#!/usr/bin/env python3
"""
Simplified Audiobook Generation Workflow Script

This module provides a streamlined workflow for generating audiobooks from text content.
It orchestrates the entire process from content generation to final audio mixing.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from api import speaker_matching
from config import get_config
from generator import GeneratorConfig, UnifiedAudioGenerator
from llm import LLMConfig, LLMService
from mix import AudioMixer
from utils import DirectoryManager, FileHandler, StepExecutor, workflow_error_handler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get configuration
_config = get_config()


@dataclass
class WorkflowResult:
    """Result of a workflow execution"""
    success: bool
    message: str
    output_files: Dict[str, str]
    execution_time: float = 0.0


class AudioBookWorkflow:
    """
    Audiobook Generation Workflow

    Orchestrates the complete audiobook generation process including:
    - Content script generation
    - Speaker matching
    - Voice synthesis
    - Audio effects generation
    - Final mixing
    """

    def __init__(self, llm_config: Optional[LLMConfig] = None, output_base_dir: Optional[str] = None):
        """Initialize the audiobook workflow with necessary components."""
        self.llm_config = llm_config or _config.llm
        self.llm_service = LLMService(self.llm_config)
        self.audio_generator = UnifiedAudioGenerator(GeneratorConfig(llm_config=self.llm_config))

        # Set output base directory for intermediate files
        self.output_base_dir = output_base_dir or _config.directories.base_dir

        # Initialize step executors for different workflow phases
        self.content_executor = StepExecutor("Content Generation", 1)
        self.speaker_executor = StepExecutor("Speaker Matching", 2)
        self.dialogue_executor = StepExecutor("Dialogue Generation", 3)
        self.speech_executor = StepExecutor("Speech Synthesis", 4)
        self.script_executor = StepExecutor("Script Generation", 5)
        self.effects_executor = StepExecutor("Effects Generation", 6)
        self.mixing_executor = StepExecutor("Audio Mixing", 7)

        logger.info(f"AudioBookWorkflow initialized successfully (output dir: {self.output_base_dir})")

    @workflow_error_handler 
    def generate_step1_jsonl(self, user_input: str) -> str:
        """Generate the first step JSONL file (contains role and dialogue structure) based on user input."""
        return self.content_executor.execute_step(self._generate_content_structure, user_input)

    def _generate_content_structure(self, user_input: str) -> str:
        """Internal method to build system prompt and call LLM to generate content structure."""
        self.content_executor.validate_prompt_file("prompts/1.prompt_en")

        system_prompt = Path("prompts/1.prompt_en").read_text(encoding='utf-8')
        
        # Merge the original segmented input into a unified user prompt
        user_prompt = f"User Request: {user_input}"

        output_path = Path(self.output_base_dir) / "1.jsonl"

        success = self.llm_service.generate_and_save(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_path=str(output_path)
        )

        if not success:
            raise Exception("LLM generation unsuccessful")

        return str(output_path)

    @workflow_error_handler
    def generate_speaker_matching(self, jsonl_file: str) -> str:
        """Perform speaker matching using the speaker matching API."""
        return self.speaker_executor.execute_step(self._perform_speaker_matching, jsonl_file)

    def _perform_speaker_matching(self, jsonl_file: str) -> str:
        """Internal method to perform speaker matching."""
        self.speaker_executor.validate_input_file(jsonl_file)

        with open(jsonl_file, 'r', encoding='utf-8') as f:
            jsonl_content = f.read()

        output_path = Path(self.output_base_dir) / "name.json"
        result = speaker_matching(jsonl_content=jsonl_content, output_path=str(output_path))
        total_matches = result.get("total_matches", 0)
        logger.info(f"Speaker matching completed: {total_matches} characters matched")

        return str(output_path)

    @workflow_error_handler
    def generate_step2_jsonl(self, step1_file: str) -> str:
        """Generate the second step JSONL file with detailed dialogue content."""
        return self.dialogue_executor.execute_step(self._generate_dialogue_content, step1_file)

    def _generate_dialogue_content(self, step1_file: str) -> str:
        """Internal method to generate dialogue content."""
        self.dialogue_executor.validate_input_file(step1_file)
        self.dialogue_executor.validate_prompt_file("prompts/2.prompt_en")

        system_prompt = Path("prompts/2.prompt_en").read_text(encoding='utf-8')

        # Build user prompt from step 1 data
        step1_data = FileHandler.load_jsonl_file(step1_file)
        user_prompt = '\n'.join(json.dumps(item, ensure_ascii=False) for item in step1_data)

        output_path = Path(self.output_base_dir) / "2.jsonl"
        success = self.llm_service.generate_and_save(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            output_path=str(output_path)
        )

        if not success:
            raise Exception("LLM generation unsuccessful")

        return str(output_path)

    @workflow_error_handler
    def generate_audio_from_jsonl(self, jsonl_file: str) -> str:
        """Generate speech audio files and timestamp data from dialogue JSONL."""
        return self.speech_executor.execute_step(self._generate_speech_audio, jsonl_file)

    def _generate_speech_audio(self, jsonl_file: str) -> str:
        """Internal method to generate speech audio and timestamps."""
        self.speech_executor.validate_input_file(jsonl_file)

        # Create output directory
        speech_dir_name = f"{self.output_base_dir}/speech"
        audio_dir = DirectoryManager.create_output_dirs(speech_dir_name)[speech_dir_name]

        # Load required data
        speaker_mapping = self._load_speaker_mapping()
        dialogue_lines = FileHandler.load_jsonl_file(jsonl_file)

        if not dialogue_lines:
            logger.warning("No valid dialogue lines found in input file")
            output_path = Path(self.output_base_dir) / "3.jsonl"
            FileHandler.save_jsonl_file([], str(output_path))
            return str(output_path)

        logger.info(f"Processing {len(dialogue_lines)} dialogue lines...")
        timestamps_data = []

        for idx, line_data in enumerate(dialogue_lines, 1):
            result = self._process_single_dialogue_line(line_data, idx, dialogue_lines, speaker_mapping, audio_dir)
            if result:
                timestamps_data.append(result)

        # Save timestamps
        output_path = Path(self.output_base_dir) / "3.jsonl"
        FileHandler.save_jsonl_file(timestamps_data, str(output_path))
        logger.info(f"Generated {len(timestamps_data)} speech files and timestamps")

        return str(output_path)

    def _process_single_dialogue_line(self, line_data: Dict[str, Any], idx: int,
                                    all_lines: List[Dict[str, Any]], speaker_mapping: Dict[str, Any],
                                    audio_dir: Path) -> Optional[Dict[str, Any]]:
        """Process a single dialogue line for speech generation."""
        speaker = line_data.get("speaker", "")
        content = line_data.get("content", "").strip()
        emo_vector = line_data.get("emo_vector", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

        if not content:
            return None

        context = self._build_context(all_lines, idx, emo_vector)
        prompt_path = self._get_speaker_prompt_path(speaker, speaker_mapping)
        audio_path = audio_dir / f"speech_{idx}.wav"

        logger.info(f"Generating speech {idx}: {speaker} - {content[:30]}...")

        result = self.audio_generator.generate_speech(
            text=content,
            prompt_path=prompt_path,
            output_path=audio_path,
            emo_vector=emo_vector,
            context=context,
            enable_check=False
        )

        if result["status"] == "success":
            return {
                "speaker": speaker,
                "content": content,
                "timestamps": result["timestamps"],
                "emo_vector": emo_vector
            }
        else:
            logger.warning(f"Generation failed: {result.get('status', 'unknown error')}")
            return None

    def _load_speaker_mapping(self) -> Dict[str, Any]:
        """Load speaker matching results from various possible locations."""
        base_path = Path(self.output_base_dir)
        for file_name in ["name.jsonl", "name.json"]:
            file_path = base_path / file_name
            if file_path.exists():
                try:
                    data = FileHandler.load_json_file(str(file_path)) if file_name.endswith('.json') else FileHandler.load_jsonl_file(str(file_path))
                    speaker_mapping = data.get("results", data) if file_name.endswith('.json') else data
                    logger.info(f"Loaded {len(speaker_mapping)} speaker matching results from {file_path}")
                    return speaker_mapping
                except Exception as e:
                    logger.warning(f"Failed to load speaker mapping from {file_path}: {e}")
        return {}

    def _build_context(self, all_lines: List[Dict[str, Any]], current_idx: int,
                      emo_vector: List[float]) -> Dict[str, Any]:
        """Build context information for audio generation."""
        return {
            "previous": all_lines[current_idx-2]["content"] if current_idx > 1 else "None",
            "current": all_lines[current_idx-1]["content"],
            "next": all_lines[current_idx]["content"] if current_idx < len(all_lines) else "None",
            "emo_vector": emo_vector
        }

    def _get_speaker_prompt_path(self, speaker: str, speaker_mapping: Dict[str, Any]) -> str:
        """Get the appropriate prompt path for a speaker."""
        if speaker in speaker_mapping:
            matched_info = speaker_mapping[speaker]
            if isinstance(matched_info, dict):
                # Try different possible key names for wav_path
                for key in ["wav_path", "name"]:
                    if key in matched_info:
                        nested_info = matched_info[key]
                        if isinstance(nested_info, dict) and "wav_path" in nested_info:
                            prompt_path = nested_info["wav_path"]
                        elif isinstance(nested_info, str):
                            prompt_path = matched_info["wav_path"]
                        else:
                            continue

                        logger.debug(f"Using matched speaker: {speaker} -> {Path(prompt_path).name}")
                        return prompt_path
        return "default_prompt.wav"



    @workflow_error_handler
    def generate_audio_script(self, timestamp_file: str) -> str:
        """Generate audio production script from timestamp data."""
        return self.script_executor.execute_step(self._generate_audio_script, timestamp_file)

    def _generate_audio_script(self, timestamp_file: str) -> str:
        """Internal method to generate audio script."""
        self.script_executor.validate_input_file(timestamp_file)
        self.script_executor.validate_prompt_file("prompts/3.prompt")

        with open(timestamp_file, 'r', encoding='utf-8') as f:
            jsonl_content = f.read()

        system_prompt = Path("prompts/3.prompt").read_text(encoding='utf-8')

        output_path = Path(self.output_base_dir) / "4.jsonl"
        success = self.llm_service.generate_and_save(
            system_prompt=system_prompt,
            user_prompt=jsonl_content,
            output_path=str(output_path)
        )

        if not success:
            raise Exception("Audio script generation failed")

        return str(output_path)

    @workflow_error_handler
    def generate_effects_from_script(self, script_file: str) -> WorkflowResult:
        """Generate BGM and SFX audio files from the production script."""
        return self.effects_executor.execute_step(self._generate_effects, script_file)

    def _generate_effects(self, script_file: str) -> WorkflowResult:
        """Internal method to generate audio effects."""
        self.effects_executor.validate_input_file(script_file)

        # Create output directories
        bgm_dir_name = f"{self.output_base_dir}/bgm"
        sfx_dir_name = f"{self.output_base_dir}/sfx"
        dirs = DirectoryManager.create_output_dirs(bgm_dir_name, sfx_dir_name)
        bgm_dir, sfx_dir = dirs[bgm_dir_name], dirs[sfx_dir_name]

        # Generate effects
        elements = FileHandler.load_jsonl_file(script_file)
        bgm_count, sfx_count = self._process_audio_effects(elements, bgm_dir, sfx_dir)

        message = f"BGM/SFX generation completed: {bgm_count} BGM, {sfx_count} SFX"
        return WorkflowResult(
            success=True,
            message=message,
            output_files={"bgm_directory": str(bgm_dir), "sfx_directory": str(sfx_dir)}
        )

    def _process_audio_effects(self, elements: List[Dict[str, Any]], bgm_dir: Path, sfx_dir: Path) -> tuple[int, int]:
        """Process audio effects generation for all elements."""
        bgm_count = sfx_count = 0

        for element in elements:
            audio_type = element.get("audio_type")
            if audio_type == "bgm" and element.get("action") == "start":
                if self._generate_single_effect(element, bgm_dir, "bgm"):
                    bgm_count += 1
            elif audio_type == "sfx":
                if self._generate_single_effect(element, sfx_dir, "sfx"):
                    sfx_count += 1

        return bgm_count, sfx_count

    def _generate_single_effect(self, element: Dict[str, Any], output_dir: Path, effect_type: str) -> bool:
        """Generate a single audio effect (BGM or SFX)."""
        desc = element.get("desc", "")
        clip_id = element.get("clip_id")
        output_path = output_dir / f"{effect_type}_{clip_id}.wav"

        logger.info(f"Generating {effect_type.upper()} {clip_id}: {desc[:30]}...")

        try:
            generator_method = getattr(self.audio_generator, f"generate_{effect_type}")
            kwargs = {
                "prompt": desc,
                "output_path": output_path,
                "clip_id": clip_id
            }

            if effect_type == "sfx":
                kwargs.update({
                    "duration": element.get("duration", 15.0),
                    "base_seed": clip_id
                })

            result = generator_method(**kwargs)

            if result.get("success"):
                logger.info(f"{effect_type.upper()} {clip_id} generated successfully")
                return True
            else:
                logger.warning(f"{effect_type.upper()} {clip_id} generation failed")
                return False

        except Exception as e:
            logger.error(f"{effect_type.upper()} {clip_id} generation error: {e}")
            return False

    @workflow_error_handler
    def generate_final_mix(self, script_file: str) -> Dict[str, str]:
        """Generate final audio mix using the audio mixer."""
        return self.mixing_executor.execute_step(self._perform_final_mix, script_file)

    def _perform_final_mix(self, script_file: str) -> Dict[str, str]:
        """Internal method to perform final audio mixing."""
        self.mixing_executor.validate_input_file(script_file)

        resource_dirs = DirectoryManager.get_resource_dirs(self.output_base_dir)
        DirectoryManager.create_output_dirs(self.output_base_dir)

        mixer = AudioMixer(
            script_path=script_file,
            output_dir=self.output_base_dir,
            resource_dirs=resource_dirs,
            gap_ms=500
        )

        result = mixer.run()
        self._log_mix_results(result)

        return result

    def _log_mix_results(self, result: Dict[str, str]) -> None:
        """Log the results of the audio mixing process."""
        logger.info("Final audio mixing completed successfully!")
        logger.info("Generated files:")

        for track_type, file_path in result.items():
            if Path(file_path).exists():
                file_size = Path(file_path).stat().st_size / 1024 / 1024  # MB
                logger.info(f"  {track_type}: {file_path} ({file_size:.2f} MB)")
            else:
                logger.warning(f"  {track_type}: {file_path} (file not found)")

    def run_complete_workflow(self, user_input: str) -> WorkflowResult:
        """
        Run the complete audiobook generation workflow.

        Args:
            user_input: User input for audio generation

        Returns:
            WorkflowResult with execution details
        """
        import time

        start_time = time.time()
        logger.info(f"Starting complete workflow: {user_input}")

        try:
            # Step 1: Generate content structure
            step1_file = self.generate_step1_jsonl(user_input)

            # Step 2: Speaker matching
            name_file = self.generate_speaker_matching(step1_file)

            # Step 3: Generate detailed dialogue
            step2_file = self.generate_step2_jsonl(step1_file)

            # Step 4: Generate speech and timestamps
            step3_file = self.generate_audio_from_jsonl(step2_file)

            # Step 5: Generate audio script
            step4_file = self.generate_audio_script(step3_file)

            # Step 6: Generate effects
            effects_result = self.generate_effects_from_script(step4_file)

            # Step 7: Final mix
            final_result = self.generate_final_mix(step4_file)

            execution_time = time.time() - start_time

            logger.info(f"Workflow completed successfully in {execution_time:.2f} seconds")

            return WorkflowResult(
                success=True,
                message="Audiobook generation completed successfully",
                output_files={
                    "final_mix": final_result.get("master", ""),
                    "speech_dir": f"{self.output_base_dir}/speech",
                    "bgm_dir": f"{self.output_base_dir}/bgm",
                    "sfx_dir": f"{self.output_base_dir}/sfx"
                },
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Workflow failed: {e}"
            logger.error(error_msg)

            return WorkflowResult(
                success=False,
                message=error_msg,
                output_files={},
                execution_time=execution_time
            )


def main():
    """Main function for command-line execution."""
    logger.info("Starting AuDirector workflow execution")

    workflow = AudioBookWorkflow()


if __name__ == "__main__":
    main()
