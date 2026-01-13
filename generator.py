#!/usr/bin/env python3
"""
Unified Audio Generator Module

This module integrates speech (TTS), background music (BGM), and sound effects (SFX)
generation, evaluation, and optimization logic.
"""

import json
import logging
import os
import random
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from config import get_config
from llm import LLMConfig, LLMService

# Import API modules
try:
    from api import clap, mimoaudio, timestamps, tta, ttm, tts
except ImportError:
    # For development/demo purposes, prevent IDE errors
    # Actual runtime requires these modules
    logging.warning("Audio API modules not available - functionality will be limited")
    pass

logger = logging.getLogger(__name__)

# Get configuration
_config = get_config()

@dataclass
class GeneratorConfig:
    """Unified configuration class for audio generation"""
    # TTS Configuration
    tts_max_retry: int = None
    tts_emotion_threshold: float = None
    tts_overall_threshold: float = None

    # LLM Configuration
    llm_config: Optional['LLMConfig'] = None

    # BGM/SFX Configuration
    bgm_threshold: float = None
    bgm_max_attempts: int = None
    sfx_threshold: float = None
    sfx_max_attempts: int = None

    def __post_init__(self):
        """Initialize with default values from global config."""
        if self.tts_max_retry is None:
            self.tts_max_retry = _config.audio.tts_max_retry
        if self.tts_emotion_threshold is None:
            self.tts_emotion_threshold = _config.audio.tts_emotion_threshold
        if self.tts_overall_threshold is None:
            self.tts_overall_threshold = _config.audio.tts_overall_threshold
        if self.llm_config is None:
            self.llm_config = _config.llm
        if self.bgm_threshold is None:
            self.bgm_threshold = _config.audio.bgm_threshold
        if self.bgm_max_attempts is None:
            self.bgm_max_attempts = _config.audio.bgm_max_attempts
        if self.sfx_threshold is None:
            self.sfx_threshold = _config.audio.sfx_threshold
        if self.sfx_max_attempts is None:
            self.sfx_max_attempts = _config.audio.sfx_max_attempts

class UnifiedAudioGenerator:
    """
    Unified Audio Generator for TTS, BGM, and SFX generation.

    Provides integrated audio generation capabilities with quality evaluation
    and iterative improvement for optimal results.
    """

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize the unified audio generator.

        Args:
            config: Generator configuration. If None, uses default config.
        """
        self.cfg = config or GeneratorConfig()
        # Initialize LLM service for quality evaluation
        self.llm = LLMService(self.cfg.llm_config)
        logger.info("UnifiedAudioGenerator initialized")

    # ================= General Helper Methods =================

    def _cleanup_temps(self, directory: Path, prefix: str, keep_file: Optional[Path] = None):
        """Clean up temporary files with specified prefix"""
        if not directory.exists():
            return
        for p in directory.glob(f"{prefix}*.wav"):
            if p.resolve() != (keep_file.resolve() if keep_file else None):
                try: p.unlink()
                except Exception: pass

    def _finalize_file(self, source: Path, destination: Path) -> bool:
        """Move the best file to the final destination"""
        if source and source.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            # Only move if source and destination are different (prevent self-move error)
            if source.resolve() != destination.resolve():
                shutil.move(str(source), str(destination))
            return True
        return False

    # ================= Evaluation Logic (Private) =================

    def _evaluate_clap(self, audio_path: Path, text: str) -> float:
        """BGM/SFX specific: CLAP similarity evaluation"""
        try:
            return float(clap(audio_path=str(audio_path), text=text).get("similarity", 0.0))
        except Exception as e:
            logger.warning(f"CLAP evaluation error: {e}")
            return -1.0

    def _evaluate_tts(self, audio_path: str, text: str, context: Dict) -> Tuple[float, float, Optional[list]]:
        """TTS specific: Mimo + LLM evaluation"""
        try:
            mimo_res = mimoaudio(
                audio_path=audio_path,
                text="Analyze vocal emotion expression, ignore background noise.",
                thinking=True
            )
            llm_res = self.llm.evaluate_audio(
                context=context,
                mimo_eval=mimo_res,
                prompt_override="As an audio expert, evaluate TTS quality. Return only pure JSON format without Markdown markers or additional text."
            )
            return (
                llm_res.get("emotion_score", 0),
                llm_res.get("overall_score", 0),
                llm_res.get("new_emo_vector")
            )
        except Exception as e:
            logger.warning(f"TTS evaluation process failed: {e}")
            return 0.0, 0.0, None

    # ================= Core Generation Methods =================

    def _convert_timestamps(self, ts_data: Dict) -> Dict[str, List[float]]:
        """
        Convert timestamp data to {"word": [start, end]} format

        Args:
            ts_data: Raw timestamp data containing 'results' field

        Returns:
            Dict[str, List[float]]: Converted timestamp dictionary
        """
        converted_timestamps = {}

        if not ts_data or 'results' not in ts_data:
            return converted_timestamps

        for segment in ts_data['results']:
            if 'words' in segment:
                for word_info in segment['words']:
                    word = word_info.get('word', '')
                    start = round(float(word_info.get('start', 0.0)), 2)
                    end = round(float(word_info.get('end', 0.0)), 2)

                    if word:  # Only process non-empty characters
                        converted_timestamps[word] = [start, end]

        return converted_timestamps

    def generate_speech(self,
                        text: str,
                        prompt_path: Union[str, Path],
                        output_path: Path,
                        emo_vector: list,
                        context: Dict = None,
                        enable_check: bool = True) -> Dict[str, Any]:
        """
        Generate speech (TTS)
        Includes: Generation -> Mimo/LLM quality check -> Emotion vector iteration -> Timestamp generation
        """
        output_path = Path(output_path)
        context = context or {"current": text}
        temp_prefix = f"temp_tts_{output_path.stem}_"

        best_temp_file = None
        best_score = -1.0
        final_emo = emo_vector
        is_passed = False

        attempts = self.cfg.tts_max_retry if enable_check else 0

        logger.info(f"[TTS] Generating: {text[:20]}...")

        for i in range(attempts + 1):
            temp_file = output_path.parent / f"{temp_prefix}{i}.wav"
            try:
                # 1. Generate
                tts(text=text, emo_vector=json.dumps(final_emo),
                    prompt_path=str(prompt_path), output_path=str(temp_file))

                if not enable_check:
                    best_temp_file = temp_file
                    is_passed = True
                    break

                # 2. Evaluate
                e_score, o_score, new_emo = self._evaluate_tts(str(temp_file), text, context)
                total_score = e_score + o_score
                logger.info(f"   TTS Attempt {i}: Emo={e_score}, Overall={o_score}")

                # 3. Record best
                if total_score > best_score:
                    best_score = total_score
                    best_temp_file = temp_file

                # 4. Threshold check
                if e_score >= self.cfg.tts_emotion_threshold and o_score >= self.cfg.tts_overall_threshold:
                    is_passed = True
                    break

                # 5. Update vector for next retry
                if new_emo:
                    final_emo = new_emo

            except Exception as e:
                logger.error(f"   TTS Attempt {i} failed: {e}")
                continue

        # Result processing
        result_data = {
            "status": "failed",
            "audio_path": "",
            "timestamps": None,
            "final_emo": final_emo,
            "passed_check": is_passed,
            "score": best_score
        }

        success = self._finalize_file(best_temp_file, output_path)
        self._cleanup_temps(output_path.parent, temp_prefix, keep_file=None)

        if success:
            try:
                ts_data = timestamps(audio_path=str(output_path), word_timestamps=True)
                # Convert timestamp format
                converted_timestamps = self._convert_timestamps(ts_data)
                result_data.update({
                    "status": "success",
                    "audio_path": str(output_path),
                    "timestamps": converted_timestamps
                })
            except Exception as e:
                logger.error(f"   TTS timestamp generation failed: {e}")
                result_data["status"] = "timestamps_failed"

        return result_data

    def _effect_attempt_loop(self, generate_func, prompt: str, attempts: List[Any], 
                             temp_prefix: str, output_path: Path, threshold: float) -> Dict[str, Any]:
        """Universal retry loop for BGM/SFX"""
        best_score = -1.0
        best_file = None
        used_param = attempts[0] if attempts else 0

        output_path.parent.mkdir(parents=True, exist_ok=True)

        for param in attempts:
            temp_file = output_path.parent / f"{temp_prefix}{param}.wav"
            try:
                generate_func(temp_file, param)
                score = self._evaluate_clap(temp_file, prompt)
                
                if score > best_score:
                    best_score = score
                    best_file = temp_file
                    used_param = param
                
                logger.info(f"   Attempt ({param}): Score {score:.3f} {'PASS' if score >= threshold else 'RETRY'}")
                
                if score >= threshold:
                    break
            except Exception as e:
                logger.error(f"   Effect Attempt {param} failed: {e}")
                continue

        success = self._finalize_file(best_file, output_path)
        self._cleanup_temps(output_path.parent, temp_prefix, keep_file=None)
        
        return {
            "success": success, 
            "score": best_score, 
            "used_param": used_param, 
            "path": str(output_path) if success else ""
        }

    def generate_bgm(self, prompt: str, output_path: Path, clip_id: Union[str, int]) -> Dict[str, Any]:
        """Generate background music (BGM)"""
        output_path = Path(output_path)
        logger.info(f"[BGM-{clip_id}] Generating... Prompt: {prompt[:30]}...")
        
        return self._effect_attempt_loop(
            generate_func=lambda out, _: ttm(prompt=prompt, output_path=str(out)),
            prompt=prompt,
            attempts=list(range(1, self.cfg.bgm_max_attempts + 1)),
            temp_prefix=f"temp_bgm_{clip_id}_",
            output_path=output_path,
            threshold=self.cfg.bgm_threshold
        )

    def generate_sfx(self, prompt: str, duration: float, base_seed: int,
                     output_path: Path, clip_id: Union[str, int]) -> Dict[str, Any]:
        """Generate sound effects (SFX)"""
        output_path = Path(output_path)
        # Generate random seeds directly
        seeds = [random.randint(0, 1000000) for _ in range(self.cfg.sfx_max_attempts)]
        logger.info(f"[SFX-{clip_id}] Generating... Dur: {duration}s, Random Seeds: {seeds}")

        return self._effect_attempt_loop(
            generate_func=lambda out, seed: tta(prompt=prompt, duration=duration, seed=seed, output_wav=str(out)),
            prompt=prompt,
            attempts=seeds,
            temp_prefix=f"temp_sfx_{clip_id}_",
            output_path=output_path,
            threshold=self.cfg.sfx_threshold
        )

# ================= Command Line Test Entry Point =================
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    
    parser = argparse.ArgumentParser(description="Unified Audio Generator Test CLI")
    parser.add_argument("--mode", choices=["batch", "single"], default="batch", help="Run mode")
    parser.add_argument("--jsonl", default="test/step3.jsonl", help="Batch mode input file")
    parser.add_argument("--output-dir", default="generated_audio", help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    gen = UnifiedAudioGenerator()

    if args.mode == "batch":
        input_path = Path(args.jsonl)
        if not input_path.exists():
            print(f"Input file {input_path} does not exist")
            exit(1)

        items = [json.loads(line) for line in input_path.read_text('utf-8').splitlines() if line.strip()]
        results = {}

        for item in items:
            clip_id = item.get('clip_id', 'unknown')
            atype = item.get('audio_type')
            desc = item.get('desc', '') or item.get('text', '') # Compatible with TTS text field
            out_file = out_dir / f"{atype}_{clip_id}.wav"
            
            res = {}
            if atype == "bgm":
                res = gen.generate_bgm(desc, out_file, clip_id)
            elif atype == "sfx":
                res = gen.generate_sfx(desc, item.get('duration', 5.0), item.get('seed', 142), out_file, clip_id)
            elif atype == "speech":
                # Assume JSONL contains emo_vector and prompt_path
                emo = item.get('emo_vector', [0.5]*10) # Default value
                ppath = item.get('prompt_path', 'default_prompt.wav')
                res = gen.generate_speech(desc, ppath, out_file, emo)
            else:
                continue

            results[clip_id] = res
            
            # Simple success check (TTS uses status, BGM uses success)
            is_success = res.get('success') or (res.get('status') == 'success')
            if not is_success:
                logger.error(f"{atype}-{clip_id} generation failed")

        (out_dir / "gen_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("All tasks completed")