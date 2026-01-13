#!/usr/bin/env python3
"""
LLM Service Module

Provides unified handling of LLM requests, content generation, JSON parsing, and audio quality evaluation.
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config import get_config
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Get configuration
_config = get_config()

@dataclass
class LLMConfig:
    """Configuration for LLM service."""
    api_key: str = None
    base_url: str = None
    model_name: str = None
    temperature: float = None
    max_tokens: int = None
    timeout: int = None

    def __post_init__(self):
        """Initialize with default values from global config."""
        if self.api_key is None:
            self.api_key = _config.llm.api_key
        if self.base_url is None:
            self.base_url = _config.llm.base_url
        if self.model_name is None:
            self.model_name = _config.llm.model_name
        if self.temperature is None:
            self.temperature = _config.llm.temperature
        if self.max_tokens is None:
            self.max_tokens = _config.llm.max_tokens
        if self.timeout is None:
            self.timeout = _config.llm.timeout

class LLMService:
    """
    LLM Service for content generation and audio quality evaluation.

    Handles OpenAI-compatible API calls with proper error handling and response parsing.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize the LLM service.

        Args:
            config: LLM configuration. If None, uses default configuration.
        """
        self.cfg = config or LLMConfig()
        self.client = OpenAI(
            api_key=self.cfg.api_key,
            base_url=self.cfg.base_url.rstrip('/'),
            timeout=self.cfg.timeout
        )
        logger.info(f"LLM service initialized with model: {self.cfg.model_name}")

    def _clean_response(self, text: str) -> str:
        if not text:
            return ""

        if "</think>" in text:
            text = text.split("</think>", 1)[-1].strip()

        if "```" in text:
            lines = text.splitlines()
            in_code_block = False
            block_lines = []

            for line in lines:
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    continue
                if in_code_block:
                    block_lines.append(line)

            if block_lines:
                text = "\n".join(block_lines)

        return text.strip()


    def _call(self, system: str, user: str, temperature: Optional[float] = None,
              max_tokens: Optional[int] = None) -> str:
        """
        Make a basic API call to the LLM service.

        Args:
            system: System prompt
            user: User prompt
            temperature: Sampling temperature (optional)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            Cleaned response text

        Raises:
            Exception: If API call fails
        """
        try:
            logger.debug(f"Sending request to {self.cfg.model_name}...")

            request_params = {
                "model": self.cfg.model_name,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "stream": False
            }

            # Add optional parameters if provided
            if temperature is not None:
                request_params["temperature"] = temperature
            if max_tokens is not None:
                request_params["max_tokens"] = max_tokens

            start_time = time.time()
            resp = self.client.chat.completions.create(**request_params)
            elapsed = time.time() - start_time

            raw_content = resp.choices[0].message.content
            cleaned_content = self._clean_response(raw_content)
            print(cleaned_content)
            logger.info(f"LLM request completed in {elapsed:.2f}s")
            return cleaned_content

        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise

    def generate_and_save(
        self,
        system_prompt: str,
        user_prompt: str,
        output_path: Union[str, Path],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> bool:
        """
        Generate text content and save to file.

        Args:
            system_prompt: System prompt for the LLM
            user_prompt: User prompt for content generation
            output_path: Path to save the generated content
            temperature: Sampling temperature (optional)
            max_tokens: Maximum tokens to generate (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            content = self._call(system_prompt, user_prompt, temperature, max_tokens)

            # Save to file
            out_p = Path(output_path)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            out_p.write_text(content, encoding='utf-8')

            logger.info(f"Content saved to: {out_p} (length: {len(content)} characters)")
            return True

        except Exception as e:
            logger.error(f"Failed to generate and save content: {e}")
            return False

    def evaluate_audio(
        self,
        context: Dict[str, str],
        mimo_eval: Any,
        prompt_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate audio quality and return structured data.

        Args:
            context: Context information including previous/current/next sentences
            mimo_eval: MimoAudio analysis results
            prompt_override: Optional custom system prompt

        Returns:
            Dictionary containing emotion_score, overall_score, and optionally new_emo_vector
        """
        # Prepare system prompt
        sys_prompt = prompt_override or (
            "You are a professional audio quality assessment expert. Please evaluate TTS audio "
            "based on context and acoustic analysis. Return only pure JSON format without "
            "Markdown markers or additional text."
        )

        # Prepare Mimo content
        mimo_str = json.dumps(mimo_eval, ensure_ascii=False) if isinstance(mimo_eval, dict) else str(mimo_eval)
        if "</think>" in mimo_str:
            mimo_str = mimo_str.split("</think>")[-1].strip()

        # Build user prompt
        emo_vector_info = ""
        if "emo_vector" in context and context["emo_vector"]:
            emo_vector_info = f"""
[Target Emotion Vector]
{context["emo_vector"]}
(This is the target emotional vector for this sentence)"""

        user_prompt = f"""
[Context Information]
Previous: {context.get("previous", "None")}
Current: {context.get("current", "")}
Next: {context.get("next", "None")}{emo_vector_info}

[MimoAudio Analysis]
{mimo_str}

[Evaluation Task]
Evaluate the audio quality for the "Current" sentence.
1. Emotion Score (0-100): Does it fit context? Natural transition? Matches target emotion vector?
2. Overall Score (0-100): Voice quality, intonation, fluency, pronunciation.
3. Suggest a new emo_vector if the current one doesn't fit well (optional).

[Emotion Vector Rules]
Order: [Happy, Angry, Sad, Fear, Disgust, Depressed, Surprise, Natural]
Range: 0.0-1.0 (1 decimal place).
- Natural: 0.0-0.3 (casual conversation)
- Core: 0.4-0.7 (clearly audible emotion)
- Extreme: 0.7-0.8 (intense conflict/crying)

[Output Format JSON]
{{
    "emotion_score": 85.0,
    "overall_score": 80.0,
    "new_emo_vector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]
}}
"""
        try:
            result_str = self._call(sys_prompt, user_prompt)
            data = json.loads(result_str)

            # Data validation and cleaning
            e_score = max(0.0, min(100.0, float(data.get("emotion_score", 0))))
            o_score = max(0.0, min(100.0, float(data.get("overall_score", 0))))

            vec = data.get("new_emo_vector")
            if vec and isinstance(vec, list) and len(vec) == 8:
                vec = [max(0.0, min(1.0, round(float(v), 1))) for v in vec]
            else:
                vec = None

            final_result = {
                "emotion_score": e_score,
                "overall_score": o_score,
                "new_emo_vector": vec
            }

            logger.debug("Audio evaluation completed successfully")
            logger.info(f"Evaluation scores - Emotion: {e_score}, Overall: {o_score}")

            return final_result

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed. Raw response: {result_str[:100]}...")
            return {"emotion_score": 0.0, "overall_score": 0.0, "error": "JSON parsing failed"}
        except Exception as e:
            logger.error(f"Audio evaluation failed: {e}")
            return {"emotion_score": 0.0, "overall_score": 0.0, "error": str(e)}

# ================= Unit Tests / Example Usage =================
if __name__ == "__main__":
    # Initialize service with default configuration
    config = LLMConfig()
    llm = LLMService(config)

    print("LLM Service Test Examples")
    print("=" * 40)

    # Test audio evaluation
    # print("\nTesting audio evaluation...")
    # test_context = {
    #     "previous": "Run!",
    #     "current": "I can't go on...",
    #     "next": "Don't give up."
    # }
    # mimo_mock = "Voice is shaky, low volume, sounds sad."

    # try:
    #     result = llm.evaluate_audio(context=test_context, mimo_eval=mimo_mock)
    #     print("Audio evaluation result:")
    #     print(json.dumps(result, indent=2, ensure_ascii=False))
    # except Exception as e:
    #     print(f"Audio evaluation test failed: {e}")

    # print("\nLLM service tests completed.")
