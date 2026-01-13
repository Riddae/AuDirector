"""
AuDirector API Client Module

This module provides unified client functions for various audio processing APIs
including TTS, TTA, timestamps, music generation, and quality evaluation.
"""

import base64
import json
import logging
import os
import pathlib
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import requests

from config import get_config

logger = logging.getLogger(__name__)

# Get configuration
_config = get_config()


def _ensure_dir(file_path: Union[str, Path]) -> Path:
    """
    Ensure the parent directory of the output file exists.

    Args:
        file_path: Path to the file

    Returns:
        Path object with ensured parent directory
    """
    path = Path(file_path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def tta(
    prompt: str,
    output_wav: str = "api_output.wav",
    api_url: Optional[str] = None,
    steps: Optional[int] = None,
    duration: float = 15.0,
    seed: int = 142,
    negative_prompt: Optional[str] = None,
    target_dbfs: Optional[float] = None
) -> bool:
    """
    Generate audio using TangoFlux API and save to local file.

    Args:
        prompt: Text description for audio generation
        output_wav: Output WAV file path
        api_url: API endpoint URL (uses default if None)
        steps: Number of generation steps (uses default if None)
        duration: Audio duration in seconds
        seed: Random seed for generation
        negative_prompt: Negative prompt (uses default if None)
        target_dbfs: Target loudness in dBFS (uses default if None)

    Returns:
        bool: True if successful, False otherwise
    """
    # Use defaults if not provided
    api_url = api_url or _config.api.tta
    steps = steps or _config.audio.bgm_max_attempts  # Using bgm_max_attempts as default steps
    negative_prompt = negative_prompt or "noise, distortion, low quality"
    target_dbfs = target_dbfs if target_dbfs is not None else -15.0

    payload = {
        "prompt": prompt,
        "steps": steps,
        "duration": duration,
        "seed": seed,
        "negative_prompt": negative_prompt,
        "target_dbfs": target_dbfs
    }

    logger.info(f"Requesting TTA generation: '{prompt[:50]}...'")

    try:
        start_time = time.time()

        # Send request with streaming to handle large files
        response = requests.post(api_url, json=payload, stream=True, timeout=300)

        if response.status_code == 200:
            # Write to file
            output_path = _ensure_dir(output_wav)
            downloaded = 0

            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            elapsed = time.time() - start_time
            size_mb = downloaded / 1024 / 1024

            logger.info(f"TTA generation successful! File saved: {output_path} ({size_mb:.2f} MB, {elapsed:.2f}s)")
            return True
        else:
            logger.error(f"TTA generation failed: Server returned status code {response.status_code}")
            logger.error(f"Error details: {response.text}")
            return False

    except requests.exceptions.Timeout:
        logger.error(f"TTA request timeout: {api_url}")
        return False
    except requests.exceptions.ConnectionError:
        logger.error(f"Connection failed: Unable to connect to {api_url}. Please check if service is running.")
        return False
    except Exception as e:
        logger.error(f"TTA generation error: {e}")
        return False



def tts(
    text: str,
    emo_vector: str,
    prompt_path: str,
    api_url: Optional[str] = None,
    emo_alpha: Optional[str] = None,
    use_random: Optional[str] = None,
    verbose: Optional[str] = None,
    target_dbfs: Optional[str] = None,
    output_path: str = "gen.wav",
    timeout: int = 120
) -> None:
    """
    Generate speech using local inference API.

    Args:
        text: Text to generate speech for
        emo_vector: Emotion vector string, e.g., "[0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]"
        prompt_path: Path to speaker prompt audio file
        api_url: Inference server URL (uses default if None)
        emo_alpha: Emotion weight factor (uses default if None)
        use_random: Whether to use randomness (uses default if None)
        verbose: Whether to output detailed logs (uses default if None)
        target_dbfs: Target loudness parameter (uses default if None)
        output_path: Output audio file path
        timeout: Request timeout in seconds
    """
    # Use defaults if not provided
    api_url = api_url or _config.api.tts
    emo_alpha = emo_alpha or "0.65"
    use_random = use_random or "false"
    verbose = verbose or "true"
    target_dbfs = target_dbfs or "-15.0"

    data = {
        "text": text,
        "emo_vector": emo_vector,
        "emo_alpha": emo_alpha,
        "use_random": use_random,
        "verbose": verbose,
        "target_dbfs": target_dbfs,
    }

    with open(prompt_path, "rb") as f:
        files = {"spk_audio_prompt": ("prompt.wav", f, "audio/wav")}
        resp = requests.post(api_url, data=data, files=files, timeout=timeout)
        resp.raise_for_status()

        with open(output_path, "wb") as out:
            out.write(resp.content)

    print(f"Saved to {output_path}")



def timestamps(
    audio_path: str,
    api_url: Optional[str] = None,
    beam_size: int = 5,
    word_timestamps: bool = True,
    vad_filter: bool = False,
    output_json: Optional[str] = None,
) -> dict:
    """
    Generate word-level timestamps for audio using timestamp API.

    Args:
        audio_path: Path to input audio file
        api_url: Timestamp API endpoint URL (uses default if None)
        beam_size: Beam search size for recognition
        word_timestamps: Whether to include word-level timestamps
        vad_filter: Whether to apply VAD filtering
        output_json: Optional path to save results as JSON file

    Returns:
        Dictionary containing timestamp results
    """
    api_url = api_url or _config.api.timestamps

    form = {
        "beam_size": str(beam_size),
        "word_timestamps": str(word_timestamps).lower(),
        "vad_filter": str(vad_filter).lower(),
    }

    logger.info(f"Sending timestamp request to {api_url} for {audio_path}")
    with open(audio_path, "rb") as f:
        files = {"audio_file": (Path(audio_path).name, f, "audio/wav")}
        resp = requests.post(api_url, data=form, files=files, timeout=300)
        resp.raise_for_status()

    data = resp.json()
    logger.info(f"Timestamp response received ({len(data)} entries)")

    if output_json:
        output_file = _ensure_dir(output_json)
        output_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        logger.info(f"Saved timestamp JSON to {output_file}")

    return data


    
def mimoaudio(
    audio_path: str,
    text: str,
    server_url: Optional[str] = None,
    thinking: bool = True
) -> dict:
    """
    Call MimoAudio API for audio understanding analysis

    Args:
        audio_path: Audio file path
        text: Analysis prompt text
        server_url: Server URL (uses default if None)
        thinking: Whether to enable thinking mode

    Returns:
        dict: API returned JSON result
    """
    server_url = server_url or _config.api.mimoaudio
    endpoint = server_url.rstrip("/") + "/audio-understanding"

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with open(audio_path, "rb") as f:
        files = {"audio": (os.path.basename(audio_path), f, "application/octet-stream")}
        data = {"text": text, "thinking": str(thinking).lower()}
        resp = requests.post(endpoint, files=files, data=data, timeout=600)
        resp.raise_for_status()
        return resp.json()



def clap(
    audio_path: str,
    text: str,
    server_url: Optional[str] = None,
):
    """
    Call CLAP similarity interface in FastAPI.

    Args:
        audio_path: Local audio file path
        text: Text description
        server_url: Server URL (uses default if None)

    Returns:
        dict: API response with similarity score
    """
    server_url = server_url or _config.api.clap
    url = server_url.rstrip("/") + "/similarity"

    with open(audio_path, "rb") as f:
        files = {"audio": (audio_path, f, "audio/wav")}
        data = {"text": text}
        resp = requests.post(url, data=data, files=files, timeout=60)

    resp.raise_for_status()
    return resp.json()




def ttm(
    prompt: str,
    output_path: str = "musicgen_out.wav",
    endpoint: Optional[str] = None,
    max_new_tokens: int = 1500,
    target_dbfs: float = -15
) -> str:
    """
    Generate music from a MusicGen FastAPI service.

    Args:
        prompt: Text prompt describing the desired music
        output_path: Path to save the output WAV file
        endpoint: Base URL of the service (uses default if None)
        max_new_tokens: Max tokens to generate
        target_dbfs: Target loudness in dBFS

    Returns:
        str: The path to the saved WAV file
    """
    endpoint = endpoint or _config.api.music_gen

    # Send request
    response = requests.post(
        f"{endpoint.rstrip('/')}/generate",
        json={"prompt": prompt, "max_new_tokens": max_new_tokens, "target_dbfs": target_dbfs},
        timeout=600,
    )
    response.raise_for_status()
    payload = response.json()

    # Decode and save audio
    audio_bytes = base64.b64decode(payload["audio_base64"])
    output_path_obj = pathlib.Path(output_path)
    output_path_obj.write_bytes(audio_bytes)

    logger.info(f"Saved music to {output_path} (sample_rate={payload['sample_rate']})")
    return str(output_path)



def speaker_matching(
    jsonl_content: Optional[str] = None,
    jsonl_path: Optional[str] = None,
    output_path: Optional[str] = None,
    api_url: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform speaker matching using the speaker matching API.

    Args:
        jsonl_content: JSONL format string content
        jsonl_path: Path to JSONL file
        output_path: Optional output file path to save results
        api_url: API service endpoint URL (uses default if None)

    Returns:
        Dictionary containing speaker matching results

    Raises:
        ValueError: If neither jsonl_content nor jsonl_path is provided
    """
    if not jsonl_content and not jsonl_path:
        raise ValueError("Must provide either jsonl_content or jsonl_path")

    api_url = api_url or _config.api.speaker_matching

    payload = {}
    if jsonl_content:
        payload["jsonl_content"] = jsonl_content
    elif jsonl_path:
        payload["jsonl_path"] = jsonl_path

    try:
        response = requests.post(api_url, json=payload, timeout=300)
        response.raise_for_status()

        result = response.json()

        # Save results if output path is specified
        if output_path:
            output_file = _ensure_dir(output_path)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result.get("results", {}), f, ensure_ascii=False, indent=2)

        return result

    except requests.exceptions.ConnectionError:
        logger.error(f"Cannot connect to speaker matching service: {api_url}")
        raise
    except Exception as e:
        logger.error(f"Speaker matching API call failed: {e}")
        raise


# Example calls (only enable during testing)
if __name__ == "__main__":
    # 1. TTA
    # tta(
    #     prompt="A single, clear ethereal Zen chime reverberating, as the heavy river waves gradually fade into a gentle, distant ripple and eventually silence",
    #     output_wav="tta_sfx.wav",
    #     steps=100,
    #     duration=3.0,
    #     seed=142,
    #     negative_prompt="noise, distortion, low quality",
    #     target_dbfs=-15.0
    # )

    # # 2. TTS
    # tts(
    #     text="Hey you! What are you shaking for? Do you want me to carry you over? Huh?",
    #     emo_vector="[0.4, 0.1, 0.0, 0.0, 0.4, 0.0, 0.3, 0.0]",
    #     prompt_path="/mnt/shared-storage-user/renyiming1/AudiobookAgent/cvd/swk.wav",
    #     output_path="./outputs/audio/tts_gen.wav",
    # )

    # # 3. Timestamps
    # timestamps(
    #     "/mnt/shared-storage-user/renyiming1/AudiobookAgent/cvd/zbj.wav",
    #     output_json="./outputs/json/timestamps.json",
    # )

    # # 4. MimoAudio
    # mimoaudio(
    #     audio_path="/mnt/shared-storage-user/renyiming1/AudiobookAgent/cvd/zbj.wav",
    #     text="Please analyze the emotion of this audio from a vocal perspective.",
    #     thinking=True,
    # )

    # 5. TTM
    ttm(prompt="Upbeat lo-fi hip hop podcast intro theme", output_path="musicgen_out.wav", max_new_tokens=1500)

    # 6. Speaker Matching
    # speaker_matching(
    #     jsonl_content='{"name": "John Doe", "desc": "A 30-year-old young man"}',
    #     output_path="./outputs/json/speakers.json"
    # )