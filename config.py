"""
AuDirector Configuration Module

File-based configuration management for all AuDirector components.
Configuration is loaded from and saved to config.json file.
"""

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

# Configuration file path
CONFIG_FILE = "config.json"


@dataclass
class APIEndpoints:
    """API endpoint configuration."""
    tts: str = ""
    tta: str = ""
    timestamps: str = ""
    mimoaudio: str = ""
    clap: str = ""
    speaker_matching: str = ""
    music_gen: str = ""


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    format: str = "wav"
    supported_formats: list = None

    # Mixing parameters
    default_gap_ms: int = 1200
    fade_in_ms: int = 500
    fade_out_ms: int = 500
    master_fade_in_ms: int = 2000
    master_fade_out_ms: int = 2000

    # TTS parameters
    tts_emotion_threshold: float = 95.0
    tts_overall_threshold: float = 90.0
    tts_max_retry: int = 2

    # BGM/SFX parameters
    bgm_threshold: float = 0.25
    bgm_max_attempts: int = 3
    sfx_threshold: float = 0.25
    sfx_max_attempts: int = 4

    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ["wav", "mp3", "flac"]


@dataclass
class DirectoryConfig:
    """Directory structure configuration."""
    base_dir: str = "generated_audio"
    speech_dir: str = "generated_audio/speech"
    bgm_dir: str = "generated_audio/bgm"
    sfx_dir: str = "generated_audio/sfx"
    temp_dir: str = "temp"

    def get_resource_dirs(self) -> Dict[str, str]:
        """Get resource directories mapping."""
        return {
            "speech": self.speech_dir,
            "sfx": self.sfx_dir,
            "bgm": self.bgm_dir
        }


@dataclass
class LLMConfig:
    """LLM service configuration."""
    api_key: str = None
    base_url: str = ""
    model_name: str = ""
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout: int = 60

    def __post_init__(self):
        # Use environment variables with fallbacks
        if self.api_key is None:
            self.api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")

        # Override with environment variables if set
        if os.getenv("OPENAI_BASE_URL"):
            self.base_url = os.getenv("OPENAI_BASE_URL")
        if os.getenv("OPENAI_MODEL"):
            self.model_name = os.getenv("OPENAI_MODEL")


@dataclass
class AuDirectorConfig:
    """Main AuDirector configuration."""
    api: APIEndpoints = None
    audio: AudioConfig = None
    directories: DirectoryConfig = None
    llm: LLMConfig = None

    def __post_init__(self):
        if self.api is None:
            self.api = APIEndpoints()
        if self.audio is None:
            self.audio = AudioConfig()
        if self.directories is None:
            self.directories = DirectoryConfig()
        if self.llm is None:
            self.llm = LLMConfig()


def load_config_from_file(config_file: str = CONFIG_FILE) -> AuDirectorConfig:
    """Load configuration from JSON file."""
    try:
        if not Path(config_file).exists():
            print(f"Configuration file {config_file} not found, using defaults")
            return AuDirectorConfig()

        with open(config_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Create config objects from loaded data
        api_config = APIEndpoints(**data.get('api', {}))
        audio_config = AudioConfig(**data.get('audio', {}))
        dir_config = DirectoryConfig(**data.get('directories', {}))
        llm_config = LLMConfig(**data.get('llm', {}))

        return AuDirectorConfig(
            api=api_config,
            audio=audio_config,
            directories=dir_config,
            llm=llm_config
        )

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Error loading configuration from {config_file}: {e}")
        print("Using default configuration")
        return AuDirectorConfig()


def save_config_to_file(config: AuDirectorConfig, config_file: str = CONFIG_FILE) -> bool:
    """Save configuration to JSON file."""
    try:
        # Convert config to dictionary
        config_dict = {
            'api': asdict(config.api),
            'audio': asdict(config.audio),
            'directories': asdict(config.directories),
            'llm': asdict(config.llm)
        }

        # Ensure directory exists
        Path(config_file).parent.mkdir(parents=True, exist_ok=True)

        # Save to file
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)

        print(f"Configuration saved to {config_file}")
        return True

    except Exception as e:
        print(f"Error saving configuration to {config_file}: {e}")
        return False


# Global configuration instance - loaded from file
config = load_config_from_file()


def get_config() -> AuDirectorConfig:
    """Get the global configuration instance."""
    return config


def update_config(**kwargs) -> None:
    """Update global configuration with new values."""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)


def save_config(config_file: str = CONFIG_FILE) -> bool:
    """Save current configuration to file."""
    global config
    return save_config_to_file(config, config_file)


def reload_config(config_file: str = CONFIG_FILE) -> None:
    """Reload configuration from file."""
    global config
    config = load_config_from_file(config_file)
    print(f"Configuration reloaded from {config_file}")


# Auto-save configuration on exit (optional)
import atexit
atexit.register(lambda: save_config_to_file(config, CONFIG_FILE) if config else None)