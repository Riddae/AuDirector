"""
AuDirector Utility Module

This module provides common utilities for file handling, step execution,
and error handling across the AuDirector project.
"""

import json
import logging
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class FileHandler:
    """Utility class for common file operations."""

    @staticmethod
    def ensure_path_exists(file_path: Union[str, Path]) -> Path:
        """Ensure the parent directory of the file exists."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
        """Load and parse a JSONL file, filtering for valid entries."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        entry = json.loads(line.strip())
                        if FileHandler._is_valid_entry(entry):
                            data.append(entry)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        return data

    @staticmethod
    def save_jsonl_file(data: List[Dict[str, Any]], file_path: str) -> None:
        """Save data to a JSONL file."""
        FileHandler.ensure_path_exists(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    @staticmethod
    def load_json_file(file_path: str) -> Dict[str, Any]:
        """Load a JSON file."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def _is_valid_entry(entry: Dict[str, Any]) -> bool:
        """Check if an entry is valid based on its content."""
        # For dialogue entries, check for content
        if "content" in entry:
            return bool(entry.get("content", "").strip())
        # For other entries, check for essential fields
        return bool(entry)


class StepExecutor:
    """Base class for executing workflow steps with common patterns."""

    def __init__(self, step_name: str, step_number: int):
        self.step_name = step_name
        self.step_number = step_number

    def execute_step(self, func: Callable, *args, **kwargs) -> Any:
        """Execute a step with logging and error handling."""
        logger.info(f"Step {self.step_number}: {self.step_name}")
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"Step {self.step_number} completed in {elapsed:.2f}s")
            return result
        except Exception as e:
            logger.error(f"Step {self.step_number} failed: {e}")
            raise

    def validate_input_file(self, file_path: str) -> None:
        """Validate that input file exists."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Input file not found: {file_path}")

    def validate_prompt_file(self, prompt_path: str) -> None:
        """Validate that prompt file exists."""
        if not Path(prompt_path).exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")


def workflow_error_handler(func: Callable) -> Callable:
    """Decorator for unified error handling in workflow methods."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            raise
        except FileNotFoundError:
            raise  # Re-raise file not found errors as-is
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}")
            raise
    return wrapper


class DirectoryManager:
    """Utility for managing output directories."""

    @staticmethod
    def create_output_dirs(*dirs: str) -> Dict[str, Path]:
        """Create output directories and return their paths."""
        created_dirs = {}
        for dir_name in dirs:
            dir_path = Path(dir_name)
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs[dir_name] = dir_path
        return created_dirs

    @staticmethod
    def get_resource_dirs(base_dir: str = "generated_audio") -> Dict[str, str]:
        """Get standard resource directory paths."""
        return {
            "speech": f"{base_dir}/speech",
            "sfx": f"{base_dir}/sfx",
            "bgm": f"{base_dir}/bgm"
        }