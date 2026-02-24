"""Persistent memory management with environment detection and preferences."""

import json
import multiprocessing
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..common_utils import get_cpu_count, get_gpu_info
from ..path_config import get_subdirectory

# Limits for conciseness
MAX_LEARNED_BEHAVIOR_WORDS = 50
MAX_CUSTOM_INSTRUCTIONS_WORDS = 100
MAX_LEARNED_BEHAVIORS = 10


def _get_memory_file() -> Path:
    """Get path to persistent_memory.json file."""
    memory_dir = get_subdirectory("memory")
    memory_dir.mkdir(parents=True, exist_ok=True)
    return memory_dir / "persistent_memory.json"


def _count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())


def _validate_conciseness(text: str, max_words: int, field_name: str) -> None:
    """Validate text doesn't exceed word limit."""
    word_count = _count_words(text)
    if word_count > max_words:
        raise ValueError(
            f"{field_name} exceeds {max_words} words ({word_count} words). "
            f"Please be more concise."
        )


def _get_total_ram_gb() -> float:
    """Get total system RAM in GB."""
    try:
        import psutil
        return round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        return 0.0


def _get_python_version() -> str:
    """Get Python version."""
    return platform.python_version()


def detect_environment_info() -> Dict[str, Any]:
    """Auto-detect environment (hardware + OS + runtime) information on first run."""
    usable_cores = get_cpu_count()
    gpu_count, gpu_names = get_gpu_info()
    return {
        "total_cpu_cores": multiprocessing.cpu_count(),
        "usable_cpu_cores": usable_cores,
        "total_ram_gb": _get_total_ram_gb(),
        "gpu_count": gpu_count,
        "gpu_names": gpu_names,
        "os": platform.system().lower(),
        "os_version": platform.release(),
        "architecture": platform.machine(),
        "python_version": _get_python_version(),
        "detected_at": datetime.now().isoformat(),
    }


def load_persistent_memory() -> Dict[str, Any]:
    """Load persistent memory from JSON file."""
    memory_file = _get_memory_file()
    if not memory_file.exists():
        # Initialize with environment detection
        environment_info = detect_environment_info()
        memory = {
            "environment_info": environment_info,
            "user_preferences": {
                "preferred_gpu_usage": None,
                "preferred_working_directory": None,
                "learned_behaviors": [],
                "last_updated": datetime.now().isoformat(),
            },
            "custom_instructions": "",
        }
        save_persistent_memory(memory)
        return memory

    try:
        with open(memory_file, "r", encoding="utf-8") as f:
            memory = json.load(f)
        # Migrate legacy key
        if "hardware_info" in memory and "environment_info" not in memory:
            memory["environment_info"] = memory.pop("hardware_info")
        return memory
    except (json.JSONDecodeError, IOError):
        # Fallback to defaults
        environment_info = detect_environment_info()
        return {
            "environment_info": environment_info,
            "user_preferences": {
                "preferred_gpu_usage": None,
                "preferred_working_directory": None,
                "learned_behaviors": [],
                "last_updated": datetime.now().isoformat(),
            },
            "custom_instructions": "",
        }


def save_persistent_memory(memory: Dict[str, Any]) -> None:
    """Save persistent memory to JSON file."""
    memory_file = _get_memory_file()
    with open(memory_file, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2, ensure_ascii=False)


def update_user_preference(key: str, value: Any) -> None:
    """Update a user preference."""
    memory = load_persistent_memory()
    memory["user_preferences"][key] = value
    memory["user_preferences"]["last_updated"] = datetime.now().isoformat()
    save_persistent_memory(memory)


def add_learned_behavior(behavior_text: str) -> None:
    """Add a learned behavior (max 50 words)."""
    _validate_conciseness(behavior_text, MAX_LEARNED_BEHAVIOR_WORDS, "Learned behavior")
    memory = load_persistent_memory()
    behaviors = memory["user_preferences"]["learned_behaviors"]
    behaviors.append(behavior_text)
    # Keep only most recent MAX_LEARNED_BEHAVIORS
    if len(behaviors) > MAX_LEARNED_BEHAVIORS:
        behaviors[:] = behaviors[-MAX_LEARNED_BEHAVIORS:]
    memory["user_preferences"]["last_updated"] = datetime.now().isoformat()
    save_persistent_memory(memory)


def set_custom_instructions(instructions: str) -> None:
    """Set the user's custom instructions (max 100 words). Injected into agent system prompt."""
    _validate_conciseness(instructions, MAX_CUSTOM_INSTRUCTIONS_WORDS, "Custom instructions")
    memory = load_persistent_memory()
    memory["custom_instructions"] = instructions
    save_persistent_memory(memory)


def clear_user_preferences() -> None:
    """Clear all user preferences, resetting them to defaults."""
    memory = load_persistent_memory()
    memory["user_preferences"] = {
        "preferred_gpu_usage": None,
        "preferred_working_directory": None,
        "learned_behaviors": [],
        "last_updated": datetime.now().isoformat(),
    }
    save_persistent_memory(memory)


def get_memory_summary() -> str:
    """Get formatted memory summary for agent context."""
    memory = load_persistent_memory()
    env = memory["environment_info"]
    prefs = memory["user_preferences"]
    instructions = memory.get("custom_instructions", "")

    # Format environment
    env_parts = []
    env_parts.append(f"{env.get('usable_cpu_cores', '?')}/{env.get('total_cpu_cores', '?')} CPU cores (usable/total)")
    ram = env.get("total_ram_gb")
    if ram:
        env_parts.append(f"{ram}GB RAM")
    if env.get("gpu_count", 0) > 0:
        env_parts.append(f"{env['gpu_count']} GPUs ({env['gpu_names']})")
    else:
        env_parts.append("No GPU")
    env_parts.append(f"{env.get('os', '?').title()} {env.get('os_version', '')} {env.get('architecture', '')}")

    # Format preferences
    pref_parts = []
    if prefs.get("preferred_gpu_usage") is not None:
        pref_parts.append(
            f"Prefers {'GPU' if prefs['preferred_gpu_usage'] else 'CPU'} acceleration"
        )
    if prefs.get("preferred_working_directory"):
        pref_parts.append(f"typically works in {prefs['preferred_working_directory']}")

    # Format learned behaviors
    behaviors = prefs.get("learned_behaviors", [])

    lines = [
        "=== PERSISTENT MEMORY ===",
        f"Environment: {', '.join(env_parts)}",
    ]
    if pref_parts:
        lines.append(f"User Preferences: {', '.join(pref_parts)}")
    if behaviors:
        lines.append(f"Learned Behaviors: {'; '.join(behaviors)}")
    if instructions:
        lines.append("")
        lines.append("=== USER CUSTOM INSTRUCTIONS ===")
        lines.append("The following custom instructions were provided by the user and should be followed:")
        lines.append("")
        lines.append(instructions)
        lines.append("=================================")
    lines.append("=======================")

    return "\n".join(lines)
