"""User custom instructions management."""
from pathlib import Path

from .persistent_memory import (
    MAX_CUSTOM_INSTRUCTIONS_WORDS,
    load_persistent_memory,
    set_custom_instructions,
)


def get_instructions() -> str:
    """Get current custom instructions."""
    memory = load_persistent_memory()
    return memory.get("custom_instructions", "")


def set_instructions(text: str) -> None:
    """Set custom instructions (replaces existing)."""
    set_custom_instructions(text)


def append_instructions(text: str) -> None:
    """Append to existing custom instructions."""
    current = get_instructions()
    if current:
        new_text = f"{current}\n{text}"
    else:
        new_text = text
    
    # Validate word count after appending
    word_count = len(new_text.split())
    if word_count > MAX_CUSTOM_INSTRUCTIONS_WORDS:
        raise ValueError(
            f"Appended instructions exceed {MAX_CUSTOM_INSTRUCTIONS_WORDS} words ({word_count} words). "
            f"Please be more concise."
        )
    
    set_custom_instructions(new_text)


def read_instructions_from_file(file_path: Path) -> str:
    """Read instructions from a file."""
    return file_path.read_text().strip()


def read_instructions_from_stdin() -> str:
    """Read instructions from stdin."""
    lines = []
    try:
        while True:
            lines.append(input())
    except EOFError:
        pass
    return "\n".join(lines).strip()
