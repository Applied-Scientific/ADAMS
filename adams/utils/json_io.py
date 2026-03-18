"""
Shared JSON file I/O for memory, planning, and meta_analysis.

Single place for encoding and dump conventions (UTF-8, indent=2, ensure_ascii=False)
so all modules that read/write JSON files behave consistently.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_json(file_path: Path, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Load a JSON file. Returns default if file is missing or invalid.

    Uses UTF-8 encoding. Used by session_memory, persistent_memory, and plan storage.
    """
    if not file_path.exists():
        return default if default is not None else {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return default if default is not None else {}


def save_json(
    file_path: Path,
    data: Dict[str, Any],
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """
    Save a dict to a JSON file. Creates parent directories if needed.

    Uses UTF-8 encoding. Same conventions across memory and plan storage.
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
