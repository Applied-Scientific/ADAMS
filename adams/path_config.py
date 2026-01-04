"""
Centralized path configuration for ADAMS package.

This module provides session-based path configuration using Python's contextvars,
allowing multiple concurrent sessions with different agent_data paths.

Usage in command line:
    adams --data-dir /path/to/data

Usage in Jupyter notebook:
    from adams.path_config import set_agent_data_path
    set_agent_data_path("/path/to/data")
    # Or auto-detect from input file:
    set_agent_data_path(Path("path/to/receptor.pdb").parent / "agent_data")

Usage in code:
    from adams.path_config import get_agent_data_path
    agent_data_path = get_agent_data_path()
"""

import os
from contextvars import ContextVar
from pathlib import Path
from typing import Optional

# Context variable for session-based path storage (thread-safe)
_agent_data_path: ContextVar[Optional[Path]] = ContextVar(
    "agent_data_path", default=None
)


def validate_path_safety(target_path: Path, base_path: Optional[Path] = None) -> None:
    """
    Ensure a path is contained within an allowed base directory.

    Args:
        target_path: The path to validate
        base_path: The allowed root directory. If None, uses current working directory.

    Raises:
        PermissionError: If the target_path is outside the base_path
    """
    if base_path is None:
        base_path = Path.cwd()

    safe_root = base_path.resolve()
    resolved_target = target_path.resolve()

    if not resolved_target.is_relative_to(safe_root):
        raise PermissionError(
            f"Access denied: Path '{target_path}' resolves to '{resolved_target}' "
            f"which is outside the allowed directory '{safe_root}'"
        )


def get_safe_path(user_path: str, base_path: Optional[Path] = None) -> Path:
    """
    Resolve a user-provided path and validate its safety.

    Args:
        user_path: The user-provided path string (can be absolute or relative)
        base_path: The allowed root directory. If None, uses current working directory.

    Returns:
        Path: The confirmed safe absolute path

    Raises:
        PermissionError: If the resolved path is outside base_path
    """
    if base_path is None:
        base_path = Path.cwd()

    if os.path.isabs(user_path):
        target = Path(user_path)
    else:
        target = base_path / user_path

    validate_path_safety(target, base_path)
    return target.resolve()


def set_agent_data_path(
    path: Optional[str | Path] = None,
    input_file_path: Optional[str | Path] = None,
) -> Path:
    """Set the agent data path for the current session/context.

    This is thread-safe and context-safe - different sessions can have
    different paths simultaneously.

    Args:
        path: Direct path to agent_data directory
        input_file_path: Path to input file (will use parent directory + "agent_data")

    Returns:
        The resolved agent_data path that was set

    Example (direct path):
        >>> set_agent_data_path("/path/to/my/project/data")

    Example (from input file):
        >>> set_agent_data_path(input_file_path="/path/to/receptor.pdb")
        # Sets to /path/to/agent_data
    """
    if path:
        resolved_path = Path(path).resolve()
    elif input_file_path:
        resolved_path = Path(input_file_path).resolve().parent / "agent_data"
    else:
        # Default to current working directory
        resolved_path = Path.cwd() / "agent_data"

    _agent_data_path.set(resolved_path)
    return resolved_path


def get_agent_data_path() -> Path:
    """Get the agent data path for the current session/context.

    Raises RuntimeError if not set. Path must be explicitly configured
    using set_agent_data_path() before calling this function.

    Returns:
        Path to the agent_data directory

    Raises:
        RuntimeError: If path has not been configured yet

    Example:
        >>> from adams.path_config import set_agent_data_path, get_agent_data_path
        >>> set_agent_data_path("/path/to/project")
        >>> agent_data = get_agent_data_path()
        >>> logs_dir = agent_data / "logs"
    """
    path = _agent_data_path.get()
    if path is None:
        raise RuntimeError(
            "Agent data path not configured. "
            "Call set_agent_data_path() to configure the working directory before using paths."
        )
    return path


def get_subdirectory(*parts: str) -> Path:
    """Get a subdirectory within agent_data.

    Args:
        *parts: Path components

    Returns:
        Path to the subdirectory

    Example:
        >>> from adams.path_config import get_subdirectory
        >>> logs_dir = get_subdirectory('logs')
        >>> traces_dir = get_subdirectory('traces')
    """
    return get_agent_data_path().joinpath(*parts)


def resolve_path_from_input(
    agent_data_path: Optional[str] = None,
    input_file_path: Optional[str] = None,
) -> Path:
    """Resolve agent data path from user inputs without setting context.

    Useful for one-off path resolution without affecting the session context.

    Args:
        agent_data_path: Direct path to agent_data directory
        input_file_path: Path to input file (will use parent directory)

    Returns:
        Resolved Path object

    Example:
        >>> path = resolve_path_from_input(input_file_path="data/receptor.pdb")
        >>> # Returns data/agent_data (without setting global context)
    """
    if agent_data_path:
        return Path(agent_data_path).resolve()
    elif input_file_path:
        return Path(input_file_path).resolve().parent / "agent_data"
    else:
        return Path.cwd() / "agent_data"


def reset_paths() -> None:
    """Reset the path configuration.

    This clears the current context's agent_data path, causing the next
    get_agent_data_path() call to raise RuntimeError until a new path is set.

    Primarily used for testing or when you need to reinitialize paths in a session.

    Example:
        >>> reset_paths()
        >>> # Next get_agent_data_path() will raise RuntimeError
        >>> set_agent_data_path("/new/path")  # Must set path again
    """
    _agent_data_path.set(None)
