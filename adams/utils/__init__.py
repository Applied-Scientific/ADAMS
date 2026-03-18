"""
Utility modules for the adams pipeline.

This package contains general-purpose utilities used across the codebase.
Shared helpers used by memory, planning, and meta_analysis live here (e.g. json_io).
"""

# Import functions from common_utils.py to maintain backward compatibility
# This allows imports like "from adams.utils import run_cmd" to work
from ..common_utils import get_cpu_count, get_gpu_count, list_agent_data_files, run_cmd
from .console_transcript import start_console_transcript, stop_console_transcript
from .json_io import load_json, save_json

# Re-export for backward compatibility and shared utils
__all__ = [
    "run_cmd",
    "get_cpu_count",
    "get_gpu_count",
    "list_agent_data_files",
    "load_json",
    "save_json",
    "start_console_transcript",
    "stop_console_transcript",
]
