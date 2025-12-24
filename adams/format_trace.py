"""
Format trace files for human readability.

Converts JSONL trace files to formatted JSON arrays for easier viewing.
If no trace file is specified, uses the most recent trace file.
"""

import json
from pathlib import Path
from typing import Optional

from .path_config import get_subdirectory


def format_trace_file(
    jsonl_path: Optional[str] = None, output_path: Optional[str] = None
) -> str:
    """
    Convert a JSONL trace file to a formatted JSON array for human readability.

    Args:
        jsonl_path: Path to the JSONL trace file. If None, uses most recent trace.
        output_path: Path to write formatted JSON. If None, creates .json version.

    Returns:
        Path to the formatted JSON file

    Raises:
        FileNotFoundError: If trace file doesn't exist
        ValueError: If trace file is empty or invalid
    """
    # Traces are in agent_data/traces
    traces_dir = get_subdirectory("traces")

    # Find trace file
    if jsonl_path:
        trace_path = Path(jsonl_path)
        # If relative path and doesn't exist, try relative to traces_dir
        if not trace_path.is_absolute() and not trace_path.exists():
            potential_path = traces_dir / trace_path.name
            if potential_path.exists():
                trace_path = potential_path
    else:
        # Find most recent trace file
        if not traces_dir.exists():
            raise FileNotFoundError(f"Traces directory not found: {traces_dir}")

        trace_files = sorted(traces_dir.glob("trace_*.jsonl"), reverse=True)
        if not trace_files:
            raise FileNotFoundError(f"No trace files found in {traces_dir}")
        trace_path = trace_files[0]

    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    # Determine output path
    if output_path:
        output_path = Path(output_path)
    else:
        output_path = trace_path.with_suffix(".json")

    # Read and parse JSONL
    events = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {trace_path}: {e}"
                )

    if not events:
        raise ValueError(f"Trace file is empty: {trace_path}")

    # Write formatted JSON
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)
    except (OSError, IOError) as e:
        raise IOError(f"Failed to write formatted trace to {output_path}: {e}")

    return str(output_path)
