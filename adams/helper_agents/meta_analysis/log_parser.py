"""
Log Parser - Parses pipeline log files and extracts structured information.

This module provides functions to parse log files and extract only the essential
information needed for analysis, avoiding context window overload from loading
entire log files (which can be 100KB+).
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# Mapping of module step markers to pipeline stages
MODULE_TO_STAGE = {
    "Data Preprocessing": "preprocessing",
    "Docking Inference": "docking",
    "GPU Docking Inference": "docking",
    "Search Inference": "docking",
    "Find Pocket": "docking",
    "Protein Topology": "md_analysis",
    "Ligand Preparation": "md_analysis",
    "MD Simulation": "md_analysis",
    "Stability Analysis": "md_analysis",
}


def parse_log_file_impl(log_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse a log file and extract structured information.

    This function reads a log file and extracts only the essential information
    needed for analysis, returning a structured dictionary instead of raw log
    content. This dramatically reduces context window usage (from ~100KB to ~1-2KB).

    Args:
        log_file: Path to specific log file. If None, returns error.

    Returns:
        dict with structured information:
        - log_file: Path to the log file parsed
        - steps: List of step dictionaries with:
            - name: Step name (module name)
            - stage: Pipeline stage (preprocessing, docking, md_analysis)
            - started: Start timestamp
            - completed: Completion timestamp (or None if incomplete)
            - duration_sec: Duration in seconds (or None if incomplete)
            - status: 'success', 'error', or 'incomplete'
            - mode: Mode for docking steps (e.g., 'search_dock', 'ligands')
            - workers: Number of workers (if applicable)
            - timing_breakdown: Dict with timing details (if available)
        - total_duration_sec: Total execution time (or None)
        - errors: List of error dictionaries with:
            - timestamp: When error occurred
            - message: Error message
            - context: Additional context (step name, etc.)
            - stack_trace: Stack trace if available
        - warnings: List of warning messages
        - resource_usage: Dict with resource information:
            - gpus_used: Number of GPUs (if applicable)
            - workers: Total workers (if applicable)
        - conclusion: Dict with summary statistics (if conclusion section exists)
        - error: Error message if parsing failed (or None)
    """
    if not log_file:
        return {"error": "log_file parameter is required", "log_file": None}

    log_path = Path(log_file)
    if not log_path.exists():
        return {"error": f"Log file not found: {log_path}", "log_file": str(log_path)}

    # Initialize result structure
    result = {
        "log_file": str(log_path),
        "steps": [],
        "total_duration_sec": None,
        "errors": [],
        "warnings": [],
        "resource_usage": {"gpus_used": None, "workers": None},
        "conclusion": None,
        "error": None,
    }

    try:
        # Parse log file
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Track current step
        current_step = None
        step_start_time = None

        # Track errors and warnings
        in_error_block = False
        error_lines = []
        current_error = None

        # Track conclusion section
        in_conclusion = False
        conclusion_lines = []

        for line in lines:
            # Parse timestamp and log level
            timestamp_match = re.match(
                r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) - .* - (INFO|WARNING|ERROR|DEBUG) - (.*)",
                line,
            )
            if not timestamp_match:
                # Handle continuation lines (errors, conclusions)
                if in_error_block and current_error:
                    error_lines.append(line.strip())
                elif in_conclusion:
                    conclusion_lines.append(line.strip())
                continue

            timestamp_str, level, message = timestamp_match.groups()

            try:
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue

            # Check for step start markers
            # Pattern handles: "=== Step Name Started! ===" or "=== Step Name Started for mode X! ==="
            # Match everything between === and " Started" (space before Started)
            step_start_match = re.search(r"=== (.*?)\s+Started", message)
            if step_start_match:
                # Save previous step if exists
                if current_step and step_start_time:
                    current_step["status"] = "incomplete"
                    result["steps"].append(current_step)

                # Start new step
                step_name = step_start_match.group(1)
                stage = MODULE_TO_STAGE.get(step_name, "unknown")

                # Extract mode for docking steps
                mode = None
                if (
                    "Docking Inference" in step_name
                    or "GPU Docking Inference" in step_name
                ):
                    mode_match = re.search(r"for mode (.+?)!", message)
                    if mode_match:
                        mode = mode_match.group(1)

                current_step = {
                    "name": step_name,
                    "stage": stage,
                    "started": timestamp_str,
                    "completed": None,
                    "duration_sec": None,
                    "status": "incomplete",
                    "mode": mode,
                    "workers": None,
                    "timing_breakdown": {},
                }
                step_start_time = timestamp
                in_error_block = False
                error_lines = []
                continue

            # Check for step completion markers
            # Pattern handles: "=== Step Name Completed Successfully! ===" or "=== Step Name Completed Successfully for mode X! ==="
            # Match everything between === and " Completed Successfully" (space before Completed)
            step_complete_match = re.search(
                r"=== (.*?)\s+Completed Successfully", message
            )
            if step_complete_match:
                if current_step:
                    current_step["completed"] = timestamp_str
                    if step_start_time:
                        duration = (timestamp - step_start_time).total_seconds()
                        current_step["duration_sec"] = round(duration, 2)
                    current_step["status"] = "success"
                    result["steps"].append(current_step)
                    current_step = None
                    step_start_time = None
                continue

            # Extract timing information
            if current_step:
                # Look for explicit timing logs (old format: "step_name time = X.XX (s)")
                timing_match = re.search(
                    r"(\w+(?:\s+\w+)*)\s+time\s*=\s*([\d.]+)\s*\(?s\)?",
                    message,
                    re.IGNORECASE,
                )
                if timing_match:
                    timing_name = timing_match.group(1).strip()
                    timing_value = float(timing_match.group(2))
                    current_step["timing_breakdown"][timing_name] = timing_value

                # Look for "Total time of execution" (GROMACS format: "Xm Ys")
                total_time_match = re.search(
                    r"Total time of execution:\s*(\d+)m\s*(\d+)s",
                    message,
                    re.IGNORECASE,
                )
                if total_time_match:
                    minutes = int(total_time_match.group(1))
                    seconds = int(total_time_match.group(2))
                    total_seconds = minutes * 60 + seconds
                    current_step["timing_breakdown"]["total_execution"] = total_seconds

                # Look for conclusion section timing breakdown (new format: "  - name: X.XX (s)")
                if in_conclusion or "Conclusion" in message:
                    # Pattern for timing breakdown lines: "  - name: X.XX (s)"
                    breakdown_match = re.search(
                        r"^\s*-\s+([^:]+):\s*([\d.]+)\s*\(s\)", message, re.IGNORECASE
                    )
                    if breakdown_match:
                        timing_name = breakdown_match.group(1).strip()
                        timing_value = float(breakdown_match.group(2))
                        current_step["timing_breakdown"][timing_name] = timing_value

                    # Pattern for "Total execution time: X.XX (s)" in conclusion
                    total_exec_match = re.search(
                        r"Total execution time:\s*([\d.]+)\s*\(s\)",
                        message,
                        re.IGNORECASE,
                    )
                    if total_exec_match:
                        total_seconds = float(total_exec_match.group(1))
                        current_step["timing_breakdown"][
                            "total_execution"
                        ] = total_seconds

                # Extract worker information
                if "workers" in message.lower() or "worker" in message.lower():
                    workers_match = re.search(
                        r"(\d+)\s+workers?", message, re.IGNORECASE
                    )
                    if workers_match:
                        current_step["workers"] = int(workers_match.group(1))
                        result["resource_usage"]["workers"] = int(
                            workers_match.group(1)
                        )

                # Extract GPU information
                gpu_match = re.search(r"(\d+)\s+GPU\(s\)", message, re.IGNORECASE)
                if gpu_match:
                    result["resource_usage"]["gpus_used"] = int(gpu_match.group(1))

            # Extract errors
            if level == "ERROR":
                in_error_block = True
                error_lines = [message]

                # Try to extract context
                context = None
                if current_step:
                    context = f"Step: {current_step['name']}"

                current_error = {
                    "timestamp": timestamp_str,
                    "message": message,
                    "context": context,
                    "stack_trace": None,
                }
                result["errors"].append(current_error)

                # Mark current step as error
                if current_step:
                    current_step["status"] = "error"
                    current_step["completed"] = timestamp_str
                    if step_start_time:
                        duration = (timestamp - step_start_time).total_seconds()
                        current_step["duration_sec"] = round(duration, 2)
            elif in_error_block and current_error:
                # Collect stack trace lines
                error_lines.append(line.strip())
                # Check if this looks like a stack trace line
                if "Traceback" in line or 'File "' in line or "line " in line:
                    if current_error["stack_trace"] is None:
                        current_error["stack_trace"] = []
                    current_error["stack_trace"].append(line.strip())

            # Extract warnings
            elif level == "WARNING":
                result["warnings"].append(
                    {"timestamp": timestamp_str, "message": message}
                )

            # Check for conclusion section
            if "=== Conclusion ===" in message or "=== Summary ===" in message:
                in_conclusion = True
                conclusion_lines = []
            elif in_conclusion and message.strip():
                conclusion_lines.append(message.strip())

        # Save final step if incomplete
        if current_step:
            result["steps"].append(current_step)

        # Process stack traces for errors
        for error_info in result["errors"]:
            if error_info.get("stack_trace"):
                error_info["stack_trace"] = "\n".join(error_info["stack_trace"])

        # Calculate total duration
        if result["steps"]:
            first_step = result["steps"][0]
            last_step = result["steps"][-1]
            if first_step.get("started") and last_step.get("completed"):
                try:
                    start = datetime.strptime(
                        first_step["started"], "%Y-%m-%d %H:%M:%S"
                    )
                    end = datetime.strptime(last_step["completed"], "%Y-%m-%d %H:%M:%S")
                    result["total_duration_sec"] = round(
                        (end - start).total_seconds(), 2
                    )
                except ValueError:
                    pass

        # Process conclusion if found
        if conclusion_lines:
            result["conclusion"] = "\n".join(conclusion_lines)

    except Exception as e:
        result["error"] = f"Failed to parse log file: {e}"

    return result


def search_log_file_impl(
    log_file: str, pattern: str, max_results: int = 50
) -> Dict[str, Any]:
    """
    Search a log file for specific patterns.

    This function searches a log file for lines matching a pattern and returns
    only the matching lines, avoiding loading the entire log into context.

    Args:
        log_file: Path to log file to search
        pattern: Regex pattern to search for
        max_results: Maximum number of results to return

    Returns:
        dict with:
        - log_file: Path to log file searched
        - pattern: Pattern searched
        - matches: List of matching lines with timestamps
        - count: Total number of matches found
        - error: Error message if search failed (or None)
    """
    if not log_file:
        return {"error": "log_file parameter is required", "log_file": None}

    log_path = Path(log_file)
    if not log_path.exists():
        return {"error": f"Log file not found: {log_path}", "log_file": str(log_path)}

    result = {
        "log_file": str(log_path),
        "pattern": pattern,
        "matches": [],
        "count": 0,
        "error": None,
    }

    try:
        regex = re.compile(pattern, re.IGNORECASE)

        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                if regex.search(line):
                    result["matches"].append(line.strip())
                    result["count"] += 1

                    if result["count"] >= max_results:
                        break

    except re.error as e:
        result["error"] = f"Invalid regex pattern: {e}"
    except Exception as e:
        result["error"] = f"Failed to search log file: {e}"

    return result
