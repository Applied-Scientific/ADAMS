"""
    Meta Analysis Agent - Analyzes pipeline trace files and log files to understand run state
    for resuming, error handling, and context extraction.
"""

from pathlib import Path

from agents import Agent, ModelSettings, function_tool

from ...path_config import get_subdirectory
from ...pipeline.references.reference_file_reader import read_reference_file
from .log_parser import parse_log_file_impl, search_log_file_impl
from .trace_parser import parse_trace_file_impl


@function_tool
def read_trace_file(trace_file: str = None) -> dict:
    """
    Read a trace file and return its contents for analysis.

    This function reads pipeline trace files that contain detailed execution logs
    of agent interactions, tool calls, and workflow execution. Trace files are
    in JSONL format (one JSON object per line) and contain information about
    completed steps, errors, file paths, and run status.

    Use this when:
    - User wants to resume a previous run ("continue", "resume", "pick up where we left off")
    - After an error occurs and you need to understand what happened
    - To check if there's an active/incomplete run before starting a new one
    - You need to extract file paths, output folders, or log file locations from previous runs

    Args:
        trace_file (str, optional): Path to specific trace file to read.
            If not provided (None), automatically reads the most recent trace file
            from agent_data/traces/ directory. Trace files are named with format:
            trace_YYYYMMDD_HHMMSS.jsonl
            Example: "agent_data/traces/trace_20251203_143022.jsonl"
            Default: None (reads most recent)

    Returns:
        dict: Dictionary containing:
            - 'trace_file' (str or None): Path to the trace file that was read,
                or None if no trace file was found
            - 'content' (str or None): Raw content of the trace file as a string
                (JSONL format - one JSON object per line). None if file doesn't exist
                or reading failed
            - 'error' (str or None): Error message if reading failed (e.g., file not found,
                permission denied), None if reading was successful

    Trace File Format:
        Each line in the trace file is a JSON object representing an event:
        - session_start/session_end: Session boundaries with session_id
        - workflow_start/workflow_end: Workflow execution boundaries
        - agent_start/agent_end: When agents (preprocessing_agent, docking_agent, md_agent) run
        - tool_call_start/tool_call_end: Tool executions with inputs and outputs

    Example:
        >>> result = read_trace_file()
        >>> # Reads most recent trace file from agent_data/traces/

        >>> result = read_trace_file("agent_data/traces/trace_20251203_143022.jsonl")
        >>> # Reads specific trace file
    """
    traces_dir = get_subdirectory("traces")

    if trace_file:
        trace_path = Path(trace_file)
    else:
        # Find the most recent trace file
        if not traces_dir.exists():
            return {
                "trace_file": None,
                "content": None,
                "error": f"Traces directory not found: {traces_dir}",
            }

        trace_files = sorted(traces_dir.glob("trace_*.jsonl"), reverse=True)
        if not trace_files:
            return {
                "trace_file": None,
                "content": None,
                "error": "No trace files found",
            }
        trace_path = trace_files[0]

    if not trace_path.exists():
        return {
            "trace_file": str(trace_path),
            "content": None,
            "error": f"Trace file not found: {trace_path}",
        }

    try:
        with open(trace_path, "r") as f:
            content = f.read()
        return {"trace_file": str(trace_path), "content": content, "error": None}
    except Exception as e:
        return {
            "trace_file": str(trace_path),
            "content": None,
            "error": f"Failed to read trace file: {e}",
        }


@function_tool
def parse_trace_file(trace_file: str = None) -> dict:
    """
    Parse a trace file and extract structured information (RECOMMENDED).

    This function reads a trace file and extracts only the essential information
    needed for resume/analysis, returning a structured dictionary instead of raw
    JSONL content. This dramatically reduces context window usage (from ~552KB
    to ~1-2KB).

    **PREFERRED METHOD**: Use this instead of read_trace_file() for most cases.
    Only use read_trace_file() if you need to inspect raw trace content.

    Use this when:
    - User wants to resume a previous run
    - After an error occurs and you need to understand what happened
    - To check if there's an active/incomplete run before starting a new one
    - You need to extract file paths, output folders, or log file locations

    Args:
        trace_file (str, optional): Path to specific trace file to parse.
            If not provided (None), automatically parses the most recent trace file
            from agent_data/traces/ directory.
            Default: None (parses most recent)

    Returns:
        dict: Dictionary containing structured information:
            - 'session_id' (str): Session identifier
            - 'status' (str): "completed", "error", "incomplete", or "unknown"
            - 'entry_point' (str): Inferred entry point used
            - 'output_folder' (str or None): Output folder path
            - 'log_file' (str or None): Log file path
            - 'completed_steps' (list): List of completed steps (preprocessing, docking, md_analysis)
            - 'steps_with_errors' (list): List of steps that had errors
            - 'file_paths' (dict): Dict with:
                - 'receptor' (str or None): Receptor file path
                - 'ligands_csv' (str or None): Ligands CSV path
                - 'docking_centers' (str or None): Docking centers path
                - 'docking_results' (str or None): Docking results path
            - 'last_error' (str or None): Last error message (truncated to 200 chars)
            - 'original_workflow_prompt' (str or None): User input that started the workflow
            - 'trace_file' (str): Path to the trace file parsed
            - 'error' (str or None): Error message if parsing failed

    Example:
        >>> result = parse_trace_file()
        >>> # Returns structured dict with ~1-2KB of essential information
        >>> print(result['output_folder'])  # "agent_data/outputs/run_20251203_143022"
        >>> print(result['completed_steps'])  # ["preprocessing", "docking"]

        >>> result = parse_trace_file("agent_data/traces/trace_20251203_143022.jsonl")
        >>> # Parses specific trace file
    """
    return parse_trace_file_impl(trace_file)


@function_tool
def parse_log_file(log_file: str) -> dict:
    """
    Parse a log file and extract structured information (RECOMMENDED).

    This function reads a log file and extracts only the essential information
    needed for analysis, returning a structured dictionary instead of raw log
    content. This dramatically reduces context window usage (from ~100KB to ~1-2KB).

    Use this when:
    - You need to analyze execution details from a log file
    - You want timing information, errors, or step completion status
    - The log file path is known (typically from parse_trace_file output)

    Args:
        log_file (str): Path to log file to parse.
            Typically obtained from parse_trace_file() output: result['log_file']

    Returns:
        dict: Dictionary containing structured information:
            - 'log_file' (str): Path to the log file parsed
            - 'steps' (list): List of step dictionaries with:
                - 'name' (str): Step name (module name)
                - 'stage' (str): Pipeline stage (preprocessing, docking, md_analysis)
                - 'started' (str): Start timestamp
                - 'completed' (str): Completion timestamp (or None)
                - 'duration_sec' (float): Duration in seconds (or None)
                - 'status' (str): 'success', 'error', or 'incomplete'
                - 'mode' (str): Mode for docking steps (e.g., 'search_dock')
                - 'workers' (int): Number of workers (if applicable)
                - 'timing_breakdown' (dict): Timing details (if available)
            - 'total_duration_sec' (float): Total execution time (or None)
            - 'errors' (list): List of error dictionaries with timestamp, message, context, stack_trace
            - 'warnings' (list): List of warning messages
            - 'resource_usage' (dict): Resource information (GPUs, workers)
            - 'conclusion' (str): Conclusion section if available (or None)
            - 'error' (str or None): Error message if parsing failed

    Example:
        >>> # First get log file from trace
        >>> trace_result = parse_trace_file()
        >>> log_result = parse_log_file(trace_result['log_file'])
        >>> print(log_result['total_duration_sec'])  # Total execution time
        >>> print(log_result['steps'][0]['duration_sec'])  # First step duration
    """
    return parse_log_file_impl(log_file)


@function_tool
def search_log_file(log_file: str, pattern: str, max_results: int = 50) -> dict:
    """
    Search a log file for specific patterns.

    This function searches a log file for lines matching a pattern and returns
    only the matching lines, avoiding loading the entire log into context.

    Use this when:
    - You need to find specific information in a log (e.g., "all errors", "GPU usage")
    - You want to search for specific patterns without parsing the entire log

    Args:
        log_file (str): Path to log file to search
        pattern (str): Regex pattern to search for (case-insensitive)
            Examples:
            - "ERROR" - Find all error lines
            - "GPU" - Find all GPU-related lines
            - "time.*execution" - Find timing information
            - "Worker.*Completed" - Find worker completion messages
        max_results (int): Maximum number of results to return (default: 50)

    Returns:
        dict: Dictionary containing:
            - 'log_file' (str): Path to log file searched
            - 'pattern' (str): Pattern searched
            - 'matches' (list): List of matching lines
            - 'count' (int): Total number of matches found
            - 'error' (str or None): Error message if search failed

    Example:
        >>> # Find all errors
        >>> errors = search_log_file(log_file, "ERROR")
        >>> # Find GPU usage
        >>> gpu_info = search_log_file(log_file, "GPU")
    """
    return search_log_file_impl(log_file, pattern, max_results)


@function_tool
def list_trace_files() -> dict:
    """
    List all available trace files.

    This function scans the traces directory and returns a list of all available
    trace files, sorted by modification time (most recent first). This helps
    identify which trace files exist and which one is most recent.

    Use this when:
    - You need to discover available trace files
    - User asks about a specific session or time period
    - You want to find the most recent trace file before parsing
    - You need to identify which trace file corresponds to a specific run

    Returns:
        dict: Dictionary containing:
            - 'trace_files' (list[str]): List of trace file paths, sorted by modification time (most recent first)
            - 'count' (int): Total number of trace files found
            - 'traces_dir' (str): Path to the traces directory
            - 'error' (str or None): Error message if listing failed

    Trace File Naming:
        Trace files follow pattern: trace_YYYYMMDD_HHMMSS.jsonl
        Example: "agent_data/traces/trace_20251203_143022.jsonl"
        Trace files are session-based - one file contains all runs in a session

    Example:
        >>> result = list_trace_files()
        >>> print(result['trace_files'][0])  # Most recent trace file
        >>> print(f"Found {result['count']} trace files")
    """
    traces_dir = get_subdirectory("traces")

    if not traces_dir.exists():
        return {
            "trace_files": [],
            "count": 0,
            "traces_dir": str(traces_dir),
            "error": f"Traces directory not found: {traces_dir}",
        }

    try:
        trace_files = sorted(
            traces_dir.glob("trace_*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return {
            "trace_files": [str(f) for f in trace_files],
            "count": len(trace_files),
            "traces_dir": str(traces_dir),
            "error": None,
        }
    except Exception as e:
        return {
            "trace_files": [],
            "count": 0,
            "traces_dir": str(traces_dir),
            "error": f"Failed to list trace files: {e}",
        }


@function_tool
def list_log_files() -> dict:
    """
    List all available log files and extract run identifiers.

    This function scans the logs directory and returns a list of all available
    log files, sorted by modification time (most recent first). It also extracts
    the run identifier from each log file name to help map runs to log files.

    Use this when:
    - You need to discover available log files
    - User asks about a specific run and you need to find its log file
    - You want to find log files for timing comparisons
    - You need to map run identifiers to log file paths

    Returns:
        dict: Dictionary containing:
            - 'log_files' (list[str]): List of log file paths, sorted by modification time (most recent first)
            - 'run_identifiers' (list[str]): List of run identifiers extracted from log file names
            - 'log_file_map' (dict): Dictionary mapping run identifiers to log file paths
            - 'count' (int): Total number of log files found
            - 'logs_dir' (str): Path to the logs directory
            - 'error' (str or None): Error message if listing failed

    Log File Naming:
        Log files follow pattern: adams_pipeline_run_{run_identifier}.log
        Example: "agent_data/logs/adams_pipeline_run_20251213_120500.log"
        The run_identifier matches the run folder name (e.g., "20251213_120500" or "full_docking_2ppn_9e7c")

    Example:
        >>> result = list_log_files()
        >>> print(result['log_files'][0])  # Most recent log file
        >>> # Find log file for a specific run
        >>> run_id = "20251213_120500"
        >>> log_file = result['log_file_map'].get(run_id)
        >>> if log_file:
        ...     parse_log_file(log_file)
    """
    logs_dir = get_subdirectory("logs")

    if not logs_dir.exists():
        return {
            "log_files": [],
            "run_identifiers": [],
            "log_file_map": {},
            "count": 0,
            "logs_dir": str(logs_dir),
            "error": f"Logs directory not found: {logs_dir}",
        }

    try:
        log_files = sorted(
            logs_dir.glob("adams_pipeline_run_*.log"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        log_file_list = []
        run_identifiers = []
        log_file_map = {}

        for log_file in log_files:
            log_path = str(log_file)
            log_file_list.append(log_path)

            # Extract run identifier from filename
            # Pattern: adams_pipeline_run_{run_identifier}.log
            filename = log_file.name
            if filename.startswith("adams_pipeline_run_") and filename.endswith(".log"):
                run_id = filename[len("adams_pipeline_run_") : -len(".log")]
                run_identifiers.append(run_id)
                log_file_map[run_id] = log_path

        return {
            "log_files": log_file_list,
            "run_identifiers": run_identifiers,
            "log_file_map": log_file_map,
            "count": len(log_file_list),
            "logs_dir": str(logs_dir),
            "error": None,
        }
    except Exception as e:
        return {
            "log_files": [],
            "run_identifiers": [],
            "log_file_map": {},
            "count": 0,
            "logs_dir": str(logs_dir),
            "error": f"Failed to list log files: {e}",
        }


prompt_path = Path(__file__).parent / "meta_analysis_prompt.md"
system_prompt = prompt_path.read_text()

meta_analysis_agent = Agent(
    model="gpt-5-mini",
    name="Meta Analysis Agent",
    instructions=system_prompt,
    tools=[
        read_reference_file,
        parse_trace_file,
        read_trace_file,
        parse_log_file,
        search_log_file,
        list_trace_files,
        list_log_files,
    ],
    model_settings=ModelSettings(tool_choice="auto"),
)
