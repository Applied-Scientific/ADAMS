from datetime import datetime
from pathlib import Path

from agents import Agent, ModelSettings, function_tool

from ..helper_agents.file_parser.file_parser_agent import file_parser_agent
from ..logger_utils import setup_logger
from ..path_config import get_subdirectory
from .data_preprocessing.preprocessing_agent import preprocessing_agent
from .docking.docking_agent import docking_agent
from .references.reference_file_reader import read_reference_file


@function_tool
def create_run_directory() -> str:
    """
    Create a timestamped run directory for pipeline execution.

    This function creates a new directory with format agent_data/outputs/run_YYYYMMDD_HHMMSS/
    for organizing pipeline outputs. The timestamp format matches trace file naming
    (trace_YYYYMMDD_HHMMSS.jsonl) for consistency.

    Use this when:
    - The user doesn't specify an explicit output folder
    - Starting a NEW pipeline run
    - You need a unique directory for organizing outputs

    IMPORTANT: Do NOT call this when resuming a previous run. When resuming, use the
    existing output_folder path from trace analysis instead.

    Returns:
        str: Full absolute path to the created run directory. The directory is created at:
            agent_data/outputs/run_YYYYMMDD_HHMMSS/ where YYYYMMDD_HHMMSS is the current
            timestamp (year, month, day, hour, minute, second).
            Example: "agent_data/outputs/run_20251203_143022"

    Example:
        >>> run_dir = create_run_directory()
        >>> # Returns: "agent_data/outputs/run_20251203_143022"
        >>> # Directory is created and ready for use
        >>> # IMPORTANT: Use this exact path for all subsequent operations (outpath, out_folder)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = get_subdirectory("outputs", f"run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Agent] Created run directory: {run_dir}")
    return str(run_dir)


@function_tool
def setup_pipeline_logger(log_file: str) -> str:
    """
    Set up the pipeline logger with a specified log file location.

    This function configures the centralized logging system for the entire pipeline.
    All pipeline components will automatically use this logger once it's set up.
    The logger captures all pipeline operations, errors, and status messages.

    Use this tool when:
    - Starting a new run - construct the log file name to match the run directory name
    - The user explicitly requests a custom log file location or name
    - Resuming a previous run - pass the existing log_file path to continue logging to the same file

    IMPORTANT for multiple runs (e.g. comparison): Call this immediately before the pipeline step for that run, then run that step. Do NOT call it for all runs first and then run steps—the logger is global, so only the last log_file receives output. Alternatively, pass log_file to run_docking for each run so the tool sets the logger at run time.

    Args:
        log_file (str): Full path to the log file. If the file exists, logging will append to it.
            Log files should be in agent_data/logs/ directory.
            The log file name should match the run directory name:
            - If out_folder is "agent_data/outputs/run_20251203_143022", use "agent_data/logs/adams_pipeline_run_20251203_143022.log"
            - If out_folder is "agent_data/my_experiment", use "agent_data/logs/adams_pipeline_my_experiment.log"
            - Extract folder name from output folder path (last component after '/')

    Returns:
        str: Path to the log file that was created or used. This is the same path as the input
            log_file parameter. The logger is now active and all pipeline components will use it.

    Example:
        >>> # Set up logger matching run directory
        >>> log_path = setup_pipeline_logger(log_file="agent_data/logs/adams_pipeline_20251203_143022.log")
        >>> # Returns: "agent_data/logs/adams_pipeline_20251203_143022.log"

        >>> # Set up logger with custom name
        >>> log_path = setup_pipeline_logger(log_file="agent_data/logs/adams_pipeline_my_experiment.log")
        >>> # Returns: "agent_data/logs/adams_pipeline_my_experiment.log"

        >>> # Resume run - continue logging to existing file
        >>> log_path = setup_pipeline_logger(log_file="agent_data/logs/adams_pipeline_20251203_120000.log")
        >>> # Returns: "agent_data/logs/adams_pipeline_20251203_120000.log"
    """
    logger = setup_logger(log_file=log_file)
    return log_file


# Load prompt from the same directory as this agent file
prompt_path = Path(__file__).parent / "workflow_prompt.md"
base_prompt = prompt_path.read_text()

# Reference documentation is now lazy-loaded via read_reference_file tool
system_prompt = base_prompt + "\n\nNote: Reference documentation is available via the read_reference_file tool. Use it to look up entry points, parameter defaults, and examples."

workflow_agent = Agent(
    model="gpt-5.2-pro",
    name="Molecular Docking Workflow Agent",
    instructions=system_prompt,
    tools=[
        create_run_directory,
        setup_pipeline_logger,
        read_reference_file,
        file_parser_agent.as_tool(
            tool_name="file_parser_agent",
            tool_description=(
                "An agent that extracts structured statistics from pipeline output files to enable parameter extraction and result-based decision making. "
                "Use this agent to extract parameters from previous step results. "
                "Can parse docking results CSV to extract affinity statistics, pose counts, and pocket analysis."
            ),
        ),
        preprocessing_agent.as_tool(
            tool_name="preprocessing_agent",
            tool_description=(
                "Entry Point 1: Prepare receptor PDB and process ligand CSVs for docking. "
                "Use when: raw PDB needs cleaning, CSV needs filtering/sampling, or for CUSTOM data manipulation/analysis via Python code. "
                "Required: input_pdb (raw PDB), input_data (raw SMILES CSV), OR description of custom data task."
            ),
        ),
        docking_agent.as_tool(
            tool_name="docking_agent",
            tool_description=(
                "Entry Points 2-3: Run molecular docking. "
                "Entry Point 2 (Search Docking): Discover binding sites + production docking. "
                "Required: cleaned receptor, SMILES CSV. "
                "Entry Point 3 (Production Only): Dock at known binding sites. "
                "Required: cleaned receptor, SMILES CSV, docking_centers or docking_centers_file."
            ),
        ),
    ],
    model_settings=ModelSettings(tool_choice="auto"),
)
