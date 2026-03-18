from datetime import datetime
from pathlib import Path

from agents import Agent, ModelSettings, function_tool

from ..helper_agents.file_parser.file_parser_agent import get_file_parser_agent
from ..helper_agents.meta_analysis.meta_analysis_agent import get_meta_analysis_agent
from ..logger_utils import setup_logger
from ..model_config import get_current_model_name, get_resolved_model
from ..path_config import get_subdirectory
from .data_preprocessing.preprocessing_agent import get_preprocessing_agent
from .docking.docking_agent import get_docking_agent
from .docking.protocolized_docking import run_standard_docking_job
from .md_analysis.md_agent import get_md_agent
from .references.reference_file_reader import read_reference_file
from ..user_plan_utils import (
    append_to_plan_section,
    contribute_stage_to_plan,
    create_plan_path,
    read_plan_document,
    set_plan_tags,
)


@function_tool
def create_run_directory() -> str:
    """
    Create a timestamped run directory for pipeline execution.

    This function creates a new directory with format agent_data/outputs/run_YYYYMMDD_HHMMSS/
    for organizing pipeline outputs. If that path already exists (e.g., same-second call),
    it creates run_YYYYMMDD_HHMMSS_1, run_YYYYMMDD_HHMMSS_2, etc.

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
        >>> # IMPORTANT: Use this exact path for all subsequent operations (outpath, out_folder, md_workdir)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"run_{timestamp}"
    run_dir = get_subdirectory("outputs", base_name)
    suffix = 0
    while run_dir.exists():
        suffix += 1
        run_dir = get_subdirectory("outputs", f"{base_name}_{suffix}")
    run_dir.mkdir(parents=True, exist_ok=False)
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
            - If out_folder is "agent_data/my_experiment", use "agent_data/logs/adams_pipeline_run_my_experiment.log"
            - Extract the last folder name from output_folder and remove one leading `run_` if present before building the log filename prefix.

    Returns:
        str: Path to the log file that was created or used. This is the same path as the input
            log_file parameter. The logger is now active and all pipeline components will use it.

    Example:
        >>> # Set up logger matching run directory
        >>> log_path = setup_pipeline_logger(log_file="agent_data/logs/adams_pipeline_run_20251203_143022.log")
        >>> # Returns: "agent_data/logs/adams_pipeline_run_20251203_143022.log"

        >>> # Set up logger with custom name
        >>> log_path = setup_pipeline_logger(log_file="agent_data/logs/adams_pipeline_run_my_experiment.log")
        >>> # Returns: "agent_data/logs/adams_pipeline_run_my_experiment.log"

        >>> # Resume run - continue logging to existing file
        >>> log_path = setup_pipeline_logger(log_file="agent_data/logs/adams_pipeline_run_20251203_120000.log")
        >>> # Returns: "agent_data/logs/adams_pipeline_run_20251203_120000.log"
    """
    setup_logger(log_file=log_file)
    return log_file


@function_tool
def build_pipeline_log_path(output_folder: str) -> str:
    """
    Build the canonical pipeline log-file path for an output folder.

    Rules:
    - Use the last path component of output_folder as the run identifier.
    - If that last component starts with `run_`, strip that single prefix.
    - Return: agent_data/logs/adams_pipeline_run_{identifier}.log

    Examples:
        output_folder=/.../agent_data/outputs/run_20251203_143022
        -> /.../agent_data/logs/adams_pipeline_run_20251203_143022.log

        output_folder=/.../agent_data/outputs/compare_unidock
        -> /.../agent_data/logs/adams_pipeline_run_compare_unidock.log
    """
    folder_name = Path(output_folder).name
    identifier = folder_name[4:] if folder_name.startswith("run_") else folder_name
    logs_dir = get_subdirectory("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    return str(logs_dir / f"adams_pipeline_run_{identifier}.log")


@function_tool
def run_standard_docking_job_tool(
    receptor: str,
    outdir: str,
    docking_centers: list[float],
    ligand_input: str | None = None,
    ligand_folder: str | None = None,
    backend: str = "vina_gpu",
    chain_to_keep: str = "all",
    residue_range_start: int | None = None,
    residue_range_end: int | None = None,
    keep_heterogens: str = "essential",
    required_heterogens: list[str] | None = None,
    keep_water: bool = False,
    pH: float = 7.4,
    warning_strict: bool = False,
    enumerate_microstates: bool = True,
    num_confs: int = 8,
    max_confs_to_keep: int = 2,
    conformer_energy_window_kcal: float = 3.0,
    random_seed: int = 42,
    charge_model: str = "gasteiger",
    num_pockets: int = 1,
    num_poses: int = 20,
    production_gridsize: float = 20.0,
    lock_grid_center: bool = True,
    num_gpus: int | None = 1,
    gpu_ids: list[int] | None = None,
    num_cores: int | None = None,
) -> dict:
    """
    Run the standard production-docking protocol in one tool call.

    Use this by default when the user wants a normal docking job with:
    - a raw receptor structure,
    - ligand input as a file or a folder of `.smi` files, and
    - known docking center coordinates.

    This tool performs receptor cleanup + protonation, ligand preparation,
    production docking, transcript capture, and score-only CSV generation.
    Prefer this over manual preprocessing_agent -> docking_agent orchestration
    unless the user explicitly requests a non-standard workflow.
    """
    return run_standard_docking_job(
        receptor=receptor,
        outdir=outdir,
        docking_centers=docking_centers,
        ligand_input=ligand_input,
        ligand_folder=ligand_folder,
        backend=backend,
        chain_to_keep=chain_to_keep,
        residue_range_start=residue_range_start,
        residue_range_end=residue_range_end,
        keep_heterogens=keep_heterogens,
        required_heterogens=required_heterogens,
        keep_water=keep_water,
        pH=pH,
        warning_strict=warning_strict,
        enumerate_microstates=enumerate_microstates,
        num_confs=num_confs,
        max_confs_to_keep=max_confs_to_keep,
        conformer_energy_window_kcal=conformer_energy_window_kcal,
        random_seed=random_seed,
        charge_model=charge_model,
        num_pockets=num_pockets,
        num_poses=num_poses,
        production_gridsize=production_gridsize,
        lock_grid_center=lock_grid_center,
        num_gpus=num_gpus,
        gpu_ids=gpu_ids,
        num_cores=num_cores,
    )


# Base prompt: shared orchestration rules, tools, principles. Mode-specific
# instructions (PLANNING vs EXECUTION) are injected by the workflow wrapper
# at the start of each message; see workflow_wrapper.py and
# workflow_planning_prompt.md / workflow_execution_prompt.md.
prompt_path = Path(__file__).parent / "workflow_prompt.md"
base_prompt = prompt_path.read_text()

# Reference documentation is available as a rare fallback via read_reference_file.
system_prompt = base_prompt + "\n\nNote: Use read_reference_file only when a path/entry-point rule is genuinely missing from the prompt context, or when an actual error requires agent_error_handling.md."

_workflow_agent = None
_workflow_model = None


def get_workflow_agent() -> Agent:
    global _workflow_agent, _workflow_model
    current_model = get_current_model_name()
    if _workflow_agent is None or _workflow_model != current_model:
        _workflow_agent = Agent(
            model=get_resolved_model(),
            name="Molecular Docking Workflow Agent",
            instructions=system_prompt,
            tools=[
                create_run_directory,
                build_pipeline_log_path,
                setup_pipeline_logger,
                create_plan_path,
                read_plan_document,
                append_to_plan_section,
                contribute_stage_to_plan,
                set_plan_tags,
                read_reference_file,
                run_standard_docking_job_tool,
                get_meta_analysis_agent().as_tool(
                    tool_name="meta_analysis_agent",
                    tool_description=(
                        "An agent for current-run error diagnosis and resume recommendations. "
                        "Use when a stage agent returns a failure: call with the failure context (e.g. output folder, step failed, error summary) to determine whether the run can be resumed (entry point, paths) or a similar error was seen before. "
                        "It has access to trace/log parsing, read-only session history, and read_plan_document. Use its resume recommendation to decide whether to attempt one workflow-level retry (e.g. resume from entry point) before propagating to the executive."
                    ),
                ),
                get_file_parser_agent().as_tool(
                    tool_name="file_parser_agent",
                    tool_description=(
                        "An agent that extracts structured statistics from pipeline output files. "
                        "Use this agent to surface evidence from previous step results without loading large files. "
                        "Can parse docking results CSV to extract affinity statistics, pose counts, and pocket analysis. "
                        "Can parse MD results directories to check completion status and extract pose statistics."
                    ),
                ),
                get_preprocessing_agent().as_tool(
                    tool_name="preprocessing_agent",
                    tool_description=(
                        "Entry Point 1: Prepare receptor PDB and process ligand CSVs for docking. "
                        "Use when: raw PDB needs cleaning, CSV needs filtering/sampling, or for CUSTOM data manipulation/analysis via Python code. "
                        "Required: input_pdb (raw PDB), input_data (raw SMILES CSV), OR description of custom data task."
                    ),
                ),
                get_docking_agent().as_tool(
                    tool_name="docking_agent",
                    tool_description=(
                        "Entry Points 2-3: Run molecular docking. "
                        "Entry Point 2 (Search Docking): Discover binding sites + production docking. "
                        "Required: cleaned receptor, SMILES CSV. "
                        "Entry Point 3 (Production Only): Dock at known binding sites. "
                        "Required: cleaned receptor, SMILES CSV, docking_centers or docking_centers_file."
                    ),
                ),
                get_md_agent().as_tool(
                    tool_name="md_agent",
                    tool_description=(
                        "Entry Points 4-7: Run MD stability analysis pipeline. "
                        "Entry Point 4 (ProteinTopology): Start from protein topology generation. "
                        "Required: protein_file (cleaned PDB), docking_csv, ligand_input. "
                        "Entry Point 5 (LigPrepare): Start from ligand preparation. "
                        "Required: protein_gro, protein_top, docking_csv, ligand_input. "
                        "Entry Point 6 (Gro): Start from MD simulations. "
                        "Required: poses_dir (with min.gro, system.top, index.ndx). "
                        "Entry Point 7 (Analysis): Run only stability analysis. "
                        "Required: poses_dir (with md.tpr, md.xtc, md.gro, index files)."
                    ),
                ),
            ],
            model_settings=ModelSettings(tool_choice="auto"),
        )
        _workflow_model = current_model
    return _workflow_agent
