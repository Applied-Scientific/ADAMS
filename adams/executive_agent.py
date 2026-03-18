from pathlib import Path
from typing import List, Optional

from agents import Agent, ModelSettings, function_tool
from agents.tracing import set_trace_processors

from .common_utils import get_gpu_usage_decision
from .helper_agents.file_finder.file_finder_agent import get_file_finder_agent
from .helper_agents.file_parser.file_parser_agent import get_file_parser_agent
from .helper_agents.meta_analysis.meta_analysis_agent import get_meta_analysis_agent
from .helper_agents.oversight.oversight_agent import get_oversight_agent
from .memory.memory_tools import (
    PERSISTENT_MEMORY_TOOLS,
    set_session_description_tool,
    set_session_tags_tool,
    tag_session,
)
from .memory.persistent_memory import get_memory_summary
from .memory.session_memory import get_session_context_summary, format_session_context_for_prompt
from .path_config import get_agent_data_path, set_agent_data_path, set_current_session_id
from .pipeline.workflow_wrapper import _make_workflow_agent_wrapper
from .utils import list_agent_data_files
from .utils.trace_writer import JsonTraceProcessor
from .pipeline.references.reference_file_reader import read_reference_file
from .user_plan_utils import (
    append_to_plan_section,
    clone_plan,
    get_all_plan_tags,
    list_plans_by_tag,
    read_plan_document,
)
from .pipeline.workflow_agent import create_run_directory
from .plan_questions_tool import collect_plan_answers


from .model_config import get_resolved_model, set_model


_active_trace_processor: Optional[JsonTraceProcessor] = None


@function_tool
def get_gpu_spec_from_user() -> dict:
    """
    Resolve GPU usage preference when the user did not specify CPU/GPU.

    Call this tool when hardware preference is undecided (e.g. start of a run).
    Do not call it if the user already explicitly requested GPU or CPU.

    Behavior:
    - Detect available CUDA GPUs via `nvidia-smi`
    - If GPUs exist and stdin is interactive (TTY): prompt the user on stdin (y/n)
    - If GPUs exist but context is non-interactive (e.g. TUI): return ask_user_in_chat=True;
      you MUST then ask the user in chat: "I detected N GPU(s): {gpu_names}. Do you want to
      use them for docking?" and use their reply to set use_gpu before calling workflow_agent
    - If no GPU is available: return use_gpu=False (no prompt needed)

    Returns:
        dict: Hardware decision payload with:
            - `use_gpu` (bool): True/False from user or from context; when ask_user_in_chat
              is True, this is False until you ask in chat and set it from the user's answer
            - `num_gpus` (int): Number of detected GPUs (0 if unavailable)
            - `gpu_names` (str): Comma-separated GPU names, empty if unavailable
            - `ask_user_in_chat` (bool, optional): When True, you MUST ask the user in chat
              whether to use GPUs and must not proceed until they answer
            - `decision_source` (str): Why this decision was made
    """
    return get_gpu_usage_decision()


@function_tool
def resolve_gpu_config(
    use_gpu: bool = True,
    requested_num_gpus: Optional[int] = None,
    requested_gpu_ids: Optional[List[int]] = None,
) -> dict:
    """
    Resolve and validate GPU allocation for downstream docking tool calls.

    This tool removes ambiguity around GPU selection by normalizing user intent
    against detected hardware. It should be used before calling workflow/docking
    when GPU usage is enabled.

    Rules:
    - If use_gpu=False: returns num_gpus=0 and empty gpu_ids.
    - If use_gpu=True and neither requested_num_gpus nor requested_gpu_ids is set:
      allocate all detected GPUs.
    - If requested_gpu_ids is provided, it takes precedence over requested_num_gpus.
    - If requested_num_gpus is provided, allocate GPU IDs [0..requested_num_gpus-1].
    - Validates bounds against detected GPU count.

    Returns:
        dict with keys:
            - use_gpu (bool)
            - available_gpus (int)
            - gpu_names (str)
            - num_gpus (int)
            - gpu_ids (list[int])
            - source (str): one of "disabled", "auto_all", "requested_ids", "requested_count"
    """
    from .common_utils import get_gpu_info

    available_gpus, gpu_names = get_gpu_info()

    if not use_gpu:
        return {
            "use_gpu": False,
            "available_gpus": available_gpus,
            "gpu_names": gpu_names,
            "num_gpus": 0,
            "gpu_ids": [],
            "source": "disabled",
        }

    if available_gpus <= 0:
        raise ValueError(
            "GPU requested but no GPUs were detected on this host."
        )

    if requested_gpu_ids is not None:
        if len(requested_gpu_ids) == 0:
            raise ValueError("requested_gpu_ids must not be empty.")
        if any(gid < 0 for gid in requested_gpu_ids):
            raise ValueError(f"GPU IDs must be non-negative, got: {requested_gpu_ids}")
        if any(gid >= available_gpus for gid in requested_gpu_ids):
            raise ValueError(
                f"requested_gpu_ids {requested_gpu_ids} exceed available GPU range "
                f"[0..{available_gpus - 1}]"
            )
        gpu_ids = sorted(set(requested_gpu_ids))
        return {
            "use_gpu": True,
            "available_gpus": available_gpus,
            "gpu_names": gpu_names,
            "num_gpus": len(gpu_ids),
            "gpu_ids": gpu_ids,
            "source": "requested_ids",
        }

    if requested_num_gpus is not None:
        if requested_num_gpus <= 0:
            raise ValueError(
                f"requested_num_gpus must be positive, got: {requested_num_gpus}"
            )
        if requested_num_gpus > available_gpus:
            raise ValueError(
                f"requested_num_gpus={requested_num_gpus} exceeds available GPUs={available_gpus}"
            )
        gpu_ids = list(range(requested_num_gpus))
        return {
            "use_gpu": True,
            "available_gpus": available_gpus,
            "gpu_names": gpu_names,
            "num_gpus": requested_num_gpus,
            "gpu_ids": gpu_ids,
            "source": "requested_count",
        }

    # Default: if GPU is requested and no explicit count/IDs are given, use all GPUs.
    gpu_ids = list(range(available_gpus))
    return {
        "use_gpu": True,
        "available_gpus": available_gpus,
        "gpu_names": gpu_names,
        "num_gpus": available_gpus,
        "gpu_ids": gpu_ids,
        "source": "auto_all",
    }


@function_tool
def set_working_directory_tool(
    directory_path: str = None, input_file_path: str = None
) -> dict:
    """
    Set the working directory where agent_data will be created for logs, traces, and outputs.

    **IMPORTANT**: By default, the working directory is automatically set to the current working directory
    (where adams is called from). Only call this tool when the user explicitly requests a different directory.

    Call this tool when:
    - User explicitly specifies a different directory than the current working directory
    - User tells you a project directory to work in that differs from CWD
    - User provides a file path and you need to set agent_data in that file's directory

    Args:
        directory_path: Direct path to use for agent_data parent directory (e.g., "/path/to/project")
        input_file_path: Path to an input file; agent_data will be created in the same directory

    Returns:
        dict: Contains 'agent_data_path' (str) showing where data will be stored

    Example:
        >>> # User says "my files are in /data/project1/" (different from CWD)
        >>> set_working_directory_tool(directory_path="/data/project1")
        >>> # agent_data will be at /data/project1/agent_data

        >>> # User says "my receptor is at /data/project1/receptor.pdb" (different from CWD)
        >>> set_working_directory_tool(input_file_path="/data/project1/receptor.pdb")
        >>> # agent_data will be at /data/project1/agent_data
    """
    if directory_path:
        path = set_agent_data_path(path=Path(directory_path) / "agent_data")
    elif input_file_path:
        path = set_agent_data_path(input_file_path=input_file_path)
    else:
        # Use current directory
        path = set_agent_data_path()

    return {
        "agent_data_path": str(path),
        "message": f"Working directory set. Data will be stored in: {path}",
    }


@function_tool
def list_agent_data_files_tool() -> dict:
    """
    List all files in the agent_data folder to help identify receptor and ligand files.

    This function scans the agent_data directory and returns information about
    available files, categorizing them by type (receptor PDB/PDBQT files, ligand CSV files).
    It provides a quick overview of what input files are available for pipeline execution.

    Use this when:
    - You need a quick overview of available input files
    - User asks "what files do I have?" or "what can I run?"
    - You want to identify receptor and ligand files without detailed scanning

    Note: For more structured file discovery, use the file_finder_agent.
    It searches the CWD for new runs and `agent_data/` only for resume-style requests.

    Returns:
        dict: Dictionary containing:
            - 'agent_data_path' (str): Full absolute path to the agent_data folder
            - 'receptor_files' (list[str]): List of receptor file paths (PDB/PDBQT format).
                Files are identified by extension (.pdb, .pdbqt) and location
            - 'ligand_files' (list[str]): List of ligand CSV file paths.
                Files are identified by .csv extension
            - 'all_files' (list[str]): List of all files found in agent_data directory
                (includes all file types, not just receptors and ligands)
            - 'error' (str or None): Error message if scan failed (e.g., directory not found,
                permission denied), None if scan was successful

    Example:
        >>> result = list_agent_data_files_tool()
        >>> if result['receptor_files']:
        ...     print(f"Found receptor: {result['receptor_files'][0]}")
        >>> if result['ligand_files']:
        ...     print(f"Found ligand: {result['ligand_files'][0]}")
        >>> # Returns: {'agent_data_path': '/path/to/agent_data', 'receptor_files': [...], ...}
    """
    return list_agent_data_files()


def setup_tracing(session_id: Optional[str] = None) -> JsonTraceProcessor:
    """
    Set up JSON trace processor for tracking all agent interactions.

    This is completely separate from the pipeline logger (logger_utils.py).
    Trace files are written to agent_data/traces/ with real-time updates.

    Args:
        session_id: Optional session ID to continue. If None, creates a new session.

    Returns:
        JsonTraceProcessor: The configured trace processor instance

    Raises:
        RuntimeError: If trace processor initialization fails
    """
    global _active_trace_processor
    try:
        # One launcher process -> one active trace/session. Reuse if already initialized.
        if _active_trace_processor is not None:
            set_trace_processors([_active_trace_processor])
            set_current_session_id(_active_trace_processor.session_id)
            return _active_trace_processor

        trace_output_dir = get_agent_data_path() / "traces"
        trace_processor = JsonTraceProcessor(
            output_dir=str(trace_output_dir), session_id=session_id
        )
        # Replace any existing processors so one session = one trace file (no duplicates)
        set_trace_processors([trace_processor])
        set_current_session_id(trace_processor.session_id)
        _active_trace_processor = trace_processor
        if session_id:
            print(f"[Tracing] Continuing session trace file: {trace_processor.filepath}")
        else:
            print(f"[Tracing] Session trace file: {trace_processor.filepath}")
        return trace_processor
    except Exception as e:
        raise RuntimeError(f"Failed to initialize trace processor: {e}") from e


def create_agent(session_id: Optional[str] = None, model: str = "gpt-5.4") -> Agent:
    """
    Create and configure the Biophysics Controller Agent.

    Args:
        session_id: Optional current session ID; when set, the agent is prompted to tag
            this session when starting work and update description/tags when concluding.
        model: The LLM model identifier to use (default: gpt-5.4).

    Returns:
        Agent: The configured agent instance with all tools and settings
    """
    prompt_path = Path(__file__).parent / "agent_prompt.md"
    base_prompt = prompt_path.read_text()

    # Load persistent memory to insert near end of prompt
    try:
        memory_summary = get_memory_summary()
    except Exception:
        memory_summary = ""

    # Recent session context (tags + recent sessions) for discovery and alignment with past runs
    try:
        session_ctx = get_session_context_summary(limit_sessions=10)
        session_context_block = format_session_context_for_prompt(session_ctx)
    except Exception:
        session_context_block = ""

    # Build system prompt - memory summary, recent session context, optional session_id, final note
    sections = [base_prompt]
    if memory_summary:
        sections.append(f"\n{memory_summary}")
    if session_context_block:
        sections.append(f"\n=== RECENT SESSION CONTEXT ===\n{session_context_block}\n==============================")
    if session_id:
        sections.append(f"\nCurrent session ID: {session_id}.")
    sections.append("\nNote: Reference documentation is available via the read_reference_file tool. Use it to look up entry points, parameter defaults, and examples.")
    system_prompt = "\n".join(sections)

    set_model(model)
    resolved_model = get_resolved_model()

    agent = Agent(
        model=resolved_model,
        name="Biophysics Controller Agent",
        instructions=system_prompt,
        tools=[
            set_working_directory_tool,
            list_agent_data_files_tool,
            get_gpu_spec_from_user,
            resolve_gpu_config,
            read_plan_document,
            append_to_plan_section,
            collect_plan_answers,
            get_all_plan_tags,
            list_plans_by_tag,
            clone_plan,
            create_run_directory,
            read_reference_file,
            *PERSISTENT_MEMORY_TOOLS,
            set_session_description_tool,
            set_session_tags_tool,
            tag_session,
            get_meta_analysis_agent().as_tool(
                tool_name="meta_analysis_agent",
                tool_description="""An agent for solving errors in the current run. It has access to trace/log parsing, read-only session history, and read_plan_document for context.

                Use this agent when:
                - An error or failure occurred in the current run and you need to diagnose and fix it
                - You need trace and log analysis for the current session to understand what went wrong

                The agent will use trace files, log files, session history, and the plan (when plan_path is available) to provide run state, error details, and resume recommendations.""",
            ),
            get_file_finder_agent().as_tool(
                tool_name="file_finder_agent",
                tool_description="""An intelligent agent that locates the files needed to start or resume the pipeline.

                Use this agent when:
                - User wants to start the pipeline but you're unsure which step to begin from
                - User says they have "existing files", "completed steps", or want to "resume"
                - You need to determine what intermediate files are available
                - User asks "what can I run?" or "where can I start?"

                The agent will:
                1. Scan the CWD root for new-run inputs, or `agent_data/` only for resume-style requests
                2. Identify file types: receptors, ligand inputs, docking results, protein topology, pose directories, etc.
                3. Determine the best-supported entry point from file evidence
                4. Return the path bindings needed for that entry point

                Output is concise: RECOMMENDED ENTRY POINT, RELEVANT PATHS (only the paths required for that entry point), WORKFLOW PARAMETERS (path bindings only), and brief NOTES if needed.""",
            ),
            get_file_parser_agent().as_tool(
                tool_name="file_parser_agent",
                tool_description="""An agent that extracts structured statistics from pipeline output files.

                Use this agent when:
                - You need evidence from docking or MD outputs without loading large files
                - You want to summarize docking results for the user
                - You want to check completion status or identify which poses are available
                - You need to analyze affinity distributions, pose counts, or pocket statistics

                The agent can:
                1. Parse docking results CSV files to extract:
                - Affinity statistics (best, average, median, distribution)
                - Pose counts per ligand and per pocket
                - Pocket analysis (which pockets have best affinities)
                - Affinity percentiles and ranges
                2. Parse docking results to extract statistics.
                - Pose counts, affinity stats, pocket analysis

                It returns evidence; the caller decides parameters and next actions.""",
            ),
            get_oversight_agent().as_tool(
                tool_name="oversight_agent",
                tool_description="""An agent that reviews and validates pipeline execution plans before execution.

            **BEFORE calling this tool:** (1) The plan document must exist and be filled by the workflow and stage agents. (2) User must have answered plan questions and you must have recorded them: get questions via collect_plan_answers(plan_path), present them in chat, then append_to_plan_section(plan_path, "answers", ...) when the user replies. Submit to oversight **after** that so oversight sees the finalized plan. Do NOT submit a hand-written "proposed plan" or a plan without answers when questions exist.
            - When no plan exists: Call workflow_agent in plan-only mode (user's exact message). After workflow returns plan_path, if the plan has questions get them via collect_plan_answers(plan_path), present in chat, record answers when the user replies, then read_plan_document(plan_path) and call oversight_agent once.
            - When you already have a plan_path with answers recorded: Call read_plan_document(plan_path), then call oversight_agent with that plan and the user's request.

            **CRITICAL: Submit the plan to oversight_agent only AFTER the plan exists, is filled, and user answers are recorded (Step 6 then Step 7). Call oversight once with the finalized plan; if rejected, revise and resubmit once.**

            Use this agent when:
            - You have a plan_path, have collected and recorded user answers (if the plan had questions), and have read the plan via read_plan_document(plan_path)
            - You need validation before execution

            The oversight agent returns: exact_plan, feedback, concerns, suggestions. Use exact_plan as the plan to execute.

            **WORKFLOW:**
            1. workflow_agent in plan-only mode → plan_path.
            2. collect_plan_answers(plan_path); present questions in chat; append_to_plan_section(plan_path, "answers", ...) when user replies.
            3. read_plan_document(plan_path); oversight_agent(plan, user request).
            4. If approved: workflow_agent(..., plan_path=..., session_id=...) to execute. If rejected: Revise and resubmit once.""",
            ),
            _make_workflow_agent_wrapper(),
        ],
        model_settings=ModelSettings(tool_choice="auto"),
    )

    return agent
