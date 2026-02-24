from pathlib import Path
from typing import List, Optional

from agents import Agent, ModelSettings, function_tool
from agents.tracing import add_trace_processor

from .common_utils import get_gpu_usage_decision
from .helper_agents.file_finder.file_finder_agent import file_finder_agent
from .helper_agents.file_parser.file_parser_agent import file_parser_agent
from .helper_agents.meta_analysis.meta_analysis_agent import meta_analysis_agent
from .helper_agents.oversight.oversight_agent import oversight_agent
from .memory.memory_tools import MEMORY_TOOLS
from .memory.persistent_memory import get_memory_summary
from .memory.session_memory import (
    format_session_context_for_prompt,
    get_session_context_summary,
)
from .path_config import get_agent_data_path, set_agent_data_path
from .pipeline.data_preprocessing.preprocessing_agent import preprocessing_agent
from .pipeline.workflow_agent import workflow_agent
from .utils import list_agent_data_files
from .utils.trace_writer import JsonTraceProcessor
from .pipeline.references.reference_file_reader import read_reference_file


def _resolve_model(model: str):
    """Resolve a model string to an SDK-compatible model object.

    Plain names (e.g. 'gpt-5') are passed through as-is for native OpenAI
    routing.  Names containing '/' (e.g. 'gemini/gemini-3-pro',
    'anthropic/claude-3-5-sonnet-20240620') are wrapped in LitellmModel
    so the Agents SDK routes them through LiteLLM.
    """
    if "/" in model:
        from agents.extensions.models.litellm_model import LitellmModel
        return LitellmModel(model=model)
    return model


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

    Note: For more detailed file discovery and entry point analysis, use the file_finder_agent
    which provides comprehensive scanning and entry point recommendations.

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
    try:
        trace_output_dir = get_agent_data_path() / "traces"
        trace_processor = JsonTraceProcessor(
            output_dir=str(trace_output_dir), session_id=session_id
        )
        add_trace_processor(trace_processor)
        if session_id:
            print(f"[Tracing] Continuing session trace file: {trace_processor.filepath}")
        else:
            print(f"[Tracing] Session trace file: {trace_processor.filepath}")
        return trace_processor
    except Exception as e:
        raise RuntimeError(f"Failed to initialize trace processor: {e}") from e


def create_agent(session_id: Optional[str] = None, model: str = "gpt-5") -> Agent:
    """
    Create and configure the Biophysics Controller Agent.

    Args:
        session_id: Optional current session ID; when set, the agent is prompted to tag
            this session when starting work and update description/tags when concluding.
        model: The LLM model identifier to use (default: gpt-5).

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

    try:
        ctx = get_session_context_summary(limit_sessions=10)
        session_memory_block = format_session_context_for_prompt(ctx)
    except Exception:
        session_memory_block = ""

    # Build system prompt - insert memory summary, session context, optional session_id line, final note
    sections = [base_prompt]
    if memory_summary:
        sections.append(f"\n{memory_summary}")
    if session_memory_block:
        sections.append(f"\n## Session memory\n{session_memory_block}")
    if session_id:
        sections.append(f"\nCurrent session ID: {session_id}. Tag this session when starting work and update description/tags when concluding.")
    sections.append("\nNote: Reference documentation is available via the read_reference_file tool. Use it to look up entry points, parameter defaults, and examples.")
    system_prompt = "\n".join(sections)

    resolved_model = _resolve_model(model)

    agent = Agent(
        model=resolved_model,
        name="Biophysics Controller Agent",
        instructions=system_prompt,
        tools=[
            set_working_directory_tool,
            list_agent_data_files_tool,
            get_gpu_spec_from_user,
            resolve_gpu_config,
            read_reference_file,
            *MEMORY_TOOLS,
            meta_analysis_agent.as_tool(
                tool_name="meta_analysis_agent",
                tool_description="""An agent that analyzes pipeline trace files and log files to understand run state.

                Use this agent when:
                - User wants to resume a previous run ("continue", "resume", "pick up where we left off")
                - After an error occurs and you need to understand what happened
                - To check if there's an active/incomplete run before starting a new one
                - You need deeper context from a past session (run state, errors, full trace analysis) after using get_session_plan_summary for a brief overview—pass the session's trace file path from get_session_info(session_id).

                The agent will read the trace file and log files and provide:
                - Output folder and log file paths from the previous run
                - Which pipeline steps completed successfully
                - Which steps had errors and error details
                - File paths used (receptor, ligands, docking centers, etc.)
                - Run status (completed, error, incomplete)
                - Entry point used and recommendation for resuming

                CRITICAL: When resuming a run, use the SAME output_folder from the meta analysis.""",
            ),
            file_finder_agent.as_tool(
                tool_name="file_finder_agent",
                tool_description="""An intelligent agent that scans agent_data/ to identify and classify files for the pipeline.

                Use this agent when:
                - User wants to start the pipeline but you're unsure which step to begin from
                - User says they have "existing files", "completed steps", or want to "resume"
                - You need to determine what intermediate files are available
                - User asks "what can I run?" or "where can I start?"

                The agent will:
                1. Scan agent_data/ and its subdirectories (including outputs/)
                2. Identify file types: receptors, SMILES CSVs, docking results, protein topology, pose directories, etc.
                3. Determine which pipeline entry points are available
                4. Recommend the most advanced entry point that has all required files
                5. Report any missing files

                Output is concise: RECOMMENDED ENTRY POINT, RELEVANT PATHS (only the paths required for that entry point), WORKFLOW PARAMETERS, and brief NOTES if needed.""",
            ),
            file_parser_agent.as_tool(
                tool_name="file_parser_agent",
                tool_description="""An agent that extracts structured statistics from pipeline output files to enable parameter extraction and result-based decision making.

                Use this agent when:
                - You need to analyze docking results to determine parameters for the next step (e.g., optimal `tops` parameter for MD)
                - You want to summarize docking results for the user
                - You need to extract statistics from previous runs to inform parameter decisions
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

                Use this agent in conjunction with meta_analysis_agent and file_finder_agent to gather comprehensive information about pipeline state and results.""",
            ),
            oversight_agent.as_tool(
                tool_name="oversight_agent",
                tool_description="""An agent that reviews and validates pipeline execution plans before execution.

            **Before calling this tool:** Check session memory for a similar approved plan: call get_all_session_tags or list_recent_sessions; if a session matches your task, call get_session_plan_summary(session_id) and adapt that plan, then submit to oversight.

            **CRITICAL: You MUST submit your plan to the oversight_agent for review before calling workflow_agent.**

            Use this agent when:
            - You have formulated a plan to execute the pipeline
            - Before calling workflow_agent for the first time in a conversation
            - When proposing significant parameter changes or non-standard workflows
            - When the user request is ambiguous and you need validation of your interpretation

            The oversight agent returns a structured response: exact_plan (the plan text only, no notes) and feedback/concerns/suggestions separately. Use exact_plan as the plan to execute; use feedback and suggestions to inform adjustments.

            The oversight agent will:
            1. Validate that your plan makes scientific sense in the context of molecular docking/biophysics
            2. Check that your plan aligns with the user's request
            3. Review parameter choices for reasonableness and consistency
            4. Identify potential issues before execution
            5. Return the exact plan and any notes/suggestions separately

            **WORKFLOW WITH OVERSIGHT:**
            1. Interpret user intent and gather necessary information
            2. Formulate your execution plan (what you will do, which entry point, parameters)
            3. **Submit plan to oversight_agent for review** - include:
            - The user's original request
            - Your proposed plan (description of steps)
            - Proposed parameters (as a dictionary)
            - Entry point you plan to use
            - Any relevant context (e.g., "resuming from previous run", "user agreed to GPU usage when prompted")
            4. Review the oversight feedback:
            - If approved: Proceed with execution (may still have suggestions to consider)
            - If rejected: Revise plan based on feedback and resubmit
            - Address any concerns or suggestions before proceeding
            5. Execute the validated plan using workflow_agent

            **IMPORTANT:**
            - Always submit plans before execution - oversight helps prevent errors and wasted computation
            - If oversight raises concerns, address them before proceeding
            - You can still proceed if oversight approves with minor suggestions, but consider the feedback
            - For very simple, straightforward requests (e.g., "run docking"), oversight may approve quickly
            - For complex or ambiguous requests, oversight feedback is especially valuable""",
            ),
            workflow_agent.as_tool(
                tool_name="workflow_agent",
                tool_description="An agent that coordinates the complete molecular docking workflow. Use this agent for full pipelines (preprocessing -> docking -> MD), or individual steps. It also supports custom data manipulation and Python code execution via the preprocessing agent. You can specify whether to use the GPU by passing use_gpu=True or use_gpu=False.",
            ),
        ],
        model_settings=ModelSettings(tool_choice="auto"),
    )

    return agent
