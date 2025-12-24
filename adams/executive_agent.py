from pathlib import Path

from agents import Agent, ModelSettings, function_tool
from agents.tracing import add_trace_processor

from .common_utils import ask_to_use_gpu
from .helper_agents.file_finder.file_finder_agent import file_finder_agent
from .helper_agents.file_parser.file_parser_agent import file_parser_agent
from .helper_agents.meta_analysis.meta_analysis_agent import meta_analysis_agent
from .helper_agents.oversight.oversight_agent import oversight_agent
from .path_config import get_agent_data_path, set_agent_data_path
from .pipeline.data_preprocessing.preprocessing_agent import preprocessing_agent
from .pipeline.workflow_agent import workflow_agent
from .utils import list_agent_data_files
from .utils.trace_writer import JsonTraceProcessor


def _load_reference_files(*filenames: str) -> str:
    """
    Load reference markdown files and format them for embedding in system prompts.

    Args:
        *filenames: Names of reference files to load (e.g., "entry_points.md")

    Returns:
        Formatted string containing all reference file contents
    """
    references_dir = Path(__file__).parent / "pipeline" / "references"
    sections = []

    for filename in filenames:
        file_path = references_dir / filename
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8")
            # Extract title from filename (e.g., "entry_points.md" -> "Entry Points")
            title = filename.replace(".md", "").replace("_", " ").title()
            sections.append(f"\n## {title}\n\n{content}")

    if sections:
        return "\n# Reference Documentation\n" + "\n".join(sections)
    return ""


@function_tool
def get_gpu_spec_from_user() -> dict:
    """
    Check for GPUs and ask the user if they want to use them.

    **USAGE**: Only call this function when:
    1. The user's original request did NOT mention GPU usage, AND
    2. You need to check if CUDA GPUs are available and ask the user

    This function will:
    - Check if CUDA GPUs are available (via nvidia-smi)
    - If GPUs found: Ask the user if they want to use them
    - If no GPUs found: Return False without asking

    Do NOT call this if user already requested GPU in their original prompt.

    Returns:
        dict: Contains:
            - 'use_gpu' (bool): True if GPUs are available and user wants to use them
            - 'num_gpus' (int): Number of GPUs detected (0 if none available)
            - 'gpu_names' (str): Names of detected GPUs
    """
    from .common_utils import get_gpu_info

    gpu_count, gpu_names = get_gpu_info()
    use_gpu = ask_to_use_gpu()

    return {"use_gpu": use_gpu, "num_gpus": gpu_count, "gpu_names": gpu_names}


@function_tool
def set_working_directory_tool(
    directory_path: str = None, input_file_path: str = None
) -> dict:
    """
    Set the working directory where agent_data will be created for logs, traces, and outputs.

    Call this tool when:
    - User specifies where their input files are located
    - User tells you a project directory to work in
    - Beginning a new session and need to establish the working location

    Args:
        directory_path: Direct path to use for agent_data parent directory (e.g., "/path/to/project")
        input_file_path: Path to an input file; agent_data will be created in the same directory

    Returns:
        dict: Contains 'agent_data_path' (str) showing where data will be stored

    Example:
        >>> # User says "my files are in /data/project1/"
        >>> set_working_directory_tool(directory_path="/data/project1")
        >>> # agent_data will be at /data/project1/agent_data

        >>> # User says "my receptor is at /data/project1/receptor.pdb"
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


def setup_tracing() -> JsonTraceProcessor:
    """
    Set up JSON trace processor for tracking all agent interactions.

    This is completely separate from the pipeline logger (logger_utils.py).
    Trace files are written to agent_data/traces/ with real-time updates.

    Returns:
        JsonTraceProcessor: The configured trace processor instance

    Raises:
        RuntimeError: If trace processor initialization fails
    """
    try:
        trace_output_dir = get_agent_data_path() / "traces"
        trace_processor = JsonTraceProcessor(output_dir=str(trace_output_dir))
        add_trace_processor(trace_processor)
        print(f"[Tracing] Session trace file: {trace_processor.filepath}")
        return trace_processor
    except Exception as e:
        raise RuntimeError(f"Failed to initialize trace processor: {e}") from e


def create_agent() -> Agent:
    """
    Create and configure the Biophysics Controller Agent.

    Returns:
        Agent: The configured agent instance with all tools and settings
    """
    prompt_path = Path(__file__).parent / "agent_prompt.md"
    base_prompt = prompt_path.read_text()

    # Load and embed reference documentation
    reference_docs = _load_reference_files(
        "entry_points.md",
        "parameter_defaults.md",
        "workflow_examples.md",
    )

    system_prompt = base_prompt + reference_docs

    agent = Agent(
        model="gpt-5.2-pro",
        name="Biophysics Controller Agent",
        instructions=system_prompt,
        tools=[
            set_working_directory_tool,
            list_agent_data_files_tool,
            get_gpu_spec_from_user,
            meta_analysis_agent.as_tool(
                tool_name="meta_analysis_agent",
                tool_description="""An agent that analyzes pipeline trace files and log files to understand run state.

                Use this agent when:
                - User wants to resume a previous run ("continue", "resume", "pick up where we left off")
                - After an error occurs and you need to understand what happened
                - To check if there's an active/incomplete run before starting a new one

                The agent will read the most recent trace file and log files and provide:
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

                Output includes:
                - DETECTED FILES section with paths for each file type
                - AVAILABLE ENTRY POINTS section showing which steps are READY vs MISSING requirements
                - RECOMMENDED ENTRY POINT (most advanced available step)
                - NOTES about any ambiguities or recommendations""",
            ),
            file_parser_agent.as_tool(
                tool_name="file_parser_agent",
                tool_description="""An agent that extracts structured statistics from pipeline output files to enable parameter extraction and result-based decision making.

                Use this agent when:
                - You need to analyze docking results to determine parameters for the next step (e.g., optimal `tops` parameter for MD)
                - You want to summarize docking or MD results for the user
                - You need to extract statistics from previous runs to inform parameter decisions
                - You want to check MD completion status or identify which poses completed
                - You need to analyze affinity distributions, pose counts, or pocket statistics

                The agent can:
                1. Parse docking results CSV files to extract:
                - Affinity statistics (best, average, median, distribution)
                - Pose counts per ligand and per pocket
                - Pocket analysis (which pockets have best affinities)
                - Affinity percentiles and ranges
                2. Parse MD results directories to extract:
                - Completion status (protein topology, ligand prep, MD simulations, analysis)
                - Pose statistics (total prepared, completed, with analysis)
                - File paths (protein files, analysis reports, pose directories)

                Use this agent in conjunction with meta_analysis_agent and file_finder_agent to gather comprehensive information about pipeline state and results.""",
            ),
            oversight_agent.as_tool(
                tool_name="oversight_agent",
                tool_description="""An agent that reviews and validates pipeline execution plans before execution.

            **CRITICAL: You MUST submit your plan to the oversight_agent for review before calling workflow_agent.**

            Use this agent when:
            - You have formulated a plan to execute the pipeline
            - Before calling workflow_agent for the first time in a conversation
            - When proposing significant parameter changes or non-standard workflows
            - When the user request is ambiguous and you need validation of your interpretation

            The oversight agent will:
            1. Validate that your plan makes scientific sense in the context of molecular docking/biophysics
            2. Check that your plan aligns with the user's request
            3. Review parameter choices for reasonableness and consistency
            4. Identify potential issues before execution
            5. Provide feedback and suggestions

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
