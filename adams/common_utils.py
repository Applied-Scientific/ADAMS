"""
Shared utility functions for the adams pipeline.

This module contains general-purpose utilities that are used across
multiple modules (docking, md_analysis, etc.) to avoid cross-module dependencies.
"""

import multiprocessing as mp
import subprocess
import threading
from pathlib import Path
from typing import Optional, Tuple

from .logger_utils import get_logger
from .path_config import get_agent_data_path


def run_cmd(
    cmd,
    input_str=None,
    shell=False,
    check=True,
    capture_output=True,
    env=None,
    cwd=None,
):
    """
    Unified command execution function that redirects executable output directly to logger.

    This function is used throughout the pipeline to execute external commands (GROMACS,
    vina_gpu, OpenBabel, etc.) with their output automatically logged.

    Args:
        cmd: Command as string (for shell=True) or list of arguments (for shell=False)
        input_str: Optional stdin input string
        shell: Whether to use shell execution (default: False for safety)
        check: Whether to raise exception on non-zero return code (default: True)
        capture_output: Whether to capture stdout/stderr and log them (default: True)
        env: Optional environment variables dict (default: None, uses current env)
        cwd: Optional working directory (default: None, uses current directory)

    Returns:
        subprocess.CompletedProcess object

    Raises:
        subprocess.CalledProcessError: If check=True and command fails

    Example:
        >>> from utils import run_cmd
        >>> # Run a simple command
        >>> result = run_cmd(["ls", "-l"])
        >>> # Run with custom environment
        >>> env = os.environ.copy()
        >>> env["CUDA_VISIBLE_DEVICES"] = "0"
        >>> result = run_cmd(["gpu_command"], env=env, cwd="/path/to/dir")
    """
    if isinstance(cmd, str):
        shell = True  # Force shell=True for string commands

    logger = get_logger()
    cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
    logger.info(f"Running: {cmd_str}")

    if capture_output:
        # Use PIPE and read in real-time to avoid fileno() issues with file-like objects
        # This approach works reliably with subprocess.Popen
        def log_stream(stream, log_level="info"):
            """Helper to log lines from stream in real-time."""
            log_func = getattr(logger, log_level.lower(), logger.info)
            for line in iter(stream.readline, ""):
                if line:
                    line = line.rstrip("\n\r")
                    if line:  # Only log non-empty lines
                        log_func(line)
            stream.close()

        # Use Popen with PIPE to avoid fileno() issues
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if input_str else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=shell,
            env=env,
            cwd=cwd,
            bufsize=1,  # Line buffered for real-time output
        )

        # Start threads to log stdout and stderr in real-time
        # Note: Many executables (like GROMACS) write normal output to stderr,
        # so we log both at INFO level to capture all output
        stdout_thread = threading.Thread(
            target=log_stream, args=(process.stdout, "info"), daemon=True
        )
        stderr_thread = threading.Thread(
            target=log_stream, args=(process.stderr, "info"), daemon=True
        )

        stdout_thread.start()
        stderr_thread.start()

        # Send input if provided
        if input_str:
            process.stdin.write(input_str)
            process.stdin.close()

        # Wait for process to complete
        returncode = process.wait()

        # Close pipes to signal EOF to reader threads
        # This ensures threads see EOF immediately and exit naturally
        process.stdout.close()
        process.stderr.close()

        # Wait for logging threads to finish reading all buffered output
        # Threads will exit naturally when they see EOF (empty string from readline)
        stdout_thread.join()
        stderr_thread.join()

        # Create CompletedProcess object for compatibility
        # Note: stdout/stderr are empty strings since output was streamed to logger
        result = subprocess.CompletedProcess(
            cmd,
            returncode,
            stdout="",  # Already logged to logger
            stderr="",  # Already logged to logger
        )

        if check and returncode != 0:
            raise subprocess.CalledProcessError(
                returncode, cmd, result.stdout, result.stderr
            )

        return result
    else:
        # If not capturing output, use standard subprocess.run
        result = subprocess.run(
            cmd, input=input_str, text=True, shell=shell, env=env, cwd=cwd, check=check
        )
        return result


def get_cpu_count() -> int:
    """
    Get the number of available CPU cores on the system.

    Returns:
        int: Number of CPU cores available (typically CPU count - 2 to leave two cores free)

    Example:
        >>> from adams.common_utils import get_cpu_count
        >>> num_cores = get_cpu_count()
        >>> print(f"Using {num_cores} CPU cores")
    """
    return max(1, mp.cpu_count() - 2)  # Leave two cores free for system


def get_gpu_info() -> Tuple[int, str]:
    """
    Get the number of available CUDA-enabled GPUs and their names.

    Returns:
        Tuple[int, str]: Number of GPUs available and a string containing their names,
                         or (0, "") if nvidia-smi is not found.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_names = result.stdout.strip().split("\n")
        return len(gpu_names), ", ".join(gpu_names)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # nvidia-smi not found or returned an error
        return 0, ""


def get_gpu_count() -> int:
    """
    Get the number of available CUDA-enabled GPUs.

    Returns:
        int: Number of GPUs available, or 0 if nvidia-smi is not found.
    """
    gpu_count, _ = get_gpu_info()
    return gpu_count


def ask_to_use_gpu() -> bool:
    """
    Check for GPUs and ask the user if they want to use them.

    Returns:
        bool: True if GPUs are available and the user wants to use them, False otherwise.
    """
    gpu_count, gpu_names = get_gpu_info()
    if gpu_count > 0:
        print(f"Found {gpu_count} GPU(s): {gpu_names}")
        while True:
            answer = input(
                "Do you want to use the GPU(s) for accelerated processing? (y/n): "
            ).lower()
            if answer in ["y", "yes"]:
                return True
            elif answer in ["n", "no"]:
                return False
            else:
                print("Invalid input. Please enter 'y' or 'n'.")
    return False


def list_agent_data_files(agent_data_path: Optional[str] = None) -> dict:
    """
    List all files in the agent_data folder to help identify receptor and ligand files.

    This function scans the agent_data directory and returns information about
    available files, categorizing them by type (receptor PDB/PDBQT files, ligand CSV files).

    Args:
        agent_data_path: Optional. Path to agent_data folder. .

    Returns:
        dict: Dictionary containing:
            - 'agent_data_path': str - Full path to agent_data folder
            - 'receptor_files': list[str] - List of receptor file paths (PDB/PDBQT)
            - 'ligand_files': list[str] - List of ligand CSV file paths
            - 'all_files': list[str] - List of all files found
            - 'error': str - Error message if scan failed (None if successful)

    Example:
        >>> from utils import list_agent_data_files
        >>> result = list_agent_data_files()
        >>> if result['receptor_files']:
        ...     print(f"Found receptor: {result['receptor_files'][0]}")
        >>> if result['ligand_files']:
        ...     print(f"Found ligand: {result['ligand_files'][0]}")
    """
    if agent_data_path is None:
        # Use centralized path configuration
        agent_data_path = get_agent_data_path()
    else:
        agent_data_path = Path(agent_data_path)

    try:
        if not agent_data_path.exists():
            return {
                "agent_data_path": str(agent_data_path),
                "receptor_files": [],
                "ligand_files": [],
                "all_files": [],
                "error": f"agent_data folder does not exist at {agent_data_path}",
            }

        receptor_files = []
        ligand_files = []
        all_files = []

        # Scan for files
        for file_path in agent_data_path.iterdir():
            if file_path.is_file():
                all_files.append(str(file_path))
                file_ext = file_path.suffix.lower()

                # Check for receptor files
                if file_ext in [".pdb", ".pdbqt"]:
                    receptor_files.append(str(file_path))
                # Check for ligand files
                elif file_ext == ".csv":
                    ligand_files.append(str(file_path))

        return {
            "agent_data_path": str(agent_data_path),
            "receptor_files": sorted(receptor_files),
            "ligand_files": sorted(ligand_files),
            "all_files": sorted(all_files),
            "error": None,
        }

    except Exception as e:
        return {
            "agent_data_path": str(agent_data_path),
            "receptor_files": [],
            "ligand_files": [],
            "all_files": [],
            "error": f"Error scanning agent_data folder: {str(e)}",
        }
