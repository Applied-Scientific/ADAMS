"""
    File Finding Agent - Intelligently identifies and classifies files in the current
    working directory to determine which pipeline entry points are available.
"""

from pathlib import Path

from agents import Agent, ModelSettings, function_tool

from ...pipeline.references.reference_file_reader import read_reference_file
from .file_finder_tools import (
    check_directory_contents_impl,
    check_file_exists_impl,
    read_csv_headers_impl,
    read_file_preview_impl,
    scan_directory_impl,
)


@function_tool
def scan_directory(path: str = "") -> dict:
    """
    Scan a directory and return information about all files and subdirectories.

    This function recursively scans a directory structure and returns metadata about
    all files and subdirectories found. Use this as the primary tool for discovering
    files in the current working directory.

    Use this when:
    - You need to discover what files exist in the current working directory or its subdirectories
    - You're looking for specific file types (receptors, CSVs, docking results, etc.)
    - You need to explore the directory structure to understand available files

    Args:
        path (str): Path to scan, relative to the current working directory.
            Empty string ("") scans the current working directory.
            Example: "" for root, "outputs" for outputs subdirectory,
            "outputs/run_20251203/md_analysis" for nested path.

    Returns:
        dict: Dictionary containing:
            - 'path' (str): Absolute path that was scanned
            - 'files' (list): List of dictionaries, each containing:
                - 'name' (str): Filename
                - 'size_bytes' (int): File size in bytes
                - 'extension' (str): File extension (e.g., '.pdb', '.csv')
                - 'full_path' (str): Full absolute path to the file
            - 'directories' (list): List of dictionaries, each containing:
                - 'name' (str): Directory name
                - 'full_path' (str): Full absolute path to the directory
            - 'error' (str or None): Error message if scan failed, None if successful

    Example:
        >>> result = scan_directory("")
        >>> # Scans the current working directory, returns all files and directories

        >>> result = scan_directory("outputs")
        >>> # Scans the ./outputs directory
    """
    return scan_directory_impl(path)


@function_tool
def read_csv_headers(file_path: str) -> dict:
    """
    Read the column headers from a CSV file to help identify its type.

    This function reads the first row (headers) and optionally samples data rows
    from a CSV file to help classify the file type. Use this to distinguish between
    different CSV types (ligand libraries, docking results, docking centers, etc.).

    Use this when:
    - You need to identify what type of CSV file you're looking at
    - You want to verify CSV structure matches expected format
    - You need to check column names to classify files

    Args:
        file_path (str): Path to CSV file. Can be relative to the current working
            directory or an absolute path. Example: "input.csv" or
            "agent_data/run_20251203/docking_results.csv"

    Returns:
        dict: Dictionary containing:
            - 'file_path' (str): Full path to the CSV file
            - 'columns' (list): List of column names from the header row
            - 'row_count' (int): Approximate number of data rows (sampled from first 100 rows)
            - 'sample_values' (dict): Dictionary mapping column names to lists of
                first few values from that column (useful for identification)
            - 'error' (str or None): Error message if reading failed, None if successful

    Example:
        >>> result = read_csv_headers("input.csv")
        >>> # Returns column names like ['ID', 'SMILES', 'MolWt'] for ligand CSV

        >>> result = read_csv_headers("docking_centers.csv")
        >>> # Returns column names like ['center_x', 'center_y', 'center_z'] for docking centers
    """
    return read_csv_headers_impl(file_path)


@function_tool
def check_file_exists(file_path: str) -> dict:
    """
    Check if a specific file exists and get basic metadata.

    This function verifies file existence and returns basic information about
    the file or directory. Use this to validate specific file paths before using them.

    Use this when:
    - You need to verify a specific file path exists
    - You want to check if a path points to a file or directory
    - You need basic file metadata (size, extension)

    Args:
        file_path (str): Path to check. Can be relative to the current working
            directory or an absolute path. Example: "2ppn.pdb" or
            "agent_data/run_20251203/protein.gro"

    Returns:
        dict: Dictionary containing:
            - 'exists' (bool): True if the path exists, False otherwise
            - 'is_file' (bool): True if path is a file, False otherwise
            - 'is_directory' (bool): True if path is a directory, False otherwise
            - 'size_bytes' (int or None): File size in bytes if it's a file, None otherwise
            - 'extension' (str or None): File extension (e.g., '.pdb', '.csv') if it's a file, None otherwise

    Example:
        >>> result = check_file_exists("2ppn.pdb")
        >>> # Returns: {'exists': True, 'is_file': True, 'size_bytes': 12345, 'extension': '.pdb'}

        >>> result = check_file_exists("nonexistent.txt")
        >>> # Returns: {'exists': False, 'is_file': False, ...}
    """
    return check_file_exists_impl(file_path)


@function_tool
def check_directory_contents(dir_path: str, required_files: str) -> dict:
    """
    Check if a directory contains specific required files.

    This function verifies that a directory contains all specified files. It's particularly
    useful for checking pose directories that should contain specific MD simulation files
    (e.g., min.gro, system.top, index.ndx) or completed MD directories (md.tpr, md.xtc, md.gro).

    Use this when:
    - You need to verify a pose directory has all required files for MD simulation
    - You want to check if an MD directory has completed simulation files
    - You need to validate directory completeness before proceeding

    Args:
        dir_path (str): Path to directory to check. Can be relative to the current
            working directory or an absolute path. Example:
            "agent_data/run_20251203/md_analysis/poses/ligand_pocket_0_top1"
        required_files (str): Comma-separated list of required file names.
            Example: "min.gro,system.top,index.ndx" or "md.tpr,md.xtc,md.gro"
            Files are checked by exact name match (case-sensitive).

    Returns:
        dict: Dictionary containing:
            - 'dir_path' (str): Full absolute path to the directory checked
            - 'exists' (bool): True if directory exists, False otherwise
            - 'required_files' (list): List of required file names (parsed from input)
            - 'found_files' (list): List of required files that were found in the directory
            - 'missing_files' (list): List of required files that are missing
            - 'all_present' (bool): True if all required files exist, False otherwise

    Example:
        >>> result = check_directory_contents("agent_data/run_1/poses/ligand_1", "min.gro,system.top,index.ndx")
        >>> # Returns: {'all_present': True, 'found_files': ['min.gro', 'system.top', 'index.ndx'], ...}

        >>> result = check_directory_contents("agent_data/run_1/md_analysis/poses/ligand1", "md.tpr,md.xtc,md.gro")
        >>> # Checks for completed MD simulation files
    """
    return check_directory_contents_impl(dir_path, required_files)


@function_tool
def read_file_preview(file_path: str, lines: int = 20) -> dict:
    """
    Read the first N lines of a text file to help identify its contents.

    This function reads a preview of a text file to help understand its format and content.
    Useful for examining file headers, structure, or content when file type is ambiguous.

    Use this when:
    - You need to examine file contents to understand format
    - You want to check file structure or headers
    - File type is ambiguous and you need to inspect content

    Args:
        file_path (str): Path to file to read. Can be relative to the current
            working directory or an absolute path. Example: "protein.pdb" or
            "agent_data/run_20251203/log.txt"
        lines (int): Number of lines to read from the beginning of the file.
            Default: 20. Maximum recommended: 100 lines for efficiency.

    Returns:
        dict: Dictionary containing:
            - 'file_path' (str): Full absolute path to the file
            - 'content' (str): First N lines of the file as a single string
            - 'total_lines' (int or None): Total line count if file has < 1000 lines,
                None for larger files (to avoid reading entire file)
            - 'error' (str or None): Error message if reading failed, None if successful

    Example:
        >>> result = read_file_preview("protein.pdb", lines=10)
        >>> # Returns first 10 lines of PDB file

        >>> result = read_file_preview("docking_results.csv", lines=5)
        >>> # Returns first 5 lines including header
    """
    return read_file_preview_impl(file_path, lines)


prompt_path = Path(__file__).parent / "file_finder_prompt.md"
system_prompt = prompt_path.read_text()

file_finder_agent = Agent(
    model="gpt-5-mini",
    name="File Finder Agent",
    instructions=system_prompt,
    tools=[
        read_reference_file,
        scan_directory,
        read_csv_headers,
        check_file_exists,
        check_directory_contents,
        read_file_preview,
    ],
    model_settings=ModelSettings(tool_choice="auto"),
)
