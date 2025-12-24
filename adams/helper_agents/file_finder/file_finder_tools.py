"""
    Utility functions for scanning and identifying files in the current working directory.
    These functions are used by the File Finding Agent.
"""

import csv
import os
from pathlib import Path


def scan_directory_impl(path: str = "") -> dict:
    """
    Scan a directory and return information about all files and subdirectories.

    Args:
        path: Path to scan, relative to the current working directory. Empty string scans the CWD.

    Returns:
        dict with:
            - 'path': Absolute path scanned
            - 'files': List of {name, size_bytes, extension, full_path} for files
            - 'directories': List of {name, full_path} for directories
            - 'error': Error message if any
    """
    base_path = Path.cwd()
    if path:
        target = base_path / path
    else:
        target = base_path

    result = {"path": str(target), "files": [], "directories": [], "error": None}

    try:
        if not target.exists():
            result["error"] = f"Path does not exist: {target}"
            return result

        for item in sorted(target.iterdir()):
            if item.is_file():
                result["files"].append(
                    {
                        "name": item.name,
                        "size_bytes": item.stat().st_size,
                        "extension": item.suffix.lower(),
                        "full_path": str(item),
                    }
                )
            elif item.is_dir():
                result["directories"].append(
                    {"name": item.name, "full_path": str(item)}
                )
    except Exception as e:
        result["error"] = str(e)

    return result


def read_csv_headers_impl(file_path: str) -> dict:
    """
    Read the column headers from a CSV file to help identify its type.

    Args:
        file_path: Path to CSV file (can be relative to CWD or absolute)

    Returns:
        dict with:
            - 'file_path': Path to file
            - 'columns': List of column names
            - 'row_count': Approximate number of data rows (first 100 sampled)
            - 'sample_values': Dict of column -> first few values (for identification)
            - 'error': Error message if any
    """
    base_path = Path.cwd()

    # Handle relative vs absolute paths
    if os.path.isabs(file_path):
        target = Path(file_path)
    else:
        target = base_path / file_path

    result = {
        "file_path": str(target),
        "columns": [],
        "row_count": 0,
        "sample_values": {},
        "error": None,
    }

    try:
        with open(target, "r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, None)

            if not headers:
                result["error"] = "CSV file is empty or has no headers"
                return result

            result["columns"] = [h.strip() for h in headers]

            # Sample first few rows
            sample_values = {col: [] for col in result["columns"]}
            row_count = 0

            for row in reader:
                row_count += 1
                if row_count <= 3:  # Sample first 3 rows
                    for i, val in enumerate(row):
                        if i < len(result["columns"]):
                            sample_values[result["columns"][i]].append(val.strip())
                if row_count >= 100:  # Stop counting after 100
                    break

            result["row_count"] = row_count
            result["sample_values"] = sample_values

    except Exception as e:
        result["error"] = str(e)

    return result


def check_file_exists_impl(file_path: str) -> dict:
    """
    Check if a specific file exists and get basic info.

    Args:
        file_path: Path to check (relative to CWD or absolute)

    Returns:
        dict with exists, is_file, is_directory, size_bytes, extension
    """
    base_path = Path.cwd()

    if os.path.isabs(file_path):
        target = Path(file_path)
    else:
        target = base_path / file_path

    return {
        "file_path": str(target),
        "exists": target.exists(),
        "is_file": target.is_file() if target.exists() else False,
        "is_directory": target.is_dir() if target.exists() else False,
        "size_bytes": target.stat().st_size if target.is_file() else None,
        "extension": target.suffix.lower() if target.is_file() else None,
    }


def check_directory_contents_impl(dir_path: str, required_files: str) -> dict:
    """
    Check if a directory contains specific files (useful for checking pose directories).

    Args:
        dir_path: Path to directory (relative to CWD or absolute)
        required_files: Comma-separated list of required file names (e.g., "min.gro,system.top,index.ndx")

    Returns:
        dict with:
            - 'dir_path': Full path to directory
            - 'exists': Whether directory exists
            - 'required_files': List of required files
            - 'found_files': List of required files that were found
            - 'missing_files': List of required files that are missing
            - 'all_present': True if all required files exist
    """
    base_path = Path.cwd()

    if os.path.isabs(dir_path):
        target = Path(dir_path)
    else:
        target = base_path / dir_path

    required = [f.strip() for f in required_files.split(",")]

    result = {
        "dir_path": str(target),
        "exists": target.exists() and target.is_dir(),
        "required_files": required,
        "found_files": [],
        "missing_files": [],
        "all_present": False,
    }

    if result["exists"]:
        for req_file in required:
            if (target / req_file).exists():
                result["found_files"].append(req_file)
            else:
                result["missing_files"].append(req_file)
        result["all_present"] = len(result["missing_files"]) == 0
    else:
        result["missing_files"] = required

    return result


def read_file_preview_impl(file_path: str, lines: int = 20) -> dict:
    """
    Read the first N lines of a text file to help identify its contents.

    Args:
        file_path: Path to file (relative to CWD or absolute)
        lines: Number of lines to read (default 20)

    Returns:
        dict with:
            - 'file_path': Full path
            - 'content': First N lines of file
            - 'total_lines': Total line count (if < 1000 lines)
            - 'error': Error message if any
    """
    base_path = Path.cwd()

    if os.path.isabs(file_path):
        target = Path(file_path)
    else:
        target = base_path / file_path

    result = {"file_path": str(target), "content": "", "total_lines": 0, "error": None}

    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            content_lines = []
            total = 0
            for i, line in enumerate(f):
                total += 1
                if i < lines:
                    content_lines.append(line.rstrip("\n\r"))
                if total > 1000:
                    break

            result["content"] = "\n".join(content_lines)
            result["total_lines"] = total if total <= 1000 else ">1000"

    except Exception as e:
        result["error"] = str(e)

    return result
