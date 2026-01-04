"""
Utility functions for reading reference files.
"""

from pathlib import Path

from agents import function_tool

from ...path_config import validate_path_safety


def read_reference_file_impl(reference_name: str) -> dict:
    """
    Read a reference markdown file from adams/pipeline/references/.

    Args:
        reference_name: Name of the reference file (e.g., "entry_points.md", "parameter_defaults.md")
            Can include or omit the .md extension

    Returns:
        dict with:
            - 'reference_name': Name of the file requested
            - 'file_path': Full path to the file
            - 'content': Full content of the file as string
            - 'error': Error message if reading failed, None if successful
    """
    # Get the references directory path
    references_dir = Path(__file__).parent

    # Normalize the reference name (add .md if not present)
    if not reference_name.endswith(".md"):
        reference_name = reference_name + ".md"

    file_path = references_dir / reference_name

    try:
        validate_path_safety(file_path, references_dir)
    except PermissionError as e:
        return {
            "reference_name": reference_name,
            "file_path": str(file_path),
            "content": None,
            "error": str(e),
        }

    result = {
        "reference_name": reference_name,
        "file_path": str(file_path),
        "content": None,
        "error": None,
    }

    if not file_path.exists():
        result["error"] = f"Reference file not found: {file_path}"
        return result

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        result["content"] = content
    except Exception as e:
        result["error"] = f"Failed to read reference file: {e}"

    return result


@function_tool
def read_reference_file(reference_name: str) -> dict:
    """
    Read a reference markdown file from adams/pipeline/references/.

    This function reads reference documentation files that contain important information
    about pipeline entry points, parameter defaults, directory structures, workflow examples,
    error handling, and file path mappings.

    Use this when:
    - You need to understand entry point requirements
    - You need to check parameter defaults or guidelines
    - You need to understand directory structures or file path mappings
    - You need workflow examples or error handling patterns

    Available reference files:
    - `entry_points.md` - Entry point definitions and requirements for starting the pipeline at different stages
    - `parameter_defaults.md` - Default parameter values and guidelines for all pipeline steps
    - `directory_structure.md` - Directory structure and file organization patterns
    - `file_path_mapping.md` - File path mapping rules between pipeline agents
    - `workflow_examples.md` - Example workflow prompts and patterns
    - `error_handling.md` - Error handling patterns and report formats

    Args:
        reference_name (str): Name of the reference file to read.
            Examples: "entry_points.md", "parameter_defaults.md", "directory_structure.md",
            "workflow_examples.md", "error_handling.md", "file_path_mapping.md"
            Can include or omit the .md extension.

    Returns:
        dict: Dictionary containing:
            - 'reference_name' (str): Name of the file requested
            - 'file_path' (str): Full path to the reference file
            - 'content' (str or None): Full content of the file as string, None if error
            - 'error' (str or None): Error message if reading failed, None if successful

    Example:
        >>> result = read_reference_file("entry_points.md")
        >>> # Returns full content of entry_points.md

        >>> result = read_reference_file("parameter_defaults")
        >>> # .md extension added automatically
    """
    return read_reference_file_impl(reference_name)
