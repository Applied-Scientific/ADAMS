"""
    adams/helper_agents/oversight/oversight_agent.py

    Oversight Agent - Reviews and validates pipeline execution plans
    to ensure they are scientifically sound and align with user intent.
"""

from pathlib import Path

from agents import Agent, ModelSettings

from .oversight_tools import submit_review


def _load_reference_files(*filenames: str) -> str:
    """
    Load reference markdown files and format them for embedding in system prompts.

    Args:
        *filenames: Names of reference files to load (e.g., "entry_points.md")

    Returns:
        Formatted string containing all reference file contents
    """
    references_dir = Path(__file__).parent.parent.parent / "pipeline" / "references"
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


prompt_path = Path(__file__).parent / "oversight_prompt.md"
base_prompt = prompt_path.read_text()

# Load and embed all 6 reference files needed for validation
reference_docs = _load_reference_files(
    "entry_points.md",
    "workflow_examples.md",
    "parameter_defaults.md",
    "terminology.md",
    "directory_structure.md",
    "file_path_mapping.md",
)

system_prompt = base_prompt + reference_docs

oversight_agent = Agent(
    model="gpt-5.2",
    name="Oversight Agent",
    instructions=system_prompt,
    tools=[
        submit_review,
    ],
    model_settings=ModelSettings(tool_choice="auto"),
)
