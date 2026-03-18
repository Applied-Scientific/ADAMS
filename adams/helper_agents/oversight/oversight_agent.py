"""
Oversight Agent - Reviews and validates pipeline execution plans.
"""

from pathlib import Path

from agents import Agent, ModelSettings

from ...model_config import get_current_model_name, get_resolved_model
from ...pipeline.references.reference_file_reader import read_reference_file
from ...user_plan_utils import read_plan_document
from .oversight_tools import submit_review


prompt_path = Path(__file__).parent / "oversight_prompt.md"
system_prompt = prompt_path.read_text()

_oversight_agent = None
_oversight_model = None


def get_oversight_agent() -> Agent:
    global _oversight_agent, _oversight_model
    current_model = get_current_model_name()
    if _oversight_agent is None or _oversight_model != current_model:
        _oversight_agent = Agent(
            model=get_resolved_model(),
            name="Oversight Agent",
            instructions=system_prompt,
            tools=[
                submit_review,
                read_plan_document,
                read_reference_file,
            ],
            model_settings=ModelSettings(tool_choice="auto"),
        )
        _oversight_model = current_model
    return _oversight_agent
