"""
    adams/helper_agents/oversight/oversight_tools.py

    Tools for the oversight agent to review and validate pipeline plans.
"""

from typing import Any, Dict, List, Optional

from agents import function_tool


@function_tool
def submit_review(
    approved: bool,
    confidence: str,
    exact_plan: str,
    feedback: str,
    concerns: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    parameter_issues: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Submit a structured review of a proposed pipeline execution plan.

    You MUST separate (1) the exact plan text with nothing else, and (2) all notes,
    suggestions, and reasoning. Put only the executable plan in exact_plan; put
    everything else in feedback, concerns, and suggestions.

    Args:
        approved (bool): True if plan is approved for execution, False if it needs changes
        confidence (str): Your confidence level - one of "high", "medium", or "low"
        exact_plan (str): The exact plan text only—no commentary, no notes, no suggestions.
            Copy or lightly edit the submitted plan so it is self-contained and executable.
            This value is stored and reused; keep it plan-only.
        feedback (str): Your review reasoning, notes, and any commentary (not the plan itself)
        concerns (list[str], optional): List of specific concerns or warnings. Use empty list if none.
        suggestions (list[str], optional): List of suggestions for improvement. Use empty list if none.
        parameter_issues (list[str], optional): List of parameter-specific issues found. Use empty list if none.

    Returns:
        dict: approved, confidence, exact_plan, feedback, concerns, suggestions, parameter_issues
    """
    return {
        "approved": approved,
        "confidence": confidence,
        "exact_plan": exact_plan,
        "feedback": feedback,
        "concerns": concerns or [],
        "suggestions": suggestions or [],
        "parameter_issues": parameter_issues or [],
    }
