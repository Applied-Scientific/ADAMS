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
    feedback: str,
    concerns: Optional[List[str]] = None,
    suggestions: Optional[List[str]] = None,
    parameter_issues: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Submit a structured review of a proposed pipeline execution plan.

    Use this tool to provide structured feedback after reviewing a plan. This tool
    formats your review results for the requesting agent.

    Args:
        approved (bool): True if plan is approved for execution, False if it needs changes
        confidence (str): Your confidence level - one of "high", "medium", or "low"
        feedback (str): Detailed explanation of your review decision and reasoning
        concerns (list[str], optional): List of specific concerns or warnings. Use empty list if none.
        suggestions (list[str], optional): List of suggestions for improvement. Use empty list if none.
        parameter_issues (list[str], optional): List of parameter-specific issues found. Use empty list if none.

    Returns:
        dict: Structured review result containing all the provided information

    Example:
        >>> result = submit_review(
        ...     approved=True,
        ...     confidence="high",
        ...     feedback="Plan looks good. Parameters are reasonable and workflow is correct.",
        ...     concerns=[],
        ...     suggestions=["Consider using GPU for faster execution if available"],
        ...     parameter_issues=[]
        ... )
    """
    return {
        "approved": approved,
        "confidence": confidence,
        "feedback": feedback,
        "concerns": concerns or [],
        "suggestions": suggestions or [],
        "parameter_issues": parameter_issues or [],
    }
