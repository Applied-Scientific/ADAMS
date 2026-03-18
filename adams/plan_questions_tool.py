"""
Tool for the executive agent to get plan questions so it can address them via user input in chat.

Returns the plan's questions array; the executive presents them to the user in chat,
collects the user's reply, parses answers, and calls append_to_plan_section(plan_path, "answers", {...}).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agents import function_tool


def _load_plan_questions(plan_path: str) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Load plan JSON and return (full data, questions list)."""
    path = Path(plan_path)
    if not path.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions")
    if not isinstance(questions, list):
        questions = []
    return data, questions


def _stage_label(stage: str) -> str:
    """Convert plan stage ids into readable section headings."""
    if not stage:
        return "Questions"
    stage_map = {
        "preprocessing": "Preprocessing",
        "docking": "Docking",
        "md": "MD",
    }
    return stage_map.get(stage, stage.replace("_", " ").title())


def _stage_order_from_plan(data: Dict[str, Any], questions: List[Dict[str, Any]]) -> List[str]:
    """
    Build canonical stage order for question rendering.

    Priority:
    1) Stage order from plan steps (execution order)
    2) Any remaining stages from questions in first-seen order
    3) Known fallback order preprocessing -> docking -> md
    """
    ordered: List[str] = []
    seen = set()

    steps = data.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            stage = str(step.get("stage", "") or "")
            if stage and stage not in seen:
                ordered.append(stage)
                seen.add(stage)

    for question in questions:
        stage = str(question.get("stage", "") or "")
        if stage and stage not in seen:
            ordered.append(stage)
            seen.add(stage)

    for stage in ("preprocessing", "docking", "md"):
        if stage in seen:
            continue
        if any(str(q.get("stage", "") or "") == stage for q in questions):
            ordered.append(stage)
            seen.add(stage)

    if any(str(q.get("stage", "") or "") == "" for q in questions) and "" not in seen:
        ordered.append("")
    return ordered


def _format_questions_for_chat(
    data: Dict[str, Any], questions: List[Dict[str, Any]]
) -> str:
    """
    Render questions into a user-facing chat block that preserves:
    - stage grouping
    - question text
    - human-readable choice labels
    - explicit default markers
    """
    if not questions:
        return "No plan questions."

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for question in questions:
        stage = str(question.get("stage", "") or "")
        if stage not in grouped:
            grouped[stage] = []
        grouped[stage].append(question)
    stage_order = _stage_order_from_plan(data, questions)

    lines = [
        "Please confirm the plan questions below before execution.",
        "Defaults are marked with [default].",
        "Reply with `question_id=value` pairs, one per line or comma-separated. If you want the default for a question, you can either use its default value explicitly or say `use default` for that question.",
    ]

    question_number = 1
    for stage in stage_order:
        stage_questions = grouped.get(stage, [])
        if not stage_questions:
            continue
        lines.append("")
        lines.append(_stage_label(stage))
        lines.append("")  # blank line after section heading so first question is on its own line
        for question in stage_questions:
            question_id = str(question.get("id", f"question_{question_number}"))
            question_text = str(question.get("question", question_id))
            default_value = question.get("default")
            choices = question.get("choices")

            lines.append(f"{question_number}. `{question_id}` - {question_text}")

            if isinstance(choices, list) and choices:
                default_listed = False
                for choice in choices:
                    value = str(choice.get("value", ""))
                    label = str(choice.get("label", value or ""))
                    suffix = ""
                    if default_value is not None and str(default_value) == value:
                        suffix = " [default]"
                        default_listed = True
                    lines.append(f"   - `{value}`: {label}{suffix}")
                if default_value is not None and not default_listed:
                    lines.append(f"   - `{default_value}`: default choice [default]")
            elif default_value is not None:
                lines.append(f"   - `{default_value}`: default answer [default]")
            else:
                lines.append("   - Free-form answer (no predefined choices)")

            question_number += 1
            lines.append("")  # blank line between questions for clear separation

    if lines and lines[-1] == "":
        lines.pop()  # avoid trailing blank line
    return "\n".join(lines)


@function_tool
def collect_plan_answers(plan_path: str) -> dict:
    """
    Get the plan's questions so you can present them to the user in chat and collect answers.

    Call this when you have a plan_path and the plan has a non-empty **questions**
    array. The tool returns the raw questions plus a formatted chat block that preserves
    stage grouping, choice labels, and explicit default markers. Use the formatted
    version as the template you present to the user. When the user replies with their
    answers, parse their response and call append_to_plan_section(plan_path, "answers",
    <dict>) with an object keyed by question id. Then proceed to oversight.

    Args:
        plan_path: Full path to the plan JSON file.

    Returns:
        dict: {"questions": [...], "message": "Present these to the user in chat; when they
              reply, parse answers and call append_to_plan_section(plan_path, 'answers', {...})"}.
              If no questions: {"questions": [], "message": "No questions in plan; proceed to oversight."}.
    """
    data, questions = _load_plan_questions(plan_path)

    if not questions:
        return {
            "questions": [],
            "message": "No questions in plan; proceed to oversight.",
        }

    formatted_questions_markdown = _format_questions_for_chat(data, questions)
    return {
        "questions": questions,
        "formatted_questions_markdown": formatted_questions_markdown,
        "message": "Present these questions to the user in chat using formatted_questions_markdown as the template. Keep the stage headings, question text, choice labels, and [default] markers. Invite answers as question_id=value pairs. When the user replies with their answers, parse the response and call append_to_plan_section(plan_path, 'answers', <dict keyed by question id>), then proceed to oversight.",
    }
