"""
Shared utilities for plan-mode coordination.

Plan documents are JSON files under agent_data/plans/ with structured sections:
- user_prompt: string (the user request that led to this plan; set once for matching and context)
- steps: list of { "stage", "description", "details" } (pipeline steps per stage)
- parameters: dict keyed by stage (e.g. "preprocessing", "docking", "md") with key-value params
- questions: list of { "id", "stage", "question", "choices"?, "default"? } for user prompts
- answers: dict of question_id -> value (user responses to plan questions; filled after approval)
- additional_notes: list of strings (freeform notes)
- tags: list of strings (for discovery; workflow sets tag for current plan; executive uses for list_plans_by_tag)

User interaction is only through the executive: plan questions are presented by the executive
and answers are recorded in the plan before execution. Stage agents do not request user input
during execution.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from contextlib import contextmanager
import fcntl
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import function_tool

from .path_config import get_agent_data_path


# Tags are set via set_plan_tags(), not append_to_plan_section.
PLAN_SECTIONS = ("user_prompt", "steps", "parameters", "questions", "answers", "additional_notes")
VALID_STAGES = {"preprocessing", "docking", "md"}

# Keys to clear when cloning a plan (run/input-specific). Structural params (pH, num_pockets, etc.) are preserved.
PARAMS_KEYS_TO_CLEAR: Dict[str, List[str]] = {
    "preprocessing": [
        "outpath",
        "input_pdb",
        "input_data",
        "input_ligands",
        "protonated_pdb",
        "docking_ready_ligands_csv",
    ],
    "docking": [
        "out_folder",
        "receptor",
        "input_data",
        "expected_production_results_csv",
    ],
    "md": [
        "md_workdir",
    ],
}


def _plans_dir() -> Path:
    """Return agent_data/plans path and ensure it exists."""
    plans_dir = get_agent_data_path() / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    return plans_dir


def _default_plan() -> Dict[str, Any]:
    return {
        "user_prompt": "",
        "steps": [],
        "parameters": {},
        "questions": [],
        "answers": {},
        "additional_notes": [],
        "tags": [],
    }


def _ensure_plan_shape(data: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure required top-level sections exist and tags are normalized."""
    if not isinstance(data, dict):
        raise ValueError("plan document must be a JSON object")
    default = _default_plan()
    for key, value in default.items():
        if key not in data:
            data[key] = value
    data["tags"] = _normalize_tags(data.get("tags"))
    return data


@contextmanager
def _plan_file_lock(path: Path):
    """
    Acquire an exclusive advisory lock for a plan file.

    This prevents concurrent read-modify-write races between stage-agent
    contributions that could otherwise corrupt plan JSON.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _recover_first_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """
    Recover the first complete top-level JSON object from malformed content.

    Useful when accidental trailing data exists due to interrupted/concurrent writes.
    """
    if not raw or not raw.strip():
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(raw.lstrip())
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _write_plan_data(path: Path, data: Dict[str, Any]) -> None:
    """Atomically write plan JSON to disk to avoid partial/corrupt writes."""
    data = _ensure_plan_shape(data)
    fd = -1
    tmp_path = ""
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=str(path.parent),
        )
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_file:
            fd = -1  # ownership transferred to tmp_file
            json.dump(data, tmp_file, indent=2)
            tmp_file.flush()
            os.fsync(tmp_file.fileno())
        os.replace(tmp_path, path)
    finally:
        if fd >= 0:
            os.close(fd)
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _load_plan_data(path: Path, repair_on_recoverable_error: bool = False) -> Dict[str, Any]:
    """
    Load plan JSON with optional recover-and-repair behavior.

    If parsing fails due to trailing or interleaved data, recover the first
    valid JSON object and optionally rewrite the file in canonical form.
    """
    raw = path.read_text(encoding="utf-8")
    try:
        return _ensure_plan_shape(json.loads(raw))
    except json.JSONDecodeError as e:
        recovered = _recover_first_json_object(raw)
        if recovered is None:
            raise ValueError(f"Plan file contains invalid JSON: {path} ({e})") from e
        recovered = _ensure_plan_shape(recovered)
        if repair_on_recoverable_error:
            _write_plan_data(path, recovered)
        return recovered


def _normalize_tags(value: Any) -> List[str]:
    """Normalize tags to a deduplicated list[str] preserving first-seen order."""
    if not isinstance(value, list):
        return []
    normalized: List[str] = []
    seen = set()
    for t in value:
        tag = str(t).strip()
        if not tag or tag in seen:
            continue
        normalized.append(tag)
        seen.add(tag)
    return normalized


def _sanitize_plan_name(name: Optional[str]) -> str:
    """Sanitize optional name for use in plan filename: alphanumeric and underscore only, max 32 chars."""
    if not name or not str(name).strip():
        return ""
    s = "".join(c if c.isalnum() or c == "_" else "_" for c in str(name).strip())
    return s[:32] if len(s) > 32 else s


def _generate_unique_plan_path(prefix: str = "plan", name: Optional[str] = None) -> Path:
    """Return a Path under agent_data/plans that does not exist. Uses plan_{name}_{timestamp}.json or plan_{timestamp}.json; on collision appends _1, _2, etc."""
    plans_dir = _plans_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sanitized = _sanitize_plan_name(name)
    if sanitized:
        base = f"{prefix}_{sanitized}_{timestamp}"
    else:
        base = f"{prefix}_{timestamp}"
    plan_path = plans_dir / f"{base}.json"
    suffix = 0
    while plan_path.exists():
        suffix += 1
        plan_path = plans_dir / f"{base}_{suffix}.json"
    return plan_path


def _create_plan_path_impl(prefix: str = "plan", name: Optional[str] = None) -> str:
    """Create a new plan document under agent_data/plans. Used by the tool and by the workflow wrapper. Supports plan_{name}_{timestamp}.json with collision handling."""
    plan_path = _generate_unique_plan_path(prefix=prefix, name=name)
    _write_plan_data(plan_path, _default_plan())
    return str(plan_path)


@function_tool
def create_plan_path(prefix: str = "plan", name: Optional[str] = None) -> str:
    """
    Create a new plan document (JSON) under agent_data/plans.

    The plan has six sections: user_prompt, steps, parameters, questions, answers, additional_notes.
    Use read_plan_document to inspect and append_to_plan_section to add content.

    Filename is plan_{name}_{timestamp}.json when name is provided, else plan_{timestamp}.json.
    If a file with that path already exists, a numeric suffix is added to avoid overwriting.

    Args:
        prefix: Filename prefix for the plan document (default "plan").
        name: Optional short name for the plan (e.g. "docking_only"); sanitized for filesystem.

    Returns:
        Full path to the created plan JSON file.
    """
    return _create_plan_path_impl(prefix=prefix, name=name)


@function_tool
def read_plan_document(plan_path: str) -> str:
    """
    Read the current plan document (JSON). Returns pretty-printed JSON.

    Structure: { "user_prompt": str, "steps": [...], "parameters": {...}, "questions": [...], "answers": {...}, "additional_notes": [...] }

    Args:
        plan_path: Full path to the plan JSON file.

    Returns:
        Pretty-printed JSON string of the full plan.
    """
    path = Path(plan_path)
    if not path.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")
    with _plan_file_lock(path):
        data = _load_plan_data(path, repair_on_recoverable_error=True)
    return json.dumps(data, indent=2)


def _sanitize_control_chars(s: str) -> str:
    """Replace ASCII control characters (except tab, newline, carriage return) with space."""
    return "".join(
        c if (ord(c) >= 32 or c in "\t\n\r") else " " for c in s
    )


def _parse_json_content(raw: str) -> Any:
    """
    Parse JSON content with tolerance for one trailing value separator pattern.

    LLM outputs sometimes include trailing data after one complete JSON value
    (for example: ["a", "b"],). In that case, parse up to the first complete value.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        if "Extra data" in str(e) and getattr(e, "pos", None) is not None:
            try:
                return json.loads(raw[: e.pos])
            except json.JSONDecodeError:
                raise ValueError(f"content must be valid JSON: {e}") from e
        raise ValueError(f"content must be valid JSON: {e}") from e


def _validate_step(step: Any) -> Dict[str, Any]:
    """Validate a single step object and normalize its details list."""
    if not isinstance(step, dict):
        raise ValueError("each step must be a JSON object")
    stage = step.get("stage")
    if not isinstance(stage, str) or stage not in VALID_STAGES:
        raise ValueError(
            f"step.stage must be one of {sorted(VALID_STAGES)}, got {stage!r}"
        )
    description = step.get("description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("step.description must be a non-empty string")

    details = step.get("details", [])
    if details is None:
        details = []
    if not isinstance(details, list):
        raise ValueError("step.details must be a list of strings")
    normalized_details = [str(d) for d in details]

    return {
        "stage": stage,
        "description": description,
        "details": normalized_details,
    }


def _validate_question(question: Any) -> Dict[str, Any]:
    """Validate a single question object and normalize optional fields."""
    if not isinstance(question, dict):
        raise ValueError("each question must be a JSON object")
    question_id = question.get("id")
    stage = question.get("stage")
    prompt = question.get("question")
    if not isinstance(question_id, str) or not question_id.strip():
        raise ValueError("question.id must be a non-empty string")
    if not isinstance(stage, str) or stage not in VALID_STAGES:
        raise ValueError(
            f"question.stage must be one of {sorted(VALID_STAGES)}, got {stage!r}"
        )
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("question.question must be a non-empty string")

    normalized: Dict[str, Any] = {"id": question_id, "stage": stage, "question": prompt}
    if "choices" in question:
        choices = question["choices"]
        if not isinstance(choices, list):
            raise ValueError("question.choices must be a list when provided")
        normalized["choices"] = choices
    if "default" in question:
        normalized["default"] = question["default"]
    return normalized


@function_tool
def append_to_plan_section(
    plan_path: str,
    section: str,
    content: str,
) -> str:
    """
    Append content to a specific section of the plan (or set/replace for user_prompt).
    Content must be valid JSON except for user_prompt, which accepts raw text (control characters are sanitized).

    Sections and expected content shape:
    - user_prompt: Set or replace the plan's user request (verbatim). Pass as
      a JSON-encoded string or raw text; if JSON parsing fails, content is used
      as-is after stripping and sanitizing control characters. This section is replaced, not appended.
    - steps: JSON array of objects, or single object. Each: { "stage": "preprocessing"|"docking"|"md", "description": str, "details": list of str }
    - parameters: JSON object keyed by stage, e.g. {"preprocessing": {"num_confs": 8, "pH": 7.4}}
    - questions: JSON array of objects, or single object. Each: { "id": str, "stage": str, "question": str, "choices": [{"value": str, "label": str}], "default": str (optional) }
    - answers: JSON object of question_id -> value, e.g. {"pocket_choice": "center_1"}. Merged into plan after user responds.
    - additional_notes: JSON array of strings, or single string

    Args:
        plan_path: Full path to the plan JSON file.
        section: One of "user_prompt", "steps", "parameters", "questions", "answers", "additional_notes".
        content: JSON string to merge/append into that section (or raw text for user_prompt).

    Returns:
        plan_path for convenience.
    """
    path = Path(plan_path)
    if not path.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")
    if section not in PLAN_SECTIONS:
        raise ValueError(
            f"section must be one of {PLAN_SECTIONS!r}, got {section!r}"
        )

    raw = str(content).strip()
    if section == "user_prompt":
        try:
            payload = json.loads(raw)
            text = payload if isinstance(payload, str) else str(payload)
        except json.JSONDecodeError:
            text = _sanitize_control_chars(raw)
        payload = text
    else:
        payload = _parse_json_content(raw)

    with _plan_file_lock(path):
        data = _load_plan_data(path, repair_on_recoverable_error=True)

        if section == "user_prompt":
            data["user_prompt"] = payload if isinstance(payload, str) else str(payload)
        elif section == "steps":
            if isinstance(payload, list):
                data["steps"].extend(_validate_step(step) for step in payload)
            else:
                data["steps"].append(_validate_step(payload))
        elif section == "parameters":
            if not isinstance(payload, dict):
                raise ValueError("parameters content must be a JSON object")
            for stage_key, stage_params in payload.items():
                if stage_key not in VALID_STAGES:
                    raise ValueError(
                        f"parameters keys must be stage names {sorted(VALID_STAGES)}, got {stage_key!r}"
                    )
                if not isinstance(stage_params, dict):
                    raise ValueError(
                        f"parameters for stage {stage_key!r} must be a JSON object"
                    )
                if stage_key not in data["parameters"]:
                    data["parameters"][stage_key] = {}
                data["parameters"][stage_key].update(stage_params)
        elif section == "questions":
            if isinstance(payload, list):
                data["questions"].extend(_validate_question(q) for q in payload)
            else:
                data["questions"].append(_validate_question(payload))
        elif section == "answers":
            if not isinstance(payload, dict):
                raise ValueError("answers content must be a JSON object (question_id -> value)")
            data["answers"].update(payload)
        elif section == "additional_notes":
            if isinstance(payload, list):
                data["additional_notes"].extend(str(n) for n in payload)
            else:
                data["additional_notes"].append(str(payload))

        _write_plan_data(path, data)
    return str(path)


@function_tool
def append_to_step_details(
    plan_path: str, stage: str, content: str, step_index: int = 0
) -> str:
    """
    Append implementation details to an existing step's details array (by stage and index).

    The workflow agent adds the step skeleton first (ordered steps with stage, description,
    and empty details). There can be multiple steps with the same stage (e.g. two docking
    steps). Stage agents use this tool to add implementation detail bullets for the
    step they are filling. Use step_index when the plan has more than one step for
    that stage: 0 = first step of this stage, 1 = second, etc.

    Args:
        plan_path: Full path to the plan JSON file.
        stage: Stage identifier (e.g. "preprocessing", "docking", "md").
        content: JSON string: either a single detail string or a list of detail strings.
            Example: '["run_clean_pdb", "run_protonate_receptor"]' or '"Single bullet."'
        step_index: Zero-based index among steps with this stage (default 0).
            Use 0 for the first step of this stage, 1 for the second, etc. The workflow
            specifies which step index when it calls the stage agent for that step.

    Returns:
        plan_path for convenience.

    Raises:
        FileNotFoundError: If plan file does not exist.
        ValueError: If no step with the given stage exists, or step_index is out of range.
    """
    path = Path(plan_path)
    if not path.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")
    with _plan_file_lock(path):
        data = _load_plan_data(path, repair_on_recoverable_error=True)
        steps = data.get("steps", [])
        if not isinstance(steps, list):
            steps = []
        # Collect steps with matching stage in order
        steps_with_stage = [
            s for s in steps
            if isinstance(s, dict) and s.get("stage") == stage
        ]
        if step_index < 0 or step_index >= len(steps_with_stage):
            raise ValueError(
                f"No step with stage {stage!r} at step_index={step_index}. "
                f"Plan has {len(steps_with_stage)} step(s) for stage {stage!r} (valid step_index: 0 to {len(steps_with_stage) - 1}). "
                "Workflow must add the step skeleton first."
            )
        step = steps_with_stage[step_index]
        payload = _parse_json_content(content.strip())
        details = step.get("details")
        if not isinstance(details, list):
            details = []
        if isinstance(payload, list):
            details.extend(str(d) for d in payload)
        else:
            details.append(str(payload))
        step["details"] = details
        _write_plan_data(path, data)
    return str(path)


@function_tool
def contribute_stage_to_plan(
    plan_path: str,
    stage: str,
    step_index: int,
    step_details: str,
    parameters: Optional[str] = None,
    questions: Optional[str] = None,
    additional_notes: Optional[str] = None,
) -> str:
    """
    Contribute all plan-mode updates for one stage step in a single call.

    This atomic tool lets a stage agent update its step details plus optional
    parameters/questions/additional_notes at once, reducing multi-call latency
    during planning.

    Args:
        plan_path: Full path to the plan JSON file.
        stage: Stage identifier ("preprocessing", "docking", "md").
        step_index: Zero-based index among steps of this stage.
        step_details: JSON string (list of strings or single string) appended to
            the target step's details array.
        parameters: Optional JSON string. Either:
            - object keyed by stage, e.g. {"preprocessing": {"outpath": "..."}}
            - stage-only object, e.g. {"outpath": "..."} (wrapped to the given stage)
        questions: Optional JSON string. Question object or list of question objects.
        additional_notes: Optional JSON string. String or list of strings.

    Returns:
        plan_path for convenience.
    """
    path = Path(plan_path)
    if not path.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")
    if stage not in VALID_STAGES:
        raise ValueError(f"stage must be one of {sorted(VALID_STAGES)}, got {stage!r}")
    if step_index < 0:
        raise ValueError("step_index must be >= 0")

    with _plan_file_lock(path):
        data = _load_plan_data(path, repair_on_recoverable_error=True)

        # Step details update (same semantics as append_to_step_details)
        steps = data.get("steps", [])
        if not isinstance(steps, list):
            steps = []
        steps_with_stage = [s for s in steps if isinstance(s, dict) and s.get("stage") == stage]
        if step_index >= len(steps_with_stage):
            raise ValueError(
                f"No step with stage {stage!r} at step_index={step_index}. "
                f"Plan has {len(steps_with_stage)} step(s) for stage {stage!r} "
                f"(valid step_index: 0 to {len(steps_with_stage) - 1}). "
                "Workflow must add the step skeleton first."
            )
        step = steps_with_stage[step_index]
        step_payload = _parse_json_content(str(step_details).strip())
        details = step.get("details")
        if not isinstance(details, list):
            details = []
        if isinstance(step_payload, list):
            details.extend(str(d) for d in step_payload)
        else:
            details.append(str(step_payload))
        step["details"] = details

        # Parameters update (same merge semantics as append_to_plan_section "parameters")
        if parameters is not None and str(parameters).strip():
            param_payload = _parse_json_content(str(parameters).strip())
            if not isinstance(param_payload, dict):
                raise ValueError("parameters content must be a JSON object")
            if stage in param_payload:
                merge_payload = param_payload
            else:
                merge_payload = {stage: param_payload}

            for stage_key, stage_params in merge_payload.items():
                if stage_key not in VALID_STAGES:
                    raise ValueError(
                        f"parameters keys must be stage names {sorted(VALID_STAGES)}, got {stage_key!r}"
                    )
                if not isinstance(stage_params, dict):
                    raise ValueError(
                        f"parameters for stage {stage_key!r} must be a JSON object"
                    )
                if stage_key not in data["parameters"]:
                    data["parameters"][stage_key] = {}
                data["parameters"][stage_key].update(stage_params)

        # Questions update (same semantics as append_to_plan_section "questions")
        if questions is not None and str(questions).strip():
            q_payload = _parse_json_content(str(questions).strip())
            if isinstance(q_payload, list):
                data["questions"].extend(_validate_question(q) for q in q_payload)
            else:
                data["questions"].append(_validate_question(q_payload))

        # Additional notes update (same semantics as append_to_plan_section "additional_notes")
        if additional_notes is not None and str(additional_notes).strip():
            n_payload = _parse_json_content(str(additional_notes).strip())
            if isinstance(n_payload, list):
                data["additional_notes"].extend(str(n) for n in n_payload)
            else:
                data["additional_notes"].append(str(n_payload))

        _write_plan_data(path, data)
    return str(path)


@function_tool
def set_plan_tags(plan_path: str, tags: List[str]) -> str:
    """
    Set the tags for a plan (e.g. workflow assigns a tag for the current plan).
    Replaces any existing tags. Use list_plans_by_tag / get_all_plan_tags for discovery.

    Args:
        plan_path: Full path to the plan JSON file.
        tags: List of tag strings (e.g. ["docking_only", "full_pipeline"]).

    Returns:
        plan_path for convenience.
    """
    return _set_plan_tags_impl(plan_path, tags)


def _set_plan_tags_impl(plan_path: str, tags: List[str]) -> str:
    """Implementation for setting plan tags (non-tool wrapper for internal/test use)."""
    path = Path(plan_path)
    if not path.exists():
        raise FileNotFoundError(f"Plan file not found: {plan_path}")
    with _plan_file_lock(path):
        data = _load_plan_data(path, repair_on_recoverable_error=True)
        data["tags"] = _normalize_tags(tags)
        _write_plan_data(path, data)
    return str(path)


def _clear_run_input_params(parameters: Dict[str, Any]) -> None:
    """In-place: clear run/input keys per stage; leave structural params intact."""
    for stage, keys_to_clear in PARAMS_KEYS_TO_CLEAR.items():
        if stage not in parameters or not isinstance(parameters[stage], dict):
            continue
        for key in keys_to_clear:
            if key in parameters[stage]:
                del parameters[stage][key]


@function_tool
def clone_plan(source_plan_path: str, clear_answers: bool = True) -> str:
    """
    Create a new plan file by copying a source plan and clearing run/input-specific content.
    Use when basing a new plan on a previous one (slight differences in files or options).
    The clone keeps steps, questions, and structural parameters; the workflow can then
    fill run directory and input paths (fill-parameters-only) without calling stage agents.

    Args:
        source_plan_path: Full path to the source plan JSON file.
        clear_answers: If True (default), set answers to {} so the clone goes through
            the normal answer gate. Set False to copy existing answers.

    Returns:
        Full path to the new plan JSON file.
    """
    return _clone_plan_impl(source_plan_path, clear_answers=clear_answers)


def _clone_plan_impl(source_plan_path: str, clear_answers: bool = True) -> str:
    """Implementation for cloning a plan (non-tool wrapper for internal/test use)."""
    path = Path(source_plan_path)
    if not path.exists():
        raise FileNotFoundError(f"Plan file not found: {source_plan_path}")
    with _plan_file_lock(path):
        data = _load_plan_data(path, repair_on_recoverable_error=True)

    # Clear run/request-specific content
    if clear_answers:
        data["answers"] = {}
    data["additional_notes"] = []
    if isinstance(data.get("parameters"), dict):
        _clear_run_input_params(data["parameters"])

    # Write to a unique path (collision-safe)
    new_path = _generate_unique_plan_path(prefix="plan", name=None)
    _write_plan_data(new_path, data)
    return str(new_path)


@function_tool
def list_plans_by_tag(tag: str) -> dict:
    """
    List plans that have the given tag. For executive discovery: find relevant plans to adapt or reuse.

    Args:
        tag: Tag to filter by (e.g. "docking_only", "full_pipeline").

    Returns:
        dict with plan_paths (list of str), count (int), and error (optional).
    """
    return _list_plans_by_tag_impl(tag)


def _list_plans_by_tag_impl(tag: str) -> dict:
    """Implementation for listing plans by tag (non-tool wrapper for internal/test use)."""
    plans_dir = _plans_dir()
    normalized_tag = str(tag).strip()
    plan_paths: List[str] = []
    try:
        for p in sorted(plans_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
            try:
                data = _load_plan_data(p, repair_on_recoverable_error=False)
                if normalized_tag in _normalize_tags(data.get("tags")):
                    plan_paths.append(str(p))
            except (ValueError, IOError):
                continue
        return {"plan_paths": plan_paths, "count": len(plan_paths), "tag": normalized_tag}
    except Exception as e:
        return {"plan_paths": [], "count": 0, "tag": normalized_tag, "error": str(e)}


@function_tool
def get_all_plan_tags() -> dict:
    """
    Get all tags used across plans with counts. For executive: discover tags before list_plans_by_tag.

    Returns:
        dict with tags (dict mapping tag name to count) and total_tags (int).
    """
    return _get_all_plan_tags_impl()


def _get_all_plan_tags_impl() -> dict:
    """Implementation for aggregating plan tags (non-tool wrapper for internal/test use)."""
    plans_dir = _plans_dir()
    tag_counts: Dict[str, int] = {}
    try:
        for p in plans_dir.glob("*.json"):
            try:
                data = _load_plan_data(p, repair_on_recoverable_error=False)
                for t in _normalize_tags(data.get("tags")):
                    tag_counts[t] = tag_counts.get(t, 0) + 1
            except (ValueError, IOError):
                continue
        return {"tags": tag_counts, "total_tags": len(tag_counts)}
    except Exception as e:
        return {"tags": {}, "total_tags": 0, "error": str(e)}
