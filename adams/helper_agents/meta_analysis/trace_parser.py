"""
Trace Parser - Parses trace files and extracts structured information.

This module provides functions to parse JSONL trace files and extract only
the essential information needed for resume/analysis, avoiding context window
overload from loading entire trace files.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...path_config import get_subdirectory


def parse_trace_file_impl(trace_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Parse a trace file and extract structured information.

    This function reads a JSONL trace file and extracts only the essential
    information needed for resume/analysis, returning a structured dictionary
    instead of the raw JSONL content. This dramatically reduces context window
    usage (from ~552KB to ~1-2KB).

    Args:
        trace_file: Path to specific trace file. If None, reads most recent.

    Returns:
        dict with structured information:
        - session_id: Session identifier
        - status: "completed", "error", "incomplete", or "unknown"
        - entry_point: Inferred entry point used
        - output_folder: Output folder path (or None)
        - log_file: Log file path (or None)
        - completed_steps: List of completed steps (preprocessing, docking)
        - steps_with_errors: List of steps that had errors
        - file_paths: Dict with receptor, ligands_csv, docking_centers, docking_results
        - last_error: Last error message (or None)
        - original_workflow_prompt: User input that started the workflow (or None)
        - trace_file: Path to the trace file parsed
        - error: Error message if parsing failed (or None)
    """
    traces_dir = get_subdirectory("traces")

    # Find trace file
    if trace_file:
        trace_path = Path(trace_file)
    else:
        if not traces_dir.exists():
            return {
                "error": f"Traces directory not found: {traces_dir}",
                "trace_file": None,
            }
        trace_files = sorted(traces_dir.glob("trace_*.jsonl"), reverse=True)
        if not trace_files:
            return {"error": "No trace files found", "trace_file": None}
        trace_path = trace_files[0]

    if not trace_path.exists():
        return {
            "error": f"Trace file not found: {trace_path}",
            "trace_file": str(trace_path),
        }

    # Initialize result structure
    result = {
        "session_id": None,
        "status": "unknown",
        "entry_point": None,
        "output_folder": None,
        "log_file": None,
        "completed_steps": [],
        "steps_with_errors": [],
        "file_paths": {
            "receptor": None,
            "ligands_csv": None,
            "docking_centers": None,
            "docking_results": None,
        },
        "last_error": None,
        "original_workflow_prompt": None,
        "trace_file": str(trace_path),
        "error": None,
    }

    try:
        # Track agents and their completion status
        agents_seen = set()
        agents_completed = set()
        agents_with_errors = set()

        # Track workflow state
        workflow_started = False
        workflow_ended = False
        session_ended = False

        # Read and parse JSONL
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("event")

                # Extract session_id
                if event_type == "session_start":
                    result["session_id"] = event.get("session_id")

                # Extract user input (workflow prompt)
                if event_type == "user_input":
                    result["original_workflow_prompt"] = event.get("input")

                # Track workflow boundaries
                if event_type == "workflow_start":
                    workflow_started = True
                if event_type == "workflow_end":
                    workflow_ended = True
                if event_type == "session_end":
                    session_ended = True

                # Track agent execution
                if event_type == "agent_start":
                    agent_name = event.get("agent")
                    if agent_name:
                        agents_seen.add(agent_name)

                if event_type == "agent_end":
                    agent_name = event.get("agent")
                    if agent_name:
                        agents_completed.add(agent_name)
                        if "error" in event:
                            agents_with_errors.add(agent_name)

                # Extract file paths and output folder from tool calls
                if event_type == "tool_call_end":
                    tool_name = event.get("tool")
                    tool_input = event.get("input", {})
                    tool_output = event.get("output")
                    tool_error = event.get("error")

                    # Extract output folder (prioritize out_folder, then outpath)
                    if not result["output_folder"]:
                        # Check various output folder parameters (in priority order)
                        for key in [
                            "out_folder",
                            "outpath",
                            "output_folder",
                        ]:
                            if key in tool_input:
                                value = tool_input[key]
                                if value and isinstance(value, str):
                                    # Make sure it's a directory path, not a file path
                                    # If it ends with .pdb, .csv, etc., it's a file - skip
                                    if not any(
                                        value.endswith(ext)
                                        for ext in [
                                            ".pdb",
                                            ".pdbqt",
                                            ".csv",
                                            ".gro",
                                            ".top",
                                            ".log",
                                        ]
                                    ):
                                        result["output_folder"] = value
                                        break

                    # Extract output folder from log file if available
                    if result["log_file"] and not result["output_folder"]:
                        # Extract run directory from log file name
                        # Format: agent_data/logs/adams_pipeline_run_YYYYMMDD_HHMMSS_*.log
                        match = re.search(r"run_([^_]+(?:_[^_]+)*)", result["log_file"])
                        if match:
                            run_name = match.group(1)
                            # Try to find the actual output folder
                            # Look for outputs/run_* pattern in tool inputs
                            for key, value in tool_input.items():
                                if (
                                    isinstance(value, str)
                                    and f"run_{run_name}" in value
                                ):
                                    # Extract directory path
                                    if "outputs/run_" in value:
                                        # Get the directory part
                                        parts = value.split("outputs/run_")
                                        if len(parts) > 1:
                                            result[
                                                "output_folder"
                                            ] = f"agent_data/outputs/run_{parts[1].split('/')[0]}"
                                            break

                    # Extract log file
                    if tool_name == "setup_pipeline_logger" and tool_output:
                        if isinstance(tool_output, str):
                            result["log_file"] = tool_output
                        elif (
                            isinstance(tool_output, dict) and "log_file" in tool_output
                        ):
                            result["log_file"] = tool_output["log_file"]

                    # Extract file paths
                    if "receptor" in tool_input:
                        result["file_paths"]["receptor"] = tool_input["receptor"]
                    if "input_pdb" in tool_input:
                        result["file_paths"]["receptor"] = tool_input["input_pdb"]
                    if "protein_file" in tool_input:
                        result["file_paths"]["receptor"] = tool_input["protein_file"]

                    if "input_data" in tool_input:
                        result["file_paths"]["ligands_csv"] = tool_input["input_data"]
                    if "input_file" in tool_input:
                        result["file_paths"]["ligands_csv"] = tool_input["input_file"]
                    if "ligand_input" in tool_input:
                        result["file_paths"]["ligands_csv"] = tool_input["ligand_input"]
                    # Backward compatibility: also check for old smiles_file parameter
                    elif "smiles_file" in tool_input:
                        result["file_paths"]["ligands_csv"] = tool_input["smiles_file"]

                    if "docking_centers_file" in tool_input:
                        result["file_paths"]["docking_centers"] = tool_input[
                            "docking_centers_file"
                        ]
                    if "docking_centers" in tool_input:
                        # Could be a list, extract if possible
                        pass

                    if "vina_report" in tool_input:
                        result["file_paths"]["docking_results"] = tool_input[
                            "vina_report"
                        ]

                    # Track errors
                    if tool_error:
                        result["last_error"] = str(tool_error)[
                            :200
                        ]  # Truncate long errors
                        # Map tool to step
                        if (
                            "preprocessing" in tool_name.lower()
                            or "clean_pdb" in tool_name
                            or "ligand_preprocessing" in tool_name
                        ):
                            if "preprocessing" not in result["steps_with_errors"]:
                                result["steps_with_errors"].append("preprocessing")
                        elif "docking" in tool_name.lower():
                            if "docking" not in result["steps_with_errors"]:
                                result["steps_with_errors"].append("docking")
        # Map agents to pipeline steps
        agent_to_step = {
            "Data Preprocessing Agent": "preprocessing",
            "Molecular Docking Agent": "docking",
        }

        for agent_name, step in agent_to_step.items():
            if agent_name in agents_completed:
                if agent_name not in agents_with_errors:
                    if step not in result["completed_steps"]:
                        result["completed_steps"].append(step)

        # Determine status
        if agents_with_errors:
            result["status"] = "error"
        elif workflow_ended and session_ended and not agents_with_errors:
            result["status"] = "completed"
        elif workflow_started and not workflow_ended:
            result["status"] = "incomplete"
        else:
            result["status"] = "unknown"

        # Infer entry point from workflow prompt
        if result["original_workflow_prompt"]:
            prompt_lower = result["original_workflow_prompt"].lower()
            if "clean" in prompt_lower and "receptor" in prompt_lower:
                result["entry_point"] = "preprocessing"
            elif "discover" in prompt_lower or "search" in prompt_lower:
                result["entry_point"] = "search_docking"
            elif "production docking" in prompt_lower and "search" not in prompt_lower:
                result["entry_point"] = "production_docking"
            else:
                result["entry_point"] = "full_pipeline"

        # Clean up None values in file_paths
        result["file_paths"] = {
            k: v for k, v in result["file_paths"].items() if v is not None
        }

    except Exception as e:
        result["error"] = f"Failed to parse trace file: {e}"

    return result


def _is_oversight_approved(event: Dict[str, Any]) -> bool:
    """Return True if this tool_call_end event indicates an approved oversight review.
    Prefers dict output with an explicit 'approved' boolean; falls back to string heuristics.
    """
    out = event.get("output")
    if out is None:
        return False
    if isinstance(out, dict):
        return out.get("approved") is True
    if isinstance(out, str):
        return "NOT approved" not in out and "Approved" in out
    return False


def _is_submit_review_approved(event: Dict[str, Any]) -> bool:
    """Return True if this tool_call_end is submit_review (Oversight Agent) with approved=True."""
    if event.get("tool") != "submit_review" or event.get("caller") != "Oversight Agent":
        return False
    out = event.get("output")
    return isinstance(out, dict) and out.get("approved") is True


def _extract_approved_plan_input(event: Dict[str, Any]) -> str:
    """Extract the submission text (plan) from an oversight_agent tool_call input (start or end)."""
    inp = event.get("input")
    if inp is None:
        return ""
    if isinstance(inp, dict) and "input" in inp:
        return inp["input"] if isinstance(inp["input"], str) else str(inp["input"])
    if isinstance(inp, str):
        return inp
    return str(inp)


def get_trace_plan_pairs(trace_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract user-request → approved-plan pairs from a trace file (single pass).

    Uses two signals for approval (so plans are stored reliably):
    1. submit_review tool_call_end (caller Oversight Agent) with output["approved"] is True
       — canonical source; plan is taken from the preceding oversight_agent call.
    2. oversight_agent tool_call_end with output indicating approval (dict with "approved"
       or string containing "Approved" and not "NOT approved") — fallback for older traces.

    Buffers user_input; on approval, emits (user_request_block, approved_plan) and clears.
    Supports multi-turn user requests per plan.

    Args:
        trace_path: Path to the trace JSONL file. If None, returns empty plan_pairs.

    Returns:
        dict with:
        - plan_pairs: list of {"user_request_block": str, "approved_plan": str}
        - error: str or None
    """
    result: Dict[str, Any] = {"plan_pairs": [], "error": None}
    if not trace_path:
        return result

    path = Path(trace_path)
    if not path.exists():
        result["error"] = f"Trace file not found: {path}"
        return result

    buffer: List[str] = []
    # submit_review end appears before oversight_agent end; emit pair when we see oversight_agent end with plan
    pending_emit_from_submit_review: bool = False
    pending_exact_plan: Optional[str] = None  # from submit_review output; used as approved_plan when present
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("event")
                if event_type == "user_input":
                    buffer.append(event.get("input", "") or "")
                elif event_type == "tool_call_end" and event.get("tool") == "submit_review":
                    if _is_submit_review_approved(event):
                        pending_emit_from_submit_review = True
                        out = event.get("output") if isinstance(event.get("output"), dict) else {}
                        pending_exact_plan = out.get("exact_plan") or None
                elif event_type == "tool_call_end" and event.get("tool") == "oversight_agent":
                    approved_plan = pending_exact_plan if pending_exact_plan else _extract_approved_plan_input(event)
                    if pending_emit_from_submit_review or _is_oversight_approved(event):
                        user_request_block = "\n\n".join(buffer).strip()
                        result["plan_pairs"].append({
                            "user_request_block": user_request_block,
                            "approved_plan": approved_plan,
                        })
                    buffer = []
                    pending_emit_from_submit_review = False
                    pending_exact_plan = None
    except Exception as e:
        result["error"] = f"Failed to parse trace: {e}"

    return result
