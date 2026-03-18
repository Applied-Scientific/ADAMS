"""Session memory tracking and retrieval."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..path_config import get_subdirectory
from ..utils.json_io import load_json, save_json


def _get_sessions_file() -> Path:
    """Get path to sessions.json file."""
    memory_dir = get_subdirectory("memory")
    memory_dir.mkdir(parents=True, exist_ok=True)
    return memory_dir / "sessions.json"


def _normalize_session(session: Dict) -> None:
    """In-place: ensure plan_paths is a list[str] and remove invalid entries."""
    raw_paths = session.get("plan_paths")
    if raw_paths is None:
        session["plan_paths"] = []
    elif isinstance(raw_paths, str):
        session["plan_paths"] = [raw_paths] if raw_paths.strip() else []
    elif not isinstance(raw_paths, list):
        session["plan_paths"] = []
    else:
        normalized_paths = []
        for path in raw_paths:
            if isinstance(path, str) and path.strip():
                normalized_paths.append(path)
        session["plan_paths"] = normalized_paths
    raw_tags = session.get("tags")
    if not isinstance(raw_tags, list):
        session["tags"] = []
    else:
        session["tags"] = [str(t).strip() for t in raw_tags if str(t).strip()]


def _load_sessions() -> Dict:
    """Load sessions from JSON file. Normalizes each session (e.g. plan_paths list)."""
    data = load_json(_get_sessions_file(), default={"sessions": []})
    if not isinstance(data, dict):
        data = {"sessions": []}
    sessions = data.get("sessions")
    if not isinstance(sessions, list):
        sessions = []
    normalized_sessions = []
    for session in sessions:
        if isinstance(session, dict):
            _normalize_session(session)
            normalized_sessions.append(session)
    data["sessions"] = normalized_sessions
    return data


def _save_sessions(data: Dict) -> None:
    """Save sessions to JSON file."""
    save_json(_get_sessions_file(), data)


def register_session(
    session_id: str,
    trace_file_path: str,
    description: Optional[str] = None,
    tags: Optional[List[str]] = None,
    plan_paths: Optional[List[str]] = None,
) -> None:
    """Register a new session. Optionally associate plan_paths (a session can have multiple)."""
    data = _load_sessions()
    paths = [p.strip() for p in (plan_paths or []) if isinstance(p, str) and p.strip()]
    for existing in data["sessions"]:
        if existing.get("session_id") == session_id:
            existing["trace_file"] = trace_file_path
            if description is not None:
                existing["description"] = description
            if tags is not None:
                existing["tags"] = [str(t).strip() for t in tags if str(t).strip()]
            if paths:
                current_paths = existing.get("plan_paths", [])
                for p in paths:
                    if p not in current_paths:
                        current_paths.append(p)
                existing["plan_paths"] = current_paths
            existing["timestamp"] = datetime.now().isoformat()
            _save_sessions(data)
            return
    session = {
        "session_id": session_id,
        "trace_file": trace_file_path,
        "description": description or "",
        "tags": [str(t).strip() for t in (tags or []) if str(t).strip()],
        "timestamp": datetime.now().isoformat(),
        "created_at": datetime.now().isoformat(),
        "plan_paths": paths,
    }
    data["sessions"].append(session)
    _save_sessions(data)


def ensure_session(session_id: str, trace_file_path: str) -> None:
    """Ensure a session record exists. Updates timestamp if present; registers if missing.

    Called when continuing a session so that plan linking via add_session_plan_path
    always has a target record, even if sessions.json was cleared or the original
    registration failed.
    """
    data = _load_sessions()
    for session in data["sessions"]:
        if session["session_id"] == session_id:
            session["timestamp"] = datetime.now().isoformat()
            session["trace_file"] = trace_file_path
            _save_sessions(data)
            return
    # Session not found — create a minimal record so plan linking works.
    session = {
        "session_id": session_id,
        "trace_file": trace_file_path,
        "description": "",
        "tags": [],
        "timestamp": datetime.now().isoformat(),
        "created_at": datetime.now().isoformat(),
        "plan_paths": [],
    }
    data["sessions"].append(session)
    _save_sessions(data)


def add_session_plan_path(session_id: str, plan_path: str) -> bool:
    """Add a plan path to a session (sessions can have multiple plan paths). Idempotent."""
    normalized_path = str(plan_path).strip()
    if not normalized_path:
        return False
    data = _load_sessions()
    for session in data["sessions"]:
        if session["session_id"] == session_id:
            paths = session.get("plan_paths") or []
            if normalized_path not in paths:
                paths.append(normalized_path)
                session["plan_paths"] = paths
            _save_sessions(data)
            return True
    return False


def get_session(session_id: str) -> Optional[Dict]:
    """Get session by ID."""
    data = _load_sessions()
    for session in data["sessions"]:
        if session["session_id"] == session_id:
            return session
    return None


def search_sessions(query: str) -> List[Dict]:
    """Search sessions by description/keywords/tags."""
    data = _load_sessions()
    query_lower = query.lower()
    results = []
    for session in data["sessions"]:
        desc = session.get("description", "").lower()
        tags = [t.lower() for t in session.get("tags", [])]
        session_id = session["session_id"].lower()
        # Search in description, tags, or session ID
        if (query_lower in desc or 
            query_lower in session_id or 
            any(query_lower in tag for tag in tags)):
            results.append(session)
    return results


def list_sessions(limit: int = 10) -> List[Dict]:
    """List recent sessions."""
    data = _load_sessions()
    sessions = sorted(
        data["sessions"], key=lambda s: s.get("timestamp", ""), reverse=True
    )
    return sessions[:limit]


def get_session_trace_path(session_id: str) -> Optional[str]:
    """Get trace file path for a session."""
    session = get_session(session_id)
    return session.get("trace_file") if session else None


def set_session_description(session_id: str, description: str) -> bool:
    """Set or update session description."""
    data = _load_sessions()
    for session in data["sessions"]:
        if session["session_id"] == session_id:
            session["description"] = description
            _save_sessions(data)
            return True
    return False


def set_session_tags(session_id: str, tags: List[str]) -> bool:
    """Set or replace session tags (full control; use for agent-defined tags)."""
    data = _load_sessions()
    for session in data["sessions"]:
        if session["session_id"] == session_id:
            session["tags"] = list(tags) if tags else []
            _save_sessions(data)
            return True
    return False


def add_session_tags(session_id: str, tags: List[str]) -> bool:
    """Add tags to a session (doesn't duplicate existing tags)."""
    data = _load_sessions()
    for session in data["sessions"]:
        if session["session_id"] == session_id:
            existing_tags = set(session.get("tags", []))
            new_tags = set(tags)
            session["tags"] = list(existing_tags | new_tags)
            _save_sessions(data)
            return True
    return False


def remove_session_tags(session_id: str, tags: List[str]) -> bool:
    """Remove tags from a session."""
    data = _load_sessions()
    for session in data["sessions"]:
        if session["session_id"] == session_id:
            existing_tags = session.get("tags", [])
            session["tags"] = [t for t in existing_tags if t not in tags]
            _save_sessions(data)
            return True
    return False


def get_all_tags() -> Dict[str, int]:
    """Get all tags with session counts (fast overview)."""
    data = _load_sessions()
    tag_counts: Dict[str, int] = {}
    for session in data["sessions"]:
        for tag in session.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return tag_counts


def get_sessions_by_tag(tag: str, limit: Optional[int] = None) -> List[Dict]:
    """Get lightweight summaries of sessions with a specific tag (same shape as get_session_summaries)."""
    applied_limit = 50 if limit is None else max(0, int(limit))
    return get_session_summaries(tag=tag, limit=applied_limit)


def get_session_summaries(session_ids: Optional[List[str]] = None, tag: Optional[str] = None, limit: Optional[int] = None) -> List[Dict]:
    """
    Get lightweight session summaries (fast, cheap access).
    
    Returns only essential fields: session_id, description, tags, timestamp.
    Use this for overviews before fetching full session details.
    """
    data = _load_sessions()
    sessions = data["sessions"]
    
    # Filter by tag if provided
    if tag:
        sessions = [s for s in sessions if tag in s.get("tags", [])]
    
    # Filter by session IDs if provided
    if session_ids:
        session_id_set = set(session_ids)
        sessions = [s for s in sessions if s["session_id"] in session_id_set]
    
    # Sort by timestamp, most recent first
    sessions.sort(key=lambda s: s.get("timestamp", ""), reverse=True)
    
    # Apply limit if provided
    if limit is not None:
        sessions = sessions[:limit]
    
    # Return summaries (lightweight)
    summaries = []
    for session in sessions:
        summaries.append({
            "session_id": session["session_id"],
            "description": session.get("description", ""),
            "tags": session.get("tags", []),
            "timestamp": session.get("timestamp", ""),
        })
    
    return summaries


def continue_session(session_id: str) -> Dict:
    """Restore conversation state from trace file."""
    session = get_session(session_id)
    if not session:
        return {
            "error": f"Session {session_id} not found",
            "conversation_history": None,
            "trace_file": None,
            "description": None,
        }

    trace_path = _resolve_trace_path(session)

    conversation_history = []
    if trace_path.exists():
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        event_type = event.get("event")
                        if event_type == "user_input":
                            conversation_history.append(
                                {"role": "user", "content": event.get("input", "")}
                            )
                        elif event_type == "agent_output":
                            conversation_history.append(
                                {"role": "assistant", "content": event.get("output", "")}
                            )
                    except json.JSONDecodeError:
                        continue
        except IOError:
            pass

    return {
        "session_id": session_id,
        "trace_file": str(trace_path),
        "description": session.get("description", ""),
        "conversation_history": conversation_history,
        "error": None,
    }


def _resolve_trace_path(session: Dict) -> Path:
    """Resolve trace file path (absolute) from session record."""
    trace_file = session["trace_file"]
    trace_path = Path(trace_file)
    if not trace_path.is_absolute():
        trace_path = get_subdirectory("traces") / trace_path.name
    return trace_path


def get_session_plan_summary(session_id: str) -> Dict[str, Any]:
    """
    Get session metadata only (no trace parsing). Returns session_id, description,
    tags, timestamp, and plan_paths (list; a session can have multiple).
    Use meta_analysis_agent for trace/log analysis and current-run error solving.
    """
    session = get_session(session_id)
    if not session:
        return {
            "error": f"Session {session_id} not found",
            "session_id": session_id,
            "description": "",
            "tags": [],
            "timestamp": "",
            "plan_paths": [],
        }
    return {
        "error": None,
        "session_id": session_id,
        "description": session.get("description", ""),
        "tags": session.get("tags", []),
        "timestamp": session.get("timestamp", ""),
        "plan_paths": session.get("plan_paths", []),
    }


def get_session_context_summary(limit_sessions: int = 10) -> Dict[str, Any]:
    """
    Compact summary for injection into the agent prompt: all tags with counts
    and recent session summaries (session_id, description, tags, timestamp).
    No trace reads; uses sessions.json only. Injected into the executive prompt.
    """
    tags = get_all_tags()
    summaries = get_session_summaries(limit=limit_sessions)
    return {
        "tags": tags,
        "recent_sessions": summaries,
    }


def format_session_context_for_prompt(ctx: Dict[str, Any]) -> str:
    """Format session context (from get_session_context_summary) for agent prompt."""
    tags = ctx.get("tags") or {}
    recent = ctx.get("recent_sessions") or []
    reminder = (
        "Before submitting a plan to oversight: use the tags and recent sessions below to identify "
        "relevant plan tags (e.g. docking_only, full_pipeline). Then call get_all_plan_tags() and "
        "list_plans_by_tag(tag) to find plans to adapt or reuse."
    )
    tag_line = (
        "Session tags in use: " + ", ".join(f"{t}({c})" for t, c in sorted(tags.items()))
        if tags
        else "No session tags yet."
    )
    parts = []
    if recent:
        parts = [
            f"- {s.get('session_id', '')}: {s.get('description', '') or '(no description)'} [{', '.join(s.get('tags', []))}]"
            for s in recent
        ]
    sessions_line = "Recent sessions:\n" + "\n".join(parts) if parts else ""
    return "\n".join(filter(None, [reminder, tag_line, sessions_line]))
