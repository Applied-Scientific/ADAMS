"""Session and conversation-history utilities for the CLI."""

import asyncio
import json
import logging
from pathlib import Path

from ..path_config import get_agent_data_path


def read_history_from_trace(trace_path: Path) -> list:
    """Read conversation history from a trace JSONL file. Returns list of {role, content} dicts."""
    out = []
    if not trace_path.exists():
        return out
    try:
        with open(trace_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if event.get("event") == "user_input":
                        out.append({"role": "user", "content": event.get("input", "")})
                    elif event.get("event") == "agent_output":
                        out.append({"role": "assistant", "content": event.get("output", "")})
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return out


def create_sdk_session(session_id: str, is_continue: bool):
    """
    Create an SDK SQLiteSession with persistent DB under agent_data/session_db/.
    If is_continue, restore conversation history from the session's trace file.
    Returns the SQLiteSession instance.
    """
    from agents import SQLiteSession

    from ..memory.session_memory import continue_session as load_session_history

    session_db_dir = get_agent_data_path() / "session_db"
    session_db_dir.mkdir(parents=True, exist_ok=True)
    session_db_path = str(session_db_dir / "conversation_history.db")
    session = SQLiteSession(session_id, session_db_path)

    if is_continue:
        data = load_session_history(session_id)
        history = data.get("conversation_history") or read_history_from_trace(
            get_agent_data_path() / "traces" / f"trace_{session_id}.jsonl"
        )
        if history:
            try:
                asyncio.run(session.add_items(history))
            except Exception as e:
                logging.warning("Could not restore session history from trace: %s", e)

    return session
