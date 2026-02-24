"""Memory system for persistent agent memory and session tracking."""

from .persistent_memory import (
    add_learned_behavior,
    detect_environment_info,
    get_memory_summary,
    load_persistent_memory,
    save_persistent_memory,
    set_custom_instructions,
    update_user_preference,
)
from .session_memory import (
    add_session_tags,
    continue_session,
    get_all_tags,
    get_session,
    get_session_context_summary,
    get_session_plan_summary,
    get_session_summaries,
    get_session_trace_path,
    get_sessions_by_tag,
    list_sessions,
    register_session,
    remove_session_tags,
    search_sessions,
    set_session_description,
    set_session_tags,
)

__all__ = [
    # Persistent memory
    "load_persistent_memory",
    "save_persistent_memory",
    "detect_environment_info",
    "update_user_preference",
    "add_learned_behavior",
    "set_custom_instructions",
    "get_memory_summary",
    # Session memory
    "register_session",
    "get_session",
    "search_sessions",
    "list_sessions",
    "get_session_trace_path",
    "set_session_description",
    "set_session_tags",
    "continue_session",
    "get_session_plan_summary",
    "get_session_context_summary",
    # Tag-based session organization
    "get_all_tags",
    "get_sessions_by_tag",
    "get_session_summaries",
    "add_session_tags",
    "remove_session_tags",
]
