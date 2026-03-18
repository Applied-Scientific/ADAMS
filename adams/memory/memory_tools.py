"""Agent-accessible tools for memory management."""

from agents import function_tool

from . import persistent_memory as _pm
from . import session_memory as _sm


@function_tool
def search_previous_sessions(query: str) -> dict:
    """
    Search previous agent sessions by keywords or session ID (SLOWER, use tags first).
    
    Use this when tag-based discovery doesn't work or you need text search.
    Prefer get_all_session_tags() + get_sessions_by_tag() for faster hierarchical access.
    
    Args:
        query: Search query (keywords, description text, or session ID)

    Returns:
        dict: Contains 'sessions' list with matching session info
    """
    results = _sm.search_sessions(query)
    return {"sessions": results, "count": len(results)}


@function_tool
def get_session_plan_summary(session_id: str) -> dict:
    """
    Get session metadata only (no trace parsing). Returns session_id, description,
    tags, timestamp, and plan_paths (list; a session can have multiple).
    When plan_paths is present, use read_plan_document(plan_path) for each or the most relevant.
    For trace/log analysis or current-run error solving, use meta_analysis_agent.

    Args:
        session_id: Session ID (format: YYYYMMDD_HHMMSS)

    Returns:
        dict: session_id, description, tags, timestamp, plan_paths (list), error (if any)
    """
    return _sm.get_session_plan_summary(session_id)


@function_tool
def get_session_info(session_id: str) -> dict:
    """
    Get detailed information about a specific session (SLOWER, use after summaries).
    
    Use this AFTER exploring via get_all_session_tags() or get_sessions_by_tag()
    when you need full details for a specific session. This loads complete session data.
    Use this to get the trace_file path when you want to pass a session to meta_analysis_agent.
    
    Args:
        session_id: Session ID (format: YYYYMMDD_HHMMSS)

    Returns:
        dict: Session details including trace file path, description, tags, timestamp
    """
    session = _sm.get_session(session_id)
    if not session:
        return {"error": f"Session {session_id} not found", "session": None}
    return {"session": session, "error": None}


@function_tool
def set_session_description_tool(session_id: str, description: str) -> dict:
    """
    Set or update the description for a session.

    Call this at the END of each session to record a concise summary
    of what happened, so it can be found later via search.

    Args:
        session_id: Session ID (format: YYYYMMDD_HHMMSS)
        description: Brief description of the session (keep concise)

    Returns:
        dict: Success status and updated session info
    """
    success = _sm.set_session_description(session_id, description)
    if success:
        session = _sm.get_session(session_id)
        return {"success": True, "session": session}
    return {"success": False, "error": f"Session {session_id} not found"}


@function_tool
def set_session_tags_tool(session_id: str, tags: list[str]) -> dict:
    """
    Set or replace the full tag list for a session (same control as description).

    Use whenever it helps discovery: when a workflow completes, when you
    categorize a session, or when revisiting past sessions. Workflow type
    (e.g. "docking", "md_analysis", "preprocessing"), status ("completed",
    "error", "incomplete"), and topic tags. Replaces existing tags; use
    tag_session to add tags incrementally.

    Args:
        session_id: Session ID (format: YYYYMMDD_HHMMSS)
        tags: Full list of tag strings (e.g. ["docking", "completed"])

    Returns:
        dict: Success status and updated session info
    """
    success = _sm.set_session_tags(session_id, tags)
    if success:
        session = _sm.get_session(session_id)
        return {"success": True, "session": session, "message": f"Set tags: {tags}"}
    return {"success": False, "error": f"Session {session_id} not found"}


@function_tool
def list_recent_sessions(limit: int = 10) -> dict:
    """
    List the most recent agent sessions.

    Args:
        limit: Maximum number of sessions to return (default: 10)

    Returns:
        dict: List of recent sessions with IDs, descriptions, and timestamps
    """
    sessions = _sm.list_sessions(limit)
    return {"sessions": sessions, "count": len(sessions)}


@function_tool
def get_all_session_tags() -> dict:
    """
    Get all tags used across sessions with session counts (FAST, CHEAP).
    
    Use this FIRST to understand how sessions are organized before diving deeper.
    This gives you a high-level overview without loading session details.
    
    Returns:
        dict: Dictionary mapping tag names to session counts
            Example: {"docking": 5, "md_analysis": 3, "error_recovery": 2}
    """
    tags = _sm.get_all_tags()
    return {"tags": tags, "total_tags": len(tags)}


@function_tool
def get_sessions_by_tag(tag: str, limit: int = 20) -> dict:
    """
    Get lightweight summaries of sessions with a specific tag (FAST, CHEAP).
    
    Use this AFTER get_all_session_tags() to explore sessions in a category.
    Returns only summaries (session_id, description, tags, timestamp) - not full details.
    Use get_session_info() if you need full details for a specific session.
    
    Args:
        tag: Tag to filter by (e.g., "docking", "md_analysis", "error_recovery")
        limit: Maximum number of sessions to return (default: 20)
    
    Returns:
        dict: Contains 'sessions' list with lightweight summaries
    """
    summaries = _sm.get_sessions_by_tag(tag, limit=limit)
    return {"tag": tag, "sessions": summaries, "count": len(summaries)}


@function_tool
def get_session_summaries(limit: int = 20, tag: str = None) -> dict:
    """
    Get lightweight summaries of sessions (FAST, CHEAP).
    
    Use this for quick overviews before fetching full session details.
    Returns only essential fields: session_id, description, tags, timestamp.
    
    Args:
        limit: Maximum number of sessions to return (default: 20)
        tag: Optional tag to filter by
    
    Returns:
        dict: Contains 'sessions' list with lightweight summaries
    """
    summaries = _sm.get_session_summaries(tag=tag, limit=limit)
    return {"sessions": summaries, "count": len(summaries)}


@function_tool
def tag_session(session_id: str, tags: list[str]) -> dict:
    """
    Add tags to a session for organization and discovery.
    
    Use whenever it helps: when a workflow completes, when categorizing
    a session, or when revisiting past sessions. Relevant categories:
    - Workflow type: "docking", "md_analysis", "preprocessing"
    - Status: "completed", "error", "incomplete"
    - Topic: "protein_ligand", "mutation_analysis", "parameter_optimization"
    - User context: "resume_request", "debugging", "exploration"
    
    Tags help organize sessions for fast discovery. Use set_session_tags_tool
    to replace the full tag list instead of adding.
    
    Args:
        session_id: Session ID (format: YYYYMMDD_HHMMSS)
        tags: List of tag strings (e.g., ["docking", "completed", "protein_ligand"])
    
    Returns:
        dict: Success status and updated session info
    """
    success = _sm.add_session_tags(session_id, tags)
    if success:
        session = _sm.get_session(session_id)
        return {"success": True, "session": session, "message": f"Added tags: {tags}"}
    return {"success": False, "error": f"Session {session_id} not found"}


@function_tool
def get_persistent_memory_tool() -> dict:
    """
    Get all persistent memory (hardware info, user preferences, custom instructions).

    Use this to check what hardware is available, user preferences,
    and any custom instructions that were previously stored.

    Returns:
        dict: Complete persistent memory structure
    """
    return _pm.load_persistent_memory()


@function_tool
def update_user_preference_tool(key: str, value: str) -> dict:
    """
    Update a user preference in persistent memory.

    Use this proactively when you learn user preferences (e.g., GPU usage,
    preferred working directory) so they persist across sessions.

    Args:
        key: Preference key (e.g., 'preferred_gpu_usage', 'preferred_working_directory')
        value: Preference value (will be converted to appropriate type)

    Returns:
        dict: Success status
    """
    try:
        converted = value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            converted = value.lower() == "true"
        _pm.update_user_preference(key, converted)
        return {"success": True, "message": f"Updated {key} = {converted}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@function_tool
def add_learned_behavior_tool(behavior: str) -> dict:
    """
    Add a learned behavior to persistent memory.

    Use this proactively when you observe patterns in user behavior.
    Keep descriptions extremely concise (max 50 words, prefer 10-20).

    Args:
        behavior: Concise description of learned behavior (max 50 words)

    Returns:
        dict: Success status
    """
    try:
        _pm.add_learned_behavior(behavior)
        return {"success": True, "message": "Learned behavior added"}
    except ValueError as e:
        return {"success": False, "error": str(e)}


@function_tool
def set_custom_memory(notes: str) -> dict:
    """
    Set custom instructions in persistent memory (injected into system prompt).

    These custom instructions are included in the agent's system prompt and persist across sessions.
    Users can set these via CLI: `adams memory set-instructions "your instructions"`
    Keep instructions concise (max 100 words) to avoid context overload.

    Args:
        notes: Custom instructions (max 100 words)

    Returns:
        dict: Success status
    """
    try:
        _pm.set_custom_instructions(notes)
        return {"success": True, "message": "Custom instructions updated"}
    except ValueError as e:
        return {"success": False, "error": str(e)}


# Persistent memory only (for executive: no session tools)
PERSISTENT_MEMORY_TOOLS = [
    get_persistent_memory_tool,
    update_user_preference_tool,
    add_learned_behavior_tool,
    set_custom_memory,
]

# Read-only session memory tools for diagnosis and retrieval.
SESSION_MEMORY_READ_TOOLS = [
    get_all_session_tags,
    get_sessions_by_tag,
    get_session_summaries,
    get_session_plan_summary,
    get_session_info,
    search_previous_sessions,
    list_recent_sessions,
]

# Full session memory tools, including metadata writes.
SESSION_MEMORY_TOOLS = SESSION_MEMORY_READ_TOOLS + [
    tag_session,
    set_session_description_tool,
    set_session_tags_tool,
]

# All memory tools (for backward compatibility where both are needed)
MEMORY_TOOLS = PERSISTENT_MEMORY_TOOLS + SESSION_MEMORY_TOOLS
