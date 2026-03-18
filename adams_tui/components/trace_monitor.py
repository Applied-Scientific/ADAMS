"""
Trace File Monitor

Tails the active session's .jsonl trace file and emits TraceEvent / StageChanged
messages so the dashboard and chat can reflect agent activity in real time.
"""

import json
import os
from pathlib import Path

from textual.widget import Widget

from .messages import TraceEvent, StageChanged


# Map trace event types to human-readable stage descriptions
_STAGE_MAP = {
    "session_start": ("idle", "Session started"),
    "user_input": ("input", "User input received"),
    "workflow_start": ("thinking", "Thinking..."),
    "agent_start": ("thinking", "Agent: {agent}"),
    "tool_call_start": ("tool_call", "Tool: {tool}"),
    "tool_call_end": ("tool_done", "Tool done: {tool}"),
    "generation": ("generation", "Generation ({model})"),
    "agent_end": ("complete", "Agent complete"),
    "workflow_end": ("idle", "Workflow complete"),
    "session_end": ("idle", "Session ended"),
}


class TraceMonitor(Widget):
    """
    Invisible widget that polls the active JSONL trace file for new events.
    Posts TraceEvent and StageChanged messages to the app.
    """

    DEFAULT_CSS = """
    TraceMonitor {
        display: none;
    }
    """

    def __init__(self, trace_filepath: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self._filepath = trace_filepath
        self._file_pos = 0  # bytes read so far
        self._timer = None

    def set_trace_file(self, filepath: str) -> None:
        """Set or change the trace file to monitor."""
        self._filepath = filepath
        # Jump to end of existing content so we only get new events
        try:
            self._file_pos = os.path.getsize(filepath)
        except OSError:
            self._file_pos = 0

    def on_mount(self) -> None:
        """Start polling every 500ms."""
        self._timer = self.set_interval(0.5, self._poll)

    def _poll(self) -> None:
        """Check for new lines in the JSONL file."""
        if not self._filepath or not os.path.exists(self._filepath):
            return

        try:
            file_size = os.path.getsize(self._filepath)
            if file_size <= self._file_pos:
                return

            with open(self._filepath, "r", encoding="utf-8") as f:
                f.seek(self._file_pos)
                new_data = f.read()
                self._file_pos = f.tell()

            for line in new_data.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    self._handle_event(event)
                except json.JSONDecodeError:
                    continue
        except Exception:
            pass

    def _handle_event(self, event: dict) -> None:
        """Parse a trace event and post messages."""
        event_type = event.get("event", "unknown")
        timestamp = event.get("timestamp", "")

        # Post the raw trace event
        self.post_message(TraceEvent(
            event_type=event_type,
            timestamp=timestamp,
            details=event,
        ))

        # Determine stage change
        stage_info = _STAGE_MAP.get(event_type)
        if stage_info:
            stage, template = stage_info
            detail = template.format(
                agent=event.get("agent", ""),
                tool=event.get("tool", ""),
                model=event.get("model", ""),
            )
            self.post_message(StageChanged(stage=stage, detail=detail))
