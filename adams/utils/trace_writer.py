"""
Trace Writer for OpenAI Agents SDK

This module provides a custom TracingProcessor that writes agent interactions
to a JSONL file in real-time with a simplified, readable format.

Captures only the essential information:
- Agent calls (which agent ran, when it started/finished)
- Tool/function calls (inputs and outputs)
- Errors

This is completely separate from the pipeline logger (logger_utils.py).
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from agents.tracing import (
    AgentSpanData,
    CustomSpanData,
    FunctionSpanData,
    GenerationSpanData,
    GuardrailSpanData,
    HandoffSpanData,
    Span,
    SpanData,
    Trace,
    TracingProcessor,
)


class JsonTraceProcessor(TracingProcessor):
    """
    Simplified trace processor that writes agent interactions to JSONL.

    Only captures meaningful events:
    - Agent runs (start/end with timing)
    - Tool calls (with inputs and outputs)
    - Errors

    Skips internal SDK spans like ResponseSpanData for cleaner output.
    """

    def __init__(self, output_dir: str = "agent_data/traces"):
        """
        Initialize the JSON trace processor.

        Args:
            output_dir: Directory to write trace files.
        """
        self.output_dir = output_dir
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(output_dir, f"trace_{self.session_id}.jsonl")
        self._ensure_dir()

        # Track active spans for computing durations
        self._span_start_times: Dict[str, datetime] = {}
        self._span_agents: Dict[str, str] = {}  # Map span_id to agent name for context

        self._write_session_header()

    def _ensure_dir(self) -> None:
        """Create the output directory if it doesn't exist."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _write_session_header(self) -> None:
        """Write initial session metadata."""
        header = {
            "event": "session_start",
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
        }
        self._write_event(header)

    def write_user_input(self, user_input: str) -> None:
        """Write user input to the trace file."""
        event = {
            "event": "user_input",
            "timestamp": datetime.now().isoformat(),
            "input": user_input,
        }
        self._write_event(event)

    def write_agent_output(self, agent_output: str) -> None:
        """Write agent output to the trace file."""
        event = {
            "event": "agent_output",
            "timestamp": datetime.now().isoformat(),
            "output": agent_output,
        }
        self._write_event(event)

    def _write_event(self, event: Dict[str, Any]) -> None:
        """Append an event to the JSONL file immediately."""
        try:
            # Clean up None values for readability
            clean_event = {k: v for k, v in event.items() if v is not None}
            with open(self.filepath, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        clean_event, default=self._json_serializer, ensure_ascii=False
                    )
                    + "\n"
                )
                f.flush()
        except Exception as e:
            print(f"[JsonTraceProcessor] Warning: Failed to write event: {e}")

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-standard types."""
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "__dict__"):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, bytes):
            return "<binary>"
        return str(obj)

    def _is_meaningful_span(self, span: Span) -> bool:
        """Check if this span type should be logged."""
        span_data = span.span_data
        if span_data is None:
            return False

        # Only log agent spans and function spans
        # Skip ResponseSpanData, GenerationSpanData (too verbose), etc.
        return isinstance(span_data, (AgentSpanData, FunctionSpanData, HandoffSpanData))

    def _get_span_type(self, span: Span) -> Optional[str]:
        """Get a human-readable span type."""
        span_data = span.span_data
        if isinstance(span_data, AgentSpanData):
            return "agent"
        elif isinstance(span_data, FunctionSpanData):
            return "tool_call"
        elif isinstance(span_data, HandoffSpanData):
            return "handoff"
        return None

    def _get_agent_name(self, span: Span) -> Optional[str]:
        """Extract agent name from span data."""
        if isinstance(span.span_data, AgentSpanData):
            return span.span_data.name
        return None

    def _get_function_info(self, span: Span) -> Dict[str, Any]:
        """Extract function/tool information from span data."""
        result = {}
        if isinstance(span.span_data, FunctionSpanData):
            if hasattr(span.span_data, "name") and span.span_data.name:
                result["tool"] = span.span_data.name
            if hasattr(span.span_data, "input") and span.span_data.input:
                # Try to parse JSON input for readability
                try:
                    if isinstance(span.span_data.input, str):
                        result["input"] = json.loads(span.span_data.input)
                    else:
                        result["input"] = span.span_data.input
                except:
                    result["input"] = span.span_data.input
            if hasattr(span.span_data, "output") and span.span_data.output:
                result["output"] = span.span_data.output
        return result

    def _find_parent_agent(self, span: Span) -> Optional[str]:
        """Find the parent agent name for context."""
        parent_id = span.parent_id
        while parent_id:
            if parent_id in self._span_agents:
                return self._span_agents[parent_id]
            break
        return None

    def on_trace_start(self, trace: Trace) -> None:
        """Called when a new trace begins."""
        event = {
            "event": "workflow_start",
            "timestamp": datetime.now().isoformat(),
            "trace_id": trace.trace_id,
        }
        self._write_event(event)

    def on_trace_end(self, trace: Trace) -> None:
        """Called when a trace completes."""
        event = {
            "event": "workflow_end",
            "timestamp": datetime.now().isoformat(),
            "trace_id": trace.trace_id,
        }
        self._write_event(event)

    def on_span_start(self, span: Span) -> None:
        """Called when a span begins - only log meaningful spans."""
        if not self._is_meaningful_span(span):
            return

        # Track start time for duration calculation
        self._span_start_times[span.span_id] = datetime.now()

        span_type = self._get_span_type(span)

        if span_type == "agent":
            agent_name = self._get_agent_name(span)
            self._span_agents[span.span_id] = agent_name

            event = {
                "event": "agent_start",
                "timestamp": datetime.now().isoformat(),
                "agent": agent_name,
                "span_id": span.span_id,
            }
            self._write_event(event)

        elif span_type == "tool_call":
            func_info = self._get_function_info(span)
            parent_agent = self._find_parent_agent(span)

            event = {
                "event": "tool_call_start",
                "timestamp": datetime.now().isoformat(),
                "tool": func_info.get("tool"),
                "caller": parent_agent,
                "span_id": span.span_id,
            }
            # Only add input if present
            if "input" in func_info:
                event["input"] = func_info["input"]

            self._write_event(event)

    def on_span_end(self, span: Span) -> None:
        """Called when a span completes - only log meaningful spans."""
        if not self._is_meaningful_span(span):
            return

        span_type = self._get_span_type(span)

        # Calculate duration
        duration_sec = None
        if span.span_id in self._span_start_times:
            start_time = self._span_start_times.pop(span.span_id)
            duration_sec = round((datetime.now() - start_time).total_seconds(), 2)

        # Check for errors
        error_info = None
        if hasattr(span, "error") and span.error:
            if hasattr(span.error, "message"):
                error_info = span.error.message
            elif isinstance(span.error, dict) and "message" in span.error:
                error_info = span.error["message"]
            else:
                error_info = str(span.error)

        if span_type == "agent":
            agent_name = self._get_agent_name(span)

            event = {
                "event": "agent_end",
                "timestamp": datetime.now().isoformat(),
                "agent": agent_name,
                "duration_sec": duration_sec,
                "span_id": span.span_id,
            }
            if error_info:
                event["error"] = error_info

            self._write_event(event)

            # Clean up agent tracking
            if span.span_id in self._span_agents:
                del self._span_agents[span.span_id]

        elif span_type == "tool_call":
            func_info = self._get_function_info(span)
            parent_agent = self._find_parent_agent(span)

            event = {
                "event": "tool_call_end",
                "timestamp": datetime.now().isoformat(),
                "tool": func_info.get("tool"),
                "caller": parent_agent,
                "duration_sec": duration_sec,
                "span_id": span.span_id,
            }

            # Add input parameters if present (important for tracking what was called)
            if "input" in func_info:
                event["input"] = func_info["input"]

            # Add output if present
            if "output" in func_info:
                event["output"] = func_info["output"]

            if error_info:
                event["error"] = error_info

            self._write_event(event)

    def shutdown(self) -> None:
        """Called when the application stops."""
        event = {
            "event": "session_end",
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
        }
        self._write_event(event)

    def force_flush(self) -> None:
        """Forces immediate processing - no-op since we flush after each write."""
        pass


def create_trace_processor(output_dir: str = "agent_data/traces") -> JsonTraceProcessor:
    """Factory function to create a JsonTraceProcessor."""
    return JsonTraceProcessor(output_dir=output_dir)
