from datetime import datetime
from textual.widgets import Static, Label, LoadingIndicator, RichLog
from textual.containers import Container, Vertical
from textual.app import ComposeResult
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown as RichMarkdown
from rich import box
from rich.align import Align
from rich.console import Group


class ChatBubble(Container):
    """A chat bubble widget using Rich Panels for a scientific/technical look."""
    def __init__(self, content: str, sender: str = "user", token_usage: dict = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.content = content
        self.sender = sender
        self.token_usage = token_usage
        self.timestamp = datetime.now().strftime("%H:%M:%S")
        self.add_class(f"message-{sender}")

    def _format_usage(self) -> str:
        """Format token usage as a dim subtitle string."""
        if not self.token_usage:
            return ""
        parts = []
        inp = self.token_usage.get("input_tokens", 0)
        out = self.token_usage.get("output_tokens", 0)
        total = self.token_usage.get("total_tokens", 0)
        if inp:
            parts.append(f"in:{inp:,}")
        if out:
            parts.append(f"out:{out:,}")
        if total:
            parts.append(f"total:{total:,}")
        return " [dim]" + " · ".join(parts) + " tokens[/]" if parts else ""

    def compose(self) -> ComposeResult:
        if self.sender == "agent":
            header = f"[bold #a6e3a1]ADAMS[/] [dim]· {self.timestamp}[/]"
            subtitle = self._format_usage()
            panel = Panel(
                RichMarkdown(self.content),
                title=header,
                title_align="left",
                subtitle=subtitle if subtitle else None,
                subtitle_align="right",
                border_style="#a6e3a1",
                box=box.ROUNDED,
                padding=(0, 1),
            )
            yield Static(panel, classes="bubble-content", id="bubble-static")
        else:
            header = f"[bold #cdd6f4]You[/] [dim]· {self.timestamp}[/]"
            # Compact user bubble — simple thin border
            panel = Panel(
                Text(self.content, style="#cdd6f4"),
                title=header,
                title_align="right",
                border_style="#45475a",
                box=box.ROUNDED,
                padding=(0, 1),
            )
            yield Static(Align.right(panel), classes="bubble-content", id="bubble-static")

    def update_content(self, new_content: str):
        """Updates the content of the bubble dynamically."""
        self.content = new_content
        static_widget = self.query_one("#bubble-static", Static)

        if self.sender == "agent":
            header = f"[bold #a6e3a1]ADAMS[/] [dim]· {self.timestamp}[/]"
            panel = Panel(
                RichMarkdown(self.content),
                title=header,
                title_align="left",
                border_style="#a6e3a1",
                box=box.ROUNDED,
                padding=(0, 1),
            )
            static_widget.update(panel)
        else:
            header = f"[bold #cdd6f4]You[/] [dim]· {self.timestamp}[/]"
            panel = Panel(
                Text(self.content, style="#cdd6f4"),
                title=header,
                title_align="right",
                border_style="#45475a",
                box=box.ROUNDED,
                padding=(0, 1),
            )
            static_widget.update(Align.right(panel))


class LoadingBubble(Static):
    """Shows a live action log of what the agent is doing."""

    _ICONS = {
        "thinking": "◆",
        "agent": "◆",
        "tool": "▸",
        "generation": "⚡",
        "complete": "✓",
        "input": "→",
        "idle": "●",
    }

    def __init__(self, **kwargs) -> None:
        super().__init__("", **kwargs)
        self.timestamp = datetime.now().strftime("%H:%M:%S")
        self.actions: list[str] = []
        self.current_status = "Initializing..."
        self.add_class("message-agent")
        self._collapsed = False
        self._rebuild_panel()

    def _rebuild_panel(self):
        """Build and set the panel renderable."""
        header = f"[bold #a6e3a1]ADAMS[/] [dim]· {self.timestamp}[/]"

        if not self.actions:
            body = f"  [dim italic #a6e3a1]● {self.current_status}[/]"
        else:
            lines = []
            for action in self.actions:
                lines.append(f"  [#a6e3a1]{action}[/]")
            if not self._collapsed:
                lines.append(f"  [bold #cba6f7]⟳ {self.current_status}[/]")
            body = "\n".join(lines)

        panel = Panel(
            body,
            title=header,
            title_align="left",
            border_style="#a6e3a1" if not self._collapsed else "#45475a",
            box=box.ROUNDED,
            padding=(0, 1),
        )
        self.update(panel)

    def update_status(self, text: str):
        """Called with each new trace event. Appends to action list and re-renders."""
        if not text or text == self.current_status:
            return
        self.current_status = text

        icon = "●"
        text_lower = text.lower()
        for keyword, ic in self._ICONS.items():
            if keyword in text_lower:
                icon = ic
                break

        ts = datetime.now().strftime("%H:%M:%S")
        self.actions.append(f"[dim]{ts}[/]  {icon}  {text}")
        self._rebuild_panel()

        # Auto-scroll the chat history
        try:
            history = self.app.query_one("#chat-history")
            history.scroll_end(animate=False)
        except Exception:
            pass

    def collapse(self):
        """Collapse into a compact summary after agent completes."""
        self._collapsed = True
        n = len(self.actions)
        elapsed = ""
        try:
            start = datetime.strptime(self.timestamp, "%H:%M:%S")
            end = datetime.now()
            delta = (end.replace(year=start.year, month=start.month, day=start.day) - start.replace(year=start.year, month=start.month, day=start.day)).seconds
            elapsed = f" in {delta}s"
        except Exception:
            pass

        # Replace the full action log with a single summary line
        self.actions = [f"[dim]{self.timestamp}[/]  ✓  [dim]{n} actions{elapsed}[/]"]
        self.current_status = "Complete"
        self._rebuild_panel()

    _LOG_STYLES = {
        "ERROR":   ("✗", "#f38ba8"),
        "WARNING": ("⚠", "#f9e2af"),
        "INFO":    ("ℹ", "#b4befe"),
        "DEBUG":   ("·", "#6c7086"),
    }

    def add_log_record(self, text: str, level: str = "INFO",
                       logger_name: str = "", timestamp: str = ""):
        """Add a Python log record with level-based styling, distinct from trace events."""
        icon, color = self._LOG_STYLES.get(level, ("·", "#6c7086"))
        ts = timestamp or datetime.now().strftime("%H:%M:%S")

        # Shorten logger name (e.g. "adams.pipeline.docking" → "docking")
        short_name = logger_name.rsplit(".", 1)[-1] if logger_name else ""
        prefix = f"[dim]{short_name}[/] " if short_name else ""

        self.actions.append(f"[dim]{ts}[/]  [{color}]{icon}[/]  {prefix}{text}")
        self._rebuild_panel()

        # Auto-scroll the chat history
        try:
            history = self.app.query_one("#chat-history")
            history.scroll_end(animate=False)
        except Exception:
            pass
