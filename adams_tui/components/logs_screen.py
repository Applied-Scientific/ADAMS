"""
Logs Screen — Pipeline Log File Browser

Two-pane layout:
  Left  – Log file list (from agent_data/logs/)
  Right – Log file content with level-based colouring
"""

import os
import re
from datetime import datetime
from pathlib import Path

from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Label, Static, ListView, ListItem, RichLog
from textual.app import ComposeResult
from textual.reactive import reactive
from rich.text import Text

from .colors import GREEN, RED, YELLOW, BLUE, MAUVE, TEAL, PEACH, TEXT, DIM

# Log level → colour
_LEVEL_COLORS = {
    "ERROR": RED,
    "WARNING": YELLOW,
    "INFO": BLUE,
    "DEBUG": DIM,
}

# Regex to parse log lines:  YYYY-MM-DD HH:MM:SS - logger - LEVEL - message
_LOG_LINE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+-\s+(\S+)\s+-\s+(\w+)\s+-\s+(.*)$"
)


def _load_log_files(data_dir: str) -> list[dict]:
    """Discover .log files in agent_data/logs/, newest first."""
    logs_dir = os.path.join(data_dir, "logs")
    if not os.path.isdir(logs_dir):
        return []

    files = []
    for name in os.listdir(logs_dir):
        if not name.endswith(".log"):
            continue
        full_path = os.path.join(logs_dir, name)
        try:
            stat = os.stat(full_path)
            size_kb = stat.st_size / 1024
            mtime = datetime.fromtimestamp(stat.st_mtime)
            files.append({
                "name": name,
                "path": full_path,
                "size_kb": size_kb,
                "mtime": mtime,
            })
        except OSError:
            continue

    # Newest first
    files.sort(key=lambda f: f["mtime"], reverse=True)
    return files


def _format_size(size_kb: float) -> str:
    """Format file size for display."""
    if size_kb < 1:
        return "<1 KB"
    if size_kb < 1024:
        return f"{size_kb:.0f} KB"
    return f"{size_kb / 1024:.1f} MB"


class LogFileItem(ListItem):
    """A selectable log file entry in the file list."""
    def __init__(self, file_info: dict, **kwargs):
        super().__init__(**kwargs)
        self.file_info = file_info

    def compose(self) -> ComposeResult:
        f = self.file_info
        name = f["name"]
        ts = f["mtime"].strftime("%Y-%m-%d %H:%M")
        size = _format_size(f["size_kb"])

        # Extract run timestamp from filename if possible
        # Format: adams_pipeline_run_YYYYMMDD_HHMMSS.log
        run_id = ""
        if name.startswith("adams_pipeline_run_") and name.endswith(".log"):
            run_id = name[len("adams_pipeline_run_"):-len(".log")]

        title = f"[bold {TEXT}]{name}[/]"
        subtitle = f"[dim]{ts}  ·  {size}[/]"
        if run_id:
            title = f"[bold {TEXT}]Run {run_id}[/]"
            subtitle += f"\n[dim]{name}[/]"

        yield Label(f"{title}\n{subtitle}", classes="session-label")


class LogsScreen(Container):
    """Pipeline log file browser with file list and content viewer."""

    selected_log_path = reactive("")

    def __init__(self, data_dir: str = "agent_data", **kwargs):
        super().__init__(**kwargs)
        self.data_dir = data_dir
        self._log_files: list[dict] = []

    def compose(self) -> ComposeResult:
        with Horizontal(id="logs-container"):
            # Left pane: log file list
            with Vertical(id="session-pane"):
                yield Label("[bold]Log Files[/]", classes="pane-title")
                yield ListView(id="session-list")

            # Right pane: log content viewer
            with Vertical(id="log-pane"):
                yield Label("[bold]Log Output[/]", id="log-pane-title", classes="pane-title")
                yield RichLog(id="log-content", wrap=True, markup=True)

    def on_mount(self) -> None:
        self.load_log_files()

    def load_log_files(self) -> None:
        """Load and display available log files."""
        self._log_files = _load_log_files(self.data_dir)
        file_list = self.query_one("#session-list", ListView)
        file_list.clear()

        if not self._log_files:
            file_list.append(ListItem(Label("[dim]No log files found[/]")))
            return

        for f in self._log_files:
            file_list.append(LogFileItem(f))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        """Handle selection in the file list."""
        item = event.item
        if isinstance(item, LogFileItem):
            self._display_log_file(item.file_info)

    def _display_log_file(self, file_info: dict) -> None:
        """Read and display a log file with level-based colouring."""
        self.selected_log_path = file_info["path"]

        # Update title
        try:
            self.query_one("#log-pane-title").update(
                f"[bold]Log Output[/]  [dim]{file_info['name']}[/]"
            )
        except Exception:
            pass

        log_view = self.query_one("#log-content", RichLog)
        log_view.clear()

        try:
            with open(file_info["path"], "r", encoding="utf-8") as f:
                lines = f.readlines()
        except Exception as e:
            log_view.write(f"[{RED}]Error reading file: {e}[/]")
            return

        if not lines:
            log_view.write(f"[dim]Log file is empty[/]")
            return

        for line in lines:
            line = line.rstrip("\n\r")
            if not line:
                continue

            match = _LOG_LINE_RE.match(line)
            if match:
                timestamp, logger_name, level, message = match.groups()
                color = _LEVEL_COLORS.get(level, DIM)
                short_logger = logger_name.rsplit(".", 1)[-1] if logger_name else ""
                log_view.write(
                    f"[dim]{timestamp}[/]  [{color}]{level:<7}[/]  {message}"
                )
            else:
                # Non-matching lines (continuations, multi-line output)
                log_view.write(f"[dim]{line}[/]")

    def refresh_sessions(self) -> None:
        """Reload log files from disk (called by app when switching to logs tab)."""
        self.load_log_files()
