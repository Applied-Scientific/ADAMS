import threading
from pathlib import Path

import psutil
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Label, Static, Select
from textual.app import ComposeResult
from textual.reactive import reactive
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich import box

from .colors import GREEN, RED, YELLOW, BLUE, MAUVE, TEAL, PEACH, TEXT, DIM, SURFACE, PROVIDER_COLORS


class Dashboard(Container):
    """Dashboard view with system metrics, model tag, and agent stage indicator."""

    online = reactive(False)
    current_model = reactive("gpt-5")
    current_stage = reactive("Idle")
    cpu_percent = reactive(0.0)
    mem_percent = reactive(0.0)
    thread_count = reactive(0)

    def compose(self) -> ComposeResult:
        with Vertical(id="dashboard-grid"):
            # ── Header with description ──
            header_text = (
                "[bold white]Agent-Driven Autonomous Molecular Simulations[/]\n"
                "[dim]An agentic workflow that automates molecular docking and MD\n"
                "simulation based on user-provided prompts.\n"
                "Protein preprocessing, binding pocket discovery, docking\n"
                "and stability analysis.[/]"
            )
            yield Static(Panel(
                Align.center(header_text),
                title="[bold green]ADAMS[/]",
                subtitle="[dim]by Rhizome Research[/]",
                border_style="green",
                box=box.HEAVY,
                padding=(1, 2)
            ))

            # ── System Metrics Row ──
            with Horizontal(classes="metrics-row"):
                with Container(classes="metric-card"):
                    yield Label("CPU", classes="metric-title")
                    yield Label("0 %", id="cpu-val", classes="metric-value")
                with Container(classes="metric-card"):
                    yield Label("Memory", classes="metric-title")
                    yield Label("0 %", id="mem-val", classes="metric-value")
                with Container(classes="metric-card"):
                    yield Label("GPU", classes="metric-title")
                    yield Label(self._detect_gpu(), id="gpu-val", classes="metric-value")
                with Container(classes="metric-card"):
                    yield Label("Threads", classes="metric-title")
                    yield Label("0", id="thread-val", classes="metric-value")

            # ── Status + Model Tag + Stage ──
            with Container(classes="card"):
                yield Label("System Status", classes="card-title")
                with Horizontal(classes="status-row"):
                    yield Label("● OFFLINE", id="status-label", classes="status-offline")
                    yield Label("", id="model-tag", classes="model-tag")
                yield Label("[dim]Stage:[/] Idle", id="stage-label", classes="stage-indicator")

            # ── Model Selection Card ──
            with Container(classes="card"):
                yield Label("Model Selection", classes="card-title")
                yield Select(
                    [
                        ("GPT-5.2 (Reasoning)", "gpt-5.2"),
                        ("GPT-5", "gpt-5"),
                        ("GPT-5 Mini", "gpt-5-mini"),
                        ("Claude Sonnet 4.5", "anthropic/claude-sonnet-4-5-20250929"),
                        ("Claude Opus 4.5", "anthropic/claude-opus-4-5-20251101"),
                        ("Gemini 3 Pro Preview", "gemini/gemini-3-pro-preview"),
                        ("Gemini 2.5 Pro", "gemini/gemini-2.5-pro"),
                    ],
                    prompt="Select Model",
                    id="model-select",
                    value="gpt-5",
                )

            # ── Session Info Card ──
            with Container(classes="card", id="session-card"):
                yield Label("Session Information", classes="card-title")
                yield Label("[dim]Waiting for initialization...[/]", id="work-dir-label", classes="label")
                yield Label("", id="data-dir-label", classes="label")
                yield Label("", id="trace-file-label", classes="label")

            # ── Active Protocols Card ──
            with Container(classes="card"):
                yield Label("Active Protocols", classes="card-title")
                yield Label("Docking Agent: [dim]Locked[/dim]", id="docking-status", classes="label")

    # ── Timers ──────────────────────────────────────────────────────
    def on_mount(self) -> None:
        self.set_interval(2.0, self._refresh_metrics)

    def _refresh_metrics(self) -> None:
        """Poll system metrics."""
        try:
            self.cpu_percent = psutil.cpu_percent(interval=0)
            self.mem_percent = psutil.virtual_memory().percent
            self.thread_count = threading.active_count()
        except Exception:
            pass

    # ── Reactive watchers ──────────────────────────────────────────
    def watch_cpu_percent(self, value: float) -> None:
        try:
            color = GREEN if value < 60 else (YELLOW if value < 85 else RED)
            self.query_one("#cpu-val").update(f"[{color}]{value:.0f} %[/]")
        except Exception:
            pass

    def watch_mem_percent(self, value: float) -> None:
        try:
            color = GREEN if value < 60 else (YELLOW if value < 85 else RED)
            self.query_one("#mem-val").update(f"[{color}]{value:.0f} %[/]")
        except Exception:
            pass

    def watch_thread_count(self, value: int) -> None:
        try:
            self.query_one("#thread-val").update(f"[{TEAL}]{value}[/]")
        except Exception:
            pass

    def watch_online(self, online: bool) -> None:
        try:
            status_label = self.query_one("#status-label")
            docking_label = self.query_one("#docking-status")

            if online:
                status_label.update(f"[{GREEN}]ONLINE[/]")
                status_label.remove_class("status-offline")
                status_label.add_class("status-online")
                docking_label.update(f"Docking Agent: [{GREEN}]Ready[/]")
            else:
                status_label.update(f"[{RED}]OFFLINE[/]")
                status_label.remove_class("status-online")
                status_label.add_class("status-offline")
                docking_label.update("Docking Agent: [dim]Locked[/dim]")
        except Exception:
            pass

    def watch_current_model(self, model: str) -> None:
        """Update the model tag badge."""
        try:
            provider = self._provider_for(model)
            color = PROVIDER_COLORS.get(provider, DIM)
            self.query_one("#model-tag").update(f" [{color}]{provider}[/] ")
        except Exception:
            pass

    def watch_current_stage(self, stage: str) -> None:
        try:
            self.query_one("#stage-label").update(f"[dim]Stage:[/] [bold]{stage}[/]")
        except Exception:
            pass

    # ── Public helpers ─────────────────────────────────────────────
    def set_error(self, message: str) -> None:
        try:
            status_label = self.query_one("#status-label")
            status_label.update(f"[{RED}]OFFLINE: {message}[/]")
            status_label.remove_class("status-online")
            status_label.add_class("status-offline")
        except Exception:
            pass

    def update_session_info(self, work_dir: str, data_dir: str, trace_file: str) -> None:
        """Updates the session information labels with shortened paths."""
        try:
            home = str(Path.home())
            def _short(p: str) -> str:
                return p.replace(home, "~") if p.startswith(home) else p

            self.query_one("#work-dir-label").update(f"[bold]Working Dir:[/] {_short(work_dir)}")
            self.query_one("#data-dir-label").update(f"[bold]Data Dir:[/] {_short(data_dir)}")
            self.query_one("#trace-file-label").update(f"[bold]Trace File:[/] {_short(trace_file)}")
        except Exception:
            pass

    # ── Private helpers ────────────────────────────────────────────
    @staticmethod
    def _detect_gpu() -> str:
        """Try to detect GPU info."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split("\n")[0]
        except Exception:
            pass
        try:
            import platform
            if platform.processor() == "arm" or "Apple" in platform.platform():
                return "[dim]Apple Silicon (MPS)[/]"
        except Exception:
            pass
        return "[dim]No GPU[/]"

    @staticmethod
    def _provider_for(model: str) -> str:
        """Derive provider name from model string."""
        m = model.lower()
        if "claude" in m or "anthropic" in m:
            return "Anthropic"
        if "gemini" in m:
            return "Gemini"
        return "OpenAI"
