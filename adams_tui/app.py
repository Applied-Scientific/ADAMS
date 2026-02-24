
import asyncio
import os
from pathlib import Path
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, ContentSwitcher, Select
from textual.screen import Screen

# Import new modules
from .components.dashboard import Dashboard
from .components.screens import ChatScreen, APIKeyModal
from .components.logs_screen import LogsScreen
from .components.messages import AgentResponse, LogMessage, TraceEvent, StageChanged
from .components.logging import TextualLogHandler
from .components.widgets import ChatBubble
from .components.trace_monitor import TraceMonitor

try:
    from adams.executive_agent import create_agent, setup_tracing
    from adams.path_config import set_agent_data_path
    from adams.shutdown_manager import ShutdownManager
    from agents import Runner, SQLiteSession
    IMPORT_ERROR = None
except ImportError as e:
    Runner = None
    create_agent = None
    ShutdownManager = None
    IMPORT_ERROR = str(e)


class AdamsApp(App):
    """ADAMS TUI Application - Toad Edition."""
    
    CSS_PATH = "styles.tcss"
    BINDINGS = [
        ("d", "show_dashboard", "Dashboard"),
        ("c", "show_chat", "Agent"),
        ("l", "show_logs", "Logs"),
        ("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self):
        super().__init__()
        self.agent = None
        self.session = None
        self.session_id = f"tui_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.trace_processor = None
        self.system_online = False
        self.selected_model = "gpt-5" # Default
        self.provider_map = {
            "gpt": "OpenAI",
            "anthropic": "Anthropic",
            "gemini": "Gemini"
        }
        self.trace_processor = None
        self.shutdown_manager = None

    def get_provider_for_model(self, model: str) -> str:
        model_lower = model.lower()
        for keyword, provider in self.provider_map.items():
            if keyword in model_lower:
                return provider
        return "OpenAI" # Default

    def compose(self) -> ComposeResult:
        
        with Container(id="sidebar"):
            yield Button("◈ Dashboard", id="btn-dashboard", classes="-active")
            yield Button("▸ Agent", id="btn-chat")
            yield Button("◇ Logs", id="btn-logs")
        
        with Container(id="content-area"):
            with ContentSwitcher(initial="view-dashboard"):
                yield Dashboard(id="view-dashboard")
                yield ChatScreen(id="view-chat")
                yield LogsScreen(data_dir=str(Path.cwd() / "agent_data"), id="view-logs")
        
        # Invisible trace monitor widget
        yield TraceMonitor(id="trace-monitor")
        
    def on_mount(self) -> None:
        self.title = "ADAMS // DOCKING AGENT"
        self.check_api_key()

    def check_api_key(self):
        """Check for API key and prompt if missing."""
        provider = self.get_provider_for_model(self.selected_model)
        
        # Determine Env Var based on provider
        if provider == "OpenAI":
            env_var = "OPENAI_API_KEY"
        elif provider == "Anthropic":
            env_var = "ANTHROPIC_API_KEY"
        elif provider == "Gemini":
            env_var = "GEMINI_API_KEY"
        else:
            env_var = "OPENAI_API_KEY"
            
        api_key = os.environ.get(env_var)
        
        if api_key:
            self.initialize_agent(api_key, self.selected_model)
        else:
            self.system_online = False
            self._update_status_offline()

    def on_select_changed(self, event: Select.Changed) -> None:
        """Handle model selection change."""
        if event.select.id == "model-select":
            self.selected_model = str(event.value)
            self.notify(f"Model changed to {self.selected_model}")
            # Update dashboard model tag
            try:
                dashboard = self.query(Dashboard).first()
                if dashboard:
                    dashboard.current_model = self.selected_model
            except Exception:
                pass
            # Re-check system status with new model
            self.check_api_key()

    def initialize_agent(self, api_key: str, model: str):
        """Initialize the ADAMS agent."""
        # Clean API key to remove potential smart quotes or invisible chars
        api_key = api_key.replace("\u201c", "").replace("\u201d", "").replace("\u2018", "").strip()
        # Enforce ASCII only
        api_key = api_key.encode("ascii", errors="ignore").decode("ascii")
        
        provider = self.get_provider_for_model(model)

        # Mark runtime as non-interactive so agent tools never call blocking stdin prompts.
        os.environ["ADAMS_UI_MODE"] = "tui"
        os.environ["ADAMS_NON_INTERACTIVE"] = "1"
        
        # Set Env Var
        if provider == "OpenAI":
            os.environ["OPENAI_API_KEY"] = api_key
        elif provider == "Anthropic":
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif provider == "Gemini":
            os.environ["GEMINI_API_KEY"] = api_key

        if create_agent:
            try:
                working_dir = Path.cwd()
                agent_data_path = working_dir / "agent_data"
                set_agent_data_path(path=agent_data_path)
                self.trace_processor = setup_tracing()
                
                # Create agent with selected model
                self.agent = create_agent(model=model)
                self.session = SQLiteSession(self.session_id)
                self.system_online = True
                
                # Setup graceful shutdown
                if ShutdownManager:
                    self.shutdown_manager = ShutdownManager()
                    
                    def cleanup_on_shutdown():
                        """Cleanup callback for graceful shutdown."""
                        try:
                            from adams.format_trace import format_trace_file
                            format_trace_file()
                        except Exception:
                            pass
                    
                    self.shutdown_manager.register_cleanup(cleanup_on_shutdown)
                    self.shutdown_manager.setup_handlers()
                
                self.notify(f"System Online. Agent Active ({model}).", severity="information")
                
                dashboard = self.query(Dashboard).first()
                if dashboard:
                    dashboard.online = True
                    dashboard.current_model = model
                    dashboard.update_session_info(
                        str(working_dir),
                        str(agent_data_path),
                        str(self.trace_processor.filepath)
                    )
                
                # Start trace monitor on the active trace file
                try:
                    monitor = self.query_one("#trace-monitor", TraceMonitor)
                    monitor.set_trace_file(self.trace_processor.filepath)
                except Exception:
                    pass
                    
            except Exception as e:
                self.notify(f"Initialization Failed: {e}", severity="error")
                self.system_online = False
                self._update_status_offline(str(e))
        else:
            self.system_online = False
            self._update_status_offline(IMPORT_ERROR if IMPORT_ERROR else "Agent not found")
            
            error_msg = f"Initialization Failed: {IMPORT_ERROR}" if IMPORT_ERROR else "Initialization Failed: Agent not found"
            self.notify(error_msg, severity="error", timeout=10)

    def on_unmount(self) -> None:
        """Format trace file on app exit (converts .jsonl -> .json)."""
        if self.trace_processor:
            try:
                from adams.format_trace import format_trace_file
                format_trace_file()
            except Exception:
                pass

    def _update_status_offline(self, error: str = None):
        """Helper to set dashboard to offline state."""
        dashboard = self.query(Dashboard).first()
        if dashboard:
            dashboard.online = False
            if error:
                dashboard.set_error(error)

    def action_show_dashboard(self) -> None:
        self.query_one(ContentSwitcher).current = "view-dashboard"
        self.update_sidebar("btn-dashboard")

    def action_show_chat(self) -> None:
        if not self.system_online:
            provider = self.get_provider_for_model(self.selected_model)
            modal = APIKeyModal(provider=provider)
            self.push_screen(modal, self.handle_api_key_input)
            return

        self.query_one(ContentSwitcher).current = "view-chat"
        self.update_sidebar("btn-chat")

    def action_show_logs(self) -> None:
        """Switch to the Logs view."""
        self.query_one(ContentSwitcher).current = "view-logs"
        self.update_sidebar("btn-logs")
        # Refresh sessions when switching to logs
        try:
            logs = self.query_one("#view-logs", LogsScreen)
            logs.refresh_sessions()
        except Exception:
            pass

    def handle_api_key_input(self, key: str | None) -> None:
        if key: 
            self.initialize_agent(key, self.selected_model)
            if self.system_online:
                 self.action_show_chat()

    def update_sidebar(self, active_id: str):
        sidebar = self.query_one("#sidebar")
        for btn in sidebar.query("Button"):
            if btn.id == active_id:
                btn.add_class("-active")
            else:
                btn.remove_class("-active")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-dashboard":
            self.action_show_dashboard()
        elif event.button.id == "btn-chat":
            self.action_show_chat()
        elif event.button.id == "btn-logs":
            self.action_show_logs()

    def on_log_message(self, message: LogMessage) -> None:
        try:
            from .components.widgets import LoadingBubble
            chat_view = self.query_one("ChatScreen", expect_type=ChatScreen)
            loading = chat_view.query_one("#loading-bubble", LoadingBubble)
            loading.add_log_record(
                text=message.text,
                level=message.level,
                logger_name=message.logger_name,
                timestamp=message.timestamp,
            )
        except Exception:
            pass

    def on_trace_event(self, message: TraceEvent) -> None:
        """Extract AI reasoning from trace events and display in LoadingBubble."""
        try:
            from .components.widgets import LoadingBubble
            chat_view = self.query_one("ChatScreen", expect_type=ChatScreen)
            loading = chat_view.query_one("#loading-bubble", LoadingBubble)
        except Exception:
            return

        details = message.details
        etype = message.event_type

        # Tool call completed — show a summary of its output
        if etype == "tool_call_end":
            tool = details.get("tool", "")
            output = details.get("output")
            if output and tool:
                summary = self._summarize_output(output)
                if summary:
                    loading.add_log_record(
                        text=f"[dim]{tool}[/] → {summary}",
                        level="INFO",
                        logger_name="trace",
                        timestamp=message.timestamp[:8] if len(message.timestamp) > 8 else "",
                    )

        # Log records from the pipeline
        elif etype == "log_record":
            loading.add_log_record(
                text=details.get("message", ""),
                level=details.get("level", "INFO"),
                logger_name=details.get("logger", ""),
                timestamp=message.timestamp[:8] if len(message.timestamp) > 8 else "",
            )

    @staticmethod
    def _summarize_output(output, max_len: int = 120) -> str:
        """Create a short summary from a tool output value."""
        if output is None:
            return ""
        if isinstance(output, str):
            # Truncate long strings
            text = output.replace("\n", " ").strip()
            return text[:max_len] + "…" if len(text) > max_len else text
        if isinstance(output, dict):
            # Pick the most informative key
            for key in ("message", "output", "exact_plan", "description",
                        "feedback", "error", "protonated_pdb", "output_path"):
                if key in output and output[key]:
                    val = str(output[key]).replace("\n", " ").strip()
                    return val[:max_len] + "…" if len(val) > max_len else val
            # Fallback: show keys
            keys = list(output.keys())[:5]
            return f"{{{'  '.join(keys)}{'…' if len(output) > 5 else ''}}}"
        return str(output)[:max_len]

    def on_stage_changed(self, message: StageChanged) -> None:
        """Update dashboard stage indicator and chat loading bubble."""
        try:
            dashboard = self.query(Dashboard).first()
            if dashboard:
                dashboard.current_stage = message.detail
        except Exception:
            pass

        # Also update chat loading bubble status
        try:
            chat_view = self.query_one("ChatScreen", expect_type=ChatScreen)
            if chat_view and message.stage not in ("idle",):
                chat_view.update_status(message.detail)
        except Exception:
            pass

    async def on_agent_response(self, message: AgentResponse) -> None:
        chat_view = self.query_one("ChatScreen", expect_type=ChatScreen)
        if chat_view:
            # Collapse the loading bubble into a summary instead of removing it
            try:
                from .components.widgets import LoadingBubble
                loading = chat_view.query_one("#loading-bubble", LoadingBubble)
                loading.collapse()
                loading.id = None  # Remove the ID so set_loading_state doesn't try to remove it
            except Exception:
                pass

            # Exit Loading State (hides stop button, re-enables input)
            chat_view.set_loading_state(False)
            
            # Update Usage if present
            if message.token_usage:
                chat_view.update_usage_display(message.token_usage)

            history = chat_view.query_one("#chat-history")
            
            # Instant render with token usage shown on the bubble
            history.mount(ChatBubble(message.content, sender="agent", token_usage=message.token_usage))
            history.scroll_end(animate=True)

        # Reset stage to idle after response
        try:
            dashboard = self.query(Dashboard).first()
            if dashboard:
                dashboard.current_stage = "Idle"
        except Exception:
            pass

def main():
    import atexit
    import signal

    # Suppress the noisy KeyboardInterrupt traceback from the SDK's
    # background tracing thread when the user presses Ctrl+C.
    _original_excepthook = None

    def _silence_keyboard_interrupt_in_atexit():
        """Patch threading join to swallow KeyboardInterrupt on shutdown."""
        import threading
        _orig_join = threading.Thread.join
        def _patched_join(self, timeout=None):
            try:
                _orig_join(self, timeout=timeout)
            except KeyboardInterrupt:
                pass
        threading.Thread.join = _patched_join

    _silence_keyboard_interrupt_in_atexit()

    # Force UTF-8 encoding to prevent Unicode errors in TUI
    os.environ["PYTHONIOENCODING"] = "utf-8"
    app = AdamsApp()
    try:
        app.run()
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
