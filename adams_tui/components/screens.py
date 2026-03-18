import asyncio
import logging
from textual.screen import ModalScreen
from textual.containers import Container, Vertical, VerticalScroll
from textual.widgets import Button, Input, Label, RichLog
from textual.app import ComposeResult
from .widgets import ChatBubble, LoadingBubble
from .messages import AgentResponse
from .logging import TextualLogHandler
from adams.path_config import set_agent_data_path
from pathlib import Path
from agents import Runner, RunConfig

class APIKeyModal(ModalScreen):
    """Modal dialog to request API Key."""
    
    def __init__(self, provider: str = "OpenAI"):
        super().__init__()
        self.provider = provider
        self.key_prefix = "sk-" if provider == "OpenAI" else "" # generic check

    def compose(self) -> ComposeResult:
        with Container(id="api-key-dialog"):
            yield Label(f"[bold orange1]Authentication Required[/]\n\nPlease enter your {self.provider} API Key to initialize ADAMS.")
            yield Label("", id="error-message", classes="error-text hidden")
            yield Input(placeholder="API Key...", password=True, id="key-input")
            with Container(classes="dialog-buttons"):
                yield Button("Connect", variant="primary", id="btn-submit")
                yield Button("Cancel", variant="error", id="btn-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-submit":
            self.submit_key()
        elif event.button.id == "btn-cancel":
            self.dismiss(None)
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.submit_key()

    def submit_key(self):
        key = self.query_one("#key-input").value.strip()
        error_label = self.query_one("#error-message")
        
        # Basic validation
        if not key:
            error_label.update("Invalid Key: Cannot be empty")
            error_label.remove_class("hidden")
            return
            
        if self.provider == "OpenAI" and not key.startswith("sk-"):
            error_label.update("Invalid Key: OpenAI keys usually start with 'sk-'")
            error_label.remove_class("hidden")
            return
            
        if self.provider == "Anthropic" and not key.startswith("sk-ant"):
             # Optional: Loose check for Anthropic
             pass 

        if len(key) < 10:
             error_label.update("Invalid Key: Too short")
             error_label.remove_class("hidden")
             return

        error_label.add_class("hidden")
        self.dismiss(key)

class ChatScreen(Container):
    """Main chat interface."""
    BINDINGS = [("escape", "stop_agent", "Stop Agent")]

    def action_stop_agent(self) -> None:
        """Escape key handler — stop the agent if running."""
        try:
            self.query_one("#loading-bubble")
            self.stop_agent()
        except Exception:
            pass
    
    def compose(self) -> ComposeResult:
        with Vertical():
            with VerticalScroll(id="chat-history"):
                yield ChatBubble(
                    "Ready. I can run **molecular docking**, **MD simulations**, "
                    "and **binding analysis**.\n\n"
                    "Try: *\"Dock 1M17.pdb against ligands_inSMILES.csv\"*",
                    sender="agent",
                )


            # Usage Stats Bar
            yield Label("", id="usage-stats", classes="usage-stats hidden")
            
            # ... (Input container remains same)
            with Container(id="input-container"):
                 yield Input(placeholder="> Enter command...", id="agent-input")
                 yield Button("■ Stop", id="btn-stop", classes="stop-btn hidden")

    def on_mount(self) -> None:
        self.query_one("#agent-input").focus()

    def load_initial_history(self, conversation_history: list) -> None:
        """Replace the default welcome bubble with resumed conversation history."""
        if not conversation_history:
            return
        chat_container = self.query_one("#chat-history")
        # Remove the default "Ready..." bubble (first child)
        children = list(chat_container.children)
        if children:
            children[0].remove()
        for item in conversation_history:
            role = item.get("role", "user")
            content = item.get("content", "")
            sender = "user" if role == "user" else "agent"
            chat_container.mount(ChatBubble(content, sender=sender))
        chat_container.scroll_end(animate=False)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        if not event.value.strip():
            return
            
        user_input = event.value
        event.input.value = ""
        
        # Add user message
        chat_container = self.query_one("#chat-history")
        chat_container.mount(ChatBubble(user_input, sender="user"))
        chat_container.scroll_end(animate=True)
        
        # Enter Loading State
        self.set_loading_state(True)
        
        # Run agent
        self.app.run_worker(self.run_agent_task(user_input), exclusive=True, group="agent")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-stop":
            self.stop_agent()

    def stop_agent(self) -> None:
        """Cancel the running agent worker."""
        # Cancel all workers in the "agent" group
        self.app.workers.cancel_group(self.app, "agent")
        
        # Exit loading state
        self.set_loading_state(False)
        
        # Add interrupted message to chat
        chat_container = self.query_one("#chat-history")
        chat_container.mount(ChatBubble("Agent interrupted by user.", sender="agent"))
        chat_container.scroll_end(animate=True)

        # Reset dashboard stage
        try:
            from .dashboard import Dashboard
            dashboard = self.app.query(Dashboard).first()
            if dashboard:
                dashboard.current_stage = "Idle"
        except Exception:
            pass

    def set_loading_state(self, is_loading: bool):
        input_widget = self.query_one("#agent-input")
        chat_container = self.query_one("#chat-history")
        stop_btn = self.query_one("#btn-stop")
        
        input_widget.disabled = is_loading
        
        if is_loading:
             # Show stop button, mount loading bubble
             stop_btn.remove_class("hidden")
             loading = LoadingBubble(id="loading-bubble")
             chat_container.mount(loading)
             chat_container.scroll_end(animate=True)
        else:
             # Hide stop button, remove loading bubble
             stop_btn.add_class("hidden")
             try:
                 self.query_one("#loading-bubble").remove()
             except:
                 pass
             input_widget.focus()

    def update_status(self, text: str):
        """Updates the status label with live activity."""
        try:
             # Update the loading bubble if it exists
             self.query_one("#loading-bubble", LoadingBubble).update_status(text)
        except:
             # Widget might not exist yet or anymore
             pass
    
    def update_usage_display(self, usage: dict):
        """Update the usage stats label."""
        if not usage:
            return
            
        label = self.query_one("#usage-stats")
        # Example usage dict: {'completion_tokens': 10, 'prompt_tokens': 20, 'total_tokens': 30}
        text = f"Tokens: {usage.get('total_tokens', 0)} (Prompt: {usage.get('prompt_tokens', 0)} | Completion: {usage.get('completion_tokens', 0)})"
        label.update(text)
        label.remove_class("hidden")

    def update_usage_display(self, usage):
        """Show usage stats below the last agent message."""
        try:
            history = self.query_one("#chat-history")
            parts = []
            if hasattr(usage, "input_tokens") and usage.input_tokens:
                parts.append(f"In: {usage.input_tokens}")
            if hasattr(usage, "output_tokens") and usage.output_tokens:
                parts.append(f"Out: {usage.output_tokens}")
            if hasattr(usage, "total_tokens") and usage.total_tokens:
                parts.append(f"Total: {usage.total_tokens}")
            if parts:
                from textual.widgets import Label
                usage_label = Label(f"[dim]Tokens: {' | '.join(parts)}[/]", classes="usage-stats")
                history.mount(usage_label)
        except Exception:
            pass

    async def run_agent_task(self, prompt: str):
        if not self.app.agent:
             self.app.post_message(AgentResponse("System Offline. Agent not initialized.", is_error=True))
             return

        # Write user input to trace file
        if self.app.trace_processor:
            self.app.trace_processor.write_user_input(prompt)

        # setup logging capture
        handler = TextualLogHandler(self.app)
        handler.setLevel(logging.INFO)
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        try:
            loop = asyncio.get_running_loop()
            def _run():
                # Re-set agent_data path in this executor thread's context.
                # ContextVar values from the main asyncio thread are NOT
                # inherited by run_in_executor threads, so tools that call
                # get_agent_data_path() would fail with "not configured".
                try:
                    agent_data_path = Path.cwd() / "agent_data"
                    set_agent_data_path(path=agent_data_path)
                except Exception:
                    pass

                # Configure run to include full trace data
                config = RunConfig(trace_include_sensitive_data=True) if RunConfig else None
                
                return Runner.run_sync(
                    self.app.agent, 
                    prompt, 
                    session=self.app.session, 
                    max_turns=50,
                    run_config=config
                )
            result = await loop.run_in_executor(None, _run)

            # Aggregate token usage across all model responses
            usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            try:
                for resp in result.raw_responses:
                    if hasattr(resp, "usage") and resp.usage:
                        usage["input_tokens"] += resp.usage.input_tokens or 0
                        usage["output_tokens"] += resp.usage.output_tokens or 0
                        usage["total_tokens"] += resp.usage.total_tokens or 0
            except Exception:
                pass
              
             # Write agent output to trace file
            if self.app.trace_processor:
                self.app.trace_processor.write_agent_output(result.final_output)

            self.app.post_message(AgentResponse(result.final_output, stream=True, token_usage=usage))
            
        except Exception as e:
            error_str = str(e)
            model_name = getattr(self.app, "selected_model", "the model")
            if "Incorrect API key provided" in error_str:
                clean_error = "Authorization Invalid. Please check your credentials."
            elif "401" in error_str:
                clean_error = "Authorization Invalid (401). Please check your API key."
            elif "429" in error_str:
                clean_error = (
                    f"Rate Limit Exceeded for {model_name}. "
                    "This model may have low quota — try again in a moment, "
                    "or switch to a different model from the Dashboard."
                )
            elif "404" in error_str:
                clean_error = f"Model not found: {model_name}. Please select a different model."
            else:
                clean_error = f"System Failure: {error_str[:120]}"

            self.app.post_message(AgentResponse(clean_error, is_error=True))
        finally:
            root_logger.removeHandler(handler)
