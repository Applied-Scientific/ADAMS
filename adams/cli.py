"""
    Interactive CLI entry point for the Biophysics Controller Agent.
    Provides enhanced keyboard navigation and command history.
"""

import argparse
import logging
import shutil
import textwrap
from pathlib import Path


from agents import Runner
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings

from .cli_docking import add_dock_subparser, run_dock_command
from .executive_agent import create_agent, setup_tracing
from .format_trace import format_trace_file
from .memory import add_session_tags
from .shutdown_manager import ShutdownManager
from .memory.custom_memory import (
    append_instructions,
    get_instructions,
    read_instructions_from_file,
    read_instructions_from_stdin,
    set_instructions,
)
from .memory.persistent_memory import clear_user_preferences
from .path_config import set_agent_data_path
from .utils.console_transcript import start_console_transcript
from .utils.secrets_manager import get_api_key
from .utils.session_utils import create_sdk_session

# Suppress OpenAI agents tracing warnings (e.g., "server error 503" messages)
logging.getLogger("openai.agents").setLevel(logging.ERROR)


def print_user_box(text: str):
    """
    Print user input in a full-width box with Unicode borders.

    Args:
        text: The user input text to display
    """
    term_width = shutil.get_terminal_size().columns

    # Ensure minimum width
    if term_width < 40:
        term_width = 80

    # Reserve space for box characters and padding (│ text │ = 4 chars)
    content_width = term_width - 4

    # Wrap text to fit
    wrapped_lines = textwrap.wrap(
        text,
        width=content_width,
        break_long_words=False,
        break_on_hyphens=False,
    )

    # Handle empty input
    if not wrapped_lines:
        wrapped_lines = [""]

    # Build box with Unicode characters
    title = "User"
    title_section = f"─ {title} "
    top = f"┌{title_section}{'─' * (term_width - len(title_section) - 2)}┐"
    bottom = f"└{'─' * (term_width - 2)}┘"

    print(top)
    for line in wrapped_lines:
        # Pad line to full width
        padded = line.ljust(content_width)
        print(f"│ {padded} │")
    print(bottom)


def get_framed_input(history: InMemoryHistory, key_bindings: KeyBindings) -> str:
    """
    Display a framed input box and get user input.

    Args:
        history: Command history for the input
        key_bindings: Custom key bindings

    Returns:
        The user's input string
    """

    # Create key bindings for submit
    submit_bindings = KeyBindings()

    @submit_bindings.add("enter")
    def submit(event):
        """Submit on Enter key."""
        event.current_buffer.validate_and_handle()

    # Merge with custom key bindings
    all_bindings = merge_key_bindings([key_bindings, submit_bindings])

    try:
        result = prompt(
            "\nUser > ",
            show_frame=False,
            history=history,
            key_bindings=all_bindings,
        )
        return result
    except (KeyboardInterrupt, EOFError) as e:
        raise e





def create_key_bindings() -> KeyBindings:
    """
    Create custom key bindings for enhanced CLI navigation.

    Note: prompt_toolkit has default bindings for word navigation (Ctrl+Left/Right),
    and we add Ctrl+W for word deletion (common terminal binding).

    Returns:
        KeyBindings: Configured key bindings with Ctrl+Arrow and word deletion support
    """
    kb = KeyBindings()

    # Ctrl+W: Delete word before cursor
    @kb.add("c-w")
    def delete_word_before_ctrlw(event):
        buffer = event.app.current_buffer
        start_pos = buffer.document.find_start_of_previous_word()
        if start_pos is not None:
            buffer.delete_before_cursor(count=buffer.cursor_position - start_pos)
        else:
            buffer.delete_before_cursor(count=buffer.cursor_position)

    # Delete word after cursor
    @kb.add("escape", "d")
    def delete_word_after(event):
        buff = event.app.current_buffer
        end_pos = buff.document.find_end_of_word_after_cursor()
        if end_pos is not None:
            buff.delete(count=end_pos)

    return kb


def format_session_trace():
    """
    Format the current session's trace file from JSONL to JSON.

    Attempts to format the most recent trace file. Errors are caught
    and reported but don't prevent session exit.
    """
    try:
        output_path = format_trace_file()
        print(f"\nTrace file formatted: {output_path}")
    except Exception as e:
        print(f"\nWarning: Could not format trace file: {e}")


def run_session_metadata_update(agent, session, trace_processor, max_turns=10):
    """
    Run one agent turn to say goodbye on exit. Used on exit/quit and Ctrl+C.
    Session metadata is owned by the controller; diagnostic agents may suggest it
    during analysis, but they do not need to write it directly.
    
    Args:
        agent: The agent instance
        session: The session object
        trace_processor: The trace processor for logging
        max_turns: Maximum turns for the agent (default 10, use 3 for interrupted sessions)
    """
    prompt = (
        "The user is ending the session. Respond with a brief goodbye."
    )
    trace_processor.write_user_input(prompt)
    try:
        result = Runner.run_sync(agent, prompt, session=session, max_turns=max_turns)
        trace_processor.write_agent_output(result.final_output)
        if result.final_output:
            print(f"\nAgent > {result.final_output}")
    except KeyboardInterrupt:
        pass
    except Exception:
        pass


def _ensure_agent_data_path_for_memory():
    """Set agent data path to CWD so memory/preferences can be read/written."""
    set_agent_data_path()


def _get_instruction_text(args, stdin_prompt: str):
    """Get instruction text from args.file, args.text, or stdin. Returns (text, error_msg)."""
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            return None, f"Error: File not found: {args.file}"
        return read_instructions_from_file(file_path).strip() or None, None
    if args.text:
        return args.text.strip() or None, None
    print(stdin_prompt)
    text = read_instructions_from_stdin()
    return (text or None), None


def handle_instructions_command(args):
    """Handle instructions subcommands."""
    _ensure_agent_data_path_for_memory()
    action = args.instructions_action

    if action == "get":
        instructions = get_instructions()
        if instructions:
            print(instructions)
        else:
            print("No custom instructions set.")
        return

    if action == "clear":
        try:
            set_instructions("")
            print("Custom instructions cleared")
        except Exception as e:
            print(f"Error: {e}")
        return

    if action == "set":
        text, err = _get_instruction_text(
            args, "Enter instructions (max 100 words). Press Ctrl+D when done:"
        )
    else:  # append
        text, err = _get_instruction_text(
            args, "Enter instructions to append (max 100 words). Press Ctrl+D when done:"
        )

    if err:
        print(err)
        return
    if not text:
        print("Error: No instructions provided")
        return

    try:
        if action == "set":
            set_instructions(text)
            print("Instructions updated")
        else:
            append_instructions(text)
            print("Instructions appended")
    except ValueError as e:
        print(f"Error: {e}")


def handle_preferences_command(args):
    """Handle preferences subcommands."""
    if args.preferences_action != "clear":
        return
    try:
        _ensure_agent_data_path_for_memory()
        clear_user_preferences()
        print("User preferences cleared")
    except Exception as e:
        print(f"Error: {e}")


def _parse_args(argv=None):
    """Parse CLI arguments. If argv is None, uses sys.argv[1:]."""
    parser = argparse.ArgumentParser(
        prog="adams",
        description="ADAMS - Agent-Driven Autonomous Molecular Simulations",
    )
    
    parser.add_argument(
        "--continue-session",
        dest="continue_session",
        metavar="SESSION_ID",
        help="Continue a previous session by ID (format: YYYYMMDD_HHMMSS)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Instructions command
    instructions_parser = subparsers.add_parser(
        "instructions",
        help="Manage custom instructions (get, set, append, clear). Use 'adams instructions -h' for details.",
    )
    instructions_subparsers = instructions_parser.add_subparsers(
        dest="instructions_action", help="Action", required=True
    )
    
    # Get instructions
    instructions_subparsers.add_parser("get", help="Read current instructions")
    
    # Set instructions
    set_parser = instructions_subparsers.add_parser(
        "set",
        help="Set instructions (replaces existing). Use text, --file PATH, or stdin.",
    )
    set_parser.add_argument("text", nargs="?", help="Instructions text")
    set_parser.add_argument("--file", "-f", help="Read instructions from file")
    
    # Append instructions
    append_parser = instructions_subparsers.add_parser(
        "append",
        help="Append to existing instructions. Use text, --file PATH, or stdin.",
    )
    append_parser.add_argument("text", nargs="?", help="Instructions text to append")
    append_parser.add_argument("--file", "-f", help="Read instructions from file")
    
    # Clear instructions
    instructions_subparsers.add_parser("clear", help="Clear custom instructions")
    
    # Preferences command
    preferences_parser = subparsers.add_parser(
        "preferences",
        help="Manage user preferences (clear). Use 'adams preferences -h' for details.",
    )
    preferences_subparsers = preferences_parser.add_subparsers(
        dest="preferences_action", help="Action", required=True
    )
    
    # Clear preferences
    preferences_subparsers.add_parser("clear", help="Clear all user preferences")

    add_dock_subparser(subparsers)
    
    args = parser.parse_args(argv) if argv is not None else parser.parse_args()
    
    # Default to session if no command
    if args.command is None:
        args.command = "session"
    
    return args


def main():
    """Main CLI entry point."""
    args = _parse_args()
    
    if args.command == "instructions":
        handle_instructions_command(args)
        return
    
    if args.command == "preferences":
        handle_preferences_command(args)
        return

    if args.command == "dock":
        run_dock_command(args)
        return
    
    # Default: interactive session
    _run_interactive_session(args)


def _run_interactive_session(args):
    """Run the interactive agent session."""
    working_dir = Path.cwd()
    agent_data_path = working_dir / "agent_data"
    set_agent_data_path(path=agent_data_path)
    transcript_path = start_console_transcript()
    print(f"[Transcript] {transcript_path}")

    print(
        r"""
    ╔════════════════════════════════════════════════════════════════════╗
    ║                                                                    ║
    ║        _    ____    _    __  __ ____                               ║
    ║       / \  |  _ \  / \  |  \/  / ___|                              ║
    ║      / _ \ | | | |/ _ \ | |\/| \___ \                              ║
    ║     / ___ \| |_| / ___ \| |  | |___) |                             ║
    ║    /_/   \_\____/_/   \_\_|  |_|____/                              ║
    ║                                                                    ║
    ║    Agent-Driven Autonomous Molecular Simulations                   ║
    ║                                                                    ║
    ║    An agentic workflow that automates molecular docking and MD     ║
    ║    simulation based on user-provided prompts.                      ║
    ║    Protein preprocessing, binding pocket discovery, docking        ║
    ║    and stability analysis.                                         ║
    ║                                                                    ║
    ║                                                                    ║
    ║                                        by Rhizome Research         ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝


    Press Ctrl+C to cancel agent calls, Ctrl+D to exit.
    Type 'exit' or 'quit' to end the session.


    """
    )

    # Get OpenAI API key
    api_key = get_api_key()
    if api_key is None:
        return

    print(f"Using working directory: {working_dir}")
    print(f"Data will be stored in: {agent_data_path}\n")

    session_id = args.continue_session if args.continue_session else None

    trace_processor = setup_tracing(session_id=session_id)

    actual_session_id = trace_processor.session_id

    if args.continue_session:
        print(f"[Memory] Continuing session: {actual_session_id}")
    session = create_sdk_session(actual_session_id, is_continue=bool(args.continue_session))

    # Create agent
    agent = create_agent(session_id=actual_session_id)

    # Create key bindings and history
    key_bindings = create_key_bindings()
    history = InMemoryHistory()

    # Setup graceful shutdown manager
    shutdown_manager = ShutdownManager()
    
    def cleanup_on_shutdown():
        """Cleanup callback for graceful shutdown on Ctrl+C."""
        try:
            print("\n[Saving session metadata...]", flush=True)
            run_session_metadata_update(
                agent, session, trace_processor,
                max_turns=3,
            )
        except Exception:
            try:
                add_session_tags(actual_session_id, ["interrupted"])
            except Exception:
                pass
        try:
            trace_processor.shutdown()
        except Exception:
            pass
        try:
            format_session_trace()
        except Exception:
            pass
    
    shutdown_manager.register_cleanup(cleanup_on_shutdown)
    shutdown_manager.setup_handlers()

    exit_normally = False  # True only for exit/quit or EOF (not Ctrl+C)
    try:
        while True:
            try:
                # Get user input with framed input box
                user_input = get_framed_input(history, key_bindings).strip()

                # Handle exit commands
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting...")
                    exit_normally = True
                    break

                # Skip empty input
                if not user_input:
                    continue

                # Write user input to trace file
                trace_processor.write_user_input(user_input)

                print("\nProcessing your request...")

                try:
                    result = Runner.run_sync(
                        agent, user_input, session=session, max_turns=50
                    )
                    # Display agent response
                    print(f"\nAgent > {result.final_output}")
                    # Log agent output to trace file
                    trace_processor.write_agent_output(result.final_output)
                except KeyboardInterrupt:
                    # Ctrl+C during agent execution - exit without touching session DB
                    print("\n[Session cancelled. Exiting...]")
                    break
                except Exception as e:
                    # Handle potential wrapped KeyboardInterrupt
                    if "KeyboardInterrupt" in str(e) or isinstance(
                        e.__cause__, KeyboardInterrupt
                    ):
                        print("\n[Session cancelled. Exiting...]")
                        break
                    else:
                        print(f"\n[Agent error: {e}]")
                        continue

            except KeyboardInterrupt:
                # Ctrl+C at input prompt - cancel current input and show new prompt
                print("\n[Cancelled. Type 'exit' or 'quit' to exit.]")
                continue
            except EOFError:
                print("\nExiting...")
                exit_normally = True
                break
            except Exception as e:
                print(f"\nError: {e}")
                # Don't crash the session on unexpected errors, just log and continue
                continue
    finally:
        # Restore original signal handlers
        shutdown_manager.restore_handlers()
        
        if exit_normally:
            try:
                run_session_metadata_update(agent, session, trace_processor)
            except (OSError, KeyboardInterrupt):
                pass

        # Write session_end marker (idempotent — safe if already called by shutdown handler)
        try:
            trace_processor.shutdown()
        except Exception:
            pass
        
        # Format trace file if not already done by shutdown manager
        try:
            format_session_trace()
        except OSError:
            pass


if __name__ == "__main__":
    main()
