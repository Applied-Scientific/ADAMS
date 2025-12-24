"""
    Interactive CLI entry point for the Biophysics Controller Agent.
    Provides enhanced keyboard navigation and command history.
"""

import logging
import os
import shutil
import textwrap
from getpass import getpass
from pathlib import Path
import stat

from agents import Runner, SQLiteSession
from prompt_toolkit import prompt
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings, merge_key_bindings

from .executive_agent import create_agent, setup_tracing
from .format_trace import format_trace_file
from .path_config import set_agent_data_path

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


def get_api_key():
    """
    Retrieves the OpenAI API key.

    The function first checks for the `OPENAI_API_KEY` environment variable.
    If not found, it tries to load it from a configuration file located at
    `~/.adams`. If the key is not in the environment or the config
    file, it prompts the user for the key and asks for permission to save it
    for future use.

    Returns:
        str: The OpenAI API key, or None if user cancelled.
    """
    config_file = Path.home() / ".adams"

    # Check environment variable first
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key

    # If not in env, check config file
    if config_file.exists():
        try:
            api_key = config_file.read_text().strip()
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                return api_key
        except (OSError, IOError) as e:
            print(f"Warning: Could not read API key from {config_file}: {e}")

    # If not found, prompt user
    print("OpenAI API key not found.")
    try:
        api_key = getpass("Please enter your OpenAI API key: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting...")
        return None

    # Validate that key is not empty
    if not api_key:
        print("Error: API key cannot be empty.")
        return None

    os.environ["OPENAI_API_KEY"] = api_key

    try:
        print(
            "\nSecurity note: If you choose to store this key, it will be written in PLAINTEXT to "
            f"{config_file}. Only do this on a trusted machine/user account."
        )
        save_key = input(
            "Do you want to store this key for future use in ~/.adams? (y/n): "
        )
    except (EOFError, KeyboardInterrupt):
        print("\nKey not stored.")
        return api_key

    if save_key.lower() == "y":
        try:
            config_file.write_text(api_key)
            # Best-effort permission hardening (user-read/write only)
            try:
                os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
            except OSError:
                # Non-fatal; filesystem may not support chmod semantics
                pass
            print(f"Key stored in {config_file}")
        except (OSError, IOError) as e:
            print(f"Warning: Could not save API key to {config_file}: {e}")
            print("Key not stored. It will be required for the next session.")
    else:
        print("Key not stored. It will be required for the next session.")

    return api_key


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


def main():
    """Main interactive CLI loop."""

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

    # Use current working directory
    working_dir = Path.cwd()
    agent_data_path = working_dir / "agent_data"
    print(f"Using working directory: {working_dir}")
    print(f"Data will be stored in: {agent_data_path}\n")
    set_agent_data_path(path=agent_data_path)

    # Set up tracing (now uses configured path)
    trace_processor = setup_tracing()

    # Create agent
    agent = create_agent()

    # Set up session
    user = "User"
    session = SQLiteSession(user)

    # Create key bindings and history
    key_bindings = create_key_bindings()
    history = InMemoryHistory()

    while True:
        try:
            # Get user input with framed input box
            user_input = get_framed_input(history, key_bindings).strip()

            # Handle exit commands
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting...")
                format_session_trace()
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
                # Ctrl+C during agent execution - exit session
                print("\n[Session cancelled. Exiting...]")
                format_session_trace()
                break
            except Exception as e:
                # Handle potential wrapped KeyboardInterrupt
                if "KeyboardInterrupt" in str(e) or isinstance(
                    e.__cause__, KeyboardInterrupt
                ):
                    print("\n[Session cancelled. Exiting...]")
                    format_session_trace()
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
            format_session_trace()
            break
        except Exception as e:
            print(f"\nError: {e}")
            # Don't crash the session on unexpected errors, just log and continue
            continue


if __name__ == "__main__":
    main()
