import sys

from adams.path_config import reset_paths, set_agent_data_path
from adams.utils.console_transcript import (
    get_active_transcript_path,
    start_console_transcript,
    stop_console_transcript,
)


def test_console_transcript_captures_stdout_and_stderr(tmp_path):
    reset_paths()
    agent_data = tmp_path / 'agent_data'
    set_agent_data_path(path=agent_data)

    transcript = start_console_transcript()
    try:
        print('hello transcript', flush=True)
        print('error transcript', file=sys.stderr, flush=True)
    finally:
        stop_console_transcript()
        reset_paths()

    text = transcript.read_text(encoding='utf-8', errors='replace')
    assert 'hello transcript' in text
    assert 'error transcript' in text
    assert transcript.parent == agent_data / 'logs'


def test_console_transcript_custom_path(tmp_path):
    reset_paths()
    agent_data = tmp_path / 'agent_data'
    set_agent_data_path(path=agent_data)
    custom = tmp_path / 'custom_cli.log'

    transcript = start_console_transcript(custom)
    try:
        print('custom transcript path', flush=True)
    finally:
        stop_console_transcript()
        reset_paths()

    assert transcript == custom
    assert 'custom transcript path' in custom.read_text(encoding='utf-8', errors='replace')


def test_console_transcript_reports_active_path(tmp_path):
    reset_paths()
    agent_data = tmp_path / 'agent_data'
    set_agent_data_path(path=agent_data)
    custom = tmp_path / 'active_cli.log'

    assert get_active_transcript_path() is None
    transcript = start_console_transcript(custom)
    try:
        assert get_active_transcript_path() == transcript
    finally:
        stop_console_transcript()
        reset_paths()

    assert get_active_transcript_path() is None
