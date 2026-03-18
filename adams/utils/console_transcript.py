"""Console transcript capture for ADAMS runs.

This module mirrors the QA runner behavior inside Python so CLI sessions and
custom runners can automatically persist the full terminal stream.
"""

from __future__ import annotations

import atexit
import os
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Optional, TextIO

from ..path_config import get_subdirectory


class _FdTeeThread(threading.Thread):
    """Copy bytes from a pipe to both the original stream fd and a log file."""

    def __init__(self, read_fd: int, target_fd: int, log_file: BinaryIO, lock: threading.Lock):
        super().__init__(daemon=True)
        self._read_fd = read_fd
        self._target_fd = target_fd
        self._log_file = log_file
        self._lock = lock

    def run(self) -> None:
        try:
            while True:
                chunk = os.read(self._read_fd, 8192)
                if not chunk:
                    break
                os.write(self._target_fd, chunk)
                with self._lock:
                    self._log_file.write(chunk)
                    self._log_file.flush()
        except OSError:
            # Stream teardown during process shutdown should not crash the run.
            pass
        finally:
            try:
                os.close(self._read_fd)
            except OSError:
                pass


class ConsoleTranscript:
    """Capture stdout/stderr to a transcript file while preserving terminal output."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._log_file: Optional[BinaryIO] = None
        self._stdout_dup: Optional[int] = None
        self._stderr_dup: Optional[int] = None
        self._stdout_thread: Optional[_FdTeeThread] = None
        self._stderr_thread: Optional[_FdTeeThread] = None
        self._saved_stdout: Optional[TextIO] = None
        self._saved_stderr: Optional[TextIO] = None
        self._active = False
        self._lock = threading.Lock()

    def start(self) -> Path:
        if self._active:
            return self.path

        self.path.parent.mkdir(parents=True, exist_ok=True)
        sys.stdout.flush()
        sys.stderr.flush()

        self._log_file = self.path.open('ab', buffering=0)
        self._stdout_dup = os.dup(1)
        self._stderr_dup = os.dup(2)
        self._saved_stdout = sys.stdout
        self._saved_stderr = sys.stderr

        stdout_read, stdout_write = os.pipe()
        stderr_read, stderr_write = os.pipe()

        os.dup2(stdout_write, 1)
        os.dup2(stderr_write, 2)
        os.close(stdout_write)
        os.close(stderr_write)

        stdout_encoding = getattr(self._saved_stdout, 'encoding', None) or 'utf-8'
        stdout_errors = getattr(self._saved_stdout, 'errors', None) or 'replace'
        stderr_encoding = getattr(self._saved_stderr, 'encoding', None) or 'utf-8'
        stderr_errors = getattr(self._saved_stderr, 'errors', None) or 'replace'
        sys.stdout = open(1, 'w', buffering=1, encoding=stdout_encoding, errors=stdout_errors, closefd=False)
        sys.stderr = open(2, 'w', buffering=1, encoding=stderr_encoding, errors=stderr_errors, closefd=False)

        self._stdout_thread = _FdTeeThread(stdout_read, self._stdout_dup, self._log_file, self._lock)
        self._stderr_thread = _FdTeeThread(stderr_read, self._stderr_dup, self._log_file, self._lock)
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._active = True
        return self.path

    def stop(self) -> None:
        if not self._active:
            return

        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except Exception:
            pass

        try:
            if sys.stdout is not self._saved_stdout:
                sys.stdout.close()
            if sys.stderr is not self._saved_stderr:
                sys.stderr.close()
        except Exception:
            pass

        if self._stdout_dup is not None:
            os.dup2(self._stdout_dup, 1)
        if self._stderr_dup is not None:
            os.dup2(self._stderr_dup, 2)

        if self._saved_stdout is not None:
            sys.stdout = self._saved_stdout
        if self._saved_stderr is not None:
            sys.stderr = self._saved_stderr

        for thread in (self._stdout_thread, self._stderr_thread):
            if thread is not None:
                thread.join(timeout=1.0)

        for fd in (self._stdout_dup, self._stderr_dup):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

        if self._log_file is not None:
            self._log_file.close()

        self._stdout_dup = None
        self._stderr_dup = None
        self._stdout_thread = None
        self._stderr_thread = None
        self._saved_stdout = None
        self._saved_stderr = None
        self._log_file = None
        self._active = False


_active_transcript: Optional[ConsoleTranscript] = None


def default_transcript_path(prefix: str = 'console_transcript') -> Path:
    """Return the default transcript location under agent_data/logs."""
    logs_dir = get_subdirectory('logs')
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return logs_dir / f'{prefix}_{timestamp}.log'


def start_console_transcript(path: str | Path | None = None) -> Path:
    """Start capturing the full terminal stream to a transcript file."""
    global _active_transcript
    if _active_transcript is not None:
        return _active_transcript.path

    transcript = ConsoleTranscript(Path(path) if path is not None else default_transcript_path())
    transcript.start()
    _active_transcript = transcript
    atexit.register(stop_console_transcript)
    return transcript.path


def get_active_transcript_path() -> Path | None:
    """Return the active transcript path when capture is already enabled."""
    if _active_transcript is None:
        return None
    return _active_transcript.path


def stop_console_transcript() -> None:
    """Stop transcript capture if it is active."""
    global _active_transcript
    if _active_transcript is None:
        return
    _active_transcript.stop()
    _active_transcript = None
