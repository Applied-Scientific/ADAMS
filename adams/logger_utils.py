"""
Logger utility for the adams pipeline.

This module provides a centralized logging system that writes to both
a log file and the console. All classes and functions automatically use
get_logger() internally, so you don't need to pass logger instances around.

Usage:
    # At entry point (e.g., executive_agent.py) - optional, creates default if not called:
    from adams.logger_utils import setup_logger
    logger = setup_logger(log_file="pipeline.log")

    # Or set a custom logger if needed:
    from adams.logger_utils import set_logger
    custom_logger = logging.getLogger('my_custom_logger')
    set_logger(custom_logger)

    # In modules/classes - logger is automatically retrieved:
    # No need to pass logger parameters - classes use get_logger() internally
    from adams.logger_utils import get_logger
    logger = get_logger()  # Gets the same logger instance
"""

import errno
import io
import logging
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Module-level logger instance
_pipeline_logger = None


class LoggerWriter(io.TextIOBase):
    """
    A file-like object that writes to a logger.
    This allows us to pass a logger directly as stdout/stderr to subprocess.

    Usage:
        logger = get_logger()
        stdout_writer = LoggerWriter(logger, 'info')
        stderr_writer = LoggerWriter(logger, 'error')

        process = subprocess.Popen(
            cmd,
            stdout=stdout_writer,
            stderr=stderr_writer
        )
    """

    def __init__(self, logger, level="info"):
        """
        Initialize LoggerWriter.

        Args:
            logger: Logger instance to write to
            level: Log level ('info', 'error', 'warning', 'debug')
        """
        self.logger = logger
        self.level = level.lower()
        self.buffer = ""

    def write(self, s):
        """
        Write string to logger. Buffers until newline, then logs the line.

        Args:
            s: String to write

        Returns:
            int: Number of characters written
        """
        if not s:
            return 0

        # Add to buffer
        self.buffer += s

        # Process complete lines
        lines = self.buffer.split("\n")
        # Keep the last incomplete line in buffer
        self.buffer = lines[-1]

        # Log all complete lines
        log_func = getattr(self.logger, self.level, self.logger.info)
        for line in lines[:-1]:
            line = line.rstrip("\r")
            if line:  # Only log non-empty lines
                log_func(line)

        return len(s)

    def flush(self):
        """
        Flush any remaining buffer content.
        """
        if self.buffer:
            log_func = getattr(self.logger, self.level, self.logger.info)
            line = self.buffer.rstrip("\r\n")
            if line:
                log_func(line)
            self.buffer = ""

    def fileno(self):
        """
        Return the file descriptor. Not supported for LoggerWriter.

        Raises:
            OSError: Always raises, as LoggerWriter doesn't have a file descriptor
        """
        raise OSError(errno.ENOTSUP, "LoggerWriter does not support file descriptors")

    def close(self):
        """
        Close the stream and flush any remaining content.
        """
        self.flush()
        super().close()


def setup_logger(log_file: str = None, log_level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger that writes to both a file and the console.

    Args:
        log_file: Path to the log file. If None, creates a default log file
                  in agent_data/logs/ with timestamp.
        log_level: Logging level (default: logging.INFO)

    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("adams_pipeline")
    logger.setLevel(log_level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    if log_file is None:
        # Use centralized path configuration
        from .path_config import get_subdirectory

        agent_data_logs = get_subdirectory("logs")
        agent_data_logs.mkdir(parents=True, exist_ok=True)

        # Create default log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = str(agent_data_logs / f"adams_pipeline_{timestamp}.log")

    # Ensure log directory exists
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Store as module-level logger
    global _pipeline_logger
    _pipeline_logger = logger

    return logger


def set_logger(logger: logging.Logger) -> None:
    """
    Set a custom logger instance to be used throughout the pipeline.

    This allows users to provide their own configured logger if they want
    custom formatting, handlers, or other behavior.

    Args:
        logger: A configured logging.Logger instance to use throughout the pipeline
    """
    global _pipeline_logger
    _pipeline_logger = logger


def get_logger() -> logging.Logger:
    """
    Get the existing logger instance if it exists, otherwise create a new one.

    Returns:
        logging.Logger: Logger instance
    """
    global _pipeline_logger
    if _pipeline_logger is not None:
        return _pipeline_logger

    # Fallback: try to get from logging system
    logger = logging.getLogger("adams_pipeline")
    if logger.handlers:
        _pipeline_logger = logger
        return logger

    # If no logger exists, create a default one
    _pipeline_logger = setup_logger()
    return _pipeline_logger


def setup_multiprocessing_logging():
    """
    Set up queue-based logging for multiprocessing workers.

    Call this once in the main process before spawning any workers.
    This enables safe logging from worker processes using the spawn
    start method (avoiding fork deadlocks).

    Returns:
        multiprocessing.Queue: Log queue to pass to workers

    Example:
        from adams.logger_utils import get_logger, setup_multiprocessing_logging

        # In main process
        logger = get_logger()
        log_queue = setup_multiprocessing_logging()

        # Pass log_queue to worker processes
        proc = Process(target=worker, args=(data, log_queue))
    """
    from .utils.multiprocessing_utils import setup_worker_logging

    logger = get_logger()
    return setup_worker_logging(logger)


def get_log_queue():
    """
    Get the multiprocessing log queue if worker logging is set up.

    Returns None if not set up yet. This is primarily for internal use;
    prefer using setup_multiprocessing_logging() at initialization.

    Returns:
        multiprocessing.Queue or None: Log queue if set up, None otherwise
    """
    from .utils.multiprocessing_utils import _log_queue

    return _log_queue


class StepExecutionLogger:
    """
    Context manager for logging step execution with timing, error handling, and conclusion.

    This utility handles all the verbose logging patterns for pipeline steps, allowing
    run() methods to remain clean and focused on their core logic.

    Basic Usage:
        from adams.logger_utils import log_step_execution

        def run(self):
            with log_step_execution("Step Name"):
                # Do work here - automatically logged
                self._do_something()
                return result

    With Sub-step Timing:
        def run(self):
            step_logger = log_step_execution("Step Name")
            with step_logger:
                # Main work
                self._preprocess()

                # Track sub-step timing
                with step_logger.timing("data_validation"):
                    self._validate()

                return result

    Features:
    - Automatically logs "=== Step Name Started! ==="
    - Tracks total execution time
    - Allows sub-step timing tracking via .timing() context manager
    - Logs conclusion with timing breakdown
    - Handles errors with full stack traces and context
    - Logs "=== Step Name Completed Successfully! ==="
    """

    def __init__(self, step_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize step execution logger.

        Args:
            step_name: Name of the step (will be logged as "=== {step_name} Started! ===")
            logger: Logger instance to use (defaults to get_logger())
        """
        self.step_name = step_name
        self.logger = logger or get_logger()
        self.start_time = None
        self.timing_breakdown: Dict[str, float] = {}
        self._active_timing_contexts: Dict[str, float] = {}

    def __enter__(self):
        """Enter context - log step start and record start time."""
        self.logger.info(f"=== {self.step_name} Started! ===")
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - log conclusion or error, then log completion."""
        total_duration = (
            round(time.perf_counter() - self.start_time, 2) if self.start_time else 0.0
        )

        if exc_type is None:
            # Success case - log conclusion
            self.logger.info("=== Conclusion ===")
            self.logger.info(f"Total execution time: {total_duration:.2f} (s)")

            if self.timing_breakdown:
                self.logger.info("Timing breakdown:")
                for key, value in self.timing_breakdown.items():
                    self.logger.info(f"  - {key}: {value:.2f} (s)")

            self.logger.info("Status: Success")
            self.logger.info(f"=== {self.step_name} Completed Successfully! ===")
        else:
            # Error case - log error with stack trace
            self.logger.error(f"=== Error in {self.step_name} ===")
            self.logger.error(f"Error message: {str(exc_val)}")
            self.logger.error(f"Error type: {exc_type.__name__}")
            self.logger.error(
                f"Context: Step failed after {total_duration:.2f} seconds"
            )
            self.logger.error("Stack trace:")
            for line in traceback.format_exception(exc_type, exc_val, exc_tb):
                for trace_line in line.rstrip().split("\n"):
                    if trace_line.strip():
                        self.logger.error(f"  {trace_line}")

        # Don't suppress the exception - let it propagate
        return False

    @contextmanager
    def timing(self, timing_name: str):
        """
        Context manager for tracking sub-step timing.

        Usage:
            step_logger = log_step_execution("Main Step")
            with step_logger:
                with step_logger.timing("substep1"):
                    # This timing will be recorded
                    do_something()

        Args:
            timing_name: Name for this timing measurement
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = round(time.perf_counter() - start, 2)
            self.timing_breakdown[timing_name] = duration

    def add_timing(self, timing_name: str, duration_seconds: float):
        """
        Manually add a timing measurement.

        Use this when you need to track timing that doesn't fit a context manager pattern.

        Args:
            timing_name: Name for this timing measurement
            duration_seconds: Duration in seconds
        """
        self.timing_breakdown[timing_name] = round(duration_seconds, 2)


def log_step_execution(
    step_name: str, logger: Optional[logging.Logger] = None
) -> StepExecutionLogger:
    """
    Create a context manager for logging step execution.

    This is the main entry point for step logging. It handles:
    - Step start/completion markers
    - Total execution timing
    - Sub-step timing tracking
    - Conclusion logging with timing breakdown
    - Error logging with stack traces

    Usage:
        from adams.logger_utils import log_step_execution

        def run(self):
            step_logger = log_step_execution("Data Preprocessing")
            with step_logger:
                # Main work
                self._preprocess()

                # Track sub-step timing
                with step_logger.timing("data_validation"):
                    self._validate()

                return result

    Args:
        step_name: Name of the step to log
        logger: Optional logger instance (defaults to get_logger())

    Returns:
        StepExecutionLogger context manager instance

    Example:
        >>> with log_step_execution("Docking Inference"):
        ...     result = dock_ligands()
        ...     # Automatically logs:
        ...     # "=== Docking Inference Started! ==="
        ...     # "=== Conclusion ==="
        ...     # "Total execution time: X.XX (s)"
        ...     # "=== Docking Inference Completed Successfully! ==="
    """
    return StepExecutionLogger(step_name, logger)
