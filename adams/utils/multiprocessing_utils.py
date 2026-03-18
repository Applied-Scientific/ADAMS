"""
Multiprocessing utilities with spawn context for safe parallel execution.

This module provides:
- Spawn-based multiprocessing (safer than fork, will be default in Python 3.14+)
- Queue-based logging infrastructure for workers
- Centralized multiprocessing configuration

Benefits of Spawn vs Fork:
- No deadlocks with threads, locks, or file handles
- Safe for use with logging, database connections, etc.
- Cross-platform consistency (same behavior on Linux/macOS/Windows)
- Future-proof for Python 3.14+ where spawn becomes default

Usage:
    # In main process:
    from adams.utils.multiprocessing_utils import Process, setup_worker_logging
    from adams.logger_utils import get_logger

    logger = get_logger()
    log_queue = setup_worker_logging(logger)

    # Spawn workers
    proc = Process(target=worker_func, args=(data, log_queue))
    proc.start()
    proc.join()

    # In worker process:
    from adams.utils.multiprocessing_utils import configure_worker_logging
    from adams.logger_utils import get_logger

    def worker_func(data, log_queue):
        configure_worker_logging(log_queue)
        logger = get_logger()
        logger.info("Worker started")  # Logs to main process queue
        # ... do work ...
"""

import atexit
import gc
import logging
import logging.handlers
import multiprocessing as mp
import os
import threading

# Global spawn context - safer than fork, will be default in Python 3.14+
# Spawn starts a fresh Python interpreter, avoiding fork hazards like:
# - Inherited file locks (causes logger deadlocks)
# - Inherited thread state (threads don't survive fork)
# - Database connections (invalid after fork)
_spawn_ctx = mp.get_context("spawn")

# Global logging infrastructure for queue-based worker logging
_log_queue = None
_queue_listener = None


def get_spawn_context():
    """
    Get multiprocessing context configured for spawn start method.

    Spawn is safer than fork (no deadlocks with threads/locks) and will
    be the default in Python 3.14+.

    Returns:
        multiprocessing.context.SpawnContext: Configured spawn context

    Example:
        ctx = get_spawn_context()
        proc = ctx.Process(target=worker_func)
        pool = ctx.Pool(processes=4)
    """
    return _spawn_ctx


# Convenience exports - use these instead of importing from multiprocessing directly
Process = _spawn_ctx.Process
Pool = _spawn_ctx.Pool
Queue = _spawn_ctx.Queue
cpu_count = mp.cpu_count


def is_spawn_error(exc: Exception) -> bool:
    """Return True when *exc* is the ``bad value(s) in fds_to_keep`` spawn error."""
    return isinstance(exc, (ValueError, OSError)) and "fds_to_keep" in str(exc)


def can_start_multiprocessing() -> bool:
    """Return False only when multiprocessing is explicitly disabled.

    Historically we also blocked non-main-thread callers to avoid spawn FD
    races ("bad value(s) in fds_to_keep"). The spawn safety patch below now
    serializes Process.start() and sanitizes passfds, so thread affinity is no
    longer used as a hard gate for multiprocessing.
    """
    if os.environ.get("ADAMS_DISABLE_MULTIPROCESS", "").lower() in {
        "1",
        "true",
        "yes",
    }:
        return False
    return True


# ---------------------------------------------------------------------------
# Root-cause fix for "ValueError: bad value(s) in fds_to_keep"
#
# When process args contain a Queue, pickling registers the Queue's pipe FDs
# via duplicate_for_child().  If a TUI thread or GC finalizer closed one of
# those FDs *before* spawn, os.pipe() inside _launch can reuse the same FD
# number, creating a duplicate in the fds_to_keep list.  fork_exec requires
# strictly ascending FDs and rejects duplicates.
#
# Two-layer fix:
#   1. Suspend GC + serialize spawns during Process.start() — prevents
#      finalizers from closing FDs during the spawn window.
#   2. Sanitize the FD list in spawnv_passfds right before fork_exec —
#      filters out negative, closed, and duplicate FDs that were already
#      stale before spawn was called.
# ---------------------------------------------------------------------------

_spawn_lock = threading.Lock()
_original_process_start = _spawn_ctx.Process.start


def _safe_process_start(self):
    with _spawn_lock:
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            _original_process_start(self)
        finally:
            if gc_was_enabled:
                gc.enable()


_spawn_ctx.Process.start = _safe_process_start

import multiprocessing.util as _mp_util

_original_spawnv_passfds = _mp_util.spawnv_passfds


def _safe_spawnv_passfds(path, args, passfds):
    seen = set()
    valid = []
    for fd in sorted(int(fd) for fd in passfds):
        if fd < 0 or fd in seen:
            continue
        try:
            os.fstat(fd)
        except OSError:
            continue
        seen.add(fd)
        valid.append(fd)
    return _original_spawnv_passfds(path, args, valid)


_mp_util.spawnv_passfds = _safe_spawnv_passfds


def setup_worker_logging(main_logger: logging.Logger) -> mp.Queue:
    """
    Set up queue-based logging for multiprocessing workers.

    This should be called once in the main process before spawning any workers.
    The returned queue should be passed to worker processes so they can send
    log records back to the main process.

    Architecture:
        Main Process:
            - QueueListener monitors the queue
            - Receives log records from workers
            - Writes to file/console via main logger handlers

        Worker Processes:
            - QueueHandler sends log records to queue
            - No file I/O or lock contention
            - Structured logging with timestamps

    Args:
        main_logger: The main logger instance (from get_logger())

    Returns:
        multiprocessing.Queue: Log queue to pass to workers

    Example:
        from adams.logger_utils import get_logger
        from adams.utils.multiprocessing_utils import setup_worker_logging

        logger = get_logger()
        log_queue = setup_worker_logging(logger)

        # Pass log_queue to worker processes
        proc = Process(target=worker, args=(data, log_queue))
    """
    global _log_queue, _queue_listener

    if _log_queue is not None:
        # Already set up - return existing queue
        return _log_queue

    # Create queue for worker logs (unbounded queue with -1)
    _log_queue = _spawn_ctx.Queue(-1)

    # Create listener that drains queue and logs to main logger
    # respect_handler_level=True ensures worker log levels are respected
    _queue_listener = logging.handlers.QueueListener(
        _log_queue, *main_logger.handlers, respect_handler_level=True
    )

    # Start the listener thread
    _queue_listener.start()

    # Ensure listener stops cleanly on program exit
    atexit.register(stop_worker_logging)

    return _log_queue


def stop_worker_logging():
    """
    Stop the queue listener and clean up logging infrastructure.

    This is called automatically via atexit, but can be called manually
    if you need to shut down logging before program exit.
    """
    global _queue_listener, _log_queue

    if _queue_listener is not None:
        _queue_listener.stop()
        _queue_listener = None

    _log_queue = None


def cleanup_process(process, timeout: float = 2.0) -> None:
    """
    Safely cleanup a Process object to prevent resource leaks.

    This function:
    1. Terminates the process if still running
    2. Waits for graceful shutdown with timeout
    3. Force kills if necessary
    4. Explicitly closes to cleanup semaphores

    Args:
        process: multiprocessing.Process object to cleanup
        timeout: Seconds to wait for graceful shutdown (default: 2.0)

    Example:
        proc = Process(target=worker_func)
        proc.start()
        try:
            proc.join()
        except KeyboardInterrupt:
            cleanup_process(proc)
    """
    if process is None:
        return

    try:
        # Terminate if still alive
        if process.is_alive():
            process.terminate()
            process.join(timeout=timeout)

            # Force kill if still alive after timeout
            if process.is_alive():
                process.kill()
                process.join()
        else:
            # Join to cleanup zombie processes
            process.join()
    except Exception:
        pass  # Ignore errors during termination

    try:
        # Explicit close to cleanup semaphores and resources
        process.close()
    except Exception:
        pass  # Ignore errors during close


def cleanup_pool(pool, terminate: bool = True, timeout: float = 2.0) -> None:
    """
    Safely cleanup a Pool object to prevent resource leaks.

    This function:
    1. Closes the pool to prevent new tasks
    2. Optionally terminates running workers
    3. Waits for cleanup with timeout

    Args:
        pool: multiprocessing.Pool object to cleanup
        terminate: Whether to terminate workers (default: True)
        timeout: Seconds to wait for cleanup (default: 2.0)

    Example:
        pool = Pool(processes=4)
        try:
            pool.map(worker_func, data)
        except KeyboardInterrupt:
            cleanup_pool(pool, terminate=True)
    """
    if pool is None:
        return

    try:
        # Close to prevent new tasks
        pool.close()
    except Exception:
        pass  # Already closed

    if terminate:
        try:
            # Terminate all workers
            pool.terminate()
        except Exception:
            pass  # Ignore errors during termination

    try:
        # Wait for cleanup
        pool.join()
    except Exception:
        pass  # Ignore errors during join


def configure_worker_logging(log_queue: mp.Queue) -> None:
    """
    Configure logging for a worker process to send logs to the queue.

    This should be called once at the start of each worker process.
    After calling this, workers can use get_logger() normally and logs
    will automatically be sent to the main process via the queue.

    Args:
        log_queue: Queue from setup_worker_logging()

    Example:
        def worker_func(data, log_queue):
            from adams.utils.multiprocessing_utils import configure_worker_logging
            from adams.logger_utils import get_logger

            configure_worker_logging(log_queue)
            logger = get_logger()  # Just works!

            logger.info("Worker started")  # Appears in main log
            logger.debug("Processing data")
    """
    # Get the main pipeline logger
    logger = logging.getLogger("adams_pipeline")
    logger.setLevel(logging.DEBUG)  # Capture all levels, main logger filters

    # Clear any existing handlers (inherited from parent, won't work after spawn)
    logger.handlers.clear()

    # Add queue handler - sends records to main process
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger.addHandler(queue_handler)

    # Don't propagate to root logger (we handle everything via queue)
    logger.propagate = False

    # Store process info in logger name for identification
    process = mp.current_process()
    # Add process-specific context to log records
    old_factory = logging.getLogRecordFactory()

    def record_factory(*args, **kwargs):
        record = old_factory(*args, **kwargs)
        record.processName = process.name
        return record

    logging.setLogRecordFactory(record_factory)


# Additional utility for checking if we're in a worker process
def is_main_process() -> bool:
    """
    Check if code is running in the main process.

    Returns:
        bool: True if in main process, False if in worker

    Example:
        if is_main_process():
            logger.info("Running in main process")
        else:
            logger.info("Running in worker process")
    """
    return mp.current_process().name == "MainProcess"
