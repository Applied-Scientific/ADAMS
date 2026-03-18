"""
Unified parallel executor for running tasks across CPU cores and GPUs.

Replaces the ad-hoc Process/Pool/Queue patterns scattered across ligand
preparation, MD simulation, conformer generation, etc. with a single
Pool-based dispatcher that handles:

- Automatic serial fallback when multiprocessing is unavailable or n_workers=1
- Queue-based logging setup in spawned workers
- SIGINT detection so workers exit cleanly
- GPU assignment via per-worker CUDA visibility mapping
- Consistent error collection without deadlock-prone Queue polling

Usage::

    from adams.utils.parallel_executor import ParallelExecutor, ResourceConfig, TaskResult

    def process_pose(pose_dir: str) -> TaskResult:
        try:
            do_work(pose_dir)
            return TaskResult(task_id=pose_dir, success=True)
        except Exception as e:
            return TaskResult(task_id=pose_dir, success=False, error=str(e))

    executor = ParallelExecutor(ResourceConfig(n_workers=4, n_gpus=1))
    results = executor.run(process_pose, pose_dirs)
"""

import os
import traceback
from dataclasses import dataclass
from multiprocessing import current_process
from typing import Any, Callable, List, Optional

from ..logger_utils import get_logger, setup_multiprocessing_logging
from .multiprocessing_utils import (
    Pool,
    can_start_multiprocessing,
    cleanup_pool,
    configure_worker_logging,
    is_main_process,
)


@dataclass
class TaskResult:
    """Outcome of a single parallel task."""

    task_id: str
    success: bool
    value: Any = None
    error: Optional[str] = None


@dataclass
class ResourceConfig:
    """Hardware resource allocation for parallel execution.

    Attributes:
        n_workers: Number of parallel workers (0 = auto from CPU count).
        n_gpus: Number of GPUs available (0 = CPU-only).
        gpu_strategy: How to assign GPUs to workers.
            ``"round_robin"`` — assign GPUs by worker slot. When there are
            more GPUs than workers, each worker receives a contiguous GPU
            subset via ``CUDA_VISIBLE_DEVICES``.
            ``"serialize_gpu"`` — only one worker runs GPU work at a time
            (implemented by capping the Pool to 1 for GPU-bound phases).
            ``"none"`` — do not pin workers to a specific GPU via
            ``CUDA_VISIBLE_DEVICES``; allows a task to see all GPUs.
    """

    n_workers: int = 0
    n_gpus: int = 0
    gpu_strategy: str = "round_robin"


def _resolve_n_workers(config: ResourceConfig) -> int:
    """Determine effective worker count."""
    if config.gpu_strategy == "serialize_gpu" and config.n_gpus == 1:
        return 1
    if config.n_workers > 0:
        return config.n_workers
    return max(1, (os.cpu_count() or 1) - 2)


def _get_worker_cuda_visible_devices(
    n_gpus: int, gpu_strategy: str, n_workers: int
) -> Optional[str]:
    """Derive CUDA_VISIBLE_DEVICES value for the current worker process."""
    if n_gpus <= 0:
        return None
    if gpu_strategy == "none":
        return None
    if gpu_strategy == "serialize_gpu" and n_gpus == 1:
        return "0"

    identity = current_process()._identity
    worker_slot = (identity[0] - 1) if identity else 0

    if n_workers > 0 and n_gpus > n_workers:
        # Partition GPUs into contiguous per-worker groups so each worker can
        # run a single command across multiple GPUs when available.
        base = n_gpus // n_workers
        extra = n_gpus % n_workers
        start = worker_slot * base + min(worker_slot, extra)
        count = base + (1 if worker_slot < extra else 0)
        return ",".join(str(gpu_id) for gpu_id in range(start, start + count))

    return str(worker_slot % n_gpus)


def _worker_wrapper(packed_args):
    """Top-level wrapper executed inside each Pool worker.

    Handles logging setup, SIGINT checking, GPU env injection, and
    exception capture so the caller's function stays focused on
    domain logic.
    """
    (
        func,
        task,
        task_id,
        n_gpus,
        n_workers,
        gpu_strategy,
        agent_data_path,
    ) = packed_args

    from ..error_handling import is_sigint_pending
    from ..path_config import set_agent_data_path

    # Spawned workers do not inherit contextvars. Rehydrate the path context so
    # logger/path helpers behave the same as the parent process.
    if agent_data_path:
        set_agent_data_path(path=agent_data_path)

    if not is_main_process():
        log_queue = setup_multiprocessing_logging()
        configure_worker_logging(log_queue)

    logger = get_logger()

    if is_sigint_pending():
        return TaskResult(task_id=task_id, success=False, error="SIGINT before start")

    cuda_visible_devices = _get_worker_cuda_visible_devices(
        n_gpus, gpu_strategy, n_workers
    )
    if gpu_strategy == "none":
        # Explicitly unpin to preserve multi-GPU visibility in this process.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    elif cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    try:
        result = func(task)
        if isinstance(result, TaskResult):
            return result
        return TaskResult(task_id=task_id, success=True, value=result)
    except Exception as exc:
        tb = traceback.format_exc()
        msg = f"Task {task_id} failed: {exc}\n{tb}"
        logger.error(msg)
        return TaskResult(task_id=task_id, success=False, error=msg)


class ParallelExecutor:
    """Execute a list of tasks using a spawn-based Pool with automatic fallback."""

    def __init__(self, config: Optional[ResourceConfig] = None):
        self.config = config or ResourceConfig()
        self.logger = get_logger()

    def run(
        self,
        func: Callable[[Any], TaskResult],
        tasks: list,
        task_id_fn: Callable[[Any], str] = str,
    ) -> List[TaskResult]:
        """Run *func* over *tasks* in parallel.

        Args:
            func: Callable that accepts a single task argument and returns a
                ``TaskResult`` (or any value, which is auto-wrapped).
            tasks: Iterable of task arguments.
            task_id_fn: Extracts a human-readable id from each task for logging.

        Returns:
            List of ``TaskResult`` in completion order.
        """
        if not tasks:
            return []

        n_workers = _resolve_n_workers(self.config)
        n_workers = min(n_workers, len(tasks))
        use_mp = n_workers > 1 and can_start_multiprocessing()

        setup_multiprocessing_logging()

        # Capture the current agent-data context from the parent process.
        # This contextvar does not cross spawn boundaries automatically.
        agent_data_path = None
        try:
            from ..path_config import get_agent_data_path

            agent_data_path = str(get_agent_data_path())
        except Exception:
            # Keep compatibility with call sites that do not require path_config.
            agent_data_path = None

        packed = [
            (
                func,
                task,
                task_id_fn(task),
                self.config.n_gpus,
                n_workers,
                self.config.gpu_strategy,
                agent_data_path,
            )
            for task in tasks
        ]

        results: List[TaskResult] = []

        if not use_mp:
            if n_workers > 1 and not can_start_multiprocessing():
                self.logger.info(
                    "Multiprocessing unavailable; running %d tasks serially.",
                    len(tasks),
                )
            self.logger.info("Execution mode: serial (%d tasks)", len(tasks))
            for args in packed:
                results.append(_worker_wrapper(args))
        else:
            self.logger.info(
                "Execution mode: multiprocessing (%d tasks, %d workers)",
                len(tasks),
                n_workers,
            )
            if (
                self.config.n_gpus > 1
                and self.config.gpu_strategy == "round_robin"
            ):
                assignments = ", ".join(
                    f"worker {i} -> GPU {i % self.config.n_gpus}"
                    for i in range(n_workers)
                )
                self.logger.info(
                    "GPU round-robin (%d GPUs): %s",
                    self.config.n_gpus,
                    assignments,
                )
            pool = None
            try:
                pool = Pool(processes=n_workers)
                for result in pool.imap_unordered(_worker_wrapper, packed):
                    results.append(result)
                pool.close()
                pool.join()
            except KeyboardInterrupt:
                self.logger.info("Parallel execution interrupted by user (Ctrl+C)")
                if pool is not None:
                    pool.terminate()
                    try:
                        pool.join()
                    except Exception:
                        pass
                raise
            except Exception:
                if pool is not None:
                    pool.terminate()
                    try:
                        pool.join()
                    except Exception:
                        pass
                raise
            finally:
                cleanup_pool(pool, terminate=True, timeout=3.0)

        succeeded = sum(1 for r in results if r.success)
        failed = len(results) - succeeded
        if failed:
            self.logger.warning(
                "%d/%d tasks failed.", failed, len(results)
            )
        else:
            self.logger.info("All %d tasks completed successfully.", len(results))

        return results

    def run_or_raise(
        self,
        func: Callable[[Any], TaskResult],
        tasks: list,
        task_id_fn: Callable[[Any], str] = str,
        fail_fast_threshold: float = 1.0,
    ) -> List[TaskResult]:
        """Like ``run`` but raises if too many tasks fail.

        Args:
            fail_fast_threshold: Fraction of tasks that must fail to raise
                (1.0 = raise only when ALL fail, 0.0 = raise on any failure).
        """
        results = self.run(func, tasks, task_id_fn)
        failed = [r for r in results if not r.success]
        if failed and len(failed) >= len(tasks) * fail_fast_threshold:
            first_err = failed[0].error or "unknown error"
            raise RuntimeError(
                f"{len(failed)}/{len(tasks)} tasks failed. First error: {first_err}"
            )
        return results
