"""Planning helpers layered on top of the core parallel executor."""

from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from ..logger_utils import get_logger
from .parallel_executor import ParallelExecutor, ResourceConfig, TaskResult


@dataclass
class ExecutionPhase:
    """A named execution phase with its own resource configuration."""

    name: str
    tasks: list
    config: ResourceConfig


def build_gpu_phases(
    tasks: list,
    *,
    n_gpus: int,
    n_workers: int = 0,
    hybrid_tail: bool = False,
    default_gpu_strategy: Optional[str] = None,
) -> List[ExecutionPhase]:
    """Build GPU-aware execution phases usable across pipeline modules."""
    if not tasks:
        return []

    n_gpus_val = max(0, int(n_gpus or 0))
    if default_gpu_strategy:
        strategy = default_gpu_strategy
    elif n_gpus_val == 1:
        strategy = "serialize_gpu"
    elif n_gpus_val > 1:
        strategy = "round_robin"
    else:
        strategy = "round_robin"

    if not hybrid_tail or n_gpus_val <= 1 or len(tasks) <= 1:
        return [
            ExecutionPhase(
                name="default",
                tasks=tasks,
                config=ResourceConfig(
                    n_workers=n_workers,
                    n_gpus=n_gpus_val,
                    gpu_strategy=strategy,
                ),
            )
        ]

    bulk_count = (len(tasks) // n_gpus_val) * n_gpus_val
    bulk_tasks = tasks[:bulk_count]
    tail_tasks = tasks[bulk_count:]
    phases: List[ExecutionPhase] = []

    if bulk_tasks:
        bulk_workers = n_workers if n_workers > 0 else n_gpus_val
        bulk_workers = min(bulk_workers, n_gpus_val, len(bulk_tasks))
        phases.append(
            ExecutionPhase(
                name="hybrid_bulk_single_gpu",
                tasks=bulk_tasks,
                config=ResourceConfig(
                    n_workers=bulk_workers,
                    n_gpus=n_gpus_val,
                    gpu_strategy="round_robin",
                ),
            )
        )
    if tail_tasks:
        phases.append(
            ExecutionPhase(
                name="hybrid_tail_multi_gpu",
                tasks=tail_tasks,
                config=ResourceConfig(
                    n_workers=1,
                    n_gpus=n_gpus_val,
                    gpu_strategy="none",
                ),
            )
        )
    return phases


def run_phases_or_raise(
    func: Callable[[Any], TaskResult],
    phases: List[ExecutionPhase],
    *,
    task_id_fn: Callable[[Any], str] = str,
    fail_fast_threshold: float = 1.0,
    task_transform_fn: Optional[Callable[[Any, str], Any]] = None,
) -> List[TaskResult]:
    """Execute phase plans with per-phase resource configuration."""
    logger = get_logger()
    all_results: List[TaskResult] = []
    for phase in phases:
        if not phase.tasks:
            continue
        transformed_tasks = (
            [task_transform_fn(task, phase.name) for task in phase.tasks]
            if task_transform_fn is not None
            else phase.tasks
        )
        logger.info(
            "Execution phase '%s': %d tasks (workers=%d, gpus=%d, strategy=%s)",
            phase.name,
            len(transformed_tasks),
            phase.config.n_workers,
            phase.config.n_gpus,
            phase.config.gpu_strategy,
        )
        phase_executor = ParallelExecutor(phase.config)
        phase_results = phase_executor.run_or_raise(
            func,
            transformed_tasks,
            task_id_fn=task_id_fn,
            fail_fast_threshold=fail_fast_threshold,
        )
        all_results.extend(phase_results)
    return all_results
