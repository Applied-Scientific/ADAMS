"""
Shared helpers for MD phase planning and GPU-aware task attachment.

Used by SolubleMd to build execution phases and attach ntmpi/ntomp per phase.
"""

import os
from typing import Any, Callable, List, Tuple

from ....utils.parallel_plans import (
    ExecutionPhase,
    build_gpu_phases,
)
from .gromacs_context import GromacsContext


def should_use_hybrid_gpu_mode(
    ctx: GromacsContext,
    n_poses: int,
    *,
    user_max_jobs_override: bool = False,
    user_parallel_override: bool = False,
) -> bool:
    """Enable hybrid scheduling when running many poses on multi-GPU hosts."""
    return (
        ctx.gpu
        and ctx.num_gpus > 1
        and n_poses > 1
        and not user_max_jobs_override
        and not user_parallel_override
    )


def validate_gpu_phase_exclusivity(phases: List[ExecutionPhase], gpu: bool) -> None:
    """Ensure phase configs cannot double-assign a GPU across tasks."""
    if not gpu:
        return
    for phase in phases:
        cfg = phase.config
        if cfg.n_gpus <= 0:
            continue
        if cfg.gpu_strategy == "none" and cfg.n_workers > 1:
            raise ValueError(
                "Invalid GPU scheduling in phase "
                f"'{phase.name}': gpu_strategy='none' with n_workers={cfg.n_workers} "
                "exposes all GPUs to multiple concurrent tasks."
            )
        if cfg.gpu_strategy != "none" and cfg.n_workers > cfg.n_gpus:
            raise ValueError(
                "Invalid GPU scheduling in phase "
                f"'{phase.name}': n_workers={cfg.n_workers} exceeds n_gpus={cfg.n_gpus}. "
                "This would assign one GPU to multiple concurrent tasks."
            )


def attach_phase_parallel_settings(
    task: dict,
    phase_name: str,
    ctx: GromacsContext,
    max_jobs: int,
) -> dict:
    """Attach ntmpi/ntomp to a task according to the execution phase."""
    prepared = dict(task)
    total_cpus = max(1, os.cpu_count() or 1)
    if phase_name == "hybrid_bulk_single_gpu":
        bulk_workers = max(1, min(ctx.num_gpus, max_jobs))
        prepared["ntmpi"] = 1
        prepared["ntomp"] = max(1, total_cpus // bulk_workers)
    elif phase_name == "hybrid_tail_multi_gpu":
        prepared["ntmpi"] = ctx.num_gpus
        prepared["ntomp"] = max(1, total_cpus // ctx.num_gpus)
    else:
        prepared["ntmpi"] = ctx.ntmpi
        prepared["ntomp"] = ctx.ntomp
    return prepared


def build_md_phases(
    ctx: GromacsContext,
    tasks: list,
    *,
    max_jobs: int = 0,
    hybrid_tail: bool = False,
    user_max_jobs_override: bool = False,
    user_parallel_override: bool = False,
) -> Tuple[List[ExecutionPhase], Callable[[Any, str], Any]]:
    """Build GPU-aware execution phases and return (phases, task_transform_fn).

    Encapsulates hybrid-mode detection, phase-specific MPI/OMP overrides,
    and GPU exclusivity validation. The returned transform function attaches
    ntmpi/ntomp to each task per phase.
    """
    n_gpus = ctx.num_gpus if ctx.gpu else 0
    default_gpu_strategy = (
        "none"
        if (
            ctx.gpu
            and ctx.ntmpi > 1
            and not hybrid_tail
            and max_jobs == 1
        )
        else None
    )
    phases = build_gpu_phases(
        tasks,
        n_gpus=n_gpus,
        n_workers=max_jobs,
        hybrid_tail=hybrid_tail,
        default_gpu_strategy=default_gpu_strategy,
    )
    validate_gpu_phase_exclusivity(phases, ctx.gpu)
    transform_fn = lambda t, p: attach_phase_parallel_settings(t, p, ctx, max_jobs)
    return phases, transform_fn
