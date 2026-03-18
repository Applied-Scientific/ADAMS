"""
GROMACS execution context: composition over inheritance.

Replaces GromacsRunner base class with a serializable context object used
by prepare/simulate/analyze workflows.
"""

import os
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, List, Tuple

from ....common_utils import get_gpu_count
from ....logger_utils import get_logger

from .gromacs_commands import run_md_stage
from .gromacs_paths import get_gromacs_binary
from .grompp_warnings import GromppWarningPolicy

if TYPE_CHECKING:
    from .grompp_warnings import GromppWarningPolicy as GromppPolicy  # noqa: F401


@dataclass(frozen=False)
class GromacsContext:
    """Lightweight, serializable GROMACS execution state."""

    gmx_binary: str
    binary_type: str  # "standard" | "mpi" | "cuda"
    gpu: bool
    num_gpus: int
    ntmpi: int
    ntomp: int
    max_mpi: int
    grompp_warning_policy: "GromppWarningPolicy"

    @classmethod
    def from_file_paths(
        cls,
        file_paths: dict,
        *,
        gpu: bool = False,
        num_gpus: int = -1,
        mpi_ranks: int = 0,
        omp_threads: int = 0,
        grompp_warning_policy: "GromppWarningPolicy" = None,
    ) -> "GromacsContext":
        """One-shot factory: binary resolution + hardware auto-detection."""
        if file_paths is None:
            raise ValueError("file_paths dictionary is required.")
        logger = get_logger()
        policy = (
            grompp_warning_policy
            if grompp_warning_policy is not None
            else GromppWarningPolicy()
        )
        gromacs_path = file_paths["gromacs_path"]
        base_binary_type = file_paths.get("gromacs_binary_type", "standard")
        binary_type = "cuda" if gpu else base_binary_type
        gmx_binary = get_gromacs_binary(
            gromacs_path,
            binary_type=binary_type,
            require_mpi=False,
        )
        if gpu:
            requested_all = num_gpus == -1
            if num_gpus == -1:
                num_gpus = get_gpu_count()
                logger.info("Auto-detected %d GPUs (num_gpus=-1: use all).", num_gpus)
            if num_gpus == 0:
                if requested_all:
                    logger.warning(
                        "GPU run requested but no GPUs detected. Falling back to CPU."
                    )
                else:
                    logger.info("num_gpus=0: CPU-only (no GPUs).")
                gpu = False
            elif num_gpus < 0:
                logger.warning(
                    "num_gpus=%d invalid; must be -1 (all), 0 (none), or positive. Falling back to CPU.",
                    num_gpus,
                )
                gpu = False
                num_gpus = 0
        else:
            num_gpus = 0
        available_cores = max(1, os.cpu_count() or 1)
        if mpi_ranks <= 0 or omp_threads <= 0:
            if gpu and num_gpus > 0:
                ntmpi = num_gpus
                ntomp = max(1, available_cores // ntmpi)
                logger.info("GPU mode: ntmpi=%d, ntomp=%d", ntmpi, ntomp)
            else:
                ntmpi = available_cores
                ntomp = 1
                logger.info("CPU mode: ntmpi=%d, ntomp=%d", ntmpi, ntomp)
        else:
            ntmpi = mpi_ranks
            ntomp = omp_threads
            logger.info("User-specified: ntmpi=%d, ntomp=%d", ntmpi, ntomp)
        return cls(
            gmx_binary=gmx_binary,
            binary_type=binary_type,
            gpu=gpu,
            num_gpus=num_gpus,
            ntmpi=ntmpi,
            ntomp=ntomp,
            max_mpi=32,
            grompp_warning_policy=policy,
        )

    def run_stage(
        self,
        name: str,
        mdp: str,
        input_gro: str,
        index: str,
        topology: str,
        *,
        cwd: str = None,
    ) -> None:
        """Run one MD stage: grompp + mdrun. Delegates to run_md_stage."""
        run_md_stage(
            self.gmx_binary,
            name,
            mdp,
            input_gro,
            index,
            topology,
            run_on_gpu=self.gpu,
            binary_type=self.binary_type,
            ntmpi=self.ntmpi,
            ntomp=self.ntomp,
            max_mpi=self.max_mpi,
            warning_policy=self.grompp_warning_policy,
            logger=get_logger(),
            cwd=cwd,
        )

    def run_stages(
        self,
        stages: List[Tuple[str, str, str]],
        index: str,
        topology: str,
        *,
        cwd: str = None,
    ) -> None:
        """Run a sequence of (name, mdp, input_gro) stages."""
        for name, mdp, input_gro in stages:
            self.run_stage(name, mdp, input_gro, index, topology, cwd=cwd)

    def with_overrides(self, **kwargs) -> "GromacsContext":
        """Return a new context with modified fields (e.g. for per-phase MPI/OMP)."""
        return replace(self, **kwargs)

    def to_task_params(self) -> dict:
        """Serialize for parallel worker task dicts."""
        return {
            "gmx_binary": self.gmx_binary,
            "binary_type": self.binary_type,
            "gpu": self.gpu,
            "num_gpus": self.num_gpus,
            "ntmpi": self.ntmpi,
            "ntomp": self.ntomp,
            "max_mpi": self.max_mpi,
            "grompp_approved": list(self.grompp_warning_policy.approved_fingerprints),
        }

    @classmethod
    def from_task_params(cls, params: dict) -> "GromacsContext":
        """Reconstruct context in a worker process."""
        policy = GromppWarningPolicy(approved=set(params.get("grompp_approved", [])))
        return cls(
            gmx_binary=params["gmx_binary"],
            binary_type=params["binary_type"],
            gpu=params["gpu"],
            num_gpus=params["num_gpus"],
            ntmpi=params["ntmpi"],
            ntomp=params["ntomp"],
            max_mpi=params["max_mpi"],
            grompp_warning_policy=policy,
        )
