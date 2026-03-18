"""
GROMACS MD Simulation Module - MD Analysis Pipeline Step 3

This module runs GROMACS MD simulations for all prepared ligand-protein complexes.
It executes staged equilibration (NVT + multi-stage NPT with gradual restraint
release) and production MD simulations in parallel across multiple poses.

POSITION IN PIPELINE:
    Step 3 of 4: run_md_simulation
    - Requires outputs from run_md_prepare (prepared poses with min.gro)
    - Must be executed before run_md_analysis
    - Can be skipped if MD simulations already completed

INPUTS (from file_paths dictionary):
    - poses_dir: Directory containing prepared pose subdirectories (with min.gro files)
    - gromacs_path: Path to GROMACS installation
    - ambertools_path: Path to AmberTools installation

OUTPUTS (added to file_paths dictionary):
    - poses_dir: Updated with MD-completed poses (each containing md.tpr, md.xtc, etc.)

KEY FUNCTIONALITY:
    - Runs NVT equilibration (constant volume, temperature coupling)
    - Runs multi-stage NPT equilibration with progressive restraint release
    - Runs production MD simulation
    - Supports GPU acceleration (CUDA) and multi-rank MPI execution
    - Parallel execution across multiple poses using multiprocessing
    - Auto-calculates optimal MPI ranks and OpenMP threads

GROMPP WARNING POLICY:
    All grompp invocations run WITHOUT -maxwarn when no warnings are pre-approved.
    If grompp produces warnings, a GromppWarningError is raised so the calling
    agent/user can review each warning individually.  Approved warning
    fingerprints can be passed via grompp_warning_policy (or approved_grompp_warnings
    in the agent tool) to classify repeated warning types across poses/runs.
    When pre-approved fingerprints exist, grompp runs with a high -maxwarn bound
    so it can complete in one pass; every emitted warning is still validated
    against the policy and unapproved ones raise GromppWarningError.  No blanket
    -maxwarn override (e.g. -maxwarn 100).

EXTERNAL COMMANDS:
    - gmx grompp: Prepare TPR run input files for each MD phase
    - gmx mdrun: Execute MD simulation (NVT → staged NPT → Production MD)
    - mpirun (optional): For multi-rank MPI execution on CPU

CONFIGURATION:
    - GPU mode: Uses CUDA-enabled GROMACS binary
    - CPU mode: Uses standard or MPI-enabled GROMACS binary
    - Auto-scales MPI ranks to GROMACS-friendly numbers (5-smooth numbers)
"""

import glob
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional

import pandas as pd

from ....error_handling import is_sigint_pending, setup_sigint_handler
from ....logger_utils import (
    get_logger,
    log_step_execution,
)
from ....utils.parallel_executor import TaskResult
from ....utils.parallel_plans import run_phases_or_raise
from ..shared import (
    GromacsContext,
    GromppWarningPolicy,
    LIGAND_RESNAME,
    copy_mdp_files,
    get_ndx_group_index,
    get_mdp_dir,
    make_system_index,
    validate_required_file_paths,
    validate_system_index_groups,
)
from ..shared.parallel_helpers import build_md_phases, should_use_hybrid_gpu_mode

SOLUBLE_EQ_STAGES = [
    ("nvt", "nvt.mdp", "min.gro"),
    ("npt_eq1", "npt_eq1.mdp", "nvt.gro"),
    ("npt_eq2", "npt_eq2.mdp", "npt_eq1.gro"),
    ("npt_eq3", "npt_eq3.mdp", "npt_eq2.gro"),
    ("npt_eq4", "npt_eq4.mdp", "npt_eq3.gro"),
]


class SolubleMd:
    """Run GROMACS MD simulations for prepared soluble ligand-protein poses."""

    def __init__(
        self,
        file_paths,
        gpu: bool = False,
        num_gpus: int = -1,
        mpi_ranks: int = 0,
        omp_threads: int = 0,
        topol: str = "system.top",
        index: str = "index.ndx",
        max_jobs: int = 0,
        production_nsteps: Optional[int] = None,
        production_dt_fs: float = 2.0,
        soluble_eq_nsteps_scale: Optional[float] = None,
        grompp_warning_policy: "GromppWarningPolicy" = None,
    ):
        r"""
        Args:
            file_paths: dict: File paths dictionary (required). Must include poses_dir,
                gromacs_path, ambertools_path.
            gpu: Whether to use GPU (default: False).
            num_gpus: Number of GPUs (-1 = use all available; 0 = CPU-only; N = use N; default: -1).
            mpi_ranks: MPI ranks (0 = auto).
            omp_threads: OpenMP threads (0 = auto).
            topol: Topology file name (default: "system.top").
            index: Index file name (default: "index.ndx").
            max_jobs: Max concurrent jobs (0 = auto).
            production_nsteps: Optional override for production ``md.mdp`` nsteps.
            production_dt_fs: Production timestep in fs (default: 2.0). Equilibration stays 2 fs.
            soluble_eq_nsteps_scale: Optional multiplier for copied soluble
                equilibration ``nsteps`` values.
            grompp_warning_policy: Optional shared grompp warning policy.
        """
        setup_sigint_handler()
        self.logger = get_logger()
        if file_paths is None:
            raise ValueError("file_paths dictionary is required.")
        self.file_paths = file_paths
        self.validate_files()
        self.ctx = GromacsContext.from_file_paths(
            file_paths,
            gpu=gpu,
            num_gpus=num_gpus,
            mpi_ranks=mpi_ranks,
            omp_threads=omp_threads,
            grompp_warning_policy=grompp_warning_policy,
        )
        self.topol = topol
        self.index = index
        self.md_workdir = file_paths.get("md_root", ".")
        self.case_path = self.md_workdir
        self.max_jobs = max_jobs
        self.production_nsteps = (
            int(production_nsteps)
            if production_nsteps is not None
            else None
        )
        self.production_dt_fs = float(production_dt_fs)
        self.soluble_eq_nsteps_scale = (
            float(soluble_eq_nsteps_scale)
            if soluble_eq_nsteps_scale is not None
            else None
        )
        if self.production_nsteps is not None and self.production_nsteps <= 0:
            raise ValueError(
                f"production_nsteps must be > 0 when provided (got {production_nsteps})."
            )
        if self.production_dt_fs <= 0:
            raise ValueError(
                f"production_dt_fs must be > 0 (got {production_dt_fs})."
            )
        if (
            self.soluble_eq_nsteps_scale is not None
            and self.soluble_eq_nsteps_scale <= 0
        ):
            raise ValueError(
                "soluble_eq_nsteps_scale must be > 0 when provided "
                f"(got {soluble_eq_nsteps_scale})."
            )
        self._user_max_jobs_override = max_jobs > 0
        self._user_parallel_override = (mpi_ranks > 0) or (omp_threads > 0)
        self.root_path = os.getcwd()
        self.init_str = "init"
        self.min_str = "min"
        self.nvt_str = "nvt"
        self.npt_str = "npt_eq4"
        self.md_str = "md"

    def validate_files(self):
        """
        Validate required keys exist in file_paths.

        Checks that:
        - poses_dir exists in file_paths
        - gromacs_path exists in file_paths
        - ambertools_path exists in file_paths

        Raises:
            ValueError: If required paths are missing from file_paths
        """
        validate_required_file_paths(
            self.file_paths,
            ["poses_dir"],
            context=(
                "Gro requires poses_dir with prepared poses.\n"
                "Ensure LigPrepare has run (or provide an existing poses_dir)."
            ),
        )

    def run(self) -> dict:
        """
        Run GROMACS MD simulations for all prepared poses.

        Returns:
            dict: Updated file_paths dictionary with md_completed_poses list
        """
        step_logger = log_step_execution("MD Simulation", self.logger)
        with step_logger:
            if self.ctx.grompp_warning_policy._approved:
                approved_list = (
                    self.ctx.grompp_warning_policy.get_approved_with_descriptions()
                )
                self.logger.info(
                    "Loaded %d pre-approved grompp warning fingerprint(s) from previous run(s) "
                    "for warning classification.",
                    len(approved_list),
                )
                for item in approved_list:
                    self.logger.info(
                        "  Pre-approved warning fingerprint: %s",
                        item["description"],
                    )
            folders = self._prepwork()

            self.logger.info(
                f"Starting MD simulations for {len(folders)} poses using {self.max_jobs} workers..."
            )

            from ....path_config import get_agent_data_path
            agent_data_path = get_agent_data_path()

            base_tasks = [
                {
                    "pose_name": pose_name,
                    "case_path": self.case_path,
                    "agent_data_path": agent_data_path,
                    "ctx": self.ctx.to_task_params(),
                    "topol": self.topol,
                    "index": self.index,
                    "min_str": self.min_str,
                    "md_str": self.md_str,
                    "production_nsteps": self.production_nsteps,
                    "production_dt_fs": self.production_dt_fs,
                    "soluble_eq_nsteps_scale": self.soluble_eq_nsteps_scale,
                }
                for pose_name in folders
            ]

            hybrid_mode = should_use_hybrid_gpu_mode(
                self.ctx, len(folders),
                user_max_jobs_override=self._user_max_jobs_override,
                user_parallel_override=self._user_parallel_override,
            )
            if hybrid_mode:
                self.logger.info(
                    "Hybrid GPU scheduling enabled: bulk poses use 1 GPU/job, tail uses multi-GPU."
                )

            phases, transform_fn = build_md_phases(
                self.ctx,
                base_tasks,
                max_jobs=self.max_jobs,
                hybrid_tail=hybrid_mode,
                user_max_jobs_override=self._user_max_jobs_override,
                user_parallel_override=self._user_parallel_override,
            )

            results = run_phases_or_raise(
                _gro_run_single,
                phases,
                task_id_fn=lambda t: t["pose_name"],
                fail_fast_threshold=1.0,
                task_transform_fn=transform_fn,
            )

            succeeded = [r for r in results if r.success]
            failed = [r for r in results if not r.success]
            if failed:
                for idx, r in enumerate(failed[:10], start=1):
                    self.logger.warning("MD failure %d: %s", idx, r.error)
                if len(failed) > 10:
                    self.logger.warning("... and %d more MD failures", len(failed) - 10)

            runtime_rows: List[Dict[str, Any]] = []
            for result in results:
                value = result.value if isinstance(result.value, dict) else {}
                runtime_rows.append(
                    {
                        "pose_name": result.task_id,
                        "status": "success" if result.success else "failed",
                        "production_dt_fs_requested": value.get(
                            "production_dt_fs_requested", self.production_dt_fs
                        ),
                        "production_dt_fs_used": value.get("production_dt_fs_used"),
                        "production_nsteps_requested": value.get(
                            "production_nsteps_requested", self.production_nsteps
                        ),
                        "production_nsteps_used": value.get("production_nsteps_used"),
                        "soluble_eq_nsteps_scale_requested": value.get(
                            "soluble_eq_nsteps_scale_requested",
                            self.soluble_eq_nsteps_scale,
                        ),
                        "production_dt_fallback_used": value.get(
                            "production_dt_fallback_used", False
                        ),
                        "error": (result.error or "").strip(),
                    }
                )

            runtime_summary_path = os.path.join(self.case_path, "md_runtime_summary.csv")
            pd.DataFrame(runtime_rows).to_csv(runtime_summary_path, index=False)
            self.file_paths["md_runtime_summary"] = runtime_summary_path
            self.file_paths["md_completed_poses"] = [r.task_id for r in succeeded]
            self.file_paths["md_failed_poses"] = [r.task_id for r in failed]

            if not succeeded:
                raise RuntimeError(
                    "MD simulation failed for all poses. "
                    f"See {runtime_summary_path} for details."
                )

            return self.file_paths

    def _prepwork(self) -> List[str]:
        """
        Prepare for MD runs by getting pose directory from file_paths.

        Returns:
            List[str]: List of pose folder names to process
        """
        # Files are already validated in __init__, just get poses_dir
        # Convert to absolute path for worker processes (workers may have different cwd)
        poses_dir = os.path.abspath(self.file_paths["poses_dir"])
        self.case_path = poses_dir

        folders = []
        skipped = []
        for name in sorted(os.listdir(poses_dir)):
            pose_path = os.path.join(poses_dir, name)
            if os.path.isdir(pose_path):
                # Check if this is a prepared pose (has min.gro)
                if os.path.exists(os.path.join(pose_path, "min.gro")):
                    missing_required = []
                    for rel in (self.topol, self.index):
                        if not os.path.exists(os.path.join(pose_path, rel)):
                            missing_required.append(rel)
                    if missing_required:
                        skipped.append(
                            f"{name}: missing required file(s): {', '.join(missing_required)}"
                        )
                        continue

                    missing_includes = self._repair_missing_local_topology_includes(
                        pose_path
                    )
                    if missing_includes:
                        skipped.append(
                            f"{name}: unresolved local topology include(s): "
                            f"{', '.join(missing_includes)}"
                        )
                        continue

                    try:
                        ndx_path = os.path.join(pose_path, self.index)
                        get_ndx_group_index(ndx_path, "Protein_LIG")
                        get_ndx_group_index(ndx_path, "Water_and_ions")
                    except ValueError:
                        rebuilt = self._rebuild_pose_index_for_md(pose_path)
                        if rebuilt:
                            try:
                                get_ndx_group_index(ndx_path, "Protein_LIG")
                                get_ndx_group_index(ndx_path, "Water_and_ions")
                            except ValueError as e:
                                skipped.append(
                                    f"{name}: invalid index groups after rebuild: {e}"
                                )
                                continue
                        else:
                            skipped.append(
                                f"{name}: invalid index groups and could not rebuild index"
                            )
                            continue

                    folders.append(name)

        self.logger.info(f"Found {len(folders)} prepared poses in {poses_dir}")
        self.logger.debug(f"Pose folders: {folders}")
        if skipped:
            self.logger.warning(
                "Skipped %d pose(s) due to preflight validation failures:\n%s",
                len(skipped),
                "\n".join(f"  - {msg}" for msg in skipped),
            )
        if not folders:
            detail = "\n".join(f"  - {msg}" for msg in skipped[:10])
            raise ValueError(
                "No valid prepared poses found for MD.\n"
                "Preflight validation failures:\n"
                f"{detail}"
            )

        self.root_path = os.getcwd()
        self.logger.debug(f"root_path: {self.root_path}")

        # Input mdp stage names
        self.max_mpi = 32
        self.init_str = "init"
        self.min_str = "min"
        self.nvt_str = "nvt"
        self.npt_str = "npt_eq4"
        self.md_str = "md"

        # Calculate total_cpus and cpus_per_job regardless of max_jobs value
        total_cpus = os.cpu_count()
        cpus_per_job = self.ctx.ntmpi * self.ctx.ntomp

        if cpus_per_job == 0:
            raise ValueError(
                f"cpus_per_job cannot be 0 (ntmpi={self.ctx.ntmpi}, ntomp={self.ctx.ntomp})"
            )

        if self.max_jobs <= 0:
            if should_use_hybrid_gpu_mode(
                self.ctx,
                len(folders),
                user_max_jobs_override=self._user_max_jobs_override,
                user_parallel_override=self._user_parallel_override,
            ):
                self.max_jobs = min(self.ctx.num_gpus, len(folders))
                self.logger.info(
                    "Auto-calculated max_jobs = %d (hybrid GPU bulk phase)",
                    self.max_jobs,
                )
            else:
                self.max_jobs = max(1, total_cpus // cpus_per_job) if cpus_per_job else 1
                self.logger.info(f"Auto-calculated max_jobs = {self.max_jobs}")
        else:
            self.logger.info(f"User override: max_jobs = {self.max_jobs}")

        self.logger.info(f"System has {total_cpus} CPUs")
        self.logger.info(
            f"Each job uses {cpus_per_job} CPUs ({self.ctx.ntmpi} MPI x {self.ctx.ntomp} OMP)"
        )
        if self.ctx.gpu and self.max_jobs > self.ctx.num_gpus:
            raise ValueError(
                "Invalid GPU scheduling: "
                f"max_jobs={self.max_jobs} exceeds num_gpus={self.ctx.num_gpus}. "
                "This would assign one GPU to multiple concurrent tasks. "
                "Reduce max_jobs or increase num_gpus."
            )
        self.logger.info(f"Scheduling up to {self.max_jobs} jobs in parallel")

        return folders

    def _repair_missing_local_topology_includes(self, pose_path: str) -> List[str]:
        """
        Ensure locally included ITP files referenced by system.top exist.

        For missing ``posre_*.itp`` includes, attempt self-heal by copying from any
        matching file under the pose directory tree (e.g. ``LIG.acpype/``).
        Returns unresolved include paths (relative include targets).
        """
        top_path = os.path.join(pose_path, self.topol)
        with open(top_path, "r", encoding="utf-8", errors="ignore") as fh:
            top_text = fh.read()

        includes = re.findall(r'^\s*#include\s+"([^"]+)"', top_text, flags=re.MULTILINE)
        missing = []
        for inc in includes:
            # External force-field includes are resolved by GROMACS datadir.
            if ".ff/" in inc or os.path.isabs(inc):
                continue
            include_path = os.path.join(pose_path, inc)
            if os.path.exists(include_path):
                continue

            base = os.path.basename(inc)
            repaired = False
            if base.startswith("posre_") and base.endswith(".itp"):
                candidates = glob.glob(os.path.join(pose_path, "**", base), recursive=True)
                if candidates:
                    os.makedirs(os.path.dirname(include_path), exist_ok=True)
                    shutil.copy2(candidates[0], include_path)
                    self.logger.warning(
                        "Recovered missing include '%s' for pose '%s' from '%s'.",
                        inc,
                        os.path.basename(pose_path),
                        os.path.relpath(candidates[0], pose_path),
                    )
                    repaired = True

            if not repaired:
                missing.append(inc)

        return missing

    def _rebuild_pose_index_for_md(self, pose_path: str) -> bool:
        """
        Rebuild index.ndx with required MD groups for resumed/legacy prepared poses.
        """
        gro_candidates = [
            os.path.join(pose_path, "solv_ions.gro"),
            os.path.join(pose_path, f"{self.min_str}.gro"),
        ]
        gro_path = next((p for p in gro_candidates if os.path.exists(p)), None)
        top_path = os.path.join(pose_path, self.topol)
        ndx_path = os.path.join(pose_path, self.index)
        if not gro_path or not os.path.exists(top_path):
            return False
        try:
            ligand_gro_path = os.path.join(
                pose_path,
                f"{LIGAND_RESNAME}.acpype",
                f"{LIGAND_RESNAME}_GMX.gro",
            )
            make_system_index(
                self.ctx.gmx_binary,
                gro_path,
                ndx_path,
                ligand_resname=LIGAND_RESNAME,
                top_path=top_path,
                ligand_gro_path=ligand_gro_path,
            )
            self.logger.warning(
                "Rebuilt index for pose '%s' to ensure required MD groups are present.",
                os.path.basename(pose_path),
            )
            return True
        except Exception as e:
            self.logger.warning(
                "Failed to rebuild index for pose '%s': %s",
                os.path.basename(pose_path),
                e,
            )
            return False


# ---------------------------------------------------------------------------
# Module-level worker function (picklable for spawn-based Pool workers)
# ---------------------------------------------------------------------------


def _format_dt_ps(dt_fs: float) -> str:
    """Convert fs to ps for GROMACS mdp dt field with stable formatting."""
    dt_ps = float(dt_fs) / 1000.0
    return f"{dt_ps:.6f}".rstrip("0").rstrip(".")


def _set_mdp_parameter(mdp_path: str, key: str, value: Any) -> None:
    """Set (or append) a single parameter in an MDP file."""
    key_lower = key.strip().lower()
    lines = []
    found = False
    with open(mdp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith(";") or "=" not in line:
                lines.append(line)
                continue
            lhs = line.split("=", 1)[0].strip().lower()
            if lhs == key_lower:
                lines.append(f"{key} = {value}\n")
                found = True
            else:
                lines.append(line)
    if not found:
        lines.append(f"{key} = {value}\n")
    with open(mdp_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _read_mdp_parameter(mdp_path: str, key: str) -> Optional[str]:
    """Read a parameter value from an MDP file; returns None if absent."""
    key_lower = key.strip().lower()
    with open(mdp_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith(";") or "=" not in line:
                continue
            lhs, rhs = line.split("=", 1)
            if lhs.strip().lower() == key_lower:
                return rhs.strip().split(";", 1)[0].strip()
    return None


def _read_mdp_integer_parameter(mdp_path: str, key: str) -> int:
    """Read an integer-like MDP parameter, accepting values such as 250000.0."""
    raw = _read_mdp_parameter(mdp_path, key)
    if raw is None:
        raise ValueError(f"Parameter '{key}' not found in MDP file: {mdp_path}")
    try:
        return int(round(float(raw)))
    except Exception as exc:
        raise ValueError(
            f"Parameter '{key}' in {mdp_path} is not numeric: {raw!r}"
        ) from exc


def _apply_soluble_eq_nsteps_scale(
    pose_dir: str,
    scale: float,
) -> Dict[str, Dict[str, int]]:
    """Scale copied soluble equilibration ``nsteps`` values for one pose directory."""
    applied: Dict[str, Dict[str, int]] = {}
    for stage_name, mdp_file, _ in SOLUBLE_EQ_STAGES:
        mdp_path = os.path.join(pose_dir, mdp_file)
        original = _read_mdp_integer_parameter(mdp_path, "nsteps")
        scaled = max(1, int(round(original * float(scale))))
        _set_mdp_parameter(mdp_path, "nsteps", scaled)
        applied[stage_name] = {
            "original_nsteps": original,
            "scaled_nsteps": scaled,
        }
    return applied


def _apply_production_mdp_overrides(
    mdp_path: str,
    *,
    dt_fs: float,
    nsteps: Optional[int],
) -> Dict[str, Any]:
    """Apply runtime overrides to production md.mdp and return applied values."""
    _set_mdp_parameter(mdp_path, "dt", _format_dt_ps(dt_fs))
    if nsteps is not None:
        _set_mdp_parameter(mdp_path, "nsteps", int(nsteps))

    applied_dt_ps = _read_mdp_parameter(mdp_path, "dt")
    applied_nsteps_raw = _read_mdp_parameter(mdp_path, "nsteps")
    applied_nsteps = None
    if applied_nsteps_raw is not None:
        try:
            applied_nsteps = int(float(applied_nsteps_raw))
        except Exception:
            applied_nsteps = applied_nsteps_raw
    return {
        "dt_ps": applied_dt_ps,
        "nsteps": applied_nsteps,
    }


def _gro_run_single(task):
    """Run NVT -> NPT -> production MD for a single pose.

    Uses ``cwd=`` on all subprocess calls instead of ``os.chdir()`` to
    be safe for parallel Pool workers.
    """
    from ....path_config import set_agent_data_path

    set_agent_data_path(task["agent_data_path"])
    logger = get_logger()

    pose_name = task["pose_name"]
    pose_dir = os.path.join(task["case_path"], pose_name)
    logger.info(f"MD worker: Starting simulation for pose {pose_name}")

    ndx_path = os.path.join(pose_dir, task["index"])
    validate_system_index_groups(ndx_path)

    if is_sigint_pending():
        return TaskResult(task_id=pose_name, success=False, error="SIGINT before start")

    ctx = GromacsContext.from_task_params(task["ctx"])
    if "ntmpi" in task and "ntomp" in task:
        ctx = ctx.with_overrides(ntmpi=task["ntmpi"], ntomp=task["ntomp"])

    mdp_dir = get_mdp_dir()
    stage_mdp_files = [mdp for _, mdp, _ in SOLUBLE_EQ_STAGES]
    copy_mdp_files(mdp_dir, stage_mdp_files + ["md.mdp"], pose_dir)
    md_mdp_path = os.path.join(pose_dir, "md.mdp")

    eq_scale = task.get("soluble_eq_nsteps_scale")
    if eq_scale is not None:
        eq_scale = float(eq_scale)
        applied_eq = _apply_soluble_eq_nsteps_scale(pose_dir, eq_scale)
        logger.info(
            "Applied soluble equilibration nsteps scale %.4f for %s (%s).",
            eq_scale,
            pose_name,
            ", ".join(
                f"{stage}:{values['original_nsteps']}->{values['scaled_nsteps']}"
                for stage, values in applied_eq.items()
            ),
        )

    time1 = time.perf_counter()

    for stage_name, mdp_file, input_gro in SOLUBLE_EQ_STAGES:
        if is_sigint_pending():
            return TaskResult(
                task_id=pose_name, success=False,
                error=f"SIGINT before {stage_name}",
            )
        ctx.run_stage(
            stage_name,
            mdp_file,
            input_gro,
            task["index"],
            task["topol"],
            cwd=pose_dir,
        )

    requested_dt_fs = float(task.get("production_dt_fs", 2.0))
    requested_nsteps = task.get("production_nsteps")
    if requested_dt_fs > 2.0:
        logger.warning(
            "Production timestep %.3f fs requested for %s. This is an advanced "
            "non-default mode; pipeline default is 2.0 fs and HMR is not enabled by default.",
            requested_dt_fs,
            pose_name,
        )
    applied = _apply_production_mdp_overrides(
        md_mdp_path,
        dt_fs=requested_dt_fs,
        nsteps=requested_nsteps,
    )
    used_dt_fs = requested_dt_fs
    fallback_used = False

    final_eq_stage = SOLUBLE_EQ_STAGES[-1][0]
    try:
        ctx.run_stage(
            task["md_str"],
            "md.mdp",
            f"{final_eq_stage}.gro",
            task["index"],
            task["topol"],
            cwd=pose_dir,
        )
    except Exception as exc:
        if requested_dt_fs > 2.0:
            logger.warning(
                "Production MD failed at dt=%.3f fs for %s (%s). Retrying once at dt=2.0 fs.",
                requested_dt_fs,
                pose_name,
                exc,
            )
            fallback_used = True
            used_dt_fs = 2.0
            applied = _apply_production_mdp_overrides(
                md_mdp_path,
                dt_fs=used_dt_fs,
                nsteps=requested_nsteps,
            )
            ctx.run_stage(
                task["md_str"],
                "md.mdp",
                f"{final_eq_stage}.gro",
                task["index"],
                task["topol"],
                cwd=pose_dir,
            )
        else:
            raise

    elapsed = time.perf_counter() - time1
    logger.info(f"MD worker: Completed pose {pose_name} in {elapsed:.2f}s")

    return TaskResult(
        task_id=pose_name,
        success=True,
        value={
            "production_dt_fs_requested": requested_dt_fs,
            "production_dt_fs_used": used_dt_fs,
            "production_nsteps_requested": requested_nsteps,
            "production_nsteps_used": applied.get("nsteps"),
            "soluble_eq_nsteps_scale_requested": eq_scale,
            "production_dt_fallback_used": fallback_used,
            "production_dt_ps_used": applied.get("dt_ps"),
            "elapsed_seconds": elapsed,
        },
    )
