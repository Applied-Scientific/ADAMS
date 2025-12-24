"""
GROMACS MD Simulation Module - MD Analysis Pipeline Step 3

This module runs GROMACS MD simulations for all prepared ligand-protein complexes.
It executes NVT equilibration, NPT equilibration, and production MD simulations
in parallel across multiple poses.

POSITION IN PIPELINE:
    Step 3 of 4: run_gro
    - Requires outputs from run_lig_prepare (prepared poses with min.gro)
    - Must be executed before run_stability_analysis
    - Can be skipped if MD simulations already completed

INPUTS (from file_paths dictionary):
    - poses_dir: Directory containing prepared pose subdirectories (with min.gro files)
    - gromacs_path: Path to GROMACS installation
    - ambertools_path: Path to AmberTools installation

OUTPUTS (added to file_paths dictionary):
    - poses_dir: Updated with MD-completed poses (each containing md.tpr, md.xtc, etc.)

KEY FUNCTIONALITY:
    - Runs NVT equilibration (constant volume, temperature coupling)
    - Runs NPT equilibration (constant pressure, temperature coupling)
    - Runs production MD simulation
    - Supports GPU acceleration (CUDA) and multi-rank MPI execution
    - Parallel execution across multiple poses using multiprocessing
    - Auto-calculates optimal MPI ranks and OpenMP threads

EXTERNAL COMMANDS:
    - gmx grompp: Prepare TPR run input files for each MD phase
    - gmx mdrun: Execute MD simulation (NVT → NPT → Production MD)
    - mpirun (optional): For multi-rank MPI execution on CPU

CONFIGURATION:
    - GPU mode: Uses CUDA-enabled GROMACS binary
    - CPU mode: Uses standard or MPI-enabled GROMACS binary
    - Auto-scales MPI ranks to GROMACS-friendly numbers (5-smooth numbers)
"""

import os
import shutil
import time
from bisect import bisect_right
from typing import List

from ...common_utils import get_gpu_count
from ...error_handling import is_sigint_pending, setup_sigint_handler
from ...logger_utils import (
    get_logger,
    log_step_execution,
    setup_multiprocessing_logging,
)
from ...utils import run_cmd
from ...utils.multiprocessing_utils import Pool, cleanup_pool, configure_worker_logging
from .utils import get_gromacs_binary, get_mdp_dir

# 5-smooth numbers (only prime factors 2, 3, 5) up to 256 - optimal for GROMACS domain decomposition
_GROMACS_FRIENDLY_RANKS = (
    1,
    2,
    3,
    4,
    5,
    6,
    8,
    9,
    10,
    12,
    15,
    16,
    18,
    20,
    24,
    25,
    27,
    30,
    32,
    36,
    40,
    45,
    48,
    50,
    54,
    60,
    64,
    72,
    75,
    80,
    81,
    90,
    96,
    100,
    108,
    120,
    125,
    128,
    135,
    144,
    150,
    160,
    162,
    180,
    192,
    200,
    216,
    225,
    240,
    243,
    250,
    256,
)


def _get_gromacs_friendly_ranks(n: int) -> int:
    """Find the largest 5-smooth number <= n for optimal GROMACS domain decomposition."""
    if n <= 1:
        return 1
    if n >= 256:
        return 256
    return _GROMACS_FRIENDLY_RANKS[bisect_right(_GROMACS_FRIENDLY_RANKS, n) - 1]


def _launch_gro(
    gmx_machine: str,
    run_on_gpu: bool,
    oversubscribe_str: str,
    ntmpi: int,
    ntomp: int,
    launch_name: str,
    input_mdp_filename: str,
    input_gro_filename: str,
    index_filename: str = "index.ndx",
    topol_filename: str = "topol.top",
    binary_type: str = "standard",
):
    """
    Launch GROMACS MD simulation phase (grompp + mdrun).

    Args:
        gmx_machine: Path to GROMACS binary
        run_on_gpu: Whether to use GPU acceleration
        oversubscribe_str: String for mpirun oversubscribe flag (if needed)
        ntmpi: Number of MPI ranks
        ntomp: Number of OpenMP threads
        launch_name: Base name for output files
        input_mdp_filename: Input MDP file
        input_gro_filename: Input GRO file
        index_filename: Index file name
        topol_filename: Topology file name
        binary_type: Type of GROMACS binary ("mpi", "cuda", or "standard")
    """
    logger = get_logger()

    # Run grompp to prepare TPR file
    grompp_cmd = [
        gmx_machine,
        "grompp",
        "-f",
        input_mdp_filename,
        "-c",
        input_gro_filename,
        "-r",
        input_gro_filename,
        "-n",
        index_filename,
        "-p",
        topol_filename,
        "-o",
        f"{launch_name}.tpr",
        "-po",
        f"{launch_name}_mdout.mdp",
        "-maxwarn",
        "100",
    ]
    run_cmd(grompp_cmd, check=True)

    # Adjust ntmpi to a GROMACS-friendly number (5-smooth) for better domain decomposition
    if ntmpi is not None and ntmpi > 1:
        friendly_ntmpi = _get_gromacs_friendly_ranks(ntmpi)
        if friendly_ntmpi != ntmpi:
            logger.info(
                f"Adjusted MPI ranks from {ntmpi} to {friendly_ntmpi} for optimal GROMACS domain decomposition"
            )
            ntmpi = friendly_ntmpi

    # Determine execution strategy based on binary type and configuration
    use_mpi = (
        (binary_type == "mpi")
        and (not run_on_gpu)
        and (ntmpi is not None and ntmpi > 1)
    )

    if run_on_gpu or binary_type == "cuda":
        # GPU/CUDA runs: use gmx with -ntmpi flag, no mpirun
        # CUDA builds require -ntmpi to be specified even if it's 1
        ntmpi_value = ntmpi if (ntmpi and ntmpi > 0) else 1
        mdrun_cmd = [
            gmx_machine,
            "mdrun",
            "-ntmpi",
            str(ntmpi_value),
            "-ntomp",
            str(ntomp),
            "-deffnm",
            launch_name,
        ]
        run_cmd(mdrun_cmd, check=True)
    elif use_mpi:
        # Multi-rank CPU runs with MPI: use mpirun with gmx_mpi
        mpirun_cmd = ["mpirun"]
        if oversubscribe_str.strip():
            mpirun_cmd.append("--oversubscribe")
        mpirun_cmd.extend(
            [
                "-np",
                str(ntmpi),
                gmx_machine,
                "mdrun",
                "-ntomp",
                str(ntomp),
                "-deffnm",
                launch_name,
            ]
        )
        run_cmd(mpirun_cmd, check=True)
    else:
        # Single-rank CPU runs or standard binary: use gmx directly without mpirun
        mdrun_cmd = [gmx_machine, "mdrun", "-ntomp", str(ntomp), "-deffnm", launch_name]
        run_cmd(mdrun_cmd, check=True)


class Gro:
    def __init__(
        self,
        file_paths,
        gpu: bool = False,
        num_gpus: int = 1,
        mpi_ranks: int = 0,
        omp_threads: int = 0,
        topol: str = "system.top",
        index: str = "index.ndx",
        max_jobs: int = 0,
    ):
        r"""
        Args:
            file_paths: dict: File paths dictionary - single source of truth (required).
                Must include:
                - poses_dir: Directory containing prepared pose subdirectories
                - gromacs_path: Path to GROMACS installation
                - ambertools_path: Path to AmberTools installation
                - gromacs_binary_type: Type of GROMACS binary (from discover_paths)
            gpu: bool: Whether to use GPU (default: False)
            num_gpus: int: Number of GPUs available (default: 1, used for GPU runs)
            mpi_ranks: int: Number of MPI ranks (0 = auto-calculate)
            omp_threads: int: Number of OpenMP threads (0 = auto-calculate)
            topol: str: Topology file name (default: "system.top")
            index: str: Index file name (default: "index.ndx")
            max_jobs: int: Maximum concurrent jobs (0 = auto-calculate)
        """
        self.logger = get_logger()

        # Set up multiprocessing logging queue (spawn-safe, eliminates fork deadlocks)
        # Note: Don't store queue in self when using Pool - it can't be pickled
        setup_multiprocessing_logging()

        # Set up SIGINT handler for clean shutdown on Ctrl+C
        setup_sigint_handler()

        if file_paths is None:
            raise ValueError(
                "file_paths dictionary is required. Use build_file_paths() and discover_paths() first."
            )
        self.file_paths = file_paths

        self.validate_files()

        self.gromacs_path = file_paths["gromacs_path"]
        self.ambertools_path = file_paths["ambertools_path"]
        base_binary_type = file_paths.get("gromacs_binary_type", "standard")
        if gpu:
            self.gromacs_binary_type = "cuda"
        else:
            self.gromacs_binary_type = base_binary_type
        self.gmx_binary = get_gromacs_binary(
            self.gromacs_path, binary_type=self.gromacs_binary_type, require_mpi=False
        )

        self.md_workdir = file_paths.get("md_root", ".")
        self.case_path = self.md_workdir

        self.gpu = gpu

        # Determine number of GPUs
        if self.gpu:
            if num_gpus == 0:
                self.num_gpus = get_gpu_count()
                self.logger.info(f"Auto-detected {self.num_gpus} GPUs.")
            else:
                self.num_gpus = num_gpus

            if self.num_gpus == 0:
                self.logger.warning(
                    "GPU run requested, but no GPUs were detected. Falling back to CPU."
                )
                self.gpu = False
        else:
            self.num_gpus = 0

        self.topol = topol
        self.index = index
        self.max_jobs = max_jobs

        available_cores = max(1, os.cpu_count() or 1)
        if mpi_ranks <= 0 or omp_threads <= 0:
            if self.gpu and self.num_gpus > 0:
                # GPU runs: 1 MPI rank per GPU, distribute CPU threads across GPUs
                self.ntmpi = self.num_gpus
                self.ntomp = max(1, available_cores // self.ntmpi)
                self.logger.info(
                    f"GPU mode: auto-configured ntmpi={self.ntmpi} (1 per GPU), ntomp={self.ntomp} (cores/GPUs)"
                )
            else:
                # CPU runs: maximize MPI ranks, 1 thread each
                self.ntmpi = available_cores
                self.ntomp = 1
                self.logger.info(
                    f"CPU mode: auto-configured ntmpi={self.ntmpi}, ntomp={self.ntomp}"
                )
        else:
            self.ntmpi = mpi_ranks
            self.ntomp = omp_threads
            self.logger.info(
                f"Using user-specified ntmpi={self.ntmpi}, ntomp={self.ntomp}"
            )

        # Initialize attributes that will be set in _prepwork() with safe defaults
        self.root_path = os.getcwd()
        self.max_mpi = 32
        self.init_str = "init"
        self.min_str = "min"
        self.nvt_str = "nvt"
        self.npt_str = "npt"
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
        poses_dir = self.file_paths.get("poses_dir")
        if not poses_dir:
            raise ValueError(
                "poses_dir required in file_paths.\n"
                f"Available keys: {list(self.file_paths.keys())}\n"
                "Ensure LigPrepare step has run or provide explicit poses_dir path."
            )

        if not self.file_paths.get("gromacs_path"):
            raise ValueError(
                "gromacs_path required in file_paths.\n"
                f"Available keys: {list(self.file_paths.keys())}\n"
                "Use discover_paths() to discover GROMACS and AmberTools paths."
            )

        if not self.file_paths.get("ambertools_path"):
            raise ValueError(
                "ambertools_path required in file_paths.\n"
                f"Available keys: {list(self.file_paths.keys())}\n"
                "Use discover_paths() to discover GROMACS and AmberTools paths."
            )

    def run(self) -> dict:
        """
        Run GROMACS MD simulations for all prepared poses.

        External commands called (via launch_gro for each phase):
            - gmx grompp: Prepare TPR run input files
            - gmx mdrun: Execute MD simulation (NVT → NPT → Production MD)
            - mpirun (optional): For multi-rank MPI execution on CPU

        Returns:
            dict: Updated file_paths dictionary with md_completed_poses list
        """
        step_logger = log_step_execution("MD Simulation", self.logger)
        with step_logger:
            folders = self._prepwork()

            self.logger.info(
                f"Starting MD simulations for {len(folders)} poses using {self.max_jobs} workers..."
            )

            pool = None
            try:
                # Get agent data path to pass to workers (required for logging setup in worker processes)
                from ...path_config import get_agent_data_path

                agent_data_path = get_agent_data_path()

                pool = Pool(processes=self.max_jobs)
                # Workers access log_queue via setup_multiprocessing_logging() - don't pass as arg
                pool.starmap(
                    self._gro_run, [(pose, agent_data_path) for pose in folders]
                )
                pool.close()
                pool.join()

                self.logger.info(f"MD simulations completed for all poses")

            except KeyboardInterrupt:
                self.logger.info("MD simulation interrupted by user (Ctrl+C)")
                if pool is not None:
                    # Try to close gracefully first
                    try:
                        pool.close()
                    except:
                        pass
                    # Then terminate any remaining workers
                    pool.terminate()
                    pool.join()
                self.logger.info("MD workers terminated, returning control to user")
                # Return file_paths to allow user to continue or restart
                return self.file_paths
            finally:
                # Explicit cleanup to prevent semaphore leaks
                cleanup_pool(pool, terminate=True, timeout=1.0)

            # poses_dir now contains MD-completed poses (same directory, now with MD outputs)
            # No need to update - poses_dir already points to the right place

            return self.file_paths

    def _gro_run(self, pose_name, agent_data_path):
        """
        Run MD simulation for a single pose with queue logging and error recovery.

        Args:
            pose_name: Name of the pose directory to process
            agent_data_path: Path: Agent data path for logging setup

        Returns:
            bool: True if successful, False if failed (per-pose error)
        """
        import subprocess

        # Set agent data path in worker process (required for logging)
        from ...path_config import set_agent_data_path

        set_agent_data_path(agent_data_path)

        # Configure worker logging - access global log queue
        log_queue = setup_multiprocessing_logging()
        configure_worker_logging(log_queue)
        logger = get_logger()
        logger.info(f"MD worker: Starting simulation for pose {pose_name}")

        # Check for SIGINT before starting work
        if is_sigint_pending():
            logger.info(f"MD worker: SIGINT detected for pose {pose_name}, exiting")
            return False

        # Use the appropriate GROMACS binary based on binary_type
        gmx_machine = self.gmx_binary

        run_on_gpu = self.gpu
        # Use MPI only when explicitly requested via mpi_ranks > 1 and not using GPU
        use_mpi = (not run_on_gpu) and (self.ntmpi is not None and self.ntmpi > 1)

        if self.ntmpi is not None and self.ntmpi >= self.max_mpi:
            oversubscribe_str = " --oversubscribe"
        else:
            oversubscribe_str = ""

        # Save original directory and change to pose folder
        original_dir = os.getcwd()
        try:
            os.chdir(f"{self.case_path}/{pose_name}")

            # Check if mdp files exist before copying
            mdp_dir = get_mdp_dir()
            mdp_files = ["nvt.mdp", "npt.mdp", "md.mdp"]
            for mdp_file in mdp_files:
                mdp_path = os.path.join(mdp_dir, mdp_file)
                if not os.path.exists(mdp_path):
                    logger.error(
                        f"MD worker: Required MDP file not found for pose {pose_name}: {mdp_path}"
                    )
                    return False
                shutil.copy(mdp_path, "./")

            time1 = time.perf_counter()

            # Check for SIGINT before NVT
            if is_sigint_pending():
                logger.info(
                    f"MD worker: SIGINT detected before NVT for pose {pose_name}, exiting"
                )
                return False

            # NVT equilibration
            try:
                _launch_gro(
                    gmx_machine,
                    run_on_gpu,
                    oversubscribe_str,
                    self.ntmpi,
                    self.ntomp,
                    self.nvt_str,
                    "nvt.mdp",
                    f"{self.min_str}.gro",
                    self.index,
                    self.topol,
                    binary_type=self.gromacs_binary_type,
                )
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"MD worker: NVT equilibration failed for pose {pose_name} (exit code {e.returncode})"
                )
                return False
            except Exception as e:
                logger.error(
                    f"MD worker: Unexpected error in NVT for pose {pose_name}: {e}"
                )
                return False

            # Check for SIGINT before NPT
            if is_sigint_pending():
                logger.info(
                    f"MD worker: SIGINT detected before NPT for pose {pose_name}, exiting"
                )
                return False

            # NPT equilibration
            try:
                _launch_gro(
                    gmx_machine,
                    run_on_gpu,
                    oversubscribe_str,
                    self.ntmpi,
                    self.ntomp,
                    self.npt_str,
                    "npt.mdp",
                    f"{self.nvt_str}.gro",
                    self.index,
                    self.topol,
                    binary_type=self.gromacs_binary_type,
                )
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"MD worker: NPT equilibration failed for pose {pose_name} (exit code {e.returncode})"
                )
                return False
            except Exception as e:
                logger.error(
                    f"MD worker: Unexpected error in NPT for pose {pose_name}: {e}"
                )
                return False

            # Check for SIGINT before production MD
            if is_sigint_pending():
                logger.info(
                    f"MD worker: SIGINT detected before production MD for pose {pose_name}, exiting"
                )
                return False

            # Production MD
            try:
                _launch_gro(
                    gmx_machine,
                    run_on_gpu,
                    oversubscribe_str,
                    self.ntmpi,
                    self.ntomp,
                    self.md_str,
                    "md.mdp",
                    f"{self.npt_str}.gro",
                    self.index,
                    self.topol,
                    binary_type=self.gromacs_binary_type,
                )
            except subprocess.CalledProcessError as e:
                logger.error(
                    f"MD worker: Production MD failed for pose {pose_name} (exit code {e.returncode})"
                )
                return False
            except Exception as e:
                logger.error(
                    f"MD worker: Unexpected error in production MD for pose {pose_name}: {e}"
                )
                return False

            time2 = time.perf_counter()
            t = time2 - time1
            logger.info(
                f"MD worker: Completed pose {pose_name} successfully in {t:.2f}s"
            )

            return True

        except Exception as e:
            logger.error(f"MD worker: Fatal error for pose {pose_name}: {e}")
            return False
        finally:
            # Always change back to original directory
            os.chdir(original_dir)

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
        for name in sorted(os.listdir(poses_dir)):
            pose_path = os.path.join(poses_dir, name)
            if os.path.isdir(pose_path):
                # Check if this is a prepared pose (has min.gro)
                if os.path.exists(os.path.join(pose_path, "min.gro")):
                    folders.append(name)

        self.logger.info(f"Found {len(folders)} prepared poses in {poses_dir}")
        self.logger.debug(f"Pose folders: {folders}")

        self.root_path = os.getcwd()
        self.logger.debug(f"root_path: {self.root_path}")

        # Input mdp stage names
        self.max_mpi = 32
        self.init_str = "init"
        self.min_str = "min"
        self.nvt_str = "nvt"
        self.npt_str = "npt"
        self.md_str = "md"

        # Calculate total_cpus and cpus_per_job regardless of max_jobs value
        total_cpus = os.cpu_count()
        cpus_per_job = self.ntmpi * self.ntomp

        if cpus_per_job == 0:
            raise ValueError(
                f"cpus_per_job cannot be 0 (ntmpi={self.ntmpi}, ntomp={self.ntomp})"
            )

        if self.max_jobs <= 0:
            self.max_jobs = max(1, total_cpus // cpus_per_job)
            self.logger.info(f"Auto-calculated max_jobs = {self.max_jobs}")
        else:
            self.logger.info(f"User override: max_jobs = {self.max_jobs}")

        self.logger.info(f"System has {total_cpus} CPUs")
        self.logger.info(
            f"Each job uses {cpus_per_job} CPUs ({self.ntmpi} MPI x {self.ntomp} OMP)"
        )
        self.logger.info(f"Scheduling up to {self.max_jobs} jobs in parallel")

        return folders
