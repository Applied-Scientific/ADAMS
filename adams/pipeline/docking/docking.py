"""
Unified docking pipeline: base class and single entry point with backend selection.

BaseDockingPipeline defines the template: preprocess -> build docking jobs -> run
jobs in parallel -> collect poses -> cleanup. Subclasses implement
_build_docking_jobs(), _execute_docking_job(), and _get_worker_count() to plug in
backend-specific logic.

DockingPipeline: use DockingPipeline(backend="vina"|"vina_gpu"|"unidock", **kwargs)
to run docking with the chosen engine. Backend-specific parameters are ignored
when not applicable.
"""

import inspect
import os
import queue
from typing import Any, List, Union

from ...error_handling import is_sigint_pending, setup_sigint_handler
from ...logger_utils import (
    get_logger,
    log_step_execution,
    setup_multiprocessing_logging,
)
from ...utils.multiprocessing_utils import (
    Process,
    Queue,
    can_start_multiprocessing,
    configure_worker_logging,
)
from ..file_organization import setup_docking_dirs
from .pose_collection import PoseCollectionMixin
from .preprocessing import preprocess


def _configure_mp_worker_thread_limits(logger):
    """
    Cap native intra-process thread pools for spawned workers.

    When many worker processes are launched, OpenMP/BLAS defaults can create
    dozens of threads per worker and cause oversubscription stalls. We only set
    defaults when the user has not already provided explicit overrides.
    """
    thread_env_defaults = {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
    }
    applied = []
    for key, value in thread_env_defaults.items():
        if not os.environ.get(key):
            os.environ[key] = value
            applied.append(f"{key}={value}")
    if applied:
        logger.info(
            "Configured multiprocessing worker thread limits: "
            + ", ".join(applied)
        )


class BaseDockingPipeline(PoseCollectionMixin):
    """
    Base class for docking pipelines. Holds common parameters, the run() template,
    and a unified parallel job runner.

    Subclasses must implement:
        _build_docking_jobs()                - Return a list of docking jobs for workers.
        _execute_docking_job(job, worker_id) - Run one job inside a worker process.
        _get_worker_count()                  - Return the max number of concurrent workers.

    The base _run_docking() calls _build_docking_jobs() then
    _run_docking_jobs_parallel().
    """

    def __init__(
        self,
        input_data: str,
        receptor: str,
        complex: str = None,
        mode: str = "production",
        num_pockets: int = 1,
        num_poses: int = 5,
        docking_centers: Union[List[float], None] = None,
        docking_centers_file: str = None,
        minimized_dock: bool = False,
        search_gridsize: float = 25.0,
        production_gridsize: float = None,
        lock_grid_center: bool = True,
        search_margin: float = 5.0,
        out_folder: str = "out_folder",
        pH: float = 7.4,
        charge_model: str = "gasteiger",
    ):
        if mode not in ["production", "search"]:
            raise ValueError(f"Mode must be 'production' or 'search', got: {mode}")

        self.input_data = input_data
        self.receptor = receptor
        self.complex = complex
        self.mode = mode
        self.num_poses = num_poses
        self.minimized_dock = minimized_dock
        self.search_gridsize = search_gridsize
        self.production_gridsize = production_gridsize
        if self.production_gridsize is not None and self.production_gridsize <= 0:
            raise ValueError(
                f"production_gridsize must be > 0 when provided, got: {self.production_gridsize}"
            )
        self.lock_grid_center = lock_grid_center
        self.search_margin = search_margin
        self.out_folder = out_folder
        self.docking_centers = docking_centers
        self.docking_centers_file = docking_centers_file
        self.num_pockets = num_pockets
        self.pH = pH
        self.charge_model = charge_model

        self.dir_structure = setup_docking_dirs(out_folder, mode=self.mode)
        self.logger = get_logger()
        setup_multiprocessing_logging()
        if not setup_sigint_handler():
            self.logger.info(
                "Running without custom SIGINT handler because execution is outside main thread."
            )

    # ------------------------------------------------------------------
    # Pipeline template
    # ------------------------------------------------------------------

    def run(self):
        """Run the docking pipeline: preprocess, dock, collect poses, optional cleanup."""
        step_name = f"Docking Inference for mode {self.mode}"
        step_logger = log_step_execution(step_name, self.logger)
        with step_logger:
            with step_logger.timing("preprocessing"):
                self._preprocess()
            with step_logger.timing("docking"):
                self._run_docking()
            with step_logger.timing("pose_collection"):
                output_csv = self._collectposes(
                    self.num_ligands,
                    len(self.docking_centers),
                    self.num_poses,
                )
            self._cleanup_after_run()
        return output_csv

    def _run_docking(self):
        """Build docking jobs and run them in parallel. Subclasses may override."""
        jobs = self._build_docking_jobs()
        self._run_docking_jobs_parallel(jobs)

    def _cleanup_after_run(self):
        """Override in subclasses (e.g. GPU backends) to remove temp dirs. Default no-op."""
        pass

    def _preprocess(self):
        """Validate inputs, load ligands, set docking centers. Delegates to preprocessing module."""
        preprocess(self)

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement
    # ------------------------------------------------------------------

    def _build_docking_jobs(self):
        """Return a list of docking jobs. Each job is passed to _execute_docking_job in a worker."""
        raise NotImplementedError("Subclasses must implement _build_docking_jobs()")

    def _execute_docking_job(self, job, worker_id):
        """Execute a single docking job inside a worker process. Logging is already configured."""
        raise NotImplementedError("Subclasses must implement _execute_docking_job()")

    def _get_worker_count(self):
        """Return the maximum number of concurrent worker processes."""
        raise NotImplementedError("Subclasses must implement _get_worker_count()")

    # ------------------------------------------------------------------
    # Unified parallel docking job runner
    # ------------------------------------------------------------------

    def _run_docking_jobs_parallel(self, jobs):
        """
        Run *jobs* across up to _get_worker_count() worker processes.

        Each worker pulls docking jobs from a shared queue and calls
        _execute_docking_job(). Handles graceful shutdown on KeyboardInterrupt
        and worker failure.
        """
        if not jobs:
            return

        num_workers = min(self._get_worker_count(), len(jobs))

        def _run_serial():
            """Execute docking jobs in-process without multiprocessing."""
            self.logger.info(
                f"Running {len(jobs)} docking jobs serially (1 worker)"
            )
            for job in jobs:
                if is_sigint_pending():
                    self.logger.info("SIGINT detected, stopping serial docking jobs")
                    return
                self._execute_docking_job(job, worker_id=0)

        multiprocessing_ok = can_start_multiprocessing()
        if num_workers <= 1 or not multiprocessing_ok:
            if num_workers > 1 and not multiprocessing_ok:
                self.logger.info(
                    "Multiprocessing is unavailable in this runtime context; "
                    "falling back to serial docking execution."
                )
            self.logger.info("Execution mode: serial")
            _run_serial()
            return

        job_queue = Queue()
        ready_queue = Queue()
        log_queue = setup_multiprocessing_logging()

        self.logger.info(
            f"Dispatching {len(jobs)} docking jobs to {num_workers} workers"
        )
        self.logger.info("Execution mode: multiprocessing")
        _configure_mp_worker_thread_limits(self.logger)

        procs = []
        try:
            for worker_id in range(num_workers):
                proc = Process(
                    target=self._docking_worker_loop,
                    args=(job_queue, worker_id, log_queue, ready_queue),
                )
                proc.start()
                procs.append(proc)

            ready_workers = set()
            while len(ready_workers) < num_workers:
                try:
                    ready_worker_id = ready_queue.get(timeout=30)
                except queue.Empty as exc:
                    raise RuntimeError(
                        "Timed out waiting for docking workers to initialize"
                    ) from exc
                ready_workers.add(ready_worker_id)

            # Enqueue only after all workers report ready so startup latency
            # does not let one worker drain the queue before peers are running.
            for job in jobs:
                job_queue.put(job)
            for _ in range(num_workers):
                job_queue.put(None)

            failed_workers = []
            for proc_idx, proc in enumerate(procs):
                proc.join()
                if proc.exitcode != 0:
                    failed_workers.append(proc_idx)
                    self.logger.error(
                        f"Worker {proc_idx} exited with code {proc.exitcode}"
                    )

            if failed_workers:
                self.logger.error(
                    f"Docking failed: {len(failed_workers)} worker(s) exited with errors."
                )
                raise RuntimeError(
                    f"{len(failed_workers)} docking worker(s) failed"
                )
            self.logger.info(f"All {num_workers} workers completed successfully")

        except KeyboardInterrupt:
            self.logger.info("Docking interrupted by user (Ctrl+C)")
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
            for proc in procs:
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
            self.logger.info(
                "Docking workers terminated, returning control to user"
            )
            raise
        except RuntimeError:
            raise
        except Exception as e:
            self.logger.error(
                f"Error in _run_docking_jobs_parallel(): {e}", exc_info=True
            )
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
            raise

    def _docking_worker_loop(self, job_queue, worker_id, log_queue, ready_queue):
        """
        Worker entry point: configure logging, then pull docking jobs from the
        queue and call _execute_docking_job() until a sentinel (None) is received.
        """
        configure_worker_logging(log_queue)
        logger = get_logger()
        logger.info(f"Worker {worker_id}: Started (PID={os.getpid()})")
        ready_queue.put(worker_id)

        while True:
            if is_sigint_pending():
                logger.info(f"Worker {worker_id}: SIGINT detected, exiting")
                return

            job = job_queue.get()
            if job is None:  # sentinel
                break

            try:
                self._execute_docking_job(job, worker_id)
            except Exception:
                logger.error(
                    "Worker %s crashed while executing job %s",
                    worker_id,
                    job,
                    exc_info=True,
                )
                raise

        logger.info(f"Worker {worker_id}: Finished")


from .backends import get_backend_class


class DockingPipeline:
    """
    Single entry point for molecular docking. Select the engine via backend=.

    Accepts the union of all parameters (common + backend-specific). When
    constructing the chosen backend, only parameters that backend's __init__
    accepts are passed; others are dropped.

    Common parameters: input_data, receptor, complex, mode, num_pockets,
    num_poses, docking_centers, docking_centers_file, minimized_dock,
    search_gridsize, search_margin, out_folder.

    Backend-specific (ignored when not applicable):
    - vina: num_cores, auto_dock_num_cores
    - vina_gpu: num_gpus, gpu_ids, gpu_threads, gpu_executable, gpu_opencl_binary_path
    - unidock: num_gpus, gpu_ids, gpu_executable, scoring, exhaustiveness,
      energy_range, min_rmsd, spacing, seed, refine_step, max_evals, max_step,
      max_gpu_memory, search_mode, verbosity, cpu
    """

    def __init__(self, backend: str, **kwargs: Any):
        if not backend:
            raise ValueError("backend must be non-empty")
        backend_class = get_backend_class(backend)
        if backend_class is None:
            raise ValueError(
                f"Unknown backend: {backend}. "
                f"Choose from: vina, vina_gpu, unidock"
            )
        sig = inspect.signature(backend_class.__init__)
        param_names = set(sig.parameters.keys()) - {"self"}
        filtered = {k: v for k, v in kwargs.items() if k in param_names}
        self._backend = backend_class(**filtered)

    def run(self) -> str:
        """Run the docking pipeline. Returns path to the output CSV."""
        return self._backend.run()
