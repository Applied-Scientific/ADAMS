"""
Vina (CPU) docking backend.

Parallelism: (ligand, grid) combinations are split into batches. Each batch
is one docking job, distributed across CPU worker processes.
"""

import itertools
import os
import sys
from typing import List, Union

import numpy as np
from vina import Vina

from ....common_utils import get_cpu_count
from ....error_handling import (
    PerLigandError,
    VinaExecutionError,
    is_sigint_pending,
)
from ....logger_utils import get_logger
from ..docking import BaseDockingPipeline
from ..utils import (
    SPACE_NAME_GRID,
    SPACE_NAME_POCKET,
    capture_stderr,
    get_docking_box_size,
    get_ligand_com_from_pdbqt_string,
    shift_pdbqt_to_center,
    tempstring,
)


class VinaBackend(BaseDockingPipeline):
    """CPU-based AutoDock Vina docking backend."""

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
        num_cores: int = None,
        auto_dock_num_cores: int = 1,
    ):
        if num_cores is None:
            num_cores = get_cpu_count()
        super().__init__(
            input_data=input_data,
            receptor=receptor,
            complex=complex,
            mode=mode,
            num_pockets=num_pockets,
            num_poses=num_poses,
            docking_centers=docking_centers,
            docking_centers_file=docking_centers_file,
            minimized_dock=minimized_dock,
            search_gridsize=search_gridsize,
            production_gridsize=production_gridsize,
            lock_grid_center=lock_grid_center,
            search_margin=search_margin,
            out_folder=out_folder,
            pH=pH,
            charge_model=charge_model,
        )
        self.num_cores = num_cores
        self.auto_dock_num_cores = auto_dock_num_cores

    # ------------------------------------------------------------------
    # Docking job API (required by BaseDockingPipeline)
    # ------------------------------------------------------------------

    def _get_worker_count(self):
        """One worker per CPU core (capped by the number of jobs)."""
        return self.num_cores

    def _build_docking_jobs(self):
        """
        Build batches of (ligand_idx, grid_idx) combinations.

        Each batch is one docking job dispatched to a worker process.
        """
        ligand_grid_combos = list(
            itertools.product(
                range(self.num_ligands), range(len(self.docking_centers))
            )
        )
        total_combos = len(ligand_grid_combos)
        if total_combos == 0:
            self.logger.warning(
                "No ligand-grid combinations to dock (num_ligands=%d, num_centers=%d).",
                self.num_ligands,
                len(self.docking_centers),
            )
            return []

        effective_num_cores = max(1, min(self.num_cores, total_combos))
        combo_batches = np.array_split(ligand_grid_combos, effective_num_cores)

        jobs = [batch.tolist() for batch in combo_batches if len(batch) > 0]

        self.logger.info(
            f"Starting docking for {total_combos} ligand-grid combinations "
            f"using {len(jobs)} workers..."
        )
        return jobs

    def _execute_docking_job(self, job, worker_id):
        """
        Dock one batch of (ligand_idx, grid_idx) combinations.

        Each worker creates a Vina instance, reads the required PDBQT files,
        and iterates through its assigned combinations.
        """
        batch = job
        logger = get_logger()

        print(
            f"[Worker {worker_id}] Process started, PID={os.getpid()}",
            file=sys.stderr,
            flush=True,
        )

        # Optional memory diagnostics
        try:
            import resource

            import psutil

            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            soft_limit, _ = resource.getrlimit(resource.RLIMIT_AS)
            limit_str = (
                f"{soft_limit / 1024 / 1024:.0f} MB"
                if soft_limit != resource.RLIM_INFINITY
                else "unlimited"
            )
            logger.info(
                f"Worker {worker_id}: Memory: {mem_mb:.1f} MB used, limit: {limit_str}"
            )
        except (ImportError, Exception):
            pass

        logger.info(f"Worker {worker_id}: Starting with {len(batch)} combinations")

        if len(batch) == 0:
            logger.warning(f"Worker {worker_id}: Received empty batch, skipping")
            return

        ligand_indices = [combo[0] for combo in batch]
        min_ligand_idx = min(ligand_indices)
        max_ligand_idx = max(ligand_indices)
        self._read_pdbqt_files(min_ligand_idx, max_ligand_idx)

        v = Vina(
            sf_name="vina",
            cpu=self.auto_dock_num_cores,
            no_refine=False,
            verbosity=0,
        )
        v.set_receptor(self.receptor)

        completed = 0
        failed = 0
        failed_ligands = []
        progress_interval = max(10, len(batch) // 20)

        for idx, combo in enumerate(batch):
            if is_sigint_pending():
                logger.info(f"Worker {worker_id}: SIGINT detected, exiting")
                return

            lig_idx = combo[0]
            grid_idx = combo[1]
            lig_string = self.ligands.get(lig_idx)

            if lig_string is not None:
                try:
                    self._dock_vina(
                        v,
                        lig_idx,
                        lig_string,
                        grid_idx,
                        self.docking_centers[grid_idx],
                    )
                    completed += 1
                except (VinaExecutionError, PerLigandError) as e:
                    lig_id = self.lignames.iloc[lig_idx]
                    logger.warning(
                        f"Worker {worker_id}: Skipping ligand {lig_idx} "
                        f"(ID: {lig_id}), grid {grid_idx}: {e}"
                    )
                    failed += 1
                    failed_ligands.append(lig_id)
                except SystemExit as e:
                    lig_id = self.lignames.iloc[lig_idx]
                    logger.error(
                        f"Worker {worker_id}: Vina exit({e.code}) for ligand "
                        f"{lig_idx} (ID: {lig_id}), grid {grid_idx}."
                    )
                    failed += 1
                    failed_ligands.append(lig_id)
                except Exception as e:
                    lig_id = self.lignames.iloc[lig_idx]
                    logger.error(
                        f"Worker {worker_id}: Unexpected error for ligand "
                        f"{lig_idx} (ID: {lig_id}), grid {grid_idx}: {e}"
                    )
                    failed += 1
                    failed_ligands.append(lig_id)
            else:
                logger.warning(
                    f"Worker {worker_id}: Ligand {lig_idx} is None, skipping"
                )
                failed += 1
                failed_ligands.append(self.lignames.iloc[lig_idx])

            if (idx + 1) % progress_interval == 0 or (idx + 1) == len(batch):
                progress_pct = 100 * (idx + 1) / len(batch)
                logger.info(
                    f"Worker {worker_id}: {idx + 1}/{len(batch)} "
                    f"({progress_pct:.1f}%) - Completed: {completed}, Failed: {failed}"
                )

        if failed > 0:
            logger.warning(
                f"Worker {worker_id}: FINISHED - {completed} completed, {failed} failed. "
                f"Failed ligands: {', '.join(set(failed_ligands))[:200]}..."
            )
        else:
            logger.info(
                f"Worker {worker_id}: FINISHED successfully - "
                f"{completed} completed, {failed} failed"
            )

    # ------------------------------------------------------------------
    # Vina-specific helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_outside_grid_error(exc: Exception) -> bool:
        """
        Return True when Vina reports a ligand outside the search box.
        """
        msg = str(exc).lower()
        return "outside the grid box" in msg or (
            "ligand is outside" in msg and "grid" in msg
        )

    @staticmethod
    def _pdbqt_max_span(pdbqt_string: str) -> float:
        """
        Return the maximum axis span (A) of ATOM/HETATM records in a PDBQT string.
        """
        xs, ys, zs = [], [], []
        for line in pdbqt_string.splitlines():
            if line.startswith(("ATOM", "HETATM")):
                try:
                    xs.append(float(line[30:38]))
                    ys.append(float(line[38:46]))
                    zs.append(float(line[46:54]))
                except ValueError:
                    continue
        if not xs:
            return 0.0
        return max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))

    @staticmethod
    def _expanded_box_size(box_size, required_span, margin=2.0):
        """
        Expand all box dimensions so each is at least required_span + margin.
        """
        required = max(0.0, float(required_span)) + float(margin)
        current = [float(v) for v in box_size]
        return [max(v, required) for v in current]

    def _read_pdbqt_files(self, start, stop):
        """Read PDBQT file contents for ligands in [start, stop] range."""
        self.ligands = {}
        logger = get_logger()
        for idx in range(start, stop + 1):
            pdbqt_path = self.ligand_pdbqt_files[idx]
            try:
                with open(pdbqt_path, "r") as f:
                    self.ligands[idx] = f.read()
            except Exception as e:
                logger.warning(
                    f"Failed to read PDBQT file {pdbqt_path} for ligand "
                    f"{self.lignames.iloc[idx]}: {e}"
                )
                self.ligands[idx] = None

    def _dock_vina(self, v, idx, lig_string, g_idx, center):
        """Set box size and mode-specific parameters, then delegate to _execute_docking."""
        box_size = get_docking_box_size(
            self.minimized_dock,
            self.mode,
            self.search_gridsize,
            production_gridsize=self.production_gridsize,
        )

        if self.minimized_dock:
            exhaustiveness = 8
            minimize_first = False
            space_name = SPACE_NAME_POCKET
        elif self.mode == "search":
            exhaustiveness = 32
            minimize_first = False
            space_name = SPACE_NAME_GRID
        else:
            exhaustiveness = 32
            minimize_first = True
            space_name = SPACE_NAME_POCKET

        self._execute_docking(
            v,
            lig_string,
            idx,
            g_idx,
            center,
            box_size,
            space_name,
            minimize_first,
            exhaustiveness,
            lock_grid_center=self.lock_grid_center,
        )

    def _execute_docking(
        self,
        v,
        lig_string,
        idx,
        g_idx,
        center,
        box_size,
        space_name,
        minimize_first=True,
        exhaustiveness=32,
        lock_grid_center=True,
    ):
        """
        Run a single Vina dock: set ligand, compute maps, dock, write poses.
        If Vina reports "ligand is outside the grid box", retry once with an
        expanded box derived from ligand span to avoid avoidable hard failures.
        """
        # Place ligand at grid center so it is inside the box (avoids Vina "ligand
        # is outside the grid box" when input coords are from another frame or origin).
        lig_string = shift_pdbqt_to_center(
            lig_string, center, source_center_mode="bbox"
        )
        try:
            self._run_single_docking_attempt(
                v=v,
                lig_string=lig_string,
                idx=idx,
                g_idx=g_idx,
                center=center,
                box_size=box_size,
                space_name=space_name,
                minimize_first=minimize_first,
                exhaustiveness=exhaustiveness,
                lock_grid_center=lock_grid_center,
            )
        except Exception as first_exc:
            if not self._is_outside_grid_error(first_exc):
                raise

            lig_span = self._pdbqt_max_span(lig_string)
            rescue_box_size = self._expanded_box_size(
                box_size=box_size,
                required_span=lig_span,
                margin=2.0,
            )
            if rescue_box_size == list(box_size):
                rescue_box_size = [float(v) + 2.0 for v in box_size]

            self.logger.warning(
                f"ligand {idx}, grid {g_idx}: Vina reported ligand outside box "
                f"with box={box_size}. Retrying once with expanded box="
                f"{rescue_box_size} (ligand_span={lig_span:.3f} A)."
            )
            self._run_single_docking_attempt(
                v=v,
                lig_string=lig_string,
                idx=idx,
                g_idx=g_idx,
                center=center,
                box_size=rescue_box_size,
                space_name=space_name,
                minimize_first=minimize_first,
                exhaustiveness=exhaustiveness,
                lock_grid_center=lock_grid_center,
            )

    def _run_single_docking_attempt(
        self,
        v,
        lig_string,
        idx,
        g_idx,
        center,
        box_size,
        space_name,
        minimize_first,
        exhaustiveness,
        lock_grid_center,
    ):
        """
        Execute one docking attempt for a ligand/grid pair with a fixed box size.
        """
        v.set_ligand_from_string(lig_string)
        with capture_stderr(self.logger, f"ligand {idx}, grid {g_idx}"):
            v.compute_vina_maps(center=center, box_size=box_size)

        if minimize_first:
            v.score()
            v.optimize()
            if lock_grid_center:
                self.logger.debug(
                    f"ligand {idx}, grid {g_idx}: keeping original docking center "
                    "after minimize (lock_grid_center=True)"
                )
            else:
                center = get_ligand_com_from_pdbqt_string(tempstring(v))
                with capture_stderr(
                    self.logger, f"ligand {idx}, grid {g_idx} (after minimize)"
                ):
                    v.compute_vina_maps(center=center, box_size=box_size)

        v.dock(exhaustiveness=exhaustiveness, n_poses=self.num_poses)
        pose_path = os.path.join(
            self.dir_structure["poses"],
            f"ligand_{idx}{space_name}{g_idx}_docked.pdbqt",
        )
        v.write_poses(pose_path, n_poses=self.num_poses, overwrite=True)
