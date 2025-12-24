"""
Description:
    AutoDock Vina Docking Pipeline
"""

import glob
import itertools
import os
import pickle
import re
import sys
from typing import List, Union

import numpy as np
import pandas as pd
from vina import Vina

from ...common_utils import get_cpu_count
from ...error_handling import (
    PerLigandError,
    VinaExecutionError,
    is_sigint_pending,
    setup_sigint_handler,
)
from ...logger_utils import (
    get_logger,
    log_step_execution,
    setup_multiprocessing_logging,
)
from ...utils.multiprocessing_utils import Process, configure_worker_logging
from ..file_organization import setup_docking_dirs
from .utils import (
    convert_receptor_to_pdbqt,
    generate_grid,
    get_all_model_centers,
    get_ligand_com_from_pdb,
    get_ligand_com_from_pdbqt_string,
    get_pdbqt_bounds,
    tempstring,
    write_all_centers_pdb,
)


class VinaDock:
    def __init__(
        self,
        input_data: str,
        receptor: str,
        complex: str = None,
        mode: str = "production",
        num_pockets: int = 1,
        num_poses: int = 5,
        docking_centers: Union[
            List[float],
        ] = None,
        docking_centers_file: str = None,
        minimized_dock: bool = False,
        search_gridsize: float = 25.0,
        search_margin: float = 5.0,
        auto_dock_num_cores: int = 1,
        out_folder: str = "out_folder",
        num_cores: int = get_cpu_count(),
    ):
        r"""
        Args:
            Required:
                input_data: str: input CSV file with ID and PDBQT_File columns
                receptor: str: receptor pdb or pdbqt file (will be converted to pdbqt if needed)

            Optional:
                complex: str: complex pdbqt/pdb file with ligand resname; e.g., complex.pdb,resname
                mode: str: docking mode: production (default) or search
                num_pockets: int: Number of docking centers (default: 1)
                num_poses: int: Number of poses to be calculated (default: 5)
                docking_centers: List[float]: List of docking centers as x y z (repeat for multiple pockets). Example: [0, 0, 0,  10, 10, 10,  -5, 5, 0]
                docking_centers_file: str: Path to CSV file containing docking centers (columns 2-4 are x,y,z). If provided, overrides docking_centers and num_pockets
                minimized_dock: bool: Use minimized docking (default: False)
                search_gridsize: float: grid size for docking over space target occupied (default: 25.0)
                search_margin: float: grid size for docking over space target occupied (default: 5.0)
                auto_dock_num_cores: int: Cores for Vina docking in each subprocess (default: 1)
                out_folder: str: Folder for storing results
                num_cores: int: Number of CPU cores (default: all-1)

        Note:
            All ligands must be pre-prepared as PDBQT files. The input CSV must contain
            'ID' and 'PDBQT_File' columns with paths to the PDBQT files.
        """
        self.logger = get_logger()

        self.log_queue = setup_multiprocessing_logging()

        # Set up SIGINT handler for clean shutdown on Ctrl+C
        setup_sigint_handler()

        self.input_data = input_data
        self.receptor = receptor
        self.complex = complex
        self.mode = mode

        self.num_poses = num_poses

        self.minimized_dock = minimized_dock
        self.search_margin = search_margin

        self.auto_dock_num_cores = auto_dock_num_cores
        self.out_folder = out_folder
        self.num_cores = num_cores

        self.docking_centers = docking_centers
        self.docking_centers_file = docking_centers_file
        self.num_pockets = num_pockets

        self.search_gridsize = search_gridsize

        # Set up file organization
        self.dir_structure = setup_docking_dirs(out_folder, mode=self.mode)

    def run(self):
        r"""
        Runs the docking pipeline.
        """
        step_name = f"Docking Inference for mode {self.mode}"
        step_logger = log_step_execution(step_name, self.logger)
        with step_logger:
            with step_logger.timing("preprocessing"):
                self._preprocess()
            with step_logger.timing("docking_computation"):
                self._run()
            with step_logger.timing("pose_collection"):
                output_csv = self._collectposes(
                    self.num_ligands, len(self.docking_centers), self.num_poses
                )
            return output_csv

    def _preprocess(self):
        r"""
        Preprocesses the input data.
        """
        if not os.path.exists(self.input_data):
            raise FileNotFoundError(f"Input data file not found: {self.input_data}")
        if not os.path.exists(self.receptor):
            raise FileNotFoundError(f"Receptor file not found: {self.receptor}")

        # Convert receptor to PDBQT if it's a PDB file
        if self.receptor.endswith(".pdb"):
            self.receptor = convert_receptor_to_pdbqt(self.receptor)
        elif not self.receptor.endswith(".pdbqt"):
            raise ValueError(
                f"Receptor file must be either .pdb or .pdbqt format, got: {self.receptor}"
            )

        os.makedirs(self.out_folder, exist_ok=True)

        # Read CSV with PDBQT file paths
        ligands = pd.read_csv(self.input_data)

        if "PDBQT_File" not in ligands.columns:
            raise ValueError(
                "Input CSV must contain 'PDBQT_File' column with paths to PDBQT files. "
                "All ligand preparation must be done in preprocessing module."
            )

        if "ID" not in ligands.columns:
            raise ValueError(
                "Input CSV must contain 'ID' column with ligand identifiers."
            )

        self.ligand_pdbqt_files = ligands["PDBQT_File"].tolist()
        self.lignames = ligands["ID"]
        self.num_ligands = len(self.ligand_pdbqt_files)

        # Validate PDBQT files exist
        for pdbqt_file in self.ligand_pdbqt_files:
            if not os.path.exists(pdbqt_file):
                raise FileNotFoundError(f"PDBQT file not found: {pdbqt_file}")

        self.logger.info(f"Loaded {self.num_ligands} PDBQT files for docking")

        # Calculate molecular weights from PDBQT files (needed for results output)
        self.logger.info("Calculating molecular weights from PDBQT files...")
        from .utils import get_molweight_from_pdbqt

        molweights = []
        for pdbqt_file in self.ligand_pdbqt_files:
            mw = get_molweight_from_pdbqt(pdbqt_file)
            molweights.append(mw)

        self.molweight = pd.Series(molweights)
        self.logger.info(
            f"Calculated molecular weights (range: {self.molweight.min():.1f} - {self.molweight.max():.1f} Da)"
        )

        self._set_docking_centers()

    def _set_docking_centers(self):

        # If docking_centers_file is provided, read it and override docking_centers and num_pockets
        if self.docking_centers_file and self.docking_centers_file.strip():
            if not os.path.exists(self.docking_centers_file):
                raise FileNotFoundError(
                    f"Docking centers file not found: {self.docking_centers_file}"
                )

            # Read the CSV file
            df_centers = pd.read_csv(self.docking_centers_file)

            # Intelligently detect coordinate columns based on column names
            # Support multiple formats:
            # 1. centroid_x, centroid_y, centroid_z (from clustering/search output)
            # 2. COM_x, COM_y, COM_z (from best_docking_centers.csv output)
            # 3. x, y, z (generic format)
            # 4. Fallback to columns 1, 2, 3 if no matching names found

            col_names = df_centers.columns.tolist()

            # Try to find coordinate columns by name
            x_col = y_col = z_col = None

            # Check for centroid_x/y/z
            if (
                "centroid_x" in col_names
                and "centroid_y" in col_names
                and "centroid_z" in col_names
            ):
                x_col, y_col, z_col = "centroid_x", "centroid_y", "centroid_z"
            # Check for COM_x/y/z
            elif "COM_x" in col_names and "COM_y" in col_names and "COM_z" in col_names:
                x_col, y_col, z_col = "COM_x", "COM_y", "COM_z"
            # Check for x/y/z
            elif "x" in col_names and "y" in col_names and "z" in col_names:
                x_col, y_col, z_col = "x", "y", "z"
            # Fallback to columns 1, 2, 3 (0-indexed)
            else:
                if df_centers.shape[1] < 4:
                    raise ValueError(
                        f"Could not find coordinate columns (centroid_x/y/z, COM_x/y/z, or x/y/z) "
                        f"and file has fewer than 4 columns for fallback: got {df_centers.shape[1]} columns"
                    )
                self.logger.warning(
                    f"Using columns 1-3 as coordinates. Column names: {col_names}"
                )
                centers_data = df_centers.iloc[:, 1:4].values
                self.docking_centers = [list(row) for row in centers_data]
                self.num_pockets = len(self.docking_centers)
                self.logger.info(
                    f"Read {self.num_pockets} docking centers from {self.docking_centers_file}"
                )
                return

            # Extract coordinates using column names
            centers_data = df_centers[[x_col, y_col, z_col]].values

            # Convert to list of [x, y, z] for each center
            self.docking_centers = [list(row) for row in centers_data]
            self.num_pockets = len(self.docking_centers)

            self.logger.info(
                f"Read {self.num_pockets} docking centers from {self.docking_centers_file} using columns: {x_col}, {y_col}, {z_col}"
            )
            return

        if self.mode == "production":
            if self.complex:
                try:
                    parts = self.complex.split(",")
                    complex_pdb = parts[0].strip()
                    lig_resnames = parts[1:]
                    # Check if number of ligands is less than required
                    if len(lig_resnames) < self.num_pockets:
                        raise ValueError(
                            f"Not enough ligands provided: expected at least {self.num_pockets}, got {len(lig_resnames)}"
                        )

                    # Limit lig_resnames to the first num_pockets ligands
                    if len(lig_resnames) > self.num_pockets:
                        lig_resnames = lig_resnames[: self.num_pockets]

                    self.logger.info(f"PDB file: {complex_pdb}")
                    self.logger.info(
                        f"Input {len(lig_resnames)} ligands: {lig_resnames}"
                    )
                except ValueError:
                    raise ValueError(
                        "Invalid format for --complex. Use: 'filename,resname', e.g., complex.pdb,LIG"
                    )
                self.docking_centers = []
                for i in range(self.num_pockets):
                    self.docking_centers.append(
                        get_ligand_com_from_pdb(complex_pdb, lig_resnames[i])
                    )

            else:
                if self.docking_centers is None:
                    self.docking_centers = [[0.0, 0.0, 0.0]] * self.num_pockets
                else:
                    if len(self.docking_centers) != self.num_pockets * 3:
                        raise ValueError(
                            f"--docking_location requires {self.num_pockets * 3} floats "
                            f"({self.num_pockets} pockets), but got {len(self.docking_centers)})"
                        )
                    self.docking_centers = [
                        self.docking_centers[i : i + 3]
                        for i in range(0, len(self.docking_centers), 3)
                    ]

        elif self.mode == "search":
            boundary = get_pdbqt_bounds(self.receptor)
            self.docking_centers = generate_grid(
                boundary, box_size=self.search_gridsize, margin=self.search_margin
            )

            self.logger.info(
                f"Search dock over {len(self.docking_centers)} grid centers..."
            )

        self.logger.debug(f"Docking centers: {self.docking_centers}")

    def _read_pdbqt_files(self, start, stop):
        r"""
        Read pre-prepared PDBQT files as strings.

        Args:
            start: Starting ligand index
            stop: Ending ligand index (inclusive)
        """
        logger = get_logger()
        self.ligands = {}

        print(
            f" Reading PDBQT files ({start}-{stop})",
            flush=True,
        )

        for idx in range(start, stop + 1):
            pdbqt_path = self.ligand_pdbqt_files[idx]
            try:
                with open(pdbqt_path, "r") as f:
                    self.ligands[idx] = f.read()  # Store PDBQT string
            except Exception as e:
                logger.warning(
                    f"Failed to read PDBQT file {pdbqt_path} for ligand {self.lignames.iloc[idx]}: {e}"
                )
                self.ligands[idx] = None

    def _run(self):
        # batches are likely to have smaller ranges of ligands
        ligand_grid_combos = list(
            itertools.product(range(self.num_ligands), range(len(self.docking_centers)))
        )
        total_combos = len(ligand_grid_combos)

        # Limit num_cores to not exceed the number of combinations to avoid empty batches
        effective_num_cores = min(self.num_cores, len(ligand_grid_combos))
        combo_batches = np.array_split(ligand_grid_combos, effective_num_cores)

        self.logger.info(
            f"Starting docking for {total_combos} ligand-grid combinations using {effective_num_cores} workers..."
        )

        procs = []

        try:
            for batch_idx, batch in enumerate(combo_batches):

                if len(batch) > 0:
                    proc = Process(
                        target=self._autodock_worker,
                        args=(batch, batch_idx, self.log_queue),
                    )
                    procs.append(proc)
                    proc.start()
                    self.logger.info(
                        f"Started worker {batch_idx} with {len(batch)} ligand-grid combinations"
                    )

            # Wait for all processes with timeout monitoring
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
                    f"Docking failed: {len(failed_workers)} worker(s) exited with errors. Check logs above for details."
                )
                raise RuntimeError(f"{len(failed_workers)} docking worker(s) failed")
            else:
                self.logger.info(f"All {len(procs)} workers completed successfully")

        except KeyboardInterrupt:
            self.logger.info("Docking interrupted by user (Ctrl+C)")
            # Terminate all worker processes
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
            # Wait for all to finish with timeout
            for proc in procs:
                proc.join(timeout=2)
                if proc.is_alive():
                    # Force kill if still alive after timeout
                    proc.kill()
                    proc.join()
            self.logger.info("Docking workers terminated, returning control to user")
            raise  # Re-raise to propagate to outer try-except if needed

        except Exception as e:
            self.logger.error(f"Error in _run(): {e}", exc_info=True)
            # Terminate any remaining processes
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
            raise

    def _autodock_worker(self, batch, worker_id, log_queue):
        r"""
        Runs the docking pipeline for a given mode.

        Args:
            batch: List of (ligand_idx, grid_idx) tuples
            worker_id: int: Worker process ID for logging
            log_queue: multiprocessing.Queue: Queue for sending log records to main process

        NOTE ON C++ CRASHES:
            C++ library crashes (segfaults in RDKit/Vina) CANNOT be caught by Python
            and will cause exit code 1 (or 128+signum). These are system-level issues
            that require fixing the input data or increasing resources.
        """
        # Print to stderr first (before logging is configured) for debugging
        print(
            f"[Worker {worker_id}] Process started, PID={os.getpid()}",
            file=sys.stderr,
            flush=True,
        )

        # Configure worker logging
        configure_worker_logging(log_queue)
        logger = get_logger()
        logger.info(f"Worker {worker_id}: Logging configured successfully")

        # Check memory limits and current usage
        try:
            import resource

            import psutil

            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            limit_str = (
                f"{soft_limit / 1024 / 1024:.0f} MB"
                if soft_limit != resource.RLIM_INFINITY
                else "unlimited"
            )
            logger.info(
                f"Worker {worker_id}: Memory: {mem_mb:.1f} MB used, limit: {limit_str}"
            )
        except ImportError:
            logger.debug(
                f"Worker {worker_id}: psutil not available for memory monitoring"
            )
        except Exception as e:
            logger.debug(f"Worker {worker_id}: Could not check memory: {e}")

        logger.info(f"Worker {worker_id}: Starting with {len(batch)} combinations")
        print(
            f"Worker {worker_id}: Starting with {len(batch)} combinations", flush=True
        )

        if len(batch) == 0:
            logger.warning(f"Worker {worker_id}: Received empty batch, skipping")
            return

        ligand_indices = [lig_grid[0] for lig_grid in batch]
        min_ligand_idx = min(ligand_indices)
        max_ligand_idx = max(ligand_indices)

        print(
            f"Worker {worker_id}: Processing {len(batch)} ligand-grid combinations",
            flush=True,
        )
        print(
            f"Worker {worker_id}: Reading PDBQT files {min_ligand_idx}-{max_ligand_idx}...",
            flush=True,
        )

        self._read_pdbqt_files(min_ligand_idx, max_ligand_idx)

        # Check memory after reading molecules
        try:
            import psutil

            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            mem_mb = mem_info.rss / 1024 / 1024
            if mem_mb > 2000:
                logger.warning(
                    f"Worker {worker_id}: High memory usage: {mem_mb:.1f} MB after reading molecules"
                )
        except (ImportError, Exception):
            pass

        v = Vina(
            sf_name="vina", cpu=self.auto_dock_num_cores, no_refine=False, verbosity=0
        )
        v.set_receptor(self.receptor)

        completed = 0
        failed = 0
        failed_ligands = []
        progress_interval = max(10, len(batch) // 20)

        for idx, lig_grid in enumerate(batch):
            if is_sigint_pending():
                logger.info(f"Worker {worker_id}: SIGINT detected, exiting")
                return

            lig_idx = lig_grid[0]
            grid_idx = lig_grid[1]
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
                        f"Worker {worker_id}: Skipping ligand {lig_idx} (ID: {lig_id}), grid {grid_idx}: {e}"
                    )
                    failed += 1
                    failed_ligands.append(lig_id)
                    continue
                except SystemExit as e:
                    lig_id = self.lignames.iloc[lig_idx]
                    logger.error(
                        f"Worker {worker_id}: Vina exit({e.code}) for ligand {lig_idx} (ID: {lig_id}), grid {grid_idx}. Likely missing affinity map for unusual atom type."
                    )
                    failed += 1
                    failed_ligands.append(lig_id)
                    continue
                except Exception as e:
                    lig_id = self.lignames.iloc[lig_idx]
                    logger.error(
                        f"Worker {worker_id}: Unexpected error for ligand {lig_idx} (ID: {lig_id}), grid {grid_idx}: {e}"
                    )
                    failed += 1
                    failed_ligands.append(lig_id)
                    continue
            else:
                logger.warning(
                    f"Worker {worker_id}: Ligand {lig_idx} is None, skipping"
                )
                failed += 1
                failed_ligands.append(self.lignames.iloc[lig_idx])

            if (idx + 1) % progress_interval == 0 or (idx + 1) == len(batch):
                progress_pct = 100 * (idx + 1) / len(batch)
                logger.info(
                    f"Worker {worker_id}: {idx + 1}/{len(batch)} ({progress_pct:.1f}%) - Completed: {completed}, Failed: {failed}"
                )

        if failed > 0:
            logger.warning(
                f"Worker {worker_id}: FINISHED - {completed} completed, {failed} failed. Failed ligands: {', '.join(set(failed_ligands))[:200]}..."
            )
        else:
            logger.info(
                f"Worker {worker_id}: FINISHED successfully - {completed} completed, {failed} failed"
            )

    # ----------------------------------------------------------------------
    # Docking Functions
    # ----------------------------------------------------------------------

    def _dock_vina(self, v, idx, lig_string, g_idx, center):
        """
        Run AutoDock Vina docking for a single ligand using pre-prepared PDBQT.
        Args:
            v: Vina: Pre-created Vina object
            idx: int: Index of the ligand
            lig_string: str: PDBQT string (already prepared)
            g_idx: int: Index of the grid
            center: List[float]: Center of the grid
        """
        if self.minimized_dock:
            box_size = [5, 5, 5]
            exhaustiveness = 8
            minimize_first = False
            space_name = "_pocket_"
        elif self.mode == "search":
            box_size = [self.search_gridsize] * 3
            exhaustiveness = 32
            minimize_first = False
            space_name = "_grid_"
        else:
            # Use default box size (can be refined based on ligand properties if needed)
            box_size = [20, 20, 20]
            exhaustiveness = 32
            minimize_first = True
            space_name = "_pocket_"

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
    ):
        """
        Dock pre-prepared PDBQT ligand with optional minimization.
        Args:
            v: Vina: Pre-created Vina object
            lig_string: str: PDBQT string (already prepared)
            idx: int: Index of the ligand
            g_idx: int: Index of the grid
            center: List[float]: Center of the grid
            box_size: List[float]: Box size
            space_name: str: _pocket_ or _grid_
            minimize_first: bool: Whether to minimize the ligand
            exhaustiveness: int: Exhaustiveness
        """
        logger = get_logger()
        ligand_id = self.lignames.iloc[idx]

        # NO Meeko preparation - ligand is already PDBQT
        v.set_ligand_from_string(lig_string)
        v.compute_vina_maps(center=center, box_size=box_size)

        if minimize_first:
            v.score()
            v.optimize()
            center = get_ligand_com_from_pdbqt_string(tempstring(v))
            v.compute_vina_maps(center=center, box_size=box_size)

        v.dock(exhaustiveness=exhaustiveness, n_poses=self.num_poses)
        pose_path = os.path.join(
            self.dir_structure["poses"], f"ligand_{idx}{space_name}{g_idx}_docked.pdbqt"
        )
        v.write_poses(pose_path, n_poses=self.num_poses, overwrite=True)

    # ----------------------------------------------------------------------
    # Pose Collection
    # ----------------------------------------------------------------------
    def _collectposes(self, n_lig, n_grid, n_pose):
        """
        Collect docking poses and save all results in a single .pkl file
        as a 4D NumPy array:
        model_data[feature, ligand_id, grid_id, pose_id]

        Features:
        0 = affinity
        1 = COM_x
        2 = COM_y
        3 = COM_z
        4 = RMSD_lb
        5 = RMSD_ub

        If a ligand-grid pair has fewer poses than n_pose,
        remaining entries are set to 0.

        Args:
            n_lig: int: Number of ligands
            n_grid: int: Number of grids
            n_pose: int: Number of poses
        """

        # Pre-allocate with zeros
        model_data = np.zeros((6, n_lig, n_grid, n_pose), dtype=np.float32)
        # Search for pose files in organized directory
        if self.mode == "search":
            filePaths = glob.glob(
                os.path.join(
                    self.dir_structure["poses"], "ligand_*_grid_*_docked.pdbqt"
                )
            )
        else:
            filePaths = glob.glob(
                os.path.join(
                    self.dir_structure["poses"], "ligand_*_pocket_*_docked.pdbqt"
                )
            )

        for filePath in filePaths:
            if self.mode == "search":
                matches = re.search(r"ligand_(\d+)_grid_(\d+)_docked.pdbqt", filePath)
            else:
                matches = re.search(r"ligand_(\d+)_pocket_(\d+)_docked.pdbqt", filePath)

            if not matches:
                continue
            lig_id, g_id = map(int, matches.groups())

            with open(filePath, "r") as f:
                lines = [
                    line.strip().split()
                    for line in f
                    if line.startswith("REMARK VINA RESULT:")
                ]
            lines = [
                [float(val) for val in line[3:]] for line in lines
            ]  # [affinity, RMSD_lb, RMSD_ub]

            model_centers = get_all_model_centers(filePath)

            for pose_id in range(min(n_pose, len(lines))):
                affinity, rmsd_lb, rmsd_ub = lines[pose_id]
                com = model_centers[pose_id]
                model_data[0, lig_id, g_id, pose_id] = affinity
                model_data[1:4, lig_id, g_id, pose_id] = com
                model_data[4, lig_id, g_id, pose_id] = rmsd_lb
                model_data[5, lig_id, g_id, pose_id] = rmsd_ub

        # Save metadata to organized directory
        pkl_file = os.path.join(self.dir_structure["metadata"], "dock_metadata.pkl")
        with open(pkl_file, "wb") as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(
            f"Saved model_data with shape {model_data.shape} to {self.dir_structure['metadata']}"
        )

        if self.mode == "search":
            output_csv = os.path.join(
                self.dir_structure["summaries"], "best_search_docking_centers.csv"
            )
            df_best = self._save_best_search_docking_poses(
                pkl_file, output_csv, selecttype="topN", topN=100
            )
            write_all_centers_pdb(
                df_best,
                os.path.join(
                    self.dir_structure["summaries"], "best_search_docking_centers.pdb"
                ),
            )

            output_csv = os.path.join(
                self.dir_structure["summaries"], "best_docking_centers.csv"
            )
            df_best = self._save_best_search_docking_poses(
                pkl_file, output_csv, selecttype="bestPerGrid"
            )
        else:

            output_csv = os.path.join(
                self.dir_structure["summaries"], "production_docking_results.csv"
            )
            df_best = self._save_best_search_docking_poses(
                pkl_file, output_csv, selecttype="bestPerGrid"
            )

        return output_csv

    def _save_best_search_docking_poses(
        self, pkl_file, output_csv, selecttype="topN", topN=10, threshold_fraction=0.1
    ):
        """
        Select best docking poses from search docking results.

        Parameters
        ----------
        pkl_file : str
            Path to pickle file with docking results
        output_csv : str
            Output CSV file
        selecttype : str
            'topN'          → global top N poses
            'percentile'    → all poses under percentile cutoff
            'bestPerGrid'   → best (lowest affinity) pose per ligand per grid
        topN : int
            Number of best poses if selecttype='topN'
        threshold_fraction : float
            Fraction (0–1) for percentile selection if selecttype='percentile'
        """
        # Load results
        with open(pkl_file, "rb") as f:
            results = pickle.load(f)
            # results shape: [6, num_ligands, num_grids, num_poses]
            # [0: affinity, 1-3: COM(x,y,z), 4: RMSD_lb, 5: RMSD_ub]

        affinities = results[0]  # (num_ligands, num_grids, num_poses)
        coms = results[1:4]  # (3, num_ligands, num_grids, num_poses)

        flat_affinities = affinities.flatten()
        nonzero_affinities = flat_affinities[flat_affinities != 0]

        cutoff = None
        selected_indices = set()

        # --- Selection strategy ---
        if selecttype == "topN":
            if len(nonzero_affinities) == 0:
                self.logger.warning("No valid affinities found.")
                return pd.DataFrame()
            flat_indices = np.argsort(flat_affinities)[:topN]
            selected_indices = set(flat_indices)

        elif selecttype == "percentile":
            if len(nonzero_affinities) == 0:
                self.logger.warning("No valid affinities found.")
                return pd.DataFrame()
            cutoff = np.percentile(nonzero_affinities, threshold_fraction * 100)

        elif selecttype == "bestPerGrid":
            pass

        else:
            raise ValueError(
                "selecttype must be 'topN', 'percentile', or 'bestPerGrid'"
            )

        # --- Collect best results ---
        best_records = []
        num_ligands, num_grids, num_poses = affinities.shape

        if selecttype == "bestPerGrid":
            for lig in range(num_ligands):
                for g in range(num_grids):
                    affs = affinities[lig, g, :]
                    valid_mask = affs != 0
                    if not np.any(valid_mask):
                        self.logger.debug(
                            f"[Skip] Ligand {lig}: {self.lignames.iloc[lig]}, Grid {g}: no valid affinities"
                        )
                        continue
                    best_p = np.argmin(affs[valid_mask])
                    pose_indices = np.where(valid_mask)[0]
                    p = pose_indices[best_p]
                    aff = affinities[lig, g, p]
                    record = {
                        "ligand_id": lig,
                        "ID": self.lignames.iloc[lig],
                        "grid_id": g,
                        "pose_id": p,
                        "affinity": aff,
                        "COM_x": coms[0, lig, g, p],
                        "COM_y": coms[1, lig, g, p],
                        "COM_z": coms[2, lig, g, p],
                        "MolWt": self.molweight.iloc[lig],
                    }
                    best_records.append(record)
        else:
            for lig in range(num_ligands):
                for g in range(num_grids):
                    for p in range(num_poses):
                        aff = affinities[lig, g, p]
                        if aff == 0:
                            continue
                        flat_idx = (lig * num_grids * num_poses) + (g * num_poses) + p
                        if (selecttype == "topN" and flat_idx in selected_indices) or (
                            selecttype == "percentile" and aff <= cutoff
                        ):
                            record = {
                                "ligand_id": lig,
                                "grid_id": g,
                                "pose_id": p,
                                "affinity": aff,
                                "COM_x": coms[0, lig, g, p],
                                "COM_y": coms[1, lig, g, p],
                                "COM_z": coms[2, lig, g, p],
                            }
                            best_records.append(record)

        # Save results
        df_best = pd.DataFrame(best_records)
        if not df_best.empty:
            df_best.sort_values(by="affinity", inplace=True)
            df_best.to_csv(output_csv, index=False)

            if selecttype == "topN":
                self.logger.info(f"Saved top {topN} poses to {output_csv}")
            elif selecttype == "percentile":
                self.logger.info(
                    f"Saved best {threshold_fraction*100:.0f}% poses to {output_csv}"
                )
            elif selecttype == "bestPerGrid":
                self.logger.info(
                    f"Saved best (lowest) pose per ligand per grid to {output_csv}"
                )
        else:
            self.logger.warning(
                f"No valid docking results found. Output CSV not created: {output_csv}"
            )

        return df_best
