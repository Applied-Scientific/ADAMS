"""
Description:
    AutoDock Vina GPU Docking Pipeline - Wrapper for GPU executable
"""

import glob
import os
import pickle
import re
import shutil
import subprocess
from typing import List, Union

import numpy as np
import pandas as pd

from ...error_handling import is_sigint_pending, setup_sigint_handler
from ...logger_utils import (
    get_logger,
    log_step_execution,
    setup_multiprocessing_logging,
)
from ...utils import run_cmd
from ...utils.multiprocessing_utils import (
    Process,
    Queue,
    cleanup_process,
    configure_worker_logging,
    cpu_count,
)
from ..file_organization import setup_docking_dirs
from .utils import (
    convert_receptor_to_pdbqt,
    flexible_box_size,
    generate_grid,
    get_all_model_centers,
    get_ligand_com_from_pdb,
    get_pdbqt_bounds,
    write_all_centers_pdb,
)


class VinaDockGPU:
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
        out_folder: str = "out_folder",
        num_gpus: int = 1,
        gpu_ids: List[int] = None,
    ):
        r"""
        Args:
            Required:
                input_data: str: input CSV file with ID and PDBQT_File columns
                receptor: str: receptor pdb or pdbqt file (will be converted to pdbqt if needed)

            Optional:
                complex: str: complex pdbqt/pdb file with ligand resname; e.g., complex.pdb,resname
                mode: str: docking mode: "production" or "search" (default: production)
                num_pockets: int: Number of docking centers (default: 1)
                num_poses: int: Number of poses to be calculated (default: 5)
                docking_centers: List[float]: List of docking centers as x y z (repeat for multiple pockets)
                docking_centers_file: str: Path to CSV file containing docking centers (columns 2-4 are x,y,z)
                minimized_dock: bool: Use minimized docking with small 5Å box (default: False).
                    Note: For GPU, this uses a fixed 5Å box for all ligands. Only use if you have
                    very precise binding sites and all ligands are small (<300 Da). For production
                    docking after search, typically use False to allow flexible box sizing.
                search_gridsize: float: Grid spacing for search mode OR manual box size for production (default: 25.0).
                    - In search mode: Spacing between grid points for exhaustive search across receptor
                    - In production mode: If provided, uses this fixed box size [search_gridsize, search_gridsize, search_gridsize]
                      for all ligands, regardless of molecular weight. If None, box size is determined
                      automatically from molecular weights (requires MolWt column or SMILES for calculation).
                      Example: 25.0 creates a 25×25×25 Å box.
                search_margin: float: Margin around receptor bounds for search mode (default: 5.0)
                out_folder: str: Folder for storing results
                num_gpus: int: Number of GPUs to use for parallel docking (default: 1)
                gpu_ids: List[int]: List of GPU device IDs to use (e.g., [0, 1, 2, 3]).
                    If None, uses GPUs 0 to num_gpus-1. Default: None
        Note:
            All ligands must be pre-prepared as PDBQT files. The input CSV must contain
            'ID' and 'PDBQT_File' columns with paths to the PDBQT files.
        """

        # Validate mode
        if mode not in ["production", "search"]:
            raise ValueError(f"Mode must be 'production' or 'search', got: {mode}")

        self.input_data = input_data
        self.receptor = receptor
        self.complex = complex
        self.mode = mode

        self.num_poses = num_poses

        self.minimized_dock = minimized_dock
        self.search_gridsize = search_gridsize
        self.search_margin = search_margin

        self.out_folder = out_folder

        self.docking_centers = docking_centers
        self.docking_centers_file = docking_centers_file
        self.num_pockets = num_pockets

        # GPU parallelization parameters
        self.num_gpus = num_gpus
        if num_gpus == 0:
            raise ValueError(f"Number of GPUs must be greater than 0, got: {num_gpus}")
        if gpu_ids is None:
            self.gpu_ids = list(range(num_gpus))  # Default: use GPUs 0 to num_gpus-1
        else:
            self.gpu_ids = gpu_ids
            if len(self.gpu_ids) != num_gpus:
                raise ValueError(
                    f"Number of GPU IDs ({len(self.gpu_ids)}) must match num_gpus ({num_gpus})"
                )

        vina_gpu_dir = os.path.join(os.path.dirname(__file__), "vina_gpu")
        self.gpu_executable = os.path.join(vina_gpu_dir, "AutoDock-Vina-GPU-2-1")
        self.gpu_opencl_binary_path = vina_gpu_dir
        self.gpu_threads = 8000  # Default GPU computing lanes

        # Validate GPU executable exists
        if not os.path.exists(self.gpu_executable):
            raise FileNotFoundError(
                f"GPU executable not found: {self.gpu_executable}\n"
                f"Expected location: {vina_gpu_dir}/AutoDock-Vina-GPU-2-1"
            )
        if not os.access(self.gpu_executable, os.X_OK):
            raise PermissionError(
                f"GPU executable is not executable: {self.gpu_executable}\n"
                f"Please run: chmod +x {self.gpu_executable}"
            )

        self.dir_structure = setup_docking_dirs(out_folder, mode=self.mode)

        # Set up logger
        self.logger = get_logger()

        setup_multiprocessing_logging()

        # Set up SIGINT handler for clean shutdown on Ctrl+C
        setup_sigint_handler()

    def run(self):
        r"""
        Runs the docking pipeline.
        """
        print("--- Using Vina Dock GPU ---")
        step_name = f"GPU Docking Inference for mode {self.mode}"
        step_logger = log_step_execution(step_name, self.logger)
        with step_logger:
            with step_logger.timing("preprocessing"):
                self._preprocess()
            with step_logger.timing("gpu_docking"):
                self._run()
            with step_logger.timing("pose_collection"):
                output_csv = self._collectposes(
                    self.num_ligands, len(self.docking_centers), self.num_poses
                )

            # Clean up any remaining temporary directories
            temp_dir = os.path.join(self.dir_structure["poses"], "temp")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)
                self.logger.info(f"Cleaned up temporary directory: {temp_dir}")

            # Clean up temp_chunks directory if it exists
            temp_chunks_dir = os.path.join(self.dir_structure["root"], "temp_chunks")
            if os.path.exists(temp_chunks_dir):
                shutil.rmtree(temp_chunks_dir, ignore_errors=True)
                self.logger.info(
                    f"Cleaned up temporary chunks directory: {temp_chunks_dir}"
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

        # Calculate molecular weights from PDBQT files (needed for results output and box sizing)
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

            col_names = df_centers.columns.tolist()
            x_col = y_col = z_col = None
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
                    if len(lig_resnames) < self.num_pockets:
                        raise ValueError(
                            f"Not enough ligands provided: expected at least {self.num_pockets}, got {len(lig_resnames)}"
                        )

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
            self.num_pockets = len(self.docking_centers)

            self.logger.info(
                f"Search dock over {len(self.docking_centers)} grid centers..."
            )

        else:
            raise ValueError(
                f"GPU version supports modes 'production' and 'search', got: {self.mode}"
            )

        self.logger.debug(f"Docking centers: {self.docking_centers}")

    def _split_ligands_into_chunks(self, num_chunks):
        """
        Split ligands into sequential chunks using numpy for efficiency.

        Args:
            num_chunks: int: Number of chunks to split ligands into

        Returns:
            List[List[int]]: List of chunks, each containing ligand indices
        """
        if num_chunks == 0 or self.num_ligands == 0:
            return [[]]

        num_chunks = min(num_chunks, self.num_ligands)
        all_indices = np.arange(self.num_ligands, dtype=np.int32)
        split_arrays = np.array_split(all_indices, num_chunks)
        ligand_chunks = [arr.tolist() for arr in split_arrays if len(arr) > 0]

        actual_num_chunks = len(ligand_chunks)
        self.logger.info(
            f"Split {self.num_ligands} ligands into {actual_num_chunks} sequential chunks (requested {num_chunks})"
        )
        for i, chunk in enumerate(ligand_chunks):
            self.logger.debug(
                f"Chunk {i}: {len(chunk)} ligands (indices {chunk[0]}-{chunk[-1]})"
            )

        return ligand_chunks

    def _calculate_num_chunks(self):
        """
        Calculate the number of chunks for organizing ligands for GPU docking.
        Uses two-level chunking:
        1. CPU chunks: Based on CPU cores for organizing ligand files
        2. GPU chunks: Grouped chunks for efficient GPU docking

        Returns:
            tuple: (num_cpu_chunks, num_gpu_chunks)
        """
        if self.num_pockets == 0:
            return 1, 1

        num_cpu_cores = cpu_count()
        num_cpu_chunks = min(num_cpu_cores, self.num_ligands)

        # Target: k × num_gpus / num_pockets GPU chunks (k ≈ 2-4)
        k = 3
        target_gpu_chunks = max(1, int(k * self.num_gpus / self.num_pockets))
        num_gpu_chunks = min(target_gpu_chunks, num_cpu_chunks)

        self.logger.info(
            f"Chunking: {num_cpu_chunks} CPU chunks (for {num_cpu_cores} cores), "
            f"{num_gpu_chunks} GPU chunks (k={k}, num_gpus={self.num_gpus}, num_sites={self.num_pockets})"
        )
        return num_cpu_chunks, num_gpu_chunks

    def _group_cpu_chunks_for_gpu(self, cpu_chunks, num_gpu_chunks):
        """
        Group CPU chunks into GPU chunks using simple sequential grouping.
        Distributes remainder evenly across first GPU chunks.

        Args:
            cpu_chunks: List[List[int]]: List of CPU chunks
            num_gpu_chunks: int: Target number of GPU chunks

        Returns:
            List[List[int]]: List of GPU chunks
        """
        if num_gpu_chunks >= len(cpu_chunks):
            return cpu_chunks

        gpu_chunks = []
        cpu_per_gpu = len(cpu_chunks) // num_gpu_chunks
        remainder = len(cpu_chunks) % num_gpu_chunks

        start = 0
        for gpu_idx in range(num_gpu_chunks):
            # Distribute remainder across first few GPU chunks
            size = cpu_per_gpu + (1 if gpu_idx < remainder else 0)
            end = start + size

            # Combine all indices from CPU chunks in this range
            combined = []
            for cpu_idx in range(start, end):
                combined.extend(cpu_chunks[cpu_idx])
            gpu_chunks.append(combined)

            start = end

        self.logger.info(
            f"Grouped {len(cpu_chunks)} CPU chunks into {len(gpu_chunks)} GPU chunks. "
            f"GPU chunk sizes: {[len(chunk) for chunk in gpu_chunks]}"
        )
        return gpu_chunks

    def _run(self):
        """
        Run GPU docking for all ligand-pocket combinations.
        Uses chunking to organize existing PDBQT files for efficient GPU docking.
        """
        num_cpu_chunks, num_gpu_chunks = self._calculate_num_chunks()

        cpu_chunks = self._split_ligands_into_chunks(num_cpu_chunks)
        gpu_chunks = self._group_cpu_chunks_for_gpu(cpu_chunks, num_gpu_chunks)

        # Use original PDBQT file directory (ligands are already prepared)
        self._run_docking_parallel(num_gpu_chunks, gpu_chunks, None)

    def _run_docking_parallel(self, num_chunks, ligand_chunks, prepared_ligand_dir):
        """
        Run docking in parallel across multiple GPUs.

        Args:
            num_chunks: int: Number of ligand chunks
            ligand_chunks: List[List[int]]: List of ligand index lists, one per chunk
            prepared_ligand_dir: str: Unused (kept for compatibility). Ligands are read directly from self.ligand_pdbqt_files
        """
        actual_num_chunks = len(ligand_chunks)
        task_queue = Queue()
        for site_id in range(len(self.docking_centers)):
            for chunk_id in range(actual_num_chunks):
                task_queue.put((site_id, chunk_id))

        total_jobs = task_queue.qsize()
        self.logger.info(
            f"Created {total_jobs} jobs ({len(self.docking_centers)} sites × {actual_num_chunks} chunks)"
        )
        self.logger.info(
            f"Using {self.num_gpus} GPU(s) (IDs: {self.gpu_ids}) for docking"
        )

        active_procs = {}
        completed_count = 0

        try:
            while completed_count < total_jobs:
                while len(active_procs) < self.num_gpus and not task_queue.empty():
                    site_id, chunk_id = task_queue.get()
                    center = self.docking_centers[site_id]
                    ligand_indices = ligand_chunks[chunk_id]

                    available_gpus = [
                        gid for gid in self.gpu_ids if gid not in active_procs
                    ]
                    if available_gpus:
                        gpu_id = available_gpus[0]
                    else:
                        gpu_id = self.gpu_ids[
                            (site_id * actual_num_chunks + chunk_id) % self.num_gpus
                        ]

                    # Note: For Process, we can pass log_queue directly (inherited, not pickled)
                    log_queue = setup_multiprocessing_logging()
                    proc = Process(
                        target=self._dock_pocket_chunk_worker,
                        args=(
                            ligand_indices,
                            site_id,
                            chunk_id,
                            center,
                            gpu_id,
                            log_queue,
                        ),
                    )
                    active_procs[gpu_id] = (proc, site_id, chunk_id)
                    proc.start()
                    self.logger.info(
                        f"Started site {site_id}, chunk {chunk_id} on GPU {gpu_id} ({len(active_procs)}/{self.num_gpus} GPUs in use)"
                    )

                finished_gpus = []
                for gpu_id, (proc, site_id, chunk_id) in list(active_procs.items()):
                    if not proc.is_alive():
                        proc.join()  # Clean up finished process
                        finished_gpus.append(gpu_id)
                        completed_count += 1
                        self.logger.info(
                            f"Site {site_id}, chunk {chunk_id} docking completed on GPU {gpu_id} ({completed_count}/{total_jobs} done)"
                        )

                for gpu_id in finished_gpus:
                    del active_procs[gpu_id]

            for gpu_id, (proc, site_id, chunk_id) in active_procs.items():
                proc.join()
                self.logger.info(
                    f"Final cleanup: GPU {gpu_id} (site {site_id}, chunk {chunk_id}) process finished"
                )

        except KeyboardInterrupt:
            self.logger.info("GPU docking interrupted by user (Ctrl+C)")
            for gpu_id, (proc, site_id, chunk_id) in active_procs.items():
                if proc.is_alive():
                    self.logger.info(
                        f"Terminating GPU {gpu_id} worker (site {site_id}, chunk {chunk_id})"
                    )
                    proc.terminate()

            for gpu_id, (proc, site_id, chunk_id) in active_procs.items():
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill()
                    proc.join()
            self.logger.info(
                "GPU docking workers terminated, returning control to user"
            )
            raise
        finally:
            for gpu_id, (proc, site_id, chunk_id) in list(active_procs.items()):
                cleanup_process(proc, timeout=1.0)

    def _dock_pocket_chunk_worker(
        self,
        ligand_indices,
        site_id,
        chunk_id,
        center,
        gpu_id,
        log_queue,
    ):
        """
        Worker method to dock a chunk of ligands for a single pocket on a specific GPU.
        Creates a temporary directory with symlinks to the relevant ligand PDBQT files for this chunk.

        Args:
            ligand_indices: List[int]: List of ligand indices in this chunk
            site_id: int: Docking site/pocket index
            chunk_id: int: Chunk identifier
            center: List[float]: Docking center [x, y, z]
            gpu_id: int: GPU device ID
            log_queue: multiprocessing.Queue: Queue for sending log records to main process
        """
        # Configure worker logging
        configure_worker_logging(log_queue)
        logger = get_logger()
        logger.info(
            f"GPU worker (site {site_id}, chunk {chunk_id}): Starting GPU docking at center {center} on GPU {gpu_id}"
        )

        if is_sigint_pending():
            logger.info(
                f"GPU worker (site {site_id}, chunk {chunk_id}): SIGINT detected, exiting"
            )
            return

        temp_chunk_dir = os.path.join(
            self.dir_structure["root"],
            "temp_chunks",
            f"site_{site_id}_chunk_{chunk_id}",
        )
        os.makedirs(temp_chunk_dir, exist_ok=True)

        try:
            for lig_idx in ligand_indices:
                # Use original PDBQT file path from input CSV
                ligand_file = self.ligand_pdbqt_files[lig_idx]
                if os.path.exists(ligand_file):
                    symlink_path = os.path.join(
                        temp_chunk_dir, f"ligand_{lig_idx}.pdbqt"
                    )
                    os.symlink(os.path.abspath(ligand_file), symlink_path)
                else:
                    logger.warning(
                        f"PDBQT file not found for ligand {lig_idx}: {ligand_file}"
                    )

            # Determine box size (use global max for consistency across chunks)
            if self.minimized_dock:
                box_size = [5, 5, 5]
            elif self.search_gridsize is not None and self.search_gridsize > 0:
                box_size = [self.search_gridsize] * 3
                logger.info(
                    f"Using manual box size: {box_size} Å (from search_gridsize={self.search_gridsize})"
                )
            else:
                # Use maximum box size for all ligands (GPU processes all ligands together)
                # Using max ensures all ligands fit in the docking box
                max_mw = self.molweight.max()
                box_size = flexible_box_size(max_mw)
                logger.info(
                    f"Using automatic box size: {box_size} Å (from max MW={max_mw:.1f})"
                )

            temp_dir = os.path.join(self.dir_structure["poses"], "temp")
            os.makedirs(temp_dir, exist_ok=True)
            output_dir = os.path.join(temp_dir, f"site_{site_id}_chunk_{chunk_id}_out")
            os.makedirs(output_dir, exist_ok=True)

            self._run_gpu_docking(
                ligand_dir=temp_chunk_dir,
                output_dir=output_dir,
                center=center,
                box_size=box_size,
                pocket_idx=site_id,
                gpu_id=gpu_id,
            )

            self._organize_gpu_outputs(output_dir, site_id)
            shutil.rmtree(output_dir, ignore_errors=True)
        finally:
            shutil.rmtree(temp_chunk_dir, ignore_errors=True)

    def _run_gpu_docking(
        self, ligand_dir, output_dir, center, box_size, pocket_idx, gpu_id
    ):
        """
        Run GPU docking for a single pocket using the GPU executable.

        Args:
            ligand_dir: str: Directory containing ligand PDBQT files
            output_dir: str: Output directory for this pocket
            center: List[float]: Docking center [x, y, z]
            box_size: List[float]: Box size [x, y, z]
            pocket_idx: int: Pocket index
            gpu_id: int: GPU device ID
        """
        receptor_path = os.path.abspath(self.receptor)
        ligand_dir_path = os.path.abspath(ligand_dir)
        output_dir_path = os.path.abspath(output_dir)

        cmd = [
            self.gpu_executable,
            "--receptor",
            receptor_path,
            "--ligand_directory",
            ligand_dir_path,
            "--output_directory",
            output_dir_path,
            "--center_x",
            str(center[0]),
            "--center_y",
            str(center[1]),
            "--center_z",
            str(center[2]),
            "--size_x",
            str(box_size[0]),
            "--size_y",
            str(box_size[1]),
            "--size_z",
            str(box_size[2]),
            "--num_modes",
            str(self.num_poses),
            "--thread",
            str(self.gpu_threads),
            "--opencl_binary_path",
            self.gpu_opencl_binary_path,
        ]

        # Set CUDA_VISIBLE_DEVICES to use specific GPU
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        logger = get_logger()
        logger.info(f"Running GPU docking command on GPU {gpu_id}: {' '.join(cmd)}")

        try:
            result = run_cmd(
                cmd, check=True, env=env, cwd=os.path.dirname(self.gpu_executable)
            )
            logger.info(f"GPU docking completed for pocket {pocket_idx}")

        except subprocess.CalledProcessError as e:
            logger.error(
                f"GPU docking failed for pocket {pocket_idx} with return code {e.returncode}"
            )
            if e.stderr:
                logger.error(f"GPU stderr: {e.stderr}")
            raise
        except Exception as e:
            logger.error(
                f"GPU docking error for pocket {pocket_idx}: {type(e).__name__}: {e}"
            )
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _organize_gpu_outputs(self, pocket_out_dir, pocket_idx):
        """
        Organize GPU output files by moving and renaming them to match expected format.

        Args:
            pocket_out_dir: str: Temporary output directory for this pocket/grid
            pocket_idx: int: Pocket/grid index
        """
        logger = get_logger()

        output_files = glob.glob(os.path.join(pocket_out_dir, "*_out.pdbqt"))
        if not output_files:
            output_files = glob.glob(os.path.join(pocket_out_dir, "*.pdbqt"))

        # Determine space name based on mode
        space_name = "_grid_" if self.mode == "search" else "_pocket_"

        for output_file in output_files:
            basename = os.path.basename(output_file)
            match = re.search(r"ligand_(\d+)(?:_out)?\.pdbqt", basename)
            if match:
                lig_idx = int(match.group(1))
                new_name = f"ligand_{lig_idx}{space_name}{pocket_idx}_docked.pdbqt"
                new_path = os.path.join(self.dir_structure["poses"], new_name)
                shutil.move(output_file, new_path)
                logger.debug(f"Moved {output_file} to {new_path}")
            else:
                logger.warning(
                    f"Could not extract ligand index from {basename}, skipping"
                )

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
            # Use mode-appropriate pattern
            if self.mode == "search":
                matches = re.search(r"ligand_(\d+)_grid_(\d+)_docked.pdbqt", filePath)
            else:
                matches = re.search(r"ligand_(\d+)_pocket_(\d+)_docked.pdbqt", filePath)

            if not matches:
                continue
            lig_id, g_id = map(int, matches.groups())

            # extract affinities + RMSDs
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

            # fill available poses, pad rest with zeros
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

        # Choose output filename based on mode
        if self.mode == "search":
            output_csv = os.path.join(
                self.dir_structure["summaries"], "best_docking_centers.csv"
            )
        else:
            output_csv = os.path.join(
                self.dir_structure["summaries"], "production_docking_results.csv"
            )
        df_best = self._save_best_search_docking_poses(
            pkl_file, output_csv, selecttype="bestPerGrid"
        )

        # Generate search-specific outputs
        if self.mode == "search":
            # Save top 100 poses for search visualization
            output_csv_top = os.path.join(
                self.dir_structure["summaries"], "best_search_docking_centers.csv"
            )
            df_top = self._save_best_search_docking_poses(
                pkl_file, output_csv_top, selecttype="topN", topN=100
            )

            # Write PDB file for visualization
            write_all_centers_pdb(
                df_top,
                os.path.join(
                    self.dir_structure["summaries"], "best_search_docking_centers.pdb"
                ),
            )

            # Save best per grid (already saved as output_csv above)
            self.logger.info(
                f"Search mode: Generated visualization files in {self.dir_structure['summaries']}"
            )

        return output_csv

    def _save_best_search_docking_poses(
        self, pkl_file, output_csv, selecttype="topN", topN=10, threshold_fraction=0.1
    ):
        """
        Select best docking poses from docking results.
        (Same implementation as VinaDock for consistency)

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
            # handled separately below
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
                    # translate to original pose index
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
