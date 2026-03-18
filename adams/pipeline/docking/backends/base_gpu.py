"""
Base class for GPU docking backends.

Provides chunking, parallel docking job execution across GPUs, output
organization, temp dir cleanup, and a shared subprocess runner. Concrete GPU
backends (VinaGPUBackend, UniDockBackend) subclass this and implement only
_run_gpu_docking() with their specific executable and CLI arguments.
"""

import csv
import glob
import os
import re
import shutil
import subprocess
import traceback
from typing import List, Union

import numpy as np

from ....error_handling import is_sigint_pending
from ....logger_utils import get_logger
from ....utils import run_cmd
from ....utils.multiprocessing_utils import cpu_count
from ..atom_types import sanitize_pdbqt_atom_types
from ..docking import BaseDockingPipeline
from ..utils import (
    SPACE_NAME_GRID,
    SPACE_NAME_POCKET,
    get_docking_box_size,
)

CHUNK_FACTOR = 3
"""Multiplier for GPU chunk count: target_gpu_chunks = CHUNK_FACTOR * num_gpus / num_pockets."""



class BaseGPUBackend(BaseDockingPipeline):
    """
    Base class for GPU-based docking backends.

    Handles GPU validation, ligand chunking, docking job lifecycle,
    output organization, and temp dir cleanup. Subclasses implement only
    _run_gpu_docking(ligand_dir, output_dir, center, box_size, pocket_idx, gpu_id).
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
        num_gpus: int = 1,
        gpu_ids: List[int] = None,
    ):
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
        if num_gpus <= 0:
            raise ValueError(
                f"Number of GPUs must be greater than 0, got: {num_gpus}"
            )
        self.num_gpus = num_gpus
        if gpu_ids is None:
            self.gpu_ids = list(range(num_gpus))
        else:
            self.gpu_ids = gpu_ids
            if len(self.gpu_ids) != num_gpus:
                raise ValueError(
                    f"Number of GPU IDs ({len(self.gpu_ids)}) must match "
                    f"num_gpus ({num_gpus})"
                )

    # ------------------------------------------------------------------
    # Docking job API (implements BaseDockingPipeline interface)
    # ------------------------------------------------------------------

    def _get_worker_count(self):
        """One worker per GPU."""
        return self.num_gpus

    def _build_docking_jobs(self):
        """
        Chunk ligands and create one docking job per (site, chunk) combination.

        Each job is a tuple (site_id, chunk_id, ligand_indices, center).
        """
        if self.num_ligands == 0:
            self.logger.warning("No ligands available for GPU docking.")
            return []
        if len(self.docking_centers) == 0:
            self.logger.warning("No docking centers available for GPU docking.")
            return []

        num_cpu_chunks, num_gpu_chunks = self._calculate_num_chunks()
        cpu_chunks = self._split_ligands_into_chunks(num_cpu_chunks)
        gpu_chunks = self._group_cpu_chunks_for_gpu(cpu_chunks, num_gpu_chunks)

        jobs = []
        for site_id in range(len(self.docking_centers)):
            for chunk_id, ligand_indices in enumerate(gpu_chunks):
                center = self.docking_centers[site_id]
                jobs.append((site_id, chunk_id, ligand_indices, center))

        self.logger.info(
            f"Created {len(jobs)} GPU docking jobs "
            f"({len(self.docking_centers)} sites x {len(gpu_chunks)} chunks)"
        )
        self.logger.info(
            f"Using {self.num_gpus} GPU(s) (IDs: {self.gpu_ids}) for docking"
        )
        return jobs

    def _sanitize_ligand_pdbqt_for_gpu(self, src_path: str, dst_path: str, lig_idx: int):
        """
        Copy ligand PDBQT into a GPU chunk and normalize unsupported atom types.

        Returns sanitizer result dict with replacement/unresolved details.
        """
        result = sanitize_pdbqt_atom_types(
            src_path=src_path,
            dst_path=dst_path,
            strict_unresolved=False,
        )
        if result["changed_entries"]:
            replacement_summary = ", ".join(
                f"{k}:{v}" for k, v in sorted(result["replacement_counts"].items())
            )
            self.logger.warning(
                "GPU ligand atom-type normalization for ligand "
                f"{lig_idx}: replaced {result['changed_entries']} entries "
                f"({replacement_summary}) in {os.path.basename(src_path)}"
            )
        return result

    def _append_skipped_ligand_row(
        self,
        *,
        stage: str,
        lig_idx: int,
        reason: str,
        worker_id: int,
        site_id: int,
        chunk_id: int,
        pdbqt_file: str,
    ):
        """Append one skipped-ligand record to summaries/skipped_ligands.csv."""
        csv_path = os.path.join(self.dir_structure["summaries"], "skipped_ligands.csv")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        def _safe_get(series, idx, default=""):
            try:
                if hasattr(series, "iloc"):
                    return series.iloc[idx]
                return series[idx]
            except Exception:
                return default

        row = {
            "Stage": stage,
            "Ligand_Index": lig_idx,
            "ID": _safe_get(self.lignames, lig_idx, f"ligand_{lig_idx}"),
            "Parent_ID": _safe_get(self.parent_ids, lig_idx, ""),
            "Variant_ID": _safe_get(self.variant_ids, lig_idx, ""),
            "Variant_Type": _safe_get(self.variant_types, lig_idx, ""),
            "Conformer_Index": _safe_get(self.conformer_indices, lig_idx, ""),
            "PDBQT_File": pdbqt_file,
            "Reason": reason,
            "Worker_ID": worker_id,
            "Site_ID": site_id,
            "Chunk_ID": chunk_id,
        }
        fieldnames = [
            "Stage",
            "Ligand_Index",
            "ID",
            "Parent_ID",
            "Variant_ID",
            "Variant_Type",
            "Conformer_Index",
            "PDBQT_File",
            "Reason",
            "Worker_ID",
            "Site_ID",
            "Chunk_ID",
        ]

        file_exists = os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            if not file_exists or os.path.getsize(csv_path) == 0:
                writer.writeheader()
            writer.writerow(row)
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

    def _execute_docking_job(self, job, worker_id):
        """
        Execute one GPU docking job: symlink ligands, compute box size,
        call _run_gpu_docking(), organize outputs, then clean up temp dirs.
        """
        site_id, chunk_id, ligand_indices, center = job
        gpu_id = self.gpu_ids[worker_id]
        logger = get_logger()
        logger.info(
            f"GPU worker (site {site_id}, chunk {chunk_id}): "
            f"Starting docking at center {center} on GPU {gpu_id}"
        )

        if is_sigint_pending():
            logger.info(
                f"GPU worker (site {site_id}, chunk {chunk_id}): "
                f"SIGINT detected, exiting"
            )
            return

        temp_chunk_dir = os.path.join(
            self.dir_structure["root"],
            "temp_chunks",
            f"site_{site_id}_chunk_{chunk_id}",
        )
        os.makedirs(temp_chunk_dir, exist_ok=True)

        try:
            # Stage ligand PDBQT files into the temp chunk directory.
            # GPU parser is stricter than CPU Vina for atom types (e.g., rejects CG0),
            # so normalize unsupported tokens during staging.
            staged_count = 0
            for lig_idx in ligand_indices:
                ligand_file = self.ligand_pdbqt_files[lig_idx]
                if os.path.exists(ligand_file):
                    staged_path = os.path.join(
                        temp_chunk_dir, f"ligand_{lig_idx}.pdbqt"
                    )
                    result = self._sanitize_ligand_pdbqt_for_gpu(
                        src_path=ligand_file,
                        dst_path=staged_path,
                        lig_idx=lig_idx,
                    )
                    unresolved = result.get("unresolved_details", [])
                    if unresolved:
                        reason = (
                            f"unresolved_atom_types={len(unresolved)}; "
                            f"first={unresolved[0]['atom_type']}"
                        )
                        logger.warning(
                            "Skipping ligand %s at GPU staging (site=%s chunk=%s): %s",
                            lig_idx,
                            site_id,
                            chunk_id,
                            reason,
                        )
                        self._append_skipped_ligand_row(
                            stage="gpu_staging",
                            lig_idx=lig_idx,
                            reason=reason,
                            worker_id=worker_id,
                            site_id=site_id,
                            chunk_id=chunk_id,
                            pdbqt_file=ligand_file,
                        )
                        continue
                    staged_count += 1
                else:
                    logger.warning(
                        f"PDBQT file not found for ligand {lig_idx}: "
                        f"{ligand_file}"
                    )

            if staged_count == 0:
                logger.warning(
                    "No valid ligands remained after GPU staging for site=%s, chunk=%s; skipping job.",
                    site_id,
                    chunk_id,
                )
                return

            # Determine box size based on mode (GPU uses MW-adaptive sizing
            # in production mode for tighter boxes on smaller ligands)
            max_mw = self.molweight.max()
            box_size = get_docking_box_size(
                self.minimized_dock,
                self.mode,
                self.search_gridsize,
                max_mw,
                production_gridsize=self.production_gridsize,
            )
            logger.info(f"Using box size: {box_size} Angstrom")

            # Create temp output directory
            temp_dir = os.path.join(self.dir_structure["poses"], "temp")
            os.makedirs(temp_dir, exist_ok=True)
            output_dir = os.path.join(
                temp_dir, f"site_{site_id}_chunk_{chunk_id}_out"
            )
            os.makedirs(output_dir, exist_ok=True)

            # Run the backend-specific GPU docking executable
            self._run_gpu_docking(
                ligand_dir=temp_chunk_dir,
                output_dir=output_dir,
                center=center,
                box_size=box_size,
                pocket_idx=site_id,
                gpu_id=gpu_id,
            )

            # Move output files into the standard poses directory
            self._organize_gpu_outputs(output_dir, site_id)
            shutil.rmtree(output_dir, ignore_errors=True)
        finally:
            shutil.rmtree(temp_chunk_dir, ignore_errors=True)

    # ------------------------------------------------------------------
    # Abstract method — concrete GPU backends must implement
    # ------------------------------------------------------------------

    def _run_gpu_docking(
        self, ligand_dir, output_dir, center, box_size, pocket_idx, gpu_id
    ):
        """Run the GPU docking executable. Must be implemented by subclasses."""
        raise NotImplementedError(
            "GPU backends must implement _run_gpu_docking("
            "ligand_dir, output_dir, center, box_size, pocket_idx, gpu_id)"
        )

    # ------------------------------------------------------------------
    # Shared GPU helpers
    # ------------------------------------------------------------------

    def _run_gpu_subprocess(self, cmd, env, cwd, context_str="pocket"):
        """
        Run a GPU docking subprocess with standardized logging and error handling.

        Backends call this from _run_gpu_docking() after building their command.
        """
        logger = get_logger()
        logger.info(f"Running GPU docking command: {' '.join(cmd)}")
        try:
            run_cmd(cmd, check=True, env=env, cwd=cwd)
            logger.info(f"GPU docking completed for {context_str}")
        except subprocess.CalledProcessError as e:
            logger.error(
                f"GPU docking failed for {context_str} "
                f"with return code {e.returncode}"
            )
            if e.stderr:
                logger.error(f"GPU stderr: {e.stderr}")
            raise
        except Exception as e:
            logger.error(
                f"GPU docking error for {context_str}: "
                f"{type(e).__name__}: {e}"
            )
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def _cleanup_after_run(self):
        """Remove temporary chunk and output directories after docking."""
        temp_dir = os.path.join(self.dir_structure["poses"], "temp")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
            self.logger.info(f"Cleaned up temporary directory: {temp_dir}")
        temp_chunks_dir = os.path.join(
            self.dir_structure["root"], "temp_chunks"
        )
        if os.path.exists(temp_chunks_dir):
            shutil.rmtree(temp_chunks_dir, ignore_errors=True)
            self.logger.info(
                f"Cleaned up temporary chunks directory: {temp_chunks_dir}"
            )

    # ------------------------------------------------------------------
    # Chunking helpers
    # ------------------------------------------------------------------

    def _split_ligands_into_chunks(self, num_chunks):
        """Split ligand indices into approximately equal chunks."""
        if num_chunks == 0 or self.num_ligands == 0:
            return [[]]
        num_chunks = min(num_chunks, self.num_ligands)
        all_indices = np.arange(self.num_ligands, dtype=np.int32)
        split_arrays = np.array_split(all_indices, num_chunks)
        ligand_chunks = [arr.tolist() for arr in split_arrays if len(arr) > 0]
        self.logger.info(
            f"Split {self.num_ligands} ligands into {len(ligand_chunks)} "
            f"sequential chunks (requested {num_chunks})"
        )
        return ligand_chunks

    def _calculate_num_chunks(self):
        """
        Determine number of CPU and GPU chunks for ligand splitting.

        Returns (num_cpu_chunks, num_gpu_chunks) where CPU chunks are
        first created for fine granularity, then grouped into GPU chunks.
        """
        if self.num_pockets == 0:
            return 1, 1
        num_cpu_cores = cpu_count()
        num_cpu_chunks = max(1, min(num_cpu_cores, self.num_ligands))
        target_gpu_chunks = max(
            1, int(CHUNK_FACTOR * self.num_gpus / self.num_pockets)
        )
        num_gpu_chunks = max(1, min(target_gpu_chunks, num_cpu_chunks))
        self.logger.info(
            f"Chunking: {num_cpu_chunks} CPU chunks (for {num_cpu_cores} cores), "
            f"{num_gpu_chunks} GPU chunks (k={CHUNK_FACTOR}, "
            f"num_gpus={self.num_gpus}, num_sites={self.num_pockets})"
        )
        return num_cpu_chunks, num_gpu_chunks

    def _group_cpu_chunks_for_gpu(self, cpu_chunks, num_gpu_chunks):
        """Merge fine-grained CPU chunks into fewer, larger GPU chunks."""
        if num_gpu_chunks <= 0:
            return cpu_chunks
        if num_gpu_chunks >= len(cpu_chunks):
            return cpu_chunks
        gpu_chunks = []
        cpu_per_gpu = len(cpu_chunks) // num_gpu_chunks
        remainder = len(cpu_chunks) % num_gpu_chunks
        start = 0
        for gpu_idx in range(num_gpu_chunks):
            size = cpu_per_gpu + (1 if gpu_idx < remainder else 0)
            end = start + size
            combined = []
            for cpu_idx in range(start, end):
                combined.extend(cpu_chunks[cpu_idx])
            gpu_chunks.append(combined)
            start = end
        self.logger.info(
            f"Grouped {len(cpu_chunks)} CPU chunks into "
            f"{len(gpu_chunks)} GPU chunks. "
            f"GPU chunk sizes: {[len(chunk) for chunk in gpu_chunks]}"
        )
        return gpu_chunks

    # ------------------------------------------------------------------
    # Output organization
    # ------------------------------------------------------------------

    def _organize_gpu_outputs(self, pocket_out_dir, pocket_idx):
        """Move docked PDBQT files from temp output dir to poses/ with standard naming."""
        logger = get_logger()
        output_files = glob.glob(
            os.path.join(pocket_out_dir, "*_out.pdbqt")
        )
        if not output_files:
            output_files = glob.glob(
                os.path.join(pocket_out_dir, "*.pdbqt")
            )
        space_name = SPACE_NAME_GRID if self.mode == "search" else SPACE_NAME_POCKET
        for output_file in output_files:
            basename = os.path.basename(output_file)
            match = re.search(r"ligand_(\d+)(?:_out)?\.pdbqt", basename)
            if match:
                lig_idx = int(match.group(1))
                new_name = (
                    f"ligand_{lig_idx}{space_name}{pocket_idx}_docked.pdbqt"
                )
                new_path = os.path.join(
                    self.dir_structure["poses"], new_name
                )
                shutil.move(output_file, new_path)
                logger.debug(f"Moved {output_file} to {new_path}")
            else:
                logger.warning(
                    f"Could not extract ligand index from {basename}, skipping"
                )
