"""
Vina GPU docking backend.

Uses the AutoDock-Vina-GPU-2-1 binary with --ligand_directory for batch
docking on GPUs. All chunking, parallelism, output organization, and cleanup
are inherited from BaseGPUBackend.
"""

import os
from typing import List, Union

from .base_gpu import BaseGPUBackend


class VinaGPUBackend(BaseGPUBackend):
    """GPU-based AutoDock Vina docking backend."""

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
        gpu_threads: int = 8000,
        gpu_executable: str = None,
        gpu_opencl_binary_path: str = None,
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
            num_gpus=num_gpus,
            gpu_ids=gpu_ids,
        )

        # Default paths: docking/vina_gpu/ next to docking/ (parent of backends/)
        if gpu_executable is None or gpu_opencl_binary_path is None:
            docking_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
            vina_gpu_dir = os.path.join(docking_dir, "vina_gpu")
            if gpu_executable is None:
                gpu_executable = os.path.join(
                    vina_gpu_dir, "AutoDock-Vina-GPU-2-1"
                )
            if gpu_opencl_binary_path is None:
                gpu_opencl_binary_path = vina_gpu_dir

        self.gpu_executable = gpu_executable
        self.gpu_opencl_binary_path = gpu_opencl_binary_path
        self.gpu_threads = gpu_threads
        self.gpu_libdir = os.environ.get("VINA_GPU_LIBDIR", "").strip()

        if self.gpu_libdir:
            self.logger.info(
                "Using VINA_GPU_LIBDIR for Vina-GPU runtime libraries: %s",
                self.gpu_libdir,
            )

        if not os.path.exists(self.gpu_executable):
            raise FileNotFoundError(
                f"GPU executable not found: {self.gpu_executable}"
            )
        if not os.access(self.gpu_executable, os.X_OK):
            raise PermissionError(
                f"GPU executable is not executable: {self.gpu_executable}"
            )

    def _run_gpu_docking(
        self, ligand_dir, output_dir, center, box_size, pocket_idx, gpu_id
    ):
        """Build and run the AutoDock-Vina-GPU-2-1 command."""
        cmd = [
            self.gpu_executable,
            "--receptor",
            os.path.abspath(self.receptor),
            "--ligand_directory",
            os.path.abspath(ligand_dir),
            "--output_directory",
            os.path.abspath(output_dir),
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

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        if self.gpu_libdir:
            current_ld_library_path = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = (
                f"{self.gpu_libdir}:{current_ld_library_path}"
                if current_ld_library_path
                else self.gpu_libdir
            )

        self._run_gpu_subprocess(
            cmd,
            env=env,
            cwd=os.path.dirname(self.gpu_executable),
            context_str=f"pocket {pocket_idx} on GPU {gpu_id}",
        )
