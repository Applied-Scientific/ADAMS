"""
UniDock docking backend.

Uses the UniDock CLI with --ligand_index for batch docking on GPUs.
All chunking, parallelism, output organization, and cleanup are inherited
from BaseGPUBackend.
"""

import glob
import os
import re
import shutil
import sys
from typing import List, Union

from ....logger_utils import get_logger
from .base_gpu import BaseGPUBackend


class UniDockBackend(BaseGPUBackend):
    """UniDock (GPU) docking backend."""

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
        gpu_executable: str = "unidock",
        scoring: str = None,
        exhaustiveness: int = None,
        energy_range: float = None,
        min_rmsd: float = None,
        spacing: float = None,
        seed: int = None,
        refine_step: int = None,
        max_evals: int = None,
        max_step: int = None,
        max_gpu_memory: int = None,
        search_mode: str = None,
        verbosity: int = None,
        cpu: int = None,
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

        self.gpu_executable = gpu_executable
        self.scoring = scoring
        self.exhaustiveness = exhaustiveness
        self.energy_range = energy_range
        self.min_rmsd = min_rmsd
        self.spacing = spacing
        self.seed = seed
        self.refine_step = refine_step
        self.max_evals = max_evals
        self.max_step = max_step
        self.max_gpu_memory = max_gpu_memory
        self.search_mode = search_mode
        self.verbosity = verbosity
        self.cpu = cpu

        if sys.platform == "darwin":
            raise RuntimeError(
                "UniDock backend is not supported on macOS. "
                "Use backend='vina' or backend='vina_gpu' instead."
            )

        resolved_executable = shutil.which(self.gpu_executable)
        if resolved_executable is None:
            raise FileNotFoundError(
                "UniDock executable not found on PATH: "
                f"{self.gpu_executable}. Install UniDock or pass "
                "gpu_executable='/absolute/path/to/unidock'."
            )
        self.gpu_executable = resolved_executable

        if scoring is not None and scoring not in ("ad4", "vina", "vinardo"):
            raise ValueError(
                f"scoring must be 'ad4', 'vina', or 'vinardo', got: {scoring}"
            )
        if search_mode is not None and search_mode not in (
            "fast",
            "balance",
            "detail",
        ):
            raise ValueError(
                f"search_mode must be 'fast', 'balance', or 'detail', "
                f"got: {search_mode}"
            )

    def _run_gpu_docking(
        self, ligand_dir, output_dir, center, box_size, pocket_idx, gpu_id
    ):
        """Build and run the UniDock command."""
        output_dir_path = os.path.abspath(output_dir)
        os.makedirs(output_dir_path, exist_ok=True)

        # Build ligand_index file (one path per line)
        ligand_files = sorted(
            glob.glob(os.path.join(ligand_dir, "ligand_*.pdbqt")),
            key=lambda p: int(
                re.search(r"ligand_(\d+)\.pdbqt", os.path.basename(p)).group(
                    1
                )
            ),
        )
        ligand_index_path = os.path.join(output_dir_path, "ligand_index.txt")
        with open(ligand_index_path, "w") as f:
            for p in ligand_files:
                f.write(os.path.abspath(p) + "\n")

        cmd = [
            self.gpu_executable,
            "--receptor",
            os.path.abspath(self.receptor),
            "--ligand_index",
            ligand_index_path,
            "--dir",
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
        ]

        # Parameters where 0 is a valid value:
        #   max_evals=0: use heuristic, max_step=0: use heuristic
        #   max_gpu_memory=0: use all available, cpu=0: auto-detect
        #   verbosity=0: silent mode (valid output level 0/1/2)
        #   seed=0: valid seed value (no default, random if not specified)
        ZERO_VALID_PARAMS = {
            "--max_evals",
            "--max_step",
            "--max_gpu_memory",
            "--cpu",
            "--verbosity",
            "--seed",
        }

        logger = get_logger()
        optional_args = [
            ("--scoring", self.scoring),
            ("--exhaustiveness", self.exhaustiveness),
            ("--energy_range", self.energy_range),
            ("--min_rmsd", self.min_rmsd),
            ("--spacing", self.spacing),
            ("--seed", self.seed),
            ("--refine_step", self.refine_step),
            ("--max_evals", self.max_evals),
            ("--max_step", self.max_step),
            ("--max_gpu_memory", self.max_gpu_memory),
            ("--search_mode", self.search_mode),
            ("--verbosity", self.verbosity),
            ("--cpu", self.cpu),
        ]
        for opt, val in optional_args:
            if val is None:
                continue
            if val == 0 and opt not in ZERO_VALID_PARAMS:
                logger.debug(
                    f"Skipping {opt}=0 (using unidock default instead)"
                )
                continue
            cmd.extend([opt, str(val)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        self._run_gpu_subprocess(
            cmd,
            env=env,
            cwd=output_dir_path,
            context_str=f"pocket {pocket_idx} on GPU {gpu_id}",
        )
