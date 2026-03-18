"""
Membrane MD Simulation Module - Transmembrane Protein MD Pipeline

This module runs multi-stage equilibration and production MD for transmembrane
protein systems embedded in lipid bilayers.

EQUILIBRATION PROTOCOL:
    1. NVT (100 ps) - Temperature stabilization with base POSRES restraints
    2. NPT-eq1 (100 ps) - Semi-isotropic C-rescale barostat with base POSRES restraints
    3. NPT-eq2 (100 ps) - Reduced POSRES force constants (500 kJ/mol/nm^2 variants)
    4. NPT-eq3 (200 ps) - Weak POSRES force constants (200 kJ/mol/nm^2 variants)
    5. NPT-eq4 (500 ps) - No restraints (free equilibration verification)
    6. Production MD (10 ns default) - Parrinello-Rahman semi-isotropic barostat

Key differences from soluble protein pipeline:
    - Semi-isotropic pressure coupling (xy coupled, z independent)
    - Three TC groups: Protein, Membrane, Solvent_and_ions
    - Multi-stage restraint reduction to prevent membrane artifacts
    - Longer tau_p for membrane stability (5 ps vs 2 ps)

INPUTS (from file_paths dictionary):
    - membrane_min_gro: Energy-minimized membrane system (from run_membrane_prep)
    - membrane_top: System topology with restraint variants
    - membrane_ndx: Index file with Protein/Membrane/Solvent_and_ions groups
    - gromacs_path, ambertools_path, gromacs_binary_type

OUTPUTS (added to file_paths dictionary):
    - membrane_md_tpr: Production MD TPR file
    - membrane_md_xtc: Production MD trajectory
    - membrane_md_gro: Final frame GRO from production

EXTERNAL COMMANDS:
    - gmx grompp: Prepare TPR run input files for each stage
    - gmx mdrun: Execute MD simulation
    - mpirun (optional): For multi-rank MPI execution

GROMPP WARNING POLICY:
    Same as the soluble pipeline: no blanket -maxwarn. Warnings are surfaced
    and must be approved individually.
"""

import os
import time
from typing import Optional

from ....logger_utils import get_logger, log_step_execution
from ..shared import (
    GromacsContext,
    GromppWarningPolicy,
    copy_mdp_files,
    get_membrane_mdp_dir,
    validate_required_file_paths,
)

# Equilibration stages: (name, mdp_file, description)
MEMBRANE_EQ_STAGES = [
    ("nvt",      "membrane_nvt.mdp",      "NVT equilibration (strong restraints)"),
    ("npt_eq1",  "membrane_npt_eq1.mdp",  "NPT eq. stage 1 (FC=1000)"),
    ("npt_eq2",  "membrane_npt_eq2.mdp",  "NPT eq. stage 2 (FC=500)"),
    ("npt_eq3",  "membrane_npt_eq3.mdp",  "NPT eq. stage 3 (FC=200)"),
    ("npt_eq4",  "membrane_npt_eq4.mdp",  "NPT eq. stage 4 (no restraints)"),
]


class MembraneMd:
    """
    Multi-stage equilibration and production MD for membrane systems.

    Composes ``GromacsContext`` for GPU/MPI/OMP and grompp policy.
    """

    def __init__(
        self,
        file_paths,
        gpu: bool = False,
        num_gpus: int = -1,
        mpi_ranks: int = 0,
        omp_threads: int = 0,
        topol: str = "system.top",
        index: str = "index.ndx",
        grompp_warning_policy: Optional[GromppWarningPolicy] = None,
    ):
        if file_paths is None:
            raise ValueError("file_paths dictionary is required.")
        self.file_paths = file_paths
        self.ctx = GromacsContext.from_file_paths(
            file_paths,
            gpu=gpu,
            num_gpus=num_gpus,
            mpi_ranks=mpi_ranks,
            omp_threads=omp_threads,
            grompp_warning_policy=grompp_warning_policy,
        )
        self.topol = (
            file_paths.get("membrane_top", topol)
            if topol == "system.top"
            else topol
        )
        self.index = (
            file_paths.get("membrane_ndx", index)
            if index == "index.ndx"
            else index
        )
        self.logger = get_logger()

    def validate_files(self):
        """Validate required keys in file_paths."""
        validate_required_file_paths(
            self.file_paths,
            [
                "membrane_min_gro",
                "membrane_top",
                "membrane_ndx",
            ],
            context="Ensure run_membrane_prep has been executed first.",
        )

        min_gro = self.file_paths["membrane_min_gro"]
        if not os.path.exists(min_gro):
            raise FileNotFoundError(
                f"Minimized membrane GRO not found: {min_gro}\n"
                "Run run_membrane_prep first to generate min.gro."
            )
        membrane_top = self.file_paths["membrane_top"]
        membrane_ndx = self.file_paths["membrane_ndx"]
        if not os.path.exists(membrane_top):
            raise FileNotFoundError(f"Membrane topology not found: {membrane_top}")
        if not os.path.exists(membrane_ndx):
            raise FileNotFoundError(f"Membrane index not found: {membrane_ndx}")

    def run(self) -> dict:
        """
        Run multi-stage equilibration and production MD.

        Returns:
            dict: Updated file_paths with membrane_md_tpr, membrane_md_xtc, etc.
        """
        step_logger = log_step_execution("Membrane MD Simulation", self.logger)
        with step_logger:
            membrane_dir = os.path.dirname(
                os.path.abspath(self.file_paths["membrane_min_gro"])
            )

            mdp_dir = get_membrane_mdp_dir()
            time_start = time.perf_counter()

            stage_mdp_files = [mdp for _, mdp, _ in MEMBRANE_EQ_STAGES]
            copy_mdp_files(
                mdp_dir,
                stage_mdp_files + ["membrane_md.mdp"],
                membrane_dir,
            )

            prev_gro = os.path.basename(self.file_paths["membrane_min_gro"])
            for stage_name, mdp_file, description in MEMBRANE_EQ_STAGES:
                with step_logger.timing(stage_name):
                    self.logger.info(f"--- {description} ---")
                    self._run_stage(
                        stage_name=stage_name,
                        mdp_file=mdp_file,
                        input_gro=prev_gro,
                        cwd=membrane_dir,
                    )
                    prev_gro = f"{stage_name}.gro"

            with step_logger.timing("production_md"):
                self.logger.info("--- Production MD ---")
                self._run_stage(
                    stage_name="md",
                    mdp_file="membrane_md.mdp",
                    input_gro=prev_gro,
                    cwd=membrane_dir,
                )

            elapsed = time.perf_counter() - time_start
            self.logger.info(
                f"Membrane MD completed in {elapsed:.1f}s "
                f"({elapsed/60:.1f} min)"
            )

            # Record output paths
            self.file_paths["membrane_md_tpr"] = os.path.join(membrane_dir, "md.tpr")
            self.file_paths["membrane_md_xtc"] = os.path.join(membrane_dir, "md.xtc")
            self.file_paths["membrane_md_gro"] = os.path.join(membrane_dir, "md.gro")

            return self.file_paths

    def _run_stage(self, stage_name, mdp_file, input_gro, cwd=None):
        """Run a single equilibration or production stage (grompp + mdrun)."""
        self.ctx.run_stage(
            stage_name,
            mdp_file,
            input_gro,
            self.index,
            self.topol,
            cwd=cwd,
        )
        self.logger.info(f"Stage {stage_name} completed successfully")
