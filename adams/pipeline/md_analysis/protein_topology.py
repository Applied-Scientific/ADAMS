"""
Protein Topology Module - MD Analysis Pipeline Step 1

This module prepares protein structure for MD simulation by converting PDB files
to GROMACS format using pdb2gmx. It generates protein topology files required
for subsequent ligand preparation and MD simulation steps.

POSITION IN PIPELINE:
    Step 1 of 4: run_protein_topology
    - Must be executed before run_lig_prepare
    - Can be skipped if protein topology files already exist

INPUTS (from file_paths dictionary):
    - protein_file: Path to input protein PDB file
    - protein_dir: Directory for output files
    - gromacs_path: Path to GROMACS installation
    - ambertools_path: Path to AmberTools installation

OUTPUTS (added to file_paths dictionary):
    - protein_gro: Protein structure in GRO format
    - protein_top: Protein topology file (topol.top)
    - posre_itp: Position restraints file (posre.itp)

KEY FUNCTIONALITY:
    - Converts PDB to GROMACS format using pdb2gmx
    - Assigns force field parameters
    - Generates position restraints for protein atoms
    - Creates topology file compatible with GROMACS MD simulations

EXTERNAL COMMANDS:
    - gmx pdb2gmx: Convert PDB to GROMACS format with force field assignment
"""

import os
import subprocess

from ...error_handling import FatalError
from ...logger_utils import get_logger, log_step_execution
from ...utils import run_cmd
from .utils import get_gromacs_binary


class ProteinTopology:
    """
    Class to handle protein topology preparation using GROMACS pdb2gmx.

    This class replicates the functionality of the pdb2gmx command from the
    lig_prepare_inference.sh script, providing a Python interface for
    converting protein PDB files to GROMACS format.
    """

    def __init__(
        self,
        file_paths,
        forcefield="amber03",
        water_model="tip3p",
        ignore_hydrogens=True,
    ):
        """
        Initialize the ProteinTopology class.

        Args:
            file_paths (dict): File paths dictionary - single source of truth for all paths.
                Must include:
                - protein_file: Path to input protein PDB file
                - protein_dir: Directory for output files
                - gromacs_path: Path to GROMACS installation
                - ambertools_path: Path to AmberTools installation
                - gromacs_binary_type: Type of GROMACS binary (from discover_paths)
            forcefield (str): GROMACS force field (default: "amber03")
            water_model (str): Water model (default: "tip3p")
            ignore_hydrogens (bool): Whether to ignore hydrogens (default: True)
        """
        self.logger = get_logger()

        if file_paths is None:
            raise ValueError(
                "file_paths dictionary is required. Use build_file_paths() and discover_paths() first."
            )
        self.file_paths = file_paths

        self.validate_files()

        self.gromacs_path = file_paths["gromacs_path"]
        self.ambertools_path = file_paths["ambertools_path"]
        self.gromacs_binary_type = file_paths.get("gromacs_binary_type", "standard")
        self.gmx_binary = get_gromacs_binary(
            self.gromacs_path, binary_type=self.gromacs_binary_type, require_mpi=False
        )
        self.forcefield = forcefield
        self.water_model = water_model
        self.ignore_hydrogens = ignore_hydrogens

    def validate_files(self):
        """
        Validate required keys exist in file_paths.

        Checks that:
        - protein_file exists in file_paths
        - protein_dir exists in file_paths
        - gromacs_path exists in file_paths
        - ambertools_path exists in file_paths

        Raises:
            ValueError: If required paths are missing from file_paths
        """
        if not self.file_paths.get("protein_file"):
            raise ValueError(
                "protein_file required in file_paths.\n"
                f"Available keys: {list(self.file_paths.keys())}"
            )

        protein_dir = self.file_paths.get("protein_dir")
        if not protein_dir:
            raise ValueError(
                "protein_dir required in file_paths.\n"
                f"Available keys: {list(self.file_paths.keys())}\n"
                "Use build_file_paths() with md_workdir to create directory structure."
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

        os.makedirs(protein_dir, exist_ok=True)

    def _run_pdb2gmx(self):
        """
        Run GROMACS pdb2gmx to convert protein PDB to GROMACS format.

        This method replicates the pdb2gmx command from the shell script:
        gmx_mpi pdb2gmx -f "$PROTEIN" -o "${MD_WORKDIR}/protein/protein.gro"
        -p "${MD_WORKDIR}/protein/topol.top" -ignh -ff "$FF" -water "$WATER_MODEL"

        Args:
            protein_file (str): Path to input protein PDB file
            md_workdir (str): Path to MD working directory
            forcefield (str): GROMACS force field (default: "amber03")
            water_model (str): Water model (default: "tip3p")
            ignore_hydrogens (bool): Whether to ignore hydrogens (default: True)

        Returns:
            dict: Dictionary with output file paths and success status

        Raises:
            subprocess.CalledProcessError: If pdb2gmx command fails
        """
        try:
            protein_file = self.file_paths["protein_file"]
            # Convert to absolute path to ensure it works when cwd is changed
            protein_file = os.path.abspath(protein_file)
            protein_dir = self.file_paths["protein_dir"]
            # Convert protein_dir to absolute path as well
            protein_dir = os.path.abspath(protein_dir)
            output_gro = os.path.join(protein_dir, "protein.gro")
            output_top = os.path.join(protein_dir, "topol.top")

            pdb2gmx_cmd = [
                self.gmx_binary,
                "pdb2gmx",
                "-f",
                protein_file,
                "-o",
                output_gro,
                "-p",
                output_top,
                "-ff",
                self.forcefield,
                "-water",
                self.water_model,
            ]

            if self.ignore_hydrogens:
                pdb2gmx_cmd.append("-ignh")

            self.logger.info(f"Running pdb2gmx for {protein_file}")
            self.logger.debug(f"Command: {' '.join(pdb2gmx_cmd)}")

            # pdb2gmx creates posre.itp in the current working directory, not the output directory
            result = run_cmd(pdb2gmx_cmd, cwd=protein_dir, check=True)

            self.logger.info("pdb2gmx completed successfully")
            if result.stdout:
                self.logger.debug(f"Output: {result.stdout}")

            return {
                "success": True,
                "output_gro": output_gro,
                "output_top": output_top,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }

        except subprocess.CalledProcessError as e:
            self.logger.error(f"pdb2gmx failed with return code {e.returncode}")
            self.logger.error(f"Error output: {e.stderr}")
            raise FatalError(
                f"GROMACS pdb2gmx failed: protein topology preparation is required for MD pipeline"
            )
        except Exception as e:
            self.logger.error(f"Unexpected error in run_pdb2gmx: {e}")
            raise FatalError(f"Protein topology preparation failed: {str(e)}")

    def run(self) -> dict:
        r"""
        Main method to prepare protein topology.

        External commands called:
            - gmx pdb2gmx: Convert PDB to GROMACS format with force field assignment

        Returns:
            dict: Updated file_paths dictionary with protein_gro, protein_top, and posre_itp paths
        """
        step_logger = log_step_execution("Protein Topology", self.logger)
        with step_logger:
            with step_logger.timing("pdb2gmx"):
                result = self._run_pdb2gmx()

            with step_logger.timing("topology_generation"):
                self.file_paths["protein_gro"] = result["output_gro"]
                self.file_paths["protein_top"] = result["output_top"]

                # pdb2gmx also creates posre.itp in the working directory (now protein_dir)
                protein_dir = self.file_paths["protein_dir"]
                posre_itp = os.path.join(protein_dir, "posre.itp")
                if os.path.exists(posre_itp):
                    self.file_paths["posre_itp"] = posre_itp
                else:
                    self.logger.warning(
                        f"posre.itp not found at expected location: {posre_itp}\n"
                        "This file is required for MD simulations with position restraints."
                    )

                self.logger.info("\n=== Output Files Created ===")
                self.logger.info(f"Protein GRO file: {result['output_gro']}")
                self.logger.info(f"Topology file: {result['output_top']}")
                if "posre_itp" in self.file_paths:
                    self.logger.info(
                        f"Position restraints file: {self.file_paths['posre_itp']}"
                    )

            return self.file_paths
