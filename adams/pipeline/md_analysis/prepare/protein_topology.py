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
    - protein_file: Path to input protein PDB file (must be protonated)
    - protein_dir: Directory for output files
    - gromacs_path: Path to GROMACS installation
    - ambertools_path: Path to AmberTools installation

OUTPUTS (added to file_paths dictionary):
    - protein_gro: Protein structure in GRO format
    - protein_top: Protein topology file (topol.top)
    - posre_itp: First position restraints file (posre.itp or first chain-specific posre)
    - posre_itp_files: List of all detected posre ITP files

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

from ....error_handling import FatalError
from ....logger_utils import get_logger, log_step_execution
from ....utils import run_cmd
from ..shared.forcefield_presets import get_forcefield_and_water, validate_forcefield_and_water
from ..shared import get_gromacs_binary, resolve_protein_topology_assets


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
        forcefield="amber99sb-ildn",
        water_model="tip3p",
        forcefield_preset=None,
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
            forcefield (str): GROMACS force field (default: "amber99sb-ildn"). Ignored if forcefield_preset is set.
            water_model (str): Water model (default: "tip3p"). Ignored if forcefield_preset is set.
            forcefield_preset (str, optional): Amber preset name. If set, overrides forcefield and water_model.
                One of: ff19sb_opc, ff14sb_tip3p (alias of amber14sb_tip3p),
                amber14sb_tip3p, ff99sb_ildn_tip3p, a99sb_disp, charmm36m.
            ignore_hydrogens (bool): Whether to ignore existing hydrogens so pdb2gmx adds forcefield ones (default: True; use True for protonated PDBs).
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

        normalized_forcefield_preset = (
            forcefield_preset.strip().lower()
            if isinstance(forcefield_preset, str) and forcefield_preset.strip()
            else None
        )
        if normalized_forcefield_preset == "auto":
            normalized_forcefield_preset = None

        self._forcefield_preset_name = normalized_forcefield_preset
        if normalized_forcefield_preset:
            self.forcefield, self.water_model = get_forcefield_and_water(
                normalized_forcefield_preset, water_model
            )
            self.logger.info(
                f"Using force field preset '{normalized_forcefield_preset}' -> ff={self.forcefield}, water={self.water_model}"
            )
        else:
            self.forcefield = forcefield
            self.water_model = water_model
        # Keep water-model handling deterministic and validate FF/water availability
        # before launching pdb2gmx (fail early with actionable diagnostics).
        self.water_model = (self.water_model or "tip3p").strip().lower()
        ff_dir, self.water_model = validate_forcefield_and_water(
            self.forcefield,
            self.water_model,
            self.gmx_binary,
        )
        self.logger.info(
            f"Validated force field '{self.forcefield}' in {ff_dir} with water model '{self.water_model}'."
        )
        self.ignore_hydrogens = ignore_hydrogens

    def validate_files(self):
        """
        Validate required keys exist in file_paths.

        Checks that:
        - protein_file exists in file_paths
        - protein_dir exists in file_paths

        Raises:
            ValueError: If required paths are missing from file_paths
        """
        if not self.file_paths.get("protein_file"):
            raise ValueError(
                "protein_file required in file_paths.\n"
                f"Available keys: {list(self.file_paths.keys())}"
            )
        protein_file = self.file_paths.get("protein_file")
        if protein_file:
            base = os.path.splitext(os.path.basename(protein_file))[0]
            has_pqr = os.path.exists(os.path.splitext(protein_file)[0] + ".pqr")
            if "_protonated" not in base and not has_pqr:
                raise ValueError(
                    "protein_file must be protonated (run_protonate_receptor). "
                    "Expected *_protonated.pdb (and optional matching .pqr)."
                )

        protein_dir = self.file_paths.get("protein_dir")
        if not protein_dir:
            raise ValueError(
                "protein_dir required in file_paths.\n"
                f"Available keys: {list(self.file_paths.keys())}\n"
                "Use build_file_paths() with md_workdir to create directory structure."
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
            forcefield (str): GROMACS force field (default: "amber99sb-ildn")
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
            err_out = (e.stderr or "").strip() or (e.stdout or "").strip()
            if err_out:
                self.logger.error("pdb2gmx output:\n%s", err_out)
            hint = ""
            if not self.ignore_hydrogens:
                hint = " If the PDB is already protonated, try run_protein_topology(..., ignore_hydrogens=True)."
            raise FatalError(
                f"GROMACS pdb2gmx failed: protein topology preparation is required for MD pipeline.{hint}"
                + (f"\nGROMACS output:\n{err_out}" if err_out else "")
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
                self.file_paths["water_model"] = self.water_model

                # pdb2gmx can produce either a single posre.itp or chain-specific posre_*.itp
                # referenced from included molecule topology files.
                topology_assets = resolve_protein_topology_assets(
                    protein_top=result["output_top"],
                    explicit_posre=None,
                    protein_dir=self.file_paths["protein_dir"],
                    root_path=None,
                    logger=self.logger,
                )
                deduped_posres = topology_assets["posre_files"]

                if deduped_posres:
                    self.file_paths["posre_itp"] = deduped_posres[0]
                    self.file_paths["posre_itp_files"] = deduped_posres
                else:
                    self.logger.warning(
                        "No protein position-restraint ITP files were detected. "
                        "Expected either posre.itp or POSRES includes in topology files."
                    )

                self.logger.info("\n=== Output Files Created ===")
                self.logger.info(f"Protein GRO file: {result['output_gro']}")
                self.logger.info(f"Topology file: {result['output_top']}")
                if "posre_itp" in self.file_paths:
                    self.logger.info(
                        f"Position restraints file: {self.file_paths['posre_itp']}"
                    )
                if self.file_paths.get("posre_itp_files"):
                    self.logger.info(
                        "Detected protein restraint files: "
                        + ", ".join(self.file_paths["posre_itp_files"])
                    )

            return self.file_paths
