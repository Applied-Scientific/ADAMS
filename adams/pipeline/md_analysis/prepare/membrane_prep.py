"""
Membrane System Preparation Module - Transmembrane Protein MD Pipeline

This module prepares transmembrane protein systems for MD simulation.

Supported pathway:
1. **Pre-built import** (required): Accept a complete membrane system from
   CHARMM-GUI or other builders (GRO + TOP files). Validates the system,
   creates index groups, generates multi-stage position restraints, patches
   the topology, and runs energy minimization.

INPUTS (from file_paths dictionary):
    Required:
        - membrane_system_gro: Pre-built system GRO (protein + lipids + water + ions)
        - membrane_system_top: Pre-built system topology
        - membrane_dir: Working directory for membrane preparation

OUTPUTS (added to file_paths dictionary):
    - membrane_min_gro: Energy-minimized membrane system GRO
    - membrane_top: Final system topology (patched with restraint variants)
    - membrane_ndx: Index file with Protein/Membrane/Solvent_and_ions groups
    - membrane_posre: Base position restraint file path

EXTERNAL COMMANDS:
    - gmx grompp: Prepare minimization TPR
    - gmx mdrun: Run energy minimization
    - gmx make_ndx: Create index groups
"""

import os
import shutil

from ....error_handling import FatalError
from ....logger_utils import get_logger, log_step_execution
from ..shared import (
    GromacsContext,
    GromppWarningPolicy,
    collect_posre_includes,
    copy_mdp_files,
    detect_lipid_resnames_in_gro,
    detect_lipid_resnames_in_top,
    generate_posre_with_fc,
    get_ndx_group_index,
    get_membrane_mdp_dir,
    make_membrane_index,
    patch_topology_staged_restraints,
    posre_variant_path,
    run_grompp,
    validate_topology_includes,
    validate_required_file_paths,
)
from ..shared.constants import RESTRAINT_FC_VALUES


class MembranePrep:
    """
    Prepare a transmembrane protein system for MD simulation.

    Supported pathway:
    1. Pre-built system import (CHARMM-GUI / external builder)
    """

    def __init__(
        self,
        file_paths,
        forcefield="charmm36m",
        water_model="tip3p",
        forcefield_preset=None,
        ignore_hydrogens=True,
        water_margin=1.5,
        ion_conc=0.15,
        pname="K",
        nname="CL",
        grompp_warning_policy=None,
    ):
        """
        Args:
            file_paths: dict with paths. Must include:
                membrane_system_gro, membrane_system_top, membrane_dir
            forcefield: GROMACS force field (default: "charmm36m"). Reserved for
                future membrane-builder integrations.
            water_model: Water model (default: "tip3p").
            forcefield_preset: Force field preset name (overrides forcefield/water_model).
            ignore_hydrogens: Ignore existing H atoms in pdb2gmx (default: True).
            water_margin: Water box margin in nm (default: 1.5 nm, larger for membrane).
            ion_conc: Ion concentration in mol/L (default: 0.15 M).
            pname: Cation name (default: "K").
            nname: Anion name (default: "CL").
            grompp_warning_policy: Optional GromppWarningPolicy for grompp calls.
        """
        self.logger = get_logger()

        self.grompp_warning_policy = (
            grompp_warning_policy
            if grompp_warning_policy is not None
            else GromppWarningPolicy()
        )

        if file_paths is None:
            raise ValueError("file_paths dictionary is required.")
        self.file_paths = file_paths

        # Determine pathway
        self.prebuilt = bool(
            file_paths.get("membrane_system_gro")
            and file_paths.get("membrane_system_top")
        )

        self.validate_files()

        self.gromacs_path = file_paths["gromacs_path"]
        self.ambertools_path = file_paths["ambertools_path"]
        self.ctx = GromacsContext.from_file_paths(
            file_paths,
            gpu=False,
            grompp_warning_policy=grompp_warning_policy,
        )
        self.grompp_warning_policy = self.ctx.grompp_warning_policy
        self.gmx_binary = self.ctx.gmx_binary
        self.gromacs_binary_type = self.ctx.binary_type

        self.forcefield = forcefield
        self.water_model = water_model
        self.forcefield_preset = forcefield_preset
        self.ignore_hydrogens = ignore_hydrogens
        self.water_margin = water_margin
        self.ion_conc = ion_conc
        self.pname = pname
        self.nname = nname

    def validate_files(self):
        """Validate required keys in file_paths based on the selected pathway."""
        validate_required_file_paths(
            self.file_paths,
            ["membrane_dir"],
        )
        membrane_dir = self.file_paths["membrane_dir"]
        os.makedirs(membrane_dir, exist_ok=True)

        if self.prebuilt:
            gro = self.file_paths["membrane_system_gro"]
            top = self.file_paths["membrane_system_top"]
            if not os.path.exists(gro):
                raise FileNotFoundError(f"Pre-built membrane GRO not found: {gro}")
            if not os.path.exists(top):
                raise FileNotFoundError(f"Pre-built membrane TOP not found: {top}")
        else:
            raise FatalError(
                "Membrane preparation requires a pre-built bilayer system. "
                "Provide both membrane_system_gro and membrane_system_top "
                "(e.g., from CHARMM-GUI). Building from an oriented protein "
                "without bilayer embedding is not supported for transmembrane MD."
            )

    def _copy_topology_tree(self, source_dir, target_dir):
        """Copy the full topology include tree root to preserve relative includes."""
        for root, _, files in os.walk(source_dir):
            rel_root = os.path.relpath(root, source_dir)
            dst_root = (
                target_dir if rel_root == "." else os.path.join(target_dir, rel_root)
            )
            os.makedirs(dst_root, exist_ok=True)
            for fname in files:
                src = os.path.join(root, fname)
                dst = os.path.join(dst_root, fname)
                shutil.copy2(src, dst)

    def _get_working_gro(self, membrane_dir):
        """Return current working GRO path for downstream prep stages."""
        return self.file_paths.get(
            "membrane_system_gro_work",
            os.path.join(membrane_dir, "system.gro"),
        )

    def run(self) -> dict:
        """
        Main entry point. Prepares the membrane system and returns updated file_paths.

        Returns:
            dict: Updated file_paths with membrane_min_gro, membrane_top,
                  membrane_ndx, and membrane_posre.
        """
        step_logger = log_step_execution("Membrane Preparation", self.logger)
        with step_logger:
            membrane_dir = os.path.abspath(self.file_paths["membrane_dir"])
            os.makedirs(membrane_dir, exist_ok=True)

            if self.prebuilt:
                with step_logger.timing("import_prebuilt"):
                    self._import_prebuilt_system(membrane_dir)
            else:
                raise FatalError(
                    "Unsupported membrane workflow: missing pre-built membrane_system_gro/top."
                )

            with step_logger.timing("create_index"):
                self._create_membrane_index(membrane_dir)

            with step_logger.timing("generate_restraints"):
                self._generate_restraints(membrane_dir)

            with step_logger.timing("patch_topology"):
                self._patch_topology_restraints(membrane_dir)

            with step_logger.timing("energy_minimize"):
                self._energy_minimize(membrane_dir)

            self.logger.info("\n=== Membrane Preparation Complete ===")
            self.logger.info(f"Minimized GRO: {self.file_paths['membrane_min_gro']}")
            self.logger.info(f"Topology: {self.file_paths['membrane_top']}")
            self.logger.info(f"Index: {self.file_paths['membrane_ndx']}")

            return self.file_paths

    # ------------------------------------------------------------------
    # Pre-built system import
    # ------------------------------------------------------------------

    def _import_prebuilt_system(self, membrane_dir):
        """Import and validate a pre-built membrane system."""
        src_gro = os.path.abspath(self.file_paths["membrane_system_gro"])
        src_top = os.path.abspath(self.file_paths["membrane_system_top"])
        src_top_dir = os.path.dirname(src_top)

        dst_gro = os.path.join(membrane_dir, "system.gro")
        rel_top = os.path.relpath(src_top, src_top_dir)
        dst_top = os.path.join(membrane_dir, rel_top)

        # Preserve full topology directory tree (e.g., toppar/ includes).
        self._copy_topology_tree(src_top_dir, membrane_dir)
        shutil.copy2(src_gro, dst_gro)
        if not os.path.exists(dst_top):
            raise FileNotFoundError(
                f"Copied topology is missing expected top file: {dst_top}"
            )

        missing_includes = validate_topology_includes(dst_top)
        if missing_includes:
            details = "\n".join(f"  - {entry}" for entry in missing_includes[:20])
            extra = ""
            if len(missing_includes) > 20:
                extra = f"\n  ... and {len(missing_includes) - 20} more"
            raise FatalError(
                "Pre-built topology has unresolved #include paths after import. "
                "For membrane workflows, all topology dependencies must be "
                "present under the provided topology directory tree.\n"
                f"{details}{extra}"
            )

        self._validate_membrane_system(dst_gro, dst_top)

        self.file_paths["membrane_system_gro_work"] = dst_gro
        self.file_paths["membrane_top"] = dst_top
        self.logger.info(f"Imported pre-built membrane system to {membrane_dir}")

    def _validate_membrane_system(self, gro_path, top_path):
        """Validate that the system contains protein, lipids, and solvent."""
        lipids = detect_lipid_resnames_in_gro(gro_path)
        if not lipids:
            lipids_top = detect_lipid_resnames_in_top(top_path)
            if not lipids_top:
                raise FatalError(
                    "No lipid residues detected in the membrane system. "
                    "Verify that the GRO and topology contain lipid molecules "
                    f"(checked for: POPC, POPE, DPPC, CHL1, etc.)."
                )
            lipids = lipids_top

        self.logger.info(f"Validated membrane system - lipids found: {sorted(lipids)}")

        # Check for protein
        has_protein = False
        with open(gro_path, "r") as fh:
            for line in fh.readlines()[2:-1]:
                if len(line) >= 10:
                    resname = line[5:10].strip()
                    if resname in (
                        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
                        "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
                        "THR", "TRP", "TYR", "VAL", "HIE", "HID", "HIP", "HSE",
                        "HSD", "HSP",
                    ):
                        has_protein = True
                        break
        if not has_protein:
            self.logger.warning(
                "No standard amino acid residues detected in the GRO file. "
                "Verify that the system contains a protein."
            )

    # ------------------------------------------------------------------
    # Index creation
    # ------------------------------------------------------------------

    def _create_membrane_index(self, membrane_dir):
        """Create GROMACS index file with Protein/Membrane/Solvent_and_ions groups."""
        gro_path = self._get_working_gro(membrane_dir)
        ndx_path = os.path.join(membrane_dir, "index.ndx")

        # Detect lipid residues
        lipids = detect_lipid_resnames_in_gro(gro_path)
        if not lipids:
            # For pre-built systems, try the topology
            top_path = self.file_paths.get("membrane_top")
            if top_path:
                lipids = detect_lipid_resnames_in_top(top_path)

        if not lipids:
            raise FatalError(
                "Could not detect lipid residues in membrane GRO/TOP, so a membrane-compatible "
                "index cannot be generated. Verify lipid residue naming in the input system."
            )

        make_membrane_index(self.gmx_binary, gro_path, ndx_path, lipids)
        self._validate_membrane_index_groups(ndx_path)

        self.file_paths["membrane_ndx"] = ndx_path

    def _validate_membrane_index_groups(self, ndx_path):
        """Ensure index contains all groups required by membrane MDP tc-grps."""
        required_groups = ("Protein", "Membrane", "Solvent_and_ions")
        missing = []
        for group in required_groups:
            try:
                get_ndx_group_index(ndx_path, group)
            except ValueError:
                missing.append(group)

        if missing:
            raise FatalError(
                f"Membrane index missing required group(s): {missing}. "
                "Expected groups: Protein, Membrane, Solvent_and_ions."
            )

    # ------------------------------------------------------------------
    # Position restraints
    # ------------------------------------------------------------------

    def _generate_restraints(self, membrane_dir):
        """
        Generate FC-variant restraints from existing POSRES include files.

        For scientific/topological correctness, membrane workflows require an
        input topology that already defines POSRES includes in molecule scopes.
        """
        top_path = self.file_paths["membrane_top"]
        posre_paths = collect_posre_includes(top_path)
        if not posre_paths:
            raise FatalError(
                "No POSRES include files were found in the membrane topology graph. "
                "This workflow requires pre-defined molecule-local POSRES includes "
                "(e.g., CHARMM-GUI topologies with posre_* files)."
            )

        generated = []
        for base_posre in posre_paths:
            if not os.path.exists(base_posre):
                raise FatalError(f"Required restraint file not found: {base_posre}")
            for fc in RESTRAINT_FC_VALUES:
                variant_path = posre_variant_path(base_posre, fc)
                if not os.path.exists(variant_path):
                    generate_posre_with_fc(base_posre, variant_path, fc)
                    generated.append(variant_path)
                    self.logger.info(
                        f"Generated restraint variant {os.path.basename(variant_path)} "
                        f"from {os.path.basename(base_posre)} (FC={fc:.0f})"
                    )

        # Canonical base POSRES path for file_paths (membrane_posre_files holds the full list).
        self.file_paths["membrane_posre"] = posre_paths[0]
        self.file_paths["membrane_posre_files"] = posre_paths
        self.file_paths["membrane_posre_variants"] = generated

    # ------------------------------------------------------------------
    # Topology patching
    # ------------------------------------------------------------------

    def _patch_topology_restraints(self, membrane_dir):
        """Add multi-stage restraint #ifdef blocks to the topology."""
        top_path = self.file_paths["membrane_top"]
        try:
            patch_topology_staged_restraints(top_path, strict=True)
        except (FileNotFoundError, ValueError) as exc:
            raise FatalError(
                "Failed to patch membrane topology with staged restraint blocks. "
                "Ensure the imported topology and includes are complete and contain "
                "valid #ifdef POSRES sections."
            ) from exc
        self.logger.info("Patched topology with multi-stage restraint includes")

    # ------------------------------------------------------------------
    # Energy minimization
    # ------------------------------------------------------------------

    def _energy_minimize(self, membrane_dir):
        """Run energy minimization on the membrane system."""
        gro_path = self._get_working_gro(membrane_dir)
        top_path = self.file_paths["membrane_top"]
        ndx_path = self.file_paths["membrane_ndx"]

        mdp_dir = get_membrane_mdp_dir()
        copy_mdp_files(mdp_dir, ["membrane_min.mdp"], membrane_dir)

        self.ctx.run_stage(
            "min",
            "membrane_min.mdp",
            os.path.basename(gro_path),
            os.path.basename(ndx_path),
            os.path.relpath(top_path, membrane_dir),
            cwd=membrane_dir,
        )

        min_gro = os.path.join(membrane_dir, "min.gro")
        if not os.path.exists(min_gro):
            raise FatalError(
                "Energy minimization did not produce min.gro. "
                "Check GROMACS output for errors."
            )

        self.file_paths["membrane_min_gro"] = min_gro
        self.logger.info("Membrane energy minimization completed successfully")
