"""
Ligand Preparation Module - MD Analysis Pipeline Step 2

This module processes docking results, selects top ligands, and prepares them
for MD simulation by combining with protein, solvating, adding ions, and
performing energy minimization.

POSITION IN PIPELINE:
    Step 2 of 4: run_lig_prepare
    - Requires outputs from run_protein_topology (protein_gro, protein_top)
    - Must be executed before run_md_simulation
    - Can be skipped if prepared poses already exist

INPUTS (from file_paths dictionary):
    - docking_csv: Path to docking results CSV file
    - ligand_input: Ligand structure input (SMILES string, CSV, SDF, MOL2, or directory)
    - protein_gro: Path to protein GRO file (from run_protein_topology)
    - protein_top: Path to protein topology file (from run_protein_topology)
    - poses_dir: Directory to store prepared poses
    - gromacs_path: Path to GROMACS installation
    - ambertools_path: Path to AmberTools installation

OUTPUTS (added to file_paths dictionary):
    - poses_dir: Updated with prepared pose subdirectories (each containing min.gro)

KEY FUNCTIONALITY:
    - Selects top N ligands per grid from docking results
    - Generates ligand topology using ACPYPE (from MOL2 files)
    - Combines protein and ligand into complex
    - Solvates system with water molecules
    - Adds ions to neutralize and set ionic strength
    - Performs energy minimization
    - Creates index files for restraints

LIGAND FORCE FIELDS:
    - Ligands use GAFF/GAFF2 (and charge model) explicitly: atom_type ("gaff" | "gaff2", default
      "gaff2") and charge_type ("bcc" = AM1-BCC, default, or "gas"). Set in run_lig_prepare/LigPrepare.
    - If the chosen charge method fails, ACPYPE can optionally retry with gas
      (retry_with_gas_on_failure=False by default for explicit charge control).

GROMPP WARNING POLICY:
    All grompp invocations run WITHOUT -maxwarn when no warnings are pre-approved.
    If grompp produces warnings, a GromppWarningError is raised so the calling
    agent/user can review each warning individually.  Approved warning
    fingerprints can be passed via grompp_warning_policy (or approved_grompp_warnings
    in the agent tool) to classify repeated warning types across poses/runs.
    When pre-approved fingerprints exist, grompp runs with a high -maxwarn bound
    so it can complete in one pass; every emitted warning is still validated
    against the policy and unapproved ones raise GromppWarningError.  No blanket
    -maxwarn override (e.g. -maxwarn 100).

STANDALONE / EXTERNAL DATA:
    If starting from your own docking results (not this pipeline's docking):
    - Docking CSV must have columns: ligand_id, ID, grid_id, pose_id, affinity,
      COM_x, COM_y, COM_z, MolWt. Optional: Parent_ID, SMILES.
    - Either (1) include a SMILES column in the docking CSV (per row), then
      ligand_input is still required by file_paths but SMILES are taken from the
      CSV; or (2) omit SMILES and provide ligand_input (e.g. CSV with ID and
      SMILES) whose IDs match the docking CSV "ID" column (get_smiles_by_id
      also tries base ID before "__" and Parent_ID). PDBQT files must be at
      paths derived from the CSV path: .../poses/ligand_{ligand_id}_pocket_
      {grid_id}_docked.pdbqt (or .../summaries/ -> .../poses/). If SMILES
      lookup fails, structure-only restore is used and charge is taken from the
      MOL2 (no extra code paths required).

EXTERNAL COMMANDS:
    - acpype: Generate GROMACS-compatible ligand topology from MOL2
    - gmx editconf: Create simulation box around complex
    - gmx solvate: Add water molecules to the system
    - gmx grompp: Prepare TPR files for ion addition and minimization
    - gmx genion: Add ions to neutralize and set ionic strength (solvent group auto-detected from topology; post-genion neutrality and ionic-strength sanity checks applied)
    - gmx make_ndx: Create index groups for restraints
    - gmx genrestr: Generate position restraints for ligand
    - gmx mdrun: Run energy minimization
"""

import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

import pandas as pd

from ....common_utils import get_cpu_count
from ....error_handling import setup_sigint_handler
from ....logger_utils import (
    get_logger,
    log_step_execution,
)
from ....utils.parallel_executor import ParallelExecutor, ResourceConfig
from ....utils.parallel_plans import build_gpu_phases
from .lig_prepare_workers import acpype_and_setup_single_pose, run_min_only
from .ligand_resolver import LigandResolver
from .ligand_ops import extract_pose_from_pdbqt
from ..shared import (
    GromacsContext,
    GromppWarningPolicy,
    get_solvent_box_for_water_model,
    get_solvent_box_path,
)

# ACPYPE -a atom type: gaff (GAFF), gaff2 (GAFF2), amber (AMBER14SB), amber2 (AMBER14SB+GAFF2)
ACPYPE_ATOM_TYPES = ("gaff", "gaff2", "amber", "amber2")
ACPYPE_CHARGE_TYPES = ("bcc", "gas")
SELECTION_SCOPES = ("per_grid", "per_parent_per_grid")


def _cap_parallel_workers(desired_workers: int, max_jobs: int) -> int:
    """Cap a worker count by an explicit user override when provided."""
    desired = max(1, int(desired_workers))
    if max_jobs and max_jobs > 0:
        return max(1, min(desired, int(max_jobs)))
    return desired


def _per_worker_num_cores(total_cores: int, n_workers: int) -> int:
    """Split the available CPU budget across concurrent LigPrepare workers."""
    return max(1, int(total_cores) // max(1, int(n_workers)))


class LigPrepare:
    def __init__(
        self,
        file_paths,
        tops: Optional[int] = 3,
        selection_scope: str = "per_grid",
        charge_type: str = "bcc",
        atom_type: str = "gaff2",
        retry_with_gas_on_failure: bool = False,
        num_cores: int = None,
        num_gpus: int = -1,
        max_jobs: int = 0,
        water_margin: float = 1.0,
        ion_conc: float = 0.15,
        pname: str = "K",
        nname: str = "CL",
        water_model: str = None,
        grompp_warning_policy: "GromppWarningPolicy" = None,
    ):
        r"""
        Args:
            file_paths: dict: File paths dictionary - single source of truth (required).
                Must include:
                - docking_csv: Path to docking results CSV file
                - ligand_input: Ligand structure input (SMILES string, CSV, SDF, MOL2, or directory)
                - protein_gro: Path to protein GRO file
                - protein_top: Path to protein topology file
                - poses_dir: Directory to store prepared poses
                - gromacs_path: Path to GROMACS installation
                - ambertools_path: Path to AmberTools installation
                - gromacs_binary_type: Type of GROMACS binary (from discover_paths)
            tops: int: Cap for selected docking rows.
                - per_grid mode: top rows per grid/pocket
                - per_parent_per_grid mode: top rows per parent ligand per grid/pocket
                Set tops=None or tops<=0 to disable cap.
            selection_scope: "per_grid" (default) or "per_parent_per_grid".
            charge_type: str: Charge type for Antechamber: bcc (AM1-BCC, default) or gas.
            atom_type: str: Ligand parameters: "gaff" (GAFF), "gaff2" (GAFF2, default), "amber", or "amber2".
            retry_with_gas_on_failure: bool: If True, if the chosen charge method fails, retry with gas.
                Default is False so the run fails without overriding the user's charge_type.
            num_cores: int: Number of CPU cores (None uses all-1).
            num_gpus: int: Number of GPUs (-1 = use all available; 0 = CPU-only; N = use N GPUs; default: -1).
            max_jobs: int: Maximum concurrent LigPrepare jobs (0 = auto).
            water_margin: float: Water box margin in nm (default: 1.0 nm).
            ion_conc: float: Ion concentration in mol/L (default: 0.15 M).
            pname: str: Cation name (default: K).
            nname: str: Anion name (default: CL).
            water_model: str: Optional. Water model for solvation (e.g. tip3p, spc). If None, uses
                file_paths["water_model"] from run_protein_topology, or "tip3p". Set this when
                starting from lig_prepare with pre-existing topology so solvation matches the topology.
            grompp_warning_policy: GromppWarningPolicy: Optional. Shared warning policy for grompp
                calls. If None, a new strict policy (no pre-approved warnings) is created. Pass a
                policy with pre-approved fingerprints to classify known warning types across
                repeated calls. Unapproved warnings still fail fast; when all warnings are
                approved, grompp runs with a high -maxwarn bound and every warning is
                validated against the policy (no blanket bypass).
                The same policy instance should be reused across LigPrepare and Gro for
                consistent warning categorization across pipeline steps.
        """
        self.logger = get_logger()

        self.grompp_warning_policy = (
            grompp_warning_policy
            if grompp_warning_policy is not None
            else GromppWarningPolicy()
        )

        # Set up SIGINT handler for clean shutdown on Ctrl+C
        setup_sigint_handler()

        if file_paths is None:
            raise ValueError(
                "file_paths dictionary is required. Use build_file_paths() and discover_paths() first."
            )
        self.file_paths = file_paths

        self.validate_files()

        self.docking_csv = file_paths["docking_csv"]
        self.gromacs_path = file_paths["gromacs_path"]
        self.ambertools_path = file_paths["ambertools_path"]
        gpu = file_paths.get("gromacs_binary_type", "standard") == "cuda"
        self.ctx = GromacsContext.from_file_paths(
            file_paths,
            gpu=gpu,
            num_gpus=num_gpus,
            grompp_warning_policy=grompp_warning_policy,
        )
        self.grompp_warning_policy = self.ctx.grompp_warning_policy
        self.gromacs_binary_type = self.ctx.binary_type
        self.gmx_binary = self.ctx.gmx_binary
        self.num_gpus = self.ctx.num_gpus

        self.md_workdir = file_paths.get("md_root", ".")

        # When docking CSV already has a SMILES column, skip LigandResolver and rely on
        # per-pose SMILES stored in _prepwork. Otherwise resolve from ligand_input.
        docking_csv = file_paths["docking_csv"]
        docking_has_smiles = False
        if docking_csv and os.path.isfile(docking_csv):
            try:
                df_check = pd.read_csv(docking_csv, nrows=0)
                if "SMILES" in df_check.columns:
                    df_sample = pd.read_csv(docking_csv)
                    docking_has_smiles = (
                        "SMILES" in df_sample.columns
                        and df_sample["SMILES"].notna().any()
                        and (df_sample["SMILES"].astype(str).str.strip() != "").any()
                    )
            except Exception:
                pass
        if docking_has_smiles:
            self.smiles_file = None
            self.logger.info(
                "Docking CSV contains SMILES column; using per-pose SMILES (skipping LigandResolver)."
            )
        else:
            self.ligand_resolver = LigandResolver()
            ligand_output_dir = os.path.join(self.md_workdir, "ligand_resolution")
            resolution_result = self.ligand_resolver.resolve_ligand_structures(
                ligand_input=file_paths["ligand_input"],
                docking_csv=docking_csv,
                output_dir=ligand_output_dir,
            )
            self.smiles_file = resolution_result["smiles_csv_path"]
            self.logger.info(
                f"Resolved ligand structures from {resolution_result['source']}: "
                f"{self.smiles_file}"
            )

        normalized_scope = (selection_scope or "per_grid").strip().lower()
        if normalized_scope not in SELECTION_SCOPES:
            raise ValueError(
                f"selection_scope must be one of {SELECTION_SCOPES} (got {selection_scope!r})."
            )
        self.selection_scope = normalized_scope

        if tops is None:
            self.tops = None
        else:
            try:
                tops_int = int(tops)
            except (TypeError, ValueError):
                raise ValueError(
                    f"tops must be an integer or None (got {tops!r})."
                ) from None
            self.tops = tops_int if tops_int > 0 else None

        if self.tops is None:
            self.logger.info(
                "Pose selection: scope=%s, cap=ALL (no tops limit).",
                self.selection_scope,
            )
        else:
            self.logger.info(
                "Pose selection: scope=%s, cap=%d.",
                self.selection_scope,
                self.tops,
            )
        charge_type_lower = (charge_type or "bcc").strip().lower()
        if charge_type_lower not in ACPYPE_CHARGE_TYPES:
            raise ValueError(
                f"charge_type must be one of {ACPYPE_CHARGE_TYPES} (got {charge_type!r})."
            )
        self.charge_type = charge_type_lower
        self.retry_with_gas_on_failure = retry_with_gas_on_failure
        atom_type_lower = (atom_type or "gaff2").strip().lower()
        if atom_type_lower not in ACPYPE_ATOM_TYPES:
            raise ValueError(
                f"atom_type must be one of {ACPYPE_ATOM_TYPES} (got {atom_type!r}). "
                "Use 'gaff' for GAFF or 'gaff2' for GAFF2 ligand parameters."
            )
        self.atom_type = atom_type_lower
        self.num_cores = num_cores if num_cores is not None else get_cpu_count()
        self.max_jobs = int(max_jobs) if max_jobs is not None else 0
        self.logger.info(
            f"Ligand parameters: {self.atom_type.upper()} (atom_type={self.atom_type}), charge_type={self.charge_type}, "
            f"retry_with_gas_on_failure={self.retry_with_gas_on_failure}"
        )

        self.water_margin = water_margin
        self.ion_conc = ion_conc
        self.pname = pname
        self.nname = nname

        # Water model must match protein topology (pdb2gmx -water). Prefer the
        # topology-derived value from file_paths; if the caller passes a water
        # model as well, it must agree.
        requested_water_model = (
            water_model.strip().lower()
            if isinstance(water_model, str) and water_model.strip()
            else None
        )
        topology_water_model = (
            file_paths["water_model"].strip().lower()
            if isinstance(file_paths.get("water_model"), str)
            and file_paths["water_model"].strip()
            else None
        )
        if (
            requested_water_model is not None
            and topology_water_model is not None
            and requested_water_model != topology_water_model
        ):
            raise ValueError(
                "Water model mismatch: ligand preparation was asked to use "
                f"'{requested_water_model}' but protein topology was prepared with "
                f"'{topology_water_model}'. Use a single consistent water model."
            )
        self.water_model = topology_water_model or requested_water_model or "tip3p"
        default_box = get_solvent_box_for_water_model(self.water_model)
        resolved_solvent = get_solvent_box_path(self.water_model, self.gmx_binary)
        if resolved_solvent:
            self.solvent_box = resolved_solvent
            self.logger.debug(f"Resolved solvent box to {self.solvent_box}")
        else:
            if not os.path.isabs(default_box) and not os.path.isfile(default_box):
                raise FileNotFoundError(
                    f"Solvent box file for water model '{self.water_model}' could not be found. "
                    f"Expected '{default_box}' in GROMACS share/top directory (derived from gmx binary). "
                    "Ensure GROMACS is installed with topology data (e.g. conda install gromacs) or set GMXDATA."
                )
            self.solvent_box = default_box

        self.root_path = os.getcwd()
        self.pose_tasks: List[Dict[str, Any]] = []
        self.pose_manifest_records: List[Dict[str, Any]] = []
        self.pose_manifest_by_name: Dict[str, Dict[str, Any]] = {}
        self.prepwork_failures: List[Dict[str, Any]] = []

    def validate_files(self):
        """
        Validate required keys exist in file_paths.

        Checks that:
        - Required keys exist in file_paths (docking_csv, ligand_input, protein_gro, protein_top, poses_dir)

        Raises:
            ValueError: If required paths are missing from file_paths
        """
        required_keys = [
            "docking_csv",
            "ligand_input",
            "protein_gro",
            "protein_top",
            "poses_dir",
        ]
        missing = [k for k in required_keys if not self.file_paths.get(k)]

        if missing:
            raise ValueError(
                f"Required paths missing from file_paths: {missing}\n"
                f"Available keys: {list(self.file_paths.keys())}\n"
                "Ensure previous pipeline steps have run or provide explicit paths."
            )

        poses_dir = self.file_paths["poses_dir"]
        os.makedirs(poses_dir, exist_ok=True)

    def run(self) -> dict:
        """
        Run ligand preparation workflow.

        External commands called:
            - acpype: Generate GROMACS-compatible ligand topology from MOL2
            - gmx editconf: Create simulation box around complex
            - gmx solvate: Add water molecules to the system
            - gmx grompp: Prepare TPR files for ion addition and minimization
            - gmx genion: Add ions to neutralize and set ionic strength
            - gmx make_ndx: Create index groups for restraints
            - gmx genrestr: Generate position restraints for ligand
            - gmx mdrun: Run energy minimization

        Returns:
            dict: Updated file_paths dictionary with prepared_poses list
        """
        step_logger = log_step_execution("Ligand Preparation", self.logger)
        with step_logger:
            if self.grompp_warning_policy._approved:
                approved_list = self.grompp_warning_policy.get_approved_with_descriptions()
                self.logger.info(
                    "Loaded %d pre-approved grompp warning fingerprint(s) from previous run(s) "
                    "for warning classification.",
                    len(approved_list),
                )
                for item in approved_list:
                    self.logger.info(
                        "  Pre-approved warning fingerprint: %s",
                        item["description"],
                    )
            with step_logger.timing("prepwork"):
                self._prepwork()
            with step_logger.timing("lig_prepare_batch"):
                self._run_lig_prepare_batch()

            # poses_dir is already in file_paths, no need to update
            return self.file_paths

    def _derive_parent_selection_ids(self, group: pd.DataFrame) -> pd.Series:
        """Return robust parent IDs for selection grouping."""
        fallback = group["ID"].astype(str).str.split("__").str[0]
        if "Parent_ID" not in group.columns:
            return fallback
        parent_raw = group["Parent_ID"]
        parent = parent_raw.astype(str).where(parent_raw.notna(), "")
        parent = parent.str.strip()
        return parent.where(parent != "", fallback)

    def _select_rows_for_md(self, group: pd.DataFrame) -> pd.DataFrame:
        """
        Select docking rows for MD according to selection_scope and tops.
        Expects `group` sorted by affinity ascending.
        """
        if self.selection_scope == "per_grid":
            selected = group if self.tops is None else group.head(self.tops)
            return selected.reset_index(drop=True)

        # per_parent_per_grid
        work = group.copy()
        work["_parent_selection_id"] = self._derive_parent_selection_ids(work)
        if self.tops is None:
            selected = work
        else:
            selected = work.groupby(
                "_parent_selection_id", as_index=False, sort=False, group_keys=False
            ).head(self.tops)
        return selected.drop(columns=["_parent_selection_id"], errors="ignore").reset_index(
            drop=True
        )

    @staticmethod
    def _optional(value: Any) -> Any:
        if value is None or pd.isna(value):
            return None
        return value

    @staticmethod
    def _normalize_parent_id(raw_parent: Any, ligand_id: str) -> str:
        parent = "" if raw_parent is None or pd.isna(raw_parent) else str(raw_parent).strip()
        return parent or str(ligand_id).split("__")[0]

    @staticmethod
    def _short_error(error_text: Optional[str], max_len: int = 500) -> str:
        if not error_text:
            return "unknown"
        line = str(error_text).strip().splitlines()[0]
        return line[:max_len]

    def _prepwork(self):
        """
        Prepare ligand poses from docking results and persist pose_manifest.csv.

        Reads docking CSV directly from file_paths["docking_csv"] and resolves
        source PDBQT paths relative to the CSV location.
        """
        docking_csv = self.file_paths["docking_csv"]
        output_poses_dir = self.file_paths["poses_dir"]
        os.makedirs(output_poses_dir, exist_ok=True)

        md_root = self.file_paths.get("md_root", os.path.dirname(output_poses_dir))
        top_label = "all" if self.tops is None else str(self.tops)
        top_by_grid_dir = os.path.join(md_root, f"top{top_label}_by_grid")
        os.makedirs(top_by_grid_dir, exist_ok=True)

        self.logger.info("Top-by-grid directory: %s", top_by_grid_dir)
        self.logger.info("Using docking CSV: %s", docking_csv)

        df = pd.read_csv(docking_csv)
        required_cols = [
            "ligand_id",
            "ID",
            "grid_id",
            "pose_id",
            "affinity",
            "COM_x",
            "COM_y",
            "COM_z",
            "MolWt",
        ]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Docking CSV missing required columns: {missing_cols}")

        optional_cols = [
            c
            for c in ("Parent_ID", "SMILES", "Variant_Type", "Variant_ID", "Conformer_Index")
            if c in df.columns
        ]
        df = df[required_cols + optional_cols]
        csv_dir = os.path.dirname(docking_csv)
        df["base_dir"] = csv_dir
        df = df.sort_values(by="affinity", ascending=True)

        pose_tasks: List[Dict[str, Any]] = []
        pose_manifest_records: List[Dict[str, Any]] = []
        prepwork_failures: List[Dict[str, Any]] = []

        for grid_id, group in df.groupby("grid_id"):
            top_df = self._select_rows_for_md(group).reset_index(drop=True)
            out_file = os.path.join(
                top_by_grid_dir,
                f"grid_{grid_id}_top{top_label}_{self.selection_scope}.csv",
            )
            top_df.to_csv(out_file, index=False)

            for rank_pos, row in enumerate(top_df.to_dict("records"), start=1):
                ligand_id = str(row["ligand_id"])
                grid_val = int(row["grid_id"])
                pose_id = int(row["pose_id"])
                ligand_name = str(row["ID"])
                parent_id = self._normalize_parent_id(row.get("Parent_ID"), ligand_name)
                base_dir = str(row["base_dir"])

                if "summaries" in base_dir:
                    docking_poses_dir = base_dir.replace("summaries", "poses")
                    source_pose_path = os.path.join(
                        docking_poses_dir,
                        f"ligand_{ligand_id}_pocket_{grid_val}_docked.pdbqt",
                    )
                    if not os.path.exists(source_pose_path):
                        search_poses_dir = docking_poses_dir.replace("production", "search")
                        source_pose_path = os.path.join(
                            search_poses_dir,
                            f"ligand_{ligand_id}_grid_{grid_val}_docked.pdbqt",
                        )
                else:
                    source_pose_path = os.path.join(
                        base_dir, f"ligand_{ligand_id}_pocket_{grid_val}_docked.pdbqt"
                    )

                pose_name = f"{ligand_name}_pocket_{grid_val}_top{rank_pos}"
                if not os.path.exists(source_pose_path):
                    prepwork_failures.append(
                        {
                            "phase": "selection_preflight",
                            "pose_name": pose_name,
                            "parent_id": parent_id,
                            "ligand_id": ligand_id,
                            "ID": ligand_name,
                            "grid_id": grid_val,
                            "rank": rank_pos,
                            "reason": f"source pose file not found: {source_pose_path}",
                            "source_pose_path": source_pose_path,
                        }
                    )
                    continue

                out_dir = os.path.join(output_poses_dir, pose_name)
                os.makedirs(out_dir, exist_ok=True)
                pose_pdbqt = os.path.join(out_dir, "ligand.pdbqt")
                extract_pose_from_pdbqt(source_pose_path, pose_pdbqt, pose_id)

                smiles_val = self._optional(row.get("SMILES"))
                if smiles_val is not None and str(smiles_val).strip():
                    smiles_lookup_value = str(smiles_val).strip()
                    smiles_is_direct = True
                else:
                    smiles_lookup_value = ligand_name
                    smiles_is_direct = False

                manifest_record = {
                    "pose_name": pose_name,
                    "ligand_id": ligand_id,
                    "ID": ligand_name,
                    "Parent_ID": parent_id,
                    "Variant_Type": self._optional(row.get("Variant_Type")) or "original",
                    "Variant_ID": self._optional(row.get("Variant_ID")) or "original_0",
                    "Conformer_Index": self._optional(row.get("Conformer_Index")),
                    "grid_id": grid_val,
                    "rank": rank_pos,
                    "affinity": self._optional(row.get("affinity")),
                    "source_pose_path": source_pose_path,
                    "pose_id": pose_id,
                    "pose_dir": out_dir,
                    "pose_pdbqt_path": pose_pdbqt,
                }
                pose_manifest_records.append(manifest_record)
                pose_tasks.append(
                    {
                        "pose": pose_name,
                        "out_dir": out_dir,
                        "pose_pdbqt": pose_pdbqt,
                        "smiles_lookup_value": smiles_lookup_value,
                        "smiles_is_direct": smiles_is_direct,
                        "manifest": manifest_record,
                    }
                )

        pose_manifest_path = os.path.join(output_poses_dir, "pose_manifest.csv")
        manifest_cols = [
            "pose_name",
            "ligand_id",
            "ID",
            "Parent_ID",
            "Variant_Type",
            "Variant_ID",
            "Conformer_Index",
            "grid_id",
            "rank",
            "affinity",
            "source_pose_path",
        ]
        pd.DataFrame(pose_manifest_records, columns=manifest_cols).to_csv(
            pose_manifest_path, index=False
        )

        self.file_paths["pose_manifest"] = pose_manifest_path
        self.pose_tasks = pose_tasks
        self.pose_list = [task["pose"] for task in pose_tasks]
        self.pose_manifest_records = pose_manifest_records
        self.pose_manifest_by_name = {
            rec["pose_name"]: rec for rec in pose_manifest_records
        }
        self.prepwork_failures = prepwork_failures

        self.logger.info(
            "Pose preflight complete: selected=%d, skipped_preflight=%d, manifest=%s",
            len(pose_tasks),
            len(prepwork_failures),
            pose_manifest_path,
        )
        if prepwork_failures:
            preview = prepwork_failures[:5]
            self.logger.warning(
                "Skipped %d pose(s) in preflight. Examples:\n%s",
                len(prepwork_failures),
                "\n".join(f"  - {item['pose_name']}: {item['reason']}" for item in preview),
            )

        if not pose_tasks:
            detail = "\n".join(
                f"  - {item['pose_name']}: {item['reason']}"
                for item in prepwork_failures[:10]
            )
            raise ValueError(
                "No valid poses available for MD preparation after preflight.\n"
                f"{detail}"
            )

    def _write_lig_prepare_reports(
        self,
        selected_tasks: List[Dict[str, Any]],
        phase1_results: List[Any],
        phase2_results: List[Any],
    ) -> Dict[str, Any]:
        pose_meta = {
            task["pose"]: task.get("manifest", {})
            for task in selected_tasks
        }

        phase1_failed = [r for r in phase1_results if not r.success]
        phase2_failed = [r for r in phase2_results if not r.success]
        phase2_success_ids = {r.task_id for r in phase2_results if r.success}

        failure_rows: List[Dict[str, Any]] = []
        for item in self.prepwork_failures:
            failure_rows.append(
                {
                    "phase": item.get("phase", "selection_preflight"),
                    "pose_name": item.get("pose_name"),
                    "parent_id": item.get("parent_id"),
                    "ligand_id": item.get("ligand_id"),
                    "ID": item.get("ID"),
                    "grid_id": item.get("grid_id"),
                    "rank": item.get("rank"),
                    "reason": item.get("reason"),
                    "source_pose_path": item.get("source_pose_path"),
                }
            )

        def add_parallel_failures(phase_name: str, failures: List[Any]) -> None:
            for result in failures:
                meta = pose_meta.get(result.task_id, {})
                failure_rows.append(
                    {
                        "phase": phase_name,
                        "pose_name": result.task_id,
                        "parent_id": meta.get("Parent_ID"),
                        "ligand_id": meta.get("ligand_id"),
                        "ID": meta.get("ID"),
                        "grid_id": meta.get("grid_id"),
                        "rank": meta.get("rank"),
                        "reason": self._short_error(result.error),
                        "source_pose_path": meta.get("source_pose_path"),
                    }
                )

        add_parallel_failures("phase1_acpype_setup", phase1_failed)
        add_parallel_failures("phase2_minimization", phase2_failed)

        parent_stats: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"selected": 0, "succeeded": 0, "failed": 0}
        )
        all_parent_ids = set()
        for item in self.prepwork_failures:
            parent_id = item.get("parent_id")
            if parent_id:
                all_parent_ids.add(parent_id)
                parent_stats[parent_id]["failed"] += 1
        for task in selected_tasks:
            meta = task.get("manifest", {})
            parent_id = meta.get("Parent_ID")
            if not parent_id:
                continue
            all_parent_ids.add(parent_id)
            parent_stats[parent_id]["selected"] += 1
            if task["pose"] in phase2_success_ids:
                parent_stats[parent_id]["succeeded"] += 1
            else:
                parent_stats[parent_id]["failed"] += 1

        dropped_parent_ids = sorted(
            parent_id
            for parent_id in all_parent_ids
            if parent_stats[parent_id]["succeeded"] == 0
        )

        parent_summary = [
            {
                "Parent_ID": parent_id,
                "selected": parent_stats[parent_id]["selected"],
                "succeeded": parent_stats[parent_id]["succeeded"],
                "failed": parent_stats[parent_id]["failed"],
                "status": "dropped"
                if parent_id in dropped_parent_ids
                else "kept",
            }
            for parent_id in sorted(all_parent_ids)
        ]

        phase1_selected = len(selected_tasks)
        phase1_succeeded = len([r for r in phase1_results if r.success])
        phase2_selected = len([r for r in phase1_results if r.success])
        phase2_succeeded = len(phase2_success_ids)
        overall_selected = phase1_selected + len(self.prepwork_failures)
        overall_succeeded = phase2_succeeded
        overall_failed = overall_selected - overall_succeeded

        summary_payload = {
            "selection_scope": self.selection_scope,
            "tops": self.tops,
            "pose_manifest": self.file_paths.get("pose_manifest"),
            "counts": {
                "preflight": {
                    "selected": overall_selected,
                    "skipped": len(self.prepwork_failures),
                },
                "phase1_acpype_setup": {
                    "selected": phase1_selected,
                    "succeeded": phase1_succeeded,
                    "failed": len(phase1_failed),
                },
                "phase2_minimization": {
                    "selected": phase2_selected,
                    "succeeded": phase2_succeeded,
                    "failed": len(phase2_failed),
                },
                "overall": {
                    "selected": overall_selected,
                    "succeeded": overall_succeeded,
                    "failed": overall_failed,
                },
            },
            "dropped_parent_ids": dropped_parent_ids,
            "parent_summary": parent_summary,
        }

        reports_dir = self.file_paths.get("reports_dir") or self.md_workdir
        os.makedirs(reports_dir, exist_ok=True)
        failures_path = os.path.join(reports_dir, "lig_prepare_failures.csv")
        summary_path = os.path.join(reports_dir, "lig_prepare_summary.json")

        failure_cols = [
            "phase",
            "pose_name",
            "parent_id",
            "ligand_id",
            "ID",
            "grid_id",
            "rank",
            "reason",
            "source_pose_path",
        ]
        pd.DataFrame(failure_rows, columns=failure_cols).to_csv(
            failures_path, index=False
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2)

        self.file_paths["lig_prepare_failures_path"] = failures_path
        self.file_paths["lig_prepare_summary_path"] = summary_path
        self.file_paths["dropped_parent_ids"] = dropped_parent_ids
        self.file_paths["prepared_pose_count"] = overall_succeeded
        self.file_paths["failed_pose_count"] = overall_failed
        return summary_payload

    def _run_lig_prepare_batch(self):
        """
        Run ligand preparation in two phases:
        1) ACPYPE + system setup (CPU-bound)
        2) energy minimization (GPU-aware)

        All poses are attempted in both phases where applicable. Partial failures
        are aggregated and reported; run fails only when zero poses succeed overall.
        """
        poses_dir = os.path.abspath(self.file_paths["poses_dir"])
        smiles_file = getattr(self, "smiles_file", None)
        if smiles_file:
            smiles_file = os.path.abspath(smiles_file)

        if not self.pose_tasks:
            raise ValueError("No poses available for ligand preparation.")

        tasks_acpype: List[Dict[str, Any]] = []
        for pose_task in self.pose_tasks:
            tasks_acpype.append(
                {
                    "pose": pose_task["pose"],
                    "out_dir": pose_task["out_dir"],
                    "pose_pdbqt": pose_task["pose_pdbqt"],
                    "smiles_lookup_value": pose_task["smiles_lookup_value"],
                    "smiles_is_direct": pose_task["smiles_is_direct"],
                    "manifest": pose_task["manifest"],
                    "poses_dir": poses_dir,
                    "smiles_file": smiles_file,
                    "file_paths": self.file_paths,
                    "charge_type": self.charge_type,
                    "atom_type": self.atom_type,
                    "retry_with_gas_on_failure": self.retry_with_gas_on_failure,
                    "ctx": self.ctx.to_task_params(),
                    "water_margin": self.water_margin,
                    "water_model": self.water_model,
                    "solvent_box": self.solvent_box,
                    "ion_conc": self.ion_conc,
                    "pname": self.pname,
                    "nname": self.nname,
                    "num_cores": self.num_cores,
                    "num_gpus": self.num_gpus,
                    "root_path": self.root_path,
                }
            )

        cpu_workers = _cap_parallel_workers(
            max(1, min(self.num_cores, len(tasks_acpype))),
            self.max_jobs,
        )
        phase1_task_num_cores = _per_worker_num_cores(self.num_cores, cpu_workers)
        for task in tasks_acpype:
            task["num_cores"] = phase1_task_num_cores
        self.logger.info(
            "LigPrep phase 1 (ACPYPE + setup): %d workers (CPU), %d poses",
            cpu_workers,
            len(tasks_acpype),
        )
        phase1_executor = ParallelExecutor(
            ResourceConfig(n_workers=cpu_workers, n_gpus=0, gpu_strategy="round_robin")
        )
        phase1_results = phase1_executor.run(
            acpype_and_setup_single_pose,
            tasks_acpype,
            task_id_fn=lambda t: t["pose"],
        )

        phase1_failed = [r for r in phase1_results if not r.success]
        if phase1_failed:
            for idx, result in enumerate(phase1_failed[:10], start=1):
                self.logger.warning(
                    "LigPrep phase 1 failure %d: %s",
                    idx,
                    self._short_error(result.error),
                )
            if len(phase1_failed) > 10:
                self.logger.warning(
                    "... and %d more LigPrep phase 1 failures",
                    len(phase1_failed) - 10,
                )

        phase1_success_ids = {r.task_id for r in phase1_results if r.success}
        min_tasks = [
            {
                "pose": task["pose"],
                "out_dir": task["out_dir"],
                "manifest": task["manifest"],
                "ctx": task["ctx"],
                "num_cores": self.num_cores,
                "num_gpus": self.num_gpus,
            }
            for task in tasks_acpype
            if task["pose"] in phase1_success_ids
        ]

        phase2_results: List[Any] = []
        if min_tasks:
            if self.gromacs_binary_type != "cuda" or self.num_gpus <= 0:
                min_workers = _cap_parallel_workers(
                    max(1, min(self.num_cores, len(min_tasks))),
                    self.max_jobs,
                )
                self.logger.info(
                    "LigPrep phase 2 (energy minimization): %d workers (CPU-only)",
                    min_workers,
                )
                phase2_config = ResourceConfig(
                    n_workers=min_workers,
                    n_gpus=0,
                    gpu_strategy="round_robin",
                )
            elif self.num_gpus == 1:
                self.logger.info(
                    "LigPrep phase 2 (energy minimization): 1 worker, 1 GPU (serial)."
                )
                phase2_config = ResourceConfig(
                    n_workers=1,
                    n_gpus=1,
                    gpu_strategy="serialize_gpu",
                )
            else:
                min_workers = _cap_parallel_workers(
                    max(1, min(self.num_cores, self.num_gpus, len(min_tasks))),
                    self.max_jobs,
                )
                self.logger.info(
                    "LigPrep phase 2 (energy minimization): %d workers, %d GPUs (round-robin).",
                    min_workers,
                    self.num_gpus,
                )
                phase2_config = ResourceConfig(
                    n_workers=min_workers,
                    n_gpus=self.num_gpus,
                    gpu_strategy="round_robin",
                )

            phase2_task_num_cores = _per_worker_num_cores(
                self.num_cores,
                phase2_config.n_workers,
            )
            for task in min_tasks:
                task["num_cores"] = phase2_task_num_cores

            phases = build_gpu_phases(
                min_tasks,
                n_gpus=phase2_config.n_gpus,
                n_workers=phase2_config.n_workers,
                hybrid_tail=False,
                default_gpu_strategy=phase2_config.gpu_strategy,
            )
            for phase in phases:
                if not phase.tasks:
                    continue
                self.logger.info(
                    "LigPrep phase 2 execution block '%s': %d pose(s).",
                    phase.name,
                    len(phase.tasks),
                )
                phase_executor = ParallelExecutor(phase.config)
                phase2_results.extend(
                    phase_executor.run(
                        run_min_only,
                        phase.tasks,
                        task_id_fn=lambda t: t["pose"],
                    )
                )
        else:
            self.logger.warning(
                "LigPrep phase 2 skipped: no poses passed phase 1 setup."
            )

        phase2_failed = [r for r in phase2_results if not r.success]
        if phase2_failed:
            for idx, result in enumerate(phase2_failed[:10], start=1):
                self.logger.warning(
                    "LigPrep phase 2 failure %d: %s",
                    idx,
                    self._short_error(result.error),
                )
            if len(phase2_failed) > 10:
                self.logger.warning(
                    "... and %d more LigPrep phase 2 failures",
                    len(phase2_failed) - 10,
                )

        summary_payload = self._write_lig_prepare_reports(
            selected_tasks=tasks_acpype,
            phase1_results=phase1_results,
            phase2_results=phase2_results,
        )
        prepared_pose_names = [
            task["pose"]
            for task in tasks_acpype
            if task["pose"] in {r.task_id for r in phase2_results if r.success}
        ]
        self.file_paths["prepared_poses"] = prepared_pose_names
        self.file_paths["prepared_pose_count"] = len(prepared_pose_names)
        self.file_paths["failed_pose_count"] = summary_payload["counts"]["overall"]["failed"]

        dropped_parent_ids = summary_payload.get("dropped_parent_ids", [])
        if dropped_parent_ids:
            self.logger.warning(
                "Parent ligands with zero prepared poses: %s",
                ", ".join(dropped_parent_ids),
            )

        if not prepared_pose_names:
            raise RuntimeError(
                "Ligand preparation failed: 0 poses prepared successfully. "
                f"See {self.file_paths.get('lig_prepare_summary_path')} and "
                f"{self.file_paths.get('lig_prepare_failures_path')} for details."
            )

        self.logger.info(
            "LigPrep completed with partial/final success: prepared=%d, failed=%d",
            len(prepared_pose_names),
            summary_payload["counts"]["overall"]["failed"],
        )
