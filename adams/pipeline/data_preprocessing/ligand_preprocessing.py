"""
data_preprocessing/ligand_preprocessing.py

Description:
    Process ligand CSV and generate samples for docking.
"""

import os
import re
from typing import Dict, List, Optional

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from ...logger_utils import get_logger, log_step_execution
from ..file_organization import setup_preprocessing_dirs
from .microstate_enumeration import enumerate_ligand_microstates
from .smiles_qc import (
    is_kekulize_rescue_mode,
    parse_smiles_with_kekulize_rescue,
    quick_3d_forcefield_qc,
)

METALS = {
    "Li",
    "Na",
    "K",
    "Rb",
    "Cs",
    "Fr",
    "Be",
    "Mg",
    "Ca",
    "Sr",
    "Ba",
    "Ra",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Al",
    "Ga",
    "In",
    "Tl",
    "Pb",
    "Bi",
    "Sn",
    "Sb",
    "Se",
    "Gd",
}


def check_vina_compatibility(smiles):
    """
    Check if a molecule is likely to be compatible with AutoDock Vina.

    Vina doesn't support certain AutoDock4 atom types that Meeko may generate,
    including charged sulfur/nitrogen and unusual halogen types.

    Args:
        smiles: str: SMILES string

    Returns:
        tuple: (is_compatible, reason) where is_compatible is bool and reason is str
    """
    try:
        parsed = parse_smiles_with_kekulize_rescue(smiles, prefer_restandardize=True)
        mol = parsed.mol
        if mol is None:
            return False, f"Invalid SMILES: {parsed.reason}"

        # Check for formal charges that may cause issues
        for atom in mol.GetAtoms():
            charge = atom.GetFormalCharge()
            symbol = atom.GetSymbol()

            # Charged sulfur (sulfonium ions) → S_P atom type (unsupported)
            if symbol == "S" and charge > 0:
                return False, "Positively charged sulfur (sulfonium)"

            # Charged nitrogen (quaternary ammonium) → N_P atom type (may be unsupported)
            # Allow simple protonated amines but flag quaternary ammonium
            if symbol == "N" and charge > 0:
                # Check if it's a quaternary ammonium (4 heavy atom neighbors)
                if len([n for n in atom.GetNeighbors() if n.GetSymbol() != "H"]) == 4:
                    return False, "Quaternary ammonium (N+)"

            # Charged phosphorus → may cause issues
            if symbol == "P" and abs(charge) > 0:
                return False, "Charged phosphorus"

            # Unusual elements that Vina may not support well
            unsupported_elements = {"B", "Si", "Se", "Te", "As", "Sb"}
            if symbol in unsupported_elements:
                return False, f"Unsupported element: {symbol}"

        if parsed.parse_mode == "strict":
            return True, "Compatible"
        return True, f"Compatible ({parsed.parse_mode}: {parsed.reason})"

    except Exception as e:
        return False, f"Error checking compatibility: {str(e)}"


class LigandPreprocessor:
    def __init__(
        self,
        input_data: str,
        molwt_upper_bound: float = 700,
        molwt_lower_bound: float = 0,
        check_rdmol: bool = False,
        sampling: bool = False,
        binsize: int = 100,
        sampling_frac: float = 0.01,
        output_prefix: str = "cleaned_data",
        outpath: str = "output",
        quick_start: bool = False,
        enumerate_microstates: bool = True,
        enumerate_tautomers: bool = True,
        enumerate_protonation: bool = True,
        enumerate_stereoisomers: bool = True,
        pH_range: tuple = (6.4, 8.4),
        protonation_precision: float = 0.5,
        max_generated_tautomers: Optional[int] = 64,
        top_tautomers_per_protomer: int = 2,
        tautomer_energy_window_kcal: float = 3.0,
        max_protomers: int = 16,
        max_stereoisomers: int = 16,
        max_unassigned_stereocenters: int = 2,
        max_total_microstates: int = 64,
        enumerate_all_stereocenters: bool = False,
    ):
        """
        Initialize the LigandPreprocessor.
        Args:
            input_data: str: Input CSV file with compound data
            molwt_upper_bound: float: Molecular weight upper bound (default: 700)
            molwt_lower_bound: float: Molecular weight lower bound (default: 0)
            check_rdmol: bool: Check molecule with RDKit (default: False)
            sampling: bool: Sampling molecule with MW distribution (default: False)
            binsize: int: Bin size for MolWt sampling (default: 100)
            sampling_frac: float: Fraction of total data to sample (default: 0.01)
            output_prefix: str: Prefix for output files (default: "cleaned_data")
            outpath: str: Output directory (default: ./output)
            quick_start: bool: If True, use fast processing path (skips metal context analysis) (default: False)
            enumerate_microstates: bool: Enable microstate enumeration (default: True). When True, tautomers, protonation states, and stereoisomers are enumerated before MW filtering.
            enumerate_tautomers: bool: Enumerate tautomers (default: True; only applies when enumerate_microstates=True)
            enumerate_protonation: bool: Enumerate protonation states (default: True; only applies when enumerate_microstates=True)
            enumerate_stereoisomers: bool: Enumerate stereoisomers (default: True; only applies when enumerate_microstates=True)
            pH_range: tuple: pH range for protonation enumeration (default: (6.4, 8.4))
            protonation_precision: float: Dimorphite precision setting (default: 0.5)
            max_generated_tautomers: Optional[int]: Optional hard cap for generated tautomers per protomer (default: 64; set None to disable)
            top_tautomers_per_protomer: int: Keep this many top-ranked tautomers (default: 2)
            tautomer_energy_window_kcal: float: Keep tautomers within this energy window (default: 3.0)
            max_protomers: int: Maximum protomers per ligand (default: 16)
            max_stereoisomers: int: Maximum stereoisomers per ligand (default: 16)
            max_unassigned_stereocenters: int: Skip stereo expansion above this count (default: 2)
            max_total_microstates: int: Maximum number of microstates per parent ligand,
                including original (default: 64)
            enumerate_all_stereocenters: bool: If True, enumerate all stereocenters.
                If False (default), enumerate only unassigned stereocenters.
        """

        self.input_data = input_data
        self.molwt_upper_bound = molwt_upper_bound
        self.molwt_lower_bound = molwt_lower_bound
        self.check_rdmol = check_rdmol
        self.sampling = sampling
        self.binsize = binsize
        self.sampling_frac = sampling_frac
        self.output_prefix = output_prefix
        self.outpath = outpath
        self.quick_start = quick_start
        self.enumerate_microstates = enumerate_microstates
        self.enumerate_tautomers = enumerate_tautomers
        self.enumerate_protonation = enumerate_protonation
        self.enumerate_stereoisomers = enumerate_stereoisomers
        self.enumerate_all_stereocenters = enumerate_all_stereocenters
        self.pH_range = pH_range
        self.protonation_precision = protonation_precision
        self.max_generated_tautomers = max_generated_tautomers
        self.top_tautomers_per_protomer = top_tautomers_per_protomer
        self.tautomer_energy_window_kcal = tautomer_energy_window_kcal
        self.max_protomers = max_protomers
        self.max_stereoisomers = max_stereoisomers
        self.max_unassigned_stereocenters = max_unassigned_stereocenters
        self.max_total_microstates = max_total_microstates
        self.logger = get_logger()

        # Set up organized directory structure
        self.dir_structure = setup_preprocessing_dirs(outpath)

    def _check_input_data(self):
        """
        Check the input data and optionally enumerate microstates.
        Returns:
            df_clean: pd.DataFrame: Cleaned input data (with microstates if enabled)
        """
        if not os.path.isfile(self.input_data):
            raise FileNotFoundError(f"Input file not found: {self.input_data}")

        os.makedirs(self.dir_structure["ligands"], exist_ok=True)

        # --- Read input CSV ---
        df = pd.read_csv(self.input_data, sep=",")

        # --- Check required columns ---
        required_cols = ["ID", "SMILES", "MolWt"]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Input CSV must contain column: {col}")

        # Extract the required columns
        df_extracted = df[required_cols]
        df_extracted.columns = ["ID", "inSMILES", "inMolWt"]

        # --- Drop duplicates and NaNs ---
        df_clean = df_extracted.drop_duplicates().dropna()

        # --- Microstate enumeration (BEFORE MW filtering) ---
        if self.enumerate_microstates:
            self.logger.info(
                f"Enumerating microstates for {len(df_clean)} ligands "
                f"(tautomers={self.enumerate_tautomers}, "
                f"protomers={self.enumerate_protonation}, "
                f"stereoisomers={self.enumerate_stereoisomers})"
            )

            enumerated_dfs = []
            rescued_kekulize_records: List[Dict[str, str]] = []
            rejected_ligand_records: List[Dict[str, str]] = []
            for idx, row in df_clean.iterrows():
                ligand_id = str(row["ID"])
                input_smiles = str(row["inSMILES"])
                parsed = parse_smiles_with_kekulize_rescue(
                    input_smiles, prefer_restandardize=True
                )
                if parsed.mol is None:
                    self.logger.warning(
                        "Skipping ligand %s due to invalid SMILES after parse QC: %s "
                        "(sanitize_step=%s)",
                        ligand_id,
                        parsed.reason,
                        parsed.sanitize_failure_step,
                    )
                    rejected_ligand_records.append(
                        {
                            "ID": ligand_id,
                            "Input_SMILES": input_smiles,
                            "Reason": parsed.reason,
                            "Sanitize_Failure_Step": parsed.sanitize_failure_step or "",
                            "Stage": "parse",
                        }
                    )
                    continue

                smiles_for_enumeration = parsed.canonical_smiles or input_smiles
                if is_kekulize_rescue_mode(parsed.parse_mode):
                    qc_ok, qc_reason = quick_3d_forcefield_qc(parsed.mol)
                    rescued_kekulize_records.append(
                        {
                            "ID": ligand_id,
                            "Input_SMILES": input_smiles,
                            "Canonical_SMILES": smiles_for_enumeration,
                            "Parse_Mode": parsed.parse_mode,
                            "Reason": parsed.reason,
                            "Sanitize_Failure_Step": parsed.sanitize_failure_step or "",
                            "QC_Status": "passed" if qc_ok else "failed",
                            "QC_Reason": qc_reason,
                        }
                    )
                    if not qc_ok:
                        repaired = parse_smiles_with_kekulize_rescue(
                            input_smiles,
                            prefer_restandardize=True,
                            allow_openbabel_fallback=True,
                            force_openbabel_recanonicalization=True,
                        )
                        repaired_can = repaired.canonical_smiles or input_smiles
                        repaired_improves = (
                            repaired.mol is not None
                            and repaired_can != smiles_for_enumeration
                            and repaired.parse_mode.startswith("openbabel_recanonicalized")
                        )

                        if repaired_improves:
                            repaired_qc_ok, repaired_qc_reason = quick_3d_forcefield_qc(repaired.mol)
                            rescued_kekulize_records.append(
                                {
                                    "ID": ligand_id,
                                    "Input_SMILES": input_smiles,
                                    "Canonical_SMILES": repaired_can,
                                    "Parse_Mode": repaired.parse_mode,
                                    "Reason": repaired.reason,
                                    "Sanitize_Failure_Step": repaired.sanitize_failure_step or "",
                                    "QC_Status": "passed" if repaired_qc_ok else "failed",
                                    "QC_Reason": repaired_qc_reason,
                                }
                            )
                            if repaired_qc_ok:
                                self.logger.info(
                                    "Recovered ligand %s via OpenBabel recanonicalization after rescue QC failure.",
                                    ligand_id,
                                )
                                parsed = repaired
                                smiles_for_enumeration = repaired_can
                            else:
                                self.logger.warning(
                                    "Rejecting rescued ligand %s after OpenBabel retry; initial_qc=%s; obabel_qc=%s",
                                    ligand_id,
                                    qc_reason,
                                    repaired_qc_reason,
                                )
                                rejected_ligand_records.append(
                                    {
                                        "ID": ligand_id,
                                        "Input_SMILES": input_smiles,
                                        "Reason": (
                                            f"kekulize_rescue_qc_failed: {qc_reason}; "
                                            f"openbabel_retry_qc_failed: {repaired_qc_reason}"
                                        ),
                                        "Sanitize_Failure_Step": repaired.sanitize_failure_step or parsed.sanitize_failure_step or "",
                                        "Stage": "post_rescue_qc_openbabel_retry",
                                    }
                                )
                                continue
                        else:
                            self.logger.warning(
                                "Rejecting rescued ligand %s because post-rescue QC failed: %s",
                                ligand_id,
                                qc_reason,
                            )
                            rejected_ligand_records.append(
                                {
                                    "ID": ligand_id,
                                    "Input_SMILES": input_smiles,
                                    "Reason": f"kekulize_rescue_qc_failed: {qc_reason}",
                                    "Sanitize_Failure_Step": parsed.sanitize_failure_step or "",
                                    "Stage": "post_rescue_qc",
                                }
                            )
                            continue

                try:
                    variant_df = enumerate_ligand_microstates(
                        smiles=smiles_for_enumeration,
                        input_mol=parsed.mol,
                        original_id=ligand_id,
                        do_enumerate_tautomers=self.enumerate_tautomers,
                        do_enumerate_protonation=self.enumerate_protonation,
                        do_enumerate_stereoisomers=self.enumerate_stereoisomers,
                        enumerate_all_stereocenters=self.enumerate_all_stereocenters,
                        pH_range=self.pH_range,
                        protonation_precision=self.protonation_precision,
                        max_generated_tautomers=self.max_generated_tautomers,
                        top_tautomers_per_protomer=self.top_tautomers_per_protomer,
                        tautomer_energy_window_kcal=self.tautomer_energy_window_kcal,
                        max_protomers=self.max_protomers,
                        max_stereoisomers=self.max_stereoisomers,
                        max_unassigned_stereocenters=self.max_unassigned_stereocenters,
                        max_total_microstates=self.max_total_microstates,
                    )
                    if variant_df.empty:
                        rejected_ligand_records.append(
                            {
                                "ID": ligand_id,
                                "Input_SMILES": input_smiles,
                                "Reason": "microstate_enumeration_returned_empty",
                                "Sanitize_Failure_Step": "",
                                "Stage": "microstate_enumeration",
                            }
                        )
                        continue
                    # Map SMILES column to inSMILES and MolWt to inMolWt
                    variant_df = variant_df.rename(columns={"SMILES": "inSMILES", "MolWt": "inMolWt"})
                    enumerated_dfs.append(variant_df)

                except ImportError:
                    # Mandatory enumeration backend missing (e.g., dimorphite-dl).
                    # Fail fast rather than silently falling back.
                    raise
                except Exception as e:
                    self.logger.warning(
                        f"Error enumerating microstates for {ligand_id}: {e}"
                    )
                    # Fallback: keep original
                    fallback_variant_id = "original_0"
                    fallback_df = pd.DataFrame(
                        [
                            {
                                "ID": f"{ligand_id}__{fallback_variant_id}",
                                "inSMILES": smiles_for_enumeration,
                                "inMolWt": row["inMolWt"],
                                "Variant_Type": "original",
                                "Variant_ID": fallback_variant_id,
                                "Parent_ID": ligand_id,
                            }
                        ]
                    )
                    enumerated_dfs.append(fallback_df)

            rescued_path = os.path.join(
                self.dir_structure["ligands"], "rescued_kekulize_only.csv"
            )
            rejected_path = os.path.join(
                self.dir_structure["ligands"], "rejected_ligands.csv"
            )
            pd.DataFrame(
                rescued_kekulize_records,
                columns=[
                    "ID",
                    "Input_SMILES",
                    "Canonical_SMILES",
                    "Parse_Mode",
                    "Reason",
                    "Sanitize_Failure_Step",
                    "QC_Status",
                    "QC_Reason",
                ],
            ).to_csv(rescued_path, index=False)
            pd.DataFrame(
                rejected_ligand_records,
                columns=[
                    "ID",
                    "Input_SMILES",
                    "Reason",
                    "Sanitize_Failure_Step",
                    "Stage",
                ],
            ).to_csv(rejected_path, index=False)
            self.logger.info(
                "Kekulize-rescue audit: %d rescued entries written to %s",
                len(rescued_kekulize_records),
                rescued_path,
            )
            self.logger.info(
                "Ligand rejection audit: %d entries written to %s",
                len(rejected_ligand_records),
                rejected_path,
            )

            if enumerated_dfs:
                df_clean = pd.concat(enumerated_dfs, ignore_index=True)
            else:
                df_clean = pd.DataFrame(
                    columns=[
                        "ID",
                        "inSMILES",
                        "inMolWt",
                        "Variant_Type",
                        "Variant_ID",
                        "Parent_ID",
                    ]
                )
            self.logger.info(
                f"Microstate enumeration complete: {len(df_clean)} variants "
                f"from {len(df_clean['Parent_ID'].unique()) if 'Parent_ID' in df_clean.columns else 0} original ligands"
            )

        self.logger.info(f"Input data preview:\n{df_clean.head()}")
        return df_clean

    def run(self):
        r"""
        Execute the ligand preprocessing pipeline.

        Processing steps:
        1. Validate input data format and required columns
        2. Detect and classify metal-containing compounds
        3. Filter compounds by molecular weight bounds
        4. Optionally validate SMILES with RDKit
        5. Check Vina compatibility
        6. Optionally perform stratified sampling by molecular weight

        Returns:
            dict: Dictionary mapping output file types to file paths
        """
        step_logger = log_step_execution("Ligand Data Processing", self.logger)
        with step_logger:
            with step_logger.timing("data_validation"):
                df_clean = self._check_input_data()

            # Process based on mode
            if self.quick_start:
                df_working, output_files = self._run_quick_start(df_clean, step_logger)
            else:
                df_working, output_files = self._run_regular(df_clean, step_logger)

            # Common post-processing
            output_files = self._post_processing(df_working, output_files, step_logger)

            self.logger.info("Output files:")
            for key, path in output_files.items():
                self.logger.info(f"  {key}: {path}")

            return output_files

    def _run_regular(self, df_clean, step_logger):
        """
        Regular processing path with full metal context analysis.

        Args:
            df_clean: pd.DataFrame with input data
            step_logger: Step logger context manager

        Returns:
            tuple: (df_working, output_files) where df_working has SMILES and MolWt columns
        """
        output_files = {}

        # --- Metal detection ---
        with step_logger.timing("metal_detection"):
            df_clean["Contains_Metal"] = df_clean["inSMILES"].apply(
                LigandPreprocessor._metal_quick_scan
            )
            self.logger.info(
                f"Metal detection statistics:\n{df_clean['Contains_Metal'].describe()}"
            )

            # Save metal compounds
            df_metals = df_clean[df_clean["Contains_Metal"]].copy()
            df_metals.to_csv(
                os.path.join(self.dir_structure["ligands"], "metal_compounds.csv"),
                index=False,
            )
            self.logger.info(
                f"Saved {len(df_metals)} metal-containing SMILES to 'metal_compounds.csv'"
            )
            output_files["metal_compounds"] = os.path.join(
                self.dir_structure["ligands"], "metal_compounds.csv"
            )

        # --- Metal context analysis ---
        with step_logger.timing("metal_context_analysis"):
            df_clean["metal_context"] = df_clean["inSMILES"].apply(
                LigandPreprocessor._clssify_metal_context
            )
            df_metal_organic = df_clean[
                df_clean["metal_context"].apply(LigandPreprocessor._has_metal_organic)
            ].copy()
            df_metal_organic.to_csv(
                os.path.join(
                    self.dir_structure["ligands"], "metal_organic_compounds.csv"
                ),
                index=False,
            )
            self.logger.info(
                f"Saved {len(df_metal_organic)} metal-organic-containing SMILES to 'metal_organic_compounds.csv'"
            )
            output_files["metal_organic_compounds"] = os.path.join(
                self.dir_structure["ligands"], "metal_organic_compounds.csv"
            )

            # Filter to non-metal-organic compounds for further processing
            df_working = df_clean[
                df_clean["metal_context"].apply(LigandPreprocessor._not_metal_organic)
            ].copy()
            # Process SMILES: keep largest fragment only
            df_working["SMILES"] = df_working["inSMILES"].apply(
                lambda x: LigandPreprocessor._keep_largest_fragment(
                    x, logger=self.logger
                )
            )
            removed_count = len(df_working[df_working["SMILES"].isna()])
            if removed_count > 0:
                self.logger.warning(
                    f"{removed_count} compounds removed due to SMILES processing error..."
                )
            df_working = df_working.dropna()

        # --- Molecular weight calculation ---
        with step_logger.timing("molecular_weight_calculation"):
            df_working["MolWt"] = df_working["SMILES"].apply(
                LigandPreprocessor._calc_molwt
            )

        return df_working, output_files

    def _run_quick_start(self, df_clean, step_logger):
        """
        Fast processing path that skips metal context analysis.

        Args:
            df_clean: pd.DataFrame with input data
            step_logger: Step logger context manager

        Returns:
            tuple: (df_working, output_files) where df_working has SMILES and MolWt columns
        """
        output_files = {}

        # --- Metal detection ---
        with step_logger.timing("metal_detection"):
            df_clean["Contains_Metal"] = df_clean["inSMILES"].apply(
                LigandPreprocessor._metal_quick_scan
            )

        # --- Work with all non-metal compounds directly ---
        df_working = df_clean[~df_clean["Contains_Metal"]].copy()
        df_working["SMILES"] = df_working["inSMILES"]

        # --- Molecular weight calculation ---
        with step_logger.timing("molecular_weight_calculation"):

            def calc_molwt_safe(s):
                if s is None or pd.isna(s):
                    return None
                try:
                    parsed = parse_smiles_with_kekulize_rescue(
                        s, prefer_restandardize=True
                    )
                    return (
                        rdMolDescriptors.CalcExactMolWt(parsed.mol)
                        if parsed.mol is not None
                        else None
                    )
                except Exception:
                    return None

            df_working["MolWt"] = df_working["inSMILES"].apply(calc_molwt_safe)

        return df_working, output_files

    def _post_processing(self, df_working, output_files, step_logger):
        """
        Common post-processing steps after initial data processing.

        Args:
            df_working: pd.DataFrame with SMILES and MolWt columns
            output_files: dict of output file paths created so far
            step_logger: Step logger context manager

        Returns:
            dict: Updated output_files dictionary
        """
        # --- Split by molecular weight bounds (exclude NaN values) ---
        df_working_clean = df_working[df_working["MolWt"].notna()].copy()
        if len(df_working_clean) < len(df_working):
            nan_count = len(df_working) - len(df_working_clean)
            self.logger.warning(
                f"{nan_count} compounds removed due to invalid molecular weight calculation"
            )

        df_clean_small = df_working_clean[
            (df_working_clean["MolWt"] >= self.molwt_lower_bound)
            & (df_working_clean["MolWt"] <= self.molwt_upper_bound)
        ].copy()
        df_clean_large = df_working_clean[
            df_working_clean["MolWt"] > self.molwt_upper_bound
        ].copy()
        df_clean_too_small = df_working_clean[
            df_working_clean["MolWt"] < self.molwt_lower_bound
        ].copy()

        # --- Log statistics ---
        if len(df_clean_small) > 0:
            self.logger.info(
                f"\nMolWt summary statistics (within bounds {self.molwt_lower_bound}-{self.molwt_upper_bound}):\n{df_clean_small['MolWt'].describe()}"
            )

        # --- Save output files ---
        # Include variant columns if they exist
        base_cols = ["ID", "inSMILES", "inMolWt", "SMILES", "MolWt"]
        variant_cols = ["Variant_Type", "Variant_ID", "Parent_ID"]
        
        small_outfile = os.path.join(
            self.dir_structure["ligands"], f"{self.output_prefix}_smallMW.csv"
        )
        output_cols = base_cols + [col for col in variant_cols if col in df_clean_small.columns]
        df_clean_small[output_cols].to_csv(
            small_outfile, index=False
        )
        self.logger.info(
            f"Saved {len(df_clean_small)} compounds with MW {self.molwt_lower_bound}-{self.molwt_upper_bound} to {small_outfile}"
        )

        output_files["small_mw"] = small_outfile

        if len(df_clean_large) > 0:
            large_outfile = os.path.join(
                self.dir_structure["ligands"], f"{self.output_prefix}_largeMW.csv"
            )
            output_cols = base_cols + [col for col in variant_cols if col in df_clean_large.columns]
            df_clean_large[output_cols].to_csv(
                large_outfile, index=False
            )
            self.logger.info(
                f"Saved {len(df_clean_large)} large MW compounds (>{self.molwt_upper_bound}) to {large_outfile}"
            )
            output_files["large_mw"] = large_outfile

        if len(df_clean_too_small) > 0:
            too_small_outfile = os.path.join(
                self.dir_structure["ligands"], f"{self.output_prefix}_tooSmallMW.csv"
            )
            output_cols = base_cols + [col for col in variant_cols if col in df_clean_too_small.columns]
            df_clean_too_small[output_cols].to_csv(
                too_small_outfile, index=False
            )
            self.logger.info(
                f"Saved {len(df_clean_too_small)} too small MW compounds (<{self.molwt_lower_bound}) to {too_small_outfile}"
            )
            output_files["too_small_mw"] = too_small_outfile

        # --- RDKit validation (only in regular mode) ---
        if not self.quick_start and self.check_rdmol:
            with step_logger.timing("rdkit_validation"):
                df_clean_small["ConformerGenerated"] = df_clean_small["SMILES"].apply(
                    lambda s: LigandPreprocessor._smiles_to_mol(s, logger=self.logger)
                )
                df_clean_small_temp = df_clean_small[
                    df_clean_small["ConformerGenerated"]
                ].copy()
                temp_outfile = os.path.join(
                    self.dir_structure["ligands"],
                    f"temp_{self.output_prefix}_smallMW.csv",
                )
                df_clean_small_temp.to_csv(temp_outfile, index=False)
                self.logger.info(
                    f"Saved {len(df_clean_small_temp)} small MW compounds to {temp_outfile}"
                )
                output_files["temp_small_mw"] = temp_outfile

        # --- Vina Compatibility Check (before sampling) ---
        with step_logger.timing("vina_compatibility_check"):
            self.logger.info("Checking ligand compatibility with AutoDock Vina...")
            compatibility_results = df_clean_small["SMILES"].apply(
                check_vina_compatibility
            )
            df_clean_small["Vina_Compatible"] = compatibility_results.apply(
                lambda x: x[0]
            )
            df_clean_small["Incompatibility_Reason"] = compatibility_results.apply(
                lambda x: x[1]
            )

            # Save incompatible ligands for reference
            df_incompatible = df_clean_small[~df_clean_small["Vina_Compatible"]].copy()
            if len(df_incompatible) > 0:
                incompatible_file = os.path.join(
                    self.dir_structure["ligands"],
                    f"{self.output_prefix}_vina_incompatible.csv",
                )
                df_incompatible[
                    ["ID", "SMILES", "MolWt", "Incompatibility_Reason"]
                ].to_csv(incompatible_file, index=False)
                self.logger.warning(
                    f"Found {len(df_incompatible)} ligands incompatible with Vina:"
                )
                reason_counts = df_incompatible["Incompatibility_Reason"].value_counts()
                for reason, count in reason_counts.items():
                    self.logger.warning(f"  - {reason}: {count} ligands")
                self.logger.info(f"Saved incompatible ligands to {incompatible_file}")
                output_files["vina_incompatible"] = incompatible_file

            # Filter to only compatible ligands
            df_clean_small = df_clean_small[df_clean_small["Vina_Compatible"]].copy()
            self.logger.info(
                f"Retained {len(df_clean_small)} Vina-compatible ligands for docking"
            )

            # Update the small_mw file with only compatible ligands
            output_cols = base_cols + [col for col in variant_cols if col in df_clean_small.columns]
            df_clean_small[output_cols].to_csv(
                small_outfile, index=False
            )
            output_files["small_mw"] = small_outfile

        # --- Sampling ---
        if self.sampling:
            with step_logger.timing("data_sampling"):
                sampled_df, bin_count = LigandPreprocessor._sample_by_fraction(
                    df_clean_small, binsize=self.binsize, frac_total=self.sampling_frac
                )
                self.logger.info(f"\nSampling bin counts:\n{bin_count}")
                sample_outfile = os.path.join(
                    self.dir_structure["ligands"],
                    f"{self.output_prefix}_frac{self.sampling_frac}.csv",
                )
                sampled_df.to_csv(sample_outfile, index=False)
                self.logger.info(
                    f"Random sample per {self.binsize} MolWt bin saved to {sample_outfile}"
                )
                self.logger.info(f"Total {len(sampled_df)} were saved")
                output_files["sampled"] = sample_outfile

        return output_files

    @staticmethod
    def _calc_molwt(smiles):
        r"""
        Calculate the molecular weight of a SMILES string.
        Args:
            smiles: str: SMILES string
        Returns:
            float: Molecular weight
        """
        try:
            if smiles is None or pd.isna(smiles):
                return None
            parsed = parse_smiles_with_kekulize_rescue(smiles, prefer_restandardize=True)
            mol = parsed.mol
            if mol is None:
                return None
            return Descriptors.MolWt(mol)
        except Exception:
            return None

    @staticmethod
    def _metal_quick_scan(smiles: str) -> bool:
        """
        Check if a SMILES string contains a metal.
        Args:
            smiles: str: SMILES string
        Returns:
            bool: True if the SMILES string contains a metal, False otherwise
        """

        # --- Regex for common metals in SMILES ---
        METAL_REGEX = (
            r"\[(?:" + "|".join(sorted(METALS, key=len, reverse=True)) + r")[^]]*\]"
        )
        return bool(re.search(METAL_REGEX, smiles))

    @staticmethod
    def _clssify_metal_context(smiles: str):
        """
        Classify metal atoms in a SMILES string as free ion, cluster, or metal-organic.
        Args:
            smiles: str: SMILES string
        Returns:
            list: List of tuples, each containing a metal symbol and a context
        """
        parsed = parse_smiles_with_kekulize_rescue(smiles, prefer_restandardize=True)
        mol = parsed.mol
        if mol is None:
            return [("Invalid", "Invalid SMILES")]

        def _is_phosphate_like(oxygen_atom):
            # oxygen_atom: RDKit Atom object (symbol 'O')
            for nbr in oxygen_atom.GetNeighbors():
                if nbr.GetSymbol() == "P":
                    # P connected to multiple O's is typical phosphate
                    return True
            return False

        def _is_sulfate_like(oxygen_atom):
            for nbr in oxygen_atom.GetNeighbors():
                if nbr.GetSymbol() == "S" and nbr.GetDegree() >= 3:
                    return True
            return False

        def _is_carboxylate_like(oxygen_atom):
            # carboxylate: O connected to C that has another O neighbor (C(=O)O-)
            for nbr in oxygen_atom.GetNeighbors():
                if nbr.GetSymbol() == "C":
                    oxy_neighbors = sum(
                        1 for a in nbr.GetNeighbors() if a.GetSymbol() == "O"
                    )
                    # C with 2 oxygens is likely carboxylate/ester-like (carboxylate if one O is [O-])
                    if oxy_neighbors >= 2:
                        return True
                    # also check for formal negative oxygen (rarely explicit in RDKit without charges)
                    # or check for C having a double-bonded O neighbor
                    for a in nbr.GetNeighbors():
                        if a.GetSymbol() == "O":
                            # check double bond order if available
                            bond = nbr.GetBondBetweenAtoms(nbr.GetIdx(), a.GetIdx())
                            if bond is not None and bond.GetBondTypeAsDouble() == 2.0:
                                return True
            return False

        def _is_peroxide_like(oxygen_atom):
            for nbr in oxygen_atom.GetNeighbors():
                if nbr.GetSymbol() == "O":
                    return True
            return False

        def _is_anionic_oxygen(oxygen_atom):
            return (
                oxygen_atom.GetFormalCharge() < 0
                or _is_phosphate_like(oxygen_atom)
                or _is_sulfate_like(oxygen_atom)
                or _is_carboxylate_like(oxygen_atom)
                or _is_peroxide_like(oxygen_atom)
            )

        results = []
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            if sym not in METALS:
                continue

            nbrs = list(atom.GetNeighbors())
            if len(nbrs) == 0:
                results.append((sym, "free ion"))
                continue

            # If all neighbors are metals -> metal cluster
            if all(n.GetSymbol() in METALS for n in nbrs):
                results.append((sym, "metal cluster"))
                continue

            # All neighbors are oxygens -> check for anionic O
            if all(n.GetSymbol() == "O" for n in nbrs):
                if any(_is_anionic_oxygen(o) for o in nbrs):
                    results.append((sym, "counter-ion"))
                else:
                    results.append((sym, "metal-organic/coordination"))
                continue

            # Mixed neighbors (O + C/N/S) => coordination
            results.append((sym, "metal-organic/coordination"))

        return results

    @staticmethod
    def _has_metal_organic(contexts):
        """
        Return True if any of the metal atoms are classified as metal-organic/coordination
        Args:
            contexts: list: List of tuples, each containing a metal symbol and a context
        Returns:
            bool: True if any of the metal atoms are classified as metal-organic/coordination, False otherwise
        """
        return any(ctx[1] == "metal-organic/coordination" for ctx in contexts)

    @staticmethod
    def _not_metal_organic(contexts):
        """
        Return True if none of the metal atoms are classified as metal-organic/coordination
        Args:
            contexts: list: List of tuples, each containing a metal symbol and a context
        Returns:
            bool: True if none of the metal atoms are classified as metal-organic/coordination, False otherwise
        """
        return all(ctx[1] != "metal-organic/coordination" for ctx in contexts)

    @staticmethod
    def _sample_by_fraction(df, binsize=100, frac_total=0.01, random_state=42):
        """
        Sample ~1% of total dataset, proportionally across MolWt_bin groups.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with at least column "MolWt".
        binsize : int
            Size of molecular weight bin.
        frac_total : float
            Fraction of total dataset to sample (e.g., 0.01 = 1%).
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Sampled dataframe
        """
        # Create MolWt_bin
        df = df.copy()
        df["MolWt_bin"] = (df["MolWt"] // binsize) * binsize

        total_samples = int(len(df) * frac_total)
        if total_samples < 1:
            raise ValueError("Dataset too small for requested fraction")

        # Group counts
        group_counts = df["MolWt_bin"].value_counts().sort_index()

        # Compute proportional allocation
        proportions = group_counts / group_counts.sum()
        group_samples = (proportions * total_samples).round().astype(int)

        # Ensure at least 1 sample per bin
        group_samples[group_samples == 0] = 1

        sampled = []
        for bin_value, n_samples in group_samples.items():
            if n_samples > 0:
                group_df = df[df["MolWt_bin"] == bin_value]
                n_samples = min(len(group_df), n_samples)  # safety check
                sampled.append(group_df.sample(n=n_samples, random_state=random_state))

        sampled_df = pd.concat(sampled).reset_index(drop=True)
        return sampled_df, group_counts

    @staticmethod
    def _keep_largest_fragment(smiles, remove_final_ions=True, logger=None):
        """
        Clean metals, keep the largest fragment of a SMILES string,
        avoid kekulization errors, return canonical SMILES.

        Args:
            smiles: SMILES string to process
            remove_final_ions: Whether to remove final ions
            logger: Optional logger instance. If None, uses get_logger()
        """
        if logger is None:
            logger = get_logger()
        try:
            parsed = parse_smiles_with_kekulize_rescue(
                smiles, prefer_restandardize=True
            )
            mol = parsed.mol
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}; reason={parsed.reason}")
                return None

            # Split into fragments
            frags = Chem.GetMolFrags(mol, asMols=True)
            if not frags:
                return None

            # Pick the largest fragment (by atom count)
            largest = max(frags, key=lambda m: m.GetNumAtoms())
            try:
                sanitize_ops = (
                    Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                )
                Chem.SanitizeMol(largest, sanitizeOps=sanitize_ops)
            except Exception:
                # Keep best-effort behavior for edge-case aromatic systems.
                pass

            # Return canonical SMILES
            canonical = Chem.MolToSmiles(largest, canonical=True)

            if remove_final_ions:
                # --- Remove all free metal ions at the SMILES level ---
                # Match metal in square brackets with optional charge
                metal_pattern = r"\[(" + "|".join(METALS) + r")(?:[+-]\d*)?\]"
                canonical = re.sub(metal_pattern, "", canonical)

            return canonical

        except Exception as e:
            logger.error(f"Error processing SMILES: {smiles}, Error: {e}")
            return None

    @staticmethod
    def _generate_conformer(mol, smiles, max_attempts=50, logger=None):
        """
        Generate a 3D conformer for a molecule using ETKDGv2 and optimize geometry with MMFF94s.

        Args:
            mol: RDKit molecule object
            smiles: SMILES string
            max_attempts: Maximum attempts for embedding
            logger: Optional logger instance. If None, uses get_logger()
        """
        if logger is None:
            logger = get_logger()
        ps = AllChem.ETKDGv2()
        rid = -1

        # Try to embed molecule up to max_attempts
        for _ in range(max_attempts):
            rid = AllChem.EmbedMolecule(mol, ps)
            if rid == 0:
                break

        # If embedding failed, try using random coordinates
        if rid != 0:
            logger.warning(
                "RDKit could not generate coordinates without random coords, using random coords now."
            )
            ps.useRandomCoords = True
            rid = AllChem.EmbedMolecule(mol, ps)
            if rid != 0:
                logger.warning(
                    f"Failed to generate a 3D conformer even with random coords {smiles}."
                )
                return False  # Could not generate conformer

        # Only optimize if at least one conformer exists
        if mol.GetNumConformers() > 0:
            AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=500)
            return True
        else:
            logger.warning(f"No conformer to optimize {smiles}.")
            return False

    @staticmethod
    def _smiles_to_mol(smiles, add_hs=True, generate_3d=True, logger=None):
        """
        Convert a SMILES string to an RDKit Mol object and optionally generate 3D conformer.
        Returns a boolean indicating if a 3D conformer was successfully generated.

        Args:
            smiles: SMILES string
            add_hs: Whether to add hydrogens
            generate_3d: Whether to generate 3D conformer
            logger: Optional logger instance. If None, uses get_logger()

        Returns:
            bool: True if conformer was successfully generated, False otherwise
        """
        if logger is None:
            logger = get_logger()
        parsed = parse_smiles_with_kekulize_rescue(smiles, prefer_restandardize=True)
        mol = parsed.mol
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}; reason={parsed.reason}")
            return False
        if is_kekulize_rescue_mode(parsed.parse_mode):
            logger.info(
                "Using kekulize-only rescued parse for conformer validation "
                f"(smiles={smiles}, mode={parsed.parse_mode})."
            )

        if add_hs:
            mol = Chem.AddHs(mol)

        success = True
        if generate_3d:
            success = LigandPreprocessor._generate_conformer(mol, smiles, logger=logger)

        return success
