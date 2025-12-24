"""
data_preprocessing/ligand_preprocessing.py

Description:
    Process ligand CSV and generate samples for docking.
"""

import os
import re

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

from ...logger_utils import get_logger, log_step_execution
from ..file_organization import setup_preprocessing_dirs

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
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol is None:
            return False, "Invalid SMILES"

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

        return True, "Compatible"

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
        self.logger = get_logger()

        # Set up organized directory structure
        self.dir_structure = setup_preprocessing_dirs(outpath)

    def _check_input_data(self):
        """
        Check the input data.
        Returns:
            df_clean: pd.DataFrame: Cleaned input data
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
                    m = Chem.MolFromSmiles(s)
                    return rdMolDescriptors.CalcExactMolWt(m) if m is not None else None
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
        small_outfile = os.path.join(
            self.dir_structure["ligands"], f"{self.output_prefix}_smallMW.csv"
        )
        df_clean_small[["ID", "inSMILES", "inMolWt", "SMILES", "MolWt"]].to_csv(
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
            df_clean_large[["ID", "inSMILES", "inMolWt", "SMILES", "MolWt"]].to_csv(
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
            df_clean_too_small[["ID", "inSMILES", "inMolWt", "SMILES", "MolWt"]].to_csv(
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
            df_clean_small[["ID", "inSMILES", "inMolWt", "SMILES", "MolWt"]].to_csv(
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
            mol = Chem.MolFromSmiles(smiles)
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
        mol = Chem.MolFromSmiles(smiles)
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
            # Parse molecule without automatic sanitization
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return None

            # Split into fragments
            frags = Chem.GetMolFrags(mol, asMols=True)
            if not frags:
                return None

            # Pick the largest fragment (by atom count)
            largest = max(frags, key=lambda m: m.GetNumAtoms())

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
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return False

        if add_hs:
            mol = Chem.AddHs(mol)

        success = True
        if generate_3d:
            success = LigandPreprocessor._generate_conformer(mol, smiles, logger=logger)

        return success
