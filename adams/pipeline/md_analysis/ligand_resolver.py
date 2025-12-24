"""
Ligand Structure Resolver - Common format handling for MD analysis

Handles common formats:
- SMILES strings (single or CSV)
- SDF/MOL2 files (single or multi-molecule)
- CSV files with SMILES column (flexible column names)

For unsupported formats, agent should use custom code.
"""

import os

import pandas as pd

from ...logger_utils import get_logger


class LigandResolver:
    """
    Resolves ligand molecular structures from common input formats.

    Uses standardize_ligand_data() from preprocessing module for actual work.
    """

    def __init__(self):
        self.logger = get_logger()

    def resolve_ligand_structures(
        self,
        ligand_input: str,
        docking_csv: str,
        output_dir: str,
    ) -> dict:
        """
        Resolve ligand structures from common formats.

        Args:
            ligand_input: Can be:
                - SMILES string: "CC(=O)O"
                - CSV file path: "/path/to/ligands.csv"
                - SDF/MOL2 file: "/path/to/ligands.sdf"
                - Directory: "/path/to/structures/"
            docking_csv: Path to docking results CSV
            output_dir: Directory for standardized output

        Returns:
            dict: {
                'smiles_csv_path': path_to_standardized_csv,
                'source': 'smiles_string' | 'csv' | 'sdf' | 'mol2' | 'directory'
            }

        Raises:
            ValueError: If format is not supported (agent should handle with custom code)
        """
        # Check if it's a SMILES string (simple heuristic)
        if self._is_smiles_string(ligand_input):
            return self._handle_smiles_string(ligand_input, docking_csv, output_dir)

        # Check if it's a file path
        if os.path.isfile(ligand_input):
            ext = os.path.splitext(ligand_input)[1].lower()
            if ext in [".csv", ".txt", ".tsv"]:
                return self._handle_csv(ligand_input, docking_csv, output_dir)
            elif ext in [".sdf", ".sd", ".mol2"]:
                return self._handle_structure_file(
                    ligand_input, docking_csv, output_dir
                )
            else:
                raise ValueError(
                    f"Unsupported file format: {ext}. "
                    "Supported formats: CSV, SDF, MOL2. "
                    "For other formats, use custom preprocessing code."
                )

        # Check if it's a directory
        if os.path.isdir(ligand_input):
            return self._handle_directory(ligand_input, docking_csv, output_dir)

        raise ValueError(
            f"Could not resolve ligand_input: {ligand_input}. "
            "Expected: SMILES string, CSV/SDF/MOL2 file, or directory. "
            "For other formats, use custom preprocessing code."
        )

    def _is_smiles_string(self, ligand_input: str) -> bool:
        """Heuristic to detect if input is a SMILES string."""
        # SMILES strings are typically short, contain alphanumeric and common symbols
        # and don't look like file paths
        if os.path.exists(ligand_input):
            return False

        # Check for common SMILES characters
        smiles_chars = set("()[]=+-#.@/\\")
        has_smiles_chars = any(c in ligand_input for c in smiles_chars)
        has_letters = any(c.isalpha() for c in ligand_input)

        # If it's short, has SMILES chars, and has letters, likely a SMILES string
        if len(ligand_input) < 500 and has_smiles_chars and has_letters:
            # Additional check: doesn't start with common path indicators
            if not ligand_input.startswith(("/", ".", "~", "C:", "D:")):
                return True

        return False

    def _handle_csv(self, csv_path: str, docking_csv: str, output_dir: str) -> dict:
        """Use standardize_ligand_data for CSV files."""
        from ..data_preprocessing.standardize_ligands import standardize_ligand_data

        self.logger.info(f"Standardizing CSV file: {csv_path}")
        standardized_csv = standardize_ligand_data(
            input_file=csv_path,
            output_dir=output_dir,
        )

        return {"smiles_csv_path": standardized_csv, "source": "csv"}

    def _handle_structure_file(
        self, struct_path: str, docking_csv: str, output_dir: str
    ) -> dict:
        """Use standardize_ligand_data for SDF/MOL2 files."""
        from ..data_preprocessing.standardize_ligands import standardize_ligand_data

        ext = os.path.splitext(struct_path)[1].lower()
        file_type = "sdf" if ext in [".sdf", ".sd"] else "mol2"

        self.logger.info(f"Standardizing {file_type.upper()} file: {struct_path}")
        standardized_csv = standardize_ligand_data(
            input_file=struct_path,
            output_dir=output_dir,
        )

        return {"smiles_csv_path": standardized_csv, "source": file_type}

    def _handle_smiles_string(
        self, smiles_str: str, docking_csv: str, output_dir: str
    ) -> dict:
        """Create temporary CSV from SMILES string."""
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        # Read docking CSV to get ligand IDs
        df_dock = pd.read_csv(docking_csv)

        if "ID" not in df_dock.columns:
            raise ValueError(
                f"Docking CSV missing 'ID' column. Available columns: {list(df_dock.columns)}"
            )

        unique_ids = df_dock["ID"].unique()

        if len(unique_ids) == 1:
            # Single ligand - use the SMILES string
            # Calculate molecular weight from SMILES string
            mol = Chem.MolFromSmiles(smiles_str)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smiles_str}")

            molwt = Descriptors.MolWt(mol)

            # Check if docking CSV has MolWt and compare
            if "MolWt" in df_dock.columns:
                docking_molwt = df_dock[df_dock["ID"] == unique_ids[0]]["MolWt"].iloc[0]
                if (
                    abs(molwt - docking_molwt) > 0.1
                ):  # Allow small floating point differences
                    self.logger.warning(
                        f"MolWt mismatch: SMILES string gives {molwt:.2f} Da, "
                        f"but docking CSV has {docking_molwt:.2f} Da. "
                        f"Using SMILES-derived MolWt ({molwt:.2f} Da)."
                    )

            df_smiles = pd.DataFrame(
                {"ID": [unique_ids[0]], "SMILES": [smiles_str], "MolWt": [molwt]}
            )
            self.logger.info(
                f"Using SMILES string for single ligand: {unique_ids[0]}, "
                f"calculated MolWt: {molwt:.2f} Da"
            )
        else:
            raise ValueError(
                f"SMILES string provided but docking CSV has {len(unique_ids)} unique ligands. "
                "For multiple ligands, provide a CSV file with SMILES column."
            )

        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        smiles_csv = os.path.join(output_dir, "ligand_smiles.csv")
        df_smiles.to_csv(smiles_csv, index=False)

        return {"smiles_csv_path": smiles_csv, "source": "smiles_string"}

    def _handle_directory(
        self, dir_path: str, docking_csv: str, output_dir: str
    ) -> dict:
        """Handle directory containing structure files."""
        # Look for SDF/MOL2 files in directory
        sdf_files = []
        mol2_files = []

        for file in os.listdir(dir_path):
            if file.endswith((".sdf", ".sd")):
                sdf_files.append(os.path.join(dir_path, file))
            elif file.endswith(".mol2"):
                mol2_files.append(os.path.join(dir_path, file))

        if sdf_files:
            # If multiple SDF files, combine them or use first one
            # For now, use standardize_ligand_data which handles multi-molecule files
            if len(sdf_files) == 1:
                return self._handle_structure_file(
                    sdf_files[0], docking_csv, output_dir
                )
            else:
                # Multiple files - would need to combine, but for now raise
                raise ValueError(
                    f"Directory contains {len(sdf_files)} SDF files. "
                    "Please provide a single SDF file or use custom preprocessing code."
                )
        elif mol2_files:
            if len(mol2_files) == 1:
                return self._handle_structure_file(
                    mol2_files[0], docking_csv, output_dir
                )
            else:
                raise ValueError(
                    f"Directory contains {len(mol2_files)} MOL2 files. "
                    "Please provide a single MOL2 file or use custom preprocessing code."
                )
        else:
            raise ValueError(
                f"No SDF or MOL2 files found in directory: {dir_path}. "
                "Supported files: .sdf, .sd, .mol2"
            )
