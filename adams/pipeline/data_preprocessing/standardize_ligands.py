"""
Standardize Ligand Data

Description:
    Converts various ligand input formats (SDF, MOL2, PDB, PDBQT, SMILES list)
    into a standardized CSV format (ID, SMILES, MolWt) required by the pipeline.

    Crucially:
    - If inputs have 3D coordinates, it splits them into individual files and
      uses their absolute paths in the 'SMILES' column.
    - If inputs are 2D (SMILES, 2D SDF), it extracts the SMILES string.
"""

import os
import shutil

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from ...logger_utils import get_logger


class UnsupportedFormatError(Exception):
    """Raised when standardizer cannot handle the input format."""

    pass


class Invalid3DStructureError(Exception):
    """Raised when 3D coordinates are invalid or missing."""

    pass


def _is_3d(mol):
    """Check if RDKit molecule has 3D coordinates."""
    try:
        if mol.GetNumConformers() == 0:
            return False
        conf = mol.GetConformer()
        if not conf.Is3D():
            return False
        # Double check z-coords are not all zero (common in 2D SDFs)
        pos = conf.GetPositions()
        if np.all(np.abs(pos[:, 2]) < 1e-3):
            return False
        return True
    except:
        return False


def detect_ligand_format(input_file: str) -> dict:
    """
    Analyzes input file and returns format metadata.

    Args:
        input_file: Path to ligand input file (any format)

    Returns:
        dict: {
            'has_3d': bool,           # True if valid 3D coordinates detected
            'format': str,             # 'sdf', 'mol2', 'pdb', 'pdbqt', 'csv', 'smi', etc.
            'file_paths': list,        # Paths to structure files (for 3D inputs)
            'num_molecules': int       # Number of molecules detected
        }

    Raises:
        UnsupportedFormatError: If format cannot be detected
        FileNotFoundError: If input file doesn't exist
    """
    logger = get_logger()

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    ext = os.path.splitext(input_file)[1].lower()

    # Detect format from extension
    format_map = {
        ".sdf": "sdf",
        ".sd": "sdf",
        ".mol2": "mol2",
        ".pdb": "pdb",
        ".pdbqt": "pdbqt",
        ".csv": "csv",
        ".txt": "txt",
        ".tsv": "tsv",
        ".smi": "smi",
        ".smiles": "smi",
    }

    if ext not in format_map:
        raise UnsupportedFormatError(f"Unsupported file extension: {ext}")

    file_format = format_map[ext]
    has_3d = False
    num_molecules = 0
    file_paths = []

    # Check for 3D structures
    if ext in [".sdf", ".sd"]:
        suppl = Chem.SDMolSupplier(input_file, removeHs=False)
        for mol in suppl:
            if mol:
                num_molecules += 1
                if _is_3d(mol):
                    has_3d = True

    elif ext == ".mol2":
        # Mol2 files are typically 3D
        try:
            with open(input_file, "r") as f:
                content = f.read()
            blocks = content.split("@<TRIPOS>MOLECULE")
            if not blocks[0].strip():
                blocks = blocks[1:]
            num_molecules = len(blocks)
            has_3d = True  # Assume Mol2 files are 3D
        except Exception as e:
            raise UnsupportedFormatError(f"Failed to parse Mol2 file: {e}")

    elif ext in [".pdb", ".pdbqt"]:
        # PDB/PDBQT are always 3D
        num_molecules = 1
        has_3d = True
        file_paths = [os.path.abspath(input_file)]

    elif ext in [".csv", ".txt", ".tsv"]:
        # CSV files contain 2D SMILES
        try:
            df = pd.read_csv(input_file, sep=None, engine="python")
            num_molecules = len(df)
            has_3d = False
        except Exception as e:
            raise UnsupportedFormatError(f"Failed to parse CSV file: {e}")

    elif ext in [".smi", ".smiles"]:
        # SMILES files are 2D
        try:
            with open(input_file, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            num_molecules = len(lines)
            has_3d = False
        except Exception as e:
            raise UnsupportedFormatError(f"Failed to parse SMILES file: {e}")

    logger.info(
        f"Detected format: {file_format}, has_3d: {has_3d}, num_molecules: {num_molecules}"
    )

    return {
        "has_3d": has_3d,
        "format": file_format,
        "file_paths": file_paths,
        "num_molecules": num_molecules,
    }


def convert_3d_to_pdbqt(input_file: str, output_dir: str) -> list:
    """
    Converts 3D structure files directly to PDBQT format.

    Args:
        input_file: Path to 3D structure file (SDF, MOL2, PDB, PDBQT)
        output_dir: Directory to save PDBQT files

    Returns:
        list: Paths to generated PDBQT files

    Raises:
        Invalid3DStructureError: If coordinates are not valid 3D
        UnsupportedFormatError: If conversion fails
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit.Chem import AddHs

    logger = get_logger()
    logger.info(f"Converting 3D structures to PDBQT from: {input_file}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directory
    pdbqt_dir = os.path.join(output_dir, "pdbqt_files")
    os.makedirs(pdbqt_dir, exist_ok=True)

    ext = os.path.splitext(input_file)[1].lower()
    pdbqt_paths = []
    ligprep = MoleculePreparation()

    # Handle different 3D file formats
    if ext in [".sdf", ".sd"]:
        suppl = Chem.SDMolSupplier(input_file, removeHs=False)
        for i, mol in enumerate(suppl):
            if not mol:
                logger.warning(f"Failed to read molecule {i} from SDF")
                continue

            # Validate 3D coordinates
            if not _is_3d(mol):
                raise Invalid3DStructureError(
                    f"Molecule {i} does not have valid 3D coordinates. "
                    "Use standardize_2d_to_csv() for 2D structures."
                )

            # Get molecule ID
            if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
                mol_id = mol.GetProp("_Name")
            elif mol.HasProp("ID"):
                mol_id = mol.GetProp("ID")
            else:
                mol_id = f"lig_{i+1}"

            clean_id = "".join([c if c.isalnum() else "_" for c in mol_id])

            # Add hydrogens if not present
            mol = AddHs(mol, addCoords=True)

            # Convert to PDBQT using Meeko
            try:
                mol_setups = ligprep.prepare(mol)
                pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(
                    mol_setups[0]
                )

                if is_ok:
                    pdbqt_path = os.path.join(pdbqt_dir, f"{clean_id}.pdbqt")
                    with open(pdbqt_path, "w") as f:
                        f.write(pdbqt_string)
                    pdbqt_paths.append(os.path.abspath(pdbqt_path))
                    logger.info(f"Converted {mol_id} to PDBQT")
                else:
                    logger.warning(f"Failed to convert {mol_id} to PDBQT: {error_msg}")
            except Exception as e:
                logger.warning(f"Error converting {mol_id} to PDBQT: {e}")

    elif ext == ".mol2":
        # Split multi-mol2 files
        try:
            with open(input_file, "r") as f:
                content = f.read()
            blocks = content.split("@<TRIPOS>MOLECULE")
            if not blocks[0].strip():
                blocks = blocks[1:]

            for i, block in enumerate(blocks):
                block = "@<TRIPOS>MOLECULE" + block
                mol = Chem.MolFromMol2Block(block, sanitize=False, removeHs=False)

                if not mol:
                    logger.warning(f"Failed to parse MOL2 block {i}")
                    continue

                # Get molecule ID from MOL2 name line
                lines = block.splitlines()
                if len(lines) > 1 and lines[1].strip():
                    mol_id = lines[1].strip()
                else:
                    mol_id = f"lig_{i+1}"

                clean_id = "".join([c if c.isalnum() else "_" for c in mol_id])

                # Add hydrogens
                mol = AddHs(mol, addCoords=True)

                # Convert to PDBQT
                try:
                    mol_setups = ligprep.prepare(mol)
                    pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(
                        mol_setups[0]
                    )

                    if is_ok:
                        pdbqt_path = os.path.join(pdbqt_dir, f"{clean_id}.pdbqt")
                        with open(pdbqt_path, "w") as f:
                            f.write(pdbqt_string)
                        pdbqt_paths.append(os.path.abspath(pdbqt_path))
                        logger.info(f"Converted {mol_id} to PDBQT")
                    else:
                        logger.warning(
                            f"Failed to convert {mol_id} to PDBQT: {error_msg}"
                        )
                except Exception as e:
                    logger.warning(f"Error converting {mol_id} to PDBQT: {e}")

        except Exception as e:
            raise UnsupportedFormatError(f"Failed to process MOL2 file: {e}")

    elif ext == ".pdb":
        # Single PDB file
        mol = Chem.MolFromPDBFile(input_file, removeHs=False)
        if not mol:
            raise UnsupportedFormatError(f"Failed to read PDB file: {input_file}")

        mol_id = os.path.basename(input_file).split(".")[0]
        clean_id = "".join([c if c.isalnum() else "_" for c in mol_id])

        # Add hydrogens
        mol = AddHs(mol, addCoords=True)

        # Convert to PDBQT
        try:
            mol_setups = ligprep.prepare(mol)
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(
                mol_setups[0]
            )

            if is_ok:
                pdbqt_path = os.path.join(pdbqt_dir, f"{clean_id}.pdbqt")
                with open(pdbqt_path, "w") as f:
                    f.write(pdbqt_string)
                pdbqt_paths.append(os.path.abspath(pdbqt_path))
                logger.info(f"Converted {mol_id} to PDBQT")
            else:
                raise UnsupportedFormatError(f"Failed to convert to PDBQT: {error_msg}")
        except Exception as e:
            raise UnsupportedFormatError(f"Error converting to PDBQT: {e}")

    elif ext == ".pdbqt":
        # Already PDBQT, just copy it
        mol_id = os.path.basename(input_file).split(".")[0]
        clean_id = "".join([c if c.isalnum() else "_" for c in mol_id])
        pdbqt_path = os.path.join(pdbqt_dir, f"{clean_id}.pdbqt")
        shutil.copy(input_file, pdbqt_path)
        pdbqt_paths.append(os.path.abspath(pdbqt_path))
        logger.info(f"Copied PDBQT file: {mol_id}")

    else:
        raise UnsupportedFormatError(f"Cannot convert {ext} to PDBQT as 3D structure")

    if not pdbqt_paths:
        raise Invalid3DStructureError(
            "No valid 3D structures could be converted to PDBQT"
        )

    logger.info(f"Successfully converted {len(pdbqt_paths)} structures to PDBQT")
    return pdbqt_paths


def standardize_2d_to_csv(
    input_file: str,
    output_dir: str,
    id_col: str = "ID",
    smiles_col: str = "SMILES",
    molwt_col: str = "MolWt",
) -> str:
    """
    Extracts SMILES from 2D sources into standardized CSV format.

    Args:
        input_file: Path to 2D input (CSV, SMILES file, 2D SDF)
        output_dir: Directory to save output CSV
        id_col: Name of ID column (for CSV inputs)
        smiles_col: Name of SMILES column (for CSV inputs)
        molwt_col: Name of MolWt column (for CSV inputs)

    Returns:
        str: Path to standardized CSV file (columns: ID, SMILES, MolWt)

    Raises:
        UnsupportedFormatError: If format cannot be processed
    """
    logger = get_logger()
    logger.info(f"Standardizing 2D ligands to CSV from: {input_file}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "standardized_ligands.csv")

    ext = os.path.splitext(input_file)[1].lower()
    records = []

    # Handle CSV/TSV/TXT files
    if ext in [".csv", ".txt", ".tsv"]:
        try:
            df = pd.read_csv(input_file, sep=None, engine="python")
        except:
            df = pd.read_csv(input_file)

        # Column mapping (case-insensitive)
        cols = {c.lower(): c for c in df.columns}

        # Find ID column
        tgt_id = None
        for k in cols:
            if k in ["id", "name", "identifier", id_col.lower()]:
                tgt_id = cols[k]
                break

        # Find SMILES column
        tgt_smi = None
        for k in cols:
            if k in ["smiles", "smi", "canonical_smiles", smiles_col.lower()]:
                tgt_smi = cols[k]
                break

        # Find MolWt column
        tgt_mw = None
        for k in cols:
            if k in ["molwt", "mol_wt", "molecular_weight", molwt_col.lower()]:
                tgt_mw = cols[k]
                break

        if tgt_smi:
            for i, row in df.iterrows():
                smi = str(row[tgt_smi]).strip()
                lid = str(row[tgt_id]) if tgt_id else f"lig_{i+1}"

                # Calculate molecular weight if not present
                if tgt_mw and pd.notna(row[tgt_mw]):
                    mw = float(row[tgt_mw])
                else:
                    mw = 0
                    try:
                        mol = Chem.MolFromSmiles(smi)
                        if mol:
                            mw = Descriptors.MolWt(mol)
                    except:
                        pass

                records.append({"ID": lid, "SMILES": smi, "MolWt": mw})
        else:
            # Try headerless format (first column is SMILES)
            first = str(df.iloc[0, 0]).strip()
            if Chem.MolFromSmiles(first):
                for i, row in df.iterrows():
                    smi = str(row.iloc[0]).strip()
                    lid = str(row.iloc[1]) if len(row) > 1 else f"lig_{i+1}"

                    mw = 0
                    try:
                        mol = Chem.MolFromSmiles(smi)
                        if mol:
                            mw = Descriptors.MolWt(mol)
                    except:
                        pass

                    records.append({"ID": lid, "SMILES": smi, "MolWt": mw})
            else:
                raise UnsupportedFormatError(
                    f"Could not find SMILES column in CSV. "
                    f"Available columns: {list(df.columns)}"
                )

    # Handle SMILES files (.smi, .smiles)
    elif ext in [".smi", ".smiles"]:
        with open(input_file, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) >= 2:
                    smi = parts[0]
                    lid = parts[1]
                elif len(parts) == 1:
                    smi = parts[0]
                    lid = f"lig_{i+1}"
                else:
                    continue

                mw = 0
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol:
                        mw = Descriptors.MolWt(mol)
                except:
                    pass

                records.append({"ID": lid, "SMILES": smi, "MolWt": mw})

    # Handle 2D SDF files
    elif ext in [".sdf", ".sd"]:
        suppl = Chem.SDMolSupplier(input_file, removeHs=False)
        for i, mol in enumerate(suppl):
            if not mol:
                logger.warning(f"Failed to read molecule {i} from SDF")
                continue

            # Check if it's actually 2D
            if _is_3d(mol):
                raise Invalid3DStructureError(
                    f"SDF file contains 3D structures. Use convert_3d_to_pdbqt() instead."
                )

            # Get molecule ID
            if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
                lid = mol.GetProp("_Name")
            elif mol.HasProp("ID"):
                lid = mol.GetProp("ID")
            else:
                lid = f"lig_{i+1}"

            # Convert to SMILES
            smi = Chem.MolToSmiles(mol)
            mw = Descriptors.MolWt(mol)

            records.append({"ID": lid, "SMILES": smi, "MolWt": mw})

    else:
        raise UnsupportedFormatError(f"Cannot process {ext} as 2D structure file")

    if not records:
        raise UnsupportedFormatError("No valid SMILES found in input file")

    # Create DataFrame and save
    df_out = pd.DataFrame(records)
    df_out["MolWt"] = df_out["MolWt"].fillna(0.0)
    df_out.to_csv(output_csv, index=False)

    logger.info(f"Standardized {len(records)} ligands to CSV: {output_csv}")
    return output_csv


def generate_conformers_to_pdbqt(input_csv: str, output_dir: str) -> list:
    """
    Generates 3D conformers from SMILES CSV and converts to PDBQT.

    Args:
        input_csv: Path to CSV with SMILES column (from standardize_2d_to_csv)
        output_dir: Directory to save PDBQT files

    Returns:
        list: Paths to generated PDBQT files

    Raises:
        FileNotFoundError: If input CSV doesn't exist
        ValueError: If CSV doesn't have SMILES column
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit.Chem import AddHs, MolFromSmiles
    from rdkit.Chem.AllChem import EmbedMolecule, UFFOptimizeMolecule

    logger = get_logger()
    logger.info(f"Generating 3D conformers from CSV: {input_csv}")

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Read CSV
    df = pd.read_csv(input_csv)

    if "SMILES" not in df.columns:
        raise ValueError(
            f"Input CSV must have 'SMILES' column. Found columns: {list(df.columns)}"
        )

    if "ID" not in df.columns:
        raise ValueError(
            f"Input CSV must have 'ID' column. Found columns: {list(df.columns)}"
        )

    # Create output directory
    pdbqt_dir = os.path.join(output_dir, "pdbqt_files")
    os.makedirs(pdbqt_dir, exist_ok=True)

    pdbqt_paths = []
    ligprep = MoleculePreparation()

    logger.info(f"Processing {len(df)} ligands...")

    for idx, row in df.iterrows():
        smiles = row["SMILES"]
        lig_id = str(row["ID"])

        # Generate 3D conformer
        try:
            mol = MolFromSmiles(smiles)
            if not mol:
                logger.warning(f"Failed to parse SMILES for {lig_id}: {smiles}")
                continue

            # Add hydrogens
            mol = AddHs(mol)

            # Generate 3D coordinates
            result = EmbedMolecule(mol, randomSeed=42)
            if result == -1:
                logger.warning(f"Failed to embed molecule {lig_id}")
                continue

            # Optimize geometry
            try:
                UFFOptimizeMolecule(mol)
            except:
                # UFF optimization can fail, but we can still proceed
                logger.warning(
                    f"UFF optimization failed for {lig_id}, using unoptimized conformer"
                )

            # Convert to PDBQT using Meeko
            mol_setups = ligprep.prepare(mol)
            pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(
                mol_setups[0]
            )

            if is_ok:
                clean_id = "".join([c if c.isalnum() else "_" for c in lig_id])
                pdbqt_path = os.path.join(pdbqt_dir, f"{clean_id}.pdbqt")

                with open(pdbqt_path, "w") as f:
                    f.write(pdbqt_string)

                pdbqt_paths.append(os.path.abspath(pdbqt_path))
            else:
                logger.warning(f"PDBQT conversion failed for {lig_id}: {error_msg}")

        except Exception as e:
            logger.warning(f"Error processing {lig_id}: {e}")
            continue

    logger.info(f"Generated {len(pdbqt_paths)} PDBQT files from {len(df)} SMILES")
    return pdbqt_paths


def standardize_ligand_data(
    input_file: str,
    output_dir: str = "output",
    id_col: str = "ID",
    smiles_col: str = "SMILES",
) -> str:
    """
    Reads ligand data from various file formats and saves it as a standardized CSV.

    If input contains 3D structures, they are split into individual files in
    '{output_dir}/standardized_structures/' and the CSV references these paths.

    Args:
        input_file: Path to input file (.sdf, .mol2, .pdb, .pdbqt, .csv, .smi, .txt)
        output_dir: Directory to save output CSV and split structures.
        id_col: Name of ID column (for CSV inputs).
        smiles_col: Name of SMILES column (for CSV inputs).
        molwt_col: Name of MolWt column (for CSV inputs).

    Returns:
        str: Path to the generated CSV file.
    """
    logger = get_logger()
    logger.info(f"Standardizing ligand data from: {input_file}")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Create output directories
    structures_dir = os.path.join(output_dir, "standardized_structures")
    os.makedirs(structures_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, "standardized_ligands.csv")

    records = []
    ext = os.path.splitext(input_file)[1].lower()

    # Track if we found 3D structures
    found_3d = False

    # --- SDF / MOL2 (Potential Multi-molecule 3D) ---
    if ext in [".sdf", ".sd"]:
        suppl = Chem.SDMolSupplier(input_file, removeHs=False)
        for i, mol in enumerate(suppl):
            if not mol:
                continue

            # ID extraction
            if mol.HasProp("_Name") and mol.GetProp("_Name").strip():
                lid = mol.GetProp("_Name")
            elif mol.HasProp("ID"):
                lid = mol.GetProp("ID")
            else:
                lid = f"lig_{i+1}"

            # Clean ID for filename
            clean_id = "".join([c if c.isalnum() else "_" for c in lid])
            mw = Descriptors.MolWt(mol)

            if _is_3d(mol):
                found_3d = True
                # Save individual 3D file
                out_path = os.path.join(structures_dir, f"{clean_id}.sdf")
                w = Chem.SDWriter(out_path)
                w.write(mol)
                w.close()
                # Use PATH as "SMILES"
                records.append(
                    {"ID": lid, "SMILES": os.path.abspath(out_path), "MolWt": mw}
                )
            else:
                # 2D -> Use SMILES
                smi = Chem.MolToSmiles(mol)
                records.append({"ID": lid, "SMILES": smi, "MolWt": mw})

    elif ext == ".mol2":
        # Simple splitting for mol2 to handle multi-mol files manually
        # RDKit support for Mol2 is limited, so we treat it carefully
        try:
            with open(input_file, "r") as f:
                content = f.read()
            blocks = content.split("@<TRIPOS>MOLECULE")
            if not blocks[0].strip():
                blocks = blocks[1:]

            for i, block in enumerate(blocks):
                block = "@<TRIPOS>MOLECULE" + block
                mol = Chem.MolFromMol2Block(block, sanitize=False, removeHs=False)

                # Try to get ID
                lines = block.splitlines()
                if len(lines) > 1 and lines[1].strip():
                    lid = lines[1].strip()
                else:
                    lid = f"lig_{i+1}"
                clean_id = "".join([c if c.isalnum() else "_" for c in lid])

                if mol:
                    mw = Descriptors.MolWt(mol)
                    # Assume Mol2 is 3D (it usually is)
                    found_3d = True
                    out_path = os.path.join(structures_dir, f"{clean_id}.mol2")
                    with open(out_path, "w") as out_f:
                        out_f.write(block)

                else:
                    logger.warning(f"Failed to parse Mol2 block {i}")

        except Exception as e:
            logger.warning(f"Error processing Mol2: {e}")

    # --- PDB / PDBQT (Usually Single Structure, assume 3D) ---
    elif ext in [".pdb", ".pdbqt"]:
        # If it's a single file, just copy it to standardized location or refer to it?
        # Better to copy/standardize name to ensure consistency
        lid = os.path.basename(input_file).split(".")[0]
        clean_id = "".join([c if c.isalnum() else "_" for c in lid])

        # We assume 3D for PDB/PDBQT
        found_3d = True
        out_path = os.path.join(structures_dir, f"{clean_id}{ext}")
        shutil.copy(input_file, out_path)

        # Calculate MW using RDKit if possible, else 0 (Vina will recalc or use default)
        mw = 0
        try:
            if ext == ".pdb":
                mol = Chem.MolFromPDBFile(input_file, removeHs=False)
            else:
                mol = None  # RDKit doesn't read PDBQT well

            if mol:
                mw = Descriptors.MolWt(mol)
        except:
            pass

        records.append({"ID": lid, "SMILES": os.path.abspath(out_path), "MolWt": mw})

    # --- CSV / SMILES (2D) ---
    elif ext in [".csv", ".txt", ".tsv", ".smi", ".smiles"]:
        # ... (Reuse CSV/SMILES logic from previous thought) ...
        # These are strictly 2D sources
        pass  # Will implement below to keep file clean

        if ext in [".csv", ".txt", ".tsv"]:
            try:
                df = pd.read_csv(input_file, sep=None, engine="python")
            except:
                df = pd.read_csv(input_file)

            # Column mapping
            cols = {c.lower(): c for c in df.columns}

            # Find ID
            tgt_id = None
            for k in cols:
                if k in ["id", "name", "identifier", id_col.lower()]:
                    tgt_id = cols[k]
                    break

            # Find SMILES
            tgt_smi = None
            for k in cols:
                if k in ["smiles", "smi", "canonical_smiles", smiles_col.lower()]:
                    tgt_smi = cols[k]
                    break

            if tgt_smi:
                for i, row in df.iterrows():
                    smi = row[tgt_smi]
                    lid = row[tgt_id] if tgt_id else f"lig_{i+1}"
                    mw = 0
                    try:
                        mol = Chem.MolFromSmiles(str(smi))
                        if mol:
                            mw = Descriptors.MolWt(mol)
                    except:
                        pass
                    records.append({"ID": str(lid), "SMILES": str(smi), "MolWt": mw})
            else:
                # Headerless assumption check
                first = str(df.iloc[0, 0])
                if Chem.MolFromSmiles(first):
                    for i, row in df.iterrows():
                        smi = row[0]
                        lid = row[1] if len(row) > 1 else f"lig_{i+1}"
                        records.append({"ID": str(lid), "SMILES": str(smi), "MolWt": 0})

    # --- Write Output ---
    if not records:
        logger.warning("No records found.")
        return ""

    df_out = pd.DataFrame(records)
    if "MolWt" not in df_out.columns:
        df_out["MolWt"] = 0.0
    df_out["MolWt"] = df_out["MolWt"].fillna(0.0)

    # If we found 3D structures, the "SMILES" key in records actually holds paths.
    # Rename column to "Structure_Path" for clarity.
    if found_3d:
        df_out.rename(columns={"SMILES": "Structure_Path"}, inplace=True)
        # Ensure we don't have a "SMILES" column anymore to avoid confusion,
        # unless we want both? For now, VinaDock will check for one or the other.

    df_out.to_csv(output_csv, index=False)

    if found_3d:
        logger.info(
            f"Standardization complete. Found 3D structures. Saved split files to {structures_dir}"
        )
        logger.info(f"CSV contains 'Structure_Path' column pointing to SDF/PDB files.")
        logger.info(
            "NOTE: Since 3D structures are present, you should likely SKIP ligand_preprocessing and proceed directly to docking with 'keep_local_structures=True'."
        )
    else:
        logger.info(
            f"Standardization complete. Extracted 2D SMILES. Saved to {output_csv}"
        )

    return output_csv
