"""
Conformer Generation Utility
"""

import os

import pandas as pd
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem
from rdkit.Chem import AddHs, MolFromSmiles
from rdkit.Chem.AllChem import EmbedMolecule, UFFOptimizeMolecule

from ...logger_utils import get_logger
from ...utils.multiprocessing_utils import Pool, cpu_count


def _process_single_ligand(args):
    """
    Worker function to process a single ligand.

    Args:
        args: Tuple of (smiles, lig_id, conformers_dir)

    Returns:
        tuple: (success: bool, pdbqt_path: str or None, error_msg: str or None)
    """
    smiles, lig_id, conformers_dir = args

    try:
        # Generate 3D conformer
        mol = MolFromSmiles(smiles)
        if not mol:
            return (False, None, f"Failed to parse SMILES for {lig_id}")

        # Add hydrogens
        mol = AddHs(mol)

        # Generate 3D coordinates
        result = EmbedMolecule(mol, randomSeed=42)
        if result == -1:
            return (False, None, f"Failed to embed molecule {lig_id}")

        # Optimize geometry
        try:
            UFFOptimizeMolecule(mol)
        except:
            # UFF optimization can fail, but we can still proceed
            pass  # Use unoptimized conformer

        # Convert to PDBQT using Meeko
        ligprep = MoleculePreparation()
        mol_setups = ligprep.prepare(mol)
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])

        if is_ok:
            clean_id = "".join([c if c.isalnum() else "_" for c in lig_id])
            pdbqt_path = os.path.join(conformers_dir, f"{clean_id}.pdbqt")

            with open(pdbqt_path, "w") as f:
                f.write(pdbqt_string)

            return (True, os.path.abspath(pdbqt_path), None)
        else:
            return (False, None, f"PDBQT conversion failed for {lig_id}: {error_msg}")

    except Exception as e:
        return (False, None, f"Error processing {lig_id}: {e}")


def generate_conformers_from_csv(
    input_csv: str, out_folder: str = "out_folder"
) -> list:
    """
    Reads a CSV file containing SMILES, generates 3D conformers, converts to PDBQT.

    Uses parallel processing for datasets with 20+ ligands to improve performance.
    For smaller datasets, processes sequentially to avoid multiprocessing overhead.

    Args:
        input_csv: Path to input CSV with "SMILES" and "ID" columns.
        out_folder: Folder to save generated PDBQT files.

    Returns:
        list: Paths to generated PDBQT files
    """
    logger = get_logger()

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)

    if "SMILES" not in df.columns:
        raise ValueError("Input CSV must contain a 'SMILES' column.")
    if "ID" not in df.columns:
        # Generate IDs if missing
        logger.warning("'ID' column missing. Generating sequential IDs.")
        df["ID"] = [f"lig_{i}" for i in range(len(df))]

    conformers_dir = os.path.join(out_folder, "conformers_pdbqt")
    os.makedirs(conformers_dir, exist_ok=True)

    num_ligands = len(df)
    logger.info(
        f"Generating conformers and converting to PDBQT for {num_ligands} ligands..."
    )

    # Prepare arguments for processing
    args_list = [
        (row["SMILES"], str(row["ID"]), conformers_dir) for _, row in df.iterrows()
    ]

    pdbqt_paths = []

    # Use parallel processing for larger datasets (threshold: 20 ligands)
    # For small datasets, multiprocessing overhead outweighs benefits
    PARALLEL_THRESHOLD = 20

    if num_ligands >= PARALLEL_THRESHOLD:
        # Parallel processing
        num_workers = min(cpu_count(), num_ligands)
        logger.info(f"Using parallel processing with {num_workers} workers")

        with Pool(processes=num_workers) as pool:
            results = pool.map(_process_single_ligand, args_list)

        # Collect successful results
        for success, pdbqt_path, error_msg in results:
            if success:
                pdbqt_paths.append(pdbqt_path)
            elif error_msg:
                logger.warning(error_msg)
    else:
        # Sequential processing for small datasets
        logger.debug(
            f"Using sequential processing for {num_ligands} ligands (below threshold of {PARALLEL_THRESHOLD})"
        )

        for success, pdbqt_path, error_msg in map(_process_single_ligand, args_list):
            if success:
                pdbqt_paths.append(pdbqt_path)
            elif error_msg:
                logger.warning(error_msg)

    logger.info(f"Generated {len(pdbqt_paths)} PDBQT files from {num_ligands} SMILES")
    return pdbqt_paths
