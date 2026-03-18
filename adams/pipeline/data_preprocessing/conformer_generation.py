"""
Conformer generation from SMILES: multi-conformer embedding and PDBQT output.

Row-unique filenames (Variant_ID when present, else ID) so enumerated microstates
get distinct files.
"""

import math
import os

import pandas as pd

from ...logger_utils import get_logger
from ...utils.multiprocessing_utils import cpu_count
from ...utils.parallel_executor import ParallelExecutor, ResourceConfig
from ..charge_model import validate_charge_model


def row_to_file_id(row: pd.Series) -> str:
    """Row-unique id for PDBQT filenames."""
    if "Variant_ID" in row.index and pd.notna(row.get("Variant_ID")):
        # Variant_ID values repeat across ligands (e.g., "original_0"),
        # so prefix with parent ID to avoid filename collisions.
        return f"{row['ID']}__{row['Variant_ID']}"
    return str(row["ID"])


def _process_one_ligand(args):
    """
    Generate conformers for one molecule and write PDBQT files.
    Worker-friendly: takes a single tuple, returns
    (success, paths, smiles, row_id, variant_id, parent_id, variant_type, error_msg).
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem
    from rdkit.Chem import AddHs, MolFromSmiles
    from rdkit.Chem import rdDistGeom
    from rdkit.Chem.AllChem import (
        MMFFGetMoleculeForceField,
        MMFFGetMoleculeProperties,
        MMFFOptimizeMolecule,
        UFFGetMoleculeForceField,
        UFFOptimizeMolecule,
    )

    (
        smiles,
        file_id,
        pdbqt_dir,
        num_confs,
        max_confs_to_keep,
        conformer_energy_window_kcal,
        random_seed,
        row_id,
        variant_id,
        parent_id,
        variant_type,
        charge_model,
    ) = args

    def _err(msg):
        return (
            False,
            [],
            smiles,
            str(row_id),
            variant_id,
            parent_id,
            variant_type,
            msg,
        )

    mol = MolFromSmiles(smiles)
    if not mol:
        return _err("Failed to parse SMILES")

    mol = AddHs(mol)
    params = rdDistGeom.ETKDGv2()
    params.randomSeed = random_seed
    rdDistGeom.EmbedMultipleConfs(mol, num_confs, params)
    if mol.GetNumConformers() == 0:
        return _err("Failed to embed molecule")

    clean_id = "".join(c if c.isalnum() else "_" for c in file_id)
    ligprep = MoleculePreparation(charge_model=charge_model)
    paths = []

    conf_energy = []
    for conf in mol.GetConformers():
        cid = conf.GetId()
        try:
            mmff_props = MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
            if mmff_props is not None:
                MMFFOptimizeMolecule(mol, confId=cid, mmffVariant="MMFF94s")
                ff = MMFFGetMoleculeForceField(mol, mmff_props, confId=cid)
                e = float(ff.CalcEnergy()) if ff is not None else float("inf")
            else:
                UFFOptimizeMolecule(mol, confId=cid)
                ff = UFFGetMoleculeForceField(mol, confId=cid)
                e = float(ff.CalcEnergy()) if ff is not None else float("inf")
        except Exception:
            e = float("inf")
        conf_energy.append((cid, e))

    if not conf_energy:
        return _err("No conformers available after embedding")

    # Keep only low-energy conformers to avoid combinatorial blow-up.
    conf_energy.sort(key=lambda item: item[1])
    finite_conf_energy = [(cid, e) for cid, e in conf_energy if math.isfinite(e)]
    if finite_conf_energy:
        best_e = finite_conf_energy[0][1]
        selected = [
            (cid, e)
            for cid, e in finite_conf_energy
            if (e - best_e) <= conformer_energy_window_kcal
        ]
        selected = selected[:max_confs_to_keep]
        if not selected:
            selected = [finite_conf_energy[0]]
    else:
        # Fallback if all forcefield evaluations fail.
        selected = [conf_energy[0]]

    for i, (cid, e_kcal) in enumerate(selected):
        # Write exactly one conformer per PDBQT output.
        mol_one = Chem.Mol(mol)
        mol_one.RemoveAllConformers()
        conf_copy = Chem.Conformer(mol.GetConformer(cid))
        conf_copy.SetId(0)
        mol_one.AddConformer(conf_copy, assignId=True)
        mol_setups = ligprep.prepare(mol_one)
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
        if not is_ok:
            return _err(f"PDBQT failed: {error_msg}")
        name = f"{clean_id}.pdbqt" if num_confs == 1 else f"{clean_id}_conf{i}.pdbqt"
        path = os.path.join(pdbqt_dir, name)
        with open(path, "w") as f:
            f.write(pdbqt_string)
        paths.append((os.path.abspath(path), e_kcal))

    return (
        True,
        paths,
        smiles,
        str(row_id),
        variant_id,
        parent_id,
        variant_type,
        None,
    )


def generate_conformers_to_pdbqt(
    input_csv: str,
    output_dir: str,
    num_confs: int = 8,
    max_confs_to_keep: int = 2,
    conformer_energy_window_kcal: float = 3.0,
    random_seed: int = 42,
    charge_model: str = "gasteiger",
) -> str:
    """
    Generate 3D conformers from SMILES CSV and convert to PDBQT (parallel).

    Uses row-unique filenames: Variant_ID when present, else ID.

    Args:
        input_csv: Path to CSV with SMILES and ID (optionally Variant_ID).
        output_dir: Directory for pdbqt_files/ and mapping CSV.
        num_confs: Conformers generated per molecule before pruning (default 8).
        max_confs_to_keep: Maximum conformers retained per molecule after ranking (default 2).
        conformer_energy_window_kcal: Retain conformers within this energy window from best (default 3.0).
        random_seed: Seed for reproducible embedding.
        charge_model: Meeko partial charge model (default "gasteiger"). Must match receptor.

    Returns:
        str: Path to the mapping CSV
            (ID, PDBQT_File, Parent_ID, Variant_Type, Variant_ID?,
            Conformer_Index, Charge_Model).
    """
    logger = get_logger()
    charge_model = validate_charge_model(charge_model)
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if num_confs < 1:
        raise ValueError(f"num_confs must be >= 1, got {num_confs}")
    if max_confs_to_keep < 1:
        raise ValueError(
            f"max_confs_to_keep must be >= 1, got {max_confs_to_keep}"
        )

    df = pd.read_csv(input_csv)
    if "SMILES" not in df.columns:
        raise ValueError(f"Input CSV must have 'SMILES' column. Found: {list(df.columns)}")
    if "ID" not in df.columns:
        raise ValueError(f"Input CSV must have 'ID' column. Found: {list(df.columns)}")

    pdbqt_dir = os.path.join(output_dir, "pdbqt_files")
    os.makedirs(pdbqt_dir, exist_ok=True)

    args_list = []
    for _, row in df.iterrows():
        file_id = row_to_file_id(row)
        row_id = str(row["ID"])
        variant_id = row.get("Variant_ID")
        variant_id = None if pd.isna(variant_id) else str(variant_id)
        parent_id = row.get("Parent_ID")
        if pd.isna(parent_id):
            parent_id = row_id
        parent_id = str(parent_id)

        variant_type = row.get("Variant_Type")
        if pd.isna(variant_type) or not str(variant_type).strip():
            variant_type = "original"
        variant_type = str(variant_type)
        args_list.append(
            (
                row["SMILES"],
                file_id,
                pdbqt_dir,
                num_confs,
                max_confs_to_keep,
                conformer_energy_window_kcal,
                random_seed,
                row_id,
                variant_id,
                parent_id,
                variant_type,
                charge_model,
            )
        )

    n_rows = len(args_list)
    n_workers = min(cpu_count(), n_rows) if n_rows else 1

    config = ResourceConfig(n_workers=n_workers)
    executor = ParallelExecutor(config)

    logger.info(f"Generating conformers for {n_rows} molecules...")
    task_results = executor.run(
        _process_one_ligand,
        args_list,
        task_id_fn=lambda a: str(a[7]),
    )

    mapping_rows = []
    for tr in task_results:
        raw = tr.value if tr.success and tr.value is not None else None
        if raw is None:
            if tr.error:
                logger.warning(f"Conformer task failed: {tr.error}")
            continue

        success, paths, smiles, row_id, variant_id, parent_id, variant_type, err = raw
        if not success and err:
            logger.warning(f"Row {row_id}: {err}")
            continue

        for i, (path, e_kcal) in enumerate(paths):
            row = {
                "ID": row_id,
                "SMILES": smiles,
                "PDBQT_File": path,
                "Conformer_Index": i,
                "Conformer_Energy_kcal": e_kcal,
                "Charge_Model": charge_model,
                "Parent_ID": parent_id,
                "Variant_Type": variant_type,
            }
            if variant_id is not None:
                row["Variant_ID"] = variant_id
            mapping_rows.append(row)

    if not mapping_rows:
        raise RuntimeError(
            "No PDBQT entries were generated. Check SMILES validity and Meeko conversion logs."
        )
    out_df = pd.DataFrame(mapping_rows)
    mapping_csv = os.path.join(output_dir, "docking_ready_ligands.csv")
    out_df.to_csv(mapping_csv, index=False)
    logger.info(f"Generated {len(mapping_rows)} PDBQT entries from {n_rows} rows -> {mapping_csv}")
    return mapping_csv
