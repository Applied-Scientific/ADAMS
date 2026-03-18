"""
Ligand/topology/structure conversion helpers used during MD preparation.
"""

import os
import shutil
import subprocess

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from meeko import PDBQTMolecule, RDKitMolCreate
except ImportError:
    PDBQTMolecule = None
    RDKitMolCreate = None

from ....logger_utils import get_logger
from ....utils import run_cmd
from ..shared import LIGAND_RESNAME


def combine_gro(protein_gro, ligand_gro, out_gro):
    """
    Combine a protein.gro and lig.gro into one complex.gro file.
    Assumes both files use the same box dimensions.
    """
    with open(protein_gro, "r") as f:
        prot_lines = f.readlines()
    with open(ligand_gro, "r") as f:
        lig_lines = f.readlines()

    title = prot_lines[0].strip() + " + Ligand\n"
    n_prot_atoms = int(prot_lines[1].strip())
    box_line = prot_lines[-1]
    n_lig_atoms = int(lig_lines[1].strip())

    prot_atoms = prot_lines[2:-1]
    lig_atoms = lig_lines[2:-1]

    combined_atoms = []
    atom_counter = 1
    for line in prot_atoms + lig_atoms:
        new_line = line[:15] + f"{atom_counter:5d}" + line[20:]
        combined_atoms.append(new_line)
        atom_counter += 1

    n_total = n_prot_atoms + n_lig_atoms
    with open(out_gro, "w") as f:
        f.write(title)
        f.write(f"{n_total:5d}\n")
        f.writelines(combined_atoms)
        f.write(box_line)


def add_ligand_topology_with_atomtypes(
    topol_file,
    lig_itp_file,
    output_top,
    ligand_name=None,
    ligand_count=1,
    ff_include=None,
):
    """
    Copy ligand ITP, strip [ atomtypes ], and splice atomtypes into topol.top.
    """
    if ligand_name is None:
        ligand_name = LIGAND_RESNAME

    lig_copy = lig_itp_file.replace(".itp", "_with_atomtypes.itp")
    shutil.copy(lig_itp_file, lig_copy)

    with open(lig_itp_file, "r") as f:
        lines = f.readlines()

    atomtypes_block = []
    new_lines = []
    in_atomtypes = False

    for line in lines:
        if line.strip().startswith("[ atomtypes ]"):
            in_atomtypes = True
            atomtypes_block.append(line)
            continue

        if in_atomtypes:
            if line.strip().startswith("["):
                in_atomtypes = False
                new_lines.append(line)
            else:
                atomtypes_block.append(line)
                continue
        else:
            new_lines.append(line)

    new_lig_itp_file = lig_itp_file.replace(f"{ligand_name}.acpype/", "")
    logger = get_logger()
    logger.debug(f"New ligand ITP file: {new_lig_itp_file}")
    with open(new_lig_itp_file, "w") as f:
        f.writelines(new_lines)

    with open(topol_file, "r") as f:
        top_lines = f.readlines()

    if ff_include is None:
        import re

        for line in top_lines:
            match = re.search(r'#include\s+"([^"]+\.ff/forcefield\.itp)"', line)
            if match:
                ff_include = match.group(1)
                logger.info(f"Auto-detected forcefield: {ff_include}")
                break

        if ff_include is None:
            ff_include = "amber99sb-ildn.ff/forcefield.itp"
            logger.warning(
                f"Could not auto-detect forcefield, using default: {ff_include}"
            )

    new_lines = []
    posres_block_found = False
    ff_inserted = False

    for line in top_lines:
        new_lines.append(line)
        if "#endif" in line and not posres_block_found:
            new_lines.append(f'\n; Include ligand topology\n#include "{ligand_name}_GMX.itp"\n')
            new_lines.append(
                f'\n; Ligand position restraints\n#ifdef POSRES\n#include "posre_{ligand_name}.itp"\n#endif\n'
            )
            posres_block_found = True
        if ff_include in line and not ff_inserted:
            new_lines.extend(atomtypes_block)
            ff_inserted = True

    new_lines.append(f"{ligand_name:<10}{ligand_count}\n")
    with open(output_top, "w") as f:
        f.writelines(new_lines)

    if not ff_inserted:
        logger.warning(
            f"Forcefield include '{ff_include}' not found in topology file. Atomtypes may not have been inserted."
        )
    else:
        logger.info(
            f"Ligand topology and atomtypes from {lig_itp_file} added to {output_top}."
        )


def extract_pose_from_pdbqt(pdbqt_file, out_file, pose_id):
    """Extract a specific MODEL (pose) from a PDBQT file."""
    with open(pdbqt_file, "r") as f:
        lines = f.readlines()

    model_blocks = []
    current_model = []
    inside_model = False

    for line in lines:
        if line.startswith("MODEL"):
            inside_model = True
            current_model = [line]
        elif line.startswith("ENDMDL"):
            current_model.append(line)
            model_blocks.append(current_model)
            inside_model = False
        elif inside_model:
            current_model.append(line)

    if pose_id >= len(model_blocks):
        raise ValueError(
            f"Pose {pose_id} not found in {pdbqt_file} (only {len(model_blocks)} models available)"
        )

    with open(out_file, "w") as f:
        f.writelines(model_blocks[pose_id])


def pdb_to_mol2(pdb_file, mol2_file):
    """Convert PDB to MOL2 using OpenBabel."""
    logger = get_logger()
    logger.info(f"Converting {pdb_file} to {mol2_file} using OpenBabel")
    try:
        run_cmd(["obabel", pdb_file, "-O", mol2_file], check=True)
        logger.info(f"Successfully converted {pdb_file} to {mol2_file}")
    except subprocess.CalledProcessError:
        logger.error("OpenBabel conversion failed")
        raise


def restore_from_pdbqt(smiles_file, query_id, pdbqt_file, out_mol):
    smiles = get_smiles_by_id(smiles_file, query_id)
    if smiles is None or (isinstance(smiles, float) and pd.isna(smiles)):
        raise ValueError(
            f"Could not resolve SMILES for ligand ID '{query_id}' from {smiles_file}"
        )
    orig = Chem.MolFromSmiles(smiles)
    if orig is None:
        raise ValueError(
            f"Invalid SMILES for ligand ID '{query_id}': {smiles!r} (source: {smiles_file})"
        )
    orig = Chem.AddHs(orig)

    pdbqt_mol = PDBQTMolecule.from_file(pdbqt_file, skip_typing=True)
    rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    if not rdkitmol_list:
        raise ValueError("No molecules parsed from PDBQT")

    docked = rdkitmol_list[0]
    restored = AllChem.AssignBondOrdersFromTemplate(orig, docked)
    restored = Chem.AddHs(restored, addCoords=True)

    out_pdb = f"{out_mol[:-5]}.pdb"
    Chem.MolToPDBFile(restored, out_pdb)
    pdb_to_mol2(out_pdb, out_mol)
    return restored


def restore_from_pdbqt_with_smiles(smiles: str, pdbqt_file: str, out_mol: str) -> Chem.Mol:
    """
    Restore MOL2 from PDBQT using a known SMILES string as the bond-order template.
    Use when SMILES is already available (e.g. from docking CSV) to avoid file lookup.
    """
    if not smiles or (isinstance(smiles, float) and pd.isna(smiles)):
        raise ValueError("SMILES string is required for restore_from_pdbqt_with_smiles")
    orig = Chem.MolFromSmiles(smiles)
    if orig is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    orig = Chem.AddHs(orig)

    pdbqt_mol = PDBQTMolecule.from_file(pdbqt_file, skip_typing=True)
    rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    if not rdkitmol_list:
        raise ValueError("No molecules parsed from PDBQT")

    docked = rdkitmol_list[0]
    restored = AllChem.AssignBondOrdersFromTemplate(orig, docked)
    restored = Chem.AddHs(restored, addCoords=True)

    out_pdb = f"{out_mol[:-5]}.pdb"
    Chem.MolToPDBFile(restored, out_pdb)
    pdb_to_mol2(out_pdb, out_mol)
    return restored


def restore_from_pdbqt_structure_only(pdbqt_file: str, out_mol: str) -> Chem.Mol:
    """
    Build MOL2 from PDBQT 3D structure only (no SMILES template).
    Use when AssignBondOrdersFromTemplate fails (e.g. protomer/tautomer mismatch).
    Bond orders and charge are inferred from the PDBQT/meeko conversion.
    """
    logger = get_logger()
    pdbqt_mol = PDBQTMolecule.from_file(pdbqt_file, skip_typing=True)
    rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    if not rdkitmol_list:
        raise ValueError("No molecules parsed from PDBQT")
    mol = rdkitmol_list[0]
    mol = Chem.AddHs(mol, addCoords=True)
    out_pdb = f"{out_mol[:-5]}.pdb"
    Chem.MolToPDBFile(mol, out_pdb)
    pdb_to_mol2(out_pdb, out_mol)
    logger.debug(f"Restored structure from PDBQT only (no SMILES template) to {out_mol}")
    return mol


def restore_from_structure(
    structure_file: str,
    pdbqt_file: str,
    out_mol: str,
) -> Chem.Mol:
    """
    Restore molecular structure from SDF/MOL2 file (no SMILES needed).
    """
    logger = get_logger()

    if structure_file.endswith(".sdf") or structure_file.endswith(".sd"):
        mol = Chem.MolFromMolFile(structure_file, removeHs=False)
    elif structure_file.endswith(".mol2"):
        mol = Chem.MolFromMol2File(structure_file, removeHs=False)
    else:
        raise ValueError(
            f"Unsupported structure format: {structure_file}. Expected .sdf or .mol2"
        )

    if mol is None:
        raise ValueError(f"Failed to read molecule from {structure_file}")

    pdbqt_mol = PDBQTMolecule.from_file(pdbqt_file, skip_typing=True)
    rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    if not rdkitmol_list:
        raise ValueError("No molecules parsed from PDBQT")

    docked = rdkitmol_list[0]
    restored = AllChem.AssignBondOrdersFromTemplate(mol, docked)
    restored = Chem.AddHs(restored, addCoords=True)

    out_pdb = f"{out_mol[:-5]}.pdb"
    Chem.MolToPDBFile(restored, out_pdb)
    pdb_to_mol2(out_pdb, out_mol)

    logger.debug(f"Restored structure from {structure_file} to {out_mol}")
    return restored


def get_smiles_by_id(file, query_id):
    df = pd.read_csv(file)

    smiles_col = None
    for candidate in ("SMILES", "inSMILES"):
        if candidate in df.columns:
            smiles_col = candidate
            break
    if smiles_col is None:
        raise ValueError(
            f"No SMILES column found in {file}. Expected one of: SMILES, inSMILES"
        )

    query = str(query_id).strip()

    if "ID" in df.columns:
        result = df.loc[df["ID"].astype(str) == query, smiles_col]
        if not result.empty:
            return result.values[0]

    base_id = query.split("__", 1)[0]

    if "ID" in df.columns and base_id != query:
        result = df.loc[df["ID"].astype(str) == base_id, smiles_col]
        if not result.empty:
            return result.values[0]

    if "Parent_ID" in df.columns:
        result = df.loc[df["Parent_ID"].astype(str) == base_id, smiles_col]
        if not result.empty:
            return result.values[0]

    return None


def formal_charge(input_file):
    mol = Chem.MolFromMol2File(input_file, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError("Cannot read molecule file.")
    return Chem.GetFormalCharge(mol)


def formal_charge_from_smiles(smiles: str) -> int:
    """Return formal charge for a molecule given its SMILES (e.g. for ACPYPE -n when using structure-only restore)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for charge: {smiles!r}")
    return Chem.GetFormalCharge(mol)

