import glob
import os
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

try:
    from meeko import PDBQTMolecule, RDKitMolCreate
except ImportError:
    PDBQTMolecule = None
    RDKitMolCreate = None

from ...logger_utils import get_logger
from ...utils import run_cmd


def clean_gro_file():
    """Remove cached GROMACS files from current directory."""
    patterns = ["#*", "*.tpr", "*.trr", "*.cpt", "*.edr", "*.log"]
    for pattern in patterns:
        for filepath in glob.glob(pattern):
            try:
                os.remove(filepath)
            except OSError:
                pass  # Ignore errors (file already deleted, permissions, etc.)


def get_gromacs_binary(
    gromacs_path: str, binary_type: str = "mpi", require_mpi: bool = False
):
    """
    Get the correct GROMACS binary path based on binary type.

    Args:
        gromacs_path: Path to GROMACS installation directory (containing bin/)
                     Typically $CONDA_PREFIX/gromacs-2024.4/bin
                     For CUDA, function will check $CONDA_PREFIX/gromacs-2024.4/cuda/bin
        binary_type: Type of GROMACS binary. Options:
            - "mpi": Use gmx_mpi (MPI-enabled, default)
            - "cuda": Use gmx from cuda/bin directory (CUDA-enabled)
            - "standard": Use gmx (standard, no MPI/CUDA)
        require_mpi: If True, will use gmx_mpi even if binary_type is not "mpi"
                    (useful for operations that need MPI)

    Returns:
        str: Full path to the GROMACS binary

    Raises:
        FileNotFoundError: If the requested binary is not found
    """
    if require_mpi or binary_type == "mpi":
        binary_name = "gmx_mpi"
    elif binary_type == "cuda":
        binary_name = "gmx"  # CUDA builds use 'gmx'
    else:  # binary_type == "standard"
        binary_name = "gmx"

    # For CUDA, check cuda/bin subdirectory
    if binary_type == "cuda":
        # If gromacs_path is already the CUDA path, use it directly
        if "/cuda/bin" in gromacs_path:
            binary_path = os.path.join(gromacs_path, binary_name)
            if os.path.exists(binary_path):
                return binary_path
            raise FileNotFoundError(
                f"GROMACS CUDA binary not found at: {binary_path}\n"
                f"Please ensure GROMACS CUDA is installed at {gromacs_path}"
            )

        # If gromacs_path is .../gromacs-2024.4/bin, check .../gromacs-2024.4/cuda/bin
        # TODO: Update once paths are more standardized
        if gromacs_path.endswith("/bin"):
            cuda_path = gromacs_path.replace("/bin", "/cuda/bin")
            cuda_binary_path = os.path.join(cuda_path, binary_name)
            if os.path.exists(cuda_binary_path):
                return cuda_binary_path
            # Also check if standard path exists as fallback
            standard_binary_path = os.path.join(gromacs_path, binary_name)
            if os.path.exists(standard_binary_path):
                logger = get_logger()
                logger.warning(
                    f"CUDA binary not found at {cuda_path}, using standard binary at {gromacs_path}"
                )
                return standard_binary_path
            raise FileNotFoundError(
                f"GROMACS CUDA binary not found at: {cuda_binary_path}\n"
                f"Expected CUDA path: {cuda_path}\n"
                f"Standard binary also not found at: {standard_binary_path}\n"
                f"Please ensure GROMACS CUDA is installed at $CONDA_PREFIX/gromacs-2024.4/cuda/bin"
            )

        # Unexpected path format
        raise FileNotFoundError(
            f"Invalid GROMACS path format for CUDA binary: {gromacs_path}\n"
            f"Expected format: $CONDA_PREFIX/gromacs-2024.4/bin or $CONDA_PREFIX/gromacs-2024.4/cuda/bin"
        )

    # Standard path check (for non-CUDA binary types)
    binary_path = os.path.join(gromacs_path, binary_name)

    if not os.path.exists(binary_path):
        raise FileNotFoundError(
            f"GROMACS binary not found: {binary_path}\n"
            f"Requested type: {binary_type}, Binary name: {binary_name}\n"
            f"Please ensure GROMACS is installed at {gromacs_path}"
        )

    return binary_path


def combine_gro(protein_gro, ligand_gro, out_gro):
    """
    Combine a protein.gro and lig.gro into one complex.gro file.
    Assumes both files use the same box dimensions.
    """
    # Read protein
    with open(protein_gro, "r") as f:
        prot_lines = f.readlines()
    # Read ligand
    with open(ligand_gro, "r") as f:
        lig_lines = f.readlines()

    # Extract header & box from protein
    title = prot_lines[0].strip() + " + Ligand\n"
    n_prot_atoms = int(prot_lines[1].strip())
    box_line = prot_lines[-1]

    # Ligand info
    n_lig_atoms = int(lig_lines[1].strip())

    # Coordinates (skip first 2 lines and last box line)
    prot_atoms = prot_lines[2:-1]
    lig_atoms = lig_lines[2:-1]

    # Renumber ligand atom indices after protein atoms
    combined_atoms = []
    atom_counter = 1
    for line in prot_atoms + lig_atoms:
        # .gro atom index is columns 15–20 (right aligned)
        new_line = line[:15] + f"{atom_counter:5d}" + line[20:]
        combined_atoms.append(new_line)
        atom_counter += 1

    # Total atoms
    n_total = n_prot_atoms + n_lig_atoms

    # Write combined gro
    with open(out_gro, "w") as f:
        f.write(title)
        f.write(f"{n_total:5d}\n")
        f.writelines(combined_atoms)
        f.write(box_line)


def add_ligand_topology_with_atomtypes(
    topol_file,
    lig_itp_file,
    output_top,
    ligand_name="LIG",
    ligand_count=1,
    ff_include=None,
):
    """
    - Copies ligand itp to lig_with_atomtypes.itp
    - Removes [ atomtypes ] section from the original lig.itp
    - Inserts [ atomtypes ] section into topol.top after forcefield include

    Args:
        ff_include: If None, auto-detects the forcefield from the topology file.
                    Otherwise, uses the provided forcefield include string.
    """

    # 1. Copy ligand itp
    lig_copy = lig_itp_file.replace(".itp", "_with_atomtypes.itp")
    shutil.copy(lig_itp_file, lig_copy)

    # 2. Read ligand itp and extract [ atomtypes ] section
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
            if line.strip().startswith("["):  # reached next section
                in_atomtypes = False
                new_lines.append(line)
            else:
                atomtypes_block.append(line)
                continue
        else:
            new_lines.append(line)

    # Write back modified lig.itp without atomtypes
    new_lig_itp_file = lig_itp_file.replace("LIG.acpype/", "")
    logger = get_logger()
    logger.debug(f"New ligand ITP file: {new_lig_itp_file}")
    with open(new_lig_itp_file, "w") as f:
        f.writelines(new_lines)

    # 3. Insert atomtypes block into topol.top
    with open(topol_file, "r") as f:
        top_lines = f.readlines()

    # Auto-detect forcefield include if not provided
    if ff_include is None:
        import re

        for line in top_lines:
            # Look for pattern: #include "*.ff/forcefield.itp"
            match = re.search(r'#include\s+"([^"]+\.ff/forcefield\.itp)"', line)
            if match:
                ff_include = match.group(1)
                logger.info(f"Auto-detected forcefield: {ff_include}")
                break

        if ff_include is None:
            ff_include = "amber03.ff/forcefield.itp"
            logger.warning(
                f"Could not auto-detect forcefield, using default: {ff_include}"
            )

    new_lines = []
    posres_block_found = False
    ff_inserted = False

    for line in top_lines:
        new_lines.append(line)

        # Insert ligand include after POSRES block
        if "#endif" in line and not posres_block_found:
            new_lines.append(f'\n; Include ligand topology\n#include "LIG_GMX.itp"\n')
            new_lines.append(
                f'\n; Ligand position restraints\n#ifdef POSRES\n#include "posre_LIG.itp"\n#endif\n'
            )
            posres_block_found = True

        # Insert ligand atomtypes after forcefield include
        if ff_include in line and not ff_inserted:
            new_lines.extend(atomtypes_block)
            ff_inserted = True

    # Append ligand molecule entry at the end
    new_lines.append(f"{ligand_name:<10}{ligand_count}\n")
    # Write back
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
    """
    Extracts a specific MODEL (pose) from a PDBQT file and saves it.
    """
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
    """
    Convert PDB to MOL2 using OpenBabel (obabel).
    Output is redirected directly to logger.
    """
    logger = get_logger()
    logger.info(f"Converting {pdb_file} to {mol2_file} using OpenBabel")
    try:
        # Use run_cmd to redirect output to logger
        run_cmd(["obabel", pdb_file, "-O", mol2_file], check=True)
        logger.info(f"Successfully converted {pdb_file} to {mol2_file}")
    except subprocess.CalledProcessError as e:
        logger.error(f"OpenBabel conversion failed")
        raise


def restore_from_pdbqt(smiles_file, query_id, pdbqt_file, out_mol):
    smiles = get_smiles_by_id(smiles_file, query_id)
    orig = Chem.MolFromSmiles(smiles)
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


def restore_from_structure(
    structure_file: str,
    pdbqt_file: str,
    out_mol: str,
) -> Chem.Mol:
    """
    Restore molecular structure from SDF/MOL2 file (no SMILES needed).

    Args:
        structure_file: Path to SDF or MOL2 file
        pdbqt_file: Path to docked PDBQT pose
        out_mol: Output MOL2 file path

    Returns:
        RDKit Mol object with correct bond orders
    """
    logger = get_logger()

    # Read structure file
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

    # Extract coordinates from PDBQT
    pdbqt_mol = PDBQTMolecule.from_file(pdbqt_file, skip_typing=True)
    rdkitmol_list = RDKitMolCreate.from_pdbqt_mol(pdbqt_mol)
    if not rdkitmol_list:
        raise ValueError("No molecules parsed from PDBQT")

    docked = rdkitmol_list[0]

    # Assign bond orders from structure template
    restored = AllChem.AssignBondOrdersFromTemplate(mol, docked)
    restored = Chem.AddHs(restored, addCoords=True)

    # Convert to MOL2
    out_pdb = f"{out_mol[:-5]}.pdb"
    Chem.MolToPDBFile(restored, out_pdb)
    pdb_to_mol2(out_pdb, out_mol)

    logger.debug(f"Restored structure from {structure_file} to {out_mol}")
    return restored


def get_smiles_by_id(file, query_id):
    df = pd.read_csv(file)
    result = df.loc[df["ID"] == query_id, "SMILES"]
    if not result.empty:
        return result.values[0]  # first match
    else:
        return None


def formal_charge(input_file):
    mol = Chem.MolFromMol2File(input_file, sanitize=False, removeHs=False)
    if mol is None:
        raise ValueError("Cannot read molecule file.")
    return Chem.GetFormalCharge(mol)


def parse_xvg(fname):
    """Parse .xvg file into (time, value) arrays."""
    times, values = [], []
    if os.path.exists(fname):
        with open(fname) as f:
            for line in f:
                if line.startswith(("#", "@")):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        times.append(float(parts[0]))
                        values.append(float(parts[1]))
                    except ValueError:
                        continue
    return np.array(times), np.array(values)


def parse_pose_name(pose_name):
    """Parse pose_name = {Lig_ID}_pocket_{g_id}_top{rank}."""
    match = re.match(r"^(.*)_pocket_(\d+)_top(\d+)$", pose_name)
    if match:
        lig_id, g_id, rank = match.groups()
        return lig_id, int(g_id), int(rank)
    return pose_name, None, None


def mean_std(arr):
    if arr is not None and len(arr) > 0:
        return float(arr.mean()), float(arr.std())
    else:
        return None, None


def get_mdp_dir():
    """
    Get the path to the MDP files directory.

    MDP files are always in the mdp subdirectory of md_analysis package.

    Returns:
        str: Path to the mdp directory
    """
    return os.path.join(os.path.dirname(__file__), "mdp")
