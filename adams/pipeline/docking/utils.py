import os
import subprocess
import tempfile
from contextlib import contextmanager

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from ...logger_utils import get_logger
from ...utils import run_cmd

# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------


@contextmanager
def capture_stderr(logger, context_info=""):
    """
    Context manager to capture stderr from Vina operations and log errors.

    The Vina Python bindings write error messages directly to stderr file descriptor
    without raising Python exceptions. This context manager captures those messages
    by redirecting the actual file descriptor using a pipe (no files created).
    If critical errors are detected (like missing affinity maps), it raises an exception
    to prevent crashes in subsequent operations.

    Args:
        logger: Logger instance to use for logging captured errors
        context_info: Additional context string to include in log messages

    Raises:
        RuntimeError: If critical Vina errors are detected in stderr

    Example:
        with capture_stderr(self.logger, f"ligand {idx}"):
            v.compute_vina_maps(center=center, box_size=box_size)
    """
    # Save the original stderr file descriptor
    original_stderr_fd = os.dup(2)  # Duplicate stderr file descriptor (fd 2)

    # Create a pipe for capturing stderr
    read_fd, write_fd = os.pipe()

    try:
        # Redirect stderr to the write end of the pipe
        os.dup2(write_fd, 2)
        os.close(write_fd)  # Close the original write fd (we have it as fd 2 now)

        # Execute the code block
        yield

        # Restore original stderr (this closes the write end of the pipe, flushing all data)
        os.dup2(original_stderr_fd, 2)

        # Read the captured stderr content from the pipe
        # Set non-blocking mode to avoid hanging if there's no data
        os.set_blocking(read_fd, False)
        stderr_chunks = []
        try:
            while True:
                chunk = os.read(read_fd, 4096)
                if not chunk:
                    break
                stderr_chunks.append(chunk)
        except OSError:
            # No more data to read
            pass

        stderr_content = b"".join(stderr_chunks).decode("utf-8", errors="replace")

        if stderr_content:
            # Log all stderr content
            lines = [
                line.strip()
                for line in stderr_content.strip().split("\n")
                if line.strip()
            ]
            if lines:
                error_summary = "\n".join(lines)

                # Check for critical errors that indicate operation failure
                # Vina prefixes critical errors with "ERROR:"
                has_critical_error = any("ERROR:" in line for line in lines)

                if has_critical_error:
                    # Critical error detected - raise exception to prevent crashes
                    error_msg = f"Critical Vina error detected"
                    if context_info:
                        error_msg = f"{context_info}: {error_msg}"
                    logger.error(f"{error_msg}:\n{error_summary}")
                    raise RuntimeError(f"{error_msg}: {error_summary}")
                else:
                    # Non-critical warnings - just log them
                    if context_info:
                        logger.warning(
                            f"{context_info}: Vina stderr output:\n{error_summary}"
                        )
                    else:
                        logger.warning(f"Vina stderr output:\n{error_summary}")

    finally:
        # Always restore the original stderr and clean up
        os.dup2(original_stderr_fd, 2)
        os.close(original_stderr_fd)
        os.close(read_fd)


def split_models_from_pdbqt(input_file):
    """Yield each MODEL block from a multi-model PDBQT as a string."""
    with open(input_file, "r") as f:
        lines = f.readlines()

    current_model = []
    inside_model = False

    for line in lines:
        if line.startswith("MODEL"):
            inside_model = True
            current_model = []  # start new block
        elif line.startswith("ENDMDL"):
            inside_model = False
            if current_model:
                yield "".join(current_model)
            current_model = []
        elif inside_model:
            current_model.append(line)

    # In case there's a single-model file without MODEL/ENDMDL
    if not inside_model and current_model:
        yield "".join(current_model)


def get_ligand_com_from_pdbqt_string(pdbqt_string):
    """Compute COM from a PDBQT string (single MODEL)."""
    coords = []
    for line in pdbqt_string.splitlines():
        if line.startswith("HETATM") or line.startswith("ATOM"):
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append([x, y, z])
    if not coords:
        return None
    coords = np.array(coords)
    center = coords.mean(axis=0)
    return center.tolist()


def get_all_model_centers(input_file):
    """Return COM for each MODEL in a multi-model PDBQT."""
    centers = []
    for model_str in split_models_from_pdbqt(input_file):
        com = get_ligand_com_from_pdbqt_string(model_str)
        logger = get_logger()
        logger.debug(f"Model center: {com}")
        if com is not None:
            centers.append(com)
    return centers


def get_ligand_com_from_pdb(pdb_file, lig_resname):
    """
    Calculate center of mass (geometric center) for a ligand
    with residue name `lig_resname` from a PDB file.
    """
    coords = []
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith(("HETATM", "ATOM")):
                resname = line[17:20].strip()
                if resname == lig_resname:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
    if not coords:
        raise ValueError(f"No atoms found for residue '{lig_resname}' in {pdb_file}")
    coords = np.array(coords)
    return coords.mean(axis=0).tolist()


def shift_rdkitmol_to_vector(mol, target_vec):
    """
    Shift ligand coordinates in an RDKit Mol
    so that its COM moves to target_vec.

    Args:
        mol (rdkit.Chem.Mol): input ligand molecule with 3D conformer
        target_vec (list/tuple): [x, y, z] target position
    """
    conf = mol.GetConformer()
    coords = np.array(conf.GetPositions())  # shape (N_atoms, 3)

    com = coords.mean(axis=0)  # geometric center
    shift = np.array(target_vec) - com

    for i in range(mol.GetNumAtoms()):
        new_pos = coords[i] + shift
        conf.SetAtomPosition(i, new_pos.tolist())

    return mol


def flexible_box_size(mw):
    """
    Adjust box size heuristically based on MW.
    """
    if mw < 300:
        return [15, 15, 15]
    elif mw < 500:
        return [20, 20, 20]
    else:
        return [25, 25, 25]


def get_pdbqt_bounds(pdbqt_file):
    """
    Reads a PDBQT file and returns min/max for x, y, z coordinates.
    """
    x_vals, y_vals, z_vals = [], [], []

    with open(pdbqt_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    x_vals.append(x)
                    y_vals.append(y)
                    z_vals.append(z)
                except ValueError:
                    continue

    if not x_vals:
        raise ValueError("No atom coordinates found in PDBQT file.")

    bounds = {
        "x_min": min(x_vals),
        "x_max": max(x_vals),
        "y_min": min(y_vals),
        "y_max": max(y_vals),
        "z_min": min(z_vals),
        "z_max": max(z_vals),
        "x_size": max(x_vals) - min(x_vals),
        "y_size": max(y_vals) - min(y_vals),
        "z_size": max(z_vals) - min(z_vals),
    }
    return bounds


def generate_grid(boundary, box_size=15, margin=5):
    """
    Generate cube grid centers within expanded boundary.

    Parameters
    ----------
    boundary : dict
        Dictionary with x_min, x_max, y_min, y_max, z_min, z_max
    box_size : float
        Size of the cube along each axis
    margin : float
        Extra padding distance beyond min/max boundaries

    Returns
    -------
    list of tuple
        List of (x_center, y_center, z_center) for each cube
    """

    # Expand the boundary by margin
    x_min, x_max = boundary["x_min"] - margin, boundary["x_max"] + margin
    y_min, y_max = boundary["y_min"] - margin, boundary["y_max"] + margin
    z_min, z_max = boundary["z_min"] - margin, boundary["z_max"] + margin

    # Create grid ranges
    x_centers = np.arange(x_min + box_size / 2, x_max, box_size)
    y_centers = np.arange(y_min + box_size / 2, y_max, box_size)
    z_centers = np.arange(z_min + box_size / 2, z_max, box_size)

    grid_centers = []
    for x in x_centers:
        for y in y_centers:
            for z in z_centers:
                grid_centers.append([round(x, 3), round(y, 3), round(z, 3)])

    return grid_centers


def generate_conformer(mol):
    """Generate conformer with RDKit."""
    ps = AllChem.ETKDGv2()
    for repeat in range(50):
        rid = AllChem.EmbedMolecule(mol, ps)
        if rid == 0:
            break
    if rid == -1:
        logger = get_logger()
        logger.warning(
            "rdkit coords could not be generated without using random coords. using random coords now."
        )
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)

    AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=500)


def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    """Read molecule from file into RDKit Mol object."""
    if molecule_file.endswith(".mol2"):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith(".sdf"):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith(".pdbqt"):
        # PDBQT files contain AutoDock atom types that RDKit doesn't recognize
        # Need to convert atom types to standard elements before parsing
        autodock_to_element = {
            "A": "C",
            "C": "C",
            "CA": "C",  # Carbon types
            "N": "N",
            "NA": "N",
            "NS": "N",  # Nitrogen types
            "O": "O",
            "OA": "O",
            "OS": "O",  # Oxygen types
            "S": "S",
            "SA": "S",  # Sulfur types
            "P": "P",  # Phosphorus
            "F": "F",
            "Cl": "Cl",
            "Br": "Br",
            "I": "I",  # Halogens
            "H": "H",
            "HD": "H",
            "HS": "H",  # Hydrogen types
            "B": "B",
            "Si": "Si",
            "Se": "Se",
            "As": "As",  # Others
            "Zn": "Zn",
            "Fe": "Fe",
            "Mg": "Mg",
            "Ca": "Ca",
            "Mn": "Mn",
            "Cu": "Cu",
        }

        pdb_lines = []
        with open(molecule_file) as file:
            for line in file:
                if line.startswith(("ATOM", "HETATM")):
                    # PDB format: element symbol is in columns 77-78 (1-indexed) = columns 76-77 (0-indexed)
                    # Python slice [76:78] gives characters at indices 76 and 77 (columns 77-78 in PDB)
                    element_col = line[76:78].strip() if len(line) > 77 else ""

                    # If element column has AutoDock type, use it; otherwise extract from atom name
                    if element_col in autodock_to_element:
                        element = autodock_to_element[element_col]
                    else:
                        # Extract atom type from atom name (columns 12-16)
                        atom_name = line[12:16].strip() if len(line) > 15 else ""
                        atom_type = ""
                        for char in atom_name:
                            if char.isalpha():
                                atom_type += char
                            else:
                                break

                        # Map to element symbol
                        if atom_type in autodock_to_element:
                            element = autodock_to_element[atom_type]
                        elif atom_type:
                            element = atom_type[0] if atom_type[0].isalpha() else "C"
                        elif element_col and element_col[0].isalpha():
                            element = element_col[0]
                        else:
                            element = "C"  # Default to carbon

                    # Ensure element is valid
                    if element not in autodock_to_element.values():
                        element = "C"  # Default to carbon

                    # Replace element in line (columns 77-78 in PDB format, 76-77 in 0-indexed)
                    if len(line) > 77:
                        # Replace element column: line[:76] + element (right-justified in 2 chars) + rest
                        pdb_line = line[:76] + f"{element:>2}" + line[78:]
                    else:
                        # Pad line if needed
                        pdb_line = line.rstrip().ljust(77) + f"{element:>2}\n"
                    pdb_lines.append(
                        pdb_line[:66] + "\n"
                    )  # Keep only first 66 chars (standard PDB)
                elif line.startswith(("END", "TER")):
                    pdb_lines.append(line[:66] + "\n")
                elif line.strip() and not line.startswith("REMARK"):
                    # Keep other lines but truncate to 66 chars
                    pdb_lines.append(line[:66] + "\n")

        pdb_block = "".join(pdb_lines)
        try:
            mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
        except Exception:
            # If RDKit still fails, return None
            logger = get_logger()
            logger.warning(
                f"RDKit unable to parse PDBQT file {molecule_file}, returning None"
            )
            return None
    elif molecule_file.endswith(".pdb"):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        raise ValueError(
            f"Expect format .mol2, .sdf, .pdbqt or .pdb, got {molecule_file}"
        )

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                import warnings

                warnings.warn("Unable to compute charges for the molecule.")

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)

    except Exception as e:
        logger = get_logger()
        logger.error(f"RDKit error: {e}")
        logger.error("RDKit was unable to read the molecule.")
        return None

    return mol


def tempstring(v):
    """
    Capture a Vina pose as a string without leaving files on disk.
    """
    # Get a unique temporary path, but do not keep the file open
    fd, tmp_name = tempfile.mkstemp(suffix=".pdbqt")
    os.close(fd)  # close the file descriptor immediately
    os.remove(tmp_name)  # remove the file so Vina can write it
    # Write pose to the temporary file
    v.write_pose(tmp_name)

    # Read pose back into memory
    with open(tmp_name, "r") as f:
        pose_str = f.read()

    # Clean up
    os.remove(tmp_name)

    return pose_str


def write_all_centers_pdb(
    all_centers: pd.DataFrame, output_pdb: str = "all_centers.pdb"
):
    """
    Write a PDB file from a dataframe of centers.

    DataFrame must have columns:
    ['description', 'grid', <center_0>, <center_1>, ...]
    Each center column contains a list [x, y, z].
    Index is used as atom serial number.
    """
    with open(output_pdb, "w") as f:
        atom_serial = 1
        for idx, row in all_centers.iterrows():
            atom_name = "CA"
            res_name = f"C{int(row['grid_id'])}"
            chain_id = "A"
            res_seq = idx + 1
            x, y, z = row["COM_x"], row["COM_y"], row["COM_z"]
            occupancy = 1.0
            if "best_affinity" in row:
                temp_factor = row["best_affinity"]
            elif "affinity" in row:
                temp_factor = row["affinity"]
            else:
                temp_factor = 0.0
            element = "C"

            if len(res_name) <= 3:
                f.write(
                    f"ATOM  {atom_serial:5d} {atom_name:^4s} {res_name:>3s} {chain_id}{res_seq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          {element:>2s}\n"
                )
            else:
                f.write(
                    f"ATOM  {atom_serial:5d} {atom_name:^4s} {res_name:>4s}{chain_id}{res_seq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          {element:>2s}\n"
                )

            atom_serial += 1  # increment for each atom

        f.write("END\n")


def write_cluser_centers_pdb(
    cluster_summary: pd.DataFrame, output_pdb: str = "dock_sites_clustered.pdb"
):
    with open(output_pdb, "w") as f:
        atom_serial = 1
        for idx, row in cluster_summary.iterrows():
            atom_name, res_name, chain_id = "CA", "COM", "A"
            res_seq = int(row["cluster_id"]) + 1
            x, y, z = row["centroid_x"], row["centroid_y"], row["centroid_z"]
            occupancy, temp_factor = 1, row["mean_affinity"]
            element = "C"

            if len(res_name) <= 3:
                f.write(
                    f"ATOM  {atom_serial:5d} {atom_name:^4s} {res_name:>3s} {chain_id}{res_seq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          {element:>2s}\n"
                )
            else:
                f.write(
                    f"ATOM  {atom_serial:5d} {atom_name:^4s} {res_name:>4s}{chain_id}{res_seq:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{occupancy:6.2f}{temp_factor:6.2f}          {element:>2s}\n"
                )

            atom_serial += 1  # increment for each atom

        f.write("END\n")


def get_molweight_from_pdbqt(pdbqt_file):
    """
    Calculate molecular weight from a PDBQT file.

    Args:
        pdbqt_file: str: Path to PDBQT file

    Returns:
        float: Molecular weight in Daltons
    """
    from rdkit.Chem import Descriptors

    # Try reading with RDKit first (only works for standard PDB files, not PDBQT)
    # PDBQT files contain AutoDock atom types that RDKit doesn't recognize
    mol = None
    try:
        mol = Chem.MolFromPDBFile(pdbqt_file, removeHs=False)
        if mol is not None:
            return Descriptors.MolWt(mol)
    except Exception:
        # RDKit will fail on PDBQT files due to AutoDock atom types (e.g., 'A' for aromatic carbon)
        # Fall through to manual parsing
        pass

    # Fallback: manually parse PDBQT file and map AutoDock atom types to elements
    # AutoDock atom type mapping (common types)
    autodock_to_element = {
        # Carbon types
        "A": "C",  # Aromatic carbon
        "C": "C",  # Aliphatic carbon
        "CA": "C",  # Aromatic carbon (alternative)
        # Nitrogen types
        "N": "N",  # Nitrogen
        "NA": "N",  # Aromatic nitrogen
        "NS": "N",  # Nitrogen in sulfonamide
        # Oxygen types
        "O": "O",  # Oxygen
        "OA": "O",  # Acceptor oxygen
        "OS": "O",  # Ester/ether oxygen
        # Sulfur types
        "S": "S",  # Sulfur
        "SA": "S",  # Aromatic sulfur
        # Phosphorus
        "P": "P",  # Phosphorus
        # Halogens
        "F": "F",  # Fluorine
        "Cl": "Cl",  # Chlorine
        "Br": "Br",  # Bromine
        "I": "I",  # Iodine
        # Others
        "H": "H",  # Hydrogen
        "HD": "H",  # Donor hydrogen
        "HS": "H",  # Hydrogen bonded to sulfur
        "B": "B",  # Boron
        "Si": "Si",  # Silicon
        "Se": "Se",  # Selenium
        "As": "As",  # Arsenic
        "Zn": "Zn",  # Zinc
        "Fe": "Fe",  # Iron
        "Mg": "Mg",  # Magnesium
        "Ca": "Ca",  # Calcium
        "Mn": "Mn",  # Manganese
        "Cu": "Cu",  # Copper
    }

    atomic_weights = {
        "H": 1.008,
        "C": 12.011,
        "N": 14.007,
        "O": 15.999,
        "F": 18.998,
        "P": 30.974,
        "S": 32.065,
        "Cl": 35.45,
        "Br": 79.904,
        "I": 126.904,
        "B": 10.81,
        "Si": 28.085,
        "Se": 78.971,
        "As": 74.922,
        "Zn": 65.38,
        "Fe": 55.845,
        "Mg": 24.305,
        "Ca": 40.078,
        "Mn": 54.938,
        "Cu": 63.546,
    }

    total_weight = 0.0
    with open(pdbqt_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                # In PDBQT files, the atom type is typically in the atom name (columns 12-16)
                # or can be extracted from the element column if present
                # PDB format: element symbol is in columns 77-78 (1-indexed) = columns 76-77 (0-indexed)
                # Python slice [76:78] gives characters at indices 76 and 77 (columns 77-78 in PDB)
                element = line[76:78].strip() if len(line) > 77 else ""

                # If element column is empty or contains AutoDock type, parse from atom name
                if not element or element in autodock_to_element:
                    atom_name = line[12:16].strip() if len(line) > 15 else ""
                    # Extract atom type from atom name (usually first 1-2 characters)
                    # PDBQT atom names often start with the atom type
                    atom_type = ""
                    for char in atom_name:
                        if char.isalpha():
                            atom_type += char
                        else:
                            break

                    # Map AutoDock atom type to element
                    if atom_type and atom_type in autodock_to_element:
                        element = autodock_to_element[atom_type]
                    elif atom_type:
                        # Try first character as element
                        element = (
                            atom_type[0] if atom_type[0] in atomic_weights else "C"
                        )
                    elif element in autodock_to_element:
                        element = autodock_to_element[element]
                    else:
                        # Default to carbon if we can't determine
                        element = "C"
                elif element not in atomic_weights:
                    # If element is not recognized, try mapping it
                    element = autodock_to_element.get(
                        element, element[0] if element else "C"
                    )
                    if element not in atomic_weights:
                        element = "C"  # Default to carbon

                # Get weight
                weight = atomic_weights.get(element, 12.011)  # Default to carbon weight
                total_weight += weight

    return total_weight


def convert_receptor_to_pdbqt(receptor_path):
    """
    Convert receptor PDB file to PDBQT format using Open Babel command-line tool.

    This function uses the obabel command-line tool via subprocess to avoid
    Python binding ABI compatibility issues.

    The PDBQT file is stored in the same directory as the input PDB file,
    ensuring it's co-located with the source file and independent of the
    current working directory.

    Args:
        receptor_path: str: Path to the receptor PDB file

    Returns:
        str: Path to the converted PDBQT file
    """
    # Get the directory of the input PDB file
    receptor_dir = os.path.dirname(os.path.abspath(receptor_path))

    # Extract filename without extension and create PDBQT path in the same directory
    protprefix = receptor_path.split(".pdb")[0]
    protstem = os.path.basename(protprefix)
    receptor_pdbqt = os.path.join(receptor_dir, f"{protstem}.pdbqt")

    # Check if PDBQT file already exists
    if os.path.exists(receptor_pdbqt):
        logger = get_logger()
        logger.info(f"Receptor PDBQT file already exists: {receptor_pdbqt}")
        return receptor_pdbqt

    # Convert PDB to PDBQT using Open Babel
    logger = get_logger()
    logger.info(f"Converting receptor {receptor_path} to {receptor_pdbqt}")

    # Try to find obabel in common locations (conda environment, system PATH)
    # Check if we're in a conda environment
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        obabel_cmd = os.path.join(conda_prefix, "bin", "obabel")
    else:
        # Fall back to system PATH
        obabel_cmd = "obabel"

    try:
        cmd = [
            obabel_cmd,
            receptor_path,
            "-O",
            receptor_pdbqt,
            "-p",
            "7.4",
            "--partialcharge",
            "eem",
            "-xr",
            "-xp",
            "-xn",
        ]

        # Use run_cmd to redirect output to logger
        run_cmd(cmd, check=True)
        logger.info(f"Successfully converted receptor to PDBQT format")
        return receptor_pdbqt

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to convert receptor to PDBQT")
        raise RuntimeError(f"Failed to convert receptor to PDBQT: {e}")
    except FileNotFoundError:
        raise RuntimeError(
            "Open Babel (obabel) not found. Please install Open Babel to convert PDB files to PDBQT format."
        )
