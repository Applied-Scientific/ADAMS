"""
GROMACS index and membrane group helpers.
"""

import os
import re

from ....logger_utils import get_logger
from ....utils import run_cmd
from .constants import (
    LIGAND_RESNAME,
    LIPID_RESNAMES,
    NDX_GROUP_PROTEIN_LIG,
    NDX_GROUP_WATER_IONS,
    REQUIRED_NDX_GROUPS_FOR_TC,
    WATER_ION_RESNAMES,
)
from .ion_solvent import (
    get_water_ion_resnames_from_topology,
    parse_topology_molecule_counts,
)


def resolve_ligand_resname_for_pose(top_path: str, preferred: str = None) -> str:
    """
    Resolve ligand residue/molecule name from system.top.

    Prefers the caller-provided name when it exists in [ molecules ].
    Otherwise picks the first non-protein, non-water/ion molecule type.
    """
    preferred = (preferred or LIGAND_RESNAME).strip()
    counts = parse_topology_molecule_counts(top_path)
    water_ions = {name.upper() for name in get_water_ion_resnames_from_topology(top_path)}

    if preferred in counts:
        return preferred

    candidates = []
    for name, count in counts.items():
        upper = name.upper()
        if count <= 0:
            continue
        if upper in water_ions:
            continue
        if upper.startswith("PROTEIN"):
            continue
        candidates.append(name)

    if not candidates:
        return preferred
    return candidates[0]


def detect_ligand_resname_from_gro(ligand_gro_path: str) -> str:
    """
    Detect ligand residue name from a ligand-only GRO file.
    """
    if not os.path.exists(ligand_gro_path):
        raise FileNotFoundError(f"Ligand GRO file not found: {ligand_gro_path}")
    with open(ligand_gro_path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()
    # GRO: line 1 title, line 2 atom count, atom records start at line 3.
    for line in lines[2:-1]:
        if len(line) >= 10:
            resname = line[5:10].strip()
            if resname:
                return resname
    raise ValueError(f"Could not detect residue name from ligand GRO: {ligand_gro_path}")


def read_gro_atom_count(gro_path: str) -> int:
    """Read the atom count from the second line of a GRO file."""
    if not os.path.exists(gro_path):
        raise FileNotFoundError(f"GRO file not found: {gro_path}")
    with open(gro_path, "r", encoding="utf-8", errors="ignore") as fh:
        fh.readline()
        count_line = fh.readline().strip()
    try:
        return int(count_line)
    except ValueError as exc:
        raise ValueError(
            f"Invalid GRO atom count in {gro_path}: {count_line!r}"
        ) from exc


def write_atom_index_group(
    ndx_path: str,
    group_name: str,
    atom_indices,
    atoms_per_line: int = 15,
) -> None:
    """Write one named GROMACS index group from explicit 1-based atom indices."""
    indices = [int(idx) for idx in atom_indices]
    if not indices:
        raise ValueError(f"Cannot write empty index group '{group_name}' to {ndx_path}")
    with open(ndx_path, "w", encoding="utf-8") as fh:
        fh.write(f"[ {group_name} ]\n")
        for start in range(0, len(indices), atoms_per_line):
            chunk = indices[start:start + atoms_per_line]
            fh.write(" ".join(str(idx) for idx in chunk) + "\n")


def write_bulk_solvent_index(
    newbox_gro_path: str,
    solv_gro_path: str,
    ndx_path: str,
    group_name: str = "BulkSolvent",
) -> tuple[int, int]:
    """
    Write an index group covering only solvent atoms appended by ``gmx solvate``.

    Returns the inclusive 1-based atom-index range written to the group.
    """
    pre_solvate_atoms = read_gro_atom_count(newbox_gro_path)
    total_atoms = read_gro_atom_count(solv_gro_path)
    if total_atoms <= pre_solvate_atoms:
        raise ValueError(
            "No appended solvent atoms detected for bulk-solvent genion selection "
            f"({newbox_gro_path} atoms={pre_solvate_atoms}, {solv_gro_path} atoms={total_atoms})."
        )
    start_atom = pre_solvate_atoms + 1
    write_atom_index_group(
        ndx_path,
        group_name,
        range(start_atom, total_atoms + 1),
    )
    return start_atom, total_atoms


def make_system_index(
    gmx_binary,
    gro_path,
    ndx_path,
    ligand_resname=None,
    top_path=None,
    ligand_gro_path=None,
):
    """
    Create a system index file with explicitly named groups for MDP compatibility.

    If top_path is provided, water/ion residue names for the Water_and_ions group
    are taken from the topology [ molecules ] section so they match the system.
    """
    if ligand_resname is None:
        ligand_resname = LIGAND_RESNAME

    logger = get_logger()
    if ligand_gro_path and os.path.exists(ligand_gro_path):
        gro_ligand = detect_ligand_resname_from_gro(ligand_gro_path)
        if gro_ligand != ligand_resname:
            logger.warning(
                "Using ligand residue '%s' from GRO (requested '%s').",
                gro_ligand,
                ligand_resname,
            )
        ligand_resname = gro_ligand
    elif top_path and os.path.exists(top_path):
        resolved_ligand = resolve_ligand_resname_for_pose(top_path, ligand_resname)
        if resolved_ligand != ligand_resname:
            logger.warning(
                "Ligand resname '%s' not found in topology; using detected '%s'.",
                ligand_resname,
                resolved_ligand,
            )
        ligand_resname = resolved_ligand

    if top_path and os.path.exists(top_path):
        water_ion_names = get_water_ion_resnames_from_topology(top_path)
        if not water_ion_names:
            raise ValueError(
                f"No water/ion molecule types found in topology {top_path}. "
                "Check [ molecules ] section and force field residue naming."
            )
    if os.path.exists(ndx_path):
        os.remove(ndx_path)
    # LIG group used for Protein_LIG index and for POSRES ligand restraints in min/nvt/npt (not freezing)
    _append_named_group(
        gmx_binary,
        gro_path,
        ndx_path,
        selection=f"r {ligand_resname}",
        group_name="LIG",
    )

    # Build Protein_LIG using stable numeric groups from current index.
    protein_idx = get_ndx_group_index(ndx_path, "Protein")
    lig_idx = get_ndx_group_index(ndx_path, "LIG")
    _append_named_group(
        gmx_binary,
        gro_path,
        ndx_path,
        selection=f"{protein_idx} | {lig_idx}",
        group_name=NDX_GROUP_PROTEIN_LIG,
    )

    # Build Water_and_ions from default Water and Ion groups when available.
    water_idx = get_ndx_group_index(ndx_path, "Water")
    try:
        ion_idx = get_ndx_group_index(ndx_path, "Ion")
        water_ion_sel = f"{water_idx} | {ion_idx}"
    except ValueError:
        water_ion_sel = f"{water_idx}"
    _append_named_group(
        gmx_binary,
        gro_path,
        ndx_path,
        selection=water_ion_sel,
        group_name=NDX_GROUP_WATER_IONS,
    )

    _assert_group_non_empty(ndx_path, "LIG")
    _assert_group_non_empty(ndx_path, NDX_GROUP_PROTEIN_LIG)
    _assert_group_non_empty(ndx_path, NDX_GROUP_WATER_IONS)
    logger.debug(f"Written index with named groups to {ndx_path}")


def rename_ndx_last_group(ndx_path, new_name):
    """Rename the last group in a GROMACS index file."""
    with open(ndx_path, "r") as f:
        content = f.read()
    pattern = re.compile(r"\[\s*[^\]]+\s*\]")
    headers = list(pattern.finditer(content))
    if not headers:
        raise ValueError(f"No groups found in index file {ndx_path}")
    last = headers[-1]
    new_content = content[: last.start()] + f"[ {new_name} ]" + content[last.end() :]
    with open(ndx_path, "w") as f:
        f.write(new_content)


def get_ndx_group_index(ndx_path, group_name=None):
    """Return 0-based group index from a GROMACS index file."""
    with open(ndx_path, "r") as f:
        content = f.read()
    pattern = re.compile(r"\[\s*([^\]]+)\s*\]")
    names = [m.group(1).strip() for m in pattern.finditer(content)]
    if not names:
        raise ValueError(f"No groups found in index file {ndx_path}")
    if group_name is None:
        return len(names) - 1
    try:
        return names.index(group_name.strip())
    except ValueError:
        raise ValueError(
            f"Group '{group_name}' not found in {ndx_path}. Available: {names}"
        ) from None


def get_ndx_group_names(ndx_path: str) -> list:
    """Return list of group names in a GROMACS index file."""
    with open(ndx_path, "r") as f:
        content = f.read()
    pattern = re.compile(r"\[\s*([^\]]+)\s*\]")
    return [m.group(1).strip() for m in pattern.finditer(content)]


def validate_system_index_groups(
    ndx_path: str,
    required: tuple = None,
) -> None:
    """
    Ensure the index file contains all groups required for MDP tc-grps (thermostat coupling).

    Raises ValueError if the index is missing any required group, so that grompp fails
    fast with a clear message instead of wrong or failed coupling.
    """
    if required is None:
        required = REQUIRED_NDX_GROUPS_FOR_TC
    if not os.path.exists(ndx_path):
        raise FileNotFoundError(
            f"Index file not found: {ndx_path}. "
            "Run make_system_index (e.g. during run_md_prepare or run_md_simulation index rebuild) to create "
            f"groups {list(required)} required for NVT/NPT/MD tc-grps."
        )
    names = get_ndx_group_names(ndx_path)
    missing = [g for g in required if g not in names]
    if missing:
        raise ValueError(
            f"Index {ndx_path} is missing thermostat coupling group(s): {missing}. "
            f"MDP tc-grps require: {list(required)}. "
            f"Available groups: {names}. "
            "Rebuild the index with make_system_index (e.g. during run_md_prepare or run_md_simulation)."
        )
    for group_name in required:
        _assert_group_non_empty(ndx_path, group_name)


def _append_named_group(gmx_binary, gro_path, ndx_path, selection, group_name):
    """Append one group to an index file and rename it immediately."""
    make_ndx_cmd = [gmx_binary, "make_ndx", "-f", gro_path]
    if os.path.exists(ndx_path):
        make_ndx_cmd.extend(["-n", ndx_path])
    make_ndx_cmd.extend(["-o", ndx_path])
    run_cmd(make_ndx_cmd, input_str=f"{selection}\nq\n", check=True)
    rename_ndx_last_group(ndx_path, group_name)
    # Raises if the rename did not yield a resolvable named group.
    get_ndx_group_index(ndx_path, group_name)


def _count_group_atoms(ndx_path, group_name):
    """Count atoms listed under a named index group."""
    in_group = False
    atom_count = 0
    with open(ndx_path, "r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                name = line.strip("[]").strip()
                in_group = name == group_name
                continue
            if in_group:
                atom_count += len(line.split())
    return atom_count


def _assert_group_non_empty(ndx_path, group_name):
    """Raise when an expected group exists but has no atoms."""
    atom_count = _count_group_atoms(ndx_path, group_name)
    if atom_count <= 0:
        raise ValueError(
            f"Index group '{group_name}' in {ndx_path} is empty. "
            "Check residue naming (protein/ligand/solvent/ions) in the GRO/topology."
        )


def detect_lipid_resnames_in_gro(gro_path):
    """
    Detect lipid residue names present in a GRO file.

    Returns:
        set: Set of lipid residue names found.
    """
    found = set()
    with open(gro_path, "r") as fh:
        lines = fh.readlines()
    # Skip title (line 0) and atom count (line 1), stop before box vector (last line)
    for line in lines[2:-1]:
        if len(line) >= 10:
            resname = line[5:10].strip()
            if resname.upper() in LIPID_RESNAMES:
                found.add(resname)
    return found


def detect_lipid_resnames_in_top(top_path):
    """
    Detect lipid residue names from the [ molecules ] section of a topology.

    Returns:
        set: Set of lipid residue names found.
    """
    from .ion_solvent import parse_topology_molecule_counts

    counts = parse_topology_molecule_counts(top_path)
    return {name for name in counts if name.upper() in LIPID_RESNAMES}


def make_membrane_index(gmx_binary, gro_path, ndx_path, lipid_resnames=None):
    """
    Create a GROMACS index file for membrane systems with named groups:
        - Protein
        - Membrane (all detected lipid residues)
        - Solvent_and_ions (SOL + ions)
    """
    logger = get_logger()

    if lipid_resnames is None:
        lipid_resnames = detect_lipid_resnames_in_gro(gro_path)
    if not lipid_resnames:
        raise ValueError(
            "No lipid residues detected in the GRO file. Cannot create membrane index. "
            "Provide lipid_resnames explicitly or verify the system contains lipids."
        )

    logger.info(f"Detected lipid residues for membrane index: {sorted(lipid_resnames)}")

    # Build make_ndx selections: lipids into one group, then solvent+ions.
    lipid_sel = " | ".join(f"r {r}" for r in sorted(lipid_resnames))
    water_ion_sel = " | ".join(f"r {r}" for r in WATER_ION_RESNAMES)
    if os.path.exists(ndx_path):
        os.remove(ndx_path)
    _append_named_group(
        gmx_binary,
        gro_path,
        ndx_path,
        selection=lipid_sel,
        group_name="Membrane",
    )
    _append_named_group(
        gmx_binary,
        gro_path,
        ndx_path,
        selection=water_ion_sel,
        group_name="Solvent_and_ions",
    )
    _assert_group_non_empty(ndx_path, "Membrane")
    _assert_group_non_empty(ndx_path, "Solvent_and_ions")

    logger.info(f"Written membrane index with named groups to {ndx_path}")
