"""
data_preprocessing/protonation.py

Protonation module using PDB2PQR with PROPKA for pKa-aware protonation state assignment.
"""

import shutil
import subprocess
from collections import defaultdict
from math import inf
from typing import Dict, List, Optional, Tuple

from ...logger_utils import get_logger
from ...utils import run_cmd

STANDARD_AA_RESNAMES = {
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLN",
    "GLU",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
    "ASH",
    "GLH",
    "HID",
    "HIE",
    "HIP",
    "LYN",
    "CYX",
    "MSE",
}

TWO_LETTER_ELEMENTS = {"CL", "BR", "MG", "ZN", "FE", "CA", "MN", "CU", "NA", "SE"}
SINGLE_LETTER_ELEMENTS = {"H", "B", "C", "N", "O", "F", "P", "S", "K", "I"}


def _normalize_element_symbol(raw: str) -> str:
    """Normalize a raw element token to a known 1- or 2-letter symbol."""
    token = "".join(ch for ch in (raw or "").strip() if ch.isalpha())
    if not token:
        return ""
    if len(token) >= 2:
        two = token[:2].upper()
        if two in TWO_LETTER_ELEMENTS:
            return two
    one = token[0].upper()
    if one in SINGLE_LETTER_ELEMENTS:
        return one
    return ""


def _element_from_pdb_atom_name(
    atom_name: str,
    resname: str = "",
    record: str = "",
    element_hint: str = "",
) -> str:
    """
    Infer element symbol for PDB columns 77-78.

    Key rule:
    - Protein backbone atom 'CA' (alpha carbon) must be Carbon, not Calcium.
    """
    atom = (atom_name or "").strip().upper()
    res = (resname or "").strip().upper()
    rec = (record or "").strip().upper()

    # Critical disambiguation: amino-acid alpha carbon.
    if rec == "ATOM" and res in STANDARD_AA_RESNAMES and atom == "CA":
        return " C"

    # Prefer explicit element columns if present.
    hint = _normalize_element_symbol(element_hint)
    if hint:
        return hint.rjust(2)

    # Fall back to atom-name inference.
    inferred = _normalize_element_symbol(atom)
    if inferred:
        return inferred.rjust(2)

    return " C"


def pqr_to_pdb(pqr_path: str, pdb_path: str):
    """Convert PQR to PDB by replacing charge/radius with occupancy/B-factor.
    Ensures PDB columns 77-78 (element symbol) are set so Open Babel and other
    tools do not warn about missing element."""
    with open(pqr_path, "r") as f, open(pdb_path, "w") as out:
        for line in f:
            if line.startswith(("ATOM  ", "HETATM")):
                if len(line) >= 66:
                    base = line[:54] + "  1.00  0.00"
                else:
                    base = line.rstrip().ljust(54) + "  1.00  0.00"
                # PDB columns 77-78: element symbol (required by Open Babel)
                record = line[0:6].strip()
                atom_name = line[12:16] if len(line) > 15 else ""
                resname = line[17:20] if len(line) > 19 else ""
                element_hint = line[76:78] if len(line) >= 78 else ""
                element = _element_from_pdb_atom_name(
                    atom_name=atom_name,
                    resname=resname,
                    record=record,
                    element_hint=element_hint,
                )
                # Ensure line has at least 78 characters, with element at 77-78
                if len(base) >= 78:
                    out.write(base[:76] + element + base[78:].rstrip() + "\n")
                else:
                    out.write(base.ljust(76) + element + "\n")
            else:
                out.write(line)


def _format_ter_line_from_atom(atom_line: str) -> str:
    """Format a full TER record from the preceding ATOM/HETATM line."""
    serial_text = atom_line[6:11].strip() if len(atom_line) >= 11 else ""
    try:
        serial = int(serial_text) + 1
    except ValueError:
        serial = 1

    resname = atom_line[17:20].strip() if len(atom_line) >= 20 else "UNK"
    if not resname:
        resname = "UNK"

    chain = atom_line[21] if len(atom_line) > 21 else " "
    resseq_text = atom_line[22:26].strip() if len(atom_line) >= 26 else ""
    try:
        resseq = f"{int(resseq_text):4d}"
    except ValueError:
        resseq = "   1"

    icode = atom_line[26] if len(atom_line) > 26 else " "
    return f"TER   {serial:5d}      {resname:>3} {chain}{resseq}{icode}\n"


def _normalize_ter_records(pdb_path: str) -> int:
    """
    Ensure TER records contain serial/resname/chain/resseq fields.

    Returns:
        int: number of TER lines rewritten.
    """
    with open(pdb_path, "r") as f:
        lines = f.readlines()

    rewritten = 0
    output_lines: List[str] = []
    last_atom_line: Optional[str] = None
    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            output_lines.append(line)
            last_atom_line = line
            continue

        if line.startswith("TER"):
            if last_atom_line is None:
                output_lines.append(line if line.endswith("\n") else line + "\n")
                continue
            formatted = _format_ter_line_from_atom(last_atom_line)
            if line.rstrip("\n") != formatted.rstrip("\n"):
                rewritten += 1
            output_lines.append(formatted)
            last_atom_line = None
            continue

        output_lines.append(line)

    if rewritten:
        with open(pdb_path, "w") as f:
            f.writelines(output_lines)

    return rewritten


def _pdb_atom_identity_key(line: str) -> Tuple[str, str, str, str, str, str, str, str, str]:
    """
    Build a stable identity key for ATOM/HETATM records independent of serial number.
    """
    return (
        line[12:16],  # atom name
        line[16:17],  # alt loc
        line[17:20],  # residue name
        line[21:22],  # chain ID
        line[22:26],  # residue sequence
        line[26:27],  # insertion code
        line[30:38],  # x
        line[38:46],  # y
        line[46:54],  # z
    )


def _pdb_atom_identity_key_no_chain(
    line: str,
) -> Tuple[str, str, str, str, str, str, str, str]:
    """
    Build an identity key that ignores chain ID.
    """
    return (
        line[12:16],  # atom name
        line[16:17],  # alt loc
        line[17:20],  # residue name
        line[22:26],  # residue sequence
        line[26:27],  # insertion code
        line[30:38],  # x
        line[38:46],  # y
        line[46:54],  # z
    )


def _pdb_residue_key_no_chain(line: str) -> Tuple[str, str, str, str]:
    """Build a residue-level key that ignores chain ID."""
    return (
        line[0:6],  # record type (ATOM/HETATM)
        line[17:20],  # residue name
        line[22:26],  # residue sequence
        line[26:27],  # insertion code
    )


def _pdb_residue_key_no_chain_no_name(line: str) -> Tuple[str, str, str]:
    """Build a residue-level key that ignores chain ID and residue name."""
    return (
        line[0:6],  # record type (ATOM/HETATM)
        line[22:26],  # residue sequence
        line[26:27],  # insertion code
    )


def _pdb_set_chain_id(line: str, chain_id: str) -> str:
    """Return line with chain ID replaced at PDB column 22."""
    base = line.rstrip("\n")
    if len(base) < 22:
        base = base.ljust(22)
    chain = (chain_id or " ")[:1]
    return f"{base[:21]}{chain}{base[22:]}\n"


def _pdb_xyz(line: str) -> Optional[Tuple[float, float, float]]:
    """Parse XYZ coordinates from a PDB ATOM/HETATM line."""
    try:
        return (float(line[30:38]), float(line[38:46]), float(line[46:54]))
    except ValueError:
        return None


def _residue_centroid(coords: List[Tuple[float, float, float]]) -> Optional[Tuple[float, float, float]]:
    if not coords:
        return None
    n = float(len(coords))
    sx = sum(c[0] for c in coords)
    sy = sum(c[1] for c in coords)
    sz = sum(c[2] for c in coords)
    return (sx / n, sy / n, sz / n)


def _restore_chain_ids_from_reference(reference_pdb: str, target_pdb: str) -> int:
    """
    Restore missing chain IDs in target_pdb from reference_pdb.

    Strategy:
    1) Exact atom-level match ignoring chain ID.
    2) Residue-level fallback using centroid/atom-name similarity with a
       chainless residue key.
    3) If residue naming changed during protonation (e.g., HIS->HID/HIE/HIP),
       retry residue-level matching with residue-name-agnostic keys.

    Returns:
        Number of ATOM/HETATM records where chain ID was restored.
    """

    def _pick_best_candidate(
        candidates: List[dict],
        available_indices: List[int],
        target_atom_names: set,
        target_centroid: Optional[Tuple[float, float, float]],
    ) -> int:
        best_idx = available_indices[0]
        best_overlap = -1
        best_dist2 = inf
        for idx in available_indices:
            candidate = candidates[idx]
            overlap = len(target_atom_names.intersection(candidate.get("atom_names", set())))
            centroid = candidate.get("centroid")
            if target_centroid is not None and centroid is not None:
                dx = target_centroid[0] - centroid[0]
                dy = target_centroid[1] - centroid[1]
                dz = target_centroid[2] - centroid[2]
                dist2 = dx * dx + dy * dy + dz * dz
            else:
                dist2 = inf

            if (
                overlap > best_overlap
                or (overlap == best_overlap and dist2 < best_dist2)
                or (overlap == best_overlap and dist2 == best_dist2 and idx < best_idx)
            ):
                best_idx = idx
                best_overlap = overlap
                best_dist2 = dist2
        return best_idx

    ref_atom_chain_by_key: Dict[Tuple[str, str, str, str, str, str, str, str], str] = {}
    ref_residue_data_named: Dict[Tuple[str, str, str, str], List[dict]] = defaultdict(list)
    ref_residue_data_noname: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)

    residue_entries_by_full_key: Dict[Tuple[str, str, str, str, str], dict] = {}
    residue_seen_order: List[Tuple[str, str, str, str, str]] = []

    with open(reference_pdb, "r") as f:
        for line in f:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue

            chain = line[21:22] if len(line) > 21 else " "
            atom_key = _pdb_atom_identity_key_no_chain(line)
            ref_atom_chain_by_key.setdefault(atom_key, chain)

            named_key = _pdb_residue_key_no_chain(line)
            noname_key = _pdb_residue_key_no_chain_no_name(line)
            full_key = (*named_key, chain)

            if full_key not in residue_entries_by_full_key:
                residue_entries_by_full_key[full_key] = {
                    "named_key": named_key,
                    "noname_key": noname_key,
                    "chain": chain,
                    "coords": [],
                    "atom_names": set(),
                }
                residue_seen_order.append(full_key)

            entry = residue_entries_by_full_key[full_key]
            xyz = _pdb_xyz(line)
            if xyz is not None:
                entry["coords"].append(xyz)
            atom_name = line[12:16].strip()
            if atom_name:
                entry["atom_names"].add(atom_name)

    if not ref_atom_chain_by_key:
        return 0

    for full_key in residue_seen_order:
        entry = residue_entries_by_full_key[full_key]
        record = {
            "chain": entry["chain"],
            "centroid": _residue_centroid(entry["coords"]),
            "atom_names": entry["atom_names"],
        }
        ref_residue_data_named[entry["named_key"]].append(record)
        ref_residue_data_noname[entry["noname_key"]].append(record)

    with open(target_pdb, "r") as f:
        lines = f.readlines()

    target_residues = []
    current = None
    for idx, line in enumerate(lines):
        if line.startswith(("ATOM  ", "HETATM")) and (line[21:22].strip() == ""):
            named_key = _pdb_residue_key_no_chain(line)
            noname_key = _pdb_residue_key_no_chain_no_name(line)
            if current is None or named_key != current["named_key"]:
                if current is not None:
                    current["centroid"] = _residue_centroid(current["coords"])
                    target_residues.append(current)
                current = {
                    "named_key": named_key,
                    "noname_key": noname_key,
                    "line_indices": [],
                    "coords": [],
                    "atom_names": set(),
                }
            current["line_indices"].append(idx)
            xyz = _pdb_xyz(line)
            if xyz is not None:
                current["coords"].append(xyz)
            atom_name = line[12:16].strip()
            if atom_name:
                current["atom_names"].add(atom_name)
        else:
            if current is not None:
                current["centroid"] = _residue_centroid(current["coords"])
                target_residues.append(current)
                current = None
    if current is not None:
        current["centroid"] = _residue_centroid(current["coords"])
        target_residues.append(current)

    used_named: Dict[Tuple[str, str, str, str], set] = defaultdict(set)
    used_noname: Dict[Tuple[str, str, str], set] = defaultdict(set)

    restored = 0
    for residue in target_residues:
        named_key = residue["named_key"]
        noname_key = residue["noname_key"]
        line_indices = residue["line_indices"]
        residue_centroid = residue.get("centroid")
        residue_atom_names = residue.get("atom_names", set())

        chain_votes: Dict[str, int] = defaultdict(int)
        for line_idx in line_indices:
            atom_line = lines[line_idx]
            atom_key = _pdb_atom_identity_key_no_chain(atom_line)
            chain = ref_atom_chain_by_key.get(atom_key)
            if chain is not None and chain.strip() != "":
                chain_votes[chain] += 1

        selected_chain: Optional[str] = None
        if chain_votes:
            selected_chain = sorted(chain_votes.items(), key=lambda x: (-x[1], x[0]))[0][0]
        else:
            candidates = ref_residue_data_named.get(named_key, [])
            used = used_named[named_key]
            if not candidates:
                candidates = ref_residue_data_noname.get(noname_key, [])
                used = used_noname[noname_key]

            if candidates:
                available = [i for i in range(len(candidates)) if i not in used]
                if not available:
                    available = list(range(len(candidates)))
                chosen_idx = _pick_best_candidate(
                    candidates,
                    available,
                    residue_atom_names,
                    residue_centroid,
                )
                selected_chain = candidates[chosen_idx]["chain"]
                used.add(chosen_idx)

        if selected_chain is None:
            continue

        for line_idx in line_indices:
            line = lines[line_idx]
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            if line[21:22].strip() != "":
                continue
            lines[line_idx] = _pdb_set_chain_id(line, selected_chain)
            restored += 1

    if restored:
        with open(target_pdb, "w") as f:
            f.writelines(lines)

    return restored


def _renumber_pdb_atom_serial(line: str, serial: int) -> str:
    """
    Replace atom serial (columns 7-11) while preserving remaining columns.
    """
    base = line.rstrip("\n")
    if len(base) < 11:
        base = base.ljust(11)
    return f"{base[:6]}{serial:5d}{base[11:]}\n"


def _merge_preserved_hetatm(input_pdb: str, output_pdb: str) -> int:
    """
    Append HETATM records from input_pdb into output_pdb, preserving waters/cofactors
    that PDB2PQR may drop.

    Returns:
        Number of HETATM records appended to output_pdb.
    """
    with open(input_pdb, "r") as f:
        preserved = [line for line in f if line.startswith("HETATM")]
    if not preserved:
        return 0

    with open(output_pdb, "r") as f:
        out_lines = f.readlines()

    existing_keys = {
        _pdb_atom_identity_key(line)
        for line in out_lines
        if line.startswith(("ATOM  ", "HETATM"))
    }

    max_serial = 0
    for line in out_lines:
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                max_serial = max(max_serial, int(line[6:11]))
            except ValueError:
                continue

    appended = []
    for line in preserved:
        key = _pdb_atom_identity_key(line)
        if key in existing_keys:
            continue
        max_serial += 1
        appended.append(_renumber_pdb_atom_serial(line, max_serial))
        existing_keys.add(key)

    if not appended:
        return 0

    # Insert before first END* record if present, otherwise append.
    insert_idx = len(out_lines)
    for i, line in enumerate(out_lines):
        if line.startswith("END"):
            insert_idx = i
            break

    merged = out_lines[:insert_idx] + appended + out_lines[insert_idx:]
    with open(output_pdb, "w") as f:
        f.writelines(merged)

    return len(appended)


def run_pdb2pqr(
    input_pdb: str,
    output_pqr: str,
    output_pdb: str,
    pH: float = 7.4,
    ff: str = "AMBER",
    ffout: str = "AMBER",
) -> Tuple[str, str]:
    """
    Run PDB2PQR with PROPKA to assign protonation states.

    Args:
        input_pdb: Path to input PDB file (cleaned, without hydrogens)
        output_pqr: Path to output PQR file
        output_pdb: Path to output PDB file
        pH: pH value for protonation (default: 7.4)
        ff: Force field (default: "AMBER")
        ffout: Output force field (default: "AMBER")

    Returns:
        Tuple[str, str]: (output_pdb_path, output_pqr_path)
    """
    logger = get_logger()

    if shutil.which("pdb2pqr") is None:
        raise RuntimeError("Missing pdb2pqr. Install: conda install -c conda-forge pdb2pqr")
    if shutil.which("propka3") is None:
        raise RuntimeError("Missing propka3. Install: conda install -c conda-forge propka")

    cmd_common = [
        "pdb2pqr",
        "--ff", ff,
        "--ffout", ffout,
        "--with-ph", str(pH),
        "--titration-state-method=propka",
    ]

    logger.info(f"Running PDB2PQR with PROPKA (pH={pH}) on {input_pdb}")
    # Preserve chain IDs in PDB2PQR output when supported. Fall back if flag
    # is unavailable in older/newer PDB2PQR builds.
    cmd_with_keep_chain = cmd_common + ["--keep-chain", input_pdb, output_pqr]
    cmd_without_keep_chain = cmd_common + [input_pdb, output_pqr]
    try:
        run_cmd(cmd_with_keep_chain, check=True)
    except subprocess.CalledProcessError as e:
        combined = f"{(e.stderr or '')}\n{(e.stdout or '')}".lower()
        if "keep-chain" in combined and (
            "unrecognized" in combined or "no such option" in combined or "usage:" in combined
        ):
            logger.warning(
                "pdb2pqr does not support --keep-chain; retrying without it."
            )
            run_cmd(cmd_without_keep_chain, check=True)
        else:
            raise

    restored_pqr = 0
    try:
        restored_pqr = _restore_chain_ids_from_reference(input_pdb, output_pqr)
        if restored_pqr:
            logger.info(
                f"Restored chain IDs for {restored_pqr} records in protonated PQR from input structure."
            )
    except Exception as e:
        logger.warning(f"Failed to restore chain IDs in protonated PQR: {e}")

    pqr_to_pdb(output_pqr, output_pdb)
    try:
        restored_pdb = _restore_chain_ids_from_reference(input_pdb, output_pdb)
        if restored_pdb:
            logger.info(
                f"Restored chain IDs for {restored_pdb} records in protonated PDB from input structure."
            )
        elif restored_pqr:
            logger.info(
                "Protonated PDB inherited chain IDs from restored protonated PQR."
            )
    except Exception as e:
        logger.warning(f"Failed to restore chain IDs in protonated PDB: {e}")
    try:
        appended = _merge_preserved_hetatm(input_pdb, output_pdb)
        if appended:
            logger.info(
                f"Preserved {appended} HETATM records from input structure in protonated output."
            )
    except Exception as e:
        logger.warning(f"Failed to merge preserved HETATM records: {e}")
    try:
        normalized_ter = _normalize_ter_records(output_pdb)
        if normalized_ter:
            logger.info(
                f"Normalized {normalized_ter} TER record(s) in protonated PDB output."
            )
    except Exception as e:
        logger.warning(f"Failed to normalize TER records in protonated PDB: {e}")
    logger.info(f"Protonation complete: {output_pdb}")
    
    return output_pdb, output_pqr
