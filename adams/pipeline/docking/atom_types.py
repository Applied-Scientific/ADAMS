"""Utilities for validating and normalizing AutoDock atom types in ligand PDBQT files."""

from __future__ import annotations

import collections
import os
import re
import tempfile
from typing import Dict, List

# Vina/Vina-GPU-supported AutoDock atom types.
SUPPORTED_AD_TYPES = {
    "A",
    "C",
    "N",
    "NA",
    "OA",
    "O",
    "S",
    "SA",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "H",
    "HD",
    "HS",
    "Mg",
    "Mn",
    "Zn",
    "Ca",
    "Fe",
    "Cu",
}

AD_TYPE_BY_ELEMENT = {
    "C": "C",
    "N": "N",
    "O": "O",
    "S": "S",
    "P": "P",
    "F": "F",
    "Cl": "Cl",
    "Br": "Br",
    "I": "I",
    "H": "H",
    "Mg": "Mg",
    "Mn": "Mn",
    "Zn": "Zn",
    "Ca": "Ca",
    "Fe": "Fe",
    "Cu": "Cu",
}


def _normalize_symbol(raw_symbol: str) -> str:
    if not raw_symbol:
        return ""
    raw_symbol = raw_symbol.strip()
    if not raw_symbol:
        return ""
    two = raw_symbol[:2]
    one = raw_symbol[:1]
    normalized_two = (
        two[0].upper() + two[1:].lower()
        if len(two) == 2
        else two.upper()
    )
    if normalized_two in AD_TYPE_BY_ELEMENT:
        return normalized_two
    return one.upper()


def extract_element_hint_from_atom_line(line: str) -> str:
    """Best-effort element hint from a PDBQT ATOM/HETATM line."""
    # Preferred: explicit element column when present.
    element = line[76:78].strip() if len(line) >= 78 else ""
    symbol = _normalize_symbol(element)
    if symbol:
        return symbol

    # Tokenized parse is more robust than fixed columns for loosely formatted PDBQT.
    tokens = line.split()
    if len(tokens) >= 3 and tokens[0] in {"ATOM", "HETATM"}:
        atom_name = tokens[2]
        match = re.match(r"^[0-9]*([A-Za-z]{1,2})", atom_name)
        if match:
            symbol = _normalize_symbol(match.group(1))
            if symbol:
                return symbol

    # Fallback: fixed-width atom-name column.
    atom_name = line[12:16].strip() if len(line) >= 16 else ""
    match = re.match(r"^[0-9]*([A-Za-z]{1,2})", atom_name)
    if not match:
        return ""
    return _normalize_symbol(match.group(1))


def normalize_autodock_atom_type(atom_type: str, element_hint: str):
    """Map unsupported atom types (e.g., CG0, G0) to Vina-compatible types."""
    atom_type = atom_type.strip()
    if atom_type in SUPPORTED_AD_TYPES:
        return atom_type

    # Common unsupported variants observed in practice.
    if re.fullmatch(r"C[Gg]\d*", atom_type):
        return "C"
    if re.fullmatch(r"N[Gg]\d*", atom_type):
        return "N"
    if re.fullmatch(r"O[Gg]\d*", atom_type):
        return "O"
    if re.fullmatch(r"S[Gg]\d*", atom_type):
        return "S"
    if re.fullmatch(r"H[Gg]\d*", atom_type):
        return "H"
    if re.fullmatch(r"[Gg]\d*", atom_type):
        # Seen in rescued ligands (e.g., 3mxf): treat as carbon-like.
        return "C"

    alpha_only = re.sub(r"[^A-Za-z]", "", atom_type)
    for candidate in (alpha_only, alpha_only[:2], alpha_only[:1]):
        if candidate in SUPPORTED_AD_TYPES:
            return candidate

    element_hint = element_hint.strip()
    if element_hint:
        canonical_element = element_hint[0].upper() + element_hint[1:].lower()
        mapped = AD_TYPE_BY_ELEMENT.get(canonical_element)
        if mapped is not None:
            return mapped

    return None


def _replace_file_atomically(path: str, lines: List[str]) -> None:
    directory = os.path.dirname(os.path.abspath(path)) or "."
    with tempfile.NamedTemporaryFile(
        mode="w",
        delete=False,
        dir=directory,
        prefix=".atomtypes_tmp_",
    ) as tmp:
        tmp.writelines(lines)
        tmp_path = tmp.name
    os.replace(tmp_path, path)


def sanitize_pdbqt_atom_types(
    src_path: str,
    dst_path: str | None = None,
    strict_unresolved: bool = True,
):
    """
    Normalize unsupported AutoDock atom types in a ligand PDBQT file.

    Returns a dict with counts and detailed replacement/unresolved information.
    If strict_unresolved=True, raises ValueError when unresolved atom types remain.
    """
    changed_entries = 0
    replacement_counts: Dict[str, int] = collections.Counter()
    unresolved_details: List[dict] = []
    output_lines: List[str] = []
    atom_line_count = 0

    with open(src_path, "r", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.startswith(("ATOM", "HETATM")):
                atom_line_count += 1
                stripped = line.rstrip("\n")
                token_match = re.match(r"^(.*\s)(\S+)\s*$", stripped)
                if token_match:
                    original_type = token_match.group(2)
                    element_hint = extract_element_hint_from_atom_line(line)
                    normalized_type = normalize_autodock_atom_type(
                        original_type,
                        element_hint,
                    )
                    if normalized_type is None:
                        unresolved_details.append(
                            {
                                "line_number": line_number,
                                "atom_type": original_type,
                                "element_hint": element_hint,
                            }
                        )
                    elif normalized_type != original_type:
                        replacement_counts[f"{original_type}->{normalized_type}"] += 1
                        changed_entries += 1
                        line = f"{token_match.group(1)}{normalized_type}\n"
            output_lines.append(line)

    if strict_unresolved and unresolved_details:
        first = unresolved_details[0]
        raise ValueError(
            "Unsupported ligand atom type for Vina/Vina-GPU "
            f"in {os.path.basename(src_path)} at line {first['line_number']}: "
            f"{first['atom_type']!r}, element_hint={first['element_hint']!r}"
        )

    target_path = dst_path or src_path
    if dst_path is not None:
        with open(dst_path, "w") as handle:
            handle.writelines(output_lines)
    elif changed_entries:
        _replace_file_atomically(src_path, output_lines)

    return {
        "source_path": src_path,
        "output_path": target_path,
        "atom_line_count": atom_line_count,
        "changed_entries": changed_entries,
        "replacement_counts": dict(replacement_counts),
        "unresolved_details": unresolved_details,
    }
