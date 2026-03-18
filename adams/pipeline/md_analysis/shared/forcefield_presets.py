"""
Amber force field presets for the MD analysis pipeline.

Maps named presets to GROMACS pdb2gmx (forcefield, water_model) and documents
recommended Amber combinations, including preset-default ligand settings
(GAFF2 + AM1-BCC). run_lig_prepare can still override ligand parameters.

Compatibility: conda GROMACS 2024.4. ff99sb_ildn_tip3p is in the standard conda GROMACS;
amber14sb and a99SBdisp are added by the ADAMS install script (Step 5.5). Presets that
use water models outside the pdb2gmx -water enum (e.g. opc) are not available in the
standard build.

References:
- https://ambermd.org/AmberModels.php
- https://ambermd.org/AmberModels_proteins.php
"""

import os
from typing import List, Optional, Set, Tuple

# Water models accepted by pdb2gmx -water in conda GROMACS 2024.4
CONDA_GROMACS_WATER_MODELS = frozenset(
    {"select", "none", "spc", "spce", "tip3p", "tip4p", "tip5p", "tips3p"}
)

# Preset metadata. Protein/water are used by pdb2gmx; ligand defaults document the
# intended stack for run_lig_prepare (can be overridden per run).
AMBER_FF_PRESETS = {
    # ff19SB / OPC (Amber recommended: ff19SB pairs with OPC). No .ff in conda.
    "ff19sb_opc": {
        "gromacs_ff": "amber19sb_opc",
        "gromacs_water": "opc",
        "ligand_atom_type_default": "gaff2",
        "ligand_charge_type_default": "bcc",
        "description": "ff19SB protein, OPC water, GAFF2 ligands, AM1-BCC charges (preset defaults; configurable)",
    },
    # Amber14SB / TIP3P. .ff added by ADAMS install script.
    # ff14SB alias retained for user-facing naming parity.
    "ff14sb_tip3p": {
        "gromacs_ff": "amber14sb",
        "gromacs_water": "tip3p",
        "ligand_atom_type_default": "gaff2",
        "ligand_charge_type_default": "bcc",
        "description": "ff14SB protein (amber14sb), TIP3P water, GAFF2 ligands, AM1-BCC charges (preset defaults; configurable)",
    },
    "amber14sb_tip3p": {
        "gromacs_ff": "amber14sb",
        "gromacs_water": "tip3p",
        "ligand_atom_type_default": "gaff2",
        "ligand_charge_type_default": "bcc",
        "description": "Amber14SB protein, TIP3P water, GAFF2 ligands, AM1-BCC charges (preset defaults; configurable)",
    },
    # ff99SB-ILDN / TIP3P. In standard conda GROMACS.
    "ff99sb_ildn_tip3p": {
        "gromacs_ff": "amber99sb-ildn",
        "gromacs_water": "tip3p",
        "ligand_atom_type_default": "gaff2",
        "ligand_charge_type_default": "bcc",
        "description": "ff99SB-ILDN protein, TIP3P water, GAFF2 ligands, AM1-BCC charges (preset defaults; configurable)",
    },
    # a99SB-disp (Robustelli port a99SBdisp.ff). .ff added by ADAMS install script.
    # Water model "a99SBdisp_water" is a TIP4P-D variant defined in a99SBdisp.ff/watermodels.dat.
    "a99sb_disp": {
        "gromacs_ff": "a99SBdisp",
        "gromacs_water": "a99SBdisp_water",
        "ligand_atom_type_default": "gaff2",
        "ligand_charge_type_default": "bcc",
        "description": "a99SB-disp protein, a99SBdisp water (TIP4P-D variant), GAFF2 ligands, AM1-BCC charges (preset defaults; configurable)",
    },
    # CHARMM36m / TIP3P. .ff from charmm2gmx or MacKerell port; not provided by ADAMS install.
    "charmm36m": {
        "gromacs_ff": "charmm36m",
        "gromacs_water": "tip3p",
        "ligand_atom_type_default": "gaff2",
        "ligand_charge_type_default": "bcc",
        "description": "CHARMM36m protein, TIP3P water, GAFF2 ligands, AM1-BCC charges (preset defaults; configurable)",
    },
}

# Preset names whose .ff are in the standard conda GROMACS package
PRESETS_IN_STANDARD_GROMACS = frozenset({"ff99sb_ildn_tip3p"})

# Presets whose .ff are not in standard GROMACS (amber14sb, a99SBdisp added by ADAMS install script)
PRESETS_REQUIRING_EXTRA_FF = frozenset(
    {"ff19sb_opc", "ff14sb_tip3p", "amber14sb_tip3p", "a99sb_disp", "charmm36m"}
)


def get_forcefield_and_water(
    forcefield_preset_or_ff: str,
    water_model: str = "tip3p",
) -> Tuple[str, str]:
    """
    Resolve a preset name or raw forcefield to (gromacs_ff, gromacs_water).

    Args:
        forcefield_preset_or_ff: Either a preset key from AMBER_FF_PRESETS
            (e.g. "ff19sb_opc", "ff14sb_tip3p", "amber14sb_tip3p",
            "ff99sb_ildn_tip3p", "a99sb_disp", "charmm36m")
            or a raw GROMACS force field name (e.g. "amber99sb-ildn").
        water_model: Used only when forcefield_preset_or_ff is a raw force field name
            (not a preset). Default "tip3p".

    Returns:
        (gromacs_ff, gromacs_water) for use with pdb2gmx -ff and -water.

    Examples:
        >>> get_forcefield_and_water("ff99sb_ildn_tip3p")
        ('amber99sb-ildn', 'tip3p')
        >>> get_forcefield_and_water("amber99sb-ildn", "tip3p")
        ('amber99sb-ildn', 'tip3p')
    """
    preset = forcefield_preset_or_ff.strip().lower()
    if preset in AMBER_FF_PRESETS:
        entry = AMBER_FF_PRESETS[preset]
        return entry["gromacs_ff"], entry["gromacs_water"]
    # Treat as raw forcefield name
    return forcefield_preset_or_ff, water_model


def list_presets():
    """Return list of preset names for documentation/CLI."""
    return list(AMBER_FF_PRESETS.keys())


def get_gromacs_top_dirs(gromacs_bin_path: str) -> List[str]:
    """
    Resolve candidate GROMACS top directories from a gmx binary path.

    Handles both standard (.../bin/gmx) and CUDA (.../cuda/bin/gmx) layouts.
    Returned list preserves search priority.
    """
    bin_dir = os.path.dirname(os.path.abspath(gromacs_bin_path))
    prefix_candidates = []
    current = bin_dir
    for _ in range(3):
        current = os.path.dirname(current)
        if current and current not in prefix_candidates:
            prefix_candidates.append(current)

    top_dirs = []
    for prefix in prefix_candidates:
        for candidate in (
            os.path.join(prefix, "share", "gromacs", "top"),
            os.path.join(prefix, "share", "top"),
        ):
            if candidate not in top_dirs:
                top_dirs.append(candidate)
    return top_dirs


def get_gromacs_top_dir(gromacs_bin_path: str) -> str:
    """
    Resolve GROMACS share/top directory from the bin path (e.g. .../cuda/bin or .../bin).

    pdb2gmx looks for .ff directories in share/gromacs/top relative to the installation.
    """
    top_dirs = get_gromacs_top_dirs(gromacs_bin_path)
    for top in top_dirs:
        if os.path.isdir(top):
            return top
    # Return first expected candidate even if missing; caller can report diagnostics.
    return top_dirs[0] if top_dirs else ""


def find_forcefield_dir(forcefield: str, top_dirs: List[str]) -> Optional[str]:
    """Return absolute path to <forcefield>.ff if present in any top directory."""
    ff_target = f"{forcefield}.ff"
    ff_target_lower = ff_target.lower()
    for top_dir in top_dirs:
        if not os.path.isdir(top_dir):
            continue
        direct = os.path.join(top_dir, ff_target)
        if os.path.isdir(direct):
            return direct
        try:
            for entry in os.listdir(top_dir):
                if entry.lower() == ff_target_lower:
                    candidate = os.path.join(top_dir, entry)
                    if os.path.isdir(candidate):
                        return candidate
        except OSError:
            continue
    return None


def _read_watermodels_from_ff_dir(forcefield_dir: str) -> Set[str]:
    """
    Read water-model keys from forcefield_dir/watermodels.dat.

    Returns lowercase keys.
    """
    watermodels_path = os.path.join(forcefield_dir, "watermodels.dat")
    if not os.path.isfile(watermodels_path):
        return set()
    models = set()
    try:
        with open(watermodels_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.split(";", 1)[0].strip()
                if not line:
                    continue
                token = line.split()[0].strip()
                if token:
                    models.add(token.lower())
    except OSError:
        return set()
    return models


def validate_forcefield_and_water(
    forcefield: str,
    water_model: str,
    gromacs_bin_path: str,
) -> Tuple[str, str]:
    """
    Validate that forcefield and water model are usable before running pdb2gmx.

    Returns:
        (forcefield_dir, normalized_water_model)

    Raises:
        FileNotFoundError: forcefield directory is missing.
        ValueError: water model is not supported for the resolved force field.
    """
    top_dirs = get_gromacs_top_dirs(gromacs_bin_path)
    forcefield_dir = find_forcefield_dir(forcefield, top_dirs)
    existing_top_dirs = [d for d in top_dirs if os.path.isdir(d)]

    if forcefield_dir is None:
        searched = existing_top_dirs if existing_top_dirs else top_dirs
        searched_text = ", ".join(searched) if searched else "(none)"
        raise FileNotFoundError(
            f"Force field '{forcefield}' is not installed (expected '{forcefield}.ff'). "
            f"Searched GROMACS top directories: {searched_text}. "
            "Install the force field (e.g. via ADAMS install step 5.5) or choose another preset."
        )

    normalized_water = (water_model or "").strip().lower()
    if not normalized_water:
        normalized_water = "tip3p"

    ff_specific_water = _read_watermodels_from_ff_dir(forcefield_dir)
    # Prefer forcefield-local watermodels.dat when present; this is the most
    # accurate source for what pdb2gmx will accept with the selected .ff.
    if ff_specific_water:
        water_ok = normalized_water in ff_specific_water
    else:
        water_ok = normalized_water in CONDA_GROMACS_WATER_MODELS
    if not water_ok:
        available = sorted(CONDA_GROMACS_WATER_MODELS | ff_specific_water)
        raise ValueError(
            f"Water model '{normalized_water}' is not available for force field '{forcefield}'. "
            f"Resolved force field directory: {forcefield_dir}. "
            f"Available water models: {', '.join(available) if available else '(none detected)'}."
        )
    return forcefield_dir, normalized_water
