"""
Ion and solvent helpers for MD preparation (genion, neutrality, ionic strength).
"""

import os
import re
from typing import Optional

from ....utils import run_cmd
from .grompp_warnings import parse_grompp_warnings

# Map pdb2gmx -water model name to gmx solvate -cs solvent box filename.
WATER_MODEL_TO_SOLVENT_BOX = {
    "tip3p": "tip3p.gro",
    "spc": "spc216.gro",
    "spce": "spce216.gro",
    "tip4p": "tip4p.gro",
    "tip5p": "tip5p.gro",
    "tips3p": "tips3p.gro",
    "a99sbdisp_water": "a99SBdisp_water.gro",
    "opc": "opc.gro",
}

# Water models for which we conservatively use SOL as genion solvent group.
WATER_MODELS_WITH_CONFIDENT_SOL_GROUP = (
    "tip3p",
    "spc",
    "spce",
    "tip4p",
    "tip5p",
)

# Common solvent residue/group names used by GROMACS force fields.
WATER_RESNAME_CANDIDATES = (
    "SOL",
    "WAT",
    "HOH",
    "TIP3",
    "TIP4",
    "TIP5",
    "SPC",
    "SPCE",
    "OPC",
    "W",
)
DEFAULT_SOLVENT_GROUP_FOR_GENION = "SOL"

# Substring in grompp stderr indicating the system is not neutral.
GROMPP_NONZERO_CHARGE_MARKER = "System has non-zero total charge"


def get_solvent_box_for_water_model(water_model: str) -> str:
    """Return the gmx solvate -cs filename for the given pdb2gmx water model."""
    key = (water_model or "tip3p").strip().lower()
    return WATER_MODEL_TO_SOLVENT_BOX.get(key, "tip3p.gro")


# Fallback: many GROMACS installs (e.g. conda) ship spc216.gro but not tip3p.gro.
# spc216.gro is the standard 3-site water box and is suitable for TIP3P/SPC/SPCE.
TIP3P_FALLBACK_SOLVENT_BOX = "spc216.gro"


def get_solvent_box_path(water_model: str, gromacs_bin_path: str) -> Optional[str]:
    """
    Resolve the solvent box coordinate file (-cs for gmx solvate) to an absolute path
    so solvate finds it regardless of current working directory.
    Uses the same GROMACS share/top dirs as pdb2gmx. Returns None if not found.
    For TIP3P, falls back to spc216.gro when tip3p.gro is not present (e.g. conda GROMACS).
    """
    from .forcefield_presets import get_gromacs_top_dirs

    key = (water_model or "tip3p").strip().lower()
    filename = WATER_MODEL_TO_SOLVENT_BOX.get(key, "tip3p.gro")
    top_dirs = get_gromacs_top_dirs(gromacs_bin_path)
    fallback_filename = TIP3P_FALLBACK_SOLVENT_BOX if key in ("tip3p", "tips3p") else None

    def find_in_dirs(fname: str) -> Optional[str]:
        for top_dir in top_dirs:
            if not os.path.isdir(top_dir):
                continue
            candidate = os.path.join(top_dir, fname)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
            try:
                for entry in os.listdir(top_dir):
                    if entry.lower() == fname.lower():
                        p = os.path.join(top_dir, entry)
                        if os.path.isfile(p):
                            return os.path.abspath(p)
            except OSError:
                continue
        return None

    path = find_in_dirs(filename)
    if path is not None:
        return path
    if fallback_filename and fallback_filename != filename:
        return find_in_dirs(fallback_filename)
    return None


def parse_topology_molecule_counts(top_path: str) -> dict:
    """Parse [ molecules ] entries from a GROMACS topology into {name: count}."""
    if not os.path.exists(top_path):
        raise FileNotFoundError(f"Topology file not found: {top_path}")
    with open(top_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    in_molecules = False
    counts = {}
    for raw in lines:
        line = raw.split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            in_molecules = line.lower() == "[ molecules ]"
            continue
        if not in_molecules:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        if re.fullmatch(r"[+-]?\d+", parts[1]) is None:
            continue
        name = parts[0]
        counts[name] = counts.get(name, 0) + int(parts[1])
    if not counts:
        raise RuntimeError(
            f"Could not parse [ molecules ] counts from topology: {top_path}"
        )
    return counts


def get_water_ion_resnames_from_topology(top_path: str) -> list:
    """
    Return molecule names from the topology that are water or ions, in the order
    they appear in [ molecules ]. Use this for Water_and_ions index selection so
    the selection matches the actual residue names in the system.
    """
    from .constants import ION_RESNAMES

    counts = parse_topology_molecule_counts(top_path)
    water_ion = []
    for name in counts:
        if name.upper() in WATER_RESNAME_CANDIDATES or name.upper() in ION_RESNAMES:
            water_ion.append(name)
    return water_ion


def detect_solvent_group_for_genion(system_top_path: str) -> str:
    """Detect the solvent group name used in topology (e.g. SOL, WAT, HOH)."""
    counts = parse_topology_molecule_counts(system_top_path)
    best_name = None
    best_count = -1
    for name, count in counts.items():
        if name.upper() in WATER_RESNAME_CANDIDATES and count > best_count:
            best_name = name
            best_count = count
    if best_name is None:
        candidates = ", ".join(WATER_RESNAME_CANDIDATES)
        raise RuntimeError(
            "Could not detect solvent residue name for genion in system.top. "
            f"Tried candidates: {candidates}. Parsed molecules: {sorted(counts.keys())}"
        )
    return best_name


def choose_solvent_group_for_genion(
    system_top_path: str, known_water_model: str = None
) -> str:
    """
    Choose solvent group for genion.
    Fast path: use SOL only for conservative known-safe water models.
    Fallback: detect from topology for standalone/custom runs.
    """
    normalized = (known_water_model or "").strip().lower()
    if normalized in WATER_MODELS_WITH_CONFIDENT_SOL_GROUP:
        return DEFAULT_SOLVENT_GROUP_FOR_GENION
    return detect_solvent_group_for_genion(system_top_path)


def validate_ionic_strength_after_genion(
    system_top_path: str, pname: str, nname: str, target_conc: float, logger
) -> None:
    """
    Lightweight ionic-strength sanity check based on topology molecule counts.
    Uses water ratio approximation: c ~= 55.5 * N_ion / N_water.
    """
    if target_conc <= 0:
        return

    counts = parse_topology_molecule_counts(system_top_path)
    upper_counts = {k.upper(): v for k, v in counts.items()}
    n_pos = upper_counts.get((pname or "").upper(), 0)
    n_neg = upper_counts.get((nname or "").upper(), 0)

    water_name = None
    n_water = 0
    for name, count in counts.items():
        if name.upper() in WATER_RESNAME_CANDIDATES and count > n_water:
            water_name = name
            n_water = count
    if n_water <= 0:
        raise RuntimeError(
            "Could not validate ionic strength: no water molecules found in system.top."
        )
    if n_pos == 0 or n_neg == 0:
        raise RuntimeError(
            f"Ionic-strength sanity check failed: target concentration is {target_conc:.3f} M "
            f"but ion counts are {pname}={n_pos}, {nname}={n_neg}."
        )

    approx_pair_conc = 55.5 * (min(n_pos, n_neg) / n_water)
    approx_ionic_strength = 55.5 * (0.5 * (n_pos + n_neg) / n_water)
    lower_bound = max(0.01, target_conc * 0.50)
    if approx_pair_conc < lower_bound:
        raise RuntimeError(
            f"Ionic-strength sanity check failed: estimated salt concentration "
            f"{approx_pair_conc:.3f} M (water={water_name}:{n_water}, {pname}={n_pos}, {nname}={n_neg}) "
            f"is too low for target {target_conc:.3f} M."
        )
    logger.info(
        f"Ionic-strength sanity check passed: estimated pair concentration {approx_pair_conc:.3f} M, "
        f"estimated ionic strength {approx_ionic_strength:.3f} M (target {target_conc:.3f} M)."
    )


_CHARGE_CHECK_MDP_CONTENT = """\
; Minimal MDP for charge-check grompp only (no simulation).
; Intentionally avoids freezegrps / tc-grps / energygrps so that
; no custom index groups are required.
integrator  = steep
nsteps      = 0
"""


def validate_system_neutrality_after_genion(
    gmx_machine, mdp_dir: str, pose_name: str
) -> None:
    """
    Validate system neutrality with a check-only grompp invocation.

    Uses a minimal throw-away MDP rather than min.mdp so that custom index
    groups (e.g. LIG, Backbone) are not required at this stage.
    """
    check_mdp = os.path.join(pose_name, "_charge_check.mdp")
    check_tpr = os.path.join(pose_name, "_charge_check.tpr")
    check_mdout = os.path.join(pose_name, "_charge_check_mdout.mdp")
    tmp_files = [check_mdp, check_tpr, check_mdout]

    with open(check_mdp, "w", encoding="utf-8") as fh:
        fh.write(_CHARGE_CHECK_MDP_CONTENT)

    grompp_cmd = [
        gmx_machine,
        "grompp",
        "-f",
        check_mdp,
        "-c",
        f"{pose_name}/solv_ions.gro",
        "-p",
        f"{pose_name}/system.top",
        "-o",
        check_tpr,
        "-po",
        check_mdout,
    ]
    try:
        result = run_cmd(grompp_cmd, check=False)
        combined = (result.stdout or "") + (result.stderr or "")
        if GROMPP_NONZERO_CHARGE_MARKER in combined:
            raise RuntimeError(
                "System is not neutral after genion. grompp reported a non-zero total charge. "
                "Check ligand/protein charges and that genion ran with -neutral. "
                f"grompp output (excerpt): {combined[:500]}"
            )
        if result.returncode != 0:
            warnings = parse_grompp_warnings(combined)
            excerpt_len = 1500
            excerpt = (
                combined[-excerpt_len:]
                if len(combined) > excerpt_len
                else combined
            )
            if not warnings:
                raise RuntimeError(
                    "Neutrality validation failed because grompp did not complete successfully. "
                    "grompp output (tail, where the error usually appears):\n"
                    f"{excerpt}"
                )
    finally:
        for tmp in tmp_files:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

