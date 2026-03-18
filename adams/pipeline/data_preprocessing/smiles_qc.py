"""
Utilities for robust SMILES parsing and controlled kekulization fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
from typing import Dict, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize


@dataclass
class SmilesParseResult:
    mol: Optional[Chem.Mol]
    canonical_smiles: Optional[str]
    parse_mode: str
    reason: str
    sanitize_failure_step: Optional[str]


def _sanitize_flag_name(flag_value: int) -> Optional[str]:
    if flag_value == 0:
        return None

    sanitize_flags: Dict[int, str] = {}
    for attr in dir(Chem.SanitizeFlags):
        if not attr.startswith("SANITIZE_"):
            continue
        value = getattr(Chem.SanitizeFlags, attr)
        try:
            sanitize_flags[int(value)] = attr
        except Exception:
            continue

    return sanitize_flags.get(flag_value, f"SANITIZE_UNKNOWN_{flag_value}")


def _sanitize_failure_step(mol: Chem.Mol, sanitize_ops: int) -> Optional[str]:
    step = Chem.SanitizeMol(mol, sanitizeOps=sanitize_ops, catchErrors=True)
    try:
        step_val = int(step)
    except Exception:
        # Best effort fallback; unknown object from RDKit bindings.
        return "SANITIZE_UNKNOWN"
    return _sanitize_flag_name(step_val)


def is_kekulize_rescue_mode(parse_mode: str) -> bool:
    return parse_mode.startswith("kekulize_rescued")


def _openbabel_canonical_smiles(smiles: str) -> Optional[str]:
    """
    Best-effort SMILES recanonicalization using local OpenBabel (offline).
    Returns None when OpenBabel is unavailable or conversion fails.
    """
    obabel = shutil.which("obabel")
    if not obabel:
        return None

    try:
        proc = subprocess.run(
            [obabel, f"-:{smiles}", "-ocan"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:
        return None

    if proc.returncode != 0 and not proc.stdout:
        return None

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        return None

    # OpenBabel canonical output is typically "<smiles>	<title>".
    return lines[-1].split()[0]


def _openbabel_rescue_parse_result(
    smiles: str,
    prefer_restandardize: bool,
) -> Optional[SmilesParseResult]:
    repaired = _openbabel_canonical_smiles(smiles)
    if not repaired or repaired == smiles:
        return None

    repaired_result = parse_smiles_with_kekulize_rescue(
        repaired,
        prefer_restandardize=prefer_restandardize,
        allow_openbabel_fallback=False,
    )
    if repaired_result.mol is None:
        return None

    repaired_reason = (
        "rescued via OpenBabel recanonicalization"
        if not repaired_result.reason
        else f"{repaired_result.reason}; rescued via OpenBabel recanonicalization"
    )
    return SmilesParseResult(
        mol=repaired_result.mol,
        canonical_smiles=repaired_result.canonical_smiles,
        parse_mode=f"openbabel_recanonicalized->{repaired_result.parse_mode}",
        reason=repaired_reason,
        sanitize_failure_step=repaired_result.sanitize_failure_step,
    )


def parse_smiles_with_kekulize_rescue(
    smiles: str,
    prefer_restandardize: bool = True,
    allow_openbabel_fallback: bool = True,
    force_openbabel_recanonicalization: bool = False,
) -> SmilesParseResult:
    """
    Parse SMILES strictly first, then allow rescue only for kekulization-only failures.

    Rescue policy:
    - Strict parse (sanitize=True) succeeds -> accept.
    - Strict parse fails:
      1) Parse with sanitize=False
      2) Detect failing sanitize step with catchErrors=True
      3) If failing step is NOT SANITIZE_KEKULIZE -> reject (with optional OpenBabel fallback)
      4) If failing step is SANITIZE_KEKULIZE:
         - sanitize with all ops except kekulization
         - optionally re-standardize through canonical SMILES and try strict parse again
         - accept rescued representation
      5) Final fallback (offline): recanonicalize with local OpenBabel and retry strict parsing

    force_openbabel_recanonicalization:
      - If True, try OpenBabel recanonicalization first (before RDKit strict parse).
    """

    if force_openbabel_recanonicalization and allow_openbabel_fallback:
        forced_obabel = _openbabel_rescue_parse_result(
            smiles=smiles,
            prefer_restandardize=prefer_restandardize,
        )
        if forced_obabel is not None:
            return forced_obabel

    def _final_fail(reason: str, sanitize_failure_step: Optional[str]) -> SmilesParseResult:
        if allow_openbabel_fallback:
            obabel_rescue = _openbabel_rescue_parse_result(
                smiles=smiles,
                prefer_restandardize=prefer_restandardize,
            )
            if obabel_rescue is not None:
                return obabel_rescue

        return SmilesParseResult(
            mol=None,
            canonical_smiles=None,
            parse_mode="failed",
            reason=reason,
            sanitize_failure_step=sanitize_failure_step,
        )

    strict = Chem.MolFromSmiles(smiles, sanitize=True)
    if strict is not None:
        can = Chem.MolToSmiles(strict, canonical=True, isomericSmiles=True)
        return SmilesParseResult(
            mol=strict,
            canonical_smiles=can,
            parse_mode="strict",
            reason="",
            sanitize_failure_step=None,
        )

    raw = Chem.MolFromSmiles(smiles, sanitize=False)
    if raw is None:
        return _final_fail("RDKit could not parse SMILES", None)

    fail_step = _sanitize_failure_step(Chem.Mol(raw), int(Chem.SanitizeFlags.SANITIZE_ALL))
    if fail_step is None:
        # Rare case: catchErrors path reports success although strict parser returned None.
        # Build a sanitized molecule and continue.
        rebuilt = Chem.Mol(raw)
        Chem.SanitizeMol(rebuilt)
        can = Chem.MolToSmiles(rebuilt, canonical=True, isomericSmiles=True)
        return SmilesParseResult(
            mol=rebuilt,
            canonical_smiles=can,
            parse_mode="resanitized",
            reason="strict parse failed but full sanitize succeeded via sanitize=False path",
            sanitize_failure_step=None,
        )

    if fail_step != "SANITIZE_KEKULIZE":
        return _final_fail(f"Sanitization failed at {fail_step}", fail_step)

    rescued = Chem.Mol(raw)
    relaxed_ops = int(Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
    relaxed_fail = _sanitize_failure_step(rescued, relaxed_ops)
    if relaxed_fail is not None:
        return _final_fail(
            f"Kekulize-rescue sanitization failed at {relaxed_fail}",
            relaxed_fail,
        )

    canonical_relaxed = Chem.MolToSmiles(rescued, canonical=True, isomericSmiles=True)
    if prefer_restandardize:
        candidates = [canonical_relaxed]
        try:
            standardized = rdMolStandardize.StandardizeSmiles(canonical_relaxed)
            if standardized and standardized not in candidates:
                candidates.insert(0, standardized)
        except Exception:
            # Keep fallback path robust even if MolStandardize is unavailable.
            pass

        for candidate in candidates:
            strict_after = Chem.MolFromSmiles(candidate, sanitize=True)
            if strict_after is not None:
                canonical_strict = Chem.MolToSmiles(
                    strict_after, canonical=True, isomericSmiles=True
                )
                return SmilesParseResult(
                    mol=strict_after,
                    canonical_smiles=canonical_strict,
                    parse_mode="kekulize_rescued_restandardized",
                    reason=(
                        "strict parse failed due to SANITIZE_KEKULIZE; rescued by "
                        "sanitize-without-kekulize and strict reparse of standardized representation"
                    ),
                    sanitize_failure_step="SANITIZE_KEKULIZE",
                )

    return SmilesParseResult(
        mol=rescued,
        canonical_smiles=canonical_relaxed,
        parse_mode="kekulize_rescued_relaxed",
        reason=(
            "strict parse failed due to SANITIZE_KEKULIZE; kept sanitize-without-kekulize representation"
        ),
        sanitize_failure_step="SANITIZE_KEKULIZE",
    )



def quick_3d_forcefield_qc(mol: Chem.Mol) -> Tuple[bool, str]:
    """
    Lightweight geometry/forcefield gate for rescued molecules.
    """
    try:
        m = Chem.AddHs(Chem.Mol(mol))
    except Exception as exc:
        return False, f"add_hs_failed: {exc}"

    try:
        if AllChem.EmbedMolecule(m, AllChem.ETKDGv3()) != 0:
            return False, "embed_failed"
    except Exception as exc:
        return False, f"embed_exception: {exc}"

    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(m, mmffVariant="MMFF94s")
        if mmff_props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(m, mmff_props)
            if ff is not None:
                ff.Minimize(maxIts=100)
                _ = ff.CalcEnergy()
                return True, "mmff_ok"
    except Exception:
        # Fall back to UFF if MMFF fails.
        pass

    try:
        uff = AllChem.UFFGetMoleculeForceField(m)
        if uff is None:
            return False, "no_mmff_or_uff"
        uff.Minimize(maxIts=100)
        _ = uff.CalcEnergy()
        return True, "uff_ok"
    except Exception as exc:
        return False, f"uff_exception: {exc}"
