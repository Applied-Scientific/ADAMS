"""
data_preprocessing/microstate_enumeration.py

Ligand microstate enumeration: tautomers, protomers (protonation states), and stereoisomers.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.EnumerateStereoisomers import (
    EnumerateStereoisomers,
    StereoEnumerationOptions,
)
from rdkit.Chem.MolStandardize import rdMolStandardize

from ...logger_utils import get_logger
from .smiles_qc import is_kekulize_rescue_mode, parse_smiles_with_kekulize_rescue

# Lazy logger to avoid requiring agent data path at import time (set by CLI on startup).
_logger = None


def _get_logger():
    global _logger
    if _logger is None:
        _logger = get_logger()
    return _logger


def _canonicalize_smiles(smiles: str) -> Optional[str]:
    """Canonicalize SMILES via RDKit for consistent dedup across toolchains."""
    parsed = parse_smiles_with_kekulize_rescue(smiles, prefer_restandardize=True)
    m = parsed.mol
    if m is None:
        return None
    try:
        return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)
    except Exception:
        return None


def _variant_id_to_record_id(parent_id: str, variant_id: str) -> str:
    """Create a stable unique ID per microstate record."""
    return f"{parent_id}__{variant_id}"


def _estimate_mol_energy_kcal(mol: Chem.Mol) -> float:
    """
    Estimate molecule strain energy (kcal/mol) from a single embedded conformer.

    Uses MMFF94s when parameters are available, otherwise falls back to UFF.
    Returns +inf if no forcefield energy can be computed.
    """
    try:
        mol_h = Chem.AddHs(Chem.Mol(mol))
        if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) != 0:
            return float("inf")

        mmff_props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
        if mmff_props is not None:
            mmff_ff = AllChem.MMFFGetMoleculeForceField(mol_h, mmff_props)
            if mmff_ff is not None:
                mmff_ff.Minimize(maxIts=200)
                return float(mmff_ff.CalcEnergy())

        uff_ff = AllChem.UFFGetMoleculeForceField(mol_h)
        if uff_ff is not None:
            uff_ff.Minimize(maxIts=200)
            return float(uff_ff.CalcEnergy())
    except Exception:
        pass

    return float("inf")


def enumerate_tautomers(
    mol: Chem.Mol,
    top_tautomers_per_protomer: int = 2,
    energy_window_kcal: float = 3.0,
    hard_max_generated_tautomers: Optional[int] = None,
) -> List[Tuple[Chem.Mol, str, float]]:
    """
    Enumerate tautomers of a molecule using RDKit's TautomerEnumerator.

    Tautomers are canonicalized and deduplicated by canonical SMILES, then ranked
    by force-field energy.

    Args:
        mol: RDKit molecule object
        top_tautomers_per_protomer: Maximum number of ranked tautomers to keep.
        energy_window_kcal: Keep candidates within this energy window from best.
        hard_max_generated_tautomers: Optional safety cap for generated unique tautomers.

    Returns:
        List of tuples: [(mol, canonical_smiles, energy_kcal), ...]
    """
    try:
        # Enumerate all available tautomers first, then rank and prune.
        enumerator = rdMolStandardize.TautomerEnumerator()
        tautomers = list(enumerator.Enumerate(mol))

        unique: Dict[str, Chem.Mol] = {}
        for t in tautomers:
            try:
                smi = Chem.MolToSmiles(t, canonical=True, isomericSmiles=True)
                can = _canonicalize_smiles(smi)
                if can is None:
                    continue
                if can not in unique:
                    unique[can] = Chem.Mol(t)
                    if (
                        hard_max_generated_tautomers is not None
                        and len(unique) >= hard_max_generated_tautomers
                    ):
                        _get_logger().warning(
                            "Tautomer enumeration hit hard safety cap at "
                            f"hard_max_generated_tautomers={hard_max_generated_tautomers}"
                        )
                        break
            except Exception as e:
                _get_logger().warning(f"Error processing tautomer: {e}")
                continue

        ranked: List[Tuple[Chem.Mol, str, float]] = []
        for smi, taut_mol in unique.items():
            e = _estimate_mol_energy_kcal(taut_mol)
            ranked.append((Chem.Mol(taut_mol), smi, e))

        ranked.sort(key=lambda item: (item[2], item[1]))
        if not ranked:
            return []

        best_e = ranked[0][2]
        if best_e != float("inf"):
            kept = [item for item in ranked if (item[2] - best_e) <= energy_window_kcal]
        else:
            kept = ranked
        kept = kept[: max(1, top_tautomers_per_protomer)]
        return kept

    except Exception as e:
        _get_logger().error(f"Error enumerating tautomers: {e}")
        # Return original molecule if enumeration fails
        try:
            return [
                (
                    Chem.Mol(mol),
                    Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
                    float("inf"),
                )
            ]
        except Exception:
            return []


def enumerate_stereoisomers(
    mol: Chem.Mol, only_unassigned: bool = True, max_stereoisomers: int = 16
) -> List[Tuple[Chem.Mol, str]]:
    """
    Enumerate stereoisomers of a molecule using RDKit's EnumerateStereoisomers.

    By default, only unassigned stereocenters are enumerated to preserve existing
    stereochemical assignments. EnumerateStereoisomers with unique=True deduplicates
    based on CXSMILES. RDKit always returns at least 1 molecule (the original).

    Args:
        mol: RDKit molecule object
        only_unassigned: If True (default), only enumerate unassigned stereocenters.
        max_stereoisomers: Maximum number of stereoisomers to return (default: 16)

    Returns:
        List of tuples: [(mol, canonical_smiles), ...] for each unique stereoisomer
    """
    try:
        parent = Chem.Mol(mol)
        Chem.AssignStereochemistry(parent, force=True, cleanIt=True)

        if only_unassigned:
            centers = Chem.FindMolChiralCenters(
                parent, includeUnassigned=True, includeCIP=False
            )
            n_unassigned = sum(1 for _, tag in centers if tag == "?")
            if n_unassigned == 0:
                smi = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
                return [(parent, smi)]

        # Configure enumeration options
        opts = StereoEnumerationOptions()
        opts.unique = True  # Deduplicates based on CXSMILES
        opts.maxIsomers = max_stereoisomers
        opts.onlyUnassigned = only_unassigned
        opts.tryEmbedding = True

        # Enumerate stereoisomers (always returns at least the original molecule)
        stereo_isomers = list(EnumerateStereoisomers(parent, options=opts))

        # Convert to list of (mol, smiles) tuples
        # No sanitization needed for enumerated stereoisomers
        results = []

        for stereo_mol in stereo_isomers:
            try:
                smi = Chem.MolToSmiles(stereo_mol, canonical=True, isomericSmiles=True)
                results.append((Chem.Mol(stereo_mol), smi))
            except Exception as e:
                _get_logger().warning(f"Error processing stereoisomer: {e}")
                continue

        return results

    except Exception as e:
        _get_logger().error(f"Error enumerating stereoisomers: {e}")
        # Return original molecule if enumeration fails
        try:
            parent = Chem.Mol(mol)
            Chem.AssignStereochemistry(parent, force=True, cleanIt=True)
            orig_smiles = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
            return [(parent, orig_smiles)]
        except Exception:
            return []


def enumerate_protonation_states(
    mol: Chem.Mol,
    pH_range: Tuple[float, float] = (6.4, 8.4),
    precision: float = 0.5,
    ligand_id: Optional[str] = None,
) -> List[Tuple[Chem.Mol, str, int]]:
    """
    Enumerate protonation states using Dimorphite-DL.

    States are enumerated for pH within [min_pH, max_pH]. Default (6.4, 8.4) is
    centered around physiological pH while avoiding over-expansion.

    Args:
        mol: RDKit molecule object
        pH_range: Tuple of (min_pH, max_pH) for enumeration (default: (6.4, 8.4))
        precision: pKa precision factor (controls ionization sampling, default: 0.5)
        ligand_id: Optional parent ligand identifier for logging/debugging.

    Returns:
        List of tuples: [(mol, canonical_smiles, formal_charge), ...] for each unique protonation state
    """
    ph_min, ph_max = pH_range

    try:
        from dimorphite_dl import protonate_smiles
    except ImportError as e:
        raise ImportError(
            "dimorphite-dl is required for protonation state enumeration."
        ) from e

    try:
        # Step 1: Get canonical SMILES for input
        input_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        log_target = ligand_id or input_smiles

        # Step 2: Call Dimorphite-DL to protonate/deprotonate
        protonated_smiles = protonate_smiles(
            input_smiles,
            ph_min=ph_min,
            ph_max=ph_max,
            precision=precision
        )
        if protonated_smiles:
            _get_logger().info(
                "Dimorphite generated %d raw protomer candidate(s) for %s.",
                len(protonated_smiles),
                log_target,
            )

        # Step 3: Convert to RDKit Mol and dedupe
        unique: Dict[str, Tuple[Chem.Mol, int]] = {}
        invalid_candidates: set = set()
        rejected_reasons: Dict[str, str] = {}
        for smi in protonated_smiles:
            parsed = parse_smiles_with_kekulize_rescue(smi, prefer_restandardize=True)
            rdkit_m = parsed.mol
            if rdkit_m is None:
                if smi not in invalid_candidates:
                    _get_logger().warning(
                        "Rejected protonation candidate for "
                        f"{log_target}: candidate={smi}; reason={parsed.reason}; "
                        f"sanitize_step={parsed.sanitize_failure_step}"
                    )
                    invalid_candidates.add(smi)
                    rejected_reasons[smi] = parsed.reason
                continue
            try:
                can = Chem.MolToSmiles(rdkit_m, canonical=True, isomericSmiles=True)

                if can not in unique:
                    if is_kekulize_rescue_mode(parsed.parse_mode):
                        _get_logger().info(
                            "Accepted protonation candidate for %s via kekulize-only rescue "
                            "(candidate=%s, mode=%s).",
                            log_target,
                            smi,
                            parsed.parse_mode,
                        )
                    unique[can] = (rdkit_m, Chem.GetFormalCharge(rdkit_m))
            except Exception as e:
                if smi not in invalid_candidates:
                    reason = str(e)
                    _get_logger().warning(
                        "Rejected protonation candidate for "
                        f"{log_target}: candidate={smi}; reason={reason}"
                    )
                    invalid_candidates.add(smi)
                    rejected_reasons[smi] = reason
                continue

        # Step 4: Preserve Dimorphite output order (first-seen canonical states).
        results = [(mol_obj, smi, charge) for smi, (mol_obj, charge) in unique.items()]
        _get_logger().info(
            "Protomer filtering for %s: raw=%d, accepted_unique=%d, rejected=%d.",
            log_target,
            len(protonated_smiles),
            len(results),
            len(rejected_reasons),
        )
        if results:
            accepted_debug = ", ".join(
                [f"{smi} (q={charge:+d})" for _mol_obj, smi, charge in results]
            )
            _get_logger().info(
                "Accepted protomer states for %s: %s",
                log_target,
                accepted_debug,
            )

        if not results:
            # Fallback to original molecule
            try:
                orig_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                return [(Chem.Mol(mol), orig_smiles, Chem.GetFormalCharge(mol))]
            except Exception:
                return []

        return results

    except Exception as e:
        _get_logger().warning(f"Error enumerating protonation states with Dimorphite-DL: {e}")
        # Fallback to original molecule
        try:
            orig_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            return [(Chem.Mol(mol), orig_smiles, Chem.GetFormalCharge(mol))]
        except Exception:
            return []




def enumerate_ligand_microstates(
    smiles: str,
    original_id: str,
    do_enumerate_tautomers: bool = True,
    do_enumerate_protonation: bool = True,
    do_enumerate_stereoisomers: bool = True,
    enumerate_all_stereocenters: bool = False,
    pH_range: Tuple[float, float] = (6.4, 8.4),
    protonation_precision: float = 0.5,
    max_generated_tautomers: Optional[int] = 64,
    top_tautomers_per_protomer: int = 2,
    tautomer_energy_window_kcal: float = 3.0,
    max_protomers: int = 16,
    max_stereoisomers: int = 16,
    max_unassigned_stereocenters: int = 2,
    max_total_microstates: int = 64,
    input_mol: Optional[Chem.Mol] = None,
) -> pd.DataFrame:
    """
    Enumerate all microstates (protomers, tautomers, stereoisomers) for a ligand
    via combinatorial enumeration: protomers × tautomers × stereoisomers. Output
    is original plus final combinatorial microstates only (no intermediate layers).
    Global deduplication uses RDKit canonical SMILES so equivalent SMILES from
    other toolchains are treated as the same.

    Args:
        smiles: SMILES string of the ligand
        original_id: Original ligand ID
        do_enumerate_tautomers: Whether to enumerate tautomers (default: True)
        do_enumerate_protonation: Whether to enumerate protonation states (default: True)
        do_enumerate_stereoisomers: Whether to enumerate stereoisomers (default: True)
        enumerate_all_stereocenters: If True, enumerate all stereocenters (assigned and
            unassigned). If False, enumerate only unassigned stereocenters.
        pH_range: pH range for protonation state enumeration (default: (6.4, 8.4)).
        protonation_precision: Dimorphite precision parameter controlling protonation expansion.
        max_generated_tautomers: Optional hard cap for generated tautomers per protomer.
            Set to None to disable this safety cap.
        top_tautomers_per_protomer: Number of low-energy tautomer candidates retained.
        tautomer_energy_window_kcal: Energy window around best tautomer for retention.
        max_protomers: Maximum number of protomers per ligand (default: 16)
        max_stereoisomers: Maximum number of stereoisomers per ligand (default: 16)
        max_unassigned_stereocenters: Skip stereoisomer expansion when unassigned centers exceed this value.
        max_total_microstates: Maximum total number of records (including original) per
            ligand to prevent combinatorial blow-up.
        input_mol: Optional already-parsed RDKit mol to avoid reparsing fragile SMILES.

    Returns:
        DataFrame with columns: ID, SMILES, MolWt, Variant_Type, Variant_ID, Parent_ID
    """
    if input_mol is not None:
        mol = Chem.Mol(input_mol)
    else:
        parsed = parse_smiles_with_kekulize_rescue(smiles, prefer_restandardize=True)
        mol = parsed.mol
        if mol is None:
            _get_logger().warning(
                f"Invalid SMILES for {original_id}: {smiles} "
                f"(reason: {parsed.reason}; sanitize_step={parsed.sanitize_failure_step})"
            )
            return pd.DataFrame(
                columns=["ID", "SMILES", "MolWt", "Variant_Type", "Variant_ID", "Parent_ID"]
            )
        if is_kekulize_rescue_mode(parsed.parse_mode):
            _get_logger().info(
                "Using kekulize-only rescued parse for %s (mode=%s).",
                original_id,
                parsed.parse_mode,
            )

    orig_smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    orig_mw = Descriptors.MolWt(mol)

    records: list = []
    variant_counter: dict = {}

    # Original
    original_variant_id = "original_0"
    records.append(
        {
            "ID": _variant_id_to_record_id(original_id, original_variant_id),
            "SMILES": orig_smiles,
            "MolWt": orig_mw,
            "Variant_Type": "original",
            "Variant_ID": original_variant_id,
            "Parent_ID": original_id,
        }
    )

    # Level 1: protomers (always include original state)
    protomer_states: List[Tuple[Chem.Mol, str, List[str]]] = [(mol, orig_smiles, [])]
    if do_enumerate_protonation:
        try:
            prot_list = enumerate_protonation_states(
                mol,
                pH_range=pH_range,
                precision=protonation_precision,
                ligand_id=original_id,
            )
            n = 0
            for prot_mol, prot_smiles, _formal_charge in prot_list:
                if prot_smiles == orig_smiles:
                    continue
                if n >= max_protomers:
                    _get_logger().warning(
                        f"Protomer enumeration truncated for {original_id} "
                        f"at max_protomers={max_protomers}"
                    )
                    break
                protomer_states.append((prot_mol, prot_smiles, ["protomer"]))
                n += 1
        except ImportError:
            raise
        except Exception as e:
            _get_logger().warning(f"Error enumerating protomers for {original_id}: {e}")

    # Level 2: for each protomer, enumerate tautomers (and keep protomer baseline)
    level2: List[Tuple[Chem.Mol, str, List[str], float]] = []
    for prot_mol, prot_smiles, path in protomer_states:
        level2.append((prot_mol, prot_smiles, path, float("inf")))
        if not do_enumerate_tautomers:
            continue
        try:
            taut_list = enumerate_tautomers(
                prot_mol,
                top_tautomers_per_protomer=top_tautomers_per_protomer,
                energy_window_kcal=tautomer_energy_window_kcal,
                hard_max_generated_tautomers=max_generated_tautomers,
            )
            for taut_mol, taut_smiles, taut_e in taut_list:
                if taut_smiles == prot_smiles:
                    continue
                level2.append((taut_mol, taut_smiles, path + ["tautomer"], taut_e))
        except Exception as e:
            _get_logger().warning(f"Error enumerating tautomers for {original_id}: {e}")

    # Level 3: for each level2, stereoisomers (skip when same as parent)
    level3: List[Tuple[Chem.Mol, str, List[str], float]] = []
    if do_enumerate_stereoisomers:
        for parent_mol, parent_smiles, path, parent_e in level2:
            try:
                centers = Chem.FindMolChiralCenters(
                    parent_mol, includeUnassigned=True, includeCIP=False
                )
                n_unassigned = sum(1 for _, tag in centers if tag == "?")
                if n_unassigned > max_unassigned_stereocenters:
                    _get_logger().info(
                        f"Skipping stereo expansion for {original_id}: "
                        f"{n_unassigned} unassigned centers exceeds "
                        f"max_unassigned_stereocenters={max_unassigned_stereocenters}"
                    )
                    level3.append((parent_mol, parent_smiles, path, parent_e))
                    continue

                stereo_list = enumerate_stereoisomers(
                    parent_mol,
                    only_unassigned=(not enumerate_all_stereocenters),
                    max_stereoisomers=max_stereoisomers,
                )
                for stereo_mol, stereo_smiles in stereo_list:
                    if stereo_smiles == parent_smiles:
                        continue
                    level3.append((stereo_mol, stereo_smiles, path + ["stereoisomer"], parent_e))
            except Exception as e:
                _get_logger().warning(f"Error enumerating stereoisomers for {original_id}: {e}")
    else:
        level3 = list(level2)

    # Final pooled selection avoids level-order bias when max_total_microstates is hit.
    best_by_smiles: Dict[str, Tuple[float, Chem.Mol, List[str]]] = {}
    pooled = level2 + level3
    for cand_mol, cand_smiles, path, energy_hint in pooled:
        key_smi = _canonicalize_smiles(cand_smiles)
        if key_smi is None or key_smi == orig_smiles:
            continue
        if energy_hint == float("inf"):
            energy = _estimate_mol_energy_kcal(cand_mol)
        else:
            energy = energy_hint
        prev = best_by_smiles.get(key_smi)
        if prev is None or energy < prev[0]:
            best_by_smiles[key_smi] = (energy, cand_mol, path)

    ranked_variants = sorted(
        best_by_smiles.items(),
        key=lambda item: (item[1][0], len(item[1][2]), item[0]),
    )

    for key_smi, (_energy, cand_mol, path) in ranked_variants:
        if len(records) >= max_total_microstates:
            _get_logger().warning(
                f"Reached max_total_microstates={max_total_microstates} for {original_id}; "
                "truncating additional variants."
            )
            break
        key = ",".join(path)
        variant_counter[key] = variant_counter.get(key, 0) + 1
        variant_id = f"{key}_{variant_counter[key]}"
        records.append(
            {
                "ID": _variant_id_to_record_id(original_id, variant_id),
                "SMILES": key_smi,
                "MolWt": Descriptors.MolWt(cand_mol),
                "Variant_Type": key,
                "Variant_ID": variant_id,
                "Parent_ID": original_id,
            }
        )

    df = pd.DataFrame(records)

    if len(df) > 0:
        _get_logger().info(
            f"Generated {len(df)} microstates for {original_id} "
            f"(original + {len(df)-1} variants)"
        )

    return df
