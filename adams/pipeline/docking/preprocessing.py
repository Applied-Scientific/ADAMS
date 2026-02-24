"""
Preprocessing for docking pipelines: input validation, receptor conversion,
ligand loading, and docking center setup.
"""

import os

import pandas as pd

from ..charge_model import validate_charge_model
from .atom_types import sanitize_pdbqt_atom_types
from .utils import (
    convert_receptor_to_pdbqt,
    generate_grid,
    get_ligand_com_from_pdb,
    get_pdbqt_bounds,
)


LIGAND_STYLE_PDBQT_TAGS = {
    "ROOT",
    "ENDROOT",
    "BRANCH",
    "ENDBRANCH",
    "TORSDOF",
}


def _find_ligand_style_pdbqt_tag(pdbqt_path: str):
    """Return first ligand-style PDBQT tag found, or None."""
    with open(pdbqt_path, "r", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            tag = stripped.split(maxsplit=1)[0]
            if tag in LIGAND_STYLE_PDBQT_TAGS:
                return tag
    return None


def _infer_sibling_protonated_pdb(pdbqt_path: str):
    """Infer sibling protonated PDB path from a receptor PDBQT path."""
    receptor_dir = os.path.dirname(os.path.abspath(pdbqt_path))
    stem = os.path.splitext(os.path.basename(pdbqt_path))[0]

    candidates = [os.path.join(receptor_dir, f"{stem}.pdb")]
    if "__q_" in stem:
        source_stem = stem.split("__q_", 1)[0]
        candidates.insert(0, os.path.join(receptor_dir, f"{source_stem}.pdb"))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if not os.path.exists(candidate):
            continue
        base = os.path.splitext(candidate)[0]
        if "_protonated" in os.path.basename(base) or os.path.exists(base + ".pqr"):
            return candidate
    return None


def _ensure_valid_receptor_pdbqt(pipeline):
    """
    Ensure receptor PDBQT is rigid-receptor format (no ROOT/BRANCH/TORSDOF).
    Attempt recovery by regenerating from sibling protonated PDB when possible.
    """
    offending_tag = _find_ligand_style_pdbqt_tag(pipeline.receptor)
    if offending_tag is None:
        return

    sibling_pdb = _infer_sibling_protonated_pdb(pipeline.receptor)
    if sibling_pdb:
        pipeline.logger.warning(
            "Detected ligand-style tag '%s' in receptor PDBQT (%s). "
            "Attempting regeneration from sibling protonated PDB: %s",
            offending_tag,
            pipeline.receptor,
            sibling_pdb,
        )

        regenerated = convert_receptor_to_pdbqt(
            sibling_pdb,
            protonate=False,
            pH=pipeline.pH,
            charge_model=pipeline.charge_model,
        )

        regenerated_tag = _find_ligand_style_pdbqt_tag(regenerated)
        if (
            regenerated_tag is not None
            and os.path.abspath(regenerated) == os.path.abspath(pipeline.receptor)
        ):
            pipeline.logger.warning(
                "Stale invalid receptor PDBQT detected at %s. "
                "Removing it and regenerating from %s.",
                regenerated,
                sibling_pdb,
            )
            try:
                os.remove(regenerated)
            except OSError:
                pass
            regenerated = convert_receptor_to_pdbqt(
                sibling_pdb,
                protonate=False,
                pH=pipeline.pH,
                charge_model=pipeline.charge_model,
            )
            regenerated_tag = _find_ligand_style_pdbqt_tag(regenerated)

        if regenerated_tag is None:
            pipeline.receptor = regenerated
            pipeline.logger.info(
                "Recovered receptor PDBQT by regenerating from %s", sibling_pdb
            )
            return

    raise ValueError(
        "Receptor PDBQT appears to be ligand-style (contains ROOT/BRANCH/TORSDOF tags), "
        "which is invalid for rigid receptor docking. Use a protonated receptor .pdb from "
        "run_protonate_receptor and let docking convert it to receptor PDBQT."
    )


def _audit_and_sanitize_ligand_pdbqts(pipeline, ligands: pd.DataFrame) -> pd.DataFrame:
    """Validate/normalize ligand atom types and prune unresolved entries before docking."""
    preproc_lig_dir = os.path.join(pipeline.out_folder, "preprocessing", "ligands")
    os.makedirs(preproc_lig_dir, exist_ok=True)

    audit_csv = os.path.join(preproc_lig_dir, "atom_type_audit.csv")
    skipped_csv = os.path.join(pipeline.dir_structure["summaries"], "skipped_ligands.csv")

    audit_rows = []
    skipped_rows = []
    kept_indices = []

    for idx, row in ligands.iterrows():
        ligand_id = str(row.get("ID", f"ligand_{idx}"))
        parent_id = str(row.get("Parent_ID", ligand_id))
        variant_type = str(row.get("Variant_Type", "original"))
        variant_id = str(row.get("Variant_ID", ""))
        conformer_index = row.get("Conformer_Index", "")
        pdbqt_file = row["PDBQT_File"]

        result = sanitize_pdbqt_atom_types(
            src_path=pdbqt_file,
            strict_unresolved=False,
        )
        unresolved = result["unresolved_details"]
        replacement_counts = result["replacement_counts"]
        replacement_summary = "; ".join(
            f"{k}:{v}" for k, v in sorted(replacement_counts.items())
        )
        unresolved_summary = "; ".join(
            f"line {d['line_number']}:{d['atom_type']} (hint={d['element_hint']})"
            for d in unresolved[:10]
        )

        action = "ok"
        reason = ""

        if unresolved:
            action = "unresolved"
            reason = (
                f"unresolved_atom_types={len(unresolved)}"
                + (f"; {unresolved_summary}" if unresolved_summary else "")
            )
            skipped_rows.append(
                {
                    "Stage": "preflight_atom_type_audit",
                    "Ligand_Index": idx,
                    "ID": ligand_id,
                    "Parent_ID": parent_id,
                    "Variant_ID": variant_id,
                    "Variant_Type": variant_type,
                    "Conformer_Index": conformer_index,
                    "PDBQT_File": pdbqt_file,
                    "Reason": reason,
                    "Worker_ID": "",
                    "Site_ID": "",
                    "Chunk_ID": "",
                }
            )
            pipeline.logger.warning(
                "Skipping ligand %s during atom-type preflight audit: %s",
                ligand_id,
                reason,
            )
        else:
            kept_indices.append(idx)
            if result["changed_entries"] > 0:
                action = "normalized"

        audit_rows.append(
            {
                "Ligand_Index": idx,
                "ID": ligand_id,
                "Parent_ID": parent_id,
                "Variant_ID": variant_id,
                "Variant_Type": variant_type,
                "Conformer_Index": conformer_index,
                "PDBQT_File": pdbqt_file,
                "Action": action,
                "Atom_Lines": result["atom_line_count"],
                "Changed_Atom_Entries": result["changed_entries"],
                "Replacement_Map": replacement_summary,
                "Unresolved_Count": len(unresolved),
                "Unresolved_Details": unresolved_summary,
                "Reason": reason,
            }
        )

    pd.DataFrame(audit_rows).to_csv(audit_csv, index=False)

    skipped_columns = [
        "Stage",
        "Ligand_Index",
        "ID",
        "Parent_ID",
        "Variant_ID",
        "Variant_Type",
        "Conformer_Index",
        "PDBQT_File",
        "Reason",
        "Worker_ID",
        "Site_ID",
        "Chunk_ID",
    ]
    pd.DataFrame(skipped_rows, columns=skipped_columns).to_csv(
        skipped_csv,
        index=False,
    )

    total = len(ligands)
    normalized = sum(1 for row in audit_rows if row["Action"] == "normalized")
    unresolved_count = len(skipped_rows)
    ok_count = total - normalized - unresolved_count
    pipeline.logger.info(
        "Atom-type preflight audit complete: total=%d, ok=%d, normalized=%d, unresolved=%d. Audit: %s",
        total,
        ok_count,
        normalized,
        unresolved_count,
        audit_csv,
    )
    if unresolved_count:
        pipeline.logger.warning(
            "Skipped %d ligand(s) with unresolved atom types. Details: %s",
            unresolved_count,
            skipped_csv,
        )

    kept_ligands = ligands.loc[kept_indices].reset_index(drop=True)
    if kept_ligands.empty:
        raise ValueError(
            "No ligands remain after PDBQT atom-type preflight audit. "
            "Check preprocessing atom typing and rescued ligand entries."
        )
    return kept_ligands


def preprocess(pipeline):
    """
    Validate inputs, convert receptor to PDBQT, load ligands, set docking centers.
    Mutates pipeline with: receptor, ligand_pdbqt_files, lignames, num_ligands,
    molweight, and calls set_docking_centers(pipeline).
    """
    if not os.path.exists(pipeline.input_data):
        raise FileNotFoundError(
            f"Input data file not found: {pipeline.input_data}"
        )
    if not os.path.exists(pipeline.receptor):
        raise FileNotFoundError(f"Receptor file not found: {pipeline.receptor}")
    pipeline.charge_model = validate_charge_model(pipeline.charge_model)

    if pipeline.receptor.endswith(".pdb"):
        base = os.path.splitext(pipeline.receptor)[0]
        has_pqr = os.path.exists(base + ".pqr")
        is_protonated = "_protonated" in os.path.basename(base)
        if not (is_protonated or has_pqr):
            raise ValueError(
                "Receptor must be protonated via run_protonate_receptor before docking. "
                "Expected *_protonated.pdb (and optional matching .pqr)."
            )
        pipeline.receptor = convert_receptor_to_pdbqt(
            pipeline.receptor,
            protonate=False,
            pH=pipeline.pH,
            charge_model=pipeline.charge_model,
        )
    elif not pipeline.receptor.endswith(".pdbqt"):
        raise ValueError(
            f"Receptor file must be either .pdb or .pdbqt format, got: {pipeline.receptor}"
        )

    _ensure_valid_receptor_pdbqt(pipeline)

    os.makedirs(pipeline.out_folder, exist_ok=True)
    ligands = pd.read_csv(pipeline.input_data)

    if "PDBQT_File" not in ligands.columns:
        raise ValueError(
            "Input CSV must contain a PDBQT_File column. "
            "Please run the preprocessing agent first: run_standardize_ligand_data, then "
            "run_smiles_to_pdbqt (and optionally run_ligand_preprocessing in between). "
            "Use the returned mapping CSV as input_data for docking."
        )
    if "ID" not in ligands.columns:
        raise ValueError(
            "Input CSV must contain 'ID' column with ligand identifiers."
        )

    if "Charge_Model" in ligands.columns:
        model_series = ligands["Charge_Model"].dropna().astype(str).str.strip().str.lower()
        if model_series.empty:
            pipeline.logger.warning(
                "Input CSV has a Charge_Model column, but all values are empty. "
                "Skipping charge-model consistency check."
            )
        else:
            unique_models = sorted(set(model_series.tolist()))
            if len(unique_models) > 1:
                raise ValueError(
                    f"Input CSV contains multiple ligand charge models: {unique_models}. "
                    "Regenerate ligand PDBQT with a single charge_model before docking."
                )
            ligand_model = unique_models[0]
            if ligand_model != pipeline.charge_model:
                raise ValueError(
                    "Ligand/receptor charge-model mismatch: "
                    f"ligands were prepared with '{ligand_model}', "
                    f"but docking is running with charge_model='{pipeline.charge_model}'. "
                    "Use the same charge_model in preprocessing and docking."
                )
            pipeline.logger.info(
                f"Charge-model consistency check passed: {pipeline.charge_model}"
            )
    else:
        pipeline.logger.warning(
            "Input CSV has no Charge_Model column. "
            "Cannot verify ligand/receptor charge-model consistency."
        )

    missing_files = [f for f in ligands["PDBQT_File"].tolist() if not os.path.exists(f)]
    if missing_files:
        error_msg = f"Missing {len(missing_files)} PDBQT file(s):\n"
        for f in missing_files[:10]:
            error_msg += f"  - {f}\n"
        if len(missing_files) > 10:
            error_msg += f"  ... and {len(missing_files) - 10} more"
        raise FileNotFoundError(error_msg.strip())

    ligands = _audit_and_sanitize_ligand_pdbqts(pipeline, ligands)

    pipeline.ligand_pdbqt_files = ligands["PDBQT_File"].tolist()
    pipeline.lignames = ligands["ID"]
    pipeline.parent_ids = (
        ligands["Parent_ID"] if "Parent_ID" in ligands.columns else ligands["ID"]
    )
    pipeline.variant_types = (
        ligands["Variant_Type"]
        if "Variant_Type" in ligands.columns
        else pd.Series(["original"] * len(ligands))
    )
    pipeline.variant_ids = (
        ligands["Variant_ID"]
        if "Variant_ID" in ligands.columns
        else pd.Series([""] * len(ligands))
    )
    pipeline.conformer_indices = (
        ligands["Conformer_Index"]
        if "Conformer_Index" in ligands.columns
        else pd.Series([0] * len(ligands))
    )
    for candidate in ("SMILES", "inSMILES"):
        if candidate in ligands.columns:
            pipeline.ligand_smiles = ligands[candidate]
            break
    else:
        pipeline.ligand_smiles = pd.Series([None] * len(ligands))
    pipeline.num_ligands = len(pipeline.ligand_pdbqt_files)
    if pipeline.num_ligands == 0:
        raise ValueError(
            "No ligands available for docking (input CSV has zero rows). "
            "Check upstream filtering/enumeration settings or input_data path."
        )

    pipeline.logger.info(
        f"Loaded {pipeline.num_ligands} PDBQT files for docking"
    )

    from .utils import get_molweight_from_pdbqt

    pipeline.logger.info("Calculating molecular weights from PDBQT files...")
    molweights = [
        get_molweight_from_pdbqt(f) for f in pipeline.ligand_pdbqt_files
    ]
    pipeline.molweight = pd.Series(molweights)
    pipeline.logger.info(
        f"Calculated molecular weights (range: {pipeline.molweight.min():.1f} - {pipeline.molweight.max():.1f} Da)"
    )

    set_docking_centers(pipeline)


def set_docking_centers(pipeline):
    """
    Set docking centers from file, complex, manual list, or search grid.
    Mutates pipeline.docking_centers and pipeline.num_pockets.
    """
    if pipeline.docking_centers_file and pipeline.docking_centers_file.strip():
        if not os.path.exists(pipeline.docking_centers_file):
            raise FileNotFoundError(
                f"Docking centers file not found: {pipeline.docking_centers_file}"
            )
        df_centers = pd.read_csv(pipeline.docking_centers_file)
        col_names = df_centers.columns.tolist()
        if (
            "centroid_x" in col_names
            and "centroid_y" in col_names
            and "centroid_z" in col_names
        ):
            x_col, y_col, z_col = "centroid_x", "centroid_y", "centroid_z"
        elif (
            "COM_x" in col_names
            and "COM_y" in col_names
            and "COM_z" in col_names
        ):
            x_col, y_col, z_col = "COM_x", "COM_y", "COM_z"
        elif "x" in col_names and "y" in col_names and "z" in col_names:
            x_col, y_col, z_col = "x", "y", "z"
        else:
            if df_centers.shape[1] < 4:
                raise ValueError(
                    f"Could not find coordinate columns and file has fewer than 4 columns: got {df_centers.shape[1]}"
                )
            pipeline.logger.warning(
                f"Using columns 1-3 as coordinates. Column names: {col_names}"
            )
            centers_data = df_centers.iloc[:, 1:4].values
            pipeline.docking_centers = [list(row) for row in centers_data]
            pipeline.num_pockets = len(pipeline.docking_centers)
            pipeline.logger.info(
                f"Read {pipeline.num_pockets} docking centers from {pipeline.docking_centers_file}"
            )
            return
        centers_data = df_centers[[x_col, y_col, z_col]].values
        pipeline.docking_centers = [list(row) for row in centers_data]
        pipeline.num_pockets = len(pipeline.docking_centers)
        pipeline.logger.info(
            f"Read {pipeline.num_pockets} docking centers from {pipeline.docking_centers_file} using columns: {x_col}, {y_col}, {z_col}"
        )
        return

    if pipeline.mode == "production":
        if pipeline.complex:
            parts = pipeline.complex.split(",")
            complex_pdb = parts[0].strip()
            if not os.path.exists(complex_pdb):
                raise FileNotFoundError(
                    f"Complex PDB file not found: {complex_pdb}. "
                    f"Please verify the path is correct."
                )
            lig_selectors = [selector.strip() for selector in parts[1:] if selector.strip()]
            if len(lig_selectors) < pipeline.num_pockets:
                raise ValueError(
                    f"Not enough ligand selectors provided: expected at least {pipeline.num_pockets}, got {len(lig_selectors)}"
                )
            if len(lig_selectors) > pipeline.num_pockets:
                lig_selectors = lig_selectors[: pipeline.num_pockets]
            pipeline.logger.info(f"PDB file: {complex_pdb}")
            pipeline.logger.info(
                f"Input {len(lig_selectors)} ligand selectors: {lig_selectors}"
            )
            pipeline.docking_centers = [
                get_ligand_com_from_pdb(complex_pdb, lig_selectors[i])
                for i in range(pipeline.num_pockets)
            ]
        else:
            if pipeline.docking_centers is None:
                raise ValueError(
                    "Production docking requires explicit docking centers. "
                    "Provide one of: docking_centers_file, complex='complex.pdb,RES[:CHAIN[:RESSEQ]]', "
                    "or manual docking_centers=[x1,y1,z1,...]."
                )
            else:
                if len(pipeline.docking_centers) != pipeline.num_pockets * 3:
                    raise ValueError(
                        f"docking_centers requires {pipeline.num_pockets * 3} floats "
                        f"({pipeline.num_pockets} pockets × 3 coordinates), but got {len(pipeline.docking_centers)}. "
                        f"Example format: [x1, y1, z1, x2, y2, z2] for 2 pockets."
                    )
                pipeline.docking_centers = [
                    pipeline.docking_centers[i : i + 3]
                    for i in range(0, len(pipeline.docking_centers), 3)
                ]
    else:  # mode == "search"
        boundary = get_pdbqt_bounds(pipeline.receptor)
        pipeline.docking_centers = generate_grid(
            boundary,
            box_size=pipeline.search_gridsize,
            margin=pipeline.search_margin,
        )
        pipeline.num_pockets = len(pipeline.docking_centers)
        pipeline.logger.info(
            f"Search dock over {len(pipeline.docking_centers)} grid centers..."
        )

    pipeline.logger.debug(f"Docking centers: {pipeline.docking_centers}")
