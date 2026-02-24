"""
Pose collection for docking pipelines: aggregate pose files into model_data pkl
and write summary CSVs (best per grid, top N, etc.).
"""

import os
import pickle
import re
import shutil

import numpy as np
import pandas as pd

from .utils import get_all_model_centers, write_all_centers_pdb


class PoseCollectionMixin:
    """
    Mixin that provides _collectposes and _save_best_search_docking_poses.
    Expects pipeline to have: dir_structure, mode, logger, lignames,
    parent_ids, variant_types, variant_ids, conformer_indices, molweight.
    """

    def _get_pose_file_path(self, lig_idx, grid_idx):
        """Return the expected pose file path for a ligand/grid combination."""
        if self.mode == "search":
            pose_name = f"ligand_{lig_idx}_grid_{grid_idx}_docked.pdbqt"
        else:
            pose_name = f"ligand_{lig_idx}_pocket_{grid_idx}_docked.pdbqt"
        return os.path.join(self.dir_structure["poses"], pose_name)

    @staticmethod
    def _sanitize_pose_token(value):
        """Convert arbitrary token values into filesystem-safe pose filename parts."""
        if pd.isna(value):
            return "na"
        token = str(value).strip()
        token = token.replace(",", "-")
        token = re.sub(r"[^A-Za-z0-9._+-]+", "_", token)
        token = token.strip("._")
        return token if token else "na"

    def _create_named_pose_copies(self, df_best):
        """
        Create readable pose-file copies for easier visual inspection and
        return df_best with an added Pose_PDBQT_File_Named column.
        """
        if df_best.empty:
            return df_best

        named_dir = os.path.join(self.dir_structure["root"], "poses_named")
        os.makedirs(named_dir, exist_ok=True)

        df_named = df_best.copy()
        named_paths = []

        for _, row in df_named.iterrows():
            lig_idx = int(row["ligand_id"])
            grid_id = int(row["grid_id"])
            source_pose = self._get_pose_file_path(lig_idx, grid_id)

            parent_id = self._sanitize_pose_token(row.get("Parent_ID"))
            variant_id = self._sanitize_pose_token(row.get("Variant_ID"))
            variant_type = self._sanitize_pose_token(row.get("Variant_Type"))
            conf_idx = self._sanitize_pose_token(row.get("Conformer_Index"))
            pose_name = (
                f"{parent_id}__{variant_type}__{variant_id}"
                f"__conf{conf_idx}__lig{lig_idx}_pocket{grid_id}_docked.pdbqt"
            )
            named_pose = os.path.join(named_dir, pose_name)

            if os.path.exists(source_pose):
                try:
                    if os.path.exists(named_pose):
                        os.remove(named_pose)
                    try:
                        os.link(source_pose, named_pose)
                    except OSError:
                        shutil.copy2(source_pose, named_pose)
                    named_paths.append(named_pose)
                except Exception as e:
                    self.logger.warning(
                        f"Could not create named pose copy for ligand {lig_idx}, "
                        f"grid {grid_id}: {e}"
                    )
                    named_paths.append("")
            else:
                self.logger.warning(
                    f"Source pose file missing for ligand {lig_idx}, grid {grid_id}: "
                    f"{source_pose}"
                )
                named_paths.append("")

        df_named["Pose_PDBQT_File_Named"] = named_paths
        self.logger.info(
            f"Saved readable production pose files to {named_dir}"
        )
        return df_named

    def _write_failed_combinations_summary(self, failures):
        """
        Write a structured CSV of ligand/grid combinations that produced no pose file
        or no valid affinity records.
        """
        columns = [
            "ligand_id",
            "ID",
            "Parent_ID",
            "Variant_Type",
            "Variant_ID",
            "Conformer_Index",
            "grid_id",
            "Pose_PDBQT_File",
            "failure_reason",
            "details",
        ]
        df_failed = pd.DataFrame(failures, columns=columns)

        failed_csv_summaries = os.path.join(
            self.dir_structure["summaries"], "failed_combinations.csv"
        )
        failed_csv_metadata = os.path.join(
            self.dir_structure["metadata"], "failed_combinations.csv"
        )

        df_failed.to_csv(failed_csv_summaries, index=False)
        # Keep a metadata copy for scripts that already look there.
        if failed_csv_metadata != failed_csv_summaries:
            df_failed.to_csv(failed_csv_metadata, index=False)

        if df_failed.empty:
            self.logger.info(
                f"No failed docking combinations detected (wrote empty file): "
                f"{failed_csv_summaries}"
            )
        else:
            reason_counts = (
                df_failed["failure_reason"].value_counts().to_dict()
                if "failure_reason" in df_failed.columns
                else {}
            )
            self.logger.warning(
                f"Detected {len(df_failed)} failed docking combinations. "
                f"Reasons: {reason_counts}. "
                f"Saved to {failed_csv_summaries}"
            )

    def _write_production_readability_summaries(self, df_best):
        """
        Write compact production summaries that are easier to inspect:
        - best pose per parent ligand (per grid)
        - direct pose manifest with pose file paths for visualization
        """
        if df_best.empty:
            return df_best

        df_best_named = self._create_named_pose_copies(df_best)

        summaries_dir = self.dir_structure["summaries"]

        # 1) Best microstate/conformer per parent ligand (per grid), sorted by affinity.
        df_parent_best = (
            df_best_named.sort_values(by="affinity")
            .groupby(["Parent_ID", "grid_id"], as_index=False)
            .first()
            .sort_values(by="affinity")
        )
        parent_csv = os.path.join(
            summaries_dir, "production_best_per_parent.csv"
        )
        df_parent_best.to_csv(parent_csv, index=False)
        self.logger.info(
            f"Saved best pose per parent ligand to {parent_csv}"
        )

        # 2) Pose manifest for direct loading in visualization tools.
        manifest_cols = [
            "Parent_ID",
            "ID",
            "Variant_Type",
            "Variant_ID",
            "Conformer_Index",
            "grid_id",
            "pose_id",
            "affinity",
            "Ligand_PDBQT_File",
            "Pose_PDBQT_File",
            "Pose_PDBQT_File_Named",
        ]
        available_manifest_cols = [
            col for col in manifest_cols if col in df_best_named.columns
        ]
        manifest_csv = os.path.join(
            summaries_dir, "production_pose_manifest.csv"
        )
        (
            df_best_named.loc[:, available_manifest_cols]
            .sort_values(by=["affinity", "Parent_ID", "ID", "grid_id"])
            .to_csv(manifest_csv, index=False)
        )
        self.logger.info(
            f"Saved production pose manifest to {manifest_csv}"
        )
        return df_best_named

    def _collectposes(self, n_lig, n_grid, n_pose):
        """Collect docking poses into model_data pkl and write summary CSVs."""
        model_data = np.zeros((6, n_lig, n_grid, n_pose), dtype=np.float32)
        # Use NaN sentinel for missing affinities so valid 0.0 scores are retained.
        model_data[0, ...] = np.nan
        failures = []
        expected_combinations = n_lig * n_grid
        combinations_with_pose_file = 0
        combinations_with_valid_affinity = 0
        partial_pose_files = 0

        for lig_id in range(n_lig):
            for g_id in range(n_grid):
                file_path = self._get_pose_file_path(lig_id, g_id)
                if not os.path.exists(file_path):
                    failures.append(
                        {
                            "ligand_id": lig_id,
                            "ID": self.lignames.iloc[lig_id],
                            "Parent_ID": self.parent_ids.iloc[lig_id],
                            "Variant_Type": self.variant_types.iloc[lig_id],
                            "Variant_ID": self.variant_ids.iloc[lig_id],
                            "Conformer_Index": self.conformer_indices.iloc[lig_id],
                            "grid_id": g_id,
                            "Pose_PDBQT_File": file_path,
                            "failure_reason": "missing_pose_file",
                            "details": "No pose file generated for this ligand/grid combination.",
                        }
                    )
                    continue

                combinations_with_pose_file += 1
                try:
                    with open(file_path, "r") as f:
                        lines = [
                            line.strip().split()
                            for line in f
                            if line.startswith("REMARK VINA RESULT:")
                        ]
                    lines = [[float(val) for val in line[3:]] for line in lines]
                except Exception as e:
                    failures.append(
                        {
                            "ligand_id": lig_id,
                            "ID": self.lignames.iloc[lig_id],
                            "Parent_ID": self.parent_ids.iloc[lig_id],
                            "Variant_Type": self.variant_types.iloc[lig_id],
                            "Variant_ID": self.variant_ids.iloc[lig_id],
                            "Conformer_Index": self.conformer_indices.iloc[lig_id],
                            "grid_id": g_id,
                            "Pose_PDBQT_File": file_path,
                            "failure_reason": "pose_read_error",
                            "details": str(e),
                        }
                    )
                    continue

                if len(lines) == 0:
                    failures.append(
                        {
                            "ligand_id": lig_id,
                            "ID": self.lignames.iloc[lig_id],
                            "Parent_ID": self.parent_ids.iloc[lig_id],
                            "Variant_Type": self.variant_types.iloc[lig_id],
                            "Variant_ID": self.variant_ids.iloc[lig_id],
                            "Conformer_Index": self.conformer_indices.iloc[lig_id],
                            "grid_id": g_id,
                            "Pose_PDBQT_File": file_path,
                            "failure_reason": "no_affinity_records",
                            "details": "Pose file exists but contains no REMARK VINA RESULT lines.",
                        }
                    )
                    continue

                combinations_with_valid_affinity += 1
                if len(lines) < n_pose:
                    partial_pose_files += 1

                model_centers = get_all_model_centers(file_path)

                for pose_id in range(min(n_pose, len(lines))):
                    affinity, rmsd_lb, rmsd_ub = lines[pose_id]
                    # Handle case where model_centers has fewer entries than poses
                    if pose_id < len(model_centers):
                        com = model_centers[pose_id]
                    else:
                        self.logger.warning(
                            f"Missing COM for pose {pose_id} in {file_path} "
                            f"(found {len(model_centers)} models, expected at least {pose_id + 1})"
                        )
                        com = [0.0, 0.0, 0.0]  # Default COM
                    model_data[0, lig_id, g_id, pose_id] = affinity
                    model_data[1:4, lig_id, g_id, pose_id] = com
                    model_data[4, lig_id, g_id, pose_id] = rmsd_lb
                    model_data[5, lig_id, g_id, pose_id] = rmsd_ub

        self.logger.info(
            "Docking combination summary: "
            f"expected={expected_combinations}, "
            f"pose_files={combinations_with_pose_file}, "
            f"with_valid_affinity={combinations_with_valid_affinity}, "
            f"failures={len(failures)}, "
            f"partial_pose_files={partial_pose_files}"
        )
        self._write_failed_combinations_summary(failures)

        pkl_file = os.path.join(
            self.dir_structure["metadata"], "dock_metadata.pkl"
        )
        with open(pkl_file, "wb") as f:
            pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.logger.info(
            f"Saved model_data with shape {model_data.shape} to {self.dir_structure['metadata']}"
        )

        if self.mode == "search":
            output_csv = os.path.join(
                self.dir_structure["summaries"], "best_docking_centers.csv"
            )
        else:
            output_csv = os.path.join(
                self.dir_structure["summaries"],
                "production_docking_results.csv",
            )
        self._save_best_search_docking_poses(
            pkl_file, output_csv, selecttype="bestPerGrid"
        )

        if self.mode == "search":
            output_csv_top = os.path.join(
                self.dir_structure["summaries"],
                "best_search_docking_centers.csv",
            )
            df_top = self._save_best_search_docking_poses(
                pkl_file, output_csv_top, selecttype="topN", topN=100
            )
            write_all_centers_pdb(
                df_top,
                os.path.join(
                    self.dir_structure["summaries"],
                    "best_search_docking_centers.pdb",
                ),
            )
            self.logger.info(
                f"Search mode: Generated visualization files in {self.dir_structure['summaries']}"
            )

        return output_csv

    def _save_best_search_docking_poses(
        self,
        pkl_file,
        output_csv,
        selecttype="topN",
        topN=10,
        threshold_fraction=0.1,
    ):
        """Select best docking poses (topN, percentile, or bestPerGrid)."""
        with open(pkl_file, "rb") as f:
            results = pickle.load(f)
        affinities = results[0]
        coms = results[1:4]

        flat_affinities = affinities.flatten()
        valid_affinity_mask = np.isfinite(flat_affinities)
        valid_affinities = flat_affinities[valid_affinity_mask]
        cutoff = None
        selected_indices = set()

        if selecttype == "topN":
            if len(valid_affinities) == 0:
                self.logger.warning("No valid affinities found.")
                return pd.DataFrame()
            valid_flat_indices = np.where(valid_affinity_mask)[0]
            ranked_valid = valid_flat_indices[
                np.argsort(flat_affinities[valid_flat_indices])
            ]
            flat_indices = ranked_valid[:topN]
            selected_indices = set(flat_indices)
        elif selecttype == "percentile":
            if len(valid_affinities) == 0:
                self.logger.warning("No valid affinities found.")
                return pd.DataFrame()
            cutoff = np.percentile(
                valid_affinities, threshold_fraction * 100
            )
        elif selecttype == "bestPerGrid":
            pass
        else:
            raise ValueError(
                "selecttype must be 'topN', 'percentile', or 'bestPerGrid'"
            )

        best_records = []
        num_ligands, num_grids, num_poses = affinities.shape

        if selecttype == "bestPerGrid":
            for lig in range(num_ligands):
                for g in range(num_grids):
                    affs = affinities[lig, g, :]
                    valid_mask = np.isfinite(affs)
                    if not np.any(valid_mask):
                        self.logger.debug(
                            f"[Skip] Ligand {lig}: {self.lignames.iloc[lig]}, Grid {g}: no valid affinities"
                        )
                        continue
                    best_p = np.argmin(affs[valid_mask])
                    pose_indices = np.where(valid_mask)[0]
                    p = pose_indices[best_p]
                    aff = affinities[lig, g, p]
                    rec = {
                        "ligand_id": lig,
                        "ID": self.lignames.iloc[lig],
                        "Parent_ID": self.parent_ids.iloc[lig],
                        "Variant_Type": self.variant_types.iloc[lig],
                        "Variant_ID": self.variant_ids.iloc[lig],
                        "Conformer_Index": self.conformer_indices.iloc[lig],
                        "Ligand_PDBQT_File": self.ligand_pdbqt_files[lig],
                        "Pose_PDBQT_File": self._get_pose_file_path(lig, g),
                        "grid_id": g,
                        "pose_id": p,
                        "affinity": aff,
                        "COM_x": coms[0, lig, g, p],
                        "COM_y": coms[1, lig, g, p],
                        "COM_z": coms[2, lig, g, p],
                        "MolWt": self.molweight.iloc[lig],
                    }
                    if hasattr(self, "ligand_smiles"):
                        smi = self.ligand_smiles.iloc[lig]
                        rec["SMILES"] = None if pd.isna(smi) else smi
                    best_records.append(rec)
        else:
            for lig in range(num_ligands):
                for g in range(num_grids):
                    for p in range(num_poses):
                        aff = affinities[lig, g, p]
                        if not np.isfinite(aff):
                            continue
                        flat_idx = (
                            (lig * num_grids * num_poses)
                            + (g * num_poses)
                            + p
                        )
                        if (
                            selecttype == "topN"
                            and flat_idx in selected_indices
                        ) or (
                            selecttype == "percentile" and aff <= cutoff
                        ):
                            rec = {
                                "ligand_id": lig,
                                "ID": self.lignames.iloc[lig],
                                "Parent_ID": self.parent_ids.iloc[lig],
                                "Variant_Type": self.variant_types.iloc[lig],
                                "Variant_ID": self.variant_ids.iloc[lig],
                                "Conformer_Index": self.conformer_indices.iloc[lig],
                                "Ligand_PDBQT_File": self.ligand_pdbqt_files[lig],
                                "Pose_PDBQT_File": self._get_pose_file_path(lig, g),
                                "grid_id": g,
                                "pose_id": p,
                                "affinity": aff,
                                "COM_x": coms[0, lig, g, p],
                                "COM_y": coms[1, lig, g, p],
                                "COM_z": coms[2, lig, g, p],
                                "MolWt": self.molweight.iloc[lig],
                            }
                            if hasattr(self, "ligand_smiles"):
                                smi = self.ligand_smiles.iloc[lig]
                                rec["SMILES"] = None if pd.isna(smi) else smi
                            best_records.append(rec)

        df_best = pd.DataFrame(best_records)
        if not df_best.empty:
            df_best.sort_values(by="affinity", inplace=True)
            if selecttype == "bestPerGrid" and self.mode == "production":
                df_best = self._write_production_readability_summaries(df_best)
            df_best.to_csv(output_csv, index=False)
            if selecttype == "topN":
                self.logger.info(
                    f"Saved top {topN} poses to {output_csv}"
                )
            elif selecttype == "percentile":
                self.logger.info(
                    f"Saved best {threshold_fraction*100:.0f}% poses to {output_csv}"
                )
            elif selecttype == "bestPerGrid":
                self.logger.info(
                    f"Saved best (lowest) pose per ligand per grid to {output_csv}"
                )
        else:
            self.logger.warning(
                f"No valid docking results found. Output CSV not created: {output_csv}"
            )
        return df_best
