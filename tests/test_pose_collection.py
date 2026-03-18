from pathlib import Path

import pandas as pd

from adams.pipeline.docking.pose_collection import PoseCollectionMixin


class _Logger:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def debug(self, *_args, **_kwargs):
        return None


class _DummyCollector(PoseCollectionMixin):
    def __init__(self, root: Path):
        self.mode = "production"
        self.logger = _Logger()
        self.dir_structure = {
            "root": str(root),
            "poses": str(root / "poses"),
            "summaries": str(root / "summaries"),
        }


def test_write_production_readability_summaries_creates_ranked_best_entity_file(tmp_path: Path):
    root = tmp_path / "production"
    poses_dir = root / "poses"
    summaries_dir = root / "summaries"
    poses_dir.mkdir(parents=True)
    summaries_dir.mkdir(parents=True)

    for lig_idx in range(4):
        (poses_dir / f"ligand_{lig_idx}_pocket_0_docked.pdbqt").write_text("MODEL\nENDMDL\n", encoding="utf-8")

    df = pd.DataFrame(
        [
            {
                "ligand_id": 0,
                "Parent_ID": "LIG_A",
                "ID": "LIG_A__protomer_1",
                "Variant_Type": "protomer",
                "Variant_ID": "protomer_1",
                "Conformer_Index": 0,
                "grid_id": 0,
                "pose_id": 0,
                "affinity": -8.5,
                "Ligand_PDBQT_File": "lig0.pdbqt",
                "Pose_PDBQT_File": str(poses_dir / "ligand_0_pocket_0_docked.pdbqt"),
            },
            {
                "ligand_id": 1,
                "Parent_ID": "LIG_A",
                "ID": "LIG_A__original_0",
                "Variant_Type": "original",
                "Variant_ID": "original_0",
                "Conformer_Index": 0,
                "grid_id": 0,
                "pose_id": 0,
                "affinity": -9.1,
                "Ligand_PDBQT_File": "lig1.pdbqt",
                "Pose_PDBQT_File": str(poses_dir / "ligand_1_pocket_0_docked.pdbqt"),
            },
            {
                "ligand_id": 2,
                "Parent_ID": "LIG_B",
                "ID": "LIG_B__tautomer_1",
                "Variant_Type": "tautomer",
                "Variant_ID": "tautomer_1",
                "Conformer_Index": 1,
                "grid_id": 0,
                "pose_id": 0,
                "affinity": -7.4,
                "Ligand_PDBQT_File": "lig2.pdbqt",
                "Pose_PDBQT_File": str(poses_dir / "ligand_2_pocket_0_docked.pdbqt"),
            },
            {
                "ligand_id": 3,
                "Parent_ID": "LIG_C",
                "ID": "LIG_C__original_0",
                "Variant_Type": "original",
                "Variant_ID": "original_0",
                "Conformer_Index": 0,
                "grid_id": 0,
                "pose_id": 0,
                "affinity": -8.0,
                "Ligand_PDBQT_File": "lig3.pdbqt",
                "Pose_PDBQT_File": str(poses_dir / "ligand_3_pocket_0_docked.pdbqt"),
            },
        ]
    )

    collector = _DummyCollector(root)
    df_named = collector._write_production_readability_summaries(df)
    assert "Pose_PDBQT_File_Named" in df_named.columns

    ranked_csv = summaries_dir / "production_best_entity_per_ligand_ranked.csv"
    assert ranked_csv.exists()

    ranked = pd.read_csv(ranked_csv)
    assert list(ranked["LigandName"]) == ["LIG_A", "LIG_C", "LIG_B"]
    assert list(ranked["affinity"]) == [-9.1, -8.0, -7.4]
    assert list(ranked["rank_overall"]) == [1, 2, 3]
    assert list(ranked["ID"]) == ["LIG_A__original_0", "LIG_C__original_0", "LIG_B__tautomer_1"]
