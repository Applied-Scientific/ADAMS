import csv
import sys
import types
from pathlib import Path

from adams.pipeline.docking import protocolized_docking as protocolized_module
from adams.pipeline.docking.protocolized_docking import (
    build_smiles_csv_from_folder,
    parse_docking_centers,
    run_standard_docking_job,
    write_score_only_csv,
)


def test_build_smiles_csv_from_folder_supports_multiple_files_and_lines(tmp_path):
    folder = tmp_path / "ligands"
    folder.mkdir()
    (folder / "a.smi").write_text("CCO\nCCC custom_id\n", encoding="utf-8")
    (folder / "b.smiles").write_text("NCCN\n", encoding="utf-8")

    output_csv = tmp_path / "ligands.csv"
    result = build_smiles_csv_from_folder(folder, output_csv)

    with result.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"ID": "a", "SMILES": "CCO"},
        {"ID": "custom_id", "SMILES": "CCC"},
        {"ID": "b", "SMILES": "NCCN"},
    ]


def test_parse_docking_centers_requires_xyz_triplets():
    assert parse_docking_centers("1,2,3,4,5,6") == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_write_score_only_csv_falls_back_to_production_results(tmp_path):
    ranked = tmp_path / "production_best_entity_per_ligand_ranked.csv"
    production = tmp_path / "production_docking_results.csv"
    production.write_text(
        "Parent_ID,affinity\nligA,-8.1\nligA,-7.4\nligB,-9.0\n",
        encoding="utf-8",
    )

    out = write_score_only_csv(ranked, production)
    with out.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows == [
        {"LigandName": "ligB", "affinity": "-9.0"},
        {"LigandName": "ligA", "affinity": "-8.1"},
    ]


def test_run_standard_docking_job_forwards_residue_range_to_clean_pdb(
    tmp_path, monkeypatch
):
    class _StopAfterClean(RuntimeError):
        pass

    outdir = tmp_path / "out"
    receptor = tmp_path / "receptor.pdb"
    ligand_csv = tmp_path / "ligands.csv"
    receptor.write_text("ATOM\n", encoding="utf-8")
    ligand_csv.write_text("ID,SMILES\nlig,C\n", encoding="utf-8")

    captured: dict[str, object] = {}

    clean_mod = types.ModuleType("adams.pipeline.data_preprocessing.clean_pdb")

    class FakeCleanPDB:
        def __init__(self, **kwargs):
            captured["clean_pdb_kwargs"] = dict(kwargs)

        def clean(self):
            cleaned = outdir / "receptor_clean.pdb"
            cleaned.parent.mkdir(parents=True, exist_ok=True)
            cleaned.write_text("ATOM\n", encoding="utf-8")
            return str(cleaned)

    clean_mod.CleanPDB = FakeCleanPDB
    monkeypatch.setitem(
        sys.modules,
        "adams.pipeline.data_preprocessing.clean_pdb",
        clean_mod,
    )

    conformer_mod = types.ModuleType(
        "adams.pipeline.data_preprocessing.conformer_generation"
    )
    conformer_mod.generate_conformers_to_pdbqt = (
        lambda *args, **kwargs: str(outdir / "docking_ready_ligands.csv")
    )
    monkeypatch.setitem(
        sys.modules,
        "adams.pipeline.data_preprocessing.conformer_generation",
        conformer_mod,
    )

    ligand_prep_mod = types.ModuleType(
        "adams.pipeline.data_preprocessing.ligand_preprocessing"
    )

    class FakeLigandPreprocessor:
        def __init__(self, *args, **kwargs):
            return None

        def run(self):
            return {"small_mw": str(ligand_csv)}

    ligand_prep_mod.LigandPreprocessor = FakeLigandPreprocessor
    monkeypatch.setitem(
        sys.modules,
        "adams.pipeline.data_preprocessing.ligand_preprocessing",
        ligand_prep_mod,
    )

    protonation_mod = types.ModuleType("adams.pipeline.data_preprocessing.protonation")
    protonation_mod.run_pdb2pqr = lambda *args, **kwargs: (_ for _ in ()).throw(
        _StopAfterClean("stop after CleanPDB")
    )
    monkeypatch.setitem(
        sys.modules,
        "adams.pipeline.data_preprocessing.protonation",
        protonation_mod,
    )

    standardize_mod = types.ModuleType(
        "adams.pipeline.data_preprocessing.standardize_ligands"
    )
    standardize_mod.convert_3d_to_pdbqt = (
        lambda *args, **kwargs: str(outdir / "docking_ready_ligands.csv")
    )
    standardize_mod.detect_ligand_format = lambda *args, **kwargs: {"has_3d": False}

    def _standardize_2d_to_csv(*args, **kwargs):
        standardized = outdir / "standardized.csv"
        standardized.parent.mkdir(parents=True, exist_ok=True)
        standardized.write_text("ID,SMILES\nlig,C\n", encoding="utf-8")
        return str(standardized)

    standardize_mod.standardize_2d_to_csv = _standardize_2d_to_csv
    monkeypatch.setitem(
        sys.modules,
        "adams.pipeline.data_preprocessing.standardize_ligands",
        standardize_mod,
    )

    file_org_mod = types.ModuleType("adams.pipeline.file_organization")
    file_org_mod.setup_preprocessing_dirs = lambda outdir_str: {
        "receptors": str(Path(outdir_str) / "preprocessing" / "receptors")
    }
    monkeypatch.setitem(sys.modules, "adams.pipeline.file_organization", file_org_mod)

    docking_mod = types.ModuleType("adams.pipeline.docking.docking")
    docking_mod.DockingPipeline = object
    monkeypatch.setitem(sys.modules, "adams.pipeline.docking.docking", docking_mod)

    monkeypatch.setattr(protocolized_module, "set_agent_data_path", lambda path=None: Path(path))
    monkeypatch.setattr(protocolized_module, "start_console_transcript", lambda path=None: Path(path))
    monkeypatch.setattr(protocolized_module, "setup_logger", lambda *args, **kwargs: None)

    try:
        run_standard_docking_job(
            receptor=str(receptor),
            ligand_input=str(ligand_csv),
            outdir=str(outdir),
            docking_centers="1,2,3",
            residue_range_start=544,
            residue_range_end=667,
            enumerate_microstates=False,
        )
    except _StopAfterClean:
        pass
    else:
        raise AssertionError("expected test stop after CleanPDB forwarding check")

    assert captured["clean_pdb_kwargs"]["residue_range_start"] == 544
    assert captured["clean_pdb_kwargs"]["residue_range_end"] == 667
