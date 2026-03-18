from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path


def _ensure_package(name: str, rel_path: str, repo_root: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    module.__path__ = [str(repo_root / rel_path)]


def _load_protonation_module():
    repo_root = Path(__file__).resolve().parents[1]
    _ensure_package("adams", "adams", repo_root)
    _ensure_package("adams.pipeline", "adams/pipeline", repo_root)
    _ensure_package(
        "adams.pipeline.data_preprocessing",
        "adams/pipeline/data_preprocessing",
        repo_root,
    )
    return importlib.import_module("adams.pipeline.data_preprocessing.protonation")


PROTONATION = _load_protonation_module()


def _pdb_line(
    record: str,
    serial: int,
    atom: str,
    resname: str,
    chain: str,
    resseq: int,
    x: float,
    y: float,
    z: float,
    element: str,
) -> str:
    return (
        f"{record:<6}{serial:5d} {atom:>4} {resname:>3} {chain}{resseq:4d}"
        f"    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {element:>2}\n"
    )


def test_merge_preserved_hetatm_skips_duplicate_waters_with_renamed_atoms(tmp_path: Path):
    input_pdb = tmp_path / "input.pdb"
    output_pdb = tmp_path / "output.pdb"

    water_coords = {
        "O": (26.324, 47.494, 5.174),
        "H1": (26.900, 47.900, 5.500),
        "H2": (25.800, 47.800, 5.600),
    }

    input_pdb.write_text(
        "".join(
            [
                _pdb_line("HETATM", 1, "O", "HOH", "A", 128, *water_coords["O"], "O"),
                _pdb_line("HETATM", 2, "H1", "HOH", "A", 128, *water_coords["H1"], "H"),
                _pdb_line("HETATM", 3, "H2", "HOH", "A", 128, *water_coords["H2"], "H"),
                _pdb_line("HETATM", 4, "C1", "LIG", "A", 501, 10.0, 11.0, 12.0, "C"),
                "END\n",
            ]
        ),
        encoding="utf-8",
    )
    output_pdb.write_text(
        "".join(
            [
                _pdb_line("ATOM", 100, "CA", "SER", "A", 1, 1.0, 2.0, 3.0, "C"),
                _pdb_line("HETATM", 2122, "OW", "WAT", "A", 128, *water_coords["O"], "O"),
                _pdb_line("HETATM", 2123, "HW1", "WAT", "A", 128, *water_coords["H1"], "H"),
                _pdb_line("HETATM", 2124, "HW2", "WAT", "A", 128, *water_coords["H2"], "H"),
                "END\n",
            ]
        ),
        encoding="utf-8",
    )

    appended = PROTONATION._merge_preserved_hetatm(str(input_pdb), str(output_pdb))

    assert appended == 1

    merged_lines = output_pdb.read_text(encoding="utf-8").splitlines()
    water_lines = [
        line
        for line in merged_lines
        if line.startswith("HETATM") and line[21:22] == "A" and line[22:26].strip() == "128"
    ]
    ligand_lines = [
        line for line in merged_lines if line.startswith("HETATM") and line[17:20].strip() == "LIG"
    ]

    assert len(water_lines) == 3
    assert all(line[17:20].strip() == "WAT" for line in water_lines)
    assert len(ligand_lines) == 1
