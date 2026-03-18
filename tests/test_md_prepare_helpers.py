from __future__ import annotations

import importlib
import subprocess
import sys
import types
from pathlib import Path


def _ensure_package(name: str, rel_path: str, repo_root: Path) -> None:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    module.__path__ = [str(repo_root / rel_path)]


def _load_md_modules():
    repo_root = Path(__file__).resolve().parents[1]
    _ensure_package("adams", "adams", repo_root)
    _ensure_package("adams.pipeline", "adams/pipeline", repo_root)
    _ensure_package("adams.utils", "adams/utils", repo_root)
    _ensure_package("adams.pipeline.md_analysis", "adams/pipeline/md_analysis", repo_root)
    _ensure_package(
        "adams.pipeline.md_analysis.prepare",
        "adams/pipeline/md_analysis/prepare",
        repo_root,
    )
    _ensure_package(
        "adams.pipeline.md_analysis.simulate",
        "adams/pipeline/md_analysis/simulate",
        repo_root,
    )
    if "pandas" not in sys.modules:
        pandas_stub = types.ModuleType("pandas")

        class _DataFrame:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def to_csv(self, *args, **kwargs):
                raise RuntimeError("pandas.DataFrame.to_csv should not be used in this test")

        pandas_stub.DataFrame = _DataFrame
        sys.modules["pandas"] = pandas_stub
    ligand_ops_name = "adams.pipeline.md_analysis.prepare.ligand_ops"
    if ligand_ops_name not in sys.modules:
        ligand_ops_stub = types.ModuleType(ligand_ops_name)
        ligand_ops_stub.formal_charge = lambda *args, **kwargs: 0
        sys.modules[ligand_ops_name] = ligand_ops_stub
    sys.modules.pop("adams.pipeline.md_analysis.shared", None)
    importlib.import_module("adams.pipeline.md_analysis.shared")
    acpype_runner = importlib.import_module(
        "adams.pipeline.md_analysis.prepare.acpype_runner"
    )
    index_ops = importlib.import_module("adams.pipeline.md_analysis.shared.index_ops")
    soluble_md = importlib.import_module(
        "adams.pipeline.md_analysis.simulate.soluble_md"
    )
    return acpype_runner, index_ops, soluble_md


ACPYPE_RUNNER, INDEX_OPS, SOLUBLE_MD = _load_md_modules()


def _write_gro(path: Path, atom_count: int) -> None:
    atoms = []
    for idx in range(1, atom_count + 1):
        atoms.append(
            f"{1:5d}{'SOL':>5}{'OW':>5}{idx:5d}{0.0:8.3f}{0.0:8.3f}{0.0:8.3f}\n"
        )
    path.write_text(
        "generated\n"
        f"{atom_count:5d}\n"
        + "".join(atoms)
        + "   1.00000   1.00000   1.00000\n",
        encoding="utf-8",
    )


def test_write_bulk_solvent_index_uses_appended_atom_block(tmp_path: Path):
    newbox = tmp_path / "newbox.gro"
    solv = tmp_path / "solv.gro"
    ndx = tmp_path / "bulk_solvent.ndx"

    _write_gro(newbox, atom_count=6)
    _write_gro(solv, atom_count=15)

    start_atom, end_atom = INDEX_OPS.write_bulk_solvent_index(
        newbox,
        solv,
        ndx,
        group_name="SOL",
    )

    assert (start_atom, end_atom) == (7, 15)
    text = ndx.read_text(encoding="utf-8")
    assert "[ SOL ]" in text
    assert "7 8 9 10 11 12 13 14 15" in text


def test_apply_soluble_eq_nsteps_scale_scales_all_equilibration_files(tmp_path: Path):
    for mdp_name, nsteps in (
        ("nvt.mdp", 250000),
        ("npt_eq1.mdp", 125000),
        ("npt_eq2.mdp", 125000),
        ("npt_eq3.mdp", 125000),
        ("npt_eq4.mdp", 125000),
    ):
        (tmp_path / mdp_name).write_text(
            f"integrator = md\nnsteps = {nsteps}\ndt = 0.002\n",
            encoding="utf-8",
        )

    applied = SOLUBLE_MD._apply_soluble_eq_nsteps_scale(str(tmp_path), 0.1)

    assert applied["nvt"]["scaled_nsteps"] == 25000
    assert applied["npt_eq1"]["scaled_nsteps"] == 12500
    assert "nsteps = 25000" in (tmp_path / "nvt.mdp").read_text(encoding="utf-8")
    assert "nsteps = 12500" in (tmp_path / "npt_eq4.mdp").read_text(encoding="utf-8")


def test_acpype_odd_electron_failure_triggers_gas_retry():
    exc = subprocess.CalledProcessError(
        1,
        ["acpype"],
        output="",
        stderr="The number of electrons is odd (239). Please check charge and spin multiplicity.",
    )

    assert ACPYPE_RUNNER._should_retry_with_gas_after_failure(
        exc, charge_type="bcc"
    ) is True
    assert ACPYPE_RUNNER._should_retry_with_gas_after_failure(
        exc, charge_type="gas"
    ) is False
