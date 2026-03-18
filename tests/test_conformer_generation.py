from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys


class _Logger:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "adams/pipeline/data_preprocessing/conformer_generation.py"

# Minimal package/module scaffolding so we can import the module without the full runtime deps.
adams_pkg = ModuleType("adams")
adams_pkg.__path__ = [str(REPO_ROOT / "adams")]
pipeline_pkg = ModuleType("adams.pipeline")
pipeline_pkg.__path__ = [str(REPO_ROOT / "adams/pipeline")]
dp_pkg = ModuleType("adams.pipeline.data_preprocessing")
dp_pkg.__path__ = [str(REPO_ROOT / "adams/pipeline/data_preprocessing")]
logger_utils = ModuleType("adams.logger_utils")
logger_utils.get_logger = lambda: _Logger()
mp_utils = ModuleType("adams.utils.multiprocessing_utils")
mp_utils.cpu_count = lambda: 1
parallel_executor = ModuleType("adams.utils.parallel_executor")
parallel_executor.ResourceConfig = lambda n_workers: SimpleNamespace(n_workers=n_workers)


class _TaskResult:
    def __init__(self, task_id, success, value=None, error=None):
        self.task_id = task_id
        self.success = success
        self.value = value
        self.error = error


class _DummyParallelExecutor:
    def __init__(self, *_args, **_kwargs):
        pass

    def run(self, *_args, **_kwargs):
        return []


parallel_executor.ParallelExecutor = _DummyParallelExecutor
parallel_executor.TaskResult = _TaskResult
charge_model = ModuleType("adams.pipeline.charge_model")
charge_model.validate_charge_model = lambda value: value
pandas = ModuleType("pandas")
pandas.Series = object
pandas.notna = lambda value: value is not None
pandas.isna = lambda value: value is None
pandas.read_csv = lambda *_args, **_kwargs: None

sys.modules.setdefault("adams", adams_pkg)
sys.modules.setdefault("adams.pipeline", pipeline_pkg)
sys.modules.setdefault("adams.pipeline.data_preprocessing", dp_pkg)
sys.modules.setdefault("adams.logger_utils", logger_utils)
sys.modules.setdefault("adams.utils.multiprocessing_utils", mp_utils)
sys.modules.setdefault("adams.utils.parallel_executor", parallel_executor)
sys.modules.setdefault("adams.pipeline.charge_model", charge_model)
sys.modules.setdefault("pandas", pandas)

SPEC = spec_from_file_location(
    "adams.pipeline.data_preprocessing.conformer_generation",
    MODULE_PATH,
)
cg = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(cg)


class _FakeMol:
    def __init__(self, conformers=None):
        self._conformers = list(conformers or [])

    def GetConformers(self):
        return list(self._conformers)

    def GetNumConformers(self):
        return len(self._conformers)


def _install_fake_modules(monkeypatch, embed_result):
    meeko = ModuleType("meeko")
    meeko.MoleculePreparation = type(
        "MoleculePreparation",
        (),
        {"__init__": lambda self, *args, **kwargs: None},
    )
    meeko.PDBQTWriterLegacy = type(
        "PDBQTWriterLegacy",
        (),
        {"write_string": staticmethod(lambda *_args, **_kwargs: ("", True, None))},
    )

    rdkit = ModuleType("rdkit")
    chem = ModuleType("rdkit.Chem")
    rddg = ModuleType("rdkit.Chem.rdDistGeom")
    all_chem = ModuleType("rdkit.Chem.AllChem")

    chem.MolFromSmiles = lambda smiles: _FakeMol()
    chem.AddHs = lambda mol: mol
    chem.Mol = lambda mol: mol
    chem.Conformer = lambda conf: conf

    rddg.ETKDGv2 = lambda: SimpleNamespace(randomSeed=None)
    rddg.EmbedMultipleConfs = lambda mol, num_confs, params: embed_result

    all_chem.MMFFGetMoleculeForceField = lambda *args, **kwargs: None
    all_chem.MMFFGetMoleculeProperties = lambda *args, **kwargs: None
    all_chem.MMFFOptimizeMolecule = lambda *args, **kwargs: 0
    all_chem.UFFGetMoleculeForceField = lambda *args, **kwargs: None
    all_chem.UFFOptimizeMolecule = lambda *args, **kwargs: 0

    monkeypatch.setitem(sys.modules, "meeko", meeko)
    monkeypatch.setitem(sys.modules, "rdkit", rdkit)
    monkeypatch.setitem(sys.modules, "rdkit.Chem", chem)
    monkeypatch.setitem(sys.modules, "rdkit.Chem.rdDistGeom", rddg)
    monkeypatch.setitem(sys.modules, "rdkit.Chem.AllChem", all_chem)


def test_process_one_ligand_returns_clean_error_when_no_conformers_embedded(monkeypatch, tmp_path):
    _install_fake_modules(monkeypatch, embed_result=[])

    result = cg._process_one_ligand((
        "CC",
        "lig",
        str(tmp_path),
        8,
        2,
        3.0,
        42,
        "row1",
        None,
        "row1",
        "original",
        "gasteiger",
    ))

    assert result[0] is False
    assert result[-1] == "Failed to embed molecule"


def test_process_one_ligand_handles_nonempty_embed_result_with_zero_conformers(monkeypatch, tmp_path):
    _install_fake_modules(monkeypatch, embed_result=[0, 1])

    result = cg._process_one_ligand((
        "CC",
        "lig",
        str(tmp_path),
        8,
        2,
        3.0,
        42,
        "row1",
        None,
        "row1",
        "original",
        "gasteiger",
    ))

    assert result[0] is False
    assert result[-1] == "Failed to embed molecule"
