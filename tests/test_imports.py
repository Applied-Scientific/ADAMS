import importlib
import pathlib

import pytest


def get_module_paths():
    """
    Dynamically discover all Python modules in the 'adams' package.
    """
    adams_path = pathlib.Path("adams")
    module_paths = []
    for path in adams_path.rglob("*.py"):
        if path.stem == "__init__":
            continue
        # Construct the module path from the file path
        module_path = ".".join(path.with_suffix("").parts)
        module_paths.append(module_path)
    return module_paths


@pytest.mark.parametrize("module_path", get_module_paths())
def test_import_adams_module(module_path):
    """
    Test that a given adams module can be imported.
    """
    try:
        importlib.import_module(module_path)
    except ImportError as e:
        pytest.fail(f"Failed to import module {module_path}: {e}")


# Optional dependencies that may not be installed
OPTIONAL_DEPENDENCIES = [
    "psutil",  # Used for memory monitoring in vina_dock.py
    "resource",  # Used for memory limits in vina_dock.py (Unix-only)
]


@pytest.mark.parametrize("module_name", OPTIONAL_DEPENDENCIES)
def test_optional_dependency_import(module_name):
    """
    Test that optional dependencies can be imported if available.
    These are dependencies used in try/except blocks in the codebase.
    This test will be skipped if the dependency is not installed.
    """
    try:
        importlib.import_module(module_name)
    except ImportError:
        pytest.skip(f"Optional dependency {module_name} not installed")


def test_meeko_import():
    """
    Test that meeko can be imported.
    Note: meeko is listed as required in setup.py but imported with try/except
    in md_analysis/utils.py. This test ensures it's actually available.
    """
    try:
        from meeko import PDBQTMolecule, RDKitMolCreate
    except ImportError as e:
        pytest.fail(f"Failed to import meeko (required dependency): {e}")
