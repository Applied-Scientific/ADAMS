import ast
import os
import subprocess
import tempfile
import textwrap
from typing import Dict, List, Set

from ...logger_utils import get_logger

# Whitelist of allowed modules (standard lib + installed packages)
ALLOWED_MODULES = {
    # Standard Library (Safe subset)
    "math",
    "re",
    "glob",
    "pathlib",
    "itertools",
    "collections",
    "json",
    "csv",
    "pickle",
    "random",
    "time",
    "datetime",
    "typing",
    "logging",
    "argparse",
    "functools",
    "copy",
    "io",
    "sys",
    "os",
    "shutil",  # os/shutil allowed but specific functions are blacklisted
    # Installed Packages
    "numpy",
    "scipy",
    "pandas",
    "matplotlib",
    "sklearn",
    "acpype",
    "meeko",
    "MolKit",
    "vina",
    "gemmi",
    "psutil",
    "rdkit",
    "openbabel",
    "openmm",
    "pdbfixer",
    "simtk",
}

# Blacklisted functions to prevent deletion and dangerous operations
BLACKLISTED_CALLS = {
    "os": {
        "remove",
        "unlink",
        "rmdir",
        "removedirs",
        "rename",
        "replace",
        "system",
        "popen",
        "spawn",
    },
    "shutil": {"rmtree", "move"},
    "sys": {"exit"},
    "subprocess": {"*"},  # All subprocess calls are banned
}


class SecurityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.errors = []
        self.imported_modules = set()

    def visit_Import(self, node):
        for alias in node.names:
            self._check_import(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self._check_import(node.module)
        self.generic_visit(node)

    def _check_import(self, module_name):
        base_module = module_name.split(".")[0]
        if base_module not in ALLOWED_MODULES:
            self.errors.append(f"Importing '{module_name}' is not allowed.")
        self.imported_modules.add(base_module)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                module_name = node.func.value.id
                func_name = node.func.attr
                self._check_call(module_name, func_name)
        self.generic_visit(node)

    def _check_call(self, module_name, func_name):
        if module_name in BLACKLISTED_CALLS:
            blacklist = BLACKLISTED_CALLS[module_name]
            if "*" in blacklist or func_name in blacklist:
                self.errors.append(
                    f"Calling '{module_name}.{func_name}' is not allowed."
                )


def validate_code(code: str) -> List[str]:
    """
    Statically analyze code for disallowed imports and unsafe calls.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    visitor = SecurityVisitor()
    visitor.visit(tree)
    return visitor.errors


def run_python_in_conda(code: str, env_name: str = None) -> Dict[str, str]:
    """
    Executes a Python code snippet in the specified conda environment after security validation.

    Args:
        code: The Python code to execute.
        env_name: The name of the conda environment. If None, defaults to current active env or "adams".

    Returns:
        Dict containing "stdout", "stderr", and "returncode".
    """
    logger = get_logger()

    if env_name is None:
        env_name = os.environ.get("CONDA_DEFAULT_ENV", "adams")

    # 1. Deduct and Validate
    code = textwrap.dedent(code)
    errors = validate_code(code)
    if errors:
        error_msg = "Code validation failed:\n" + "\n".join(errors)
        logger.error(error_msg)
        return {"stdout": "", "stderr": error_msg, "returncode": -1}

    # 2. Write to temp file
    # Using delete=False because we need to pass the path to the subprocess
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        # 3. Execute via conda run
        # We use 'conda run -n env python script'
        # Note: This assumes 'conda' is in the PATH.
        cmd = [
            "conda",
            "run",
            "-n",
            env_name,
            "--no-capture-output",  # Important to let stdout/stderr pass through to subprocess capture
            "python",
            script_path,
        ]

        logger.info(f"Executing custom python script via conda env '{env_name}'...")

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,  # Don't raise exception on non-zero exit, return it
        )

        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    except Exception as e:
        logger.error(f"Failed to execute script: {e}")
        return {"stdout": "", "stderr": str(e), "returncode": -1}
    finally:
        # 4. Cleanup
        if os.path.exists(script_path):
            os.remove(script_path)
