"""
GROMACS binary and MDP path helpers.
"""

import glob
import os


def clean_gro_file(directory: str = None):
    """Remove cached GROMACS files from *directory* (default: current directory)."""
    target = directory or "."
    patterns = ["#*", "*.tpr", "*.trr", "*.cpt", "*.edr", "*.log"]
    for pattern in patterns:
        for filepath in glob.glob(os.path.join(target, pattern)):
            try:
                os.remove(filepath)
            except OSError:
                pass


def get_gromacs_binary(
    gromacs_path: str, binary_type: str = "mpi", require_mpi: bool = False
):
    """Return the GROMACS binary path for the given type. One lightweight existence check."""
    if require_mpi or binary_type == "mpi":
        binary_name = "gmx_mpi"
    elif binary_type == "cuda":
        binary_name = "gmx"
    else:
        binary_name = "gmx"

    if binary_type == "cuda":
        if "/cuda/bin" in gromacs_path:
            binary_path = os.path.join(gromacs_path, binary_name)
        elif gromacs_path.endswith("/bin"):
            cuda_path = gromacs_path.replace("/bin", "/cuda/bin")
            binary_path = os.path.join(cuda_path, binary_name)
        else:
            binary_path = os.path.join(gromacs_path, "cuda", "bin", binary_name)
    else:
        binary_path = os.path.join(gromacs_path, binary_name)

    if not os.path.isfile(binary_path):
        raise FileNotFoundError(
            f"GROMACS binary not found: {binary_path}\n"
            f"Ensure GROMACS is installed (scripts/install.sh) or set gromacs_path."
        )
    return binary_path


def get_mdp_dir():
    """Get the path to the bundled MDP files directory."""
    # This file lives in md_analysis/shared/, so one level up is md_analysis/.
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "mdp")


def get_membrane_mdp_dir():
    """Get the path to the membrane-specific MDP files directory."""
    return os.path.join(get_mdp_dir(), "membrane")

