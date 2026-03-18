"""
Small analysis helpers shared by MD reporting code.

These utilities are used by both the soluble-protein and membrane pipelines.
"""

import os
import re

import numpy as np

from ....utils import run_cmd
from .gromacs_commands import pbc_reimage_trajectory


def parse_xvg(fname):
    """Parse .xvg file into (time/x, first-y-column) arrays."""
    times, values = [], []
    if os.path.exists(fname):
        with open(fname) as f:
            for line in f:
                if line.startswith(("#", "@")):
                    continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        times.append(float(parts[0]))
                        values.append(float(parts[1]))
                    except ValueError:
                        continue
    return np.array(times), np.array(values)


def parse_xvg_columns(fname, columns=(1, 2)):
    """
    Parse an XVG file returning specified data columns as separate arrays.

    Column 0 is the x-axis (time).  ``columns`` selects 1-based data columns.
    For example ``columns=(1, 2)`` returns columns 1 and 2 (the first two
    data columns after the x-axis).

    Args:
        fname: Path to the XVG file.
        columns: Tuple of 1-based column indices to extract.

    Returns:
        Tuple of numpy arrays, one per requested column.
    """
    data = {c: [] for c in columns}
    if os.path.exists(fname):
        with open(fname) as f:
            for line in f:
                if line.startswith(("#", "@")):
                    continue
                parts = line.strip().split()
                for c in columns:
                    if len(parts) > c:
                        try:
                            data[c].append(float(parts[c]))
                        except (ValueError, IndexError):
                            pass
    return tuple(np.array(data[c]) for c in columns)


def select_analysis_range(values, analysis_range="all", last_frames=None):
    """
    Apply an analysis-range selection to a value array.

    Shared by both ``StabilityAnalysis`` and ``MembraneAnalysis``.

    Args:
        values: numpy array of values.
        analysis_range: ``"all"`` (return everything) or ``"last"``
            (return the last *last_frames* entries).
        last_frames: Number of trailing frames to keep when
            ``analysis_range="last"``.

    Returns:
        numpy array (possibly truncated).
    """
    if analysis_range == "all" or len(values) == 0:
        return values
    if analysis_range == "last" and last_frames:
        n = min(last_frames, len(values))
        return values[-n:]
    return values


def parse_pose_name(pose_name):
    """Parse pose_name = {Lig_ID}_pocket_{g_id}_top{rank}."""
    match = re.match(r"^(.*)_pocket_(\d+)_top(\d+)$", pose_name)
    if match:
        lig_id, g_id, rank = match.groups()
        return lig_id, int(g_id), int(rank)
    return pose_name, None, None


def mean_std(arr):
    """Return mean/std as floats or (None, None) for empty inputs."""
    if arr is not None and len(arr) > 0:
        return float(arr.mean()), float(arr.std())
    return None, None


def ensure_pbc_corrected_trajectory(
    gmx_binary: str,
    tpr: str,
    xtc: str,
    ndx: str,
    output_xtc: str,
    center_group: str = "Protein",
    output_group: str = "System",
) -> str:
    """Return output_xtc if it already exists; otherwise run pbc_reimage_trajectory and return it."""
    if os.path.exists(output_xtc):
        return output_xtc
    return pbc_reimage_trajectory(
        gmx_binary, tpr, xtc, ndx, output_xtc,
        center_group=center_group,
        output_group=output_group,
    )


def compute_rmsd(
    gmx_binary: str,
    tpr: str,
    xtc: str,
    ndx: str,
    output_xvg: str,
    fit_group: int,
    ref_group: int,
) -> None:
    """Run gmx rms to compute RMSD; fit_group and ref_group are 0-based index group numbers."""
    cmd = [
        gmx_binary, "rms",
        "-s", tpr, "-f", xtc, "-n", ndx,
        "-o", output_xvg, "-tu", "ns",
    ]
    run_cmd(cmd, input_str=f"{fit_group} {ref_group}\n", check=True)


def compute_rmsf(
    gmx_binary: str,
    tpr: str,
    xtc: str,
    ndx: str,
    output_xvg: str,
    group: int,
) -> None:
    """Run gmx rmsf to compute RMSF; group is 0-based index group number."""
    cmd = [
        gmx_binary, "rmsf",
        "-s", tpr, "-f", xtc, "-n", ndx,
        "-o", output_xvg, "-res",
    ]
    run_cmd(cmd, input_str=f"{group}\n", check=True)

