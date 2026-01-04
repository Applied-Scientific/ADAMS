"""
    Utility functions for parsing pipeline output files and extracting statistics.
    These functions are used by the File Parser Agent.
"""

import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from ...path_config import get_agent_data_path, get_safe_path


def parse_docking_results_impl(csv_path: str) -> Dict[str, Any]:
    """
    Parse docking results CSV and extract statistics.

    This function reads a docking results CSV file (e.g., production_docking_results.csv)
    and extracts comprehensive statistics including affinity metrics, pose counts,
    pocket analysis, and affinity distributions. Returns structured data for
    parameter extraction and decision-making.

    Args:
        csv_path: Path to docking results CSV file. Can be relative to agent_data or absolute.
            Expected columns: ligand_id, grid_id, pose_id, affinity, COM_x, COM_y, COM_z, MolWt (optional)

    Returns:
        dict with:
            - 'csv_path': Full path to CSV file
            - 'statistics': Dict with best_affinity, worst_affinity, avg_affinity, median_affinity, affinity_std
            - 'counts': Dict with total_poses, unique_ligands, unique_pockets, poses_per_ligand_avg/min/max
            - 'pocket_stats': Dict mapping grid_id to {count, best_affinity, avg_affinity}
            - 'top_pockets': List of grid_ids sorted by best affinity
            - 'affinity_percentiles': Dict with p10, p25, p50, p75, p90
            - 'affinity_ranges': Dict with counts for different affinity ranges
            - 'error': Error message if parsing failed, None if successful
    """
    base_path = get_agent_data_path()

    # Handle relative vs absolute paths
    try:
        target = get_safe_path(csv_path, base_path)
    except PermissionError as e:
        return {
            "csv_path": csv_path,
            "statistics": {},
            "counts": {},
            "pocket_stats": {},
            "top_pockets": [],
            "affinity_percentiles": {},
            "affinity_ranges": {},
            "error": str(e),
        }

    result = {
        "csv_path": str(target),
        "statistics": {},
        "counts": {},
        "pocket_stats": {},
        "top_pockets": [],
        "affinity_percentiles": {},
        "affinity_ranges": {},
        "error": None,
    }

    try:
        if not target.exists():
            result["error"] = f"CSV file not found: {target}"
            return result

        # Read CSV
        df = pd.read_csv(target)

        # Validate required columns
        required_cols = ["ligand_id", "grid_id", "pose_id", "affinity"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            result["error"] = f"CSV missing required columns: {missing_cols}"
            return result

        # Extract statistics
        affinities = df["affinity"].values

        result["statistics"] = {
            "best_affinity": float(affinities.min()),  # Most negative (best)
            "worst_affinity": float(affinities.max()),  # Least negative (worst)
            "avg_affinity": float(affinities.mean()),
            "median_affinity": float(np.median(affinities)),
            "affinity_std": float(affinities.std()),
        }

        # Extract counts
        result["counts"] = {
            "total_poses": int(len(df)),
            "unique_ligands": int(df["ligand_id"].nunique()),
            "unique_pockets": int(df["grid_id"].nunique()),
            "poses_per_ligand_avg": float(df.groupby("ligand_id").size().mean()),
            "poses_per_ligand_min": int(df.groupby("ligand_id").size().min()),
            "poses_per_ligand_max": int(df.groupby("ligand_id").size().max()),
        }

        # Pocket analysis
        pocket_stats = {}
        for grid_id, group in df.groupby("grid_id"):
            pocket_affinities = group["affinity"].values
            pocket_stats[int(grid_id)] = {
                "count": int(len(group)),
                "best_affinity": float(pocket_affinities.min()),
                "avg_affinity": float(pocket_affinities.mean()),
            }
        result["pocket_stats"] = pocket_stats

        # Top pockets sorted by best affinity
        result["top_pockets"] = sorted(
            pocket_stats.keys(), key=lambda gid: pocket_stats[gid]["best_affinity"]
        )

        # Affinity percentiles
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = pd.Series(affinities).quantile(
            [p / 100 for p in percentiles]
        )
        result["affinity_percentiles"] = {
            f"p{p}": float(percentile_values[p / 100]) for p in percentiles
        }

        # Affinity ranges (count poses in different affinity ranges)
        # Ranges: < -8.0, -8.0 to -6.0, -6.0 to -4.0, > -4.0
        result["affinity_ranges"] = {
            "very_strong": int((affinities < -8.0).sum()),  # < -8.0
            "strong": int(
                ((affinities >= -8.0) & (affinities < -6.0)).sum()
            ),  # -8.0 to -6.0
            "moderate": int(
                ((affinities >= -6.0) & (affinities < -4.0)).sum()
            ),  # -6.0 to -4.0
            "weak": int((affinities >= -4.0).sum()),  # > -4.0
        }

    except Exception as e:
        result["error"] = f"Failed to parse docking results CSV: {e}"

    return result


def parse_md_results_impl(md_dir: str) -> Dict[str, Any]:
    """
    Parse MD results directory and extract completion status and statistics.

    This function analyzes an MD analysis directory structure and extracts
    completion status, pose statistics, and file paths. It checks for
    protein topology files, prepared poses, completed MD simulations,
    and analysis reports.

    Args:
        md_dir: Path to MD analysis directory or parent directory containing md_analysis/.
            Can be relative to agent_data or absolute.
            Examples:
            - "outputs/run_20251203/md_analysis" (direct path)
            - "outputs/run_20251203" (parent dir, will look for md_analysis/ subdirectory)

    Returns:
        dict with:
            - 'md_dir': Full path to MD analysis directory
            - 'completion_status': Dict with protein_topology_complete, ligand_prep_complete,
                md_simulations_complete (count), analysis_complete
            - 'pose_statistics': Dict with total_poses_prepared, poses_with_md_complete, poses_with_analysis
            - 'file_paths': Dict with protein_gro, protein_top, analysis_reports (list), pose_directories (list)
            - 'error': Error message if parsing failed, None if successful
    """
    base_path = get_agent_data_path()

    # Handle relative vs absolute paths
    try:
        target = get_safe_path(md_dir, base_path)
    except PermissionError as e:
        return {
            "md_dir": md_dir,
            "completion_status": {},
            "pose_statistics": {},
            "file_paths": {
                "protein_gro": None,
                "protein_top": None,
                "analysis_reports": [],
                "pose_directories": [],
            },
            "error": str(e),
        }

    result = {
        "md_dir": None,
        "completion_status": {},
        "pose_statistics": {},
        "file_paths": {
            "protein_gro": None,
            "protein_top": None,
            "analysis_reports": [],
            "pose_directories": [],
        },
        "error": None,
    }

    try:
        # Check if target exists
        if not target.exists():
            result["error"] = f"Directory not found: {target}"
            return result

        # Determine MD analysis directory
        # If target is a directory, check if it contains md_analysis/ or is md_analysis itself
        if target.is_dir():
            md_analysis_dir = target
            # Check if it's the parent directory (contains md_analysis subdirectory)
            if (target / "md_analysis").exists():
                md_analysis_dir = target / "md_analysis"
            # Check if target itself is named md_analysis
            elif target.name == "md_analysis":
                md_analysis_dir = target
            # Otherwise assume target is the md_analysis directory
        else:
            result["error"] = f"Path is not a directory: {target}"
            return result

        result["md_dir"] = str(md_analysis_dir)

        # Check protein topology
        protein_dir = md_analysis_dir / "protein"
        protein_gro = protein_dir / "protein.gro" if protein_dir.exists() else None
        protein_top = protein_dir / "topol.top" if protein_dir.exists() else None

        result["completion_status"]["protein_topology_complete"] = (
            protein_gro is not None
            and protein_gro.exists()
            and protein_top is not None
            and protein_top.exists()
        )

        if protein_gro and protein_gro.exists():
            result["file_paths"]["protein_gro"] = str(protein_gro)
        if protein_top and protein_top.exists():
            result["file_paths"]["protein_top"] = str(protein_top)

        # Check poses directory
        poses_dir = md_analysis_dir / "poses"
        pose_directories = []
        poses_with_md_complete = 0
        poses_with_analysis = 0

        if poses_dir.exists() and poses_dir.is_dir():
            # Find all pose subdirectories
            for item in poses_dir.iterdir():
                if item.is_dir():
                    pose_directories.append(str(item))

                    # Check for completed MD simulation files
                    md_tpr = item / "md.tpr"
                    md_xtc = item / "md.xtc"
                    md_gro = item / "md.gro"
                    if md_tpr.exists() and md_xtc.exists() and md_gro.exists():
                        poses_with_md_complete += 1

                    # Check for analysis files (any CSV in the directory)
                    analysis_files = list(item.glob("*.csv"))
                    if analysis_files:
                        poses_with_analysis += 1

        result["completion_status"]["ligand_prep_complete"] = len(pose_directories) > 0
        result["completion_status"]["md_simulations_complete"] = poses_with_md_complete
        result["file_paths"]["pose_directories"] = pose_directories

        result["pose_statistics"] = {
            "total_poses_prepared": len(pose_directories),
            "poses_with_md_complete": poses_with_md_complete,
            "poses_with_analysis": poses_with_analysis,
        }

        # Check analysis reports
        reports_dir = md_analysis_dir / "reports"
        analysis_reports = []
        if reports_dir.exists() and reports_dir.is_dir():
            # Find all analysis CSV files
            for report_file in reports_dir.glob("*.csv"):
                if report_file.is_file():
                    analysis_reports.append(str(report_file))

        result["completion_status"]["analysis_complete"] = len(analysis_reports) > 0
        result["file_paths"]["analysis_reports"] = analysis_reports

    except Exception as e:
        result["error"] = f"Failed to parse MD results directory: {e}"

    return result
