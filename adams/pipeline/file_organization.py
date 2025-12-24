"""
File organization utilities for the complete pipeline.

This module provides explicit file organization functionality to organize
intermediate files during runtime into a structured directory hierarchy.
All pipeline stages (preprocessing, docking, MD) can use these utilities.
"""

import os


def setup_docking_dirs(out_folder, mode="production"):
    """
    Create organized directory structure for docking outputs.

    Args:
        out_folder: Root output folder
        mode: "search" or "production"

    Returns:
        dict: Dictionary with 'poses', 'summaries', 'metadata' directory paths
    """
    if mode == "search":
        base = os.path.join(out_folder, "docking", "search")
    else:
        base = os.path.join(out_folder, "docking", "production")

    poses_dir = os.path.join(base, "poses")
    summaries_dir = os.path.join(base, "summaries")
    metadata_dir = os.path.join(base, "metadata")

    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    return {
        "poses": poses_dir,
        "summaries": summaries_dir,
        "metadata": metadata_dir,
        "root": base,
    }


def setup_preprocessing_dirs(outpath):
    """
    Create organized directory structure for receptor preparation and ligand data processing outputs.
    Note: These operations are independent and can be run in any order.

    Args:
        outpath: Root output folder

    Returns:
        dict: Dictionary with 'receptors', 'ligands' directory paths
    """
    base = os.path.join(outpath, "preprocessing")
    receptors_dir = os.path.join(base, "receptors")
    ligands_dir = os.path.join(base, "ligands")

    os.makedirs(receptors_dir, exist_ok=True)
    os.makedirs(ligands_dir, exist_ok=True)

    return {"receptors": receptors_dir, "ligands": ligands_dir, "root": base}
