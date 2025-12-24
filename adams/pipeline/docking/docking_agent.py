"""
Docking Agent - Module Execution Order and Dependencies

This module provides function tools for the molecular docking pipeline.
The agent must execute steps in the strict sequence defined below.

MODULE EXECUTION ORDER (strict sequence):
    1. run_vina_dock (search mode) - Discover binding sites by systematic docking
    2. run_find_pocket - Cluster search results to identify binding pockets
    3. run_vina_dock (production mode) OR run_vina_dock_gpu - Production docking at identified sites

MODULE DEPENDENCIES:

    run_vina_dock (search mode):
        Inputs: input_data, receptor
        Outputs: best_search_docking_centers.csv

    run_find_pocket:
        Inputs: input_file (from search output), out_path
        Outputs: docking_centers.csv (top N binding pockets)

    run_vina_dock (production mode) / run_vina_dock_gpu:
        Inputs: input_data, receptor, docking_centers_file (from find_pocket output)
        Outputs: production_docking_results.csv
"""

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from agents import Agent, function_tool

from ...common_utils import get_cpu_count, get_gpu_count
from ...helper_agents.file_parser.file_parser_agent import file_parser_agent
from ...logger_utils import get_logger
from ..file_organization import setup_docking_dirs
from ..references.reference_file_reader import read_reference_file
from .conformer_generation import generate_conformers_from_csv
from .find_pocket import FindPocket
from .vina_dock import VinaDock
from .vina_dock_gpu import VinaDockGPU


@function_tool
def generate_3d_conformers(input_csv: str, out_folder: str = "out_folder") -> str:
    """
    Generates 3D conformers for ligands in a CSV file (SMILES) and saves them as SDF files.
    Returns the path to a new CSV file pointing to these SDF files.

    Use this tool BEFORE docking if you have SMILES strings and want to generate 3D structures
    explicitly, or if you want to ensure consistent 3D structures are used.

    Args:
        input_csv: Path to input CSV with "SMILES" and "ID" columns.
        out_folder: Folder to save generated conformers. Default: "out_folder"

    Returns:
        str: Path to the updated CSV file containing paths to the generated 3D structure files.
    """
    return generate_conformers_from_csv(input_csv, out_folder)


@function_tool
def run_vina_dock(
    input_data: str,
    receptor: str,
    complex: str = None,
    mode: str = "production",
    num_pockets: int = 1,
    num_poses: int = 5,
    docking_centers: Optional[List[float]] = None,
    docking_centers_file: str = None,
    minimized_dock: bool = False,
    search_gridsize: float = 25.0,
    search_margin: float = 5.0,
    auto_dock_num_cores: int = 1,
    out_folder: str = "out_folder",
    num_cores: Optional[int] = None,
) -> str:
    """
    Run molecular docking using AutoDock Vina (CPU-based).

    This function can perform both search docking (to discover binding sites) and production
    docking (at known binding sites) depending on the mode parameter.

    Args:
        input_data: Path to CSV file containing ligand dataset with ID and PDBQT_File columns.
            All ligands must be pre-prepared as PDBQT files in the preprocessing module.
        receptor: Path to receptor protein structure file (PDB or PDBQT format).
            If PDB format is provided, it will be automatically converted to PDBQT.
        mode: Docking mode. "search" for exploratory binding site discovery,
            "production" for production docking at known sites. Default: "production"
        complex: Optional. Path to complex structure with bound ligand for automatic center
            detection. Format: "filename,resname1[,resname2,...]".
            Example: "complex.pdb,LIG1,LIG2". If provided, overrides docking_centers. Default: None
        docking_centers: Optional. Manual specification of binding site coordinates in Angstroms.
            Format: [x1, y1, z1, x2, y2, z2, ...] for multiple sites.
            Example: [10.5, 15.2, 8.7, -5.0, 3.2, 12.1]. Ignored if complex or docking_centers_file
            is provided. Default: None
        docking_centers_file: Optional. Path to CSV file with docking center coordinates,
            typically from run_find_pocket output (docking_centers.csv). If provided,
            overrides docking_centers and num_pockets. Default: None
        num_pockets: Number of binding sites to dock into. Default: 1
        num_poses: Number of binding poses to generate per ligand per site. Default: 5
        minimized_dock: If True, performs energy minimization before docking. Default: False
        search_gridsize: Size of the docking box in Angstroms. Default: 25.0
        search_margin: Additional margin in Angstroms added to docking box boundaries. Default: 5.0
        auto_dock_num_cores: Number of CPU cores per AutoDock Vina subprocess. Default: 1
        out_folder: Output directory for all docking results. Default: "out_folder"
        num_cores: Number of parallel CPU cores for running multiple docking jobs simultaneously.
            If None, uses (CPU count - 1). Default: None

    Returns:
        str: Path to the output CSV file containing docking results.
             - For search mode: {out_folder}/docking/search/summaries/best_search_docking_centers.csv
             - For production mode: {out_folder}/docking/production/summaries/production_docking_results.csv
    """
    if num_cores is not None and num_cores <= 0:
        raise ValueError(
            f"num_cores must be None (for auto-detection) or a positive integer, got {num_cores}. "
            "To use auto-detection, omit the num_cores parameter entirely (do not pass 0)."
        )
    if num_cores is None:
        num_cores = get_cpu_count()

    vina_dock = VinaDock(
        input_data=input_data,
        receptor=receptor,
        complex=complex,
        mode=mode,
        num_pockets=num_pockets,
        num_poses=num_poses,
        docking_centers=docking_centers,
        docking_centers_file=docking_centers_file,
        minimized_dock=minimized_dock,
        search_gridsize=search_gridsize,
        search_margin=search_margin,
        auto_dock_num_cores=auto_dock_num_cores,
        out_folder=out_folder,
        num_cores=num_cores,
    )

    return vina_dock.run()


@function_tool
def run_vina_dock_gpu(
    input_data: str,
    receptor: str,
    complex: str = None,
    mode: str = "production",
    docking_centers: Optional[List[float]] = None,
    docking_centers_file: str = None,
    num_pockets: int = 1,
    num_poses: int = 5,
    minimized_dock: bool = False,
    search_gridsize: float = 25.0,
    search_margin: float = 5.0,
    out_folder: str = "out_folder",
    num_gpus: int = get_gpu_count(),
    gpu_ids: Optional[List[int]] = None,
) -> str:
    """
    Run GPU-accelerated molecular docking (search or production mode).

    This is the GPU-accelerated version using AutoDock-Vina-GPU-2-1 for significantly
    faster docking of large ligand libraries. The GPU version processes all ligands in
    batch mode per pocket, making it ideal for high-throughput virtual screening.

    Supports both search mode (discover binding sites) and production mode (dock at known sites).

    Args:
        input_data: Path to CSV file containing ligand dataset with ID and PDBQT_File columns.
            All ligands must be pre-prepared as PDBQT files in the preprocessing module.
        receptor: Path to receptor protein structure file (PDB or PDBQT format).
            If PDB format is provided, it will be automatically converted to PDBQT.
        complex: Optional. Path to complex structure with bound ligand for automatic center
            detection. Format: "filename,resname1[,resname2,...]".
            Example: "complex.pdb,LIG1,LIG2". If provided, overrides docking_centers. Default: None
        mode: Docking mode: "production" (dock at known sites) or "search" (discover binding sites).
            Default: "production"
        docking_centers: Optional. Manual specification of binding site coordinates in Angstroms.
            Format: [x1, y1, z1, x2, y2, z2, ...] for multiple sites.
            Example: [10.5, 15.2, 8.7, -5.0, 3.2, 12.1]. Ignored if complex or docking_centers_file
            is provided. Default: None
        docking_centers_file: Optional. Path to CSV file with docking center coordinates,
            typically from run_find_pocket output (docking_centers.csv). If provided,
            overrides docking_centers and num_pockets. Default: None
        num_pockets: Number of binding sites to dock into. Default: 1
        num_poses: Number of binding poses to generate per ligand per site. Default: 5
        minimized_dock: If True, uses a fixed 5Å docking box (only for very precise
            binding sites with small ligands <300 Da). For production docking, typically use False.
            Default: False
        search_gridsize: Grid spacing for search mode OR manual box size for production (default: 25.0).
            - In search mode: Spacing between grid points for exhaustive search
            - In production mode: If provided, uses this as fixed box size for all ligands
        search_margin: Margin around receptor bounds for search mode (default: 5.0)
        out_folder: Output directory for all docking results. Default: "out_folder"
        num_gpus: Number of GPUs to use for parallel docking across multiple pockets.
            Each GPU processes one pocket at a time. Default: 1
        gpu_ids: Optional. Specific GPU device IDs to use (e.g., [0, 1, 2, 3]).
            If None, uses GPUs 0 to num_gpus-1. Default: None

    Returns:
        str: Path to the output CSV file containing docking results:
             - Search mode: {out_folder}/docking/search/summaries/best_docking_centers.csv
             - Production mode: {out_folder}/docking/production/summaries/production_docking_results.csv
    """
    vina_dock_gpu = VinaDockGPU(
        input_data=input_data,
        receptor=receptor,
        complex=complex,
        mode=mode,
        num_pockets=num_pockets,
        num_poses=num_poses,
        docking_centers=docking_centers,
        docking_centers_file=docking_centers_file,
        minimized_dock=minimized_dock,
        search_gridsize=search_gridsize,
        search_margin=search_margin,
        out_folder=out_folder,
        num_gpus=num_gpus,
        gpu_ids=gpu_ids,
    )

    return vina_dock_gpu.run()


@function_tool
def run_find_pocket(
    input_file: str,
    affinity_cutoff: float = -4.0,
    out_path: str = "out_folder",
    top_n_clusters: int = 3,
) -> str:
    """
    Run pocket identification step.

    This function analyzes the output from search docking (run_vina_dock with mode="search")
    and clusters high-affinity poses to identify distinct binding pockets. It generates
    cluster summaries and extracts top N binding pockets for production docking.

    Args:
        input_file: Path to CSV file containing docking coordinates from search docking.
            Expected columns: COM_x, COM_y, COM_z, affinity
            Example: "out_folder/docking/search/summaries/best_search_docking_centers.csv"
        affinity_cutoff: Minimum binding affinity threshold in kcal/mol for including poses
            in clustering analysis. Only poses with affinity <= this value (more negative = stronger)
            are considered. Example: -4.0 means only poses with -4.0 kcal/mol or better. Default: -4.0
        out_path: Output directory for clustering results. Should match the out_folder
            used in search docking. Default: "out_folder"
        top_n_clusters: Number of top-ranked binding pockets to extract and save to
            docking_centers.csv. These will be ranked by mean affinity. Default: 3

    Returns:
        str: Path to docking_centers.csv file containing top N binding pocket coordinates
             at {out_path}/docking/search/summaries/docking_centers.csv
    """
    find_pocket = FindPocket(
        input_file=input_file, affinity_cutoff=affinity_cutoff, out_path=out_path
    )

    find_pocket.run()

    # Extract top N clusters
    dir_structure = setup_docking_dirs(out_path, mode="search")
    cluster_summary_path = os.path.join(
        dir_structure["summaries"], "cluster_summary.csv"
    )
    docking_centers_path = os.path.join(
        dir_structure["summaries"], "docking_centers.csv"
    )

    if os.path.exists(cluster_summary_path):
        # Read cluster summary and take top N clusters (assumes sorted by affinity)
        df = pd.read_csv(cluster_summary_path)
        top_clusters = df.head(top_n_clusters)
        top_clusters.to_csv(docking_centers_path, index=False)
        logger = get_logger()
        logger.info(
            f"Extracted top {top_n_clusters} clusters to {docking_centers_path}"
        )
    else:
        logger = get_logger()
        logger.warning(f"cluster_summary.csv not found at {cluster_summary_path}")

    return docking_centers_path


prompt_path = Path(__file__).parent / "docking_agent_prompt.md"
system_prompt = prompt_path.read_text()

docking_agent = Agent(
    model="gpt-5.2",
    name="Molecular Docking Agent",
    tools=[
        read_reference_file,
        file_parser_agent.as_tool(
            tool_name="file_parser_agent",
            tool_description=(
                "An agent that extracts structured statistics from docking results to enable parameter extraction. "
                "Use this agent to analyze docking results CSV files to extract affinity statistics, pose counts, "
                "and pocket analysis. Can help determine optimal parameters for MD based on docking results."
            ),
        ),
        generate_3d_conformers,
        run_vina_dock,
        run_vina_dock_gpu,
        run_find_pocket,
    ],
    instructions=system_prompt,
)
