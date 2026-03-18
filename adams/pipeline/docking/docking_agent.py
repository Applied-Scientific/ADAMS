"""
Docking Agent - Module Execution Order and Dependencies

This module provides function tools for the molecular docking pipeline.
The agent must execute steps in the strict sequence defined below.

MODULE EXECUTION ORDER (strict sequence):
    1. run_docking (search mode) - Discover binding sites by systematic docking
    2. run_find_pocket - Cluster search results to identify binding pockets
    3. run_docking (production mode) - Production docking at identified sites

MODULE DEPENDENCIES:

    run_docking (search mode):
        Inputs: input_data, receptor, backend
        Outputs: best_search_docking_centers.csv

    run_find_pocket:
        Inputs: input_file (from search output), out_path
        Outputs: docking_centers.csv (top N binding pockets)

    run_docking (production mode):
        Inputs: input_data, receptor, backend, docking_centers_file (from find_pocket output)
        Outputs: production_docking_results.csv
"""

import os
from pathlib import Path
from typing import List, Optional

import pandas as pd
from agents import Agent, function_tool

from ...common_utils import get_gpu_count
from ...helper_agents.file_parser.file_parser_agent import get_file_parser_agent
from ...logger_utils import get_logger, setup_logger
from ...model_config import get_current_model_name, get_resolved_model
from ...user_plan_utils import (
    append_to_plan_section,
    contribute_stage_to_plan,
    read_plan_document,
)
from ..charge_model import validate_charge_model
from ..file_organization import setup_docking_dirs
from ..references.reference_file_reader import read_reference_file
from .docking import DockingPipeline
from .find_pocket import FindPocket


@function_tool
def run_docking(
    input_data: str,
    receptor: str,
    backend: str = "vina",
    # Common parameters
    complex: Optional[str] = None,
    mode: str = "production",
    num_pockets: int = 1,
    num_poses: int = 5,
    docking_centers: Optional[List[float]] = None,
    docking_centers_file: Optional[str] = None,
    minimized_dock: bool = False,
    search_gridsize: float = 25.0,
    production_gridsize: Optional[float] = None,
    lock_grid_center: bool = True,
    search_margin: float = 5.0,
    out_folder: str = "out_folder",
    log_file: Optional[str] = None,
    pH: float = 7.4,
    charge_model: str = "gasteiger",
    # CPU (vina) specific
    num_cores: Optional[int] = None,
    auto_dock_num_cores: int = 1,
    # GPU (vina_gpu, unidock) specific
    num_gpus: Optional[int] = None,
    gpu_ids: Optional[List[int]] = None,
    # UniDock specific (passed through, None = use unidock defaults)
    scoring: Optional[str] = None,
    exhaustiveness: Optional[int] = None,
    search_mode: Optional[str] = None,
    energy_range: Optional[float] = None,
    min_rmsd: Optional[float] = None,
    spacing: Optional[float] = None,
    seed: Optional[int] = None,
    refine_step: Optional[int] = None,
    max_evals: Optional[int] = None,
    max_step: Optional[int] = None,
    max_gpu_memory: Optional[int] = None,
    verbosity: Optional[int] = None,
    cpu: Optional[int] = None,
) -> str:
    """
    Run molecular docking with selectable backend engine.

    Supports three backends:
    - "vina": CPU-based AutoDock Vina (parallelized across CPU cores)
    - "vina_gpu": GPU-accelerated AutoDock-Vina-GPU-2-1 (batch processing per pocket)
    - "unidock": UniDock GPU engine with extensive tuning options

    All backends support both search mode (discover binding sites) and production mode
    (dock at known sites). Parameters not applicable to the selected backend are ignored.

    Args:
        input_data: Path to CSV file with ID and PDBQT_File columns.
            All ligands must be pre-prepared as PDBQT files.
        receptor: Path to receptor file (PDB or PDBQT). PDB is auto-converted to PDBQT.
        backend: Docking engine to use. Options:
            - "vina": CPU-based, good for small datasets or when GPUs unavailable
            - "vina_gpu": GPU-accelerated, ideal for large ligand libraries
            - "unidock": GPU-accelerated with advanced tuning options
            Default: "vina"

        Common parameters (all backends):
            mode: "search" (discover binding sites) or "production" (dock at known sites).
                Default: "production"
            complex: Complex structure for auto center detection.
                Format: "file.pdb,SEL1,SEL2" where each selector is:
                - "RES" (must be unique in complex)
                - "RES:CHAIN"
                - "RES:CHAIN:RESSEQ" (optional insertion code, e.g., 123A)
            docking_centers: Manual coordinates [x1,y1,z1, x2,y2,z2, ...] in Angstroms.
            docking_centers_file: CSV file with docking centers (from run_find_pocket).
            num_pockets: Number of binding sites. Default: 1
            num_poses: Poses per ligand per site. Default: 5
            minimized_dock: Use small 5Å box for precise sites. Default: False
            search_gridsize: Box size in Angstroms. Default: 25.0
            production_gridsize: Optional production-mode box size in Angstroms.
                When provided, overrides backend defaults for production mode.
            lock_grid_center: If True (default), keep the user-defined docking center
                fixed after pre-minimization. Set False to allow recentering around
                minimized ligand coordinates.
            search_margin: Margin around receptor bounds for search. Default: 5.0
            out_folder: Output directory. Default: "out_folder"
            log_file: Optional path for this run's log file (e.g. agent_data/logs/adams_pipeline_run_<run_name>.log).
                If set, all log output for this run is written here. Use for comparison runs so each run has its own log.
            pH: float: pH value for receptor protonation state (default: 7.4). Used when converting
                PDB receptor to PDBQT format. Must match the pH used in preprocessing (run_clean_pdb)
                to ensure protonation consistency across the pipeline.
            charge_model: Charge model for receptor PDBQT conversion (default: "gasteiger"). Must match
                ligand preparation (run_smiles_to_pdbqt / convert_3d_to_pdbqt). Use same value for both.

        CPU backend (vina) specific:
            num_cores: Parallel CPU cores. None = auto-detect.
            auto_dock_num_cores: Cores per Vina subprocess. Default: 1

        GPU backends (vina_gpu, unidock) specific:
            num_gpus: Number of GPUs to use. None = auto-detect.
            gpu_ids: Specific GPU IDs, e.g. [0, 1]. None = use 0 to num_gpus-1.

        UniDock specific (backend="unidock", None = use unidock defaults):
            scoring: Scoring function "ad4", "vina", or "vinardo". Default: vina
            exhaustiveness: Global search depth. Default: 8
            search_mode: Preset "fast", "balance", or "detail"
            energy_range: Max energy difference in kcal/mol. Default: 3
            min_rmsd: Min RMSD between poses in Å. Default: 1
            spacing: Grid spacing in Å. Default: 0.375
            seed: Random seed for reproducibility
            refine_step: Refinement steps. Default: 3
            max_evals: Max MC evaluations (0 = heuristic)
            max_step: Max MC steps (0 = heuristic)
            max_gpu_memory: GPU memory limit in bytes (0 = all)
            verbosity: Output level 0/1/2. Default: 1
            cpu: CPU count for unidock (0 = auto)

    Returns:
        str: Path to output CSV:
            - Search: {out_folder}/docking/search/summaries/best_search_docking_centers.csv
            - Production: {out_folder}/docking/production/summaries/production_docking_results.csv
    """
    if log_file:
        setup_logger(log_file=log_file)

    charge_model = validate_charge_model(charge_model)

    # Validate backend early
    SUPPORTED_BACKENDS = {"vina", "vina_gpu", "unidock"}
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend: {backend}. "
            f"Choose from: {', '.join(sorted(SUPPORTED_BACKENDS))}"
        )

    # Validate and set defaults (backend-specific; backends also default num_cores/num_gpus when None)
    # Treat 0 as "use default" so callers (e.g. LLM tools) passing 0 get backend default instead of error
    if production_gridsize is not None and production_gridsize <= 0:
        production_gridsize = None

    if backend == "vina":
        if num_cores is not None and num_cores <= 0:
            raise ValueError(
                f"num_cores must be None (for auto-detection) or positive, got {num_cores}"
            )
    elif backend in ("vina_gpu", "unidock"):
        available_gpus = get_gpu_count()
        if num_gpus is None:
            num_gpus = available_gpus
            if num_gpus == 0:
                raise ValueError(
                    f"GPU backend '{backend}' requires GPUs, but none were detected. "
                    f"Please either:\n"
                    f"  1. Use CPU backend 'vina' instead, or\n"
                    f"  2. Ensure CUDA GPUs are available (check with 'nvidia-smi')"
                )
        elif num_gpus <= 0:
            raise ValueError(
                f"num_gpus must be positive, got: {num_gpus}"
            )
        if gpu_ids is not None:
            if len(gpu_ids) != num_gpus:
                raise ValueError(
                    f"gpu_ids length ({len(gpu_ids)}) must match num_gpus ({num_gpus})"
                )
            if any(gid < 0 for gid in gpu_ids):
                raise ValueError(f"gpu_ids must be non-negative, got: {gpu_ids}")

    # Build kwargs, excluding 'backend' (passed separately to DockingPipeline)
    kwargs = {
        "input_data": input_data,
        "receptor": receptor,
        "complex": complex,
        "mode": mode,
        "num_pockets": num_pockets,
        "num_poses": num_poses,
        "docking_centers": docking_centers,
        "docking_centers_file": docking_centers_file,
        "minimized_dock": minimized_dock,
        "search_gridsize": search_gridsize,
        "production_gridsize": production_gridsize,
        "lock_grid_center": lock_grid_center,
        "search_margin": search_margin,
        "out_folder": out_folder,
        "pH": pH,
        "charge_model": charge_model,
        "num_cores": num_cores,
        "auto_dock_num_cores": auto_dock_num_cores,
        "num_gpus": num_gpus,
        "gpu_ids": gpu_ids,
        "scoring": scoring,
        "exhaustiveness": exhaustiveness,
        "search_mode": search_mode,
        "energy_range": energy_range,
        "min_rmsd": min_rmsd,
        "spacing": spacing,
        "seed": seed,
        "refine_step": refine_step,
        "max_evals": max_evals,
        "max_step": max_step,
        "max_gpu_memory": max_gpu_memory,
        "verbosity": verbosity,
        "cpu": cpu,
    }

    try:
        pipeline = DockingPipeline(backend=backend, **kwargs)
        output_csv = pipeline.run()
        
        # Validate that output file was created
        if not os.path.exists(output_csv):
            raise RuntimeError(
                f"Docking pipeline completed but output file not found: {output_csv}"
            )
        return output_csv
    except ValueError as e:
        from ...utils.multiprocessing_utils import is_spawn_error

        if is_spawn_error(e):
            raise RuntimeError(
                f"Multiprocessing spawn error (backend={backend}): {e}"
            ) from e
        raise ValueError(
            f"Docking pipeline configuration error (backend={backend}): {e}"
        ) from e
    except FileNotFoundError as e:
        # Re-raise with more context
        raise FileNotFoundError(
            f"Docking pipeline file error (backend={backend}): {e}"
        ) from e
    except Exception as e:
        # Catch-all for unexpected errors
        logger = get_logger()
        logger.error(f"Docking pipeline failed (backend={backend}): {e}", exc_info=True)
        raise RuntimeError(
            f"Docking pipeline execution failed (backend={backend}): {e}"
        ) from e


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
    try:
        find_pocket = FindPocket(
            input_file=input_file, affinity_cutoff=affinity_cutoff, out_path=out_path
        )
        find_pocket.run()
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"FindPocket file error: {e}. "
            f"Make sure search docking completed successfully."
        ) from e
    except Exception as e:
        logger = get_logger()
        logger.error(f"FindPocket failed: {e}", exc_info=True)
        raise RuntimeError(f"FindPocket execution failed: {e}") from e

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
        if df.empty:
            raise ValueError(
                f"Cluster summary is empty. No binding pockets found above affinity cutoff ({affinity_cutoff} kcal/mol). "
                f"Try lowering the affinity_cutoff or check that search docking produced valid results."
            )
        top_clusters = df.head(top_n_clusters)
        top_clusters.to_csv(docking_centers_path, index=False)
        
        if not os.path.exists(docking_centers_path):
            raise RuntimeError(
                f"Failed to create docking_centers.csv at {docking_centers_path}"
            )
        logger = get_logger()
        logger.info(
            f"Extracted top {top_n_clusters} clusters to {docking_centers_path}"
        )
    else:
        raise FileNotFoundError(
            f"Cluster summary not found at {cluster_summary_path}. "
            f"FindPocket may have failed. Check the logs for errors."
        )

    return docking_centers_path


prompt_path = Path(__file__).parent / "docking_agent_prompt.md"
system_prompt = prompt_path.read_text()

_docking_agent = None
_docking_model = None


def get_docking_agent():
    global _docking_agent, _docking_model
    current_model = get_current_model_name()
    if _docking_agent is None or _docking_model != current_model:
        _docking_agent = Agent(
            model=get_resolved_model(),
            name="Molecular Docking Agent",
            tools=[
                read_reference_file,
                read_plan_document,
                append_to_plan_section,
                contribute_stage_to_plan,
                get_file_parser_agent().as_tool(
                    tool_name="file_parser_agent",
                    tool_description=(
                        "An agent that extracts structured statistics from docking results. "
                        "Use this agent to analyze docking results CSV files to extract affinity statistics, pose counts, "
                        "and pocket analysis. Use the returned evidence for downstream reasoning rather than asking it to choose parameters."
                    ),
                ),
                run_docking,
                run_find_pocket,
            ],
            instructions=system_prompt,
        )
        _docking_model = current_model
    return _docking_agent
