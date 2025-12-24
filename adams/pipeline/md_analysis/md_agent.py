"""
MD Analysis Agent - Module Execution Order and Dependencies

This module provides function tools for the MD stability analysis pipeline.
The agent must execute steps in the strict sequence defined below, passing
the file_paths dictionary between steps.

MODULE EXECUTION ORDER (strict sequence):
    1. build_file_paths - Initialize file_paths dictionary
    2. discover_paths - Discover GROMACS/AmberTools paths and merge into file_paths
    3. run_protein_topology - Prepare protein topology (pdb2gmx)
    4. run_lig_prepare - Prepare ligands and combine with protein
    5. run_gro - Run MD simulations (NVT, NPT, production)
    6. run_stability_analysis - Analyze trajectories and generate reports

MODULE DEPENDENCIES:

    run_protein_topology:
        Inputs: protein_file, protein_dir, gromacs_path, ambertools_path
        Outputs: protein_gro, protein_top, posre_itp

    run_lig_prepare:
        Inputs: docking_csv, ligand_input, protein_gro, protein_top, poses_dir,
                gromacs_path, ambertools_path
        Outputs: poses_dir (updated with prepared poses)

    run_gro:
        Inputs: poses_dir, gromacs_path, ambertools_path
        Outputs: poses_dir (updated with MD-completed poses)

    run_stability_analysis:
        Inputs: poses_dir, reports_dir, gromacs_path, ambertools_path
        Outputs: summary_report, brief_report

WORKFLOW:
    The file_paths dictionary is the single source of truth for all paths.
    Each module function requires file_paths as the first parameter and returns
    an updated file_paths dictionary that must be passed to the next step.
"""

from pathlib import Path
from typing import TypedDict

from agents import Agent, function_tool

from ...helper_agents.file_parser.file_parser_agent import file_parser_agent
from ..references.reference_file_reader import read_reference_file
from .agent_utils import build_file_paths, discover_paths
from .lig_prepare import LigPrepare
from .protein_topology import ProteinTopology
from .run_gro import Gro
from .stability_analysis import StabilityAnalysis


class FilePathsDict(TypedDict, total=False):
    """Type definition for file_paths dictionary used throughout MD pipeline."""

    md_root: str
    protein_dir: str
    poses_dir: str
    reports_dir: str
    protein_file: str
    protein_gro: str
    protein_top: str
    posre_itp: str
    docking_csv: str
    ligand_input: str
    gromacs_path: str
    ambertools_path: str
    gromacs_binary_type: str  # Auto-detected by discover_paths
    summary_report: str
    brief_report: str


@function_tool
def run_protein_topology(
    file_paths: FilePathsDict,
    forcefield: str = "amber03",
    water_model: str = "tip3p",
    ignore_hydrogens: bool = True,
) -> FilePathsDict:
    """
    Run protein topology preparation step.

    This function prepares protein structure for MD simulation by converting PDB to GROMACS format
    using pdb2gmx. It generates protein.gro, topol.top, and posre.itp files.

    Args:
        file_paths: Required dict containing file paths. Must include:
            - protein_file: Path to input protein PDB file
            - protein_dir: Directory for output files
            - gromacs_path: Path to GROMACS installation
            - ambertools_path: Path to AmberTools installation
            - gromacs_binary_type: Auto-detected binary type (from discover_paths)
        forcefield: GROMACS forcefield (default: "amber03")
        water_model: Water model for pdb2gmx (default: "tip3p")
        ignore_hydrogens: Whether to ignore hydrogens (default: True)

    Returns:
        dict: Updated file_paths dictionary with protein_gro, protein_top, and posre_itp paths
    """
    protein_topology = ProteinTopology(
        file_paths=file_paths,
        forcefield=forcefield,
        water_model=water_model,
        ignore_hydrogens=ignore_hydrogens,
    )

    return protein_topology.run()


@function_tool
def run_lig_prepare(
    file_paths: FilePathsDict,
    tops: int = 50,
    num_cores: int = 0,
    num_gpus: int = 0,
    charge_type: str = "bcc",
    water_margin: float = 1.0,
    ion_conc: float = 0.15,
    pname: str = "K",
    nname: str = "CL",
) -> FilePathsDict:
    """
    Run ligand preparation step.

    This function selects top docking ligands and prepares them for MD simulation by combining
    with protein, solvating, adding ions, and minimizing.

    Args:
        file_paths: Required dict containing file paths. Must include:
            - docking_csv: Path to docking results CSV
            - ligand_input: Ligand structure input (SMILES string, CSV, SDF, MOL2, or directory)
            - protein_gro: Path to protein GRO file
            - protein_top: Path to protein topology file
            - poses_dir: Directory to store prepared poses
            - gromacs_path: Path to GROMACS installation
            - ambertools_path: Path to AmberTools installation
            - gromacs_binary_type: Auto-detected binary type (from discover_paths)
        tops: Number of top ligands per grid (default: 50)
        num_cores: CPU cores for ligand prep (0 uses all-1)
        num_gpus: Number of GPUs for energy minimization (default: 1)
        charge_type: Charge type for Antechamber [bcc|gas] (default: "bcc")
        water_margin: Water box margin in nm (default: 1.0)
        ion_conc: Ion concentration in mol/L (default: 0.15)
        pname: Cation name (default: "K")
        nname: Anion name (default: "CL")

    Returns:
        dict: Updated file_paths dictionary with prepared poses in poses_dir
    """
    lig_prepare = LigPrepare(
        file_paths=file_paths,
        tops=tops,
        num_cores=num_cores if num_cores > 0 else None,
        num_gpus=num_gpus,
        charge_type=charge_type,
        water_margin=water_margin,
        ion_conc=ion_conc,
        pname=pname,
        nname=nname,
    )

    return lig_prepare.run()


@function_tool
def run_gro(
    file_paths: FilePathsDict,
    gpu: bool = False,
    num_gpus: int = 0,
    mpi_ranks: int = 0,
    omp_threads: int = 0,
    max_jobs: int = 0,
    topol: str = "system.top",
    index: str = "index.ndx",
) -> FilePathsDict:
    """
    Run GROMACS MD simulation step.

    This function runs equilibration and production MD simulations for all prepared poses.
    Executes NVT equilibration, NPT equilibration, and production MD.

    Args:
        file_paths: Required dict containing file paths. Must include:
            - poses_dir: Directory containing prepared pose subdirectories (with min.gro files)
            - gromacs_path: Path to GROMACS installation
            - ambertools_path: Path to AmberTools installation
            - gromacs_binary_type: Auto-detected binary type (from discover_paths)
        gpu: Whether to use GPU for MD (default: False)
        num_gpus: Number of GPUs available (default: 1)
        mpi_ranks: Number of MPI ranks (0 = auto-calculate)
        omp_threads: Number of OpenMP threads (0 = auto-calculate)
        max_jobs: Maximum concurrent jobs (0 = auto-calculate)
        topol: Topology file name (default: "system.top")
        index: Index file name (default: "index.ndx")

    Returns:
        dict: Updated file_paths dictionary (poses_dir now contains MD-completed poses)
    """
    gro = Gro(
        file_paths=file_paths,
        gpu=gpu,
        num_gpus=num_gpus,
        mpi_ranks=mpi_ranks,
        omp_threads=omp_threads,
        max_jobs=max_jobs,
        topol=topol,
        index=index,
    )

    return gro.run()


@function_tool
def run_stability_analysis(
    file_paths: FilePathsDict,
    prefix: str = "md",
    Range: str = "all",
    last_frames: int = 100,
    vina_report: str = "",
) -> FilePathsDict:
    """
    Run stability analysis step.

    This function analyzes MD trajectories for stability metrics including RMSD, RMSF,
    and generates summary reports.

    Args:
        file_paths: Required dict containing file paths. Must include:
            - poses_dir: Directory containing MD-completed pose subdirectories (with md.tpr/md.xtc)
            - reports_dir: Directory to write analysis reports
            - gromacs_path: Path to GROMACS installation
            - ambertools_path: Path to AmberTools installation
            - gromacs_binary_type: Auto-detected binary type (from discover_paths)
        prefix: Prefix for MD files (default: "md")
        Range: Analysis range: 'all' or 'last' (default: "all")
        last_frames: Number of last frames when Range='last' (default: 100)
        vina_report: Path to Vina docking report directory or file (default: "")

    Returns:
        dict: Updated file_paths dictionary with summary_report and brief_report paths
    """
    stability_analysis = StabilityAnalysis(
        file_paths=file_paths,
        prefix=prefix,
        Range=Range,
        last_frames=last_frames,
        vina_report=vina_report or None,
    )

    return stability_analysis.run()


prompt_path = Path(__file__).parent / "md_agent_prompt.md"
system_prompt = prompt_path.read_text()

md_agent = Agent(
    model="gpt-5.2",
    name="Stability MD Agent",
    tools=[
        read_reference_file,
        file_parser_agent.as_tool(
            tool_name="file_parser_agent",
            tool_description=(
                "An agent that extracts structured statistics from MD results to check completion status. "
                "Use this agent to parse MD results directories to check completion status, identify which poses "
                "have completed MD simulations, and find analysis reports."
            ),
        ),
        build_file_paths,
        discover_paths,
        run_protein_topology,
        run_lig_prepare,
        run_gro,
        run_stability_analysis,
    ],
    instructions=system_prompt,
)
