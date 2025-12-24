"""Interactive agent wrapper for receptor preparation and ligand data processing workflows."""

from pathlib import Path
from typing import Dict

from agents import Agent, function_tool

from ..references.reference_file_reader import read_reference_file
from .clean_pdb import CleanPDB
from .ligand_preprocessing import LigandPreprocessor
from .python_executor import run_python_in_conda as _run_python_in_conda
from .standardize_ligands import (
    UnsupportedFormatError,
    convert_3d_to_pdbqt,
    detect_ligand_format,
    generate_conformers_to_pdbqt,
    standardize_2d_to_csv,
)


@function_tool
def run_standardize_ligand_data(
    input_file: str,
    output_dir: str = "output",
    id_col: str = "ID",
    smiles_col: str = "SMILES",
) -> Dict:
    """
    Detects input format and prepares ligands for docking.

    This is the FIRST STEP for any ligand input. It automatically detects
    whether the input contains 2D or 3D structures and processes accordingly.

    For 3D structures (SDF/MOL2/PDB/PDBQT):
    - Converts directly to PDBQT format (ready for docking)
    - SKIP run_ligand_preprocessing (not applicable)
    - SKIP conformer generation (already have 3D)
    - Proceed directly to docking

    For 2D structures (CSV/SMILES):
    - Creates CSV with SMILES
    - OPTIONAL: run run_ligand_preprocessing for filtering/sampling
    - REQUIRED: run run_generate_conformers_to_pdbqt before docking

    Args:
        input_file: Path to input file (any format).
        output_dir: Output directory. Default: "output"
        id_col: Name of ID column (if CSV). Default: "ID"
        smiles_col: Name of SMILES column (if CSV). Default: "SMILES"

    Returns:
        dict: {
            'format_type': str,        # '2d' or '3d'
            'output_path': str/list,   # CSV path (2D) or list of PDBQT paths (3D)
            'num_molecules': int,
            'message': str             # Guidance for next steps
        }

    Raises:
        UnsupportedFormatError: If format cannot be handled (use run_python_code for custom conversion)
    """
    # Detect format
    format_info = detect_ligand_format(input_file)

    if format_info["has_3d"]:
        # 3D pathway: Convert directly to PDBQT
        pdbqt_paths = convert_3d_to_pdbqt(input_file, output_dir)

        return {
            "format_type": "3d",
            "output_path": pdbqt_paths,
            "num_molecules": len(pdbqt_paths),
            "message": (
                f"Converted {len(pdbqt_paths)} 3D structures to PDBQT. "
                "SKIP run_ligand_preprocessing and run_generate_conformers_to_pdbqt. "
                "Proceed directly to docking with these PDBQT files."
            ),
        }
    else:
        # 2D pathway: Standardize to CSV with SMILES
        csv_path = standardize_2d_to_csv(input_file, output_dir, id_col, smiles_col)

        return {
            "format_type": "2d",
            "output_path": csv_path,
            "num_molecules": format_info["num_molecules"],
            "message": (
                f"Standardized {format_info['num_molecules']} 2D ligands to CSV. "
                "OPTIONAL: run run_ligand_preprocessing for filtering/sampling. "
                "REQUIRED: run run_generate_conformers_to_pdbqt before docking."
            ),
        }


@function_tool
def run_generate_conformers_to_pdbqt(
    input_csv: str, output_dir: str = "output"
) -> list:
    """
    Generates 3D conformers from SMILES CSV and converts to PDBQT.

    Use this after run_standardize_ligand_data returns format_type='2d'.
    This is REQUIRED before docking 2D inputs.

    The function:
    1. Reads SMILES from the CSV (ID, SMILES, MolWt columns)
    2. Generates 3D conformers using RDKit
    3. Converts each conformer to PDBQT format using Meeko
    4. Returns paths to PDBQT files ready for docking

    Args:
        input_csv: Path to CSV with SMILES (from run_standardize_ligand_data or run_ligand_preprocessing)
        output_dir: Output directory for PDBQT files. Default: "output"

    Returns:
        list: Paths to PDBQT files ready for docking

    Example workflow:
        1. result = run_standardize_ligand_data("ligands.smi")
        2. # Optionally: run_ligand_preprocessing(result['output_path'])
        3. pdbqt_files = run_generate_conformers_to_pdbqt(result['output_path'])
        4. # Pass pdbqt_files to docking agent
    """
    return generate_conformers_to_pdbqt(input_csv, output_dir)


@function_tool
def run_python_code(code: str) -> Dict[str, str]:
    """
    Generate and execute a Python code snippet in the current conda environment.

    This tool allows for custom data manipulation, analysis, or fix-ups that are not covered
    by the standard preprocessing tools. It executes code in a sandboxed environment with
    access to the installed project dependencies (rdkit, pandas, numpy, etc.).

    Constraints & Safety:
    - Only specific safe modules and installed packages are allowed (numpy, pandas, rdkit, etc.).
    - File deletion and system modification (os.remove, shutil.rmtree) are BLOCKED.
    - Network access is NOT available.
    - The code runs in the current active conda environment (defaults to 'adams' if not set).

    Args:
        code: Valid Python code to execute.

    Returns:
        Dict with keys: 'stdout', 'stderr', 'returncode'.
    """
    return _run_python_in_conda(code)


@function_tool
def run_clean_pdb(
    input_pdb: str,
    outpath: str = "./output",
    ligand: bool = False,
    chain_to_keep: str = "A",
) -> str:
    r"""
    Clean and optionally augment a receptor PDB using the ``CleanPDB`` pipeline.

    This function prepares a protein structure for downstream workflows by:
    1. Chain selection: Retaining only the specified chain identifier
    2. Heterogen removal: Removing water molecules and unwanted heterogens
    3. Structure completion: Adding missing atoms and hydrogen atoms
    4. Ligand extraction: Optionally preserving or extracting bound ligands

    PIPELINE ENTRY POINT: Prepares receptor structure for docking. This operation is independent of ligand data processing and can be run in any order.

    Required Input Files:
        - input_pdb: Any valid PDB file (can be from PDB database or experimental)

    Output Files (used by downstream steps):
        - {outpath}/receptors/{prefix}_{chain}_clean_h.pdb: Cleaned protein PDB
          (Used by: Docking Agent for receptor, MD Agent for ProteinTopology)
        - {outpath}/ligands/{prefix}_{ligand_set_name}.pdb: Extracted ligand (if ligand=True)

    Use this when you need to:
    - Prepare a receptor structure for molecular docking
    - Remove unwanted chains and water molecules
    - Standardize a PDB structure for consistent processing
    - Extract ligands from a protein-ligand complex structure

    Args:
        input_pdb (str): Path to the input PDB file containing the protein
            structure to clean. Should be a valid PDB readable by standard
            parsers. Example: "data/protein.pdb"

        outpath (str): Directory to write cleaned receptor files and any
            ligand-related outputs. Created if it does not exist. Default: "./output"

        ligand (bool): If True, attempt to preserve or extract small-molecule
            ligand(s) found in the `input_pdb` and write them as separate
            outputs alongside the cleaned receptor. Only use when the pdb file
            has a ligand. Default: False

        chain_to_keep (str): Chain identifier to retain from the input
            structure; all other chains are removed prior to cleanup. Default: "A"

    Returns:
        str: Full path to the cleaned PDB file. The file is saved at:
            {outpath}/preprocessing/receptors/{prefix}_{chain}_clean_h.pdb
            where prefix is derived from the input filename.

    Example:
        >>> # Clean a receptor structure
        >>> run_clean_pdb(
        ...     input_pdb="protein.pdb",
        ...     outpath="./cleaned",
        ...     chain_to_keep="A"
        ... )
        # Outputs: "./cleaned/cleaned_protein.pdb"

        >>> # Clean and extract ligand
        >>> run_clean_pdb(
        ...     input_pdb="complex.pdb",
        ...     outpath="./cleaned",
        ...     ligand=True,
        ...     chain_to_keep="A"
        ... )
        # Outputs: cleaned protein and extracted ligand files
    """
    cleaner = CleanPDB(
        input_pdb=input_pdb,
        outpath=outpath,
        ligand=ligand,
        chain_to_keep=chain_to_keep,
    )
    return cleaner.clean()


@function_tool
def run_ligand_preprocessing(
    input_data: str,
    molwt_upper_bound: float = 700,
    molwt_lower_bound: float = 0,
    check_rdmol: bool = False,
    sampling: bool = False,
    binsize: int = 100,
    sampling_frac: float = 0.01,
    output_prefix: str = "cleaned_data",
    outpath: str = "./output",
    quick_start: bool = False,
) -> Dict[str, str]:
    r"""
    Process a ligand CSV and generate cleaned/sampled outputs for docking.

    This function processes a CSV file containing compound data by:
    1. Molecular weight filtering: Removing compounds outside the weight range
    2. Chemical validation: Optionally validating SMILES with RDKit
    3. Stratified sampling: Sampling compounds while maintaining molecular weight distribution
    4. Dataset preparation: Generating cleaned datasets for docking workflows

    PIPELINE ENTRY POINT: This step prepares ligand data for docking.

    Required Input Files:
        - input_data: CSV file with columns: ID, SMILES, MolWt

    Output Files (used by downstream steps):
        - {outpath}/ligands/{prefix}_smallMW.csv: Compounds within MW bounds
          (Used by Docking Agent: as input_data parameter in vina_dock() or vina_dock_gpu())
        - {outpath}/ligands/{prefix}_largeMW.csv: Compounds above MW upper bound
        - {outpath}/ligands/{prefix}_tooSmallMW.csv: Compounds below MW lower bound
        - {outpath}/ligands/{prefix}_frac{sampling_frac}.csv: Sampled dataset (if sampling=True)
          (Used by Docking Agent and MD Agent: as ligand_input parameter)

    Use this when you need to:
    - Filter large compound libraries by molecular weight
    - Validate chemical structures before docking
    - Create representative subsets of compound datasets
    - Prepare ligand datasets for high-throughput docking campaigns

    Args:
        input_data (str): Path to an input CSV containing compound records.
            Required columns: ID, SMILES, MolWt
            Example: "data/compounds.csv"

        molwt_upper_bound (float): Maximum molecular weight (Daltons) allowed for
            compounds retained in the cleaned dataset. Compounds above this
            threshold are removed. Default: 700

        molwt_lower_bound (float): Minimum molecular weight (Daltons) allowed for
            compounds retained in the cleaned dataset. Compounds below this
            threshold are removed. Default: 0

        check_rdmol (bool): If True, construct RDKit molecules from SMILES
            to validate chemical correctness and filter out molecules that
            fail sanitization or parsing. Can be computationally expensive for
            large datasets. Default: False

        sampling (bool): If True, perform sampling to produce a smaller set
            of compounds. When enabled, sampling is typically stratified by
            molecular weight using `binsize` to maintain distributional
            diversity. Default: False

        binsize (int): Bin width (Daltons) used to group compounds by
            molecular weight when `sampling` is enabled. Larger bins mean
            coarser stratification. Default: 100

        sampling_frac (float): Fraction of the total dataset to sample when
            `sampling` is True. For example, 0.01 keeps ~1% of the data.
            Default: 0.01

        output_prefix (str): Base filename prefix for any cleaned/sampled
            outputs written to `outpath` (e.g., "cleaned_data.csv").
            Default: "cleaned_data"

        outpath (str): Directory to write cleaned and sampled datasets.
            Will be created if it does not exist. Default: "./output"

        quick_start (bool): If True, use fast processing path that skips
            detailed metal context analysis. Faster for large datasets but
            provides less detailed metal compound classification.
            Default: False

    Returns:
        Dict[str, str]: Dictionary containing paths to all generated output files.
            Key structure:
            - 'output_file': Path to the main cleaned CSV (smallMW dataset)
            - 'small_mw_file': Path to compounds at/below MW cutoff
            - 'large_mw_file': Path to compounds above MW cutoff (if any)
            - 'sampled_file': Path to sampled dataset (only if sampling=True)
            All files are saved to {outpath}/preprocessing/ligands/ directory.

    Example:
        >>> # Filter compounds by molecular weight
        >>> run_ligand_preprocessing(
        ...     input_data="compounds.csv",
        ...     molwt_upper_bound=600,
        ...     outpath="./cleaned"
        ... )
        # Outputs: "./cleaned/cleaned_data.csv"

        >>> # Filter and sample with RDKit validation
        >>> run_ligand_preprocessing(
        ...     input_data="library.csv",
        ...     molwt_upper_bound=700,
        ...     check_rdmol=True,
        ...     sampling=True,
        ...     sampling_frac=0.05,
        ...     outpath="./sampled"
        ... )
        # Outputs: sampled and validated dataset
    """
    ligand_preprocessor = LigandPreprocessor(
        input_data,
        molwt_upper_bound,
        molwt_lower_bound,
        check_rdmol,
        sampling,
        binsize,
        sampling_frac,
        output_prefix,
        outpath,
        quick_start,
    )
    return ligand_preprocessor.run()


prompt_path = Path(__file__).parent / "preprocessing_agent_prompt.md"
system_prompt = prompt_path.read_text()

preprocessing_agent = Agent(
    model="gpt-5.2",
    name="Receptor and Ligand Preparation Agent",
    tools=[
        read_reference_file,
        run_standardize_ligand_data,
        run_generate_conformers_to_pdbqt,
        run_clean_pdb,
        run_ligand_preprocessing,
        run_python_code,
    ],
    instructions=system_prompt,
)
