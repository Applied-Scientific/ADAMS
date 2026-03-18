"""Interactive agent wrapper for receptor preparation and ligand data processing workflows."""

from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import pandas as pd
from agents import Agent, function_tool

from ..references.reference_file_reader import read_reference_file
from ...model_config import get_current_model_name, get_resolved_model
from ...user_plan_utils import (
    append_to_plan_section,
    contribute_stage_to_plan,
    read_plan_document,
)
from ..charge_model import validate_charge_model
from .clean_pdb import CleanPDB
from .ligand_preprocessing import LigandPreprocessor
from .protonation import run_pdb2pqr
from .python_executor import run_python_in_conda as _run_python_in_conda
from .conformer_generation import generate_conformers_to_pdbqt
from .standardize_ligands import (
    UnsupportedFormatError,
    convert_3d_to_pdbqt,
    detect_ligand_format,
    standardize_2d_to_csv,
)


@function_tool
def run_standardize_ligand_data(
    input_file: str,
    output_dir: str = "output",
    id_col: str = "ID",
    smiles_col: str = "SMILES",
    charge_model: str = "gasteiger",
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
    - OPTIONAL: run run_ligand_preprocessing for filtering/sampling/microstate enumeration
    - REQUIRED: run run_smiles_to_pdbqt before docking

    Args:
        input_file: Path to input file (any format).
        output_dir: Output directory. Default: "output"
        id_col: Name of ID column (if CSV). Default: "ID"
        smiles_col: Name of SMILES column (if CSV). Default: "SMILES"
        charge_model: Meeko partial charge model for 3D→PDBQT (default: "gasteiger"). Must match receptor.

    Returns:
        dict: {
            'format_type': str,   # '2d' or '3d'
            'output_path': str,   # CSV path (2D) or mapping CSV path (3D); pass as input_data to docking
            'num_molecules': int,
            'message': str         # Guidance for next steps
        }

    Raises:
        UnsupportedFormatError: If format cannot be handled (use run_python_code for custom conversion)
    """
    charge_model = validate_charge_model(charge_model)
    print(f"[Agent] Standardizing ligand data from {input_file}...")
    # Detect format
    format_info = detect_ligand_format(input_file)

    if format_info["has_3d"]:
        # 3D pathway: Convert to PDBQT and write mapping CSV
        mapping_csv = convert_3d_to_pdbqt(input_file, output_dir, charge_model=charge_model)
        num_molecules = len(pd.read_csv(mapping_csv))

        return {
            "format_type": "3d",
            "output_path": mapping_csv,
            "num_molecules": num_molecules,
            "message": (
                f"Converted {num_molecules} 3D structures to PDBQT. "
                "SKIP run_ligand_preprocessing and run_smiles_to_pdbqt. "
                "Pass the returned output_path as input_data to the docking agent."
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
                "OPTIONAL: run run_ligand_preprocessing for filtering/sampling/microstate enumeration. "
                "REQUIRED: run run_smiles_to_pdbqt before docking."
            ),
        }


@function_tool
def run_smiles_to_pdbqt(
    input_csv: str,
    output_dir: str = "output",
    num_confs: int = 8,
    max_confs_to_keep: int = 2,
    conformer_energy_window_kcal: float = 3.0,
    random_seed: int = 42,
    charge_model: str = "gasteiger",
) -> str:
    """
    Converts a SMILES CSV to docking-ready PDBQT files and a mapping CSV.

    Use this after run_standardize_ligand_data returns format_type='2d'.
    This is REQUIRED before docking 2D inputs.

    The function:
    1. Reads SMILES from the CSV (ID, SMILES, optionally Variant_ID)
    2. Generates 3D conformers per molecule (RDKit ETKDG) and converts to PDBQT
    3. Writes a mapping CSV (docking_ready_ligands.csv)
    4. Returns the path to that mapping CSV for use as docking input_data

    Args:
        input_csv: Path to CSV with SMILES (from run_standardize_ligand_data or run_ligand_preprocessing)
        output_dir: Output directory for PDBQT files and mapping CSV. Default: "output"
        num_confs: Number of conformers to generate per molecule before pruning (default: 8).
        max_confs_to_keep: Maximum low-energy conformers retained per microstate (default: 2).
        conformer_energy_window_kcal: Keep conformers within this energy window from best (default: 3.0).
        random_seed: Random seed for reproducible conformer generation (default: 42).
        charge_model: Meeko partial charge model (default: "gasteiger"). Must match receptor.

    Returns:
        str: Path to the mapping CSV (ID, PDBQT_File, ...). Pass this path as input_data to the docking agent.
    """
    charge_model = validate_charge_model(charge_model)
    print(f"[Agent] Converting SMILES CSV to PDBQT from {input_csv} (num_confs={num_confs})...")
    return generate_conformers_to_pdbqt(
        input_csv,
        output_dir,
        num_confs=num_confs,
        max_confs_to_keep=max_confs_to_keep,
        conformer_energy_window_kcal=conformer_energy_window_kcal,
        random_seed=random_seed,
        charge_model=charge_model,
    )


@function_tool
def run_python_code(code: str) -> Dict[str, str]:
    """
    Generate and execute a Python code snippet in the selected runtime environment.

    This tool allows for custom data manipulation, analysis, or fix-ups that are not covered
    by the standard preprocessing tools. It executes code in a sandboxed environment with
    access to the installed project dependencies (rdkit, pandas, numpy, etc.).

    Constraints & Safety:
    - Only specific safe modules and installed packages are allowed (numpy, pandas, rdkit, etc.).
    - File deletion and system modification (os.remove, shutil.rmtree) are BLOCKED.
    - Network access is NOT available.
    - By default runs in conda env 'adams' (or current CONDA_DEFAULT_ENV).
    - If `conda` is unavailable but the target env is already active, falls back to
      the current Python interpreter.

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
    chain_to_keep: Optional[Union[str, Sequence[str]]] = "all",
    residue_range_start: Optional[int] = None,
    residue_range_end: Optional[int] = None,
    keep_water: bool = False,
    keep_heterogens: Optional[Union[Sequence[str], str]] = "essential",
    model_missing_residues: bool = True,
    max_missing_residues_per_gap: int = 12,
    allow_terminal_missing_residues: bool = False,
    pH: float = 7.4,
) -> str:
    r"""
    Clean and optionally augment a receptor PDB using the ``CleanPDB`` pipeline.

    This function prepares a protein structure by:
    1. Chain selection: Retaining only the specified chain identifier
    2. Heterogen removal: Removing or keeping heterogens/waters
    3. Structure completion: Adding missing atoms (no hydrogens)
    4. Ligand extraction: Optionally preserving or extracting bound ligands

    PIPELINE ENTRY POINT: Prepares receptor structure for docking. This operation is independent of ligand data processing and can be run in any order.

    Required Input Files:
        - input_pdb: Any valid PDB file (can be from PDB database or experimental)

    Output Files (used by downstream steps):
        - {outpath}/receptors/{prefix}_{chain}_clean.pdb: Cleaned protein PDB (no hydrogens)
          (Used by: Protonation step via run_protonate_receptor, then Docking/MD)
        - {outpath}/ligands/{prefix}_{ligand_set_name}.pdb: Extracted ligand (if ligand=True)

    Use this when you need to:
    - Prepare a receptor structure for molecular docking
    - Remove unwanted chains and water molecules (waters removed by default; use keep_water=True to retain structural waters)
    - Keep cofactor-dependent binding sites (default keep_heterogens="essential" keeps heme, metals, NAD, etc.; use keep_heterogens=None to strip all)
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

        chain_to_keep: Chain selector for receptor cleanup.
            - "all" or None (default): keep all chains
            - "A": keep one chain
            - "A,B,C" or ["A","B","C"]: keep selected chains
        residue_range_start: Optional inclusive lower bound for residue sequence
            numbers (PDB `resseq`) applied to ATOM/HETATM filtering before fixing.
            Example: 544.
        residue_range_end: Optional inclusive upper bound for residue sequence
            numbers (PDB `resseq`). Example: 667.

        keep_water (bool): If True, retain water molecules (e.g. structural waters).
            Default: False.

        keep_heterogens: "essential" (default) = keep ESSENTIAL_HETEROGENS_TO_KEEP. None or [] = remove all.
            A list (e.g. ["HEM", "MG"]) or single str (e.g. "HEM") = keep only those.
        model_missing_residues (bool): If True (default), model selected missing-residue
            blocks during cleanup.
        max_missing_residues_per_gap (int): Safety cap on modeled gap size. Larger
            gaps are left as chain breaks. Default: 12.
        allow_terminal_missing_residues (bool): If False (default), do not model
            missing N/C terminal stretches.

        pH (float): pH value (default: 7.4). Passed to run_protonate_receptor.

    Returns:
        str: Full path to cleaned PDB file (no hydrogens).
            Use run_protonate_receptor to add hydrogens.

    Example:
        >>> # Clean a receptor structure
        >>> run_clean_pdb(
        ...     input_pdb="protein.pdb",
        ...     outpath="./cleaned",
        ...     chain_to_keep="A,B,C",
        ...     residue_range_start=544,
        ...     residue_range_end=667
        ... )
        # Outputs: "./cleaned/cleaned_protein.pdb"

        >>> # Clean and extract ligand
        >>> run_clean_pdb(
        ...     input_pdb="complex.pdb",
        ...     outpath="./cleaned",
        ...     ligand=True,
        ...     chain_to_keep="all"
        ... )
        # Outputs: cleaned protein and extracted ligand files

        >>> # Optional: retain structural waters (default is keep_water=False). Essential cofactors (e.g. HEM) are kept by default.
        >>> run_clean_pdb(
        ...     input_pdb="hemoprotein.pdb",
        ...     outpath="./cleaned",
        ...     keep_water=True
        ... )
        >>> # Remove all heterogens (strip cofactors)
        >>> run_clean_pdb(..., keep_heterogens=None)
    """
    cleaner = CleanPDB(
        input_pdb=input_pdb,
        outpath=outpath,
        ligand=ligand,
        chain_to_keep=chain_to_keep,
        residue_range_start=residue_range_start,
        residue_range_end=residue_range_end,
        keep_water=keep_water,
        keep_heterogens=keep_heterogens,
        model_missing_residues=model_missing_residues,
        max_missing_residues_per_gap=max_missing_residues_per_gap,
        allow_terminal_missing_residues=allow_terminal_missing_residues,
        pH=pH,
    )
    print(f"[Agent] Cleaning PDB structure: {input_pdb} (Chains: {chain_to_keep})...")
    return cleaner.clean()


@function_tool
def run_protonate_receptor(
    input_pdb: str,
    outpath: str = "./output",
    pH: float = 7.4,
    ff: str = "AMBER",
    ffout: str = "AMBER",
    warning_strict: bool = False,
) -> Dict[str, str]:
    """
    Protonate receptor PDB using PDB2PQR with PROPKA.

    Notes:
    - HETATM records (e.g., kept crystallographic waters/cofactors from run_clean_pdb)
      are preserved in the protonated PDB output.

    MANDATORY step after run_clean_pdb. Must be called before docking/MD.

    Args:
        input_pdb: Path to cleaned PDB file (from run_clean_pdb, without hydrogens)
        outpath: Output directory (default: "./output")
        pH: pH value for protonation (default: 7.4)
        ff: Force field (default: "AMBER")
        ffout: Output force field (default: "AMBER")
        warning_strict: If True, fail when critical protonation warnings are detected.
            Default False keeps warnings non-blocking while still writing warning reports.

    Returns:
        Dict[str, str]: {
            'protonated_pdb': path,
            'protonated_pqr': path,
            'pdb2pqr_warnings_csv': path,
            'pdb2pqr_warning_summary': path,
        }
    """
    import os
    from ..file_organization import setup_preprocessing_dirs

    if not os.path.exists(input_pdb):
        raise FileNotFoundError(f"Input PDB file not found: {input_pdb}")

    # Set up organized directory structure
    dir_structure = setup_preprocessing_dirs(outpath)
    os.makedirs(dir_structure["receptors"], exist_ok=True)

    # Generate output filenames
    input_prefix = os.path.splitext(os.path.basename(input_pdb))[0]
    # Remove _clean suffix if present
    if input_prefix.endswith("_clean"):
        input_prefix = input_prefix[:-6]
    
    output_pqr = os.path.join(
        dir_structure["receptors"], f"{input_prefix}_protonated.pqr"
    )
    output_pdb = os.path.join(
        dir_structure["receptors"], f"{input_prefix}_protonated.pdb"
    )

    print(
        f"[Agent] Protonating receptor with PDB2PQR+PROPKA (pH={pH}): "
        f"{input_pdb} -> {output_pdb}"
    )

    # Run PDB2PQR with PROPKA.
    # Backward compatibility: older runtime copies of run_pdb2pqr do not accept
    # warning_strict; retry once without the keyword if needed.
    try:
        (
            protonated_pdb,
            protonated_pqr,
            warnings_csv_path,
            warning_summary_path,
        ) = run_pdb2pqr(
            input_pdb=input_pdb,
            output_pqr=output_pqr,
            output_pdb=output_pdb,
            pH=pH,
            ff=ff,
            ffout=ffout,
            warning_strict=warning_strict,
        )
    except TypeError as e:
        if "unexpected keyword argument 'warning_strict'" not in str(e):
            raise
        print(
            "[Agent] Runtime uses legacy run_pdb2pqr signature; "
            "retrying protonation without warning_strict."
        )
        protonated_pdb, protonated_pqr = run_pdb2pqr(
            input_pdb=input_pdb,
            output_pqr=output_pqr,
            output_pdb=output_pdb,
            pH=pH,
            ff=ff,
            ffout=ffout,
        )
        warnings_csv_path = ""
        warning_summary_path = ""

    return {
        "protonated_pdb": protonated_pdb,
        "protonated_pqr": protonated_pqr,
        "pdb2pqr_warnings_csv": warnings_csv_path,
        "pdb2pqr_warning_summary": warning_summary_path,
    }


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
    enumerate_microstates: bool = True,
    enumerate_tautomers: bool = True,
    enumerate_protonation: bool = True,
    enumerate_stereoisomers: bool = True,
    pH_min: float = 6.4,
    pH_max: float = 8.4,
    protonation_precision: float = 0.5,
    max_generated_tautomers: Optional[int] = 64,
    top_tautomers_per_protomer: int = 2,
    tautomer_energy_window_kcal: float = 3.0,
    max_protomers: int = 16,
    max_stereoisomers: int = 16,
    max_unassigned_stereocenters: int = 2,
    max_total_microstates: int = 64,
    enumerate_all_stereocenters: bool = False,
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

        enumerate_microstates (bool): If True, enumerate microstates (tautomers,
            protomers, stereoisomers) BEFORE molecular weight filtering.
            Default: True

        enumerate_tautomers (bool): Enumerate tautomers when enumerate_microstates=True.
            Default: True

        enumerate_protonation (bool): Enumerate protonation states when enumerate_microstates=True.
            Default: True

        enumerate_stereoisomers (bool): Enumerate stereoisomers when enumerate_microstates=True.
            Default: True

        pH_min (float): Minimum pH for protonation state enumeration (default: 6.4)

        pH_max (float): Maximum pH for protonation state enumeration (default: 8.4)

        protonation_precision (float): Dimorphite precision for protomer expansion (default: 0.5)

        max_generated_tautomers (Optional[int]): Optional hard cap for generated tautomers per protomer (default: 64; set None to disable)

        top_tautomers_per_protomer (int): Number of top-ranked tautomers retained (default: 2)

        tautomer_energy_window_kcal (float): Tautomer retention window from best energy (default: 3.0)

        max_protomers (int): Maximum protomers per ligand (default: 16)

        max_stereoisomers (int): Maximum stereoisomers per ligand (default: 16)

        max_unassigned_stereocenters (int): Skip stereo expansion when unassigned centers exceed this (default: 2)

        max_total_microstates (int): Maximum microstates per parent ligand,
            including original (default: 64)

        enumerate_all_stereocenters (bool): If True, enumerate all stereocenters
            (assigned + unassigned). If False (default), enumerate unassigned only.

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
        input_data=input_data,
        molwt_upper_bound=molwt_upper_bound,
        molwt_lower_bound=molwt_lower_bound,
        check_rdmol=check_rdmol,
        sampling=sampling,
        binsize=binsize,
        sampling_frac=sampling_frac,
        output_prefix=output_prefix,
        outpath=outpath,
        quick_start=quick_start,
        enumerate_microstates=enumerate_microstates,
        enumerate_tautomers=enumerate_tautomers,
        enumerate_protonation=enumerate_protonation,
        enumerate_stereoisomers=enumerate_stereoisomers,
        pH_range=(pH_min, pH_max),
        protonation_precision=protonation_precision,
        max_generated_tautomers=max_generated_tautomers,
        top_tautomers_per_protomer=top_tautomers_per_protomer,
        tautomer_energy_window_kcal=tautomer_energy_window_kcal,
        max_protomers=max_protomers,
        max_stereoisomers=max_stereoisomers,
        max_unassigned_stereocenters=max_unassigned_stereocenters,
        max_total_microstates=max_total_microstates,
        enumerate_all_stereocenters=enumerate_all_stereocenters,
    )
    print(f"[Agent] Preprocessing ligand library: {input_data} (MW < {molwt_upper_bound})...")
    return ligand_preprocessor.run()


prompt_path = Path(__file__).parent / "preprocessing_agent_prompt.md"
system_prompt = prompt_path.read_text()

_preprocessing_agent = None
_preprocessing_model = None


def get_preprocessing_agent():
    global _preprocessing_agent, _preprocessing_model
    current_model = get_current_model_name()
    if _preprocessing_agent is None or _preprocessing_model != current_model:
        _preprocessing_agent = Agent(
            model=get_resolved_model(),
            name="Receptor and Ligand Preparation Agent",
            tools=[
                read_reference_file,
                read_plan_document,
                append_to_plan_section,
                contribute_stage_to_plan,
                run_standardize_ligand_data,
                run_smiles_to_pdbqt,
                run_clean_pdb,
                run_protonate_receptor,
                run_ligand_preprocessing,
                run_python_code,
            ],
            instructions=system_prompt,
        )
        _preprocessing_model = current_model
    return _preprocessing_agent
