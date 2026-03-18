# File Path Mapping Between Agents

## Overview
This document describes how file paths flow between pipeline agents. After each agent completes, extract the exact paths from outputs and pass them to the next agent.

## Preprocessing → Docking Mapping

### Receptor File Path
**Source**: `preprocessing_agent.run_protonate_receptor()` output (MANDATORY after `run_clean_pdb`)
- **Returns**: Dictionary with `'protonated_pdb'` key containing path to protonated PDB file
- **Location**: `{outpath}/preprocessing/receptors/{protein_name}_{chain}_protonated.pdb`
- **Example**: `/path/to/output/preprocessing/receptors/protein_A_protonated.pdb`
- **Workflow**: `run_clean_pdb()` → `run_protonate_receptor()` → docking/MD
  - `run_clean_pdb()` outputs: `{prefix}_{chain}_clean.pdb` (no hydrogens)
  - `run_protonate_receptor()` outputs: `{prefix}_{chain}_protonated.pdb` (with pKa-based hydrogens)

**Target**: `docking_agent` input
- **Parameter**: `receptor` (str)
- **Mapping**: Use the `'protonated_pdb'` path from `run_protonate_receptor()` output
- **Note**: Docking agent can convert PDB to PDBQT automatically if needed (protonate=False by default, expects pre-protonated file)

### Ligand Data Path
**Source**: `preprocessing_agent.run_standardize_ligand_data()` output
- **Returns**: Dictionary with keys:
  - `'format_type'`: '2d' or '3d' indicating input format
  - `'output_path'`: Path to output (CSV for 2D, path to mapping CSV for 3D)
  - `'num_molecules'`: Number of molecules processed
  - `'message'`: Status message

**2D Pathway (SMILES)**: Conformer generation is preprocessing-only
- run_standardize_ligand_data → CSV with SMILES
- run_smiles_to_pdbqt → **path to mapping CSV** (docking_ready_ligands.csv with ID, PDBQT_File, optionally Variant_ID, Conformer_Index). Uses row-unique filenames (Variant_ID when present, else ID) so enumerated microstates get distinct PDBQT files.
- Pass that mapping CSV path to docking as input_data

**3D Pathway (SDF/MOL2/PDB)**: Direct to PDBQT and mapping CSV
- run_standardize_ligand_data → **path to mapping CSV** (docking_ready_ligands.csv with ID, PDBQT_File). PDBQT files written to {output_dir}/pdbqt_files/
- Pass that mapping CSV path to docking as input_data

**Target**: `docking_agent` input
- **Parameters**: `input_data` (str - path to CSV file)
- **Required CSV Columns**:
  - `ID`: Ligand identifier
  - `PDBQT_File`: Absolute path to PDBQT file
- **CRITICAL**: All ligands MUST be pre-prepared as PDBQT files before docking
- **Location**: PDBQT files in `{output_dir}/pdbqt_files/`; mapping CSV at `{output_dir}/docking_ready_ligands.csv` (2D and 3D).

**Example**:
```python
# 2D pathway output
{
    'format_type': '2d',
    'output_path': '/output/preprocessing/ligands/cleaned_data.csv',  # Contains SMILES
    'num_molecules': 100
}
# Then: mapping_csv = run_smiles_to_pdbqt(csv_path, output_dir)
# Returns: path to docking_ready_ligands.csv (ID, PDBQT_File, ...)
# Use mapping_csv as input_data for docking

# 3D pathway output
{
    'format_type': '3d',
    'output_path': '/output/docking_ready_ligands.csv',  # mapping CSV (ID, PDBQT_File)
    'num_molecules': 50
}
# Use output_path as input_data for docking

# Docking input (must have these columns)
# CSV format:
# ID,PDBQT_File
# lig_0,/absolute/path/to/pdbqt_files/lig_0.pdbqt
# lig_1,/absolute/path/to/pdbqt_files/lig_1.pdbqt
input_data = '/output/preprocessing/ligands/ligands_with_pdbqt_paths.csv'
```

## Docking → MD Mapping

### Receptor File Path
**Source**: Same as preprocessing output (protonated receptor)
- **Path**: `{out_folder}/preprocessing/receptors/{protein_name}_{chain}_protonated.pdb`
- **Note**: Must be protonated (from `run_protonate_receptor()`), not just cleaned

**Target**: `md_agent` input
- **Parameter**: `protein_file` (for ProteinTopologyConfig)
- **Mapping**: Use the protonated receptor path from preprocessing (`run_protonate_receptor()` output)

### Docking Results Folder
**Source**: Docking agent output
- **Location**: Root docking output folder (e.g., `{out_folder}/`)
- **System automatically searches**:
  - `{source_folder}/docking/production/summaries/` (preferred)
  - `{source_folder}/docking/search/summaries/` (fallback)

**Target**: `md_agent` input
- **Parameter**: `source_folder` (for LigPrepareConfig)
- **Mapping**: Use the ROOT output folder (not subdirectories)
- **Example**: If docking outputs are in `/output/docking/`, use `/output/` as source_folder

### Ligand Structure Input
**Source**: Same as preprocessing output (processed/sampled CSV) or user-provided structure files
- **Path**: Same CSV/structure file used for docking, or user-provided ligand input

**Target**: `md_agent` input
- **Parameter**: `ligand_input` (for LigPrepareConfig)
- **Mapping**: Can be:
  - SMILES string: "CC(=O)O" (for single ligand)
  - CSV file: Path to CSV with SMILES column (prefer sampled > temp_small_mw > small_mw)
  - SDF/MOL2 file: Path to structure file
  - Directory: Path containing structure files

### Docking Centers File
**Source**: Search docking output
- **Location**: `{out_folder}/docking/search/summaries/docking_centers.csv`

**Target**: Production docking input
- **Parameter**: `docking_centers_file`
- **Mapping**: Use exact path from search output

## MD Agent Internal Mapping

### Protein Topology Files
**Source**: ProteinTopologyConfig output
- **Location**: `{md_workdir}/md_analysis/protein/protein.gro`, `topol.top`

**Target**: LigPrepareConfig input (for mid-pipeline start)
- **Parameters**: `protein_gro`, `protein_top`, `water_model` (optional but recommended)
- **Mapping**: Use exact paths from protein topology step. Set `water_model` (e.g. `tip3p`, `spc`) to match the topology so solvation uses the same water model.

### Pose Directories
**Source**: LigPrepareConfig output
- **Location**: `{md_workdir}/md_analysis/poses/{ligand_name}_pocket_{grid_id}_top{rank}/`

**Target**: GroConfig input (for mid-pipeline start)
- **Parameter**: `pose_dirs` (comma-separated directory NAMES only, not full paths)
- **Mapping**: Extract directory names from full paths
- **Example**: If pose is at `/path/md_analysis/poses/T2457_pocket_0_top1/`, use `pose_dirs="T2457_pocket_0_top1"`

### MD Trajectory Files
**Source**: GroConfig output
- **Location**: Inside pose directories: `md.tpr`, `md.xtc`, `md.gro`

**Target**: StabilityAnalysisConfig input (for mid-pipeline start)
- **Parameter**: `pose_dirs_analysis` (comma-separated directory NAMES)
- **Mapping**: Same as pose_dirs (directory names only)

## Critical Mapping Rules

1. **Exact Paths**: Always use EXACT paths returned by functions - do not modify or reconstruct
2. **Dictionary Keys**: Check which keys exist in preprocessing output dictionary before selecting
3. **Directory Names**: For pose_dirs, use only directory NAMES (not full paths)
4. **Root Folders**: For source_folder, use ROOT output folder, not subdirectories
5. **Same Output Root**: All agents should use the SAME output_root directory for consistency

## Common Mapping Patterns

### Full Pipeline Flow
```
preprocessing → docking → md_analysis
  ↓              ↓          ↓
receptor.pdb   receptor   protein_file
ligands.csv    input_data ligand_input
               (same)     source_folder (docking root)
```

### Mid-Pipeline Starts
```
External Docking → MD (Entry Point 4)
  ↓
docking_results_folder → source_folder
smiles.csv → ligand_input
cleaned_receptor → protein_file

Prepared Poses → MD (Entry Point 6)
  ↓
pose_directories → pose_dirs (names only)

Completed MD → Analysis (Entry Point 7)
  ↓
pose_directories → pose_dirs_analysis (names only)
```
