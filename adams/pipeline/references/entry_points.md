# Pipeline Entry Points

## Overview
The pipeline supports starting from ANY step, not just the beginning. This enables resuming interrupted runs, using external data, or running specific stages only.

## Entry Point Definitions

### Entry Point 1: Preprocessing
**Purpose**: Prepare raw input data for docking
**Required Files**:
- raw_receptor.pdb (raw PDB file)
- ligand_input (SMILES CSV, SDF, MOL2, PDB, or PDBQT files)

**What It Does** (two independent operations that can be run in any order):
- Receptor Preparation: Cleans receptor PDB (removes unwanted chains, water, adds hydrogens)
- Ligand Preparation:
  - Detects input format (2D vs 3D)
  - 2D pathway (SMILES): standardize → generate conformers → convert to PDBQT
  - 3D pathway (SDF/MOL2/PDB): validate 3D coords → convert to PDBQT
  - Creates CSV mapping with ID and PDBQT_File columns

**Outputs**:
- Cleaned receptor: {outpath}/preprocessing/receptors/{protein_name}_{chain}_clean_h.pdb
- PDBQT files: {outpath}/preprocessing/ligands/conformers_pdbqt/ or 3d_structures_pdbqt/
- Mapping CSV: Must contain columns: ID, PDBQT_File

**Next Steps**: Proceeds to Search Docking (Entry Point 2)

---

### Entry Point 2: Search Docking
**Purpose**: Discover binding sites and dock ligands
**Required Files**:
- cleaned_receptor.pdb or cleaned_receptor.pdbqt
- ligands_csv: CSV with columns: ID, PDBQT_File (all ligands pre-prepared as PDBQT)

**What It Does**:
- Discovers binding sites via search docking (CPU or GPU)
- Runs production docking at top N sites

**Outputs**:
- Docking centers: {out_folder}/docking/search/summaries/docking_centers.csv
- Production results: {out_folder}/docking/production/summaries/production_docking_results.csv

**Next Steps**: Proceeds to MD Analysis (Entry Point 4)

**IMPORTANT**: All ligands must be pre-prepared as PDBQT files before docking. If starting with raw SMILES or 3D structures, first run preprocessing (Entry Point 1).

---

### Entry Point 3: Production Docking Only
**Purpose**: Dock at known binding sites (skip discovery)
**Required Files**:
- cleaned_receptor.pdb or cleaned_receptor.pdbqt
- ligands_csv: CSV with columns: ID, PDBQT_File (all ligands pre-prepared as PDBQT)
- docking_centers OR docking_centers_file

**What It Does**:
- Skips search/discovery step
- Runs production docking at provided binding sites

**Outputs**:
- Production results: {out_folder}/docking/production/summaries/production_docking_results.csv

**Next Steps**: Proceeds to MD Analysis (Entry Point 4)

**IMPORTANT**: All ligands must be pre-prepared as PDBQT files before docking. If starting with raw SMILES or 3D structures, first run preprocessing (Entry Point 1).

---

### Entry Point 4: MD - Protein Topology
**Purpose**: Start MD pipeline from protein topology generation
**Required Files**:
- cleaned_receptor.pdb
- docking_results folder (with production_docking_results.csv)
- ligand_input (SMILES string, CSV, SDF, or MOL2 file)

**What It Does**:
- Generates protein topology (protein.gro, topol.top)
- Prepares ligands from docking results
- Runs MD simulations
- Performs stability analysis

**Outputs**:
- Protein topology: {md_workdir}/md_analysis/protein/protein.gro, topol.top
- Prepared poses: {md_workdir}/md_analysis/poses/{ligand_name}_pocket_{grid_id}_top{rank}/
- Analysis reports: {md_workdir}/md_analysis/reports/

---

### Entry Point 5: MD - Ligand Preparation
**Purpose**: Start MD pipeline from ligand preparation (skip protein topology)
**Required Files**:
- protein.gro
- topol.top
- docking_results folder
- ligand_input (SMILES string, CSV, SDF, or MOL2 file)

**What It Does**:
- Skips protein topology generation
- Prepares ligands from docking results
- Runs MD simulations
- Performs stability analysis

**Outputs**: Same as Entry Point 4 (poses and reports)

---

### Entry Point 6: MD - Gro (MD Simulations)
**Purpose**: Start from MD simulations (skip topology and ligand prep)
**Required Files**:
- pose_directories with min.gro, system.top, index.ndx

**What It Does**:
- Skips topology and ligand preparation
- Runs MD simulations on prepared poses
- Performs stability analysis

**Outputs**: MD trajectories and analysis reports

---

### Entry Point 7: MD - Stability Analysis Only
**Purpose**: Run only stability analysis on completed MD trajectories
**Required Files**:
- pose_directories with md.tpr, md.xtc, md.gro, index files

**What It Does**:
- Skips all previous steps
- Analyzes completed MD trajectories
- Generates stability reports

**Outputs**: Analysis reports only

---

## Entry Point Detection Signals

**Preprocessing (Entry Point 1)**:
- User says: "raw", "needs cleaning", "clean the receptor", "have SMILES", "have SDF files"
- File clues: PDB file without "_clean" or "_h" in name, or ligand files without PDBQT format

**Search Docking (Entry Point 2)**:
- User says: "already cleaned", "prepared", "discover binding sites", "have PDBQT files"
- File clues: Receptor with "_clean" or "_h", CSV with PDBQT_File column, no docking_centers

**Production Docking (Entry Point 3)**:
- User says: "binding sites known", "dock at these coordinates", "use centers"
- File clues: Has docking_centers or docking_centers_file, CSV with PDBQT_File column

**MD - Protein Topology (Entry Point 4)**:
- User says: "docking is complete", "have docking results", "run MD from docking"
- File clues: Has docking folder, no protein.gro

**MD - LigPrepare (Entry Point 5)**:
- User says: "have protein topology", "protein.gro ready", "skip protein topology"
- File clues: Has protein.gro AND topol.top

**MD - Gro (Entry Point 6)**:
- User says: "ligands are prepared", "have pose directories", "run MD simulations"
- File clues: Has poses/ directory with min.gro, system.top, index.ndx

**MD - Stability Analysis (Entry Point 7)**:
- User says: "MD is complete", "analyze trajectories", "run stability analysis"
- File clues: Has md.tpr, md.xtc, md.gro files in pose directories

---

## Quick Reference Table

| Entry Point | Required Files | Skips |
|------------|----------------|-------|
| 1. Preprocessing | raw_receptor.pdb, ligand_input (SMILES/SDF/MOL2/PDB) | None |
| 2. Search Docking | cleaned_receptor, CSV with ID+PDBQT_File | Preprocessing |
| 3. Production Docking | cleaned_receptor, CSV with ID+PDBQT_File, docking_centers | Preprocessing, Search |
| 4. MD-ProteinTopo | cleaned_receptor, docking_results, smiles.csv | Preprocessing, Docking |
| 5. MD-LigPrepare | protein.gro, topol.top, docking_results, smiles.csv | Preprocessing, Docking, ProteinTopo |
| 6. MD-Gro | pose_dirs (with min.gro, system.top, index.ndx) | All previous MD steps |
| 7. MD-Analysis | pose_dirs (with md.tpr, md.xtc, md.gro) | All MD steps except analysis |
