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
- Receptor Preparation: Cleans receptor PDB (removes unwanted chains, water, adds missing atoms) → Protonates using PDB2PQR+PROPKA (pKa-based protonation)
- Ligand Preparation:
  - Detects input format (2D vs 3D)
  - 2D pathway (SMILES): standardize → (optional ligand_preprocessing) → run_smiles_to_pdbqt → mapping CSV (docking_ready_ligands.csv)
  - 3D pathway (SDF/MOL2/PDB): validate 3D coords → convert to PDBQT
  - Conformer generation is preprocessing-only; docking does not generate conformers

**Outputs**:
- Cleaned receptor (no hydrogens): {outpath}/preprocessing/receptors/{protein_name}_{chain}_clean.pdb
- Protonated receptor (with pKa-based hydrogens): {outpath}/preprocessing/receptors/{protein_name}_{chain}_protonated.pdb
- PDBQT files: {output_dir}/pdbqt_files/ (both 2D and 3D pathways)
- Mapping CSV: run_smiles_to_pdbqt returns path to docking_ready_ligands.csv (ID, PDBQT_File, ...). Pass this as input_data to docking.

**CRITICAL**: Receptor preparation requires TWO steps:
1. `run_clean_pdb()` - outputs `_clean.pdb` (no hydrogens)
2. `run_protonate_receptor()` - outputs `_protonated.pdb` (with hydrogens)
Use the `_protonated.pdb` file for docking.

**Next Steps**: Proceeds to Search Docking (Entry Point 2)

---

### Entry Point 2: Search Docking
**Purpose**: Discover binding sites and dock ligands
**Required Files**:
- protonated_receptor.pdb or protonated_receptor.pdbqt (must be protonated, from run_protonate_receptor)
- ligands_csv: CSV with columns: ID, PDBQT_File (all ligands pre-prepared as PDBQT)

**What It Does**:
- Discovers binding sites via search docking (CPU or GPU)
- Runs production docking at top N sites

**Outputs**:
- Docking centers: {out_folder}/docking/search/summaries/docking_centers.csv
- Production results: {out_folder}/docking/production/summaries/production_docking_results.csv

**Next Steps**: End of pipeline (docking complete).

**IMPORTANT**: Docking requires a CSV with a PDBQT_File column. Conformers are generated only in preprocessing. If you have only SMILES/ID, run the preprocessing agent first (run_standardize_ligand_data → run_smiles_to_pdbqt) and use the returned mapping CSV as input_data.

---

### Entry Point 3: Production Docking Only
**Purpose**: Dock at known binding sites (skip discovery)
**Required Files**:
- protonated_receptor.pdb or protonated_receptor.pdbqt (must be protonated, from run_protonate_receptor)
- ligands_csv: CSV with columns: ID, PDBQT_File (all ligands pre-prepared as PDBQT)
- docking_centers OR docking_centers_file

**What It Does**:
- Skips search/discovery step
- Runs production docking at provided binding sites

**Outputs**:
- Production results: {out_folder}/docking/production/summaries/production_docking_results.csv

**Next Steps**: End of pipeline (docking complete).

**IMPORTANT**: Docking requires a CSV with a PDBQT_File column. Conformers are generated only in preprocessing. If you have only SMILES/ID, run the preprocessing agent first (run_standardize_ligand_data → run_smiles_to_pdbqt) and use the returned mapping CSV as input_data.

---

## Entry Point Detection Signals

**Preprocessing (Entry Point 1)**:
- User says: "raw", "needs cleaning", "clean the receptor", "protonate the receptor", "have SMILES", "have SDF files"
- File clues: PDB file without "_clean" or "_protonated" in name, or ligand files without PDBQT format

**Search Docking (Entry Point 2)**:
- User says: "already cleaned", "already protonated", "prepared", "discover binding sites", "have PDBQT files"
- File clues: Receptor with "_protonated" (not just "_clean"), CSV with PDBQT_File column, no docking_centers

**Production Docking (Entry Point 3)**:
- User says: "binding sites known", "dock at these coordinates", "use centers"
- File clues: Has docking_centers or docking_centers_file, CSV with PDBQT_File column

---

## Quick Reference Table

| Entry Point | Required Files | Skips |
|------------|----------------|-------|
| 1. Preprocessing | raw_receptor.pdb, ligand_input (SMILES/SDF/MOL2/PDB) | None |
| 2. Search Docking | cleaned_receptor, CSV with ID+PDBQT_File | Preprocessing |
| 3. Production Docking | cleaned_receptor, CSV with ID+PDBQT_File, docking_centers | Preprocessing, Search |
