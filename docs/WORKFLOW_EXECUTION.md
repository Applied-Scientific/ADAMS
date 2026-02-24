# Workflow Execution Documentation

This document provides detailed information about workflow execution, module execution order, file paths, and output organization. For agent and tool information, see `AGENTS_DOCUMENTATION.md`.

## Table of Contents

1. [Pipeline Stages](#pipeline-stages)
2. [Execution Order](#execution-order)
3. [File Paths and Output Organization](#file-paths-and-output-organization)
4. [Entry Points](#entry-points)

---

## Pipeline Stages

The pipeline consists of two main stages:

1. **Preprocessing Stage**: Cleans receptor PDBs and processes ligand CSVs
2. **Docking Stage**: Discovers binding sites and performs molecular docking

---

## Execution Order

### Docking Agent Execution Order

**Strict Sequence:**
1. `run_vina_dock` (search_dock mode) - Discover binding sites
2. `run_find_pocket` - Cluster search results
3. `run_vina_dock` (ligands mode) OR `run_vina_dock_gpu` - Production docking

*MD / stability analysis stage: coming soon in a future release.*

---

## File Paths and Output Organization

### Output Directory Structure

All outputs are organized in timestamped directories: `agent_data/outputs/run_YYYYMMDD_HHMMSS/`

```
output_folder/
├── preprocessing/
│   ├── receptors/          # Cleaned and protonated protein PDB files
│   │   ├── {protein_name}_{chain}_clean.pdb        # Cleaned (no hydrogens)
│   │   └── {protein_name}_{chain}_protonated.pdb    # Protonated (with pKa-based hydrogens)
│   ├── ligands/            # Processed ligand files
│   │   ├── metal_compounds.csv
│   │   ├── metal_organic_compounds.csv
│   │   ├── {prefix}_largeMW.csv
│   │   ├── {prefix}_smallMW.csv
│   │   ├── {prefix}_frac{sampling_frac}.csv
│   │   └── {protein_name}_{ligand_set}.pdb
│
├── docking/
│   ├── search/             # Search docking mode outputs
│   │   ├── poses/          # Individual docking pose files (PDBQT)
│   │   │   └── ligand_{idx}_grid_{grid_id}_docked.pdbqt
│   │   ├── summaries/      # Summary files
│   │   │   ├── best_search_docking_centers.csv
│   │   │   ├── best_search_docking_centers.pdb
│   │   │   ├── dock_sites_clustered.csv
│   │   │   ├── dock_sites_clustered.pdb
│   │   │   ├── cluster_summary.csv
│   │   │   └── docking_centers.csv
│   │   └── metadata/       # Docking metadata
│   │       └── dock_metadata.pkl
│   │
│   └── production/         # Production docking mode outputs
│       ├── poses/          # Individual docking pose files (PDBQT)
│       │   └── ligand_{idx}_pocket_{pocket_id}_docked.pdbqt
│       ├── summaries/      # Summary files
│       │   ├── best_docking_centers.csv
│       │   └── production_docking_results.csv
│       └── metadata/       # Docking metadata
│           └── dock_metadata.pkl
```

### Preprocessing Outputs

**`run_clean_pdb` outputs:**
- `{outpath}/preprocessing/receptors/{prefix}_{chain}_clean.pdb` - Cleaned protein PDB (no hydrogens)
- `{outpath}/preprocessing/ligands/{prefix}_{ligand_set_name}.pdb` - Extracted ligand (if `ligand=True`)

**`run_protonate_receptor` outputs (MANDATORY after run_clean_pdb):**
- `{outpath}/preprocessing/receptors/{prefix}_{chain}_protonated.pdb` - Protonated protein PDB (with pKa-based hydrogens, used by Docking Agent)
- `{outpath}/preprocessing/receptors/{prefix}_{chain}_protonated.pqr` - PQR file with charges

**`run_data_processing` outputs:**
- `{outpath}/preprocessing/ligands/{prefix}_smallMW.csv` - Compounds at/below MW cutoff (used by Docking Agent)
- `{outpath}/preprocessing/ligands/{prefix}_largeMW.csv` - Compounds above MW cutoff
- `{outpath}/preprocessing/ligands/{prefix}_frac{sampling_frac}.csv` - Sampled dataset (if `sampling=True`, used by Docking Agent)

### Docking Outputs

**Search Docking (`run_vina_dock` with `mode="search"`):**
- `{out_folder}/docking/search/summaries/best_search_docking_centers.csv` - Top 100 poses from search docking

**Pocket Finding (`run_find_pocket`):**
- `{out_path}/docking/search/summaries/cluster_summary.csv` - Statistics for all clusters
- `{out_path}/docking/search/summaries/dock_sites_clustered.csv` - All poses assigned to clusters
- `{out_path}/docking/search/summaries/docking_centers.csv` - Top N binding pocket coordinates (used for production docking)

**Production Docking (`run_vina_dock` or `run_vina_dock_gpu`):**
- `{out_folder}/docking/production/summaries/production_docking_results.csv` - Best pose per pocket from production docking
- `{out_folder}/docking/production/poses/ligand_{idx}_pocket_{pocket_id}_docked.pdbqt` - Individual docking pose files

**Important Notes:**
- Search docking uses `grid_{grid_id}` in filenames (e.g., `ligand_0_grid_5_docked.pdbqt`)
- Production docking uses `pocket_{pocket_id}` in filenames (e.g., `ligand_0_pocket_2_docked.pdbqt`)
---

## Entry Points

The pipeline can start from any of 7 entry points, allowing users to resume from any stage:

1. **Entry Point 1 (Preprocessing)**: Full preprocessing - clean receptor PDB and process ligand CSV
   - Required: `input_pdb` (raw PDB), `input_data` (raw SMILES CSV)

2. **Entry Point 2 (Search Docking)**: Search docking - discover binding sites + production docking
   - Required: Cleaned receptor, SMILES CSV

3. **Entry Point 3 (Production Docking)**: Production docking only - dock at known binding sites
   - Required: Cleaned receptor, SMILES CSV, docking_centers or docking_centers_file

---

## Trace and Log Files

### Trace Files

- **Location:** `agent_data/traces/`
- **Format:** JSONL (one JSON object per line)
- **Naming:** `trace_YYYYMMDD_HHMMSS.jsonl`
- **Content:** Detailed execution logs of agent interactions, tool calls, and workflow execution
- **Real-time:** Written in real-time during execution

### Log Files

- **Location:** `agent_data/logs/`
- **Format:** Text log files
- **Naming:** `adams_pipeline_run_{run_identifier}.log`
- **Content:** Pipeline operations, errors, and status messages
- **Note:** Log file name should match the run directory name for consistency

---

## Notes

- **Run Directories:** Created with format `agent_data/outputs/run_YYYYMMDD_HHMMSS/` where timestamp matches trace file naming
- **Resuming Runs:** When resuming, use the existing output_folder from trace analysis instead of creating a new directory
- **File Path Mapping:** See `adams/pipeline/references/file_path_mapping.md` for detailed file path mapping rules between agents
- **Parameter Defaults:** See `adams/pipeline/references/parameter_defaults.md` for default parameter values and guidelines
