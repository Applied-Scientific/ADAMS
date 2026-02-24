# Pipeline Directory Structure

## Overview
All pipeline outputs are automatically organized into a structured directory hierarchy under the output root directory.

## Complete Directory Tree

```
{output_root}/
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
│   │   ├── pdbqt_files/     # PDBQT files (from SMILES conformers or 3D conversion)
│   │   │   └── {clean_id}.pdbqt (or {clean_id}_conf{i}.pdbqt for multi-conformer)
│   │   ├── docking_ready_ligands.csv  # Mapping CSV (ID, PDBQT_File). Pass as input_data to docking.
│   │   └── {protein_name}_{ligand_set}.pdb (if ligand extraction enabled)
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
│   │   │   └── docking_centers.csv  # ← Use for production docking input
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

## Key File Locations

### Preprocessing Outputs
- **Cleaned Receptor (no hydrogens)**: `{outpath}/preprocessing/receptors/{protein_name}_{chain}_clean.pdb`
- **Protonated Receptor (with hydrogens)**: `{outpath}/preprocessing/receptors/{protein_name}_{chain}_protonated.pdb` (use this for docking)
- **Processed Ligands**: `{outpath}/preprocessing/ligands/{prefix}_smallMW.csv` (or sampled version)
- **Sampled Dataset**: `{outpath}/preprocessing/ligands/{prefix}_frac{sampling_frac}.csv` (if sampling enabled)
- **PDBQT Files**: `{output_dir}/pdbqt_files/{clean_id}.pdbqt` (from SMILES conformers or 3D conversion)
- **Mapping CSV**: `{output_dir}/docking_ready_ligands.csv` (ID, PDBQT_File). Pass this path as **input_data** to the docking agent.

### Docking Outputs
- **Docking Centers (Search)**: `{out_folder}/docking/search/summaries/docking_centers.csv`
- **Production Results**: `{out_folder}/docking/production/summaries/production_docking_results.csv`
- **Individual Poses**: `{out_folder}/docking/production/poses/ligand_{idx}_pocket_{pocket_id}_docked.pdbqt`

## Important Notes

1. **Automatic Creation**: Directory structure is created automatically - no manual directory creation needed
2. **Consistent Paths**: Always use full paths returned by functions when passing files between agents
3. **Same Output Root**: All pipeline steps use the same output_root directory
4. **Log Files**: Log files are in `agent_data/logs/`, NOT in the output folder
5. **Trace Files**: Trace files are in `agent_data/traces/`, NOT in the output folder

## File Naming Patterns

- **Cleaned Receptors (no hydrogens)**: `{protein_name}_{chain}_clean.pdb`
- **Protonated Receptors (with hydrogens)**: `{protein_name}_{chain}_protonated.pdb` (use for docking)
- **Docking Centers**: `docking_centers.csv` (in search/summaries/)
- **Production Results**: `production_docking_results.csv` (in production/summaries/)
