# Pipeline Directory Structure

## Overview
All pipeline outputs are automatically organized into a structured directory hierarchy under the output root directory.

## Complete Directory Tree

```
{output_root}/
├── preprocessing/
│   ├── receptors/          # Cleaned protein PDB files
│   │   └── {protein_name}_{chain}_clean_h.pdb
│   ├── ligands/            # Processed ligand files
│   │   ├── metal_compounds.csv
│   │   ├── metal_organic_compounds.csv
│   │   ├── {prefix}_largeMW.csv
│   │   ├── {prefix}_smallMW.csv
│   │   ├── {prefix}_frac{sampling_frac}.csv
│   │   ├── conformers_pdbqt/  # Generated PDBQT files from SMILES
│   │   │   └── {clean_id}.pdbqt
│   │   ├── 3d_structures_pdbqt/  # Converted 3D structures to PDBQT
│   │   │   └── {clean_id}.pdbqt
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
│
└── md_analysis/            # Molecular Dynamics analysis outputs
    ├── protein/            # Protein topology files
    │   ├── protein.gro
    │   └── topol.top
    ├── poses/              # Prepared ligand poses for MD
    │   └── {ligand_name}_pocket_{grid_id}_top{rank}/
    │       ├── {ligand_name}.gro
    │       ├── {ligand_name}.top
    │       └── ... (GROMACS simulation files: md.tpr, md.xtc, etc.)
    └── reports/            # MD analysis reports
        ├── md_analysis_summary_{range}.csv
        └── brief_report_{range}.csv
```

## Key File Locations

### Preprocessing Outputs
- **Cleaned Receptor**: `{outpath}/preprocessing/receptors/{protein_name}_{chain}_clean_h.pdb`
- **Processed Ligands**: `{outpath}/preprocessing/ligands/{prefix}_smallMW.csv` (or sampled version)
- **Sampled Dataset**: `{outpath}/preprocessing/ligands/{prefix}_frac{sampling_frac}.csv` (if sampling enabled)
- **PDBQT Files (from SMILES)**: `{outpath}/preprocessing/ligands/conformers_pdbqt/{clean_id}.pdbqt`
- **PDBQT Files (from 3D)**: `{outpath}/preprocessing/ligands/3d_structures_pdbqt/{clean_id}.pdbqt`

### Docking Outputs
- **Docking Centers (Search)**: `{out_folder}/docking/search/summaries/docking_centers.csv`
- **Production Results**: `{out_folder}/docking/production/summaries/production_docking_results.csv`
- **Individual Poses**: `{out_folder}/docking/production/poses/ligand_{idx}_pocket_{pocket_id}_docked.pdbqt`

### MD Analysis Outputs
- **Protein Topology**: `{md_workdir}/md_analysis/protein/protein.gro`, `topol.top`
- **Prepared Poses**: `{md_workdir}/md_analysis/poses/{ligand_name}_pocket_{grid_id}_top{rank}/`
- **Analysis Reports**: `{md_workdir}/md_analysis/reports/md_analysis_summary_{range}.csv`

## Important Notes

1. **Automatic Creation**: Directory structure is created automatically - no manual directory creation needed
2. **Consistent Paths**: Always use full paths returned by functions when passing files between agents
3. **Same Output Root**: All pipeline steps use the same output_root directory
4. **Log Files**: Log files are in `agent_data/logs/`, NOT in the output folder
5. **Trace Files**: Trace files are in `agent_data/traces/`, NOT in the output folder

## File Naming Patterns

- **Cleaned Receptors**: `{protein_name}_{chain}_clean_h.pdb`
- **Docking Centers**: `docking_centers.csv` (in search/summaries/)
- **Production Results**: `production_docking_results.csv` (in production/summaries/)
- **Pose Directories**: `{ligand_name}_pocket_{grid_id}_top{rank}/`
- **MD Trajectories**: `md.tpr`, `md.xtc`, `md.gro` (in pose directories)
