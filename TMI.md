# ADAMS: Technical Documentation

This document provides detailed implementation details, installation instructions, and file organization information.

**Note:** For standard terminology and vocabulary used throughout the codebase, see [docs/TERMINOLOGY_QUICK_REF.md](docs/TERMINOLOGY_QUICK_REF.md).

## Table of Contents

1. [Installation Steps](#installation-steps)
   - [Step 1: Base Environment](#step-1-install-packages-from-environmentyml)
   - [Step 2: AmberTools](#step-2-install-ambertools)
   - [Step 3: GROMACS](#step-3-install-gromacs)
   - [Step 4: CUDA (GPU only)](#step-4-cuda-setup-for-gpu-acceleration)
2. [Detailed Usage](#detailed-usage)
3. [Output File Organization](#output-file-organization)
4. [Pipeline Architecture](#pipeline-architecture)
   - [Agent Hierarchy](#agent-hierarchy)
   - [Stage 1: Preprocessing](#stage-1-preprocessing)
   - [Stage 2: Docking](#stage-2-docking)
   - [Stage 3: MD Analysis](#stage-3-md-analysis)
5. [Troubleshooting](#troubleshooting)
6. [Additional Documentation](#additional-documentation)

---

## Installation Steps

**IMPORTANT: Follow these steps in order.**

### Prerequisites: OpenAI API Key

Before running the pipeline, you must set up your OpenAI API key. You can do this in one of two ways:

1. **Environment variable:**
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

2. **`~/.adams`**
   Or simply provide during a session. If you choose to store it, the key will be saved to
   ```
   ~/.adams
   ```
   **Security note:** This file is stored in **plaintext**; ADAMS will attempt to set permissions to `600` (user read/write only).

The agents require this API key to function. You can obtain an API key from [OpenAI's website](https://platform.openai.com/api-keys).

### Install adams

```bash
bash scripts/install.sh
```

The script automatically detects the OS and installs all required packages.


#### Installing vina_gpu

The pre-compiled Vina-GPU binaries are included with our release. They were built using the files obtained from [Vina-GPU 2.1](https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1).


---

## Detailed Usage

### Input Preparation

1. Change directory to where the input files are:
   - Protein structure (PDB format)
   - Ligand library (SDF, MOL2, or similar)
   - Configuration files (if needed)

2. Start a session:
   ```bash
   adams
   ```
The agent will prompt you for details or use configuration defaults.

### Pipeline Stages

The pipeline consists of three main stages: preprocessing, docking, and MD analysis. Each stage has distinct operational modes or steps.

**Note:** For detailed information on how agents coordinate within each stage, see the [Pipeline Architecture](#pipeline-architecture) section below.

#### Stage 1: Preprocessing

The preprocessing stage consists of two independent operations that can be run in any order:

1. **Receptor Preparation** (`run_clean_pdb`): Cleans protein PDB structures by selecting chains, removing water molecules and unwanted heterogens, adding missing atoms and hydrogens, and optionally extracting bound ligands.

2. **Ligand Preprocessing** (`run_ligand_preprocessing`): Processes ligand CSV files by filtering compounds by molecular weight, validating SMILES structures (optional), and optionally performing stratified sampling to create representative subsets of large libraries.

**Typical Preprocessing Workflow:**
- Both operations are independent and can be run in parallel or in any order
- Receptor preparation outputs cleaned PDB files ready for docking
- Ligand preprocessing outputs filtered/sampled CSV files ready for docking

#### Stage 2: Docking

The docking stage supports two operational modes:

1. **Search Docking** (`mode="search"`): Discovers binding pockets across the entire protein surface by systematically docking ligands at grid points. This mode is CPU-based and identifies high-affinity binding sites for subsequent production docking.

2. **Production Docking** (`mode="production"`, default): Performs targeted docking at known binding sites. This mode uses flexible box sizing based on ligand molecular weight and higher exhaustiveness (32) for accurate pose generation. Can be run on CPU or GPU.

**Additional Parameter:**
- **`minimized_dock=True`**: When enabled with production mode, uses a fixed 5Å docking box and lower exhaustiveness (8) for precise refinement at well-characterized binding sites. Only recommended for small ligands (<300 Da) at energy-minimized centers.

**Typical Docking Workflow:**
1. Run search docking to discover binding sites
2. Cluster results to identify top pockets (via `run_find_pocket`)
3. Run production docking at the identified sites

#### Stage 3: MD Analysis

The MD analysis stage consists of four sequential steps:

1. **Protein Topology** (`run_protein_topology`): Generates protein topology files (protein.gro, topol.top) from cleaned PDB structures
2. **Ligand Preparation** (`run_lig_prepare`): Prepares ligand poses from docking results and combines with protein topology
3. **MD Simulations** (`run_gro`): Runs energy minimization, NVT/NPT equilibration, and production MD trajectories
4. **Stability Analysis** (`run_stability_analysis`): Analyzes trajectories for RMSD, RMSF, and binding stability metrics

For batch processing examples, see `run_inference.scr` in the project root.

---

## Output File Organization

The pipeline automatically organizes all intermediate and output files into a structured directory hierarchy. This organization is **mandatory** and happens by default during runtime.

For detailed execution order, file path specifications, and entry point documentation, see [docs/WORKFLOW_EXECUTION.md](docs/WORKFLOW_EXECUTION.md).

### Directory Structure

```
output_folder/
├── preprocessing/
│   ├── receptors/          # Cleaned protein PDB files
│   │   └── {protein_name}_{chain}_clean_h.pdb
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
│
└── md_analysis/            # Molecular Dynamics analysis outputs
    ├── protein/            # Protein topology files
    │   ├── protein.gro
    │   └── topol.top
    ├── poses/              # Prepared ligand poses for MD
    │   └── {ligand_name}_pocket_{grid_id}_top{rank}/
    │       ├── {ligand_name}.gro
    │       ├── {ligand_name}.top
    │       └── ... (GROMACS simulation files)
    └── reports/            # MD analysis reports
        ├── md_analysis_summary_{range}.csv
        └── brief_report_{range}.csv
```

### File Descriptions

#### Understanding Docking Output Files

The docking pipeline creates several CSV files with different purposes. Understanding these file names is important for correctly linking docking results to MD analysis.

##### Search Docking Files (`docking/search/summaries/`)

- **`best_search_docking_centers.csv`**: Contains the top 100 poses from search docking across all grids. Used for visualization and analysis of search results.

- **`best_docking_centers.csv`**: Contains the best pose per grid from search docking. This file includes **all grids** discovered during search (e.g., 8 grids). Used as input for clustering analysis.

- **`docking_centers.csv`**: Contains the top N clusters (typically 3) selected after clustering. This is the **input file for production docking** - it specifies which pockets to use for targeted docking.

- **`cluster_summary.csv`**: Statistics for all identified pocket clusters (mean affinity, cluster size, centroids).

- **`dock_sites_clustered.csv`**: All docking poses assigned to clusters.

##### Production Docking Files (`docking/production/summaries/`)

- **`production_docking_results.csv`**: Contains the best pose per pocket from **production docking only**. This file includes **only the top N pockets** (e.g., 3) that were selected for production docking. **This is the preferred file for MD analysis** as it matches the actual PDBQT files created during production docking.

**Note**: `best_docking_centers.csv` is a **search mode** output file (located in `docking/search/summaries/`), not a production mode file. It contains the best pose per grid from search docking and is used as input for clustering analysis.

##### Important Notes for File Matching

- **For MD analysis**: The **workflow agent** extracts the path to `production_docking_results.csv` from the docking agent's output and passes it to the MD agent. The MD agent uses whatever CSV file is provided via the `docking_csv` parameter - it does not perform automatic file prioritization or discovery.

- **File matching**: The number of PDBQT files in `docking/production/poses/` should match the number of pockets in `production_docking_results.csv`. If you see a mismatch (e.g., 8 poses but only 3 PDBQT files), check that MD is using the correct CSV file.

- **Naming convention**: 
  - Search docking uses `grid_{grid_id}` in filenames (e.g., `ligand_0_grid_5_docked.pdbqt`)
  - Production docking uses `pocket_{pocket_id}` in filenames (e.g., `ligand_0_pocket_2_docked.pdbqt`)

#### Preprocessing Outputs

Located in `preprocessing/`:
- **receptors/**: Cleaned and hydrogenated protein structures ready for docking
- **ligands/**: Filtered and processed ligand libraries (by molecular weight, metal content, etc.)

---

## Pipeline Architecture

This section describes the internal architecture and agent coordination for each pipeline stage. For usage instructions and operational modes, see the [Detailed Usage](#detailed-usage) section above.

For comprehensive agent documentation including detailed agent descriptions, available tools, and agent-to-agent call patterns, see [docs/AGENTS_DOCUMENTATION.md](docs/AGENTS_DOCUMENTATION.md).

### Agent Hierarchy

The system uses a hierarchical agent architecture where specialized agents coordinate computational workflows:

**Level 1: Coordinator Agent**
- Orchestrates the overall pipeline and interprets user prompts
- Routes tasks to appropriate stage agents

**Level 2: Stage Agents**
- **Preprocessing Agent**: Manages data preparation and format conversion
- **Docking Agent**: Handles molecular docking workflows and pose selection
- **MD Analysis Agent**: Coordinates molecular dynamics simulations and stability analysis

**Level 3: Helper Agents**
- File validation and error handling
- Parameter optimization and tuning
- Result interpretation and analysis

**Level 4: Worker Modules**
- AutoDock Vina (docking calculations)
- GROMACS (molecular dynamics simulations)
- File converters and format validators


Agents make autonomous decisions while worker modules handle the heavy computational lifting.

### Stage 1: Preprocessing

**Purpose**: Prepare protein and ligand inputs for docking

**What happens:**
1. Protein cleaning: Remove waters, add hydrogens, assign charges
2. Ligand filtering: Separate by molecular weight, identify metal-containing compounds
3. Format conversion: Convert to formats required by AutoDock Vina (PDBQT)

**Agent coordination:**
- Stage agent determines which chain(s) to use from multi-chain structures
- Helper agent validates protein preparation and fixes common issues (missing atoms, format errors)
- Stage agent sets molecular weight cutoffs for ligand filtering
- Stage agent decides sampling fraction for large libraries
- Coordinator agent approves preprocessing decisions before proceeding

**Worker modules:** PDB processors, ligand file converters, format validators

### Stage 2: Docking

**Search Phase:**
- Creates a grid covering the entire protein surface
- Performs exploratory docking across all grid points
- Identifies high-affinity binding sites
- Clusters results to find distinct pockets

**Production Phase:**
- Takes top N pockets (typically 3) from search
- Performs focused docking with increased exhaustiveness
- Generates ranked poses for each pocket
- Selects top poses for MD simulation

**Agent coordination:**

*Search Phase:*
- Stage agent determines grid spacing based on protein size
- Helper agent validates protein preparation and fixes common issues
- Stage agent interprets initial docking results and decides clustering parameters
- Coordinator agent approves number of pockets to pursue in production

*Production Phase:*
- Stage agent adjusts docking exhaustiveness based on pocket characteristics
- Helper agent evaluates pose quality and diversity
- Stage agent applies selection criteria to identify MD candidates
- Coordinator agent decides which poses warrant expensive MD simulations

**Worker modules:** AutoDock Vina, clustering algorithms, PDBQT converters, scoring functions

### Stage 3: MD Analysis

**Preparation:**
- Converts PDBQT poses to GROMACS-compatible formats
- Generates topology files for protein-ligand complexes
- Sets up simulation box and solvation

**Simulation:**
1. Energy minimization (steepest descent)
2. NVT equilibration (temperature stabilization)
3. NPT equilibration (pressure stabilization)
4. Production MD run (typically 10-100 ns)

**Analysis:**
- RMSD (stability of binding)
- RMSF (residue flexibility)
- Hydrogen bond persistence
- Binding energy estimates

**Agent coordination:**
- Stage agent configures simulation parameters (length, force field, etc.)
- Helper agent monitors simulation stability and convergence
- Stage agent interprets RMSD, RMSF, and binding energy metrics
- Coordinator agent synthesizes results and generates final report
- Helper agent handles convergence issues and decides if simulation needs extension

**Worker modules:** GROMACS, AmberTools, analysis scripts, trajectory processors

---

## Troubleshooting

### Common Issues

#### GROMACS not found
```bash
# Ensure GROMACS is in your PATH
export PATH=$CONDA_PREFIX/gromacs-2024.4/bin:$PATH
# Or for CUDA build:
export PATH=$CONDA_PREFIX/gromacs-2024.4/cuda/bin:$PATH
```

#### CUDA errors during MD
- Verify GPU is accessible: `nvidia-smi`
- Check CUDA version matches GROMACS build
- Ensure `OMPI_MCA_opal_cuda_support=true` is set

#### Missing dependencies
```bash
# Reinstall environment
bash scripts/install.sh
```
Choose "Start Fresh".

#### File not found errors
- Verify input files are in the current working directory
- Check file permissions (must be readable)
- Ensure filenames don't contain special characters

### Getting Help

1. Review the troubleshooting section above for common issues
2. Check the [Pipeline Architecture](#pipeline-architecture) section for understanding agent coordination and known limitations
3. Open an issue with:
   - Error message
   - Contents of relevant log files (`agent_data/logs/adams_pipeline_{run_id}.log`)
   - System information (OS, GPU, CUDA version)

---

## Additional Documentation

For more detailed information on specific aspects of the system:

- **[docs/AGENTS_DOCUMENTATION.md](docs/AGENTS_DOCUMENTATION.md)** - Complete agent architecture, all available tools, agent hierarchy, and agent-to-agent call patterns
- **[docs/WORKFLOW_EXECUTION.md](docs/WORKFLOW_EXECUTION.md)** - Detailed workflow execution order, module execution sequences, file paths, and entry points
- **[docs/TERMINOLOGY_QUICK_REF.md](docs/TERMINOLOGY_QUICK_REF.md)** - Standard terminology and vocabulary for pipeline stages, agents, workflows, and data organization

---
