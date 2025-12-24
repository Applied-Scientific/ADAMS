# Parameter Defaults and Guidelines

## Overview
This document provides default parameter values and guidelines for pipeline operations. When users don't specify values, use these defaults. **CRITICAL**: Never use 0 as a placeholder - omit parameters to use defaults.

## Preprocessing Parameters

### run_clean_pdb
- **outpath**: `"./output"` (default)
- **chain_to_keep**: `"A"` (default, most common)
- **ligand**: `False` (default, only enable when user explicitly requests ligand extraction)

### run_ligand_preprocessing
- **molwt_upper_bound**: `800` Da (default, typical range: 500-1000)
- **molwt_lower_bound**: `0` Da (default, use to filter out very small molecules if needed)
- **check_rdmol**: `False` (default, can be expensive for large datasets)
- **sampling**: `False` (default, only enable when user requests sampling)
- **binsize**: `100` Da (default, for stratified sampling)
- **sampling_frac**: `0.01` (default, ~1% of data)
- **output_prefix**: `"cleaned_data"` (default, do NOT infer from input filename)
- **quick_start**: `False` (default, set to True for fast processing that skips detailed metal context analysis)

## Docking Parameters

### run_vina_dock (search_dock mode)
- **search_gridsize**: `25.0` Å (default for CPU, auto-calculated for GPU)
- **search_margin**: `5.0` Å (default buffer)
- **num_poses**: `5` (default, standard for initial screening)
- **num_cores**: `None` (default, auto-detects CPU_count - 1)

### run_find_pocket
- **top_n_clusters**: `3` (default, good balance between coverage and cost)
- **affinity_cutoff**: `-4.0` kcal/mol (default)

### run_vina_dock (production mode)
- **num_pockets**: Match `top_n_clusters` from search step
- **num_poses**: `5` (default)
- **search_gridsize**: `25.0` Å (default)
- **num_cores**: `None` (default, auto-detects CPU_count - 1)

### run_vina_dock_gpu (GPU)
- **num_pockets**: Match `top_n_clusters` from search step
- **num_gpus**: `1` (default, or use user's specified value)
- **gpu_ids**: `None` (default, or use user's specified list)
- **search_gridsize**: `None` (default, auto-calculates from MolWt)

## MD Analysis Parameters

### ProteinTopologyConfig
- **forcefield**: `"amber03"` (default, always set explicitly unless user requests otherwise)
- **water_model**: Function default
- **ignore_hydrogens**: Function default
- **gromacs_path**: Auto-detected from `$CONDA_PREFIX` if not provided
- **ambertools_path**: Auto-detected from `$CONDA_PREFIX` if not provided

### LigPrepareConfig
- **tops**: `3` (default, number of ligands to carry into MD)
- **num_cores**: Function default (or use user's core budget)
- **charge_type**: Function default
- **water_margin**: Function default
- **ion_conc**: Function default
- **gromacs_path**: Auto-detected from `$CONDA_PREFIX` if not provided
- **ambertools_path**: Auto-detected from `$CONDA_PREFIX` if not provided

### GroConfig
- **gpu**: `False` (default, set to True if user requests GPU)
- **mpi_ranks**: `8` (default, or map from num_cores)
- **omp_threads**: `4` (default, or map from num_cores)
- **max_jobs**: Function default
- **gromacs_path**: Auto-detected from `$CONDA_PREFIX` if not provided
- **ambertools_path**: Auto-detected from `$CONDA_PREFIX` if not provided

### StabilityAnalysisConfig
- **prefix**: Function default
- **Range**: `"all"` (default, or `"last"` to use last_frames)
- **last_frames**: Function default
- **vina_report**: Point to production docking results CSV
- **gromacs_path**: Auto-detected from `$CONDA_PREFIX` if not provided
- **ambertools_path**: Auto-detected from `$CONDA_PREFIX` if not provided

### Common MD Parameter
- **gromacs_binary_type**: `"standard"` (default)
  - `"standard"`: Use standard GROMACS binary (gmx)
  - `"mpi"`: Use MPI-enabled binary (gmx_mpi) for multi-rank CPU MD
  - `"cuda"`: Use CUDA binary (gmx from cuda/bin) for GPU-accelerated MD

## Resource Usage Defaults

### CPU Cores
- **Default Behavior**: Use all available cores (auto-detect CPU_count - 1)
- **When to Set Explicitly**: Only when user specifies exact number (e.g., "use 4 cores")
- **Implementation**: Leave `num_cores=None` for auto-detection

### GPU Usage
- **Default**: CPU only (no GPU)
- **When to Use GPU**: Only when user explicitly requests:
  - Keywords: 'gpu', 'gpu run', 'accelerated', 'fast', 'high-throughput', 'large library'
  - Explicit statements: "use GPU", "use 2 GPUs", "GPU acceleration"
- **Search Docking**: Can use GPU when requested (GPU is preferred for large-scale search docking)
- **Production Docking**: Can use GPU when requested

## Parameter Handling Rules

### CRITICAL: Never Use 0 as Placeholder
- **BAD**: `search_gridsize=0`, `num_cores=0`, `num_gpus=0`
- **GOOD**: Omit parameter entirely to use default
- **Reason**: 0 is treated as invalid or triggers errors

### When to Use Defaults
- User doesn't specify a value
- User says "use defaults" or "standard settings"
- Parameter is optional and not critical to user's request

### When to Use Explicit Values
- User explicitly specifies a value (e.g., "use 3 binding sites", "25.0 Å grid")
- User's request implies a specific value (e.g., "50% sampling" → sampling_frac=0.5)
- Biophysical considerations require specific values

### Parameter Extraction from Natural Language
- Look for explicit numbers: "3 binding sites" → top_n_clusters=3
- Look for units: "25.0 Å" → search_gridsize=25.0
- Look for percentages: "50% sampling" → sampling_frac=0.5
- Look for keywords: "all cores" → num_cores=None (auto-detect)

## Common Parameter Mappings

### Core Budget Mapping (MD)
If user says "I have 6 cores":
- **LigPrepareConfig.num_cores**: 6
- **GroConfig.mpi_ranks**: 6 (unless user specifies otherwise)
- **GroConfig.omp_threads**: 1 (unless user specifies otherwise)

### GPU Request Mapping
If user says "use GPU" or "use 2 GPUs":
- **Docking**: Use `run_prod_docking_gpu` with `num_gpus=2`
- **MD**: Set `gromacs_binary_type="cuda"` and `gpu=True` in GroConfig

### Sampling Request Mapping
If user says "50% sampling":
- **sampling**: True
- **sampling_frac**: 0.5

## Parameter Validation

### Required Parameters
- **Preprocessing**: `input_pdb`, `input_data`
- **Docking**: `receptor`, `input_file`/`input_data`
- **MD ProteinTopology**: `protein_file`
- **MD LigPrepare**: `docking_csv`, `ligand_input`

### Optional Parameters
- All other parameters have defaults and can be omitted
- Only provide explicit values when user specifies or when biophysically necessary
