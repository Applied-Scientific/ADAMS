# Parameter Defaults and Guidelines

## Overview
This document provides default parameter values and guidelines for pipeline operations. When users don't specify values, use these defaults. **CRITICAL**: Never use 0 as a placeholder - omit parameters to use defaults.

## Preprocessing Parameters

### run_clean_pdb
- **outpath**: `"./output"` (default)
- **chain_to_keep**: `"all"` (default, keep complete receptor assembly). Also supports `"A"` (single-chain mode) and `"A,B,C"` / `["A","B","C"]` (selected chains).
- **ligand**: `False` (default, only enable when user explicitly requests ligand extraction)
- **keep_water**: `False` (default; set `True` to retain structural waters and carry them into protonated receptor output)
- **keep_heterogens**: `"essential"` (default) = keep ESSENTIAL_HETEROGENS_TO_KEEP (built-in cofactors, metal ions, nucleotides; see preprocessing_agent_prompt or clean_pdb.py). `None` or `[]` = remove all. List or single 3-letter str (e.g. `["HEM", "MG"]` or `"HEM"`) = keep only those.
- **model_missing_residues**: `True` (default). Attempt to model selected missing-residue blocks.
- **max_missing_residues_per_gap**: `12` (default). Larger missing blocks are left as chain gaps.
- **allow_terminal_missing_residues**: `False` (default). Terminal missing stretches are left as gaps unless explicitly enabled.
- **pH**: `7.4` (default) = pH value for protein protonation state. Used in run_protonate_receptor to determine protonation states.

### run_smiles_to_pdbqt (2D ligands → PDBQT + mapping CSV)
- **output_dir**: `"output"` (default)
- **num_confs**: `8` (default). Number of conformers generated per molecule before pruning.
- **max_confs_to_keep**: `2` (default). Retain only lowest-energy conformers per molecule.
- **conformer_energy_window_kcal**: `3.0` (default). Keep conformers within this window from best.
- **random_seed**: `42` (default). Use for reproducible conformers; change only if user needs a different conformer set.

### run_ligand_preprocessing (microstate defaults)
- **enumerate_microstates**: `True` (default)
- **pH_min / pH_max**: `6.4 / 8.4` (default, centered around physiological pH)
- **protonation_precision**: `0.5` (default)
- **max_generated_tautomers**: `64` (default, set to `None` to disable hard generation cap)
- **top_tautomers_per_protomer**: `2` (default)
- **tautomer_energy_window_kcal**: `3.0` (default)
- **max_protomers**: `16` (default)
- **max_stereoisomers**: `16` (default)
- **max_unassigned_stereocenters**: `2` (default)
- **max_total_microstates**: `64` (default)
- **enumerate_all_stereocenters**: `False` (default, preserve assigned stereochemistry)

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
- **Pocket definition is mandatory**: provide one of `docking_centers_file`, `complex`, or `docking_centers`. Production mode no longer defaults to `[0,0,0]`.
- **num_pockets**: Match `top_n_clusters` from search step
- **num_poses**: `5` (default)
- **production_gridsize**: `None` (default = backend default sizing; set explicit Å value to force a fixed production box)
- **lock_grid_center**: `True` (default, keep production docking maps centered on user pocket center after pre-minimize)
- **num_cores**: `None` (default, auto-detects CPU_count - 1)
- **pH**: `7.4` (default) = pH value for receptor protonation state when converting PDB to PDBQT.

### Docking charge model
- **Ligand**: Meeko uses `charge_model="gasteiger"` (default) when generating PDBQT in `run_smiles_to_pdbqt` and `convert_3d_to_pdbqt`.
- **Receptor**: OpenBabel uses `--partialcharge gasteiger` (default) in `convert_receptor_to_pdbqt`.
- Default is `gasteiger` for both for consistency. Override via `run_docking(charge_model=...)` and, for ligand prep, `run_smiles_to_pdbqt(charge_model=...)` or `run_standardize_ligand_data(charge_model=...)` (3D path).
- **pH** 7.4 remains the default for receptor protonation (unchanged).

### run_vina_dock_gpu (GPU)
- **num_pockets**: Match `top_n_clusters` from search step
- **num_gpus**: `None` (default = auto-detect all available GPUs unless user explicitly requests a count)
- **gpu_ids**: `None` (default; set only when user explicitly requests specific GPU IDs)
- **search_gridsize**: `None` (default, auto-calculates from MolWt)

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

### GPU Request Mapping
If user says "use GPU" or "use 2 GPUs":
- **Docking**: Use `run_prod_docking_gpu` with `num_gpus=2`

### Sampling Request Mapping
If user says "50% sampling":
- **sampling**: True
- **sampling_frac**: 0.5

## Parameter Validation

### Required Parameters
- **Preprocessing**: `input_pdb`, `input_data`
- **Docking**: `receptor`, `input_file`/`input_data`

### Optional Parameters
- All other parameters have defaults and can be omitted
- Only provide explicit values when user specifies or when biophysically necessary
