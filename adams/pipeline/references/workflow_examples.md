# Workflow Examples

## Overview
This document contains example prompts and workflows for common pipeline scenarios. Use these as templates when constructing prompts for the workflow_agent.

## General Principles for Stage Transitions

These principles apply to the preprocessing → docking transition and should be followed consistently:

### Principle 1: Explicit Output Folder Path Passing
- **CRITICAL**: Always explicitly include the output folder path in your natural language instruction to each agent
- **Format**: "Use {output_folder_path} as the {parameter_name} for all operations"
- **Examples**:
  - Preprocessing: "Use {output_folder} as the outpath for all preprocessing operations"
  - Docking: "Use {output_folder} as the out_folder for all docking operations"
- **Why**: Agents have default paths (like "./output") that are WRONG - you must override them
- **Verification**: After each stage completes, verify paths use your run directory, not defaults

### Principle 2: Path Extraction Before Next Stage
- **CRITICAL**: Always extract paths from the previous agent's output BEFORE calling the next agent
- **Process**:
  1. Wait for agent to complete
  2. Parse the agent's response to extract output file paths
  3. Store these paths for use in the next stage
  4. Use extracted paths (not reconstructed paths) when calling next agent
- **Key Paths to Extract**:
  - From preprocessing: cleaned receptor path; **mapping CSV path** (docking_ready_ligands.csv from run_smiles_to_pdbqt or run_standardize_ligand_data for 3D) — use as **input_data** for docking
  - From docking: production docking results CSV path, docking centers CSV path
- **Why**: Agents return exact paths - reconstructing paths can lead to errors

### Principle 2b: Docking Input Readiness
- **CRITICAL**: Docking requires ligand input that is **docking-ready**: a CSV that includes a **PDBQT_File** column (paths to PDBQT files). Docking cannot use a table that only has SMILES/ID—ligands must be prepared as 3D conformers and PDBQT files first.
- **When preprocessing runs first**: Ensure the preprocessing stage produces a PDBQT-ready table (e.g. by including conformer generation and PDBQT writing). Pass the path to that table (or the docking agent's equivalent) when calling the docking agent. Do not pass a cleaned/filtered ligand CSV that lacks a PDBQT_File column and assume docking can proceed.
- **When a step is reported incomplete**: If logs or tool output report a pipeline step as "incomplete", treat it as a signal that a required prior step (e.g. ligand/conformer preparation) may have been skipped or failed. Ensure the full sequence—including any conformer/PDBQT step—has been run before retrying.

### Principle 3: Complete Parameter Provision
- **CRITICAL**: Always provide ALL required parameters in a SINGLE agent call
- **DO NOT**: Call agent, wait for it to ask for inputs, then provide them
- **DO**: Extract all needed paths from previous stages, then call with complete parameters
- **Required Parameters by Stage**:
  - Preprocessing: input_pdb, input_data, outpath (explicit)
  - Docking: receptor, input_data, out_folder (explicit), plus any docking-specific params
- **Why**: Agents should execute immediately, not wait for additional input

### Principle 4: Consistent Output Root Directory
- **CRITICAL**: All stages must use the SAME output_root directory
- **Process**:
  1. Create run directory once (or use provided output folder)
  2. Use this EXACT path for: outpath (preprocessing), out_folder (docking)
  3. Never mix different output directories across stages
- **Verification**: All file paths should contain the same root directory path
- **Why**: Files from one stage are inputs to the next - they must be in the same location

### Principle 5: Automatic Stage Progression
- **CRITICAL**: Automatically proceed to the next stage after current stage completes
- **Process**:
  1. Call current stage agent
  2. Wait for completion
  3. Extract output paths
  4. Immediately call next stage agent with extracted paths
  5. Do NOT ask user for confirmation between stages
- **Exception**: Only stop if user explicitly requested only specific stages
- **Why**: Users requesting "full pipeline" expect automatic execution

### Principle 6: Natural Language Instruction Format
- **CRITICAL**: All agent calls use natural language instructions, not structured parameters
- **Format**: Complete sentences describing the task with embedded path information
- **Good**: "Clean the receptor at /path/to/receptor.pdb and process ligands from /path/to/ligands.csv. Use /path/to/outputs/run_xxx as the outpath for all preprocessing operations."
- **Bad**: "run_clean_pdb(input_pdb='/path/to/receptor.pdb', outpath='/path/to/outputs/run_xxx')"
- **Why**: Agents parse natural language and extract parameters themselves

### Docking engines available
- The pipeline's docking step supports **three** engines (backend parameter): **AutoDock Vina (CPU)** (`vina`), **Vina-GPU (CUDA)** (`vina_gpu`), and **UniDock (GPU)** (`unidock`). When the user or controller requests "all docking engines" or engine/speed comparison, run the same docking task with each engine in separate runs (distinct out_folder per engine, e.g. compare_vina, compare_vinagpu, compare_unidock) so results and timings can be compared.

## Example 1: Complete End-to-End Workflow (CPU)

**User Request**: "Run the complete docking workflow. Clean the receptor at {receptor_path} keeping chain {chain}, preprocess compounds in {ligand_csv_path} with a {molwt_upper_bound} Da cutoff without sampling. Discover top {N} binding sites, then dock at those sites."

**Applying General Principles**:
- **Principle 1**: Explicitly pass output_folder to preprocessing and docking
- **Principle 2**: Extract cleaned receptor and ligand CSV from preprocessing before docking
- **Principle 4**: Use same output_folder across both stages
- **Principle 5**: Automatic progression through preprocessing → docking without stopping

**Workflow Agent Prompt**:
```
Please run the ENTIRE end-to-end workflow. First, clean the receptor at {receptor_path} keeping chain {chain} and also preprocess the compounds in {ligand_csv_path} with a {molwt_upper_bound} Da cutoff without sampling. Use {output_folder} as the outpath for all preprocessing operations. Next, discover binding sites for the ligands and dock them at the top {N} sites, using search gridsize of {gridsize} and search margin of {margin}, using {output_folder} as the out_folder. I want to use all cores (no GPU).
```

---

## Example 2: Complete End-to-End Workflow (GPU)

**User Request**: "Run the complete docking workflow with GPU acceleration."

**Applying General Principles**:
- **Principle 1**: Explicitly pass output_folder to both stages
- **Principle 2**: Extract paths between stages
- **Principle 4**: Use same output_folder throughout
- **Principle 5**: Automatic progression through both stages

**Workflow Agent Prompt**:
```
Please run the ENTIRE end-to-end workflow. First, clean the receptor at {receptor_path} keeping chain {chain} and also preprocess the compounds in {ligand_csv_path} with a {molwt_upper_bound} Da cutoff without sampling. Use {output_folder} as the outpath for all preprocessing operations. Next, discover binding sites for the ligands and dock them at the top {N} sites, using {output_folder} as the out_folder. Enable GPU for docking. I want to use all cores and GPUs when applicable. Use GPU for production docking.
```

---

## Example 3: Docking Only (Data Already Prepared)

**User Request**: "Run only the docking pipeline (no preprocessing) with the ligands from {ligand_csv_path} and the protein {receptor_path}. First discover the binding sites, then dock all ligands at the top {N} sites."

**Applying General Principles**:
- **Principle 1**: Explicitly pass output_folder to docking agent
- **Principle 4**: Use consistent output_folder (even for single stage)

**Workflow Agent Prompt**:
```
Please run only the docking pipeline (no preprocessing) with the ligands from {ligand_csv_path} and the protein {receptor_path}. First discover the binding sites, then dock all ligands at the top {N} sites. Use {output_folder} as the out_folder for all docking operations. I want to use all cores and GPUs when applicable. Use GPU for production docking.
```

---

## Example 4: Production Docking with Known Binding Site Coordinates

**User Request**: "Run production docking with ligands from {ligand_csv_path} with the protein {receptor_path}. Please dock with the binding site coordinates ({x} {y} {z})."

**Applying General Principles**:
- **Principle 1**: Explicitly pass output_folder to docking agent

**Workflow Agent Prompt**:
```
Run production docking with ligands from {ligand_csv_path} with the protein {receptor_path}. Please dock with the binding site coordinates ({x} {y} {z}). Use {output_folder} as the out_folder for all docking operations. I want to use all cores and GPUs when applicable. Use GPU for production docking.
```

---

## Example 6: Complete Workflow with Sampling

**User Request**: "Run the complete docking workflow with 50% sampling."

**Applying General Principles**:
- **Principle 1**: Explicitly pass output_folder to both stages
- **Principle 2**: Extract sampled CSV path from preprocessing (not small_mw)
- **Principle 4**: Use same output_folder throughout
- **Principle 5**: Automatic progression through stages

**Workflow Agent Prompt**:
```
Please run the complete docking workflow. Clean the receptor at {receptor_path} keeping chain {chain} and preprocess the compounds in {ligand_csv_path} with a {molwt_upper_bound} Da cutoff and {sampling_frac} sampling. Use {output_folder} as the outpath for all preprocessing operations. Discover top {N} binding sites, then dock at those sites using {output_folder} as the out_folder. I want to use all cores and GPUs when applicable. Use GPU for production docking.
```

---

## Example 7: Mid-Pipeline Start - Production Docking Only (Skip Search, with Coordinates)

**User Request**: "The receptor and compounds are already prepared, and I have binding site coordinates. Run production docking only."

**Workflow Agent Prompt**:
```
The receptor and compounds are already prepared, and I have binding site coordinates. Run production docking only with ligands from {ligand_csv_path} and receptor {receptor_path}. Dock at coordinates ({x} {y} {z}). Use {output_folder} as output. I want to use all cores and GPUs when applicable. Use GPU for production docking.
```

---

## Example 11: Mid-Pipeline Start - Production Docking Only (Skip Search, with Docking Centers File)

**User Request**: "The receptor and compounds are already prepared, and I have a docking centers file from a previous search."

**Workflow Agent Prompt**:
```
The receptor and compounds are already prepared, and I have a docking centers file from a previous search. Run production docking only with ligands from {ligand_csv_path} and receptor {receptor_path}. Use docking_centers_file={docking_centers_csv_path}. Use {output_folder} as output. I want to use all cores and GPUs when applicable. Use GPU for production docking.
```

---

## Example 13: Custom Data Manipulation (Fix CSV Format)

**User Request**: "My ligand file 'ligands.txt' is tab-separated but needs to be a comma-separated CSV with specific headers. Fix it."

**Workflow Agent Prompt**:
```
Run a custom Python script to fix the ligand file format. Read 'ligands.txt' (tab-separated), rename columns to ID, SMILES, MolWt, and save it as 'ligands.csv'. Use the preprocessing_agent's run_python_code tool.
```

---

## Common Prompt Construction Guidelines

When constructing prompts, apply the general principles above and follow these guidelines:

1. **Replace Placeholders**: Replace {receptor_path}, {ligand_csv_path}, {output_folder}, etc. with actual file paths
2. **Apply Principle 1**: Always explicitly include output folder path in natural language instruction
3. **Apply Principle 2**: Extract paths from previous agent outputs before constructing next prompt
4. **Apply Principle 3**: Include all required parameters in your prompt (don't wait for agent to ask)
5. **Apply Principle 4**: Use the same output_root directory across all stages
6. **Apply Principle 5**: Automatically progress through stages without user confirmation
7. **Apply Principle 6**: Use natural language format, not structured parameters
8. **Specify Parameters**: Include exact parameter values (grid sizes, margins, number of sites, etc.)
9. **Resource Usage**: Include GPU/CPU preferences when relevant
10. **Be Explicit**: Clearly state which parts of the pipeline to run (preprocessing, docking, MD, or all)
11. **Use Full Paths**: Use full absolute paths for all files

## Mid-Pipeline Start Parameters

When starting mid-pipeline, include these parameters as needed:

- **For Production Docking entry** (skip search): Provide `docking_centers=({x} {y} {z})` OR `docking_centers_file=/path/to/centers.csv`
- **For ProteinTopology entry** (skip preprocessing/docking): Provide cleaned `receptor`, `docking_csv` (docking results), and `ligand_input`
- **For LigPrepare entry** (skip ProteinTopology): Provide `protein_gro=/path/to/protein.gro, protein_top=/path/to/topol.top`. Also pass `water_model=` (e.g. `tip3p`, `spc`) to match the topology so solvation is consistent.
- **For Gro entry** (skip LigPrepare): Provide `pose_dirs=pose1,pose2,pose3` (comma-separated list of pose directory names)
- **For StabilityAnalysis entry** (skip Gro): Provide `pose_dirs_analysis=pose1,pose2,pose3` (comma-separated list)
- **For external docking results**: Ensure user's docking CSV has `pdbqt_path` column if PDBQT files are in non-standard locations
