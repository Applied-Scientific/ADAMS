# Workflow Examples

## Overview
This document contains example prompts and workflows for common pipeline scenarios. Use these as templates when constructing prompts for the workflow_agent.

## General Principles for Stage Transitions

These principles apply to ALL stage transitions (preprocessing → docking → MD) and should be followed consistently:

### Principle 1: Explicit Output Folder Path Passing
- **CRITICAL**: Always explicitly include the output folder path in your natural language instruction to each agent
- **Format**: "Use {output_folder_path} as the {parameter_name} for all operations"
- **Examples**:
  - Preprocessing: "Use {output_folder} as the outpath for all preprocessing operations"
  - Docking: "Use {output_folder} as the out_folder for all docking operations"
  - MD: "Use {output_folder} as the md_workdir for all MD operations"
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
  - From preprocessing: cleaned receptor path, processed ligand CSV path
  - From docking: production docking results CSV path, docking centers CSV path
  - From MD: pose directories, report paths (for mid-pipeline resumes)
- **Why**: Agents return exact paths - reconstructing paths can lead to errors

### Principle 3: Complete Parameter Provision
- **CRITICAL**: Always provide ALL required parameters in a SINGLE agent call
- **DO NOT**: Call agent, wait for it to ask for inputs, then provide them
- **DO**: Extract all needed paths from previous stages, then call with complete parameters
- **Required Parameters by Stage**:
  - Preprocessing: input_pdb, input_data, outpath (explicit)
  - Docking: receptor, input_data, out_folder (explicit), plus any docking-specific params
  - MD: protein_file, docking_csv, ligand_input, md_workdir (explicit), plus MD-specific params
- **Why**: Agents should execute immediately, not wait for additional input

### Principle 4: Consistent Output Root Directory
- **CRITICAL**: All stages must use the SAME output_root directory
- **Process**:
  1. Create run directory once (or use provided output folder)
  2. Use this EXACT path for: outpath (preprocessing), out_folder (docking), md_workdir (MD)
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

## Example 1: Complete End-to-End Workflow (CPU)

**User Request**: "Run the complete docking workflow. Clean the receptor at {receptor_path} keeping chain {chain}, preprocess compounds in {ligand_csv_path} with a {molwt_upper_bound} Da cutoff without sampling. Discover top {N} binding sites, then dock at those sites."

**Applying General Principles**:
- **Principle 1**: Explicitly pass output_folder to preprocessing, docking, and MD
- **Principle 2**: Extract cleaned receptor and ligand CSV from preprocessing before docking
- **Principle 3**: Provide all MD parameters (protein_file, docking_csv, ligand_input, md_workdir) in single call
- **Principle 4**: Use same output_folder across all three stages
- **Principle 5**: Automatically proceed preprocessing → docking → MD without stopping

**Workflow Agent Prompt**:
```
Please run the ENTIRE end-to-end workflow. First, clean the receptor at {receptor_path} keeping chain {chain} and also preprocess the compounds in {ligand_csv_path} with a {molwt_upper_bound} Da cutoff without sampling. Use {output_folder} as the outpath for all preprocessing operations. Next, discover binding sites for the ligands and dock them at the top {N} sites, using search gridsize of {gridsize} and search margin of {margin}, using {output_folder} as the out_folder. Finally, run the stability MD pipeline (protein topology, ligand preparation from docking results, GROMACS MD, and stability analysis) using protein_file={cleaned_receptor_path}, docking_csv={docking_results_csv}, ligand_input={ligand_csv}, and md_workdir={output_folder}. I want to use all cores (no GPU).
```

---

## Example 2: Complete End-to-End Workflow (GPU)

**User Request**: "Run the complete docking workflow with GPU acceleration."

**Applying General Principles**:
- **Principle 1**: Explicitly pass output_folder to all stages
- **Principle 2**: Extract paths between stages
- **Principle 3**: Provide all MD parameters upfront
- **Principle 4**: Use same output_folder throughout
- **Principle 5**: Automatic progression through all stages

**Workflow Agent Prompt**:
```
Please run the ENTIRE end-to-end workflow. First, clean the receptor at {receptor_path} keeping chain {chain} and also preprocess the compounds in {ligand_csv_path} with a {molwt_upper_bound} Da cutoff without sampling. Use {output_folder} as the outpath for all preprocessing operations. Next, discover binding sites for the ligands and dock them at the top {N} sites, using {output_folder} as the out_folder. Finally, run the stability MD pipeline using protein_file={cleaned_receptor_path}, docking_csv={docking_results_csv}, ligand_input={ligand_csv}, and md_workdir={output_folder}. Enable GPU/CUDA acceleration (gromacs_binary_type=\"cuda\", GroConfig.gpu=True). I want to use all cores and GPUs when applicable. Use GPU for production docking.
```

---

## Example 3: Docking Only (Data Already Prepared)

**User Request**: "Run only the docking pipeline (no preprocessing or MD) with the ligands from {ligand_csv_path} and the protein {receptor_path}. First discover the binding sites, then dock all ligands at the top {N} sites."

**Applying General Principles**:
- **Principle 1**: Explicitly pass output_folder to docking agent
- **Principle 4**: Use consistent output_folder (even for single stage)

**Workflow Agent Prompt**:
```
Please run only the docking pipeline (no preprocessing or MD) with the ligands from {ligand_csv_path} and the protein {receptor_path}. First discover the binding sites, then dock all ligands at the top {N} sites. Use {output_folder} as the out_folder for all docking operations. I want to use all cores and GPUs when applicable. Use GPU for production docking.
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

## Example 5: MD Analysis Only (Using Existing Docking Results)

**User Request**: "Run MD analysis on my docking results."

**Applying General Principles**:
- **Principle 1**: Explicitly pass md_workdir to MD agent
- **Principle 3**: Provide all required MD parameters (protein_file, docking_csv, ligand_input, md_workdir) in single call
- **Principle 4**: Use consistent output folder

**Workflow Agent Prompt**:
```
Please run the stability MD pipeline (protein topology, ligand preparation from docking results, GROMACS MD, and stability analysis) using protein_file={receptor_path}, docking_csv={docking_results_csv}, ligand_input={ligand_csv_path}, and md_workdir={output_folder}. I want to use all cores and GPUs when applicable.
```

---

## Example 6: Complete Workflow with Sampling

**User Request**: "Run the complete docking workflow with 50% sampling."

**Applying General Principles**:
- **Principle 1**: Explicitly pass output_folder to all stages
- **Principle 2**: Extract sampled CSV path from preprocessing (not small_mw)
- **Principle 4**: Use same output_folder throughout
- **Principle 5**: Automatic progression through stages

**Workflow Agent Prompt**:
```
Please run the complete docking workflow. Clean the receptor at {receptor_path} keeping chain {chain} and preprocess the compounds in {ligand_csv_path} with a {molwt_upper_bound} Da cutoff and {sampling_frac} sampling. Use {output_folder} as the outpath for all preprocessing operations. Discover top {N} binding sites, then dock at those sites using {output_folder} as the out_folder. I want to use all cores and GPUs when applicable. Use GPU for production docking.
```

---

## Example 7: Mid-Pipeline Start - Resume at MD with External Docking Results

**User Request**: "I have completed docking externally. Run MD analysis starting from ligand preparation."

**Workflow Agent Prompt**:
```
I have completed docking externally. Run MD analysis starting from ligand preparation. Docking results folder: {docking_results_folder}, SMILES file: {ligand_csv_path}, protein topology files: protein_gro={protein_gro_path}, protein_top={protein_top_path}. Use {output_folder} as the md_workdir. I want to use all cores.
```

---

## Example 8: Mid-Pipeline Start - Resume at Gro (MD Simulations Only)

**User Request**: "Ligand preparation is complete. Run MD simulations on these pose directories."

**Workflow Agent Prompt**:
```
Ligand preparation is complete. Run MD simulations on these pose directories: {pose_dir1}, {pose_dir2}, {pose_dir3} in {md_workdir}/md_analysis/poses/. Use pose_dirs={comma_separated_pose_dirs}. I want to use all cores.
```

---

## Example 9: Mid-Pipeline Start - Stability Analysis Only

**User Request**: "MD simulations are complete. Run stability analysis only on the completed trajectories."

**Workflow Agent Prompt**:
```
MD simulations are complete. Run stability analysis only on the completed trajectories at {md_workdir}. Use pose_dirs_analysis={comma_separated_pose_dirs} if specific poses needed. I want to use all cores.
```

---

## Example 10: Mid-Pipeline Start - Production Docking Only (Skip Search, with Coordinates)

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

## Example 12: Mid-Pipeline Start - Start at ProteinTopology (Skip Preprocessing and Docking)

**User Request**: "Preprocessing and docking are complete. Run the full MD pipeline starting from protein topology."

**Applying General Principles**:
- **Principle 1**: Explicitly pass md_workdir to MD agent
- **Principle 3**: Provide all required MD parameters (protein_file, docking_csv, ligand_input, md_workdir) in single call
- **Principle 4**: Use consistent output folder (should match where preprocessing/docking outputs are)

**Workflow Agent Prompt**:
```
Preprocessing and docking are complete. Run the full MD pipeline starting from protein topology. Use protein_file={receptor_path}, docking_csv={docking_results_csv}, ligand_input={ligand_csv_path}, and md_workdir={output_folder}. I want to use all cores.
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
- **For LigPrepare entry** (skip ProteinTopology): Provide `protein_gro=/path/to/protein.gro, protein_top=/path/to/topol.top`
- **For Gro entry** (skip LigPrepare): Provide `pose_dirs=pose1,pose2,pose3` (comma-separated list of pose directory names)
- **For StabilityAnalysis entry** (skip Gro): Provide `pose_dirs_analysis=pose1,pose2,pose3` (comma-separated list)
- **For external docking results**: Ensure user's docking CSV has `pdbqt_path` column if PDBQT files are in non-standard locations
