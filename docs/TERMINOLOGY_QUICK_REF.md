# Pipeline Terminology - Quick Reference

> **Purpose**: This document defines standard terminology for clear communication between users, developers, and agents about the molecular docking and MD simulation pipeline.

---

## Table of Contents
1. [Core Pipeline Concepts](#core-pipeline-concepts)
2. [Workflow Concepts](#workflow-concepts)
3. [Code Components](#code-components)
4. [Agent Components](#agent-components)
5. [Data & File Organization](#data--file-organization)
6. [Execution & Logging](#execution--logging)
7. [Critical Boundaries](#critical-boundaries)
8. [Quick Reference](#quick-reference)

---

## Core Pipeline Concepts

### **Pipeline Stage** (or **Stage**)
One of three major sequential phases that transform data:

1. **Preprocessing** - Clean PDB structures, filter and validate ligand CSVs
2. **Docking** - Discover binding sites, dock ligands at those sites
3. **MD Analysis** - Run molecular dynamics simulations and stability analysis

**Flow**: Preprocessing → Docking → MD (can start at any stage via entry points)

---

### **Pipeline Execution**
One complete run of the workflow agent through pipeline stages.

**Properties:**
- **1 Pipeline Execution = 1 log file** (`adams_pipeline_{identifier}.log`)
- **1 Pipeline Execution = 1 run directory** (`run_{identifier}/`)
- Can span multiple stages or just one (depending on entry point)
- Managed by the workflow agent

**Example**: "Execution in `run_20251217_143022/` completed preprocessing and docking stages."

---

### **Full Pipeline Run** (or **Full Run**, **Complete Pipeline**)
A pipeline execution that includes **ALL THREE stages** in sequence: Preprocessing → Docking → MD Analysis.

**CRITICAL**: A "full run" **ALWAYS includes MD Analysis**. When users request:
- "full run"
- "full pipeline"
- "complete pipeline"
- "end-to-end pipeline"
- "run the pipeline" (from raw inputs)
- "whole pipeline"
- "entire workflow"

They mean all three stages including MD. **Do NOT ask if MD is included** - it is always included in a full run.

**Example**: "I requested a full run, and it automatically executed preprocessing, docking, and MD analysis."

---

### **Entry Point**
A specific stage where a pipeline execution can begin, enabling:
- Resuming interrupted executions
- Using pre-processed data from external sources
- Running only specific stages

**7 Entry Points:**
1. **Preprocessing** - Start from raw PDB and CSV
2. **Search Docking** - Start from cleaned receptor (discover binding sites)
3. **Production Docking** - Start with known binding sites
4. **MD - Protein Topology** - Start MD from protein parameterization
5. **MD - Ligand Prep** - Start MD with existing protein topology
6. **MD - Simulations** - Start MD with prepared poses
7. **MD - Analysis Only** - Analyze existing MD trajectories

**Example**: "Resume at Entry Point 3 (Production Docking) with known binding sites."

---

### **Agent Session**
One complete run of `cli.py` - the top-level interaction with the controller agent.

**Properties:**
- **1 Agent Session = 1 trace file** (`trace_YYYYMMDD_HHMMSS.jsonl`)
- **Can contain multiple pipeline executions**
- Trace file captures all agent interactions across ALL executions in the session

**Example**: "This session ran 3 pipeline executions with different parameters."

---

## Workflow Concepts

### **Search Docking** vs **Production Docking**

| Aspect | Search Docking | Production Docking |
|--------|----------------|-------------------|
| **Mode** | `mode="search"` | `mode="production"` |
| **Purpose** | Discover/explore binding sites | Dock full ligand set at known sites |
| **Method** | Systematically dock small ligands (<300 Da) | Focused docking at specific coordinates |
| **Workflow** | Followed by FindPocket clustering | Preceded by search docking OR user coordinates |
| **Hardware** | CPU-only | CPU or GPU (GPU for high-throughput) |
| **Output** | `best_search_docking_centers.csv` | `production_docking_results.csv` |

**Example**: "Search docking found 3 binding sites. Production docking screened 1000 ligands at those sites."

---

### **Binding Site** (or **Pocket** or **Docking Center**)
A specific 3D coordinate (x, y, z) on the protein surface where ligands are docked.

**Discovery Methods:**
- Search docking + FindPocket clustering
- User-provided coordinates
- Existing docking_centers.csv file

**Representation**: CSV file with columns: `grid_id, center_x, center_y, center_z, cluster_size, avg_affinity`

**Example**: "Pocket 1 at (25.3, 18.7, -5.2) with average affinity -8.5 kcal/mol."

---

### **File Path Handoff**
Mechanism where outputs from one stage become inputs to the next. The workflow agent extracts exact file paths from stage outputs and passes them to the next stage agent.

**Critical Handoffs:**
- **Preprocessing → Docking**: Cleaned receptor PDB, processed ligand CSV
- **Docking → MD**: Production docking results CSV, ligand CSV, cleaned receptor

**Example**: "Workflow agent passed `5LS6_A_clean_h.pdb` from preprocessing to docking agent."

---

## Code Components

### **Worker Modules**
Python modules that perform actual scientific computations (algorithms, not orchestration).

**Preprocessing** (two independent operations):
- `clean_pdb.py` - Receptor PDB structure cleaning and preparation
- `data_processing.py` - Ligand CSV filtering and validation
Note: These operations are independent and can be run in any order.

**Docking:**
- `vina_dock.py` - Molecular docking engine (CPU mode)
- `vina_dock_gpu.py` - Molecular docking engine (GPU mode)
- `find_pocket.py` - Binding site detection and clustering

**MD Analysis:**
- `protein_topology.py` - Protein parameterization for GROMACS
- `lig_prepare.py` - Ligand preparation for MD (ligprep)
- `run_gro.py` - GROMACS simulation execution
- `stability_analysis.py` - MD trajectory analysis

---

### **Utility Modules**
Python modules providing shared helper functions (no classes, no orchestration).

**Docking Utilities:**
- `utils.py` - PDBQT parsing, grid generation, coordinate transformations

**MD Analysis Utilities:**
- `_utils.py` - GROMACS utilities (file cleanup, binary detection, topology editing)
- `agent_utils.py` - Tool functions for MD agent (file path discovery, structure)

**Global Utilities:**
- `logger_utils.py` - Centralized logging
- `file_organization.py` - Directory structure management
- `reference_file_reader.py` - Access to reference documentation files

---

## Agent Components

### **Main Agents** (orchestrate the pipeline)

#### **Controller Agent** (`executive_agent.py`)
- **Role**: Top-level user interface
- **Scope**: Entire agent session, multiple executions
- **Capabilities**: Interpret user intent, plan executions, compare results
- **Example**: "Run 3 parameter sweeps and compare docking results"

#### **Workflow Agent** (`workflow_agent.py`)
- **Role**: Coordinate stages within a single execution
- **Scope**: One pipeline execution (preprocessing → docking → MD)
- **Capabilities**: Manage stage sequence, file path handoffs, logging setup
- **Example**: "Hand off cleaned PDB from preprocessing to docking stage"

---

### **Stage Agents** (also called **Domain Agents**)
LLM-powered agents that orchestrate specific pipeline stages.

- `preprocessing_agent.py` - Orchestrates preprocessing stage
- `docking_agent.py` - Orchestrates docking stage
- `md_agent.py` - Orchestrates MD analysis stage

**Key Point**: Stage agents call worker modules with appropriate parameters.

---

### **Helper Agents**
Specialized support agents that don't execute pipeline stages.

- `file_finder_agent.py` - Scan files, recommend entry points
- `meta_analysis_agent.py` - Analyze past execution traces and log files
- `file_parser_agent.py` - Extract data from various file formats
- `oversight_agent.py` - Monitor and validate execution

**Key Point**: Helper agents support main agents but don't run pipeline stages.

---

## Data & File Organization

### **Run Directory**
Timestamped output folder for a single pipeline execution.

**Format**: `agent_data/outputs/run_YYYYMMDD_HHMMSS/`

**Contains:**
- `preprocessing/` - Stage output directory
- `docking/` - Stage output directory
- `md_analysis/` - Stage output directory

**Example**: "All outputs are in `run_20251217_143022/`."

---

### **Stage Output Directory**
Subdirectory within run directory for a specific stage's outputs.

**Structure:**
- `{run_dir}/preprocessing/` - Cleaned receptors, processed CSVs
- `{run_dir}/docking/search/` - Search docking results
- `{run_dir}/docking/production/` - Production docking results  
- `{run_dir}/md_analysis/` - MD simulation outputs

**Example**: "Docking results are in `run_20251217_143022/docking/production/`."

---

## Execution & Logging

### **Trace File** vs **Log File**

| Aspect | Trace File | Log File |
|--------|-----------|----------|
| **Scope** | Entire agent session (cli.py) | Single pipeline execution |
| **Count** | 1 per cli.py run | 1 per workflow agent run |
| **Created by** | Controller agent (setup_tracing) | Workflow agent (setup_pipeline_logger) |
| **Contains** | All agent calls, tool calls, handoffs | Detailed execution steps, timing, errors |
| **Location** | `agent_data/traces/` | `agent_data/logs/` |
| **Naming** | `trace_YYYYMMDD_HHMMSS.jsonl` | `adams_pipeline_{run_id}.log` |
| **Use for** | Session analysis, agent behavior | Execution timing, debugging, resume |

**Example Scenario:**
- Run `cli.py` once → Creates `trace_20251217_143022.jsonl`
- User asks for 3 executions → Creates 3 log files:
  - `adams_pipeline_run_20251217_143100.log`
  - `adams_pipeline_run_20251217_143200.log`  
  - `adams_pipeline_run_20251217_143300.log`
- All 3 executions recorded in the SAME trace file

---

## Critical Boundaries

| Component | Uses LLM? | Calls Workers? | Executes Stage? | Purpose |
|-----------|-----------|----------------|-----------------|---------|
| **Utility Module** | ❌ No | ❌ No | ❌ No | Provides helper functions |
| **Worker Module** | ❌ No | ❌ No* | ✅ Yes | Does computational work |
| **Helper Agent** | ✅ Yes | ❌ No | ❌ No | Provides specialized support |
| **Stage Agent** | ✅ Yes | ✅ Yes | ✅ One stage | Orchestrates one pipeline stage |
| **Workflow Agent** | ✅ Yes | ❌ No** | ✅ Multiple | Coordinates multiple stages |
| **Controller Agent** | ✅ Yes | ❌ No** | ❌ No | Plans and manages executions |

\* Workers may call utility functions  
\*\* Agents call other agents; stage agents call workers

---

## Quick Reference

### Common Mistakes vs Correct Usage

| ❌ Bad Terminology | ✅ Good Terminology |
|-------------------|---------------------|
| "The vina_dock agent performed docking" | "The docking agent called the vina_dock worker module" |
| "The workflow module processed the stage" | "The workflow agent orchestrated the preprocessing stage agent" |
| "The helper executed the pipeline" | "The helper agent found entry points; the controller agent executed the pipeline" |
| "The ligprep module orchestrated ligand preparation" | "The MD agent called the lig_prepare worker module" |
| "The utils agent converts files" | "The docking worker uses utility functions from utils.py" |
| "Check the trace file for execution timing" | "Check the log file for timing; trace shows agent interactions" |
| "This run created a new trace file" | "This execution created a new log file within the session's trace" |
| "Run docking to find where ligands bind" | "Run search docking to discover sites, then production docking" |
| "The docking finished" | "Search docking found 3 pockets. Production docking completed." |
| "Start from the docking step" | "Resume at Entry Point 2 (Search Docking) with cleaned receptor" |

---

### Common Abbreviations

| Abbreviation | Full Term | Meaning |
|--------------|-----------|---------|
| **PDB** | Protein Data Bank | Standard file format for protein structures |
| **PDBQT** | PDB with Charges and Atom Types | Docking-ready protein/ligand format |
| **CSV** | Comma-Separated Values | Tabular data format for ligand lists |
| **SMILES** | Simplified Molecular Input Line Entry System | Text representation of chemical structures |
| **MD** | Molecular Dynamics | Simulation method for studying molecular motion |
| **GROMACS** | GROningen MAchine for Chemical Simulations | MD simulation software package |
| **MolWt** | Molecular Weight | Mass of a molecule in Daltons (Da) |
| **GPU** | Graphics Processing Unit | Hardware accelerator for docking/MD |
| **CPU** | Central Processing Unit | Standard processor (search docking only) |
| **RMSD** | Root Mean Square Deviation | Measure of structural similarity |

---

### Reference Documentation

For detailed information, see:

- **`terminology.md`** - Comprehensive definitions with examples
- **`entry_points.md`** - All 7 entry points with required files and detection signals
- **`parameter_defaults.md`** - Default values for all pipeline parameters
- **`workflow_examples.md`** - Example user requests and agent responses
- **`directory_structure.md`** - Output organization patterns
- **`file_path_mapping.md`** - How files are passed between stages
- **`error_handling.md`** - Error patterns and recovery strategies

All reference files are in: `adams/pipeline/references/`

---

**Last Updated**: December 2025
