# Terminology Reference

This document defines the standard terminology used throughout the codebase to ensure clear communication between users, developers, and the agentic system.

## Core Concepts

### **Pipeline Execution**
A **Pipeline Execution** (or simply **Execution**) is a single run of the computational pipeline that processes input data through one or more stages. Each execution:
- Has a unique timestamped output directory: `agent_data/outputs/run_YYYYMMDD_HHMMSS`
- Has a unique timestamped log: 'agent_data/logs/run_YYYYMMDD_HHMMSS'
- Produces structured outputs organized by stage
- Can be resumed from any entry point if interrupted
- Is managed by the workflow agent

**Example**: "I ran a pipeline execution on protein 5LS6 with 1000 ligands, and it completed preprocessing and docking."

---

### **Full Pipeline Run** (or **Full Run**, **Complete Pipeline**, **End-to-End Pipeline**)
A **Full Pipeline Run** is a pipeline execution that includes ALL THREE stages in sequence:
1. **Preprocessing Stage**: Cleans receptor PDBs and processes ligand CSVs
2. **Docking Stage**: Discovers binding sites and performs molecular docking
3. **MD Analysis Stage**: Runs molecular dynamics simulations and stability analysis

**Key Properties:**
- **ALWAYS includes MD Analysis** - A "full run" is incomplete without MD
- Executes all stages automatically without user confirmation between stages
- Starts from Entry Point 1 (Preprocessing) with raw input files
- Produces outputs from all three stages in a single run directory

**User Terminology:**
When users request any of the following, they mean a **Full Pipeline Run**:
- "full run"
- "full pipeline"
- "complete pipeline"
- "end-to-end pipeline"
- "run the pipeline" (when starting from raw inputs)
- "whole pipeline"
- "entire workflow"

**Example**: "I requested a full run, and the agent executed preprocessing, docking, and MD analysis automatically."

**Partial Runs** (for contrast):
- **Docking Run**: Preprocessing + Docking (no MD) - user must explicitly request "docking only" or "skip MD"
- **MD Run**: MD Analysis only (skips preprocessing and docking) - user must explicitly request "MD only" or provide existing docking results
- **Preprocessing Run**: Preprocessing only - user must explicitly request "preprocessing only"

---

### **Pipeline Stage** (or **Stage**)
A **Pipeline Stage** is one of the three major computational phases that transform data:
1. **Preprocessing Stage**: Consists of two independent operations that can be run in any order:
   - **Receptor Preparation**: Cleans receptor PDBs (run_clean_pdb)
   - **Ligand Preprocessing**: Processes ligand CSVs (run_ligand_preprocessing)
2. **Docking Stage**: Discovers binding sites and performs molecular docking
3. **MD Analysis Stage**: Runs molecular dynamics simulations and stability analysis

Stages are sequential (preprocessing → docking → MD), but executions can start at any stage via entry points. Within the preprocessing stage, receptor preparation and ligand data processing are independent and can be run in any order.

**Example**: "The docking stage completed successfully, producing 50 top poses."

---

### **Computational Module** (or **Core Module**)
A **Computational Module** is a Python module that performs the actual scientific computations. These are distinct from agents—they contain the algorithms, data processing logic, and scientific workflows.

**Preprocessing Modules** (independent operations):
- `clean_pdb.py`: Receptor structure cleaning (receptor preparation)
- `ligand_preprocessing.py`: Ligand CSV filtering and validation (ligand preprocessing)

**Docking Modules**:
- `vina_dock.py`: Core docking engine (handles both search and production docking)
- `find_pocket.py`: Pocket detection and clustering from docking results

**MD Analysis Modules**:
- `protein_topology.py`: Protein parameterization
- `lig_prepare.py`: Ligand preparation for MD
- `run_gro.py`: GROMACS simulation execution
- `stability_analysis.py`: Trajectory analysis

**Example**: "The `vina_dock.py` computational module handles both CPU and GPU docking."

---

### **Domain Agent** (or **Stage Agent**)
A **Domain Agent** is an LLM-powered agent that orchestrates a specific pipeline stage. Domain agents:
- Interpret natural language instructions
- Call computational modules with appropriate parameters
- Handle file path mapping and data flow between stages
- Provide stage-specific expertise

**Domain Agents**:
- `preprocessing_agent`: Orchestrates preprocessing stage
- `docking_agent`: Orchestrates docking stage
- `md_agent`: Orchestrates MD analysis stage

**Example**: "The docking agent selected GPU mode and configured the search grid based on the receptor size."

---

### **Workflow Agent**
The **Workflow Agent** is the orchestrator that coordinates multiple pipeline stages. It:
- Manages the sequence of domain agents
- Handles file path handoffs between stages
- Sets up logging and run directories
- Implements entry point detection and resume logic

**Example**: "The workflow agent automatically proceeded from preprocessing to docking without user confirmation."

---

### **Controller Agent** (or **Top-Level Agent**)
The **Controller Agent** is the highest-level agent that interacts with users. It:
- Interprets user intent and plans executions
- Manages multiple pipeline executions (comparisons, parameter sweeps)
- Delegates to workflow agent for actual pipeline execution
- Provides summaries and comparisons across runs

**Example**: "The controller agent planned three executions to compare different sampling strategies."

---

### **Agent Session** (or **Session**)
An **Agent Session** is a single interaction period with the controller agent. A session may involve:
- One or more pipeline executions
- Analysis of past executions
- Parameter exploration and comparisons
- File discovery and entry point detection

**Example**: "In this session, we ran two executions and compared their docking results."

---

### **Entry Point**
An **Entry Point** is a specific stage where a pipeline execution can begin, allowing:
- Resuming interrupted executions
- Using pre-processed data from external sources
- Running only specific stages (e.g., MD analysis on existing docking results)

**Entry Points**:
1. **Preprocessing**: Start from raw PDB and CSV
2. **Search Docking**: Start from cleaned receptor and CSV
3. **Production Docking**: Start from cleaned receptor, CSV, and known binding sites
4. **MD Analysis**: Start from various points in the MD workflow

**Example**: "We resumed the execution at Entry Point 3 (Production Docking) using the binding sites from a previous run."

---

## Helper Components

### **Helper Agent**
A **Helper Agent** provides specialized support functions but does not execute pipeline stages:
- `file_finder_agent`: Scans files and recommends entry points
- `meta_analysis_agent`: Analyzes past execution traces and log files for resume/analysis

**Example**: "The file finder agent identified that we can start at Entry Point 2."

---

### **Utility Module**
A **Utility Module** provides shared functionality used across the codebase:
- `logger_utils.py`: Centralized logging
- `file_organization.py`: Directory structure management
- `utils.py`: Common helper functions

**Example**: "The utility module handles automatic directory creation for each stage."

---

## Data Flow Terminology

### **Run Directory**
A **Run Directory** is the timestamped output folder for a single pipeline execution: `agent_data/outputs/run_YYYYMMDD_HHMMSS`. Contains all stage outputs organized by subdirectory.

**Example**: "All results from this execution are in `run_20251203_143022`."

---

### **Stage Output Directory**
A **Stage Output Directory** is a subdirectory within a run directory containing outputs from a specific stage:
- `{run_dir}/preprocessing/`: Preprocessing outputs
- `{run_dir}/docking/`: Docking outputs
- `{run_dir}/md_analysis/`: MD analysis outputs

**Example**: "The docking stage output directory contains search and production subdirectories."

---

### **File Path Handoff**
A **File Path Handoff** is the mechanism by which outputs from one stage become inputs to the next. The workflow agent extracts exact file paths from stage outputs and passes them to the next domain agent.

**Example**: "The file path handoff from preprocessing to docking passed the cleaned receptor PDB and sampled ligand CSV."

---

## Execution Context

### **Execution Plan**
An **Execution Plan** is the controller agent's strategy for running one or more pipeline executions, including:
- Number of executions
- Parameters for each
- Expected comparisons or outcomes

**Example**: "The execution plan includes three runs: baseline, high-throughput (GPU), and extended sampling."

---

### **Execution State**
An **Execution State** tracks the progress of a pipeline execution:
- Which stages have completed
- Current stage and step
- Available entry points for resume
- Error context if interrupted

**Example**: "The execution state shows preprocessing and docking completed, ready for MD analysis."

---

## Summary Table

| Term | Definition | Example |
|------|-----------|---------|
| **Pipeline Execution** | Single run through pipeline stages | "Execution in `run_20251203_143022`" |
| **Full Pipeline Run** | All three stages: Preprocessing → Docking → MD (always includes MD) | "Full run completed all stages" |
| **Pipeline Stage** | One of three major phases (preprocessing, docking, MD) | "The docking stage completed" |
| **Computational Module** | Core scientific computation code | "`vina_dock.py` module" |
| **Domain Agent** | LLM agent orchestrating a stage | "The docking agent" |
| **Workflow Agent** | Agent coordinating multiple stages | "Workflow agent handoff" |
| **Controller Agent** | Top-level user-facing agent | "Controller agent planned 3 runs" |
| **Agent Session** | User interaction period | "This session" |
| **Entry Point** | Stage where execution can begin | "Entry Point 2: Search Docking" |
| **Helper Agent** | Specialized support agent | "File finder agent" |
| **Utility Module** | Shared functionality code | "Logger utility" |
| **Run Directory** | Timestamped output folder | "`run_20251203_143022`" |
| **Stage Output Directory** | Stage-specific subdirectory | "`docking/` subdirectory" |
| **File Path Handoff** | Data flow between stages | "Path handoff from preprocessing" |
| **Execution Plan** | Strategy for runs | "Plan: 3 executions" |
| **Execution State** | Progress tracking | "State: docking complete" |

---

## Usage Guidelines

1. **When discussing code structure**: Use "computational module" for direct scientific computation, and "agent" for LLM-powered orchestration that coordinates computational modules.

2. **When discussing pipeline flow**: Use "pipeline stage" to refer to preprocessing/docking/MD, and "pipeline execution" to refer to a complete run.

3. **When discussing agent hierarchy**: Use "domain agent" for stage-specific agents, "workflow agent" for coordination, and "controller agent" for top-level interaction.

4. **When discussing user interactions**: Use "agent session" for the conversation period, which may include multiple "pipeline executions."

5. **When discussing data flow**: Use "file path handoff" to describe how outputs become inputs between stages.
