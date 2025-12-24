# Agent Documentation

This document provides an overview of all agents in the vina_dock codebase, their purposes, and available tools. For workflow execution details, see `WORKFLOW_EXECUTION.md`.

## Table of Contents

1. [Agent Groups](#agent-groups)
2. [Agent Hierarchy](#agent-hierarchy)
3. [Agent-to-Agent Call Matrix](#agent-to-agent-call-matrix)
4. [Detailed Agent Flow](#detailed-agent-flow)

---

## Agent Groups

### Universal Tools

**Available to ALL agents:**
- **`read_reference_file`** - Reads reference markdown files from `adams/pipeline/references/`

---

### Orchestration Agents

High-level agents that coordinate pipeline execution and manage workflow.

#### Biophysics Controller Agent

**Location:** `adams/executive_agent.py` | **Model:** `gpt-5.2-pro`

Main entry point agent that interprets user intent, plans pipeline executions, and manages multiple runs.

**Tools:**
- `read_reference_file` (universal)
- `list_agent_data_files_tool` - Scans `agent_data` directory for receptor/ligand files
- `meta_analysis_agent` (sub-agent)
- `file_finder_agent` (sub-agent)
- `file_parser_agent` (sub-agent)
- `oversight_agent` (sub-agent)
- `workflow_agent` (sub-agent)

#### Workflow Agent

**Location:** `adams/pipeline/workflow_agent.py` | **Model:** `gpt-5.2-pro`

Orchestrates the complete molecular docking workflow by coordinating domain agents and managing execution flow.

**Tools:**
- `read_reference_file` (universal)
- `create_run_directory` - Creates timestamped run directory
- `setup_pipeline_logger` - Sets up centralized logging
- `file_parser_agent` (sub-agent)
- `preprocessing_agent` (sub-agent)
- `docking_agent` (sub-agent)
- `md_agent` (sub-agent)

---

### Domain/Stage Agents

Agents that orchestrate specific pipeline stages and perform computational workflows.

**Shared Tools:**
- `read_reference_file` (universal)
- `file_parser_agent` (sub-agent) - Used by Docking and MD agents

#### Preprocessing Agent

**Location:** `adams/pipeline/data_preprocessing/preprocessing_agent.py` | **Model:** `gpt-5.2`

Orchestrates preprocessing: cleaning receptor PDBs and processing ligand CSVs.

**Tools:**
- `read_reference_file` (universal)
- `run_clean_pdb` - Cleans receptor PDB file
- `run_data_processing` - Processes compound CSV and generates cleaned/sampled outputs

#### Docking Agent

**Location:** `adams/pipeline/docking/docking_agent.py` | **Model:** `gpt-5.2`

Orchestrates docking: binding site discovery and molecular docking.

**Tools:**
- `read_reference_file` (universal)
- `file_parser_agent` (sub-agent)
- `run_vina_dock` - Molecular docking using AutoDock Vina (CPU-based, search_dock or production modes)
- `run_vina_dock_gpu` - Production-level molecular docking on GPU
- `run_find_pocket` - Clusters search docking results to identify binding pockets

#### MD Agent

**Location:** `adams/pipeline/md_analysis/md_agent.py` | **Model:** `gpt-5.2`

Orchestrates MD analysis: protein topology, ligand preparation, MD simulations, and stability analysis.

**Tools:**
- `read_reference_file` (universal)
- `file_parser_agent` (sub-agent)
- `build_file_paths` - Builds file_paths dictionary from paths or discovers from existing MD directory
- `discover_paths` - Discovers GROMACS and AmberTools installation paths
- `run_protein_topology` - Prepares protein structure for MD simulation
- `run_lig_prepare` - Prepares ligands for MD simulation
- `run_gro` - Runs MD simulations (NVT, NPT, production)
- `run_stability_analysis` - Analyzes MD trajectories for stability metrics

---

### Helper/Support Agents

Specialized agents that provide support functions like file discovery, result analysis, and plan validation.

**Shared Tools:**
- `read_reference_file` (universal)

#### File Finder Agent

**Location:** `adams/helper_agents/file_finder/file_finder_agent.py` | **Model:** `gpt-5-mini`

Identifies and classifies files in `agent_data/` to determine available pipeline entry points.

**Tools:**
- `read_reference_file` (universal)
- `scan_directory` - Recursively scans directory and returns metadata
- `read_csv_headers` - Reads CSV headers to identify file type
- `check_file_exists` - Checks file existence and returns metadata
- `check_directory_contents` - Checks if directory contains required files
- `read_file_preview` - Reads first N lines of a text file

**Used by:** Biophysics Controller Agent

#### File Parser Agent

**Location:** `adams/helper_agents/file_parser/file_parser_agent.py` | **Model:** `gpt-5-mini`

Extracts structured statistics from pipeline output files for parameter extraction and result-based decision making.

**Tools:**
- `read_reference_file` (universal)
- `parse_docking_results` - Parses docking results CSV and extracts statistics
- `parse_md_results` - Analyzes MD results directory and extracts completion status

**Used by:** Docking Agent, MD Agent, Workflow Agent

#### Meta Analysis Agent

**Location:** `adams/helper_agents/meta_analysis/meta_analysis_agent.py` | **Model:** `gpt-5-mini`

Analyzes pipeline trace files and log files to understand run state for resuming, error handling, and context extraction.

**Tools:**
- `read_reference_file` (universal)
- `read_trace_file` - Reads trace file and returns raw contents (JSONL)
- `parse_trace_file` (RECOMMENDED) - Parses trace file and extracts structured information
- `parse_log_file` (RECOMMENDED) - Parses log file and extracts structured information
- `search_log_file` - Searches log file for specific patterns
- `list_trace_files` - Lists all available trace files
- `list_log_files` - Lists all available log files and extracts run identifiers

**Used by:** Biophysics Controller Agent

#### Oversight Agent

**Location:** `adams/helper_agents/oversight/oversight_agent.py` | **Model:** `gpt-5.2`

Reviews and validates pipeline execution plans to ensure they are scientifically sound and align with user intent.

**Tools:**
- `read_reference_file` (universal)
- `submit_review` - Submits structured review of proposed pipeline execution plan

**Used by:** Biophysics Controller Agent (CRITICAL: must use before workflow_agent)

---

## Agent Hierarchy

```
Biophysics Controller Agent (Main Entry Point)
│
├──→ Oversight Agent (Plan Validation)
├──→ Trace Analysis Agent (Run State Analysis)
├──→ File Finder Agent (File Discovery)
├──→ File Parser Agent (Result Analysis)
│
└──→ Workflow Agent (Pipeline Orchestration)
    ├──→ File Parser Agent (Result Analysis)
    │
    ├──→ Preprocessing Agent (Stage 1)
    │
    ├──→ Docking Agent (Stage 2)
    │    └──→ File Parser Agent (Result Analysis)
    │
    └──→ MD Agent (Stage 3)
         └──→ File Parser Agent (Result Analysis)
```

---

## Agent-to-Agent Call Matrix

This table shows which agents can invoke which other agents as sub-agents.

| **Invoking Agent** | **Can Call These Sub-Agents** | **Purpose** |
|-------------------|-------------------------------|-------------|
| **Biophysics Controller** | `meta_analysis_agent` | Analyze pipeline trace files and log files and run state |
| | `file_finder_agent` | Discover and classify files in agent_data/ |
| | `file_parser_agent` | Parse result files for statistics |
| | `oversight_agent` | ⚠️ **CRITICAL**: Validate execution plans (required before workflow_agent) |
| | `workflow_agent` | Execute complete pipeline workflow |
| **Workflow Agent** | `file_parser_agent` | Parse intermediate/final results |
| | `preprocessing_agent` | Execute data preprocessing stage |
| | `docking_agent` | Execute molecular docking stage |
| | `md_agent` | Execute MD simulation stage |
| **Preprocessing Agent** | *(none)* | Pure execution agent - uses direct tools only |
| **Docking Agent** | `file_parser_agent` | Parse docking results for decision-making |
| **MD Agent** | `file_parser_agent` | Parse MD results for decision-making |
| **File Finder Agent** | *(none)* | Helper agent - no sub-agents |
| **File Parser Agent** | *(none)* | Helper agent - no sub-agents |
| **Meta Analysis Agent** | *(none)* | Helper agent - no sub-agents |
| **Oversight Agent** | *(none)* | Helper agent - no sub-agents |

### Key Patterns

1. **Helper agents never call other agents** - They are leaf nodes in the hierarchy
2. **File Parser Agent is the most reused** - Called by 4 different agents (Controller, Workflow, Docking, MD)
3. **Critical validation flow**: Biophysics Controller → Oversight Agent → Workflow Agent
4. **Stage agents have limited sub-agents** - Domain agents mostly use direct computational tools

---

## Detailed Agent Flow

### Complete Execution Flow

```
USER REQUEST
    ↓
┌─────────────────────────────────────────────────────────────────┐
│ BIOPHYSICS CONTROLLER AGENT (Entry Point)                       │
│ Model: gpt-5.2-pro                                              │
├─────────────────────────────────────────────────────────────────┤
│ Phase 1: Discovery & Analysis                                   │
│   • list_agent_data_files_tool → Find available inputs         │
│   • file_finder_agent → Classify files & determine entry points│
│   • meta_analysis_agent → Check for existing runs to resume   │
│   • file_parser_agent → Analyze results from previous runs     │
│                                                                  │
│ Phase 2: Planning & Validation                                  │
│   • Build execution plan based on user intent + available data  │
│   • oversight_agent → ⚠️ REQUIRED validation of plan           │
│                                                                  │
│ Phase 3: Execution                                              │
│   • workflow_agent → Execute validated pipeline                │
└─────────────────────────────────────────────────────────────────┘
                          ↓
         ┌────────────────────────────────────────┐
         │ WORKFLOW AGENT (Pipeline Orchestrator) │
         │ Model: gpt-5.2-pro                     │
         ├────────────────────────────────────────┤
         │ • create_run_directory                 │
         │ • setup_pipeline_logger                │
         │ • Coordinate stage agents sequentially │
         └────────────────────────────────────────┘
                          ↓
    ┌──────────────┬──────────────┬──────────────┐
    ↓              ↓              ↓              ↓
┌─────────┐  ┌──────────┐  ┌────────────┐  ┌──────────┐
│  FILE   │  │PREPROC   │  │  DOCKING   │  │    MD    │
│ PARSER  │  │  AGENT   │  │   AGENT    │  │  AGENT   │
│  AGENT  │  │          │  │            │  │          │
├─────────┤  ├──────────┤  ├────────────┤  ├──────────┤
│Parse    │  │Clean PDB │  │Find Pocket │  │Topology  │
│results  │  │Process   │  │Run Docking │  │Lig Prep  │
│Extract  │  │CSV data  │  │   ↓        │  │Run MD    │
│stats    │  │Generate  │  │Parse       │  │Stability │
│         │  │configs   │  │results ────┼──→Analysis  │
│         │  │          │  │via File    │  │   ↓      │
│         │  │          │  │Parser      │  │Parse via │
│         │  │          │  │            │  │File      │
│         │  │          │  │            │  │Parser    │
└─────────┘  └──────────┘  └────────────┘  └──────────┘
    ↑                            ↑              ↑
    └────────────────────────────┴──────────────┘
           (Called by multiple agents)
```

### Agent Communication Pattern

**Sequential Stage Execution:**
```
Workflow Agent coordinates:
  1. Preprocessing Agent (parallel if multiple receptors)
  2. Docking Agent (sequential: search → find_pocket → production)
  3. MD Agent (sequential: topology → ligands → NVT → NPT → production → analysis)
```

**Parallel Parsing:**
```
File Parser Agent can be called concurrently by:
  • Workflow Agent (checking overall progress)
  • Docking Agent (extracting best poses)
  • MD Agent (verifying completion status)
```

**Critical Decision Points:**
```
┌─ Oversight Agent validates plan ────→ APPROVED? ─┐
│                                                    │
│  YES: Proceed to Workflow Agent                   │
│  NO:  Revise plan or abort execution              │
└────────────────────────────────────────────────────┘
```
