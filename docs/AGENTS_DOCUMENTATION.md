# Agent Documentation

This document provides an overview of all agents in the vina_dock codebase, their purposes, and available tools. For workflow execution details, see `WORKFLOW_EXECUTION.md`.

## Table of Contents

1. [Agent Groups](#agent-groups)
2. [Agent Hierarchy](#agent-hierarchy)
3. [Agent-to-Agent Call Matrix](#agent-to-agent-call-matrix)
4. [Detailed Agent Flow](#detailed-agent-flow)
5. [Planning Capabilities and Tool Integration](#planning-capabilities-and-tool-integration)

---

## Agent Groups

**Model configuration:** All agents use the same LLM model, set at agent creation via `create_agent(session_id, model=...)`. The default is `gpt-5.4` (see `adams/model_config.py`). Documentation may refer to other models (e.g. gpt-5.2-pro) as typical deployment choices.

### Universal Tools

**Shared across orchestration and domain agents:**
- **`read_reference_file`** - Reads reference markdown files from `adams/pipeline/references/`

---

### Orchestration Agents

High-level agents that coordinate pipeline execution and manage workflow.

#### Biophysics Controller Agent

**Location:** `adams/executive_agent.py` | **Model:** `gpt-5.2-pro`

**Role:** Top-level user interface. Interprets user intent, decides multi-run strategy, obtains plan approval, delegates single-run execution to the workflow agent. Does not specify pipeline mechanics or stage-level details.

**Tools:**
- `read_reference_file` (universal)
- `read_plan_document` / `append_to_plan_section` - Load plan from path; append user answers or notes to plan sections
- `get_all_plan_tags` / `list_plans_by_tag` - Discover plans by tag (primary for plan reuse); no session tools
- `list_agent_data_files_tool` - Scans `agent_data` directory for receptor/ligand files
- **Persistent memory** (no other session tools): `get_persistent_memory_tool`, `update_user_preference_tool`, `add_learned_behavior_tool`, `set_custom_memory`
- **Session–plan linking:** Done only by the **workflow wrapper** when the executive calls the workflow agent with session_id: the wrapper links the plan (existing or newly created) to the session so meta_analysis can use get_session_plan_summary and read_plan_document. Session description/tags are set by meta_analysis only.
- **Recent session context:** Executive prompt includes a short recent-session summary (tags + recent sessions) for discovery and alignment with past runs.
- `meta_analysis_agent` (sub-agent) - For current-run error solving; has full session memory (read + tag/description)
- `file_finder_agent` (sub-agent)
- `file_parser_agent` (sub-agent)
- `oversight_agent` (sub-agent)
- `workflow_agent` (sub-agent; wrapped so session–plan linkage is automatic on each invocation)

**Plan/Execution split:** Plans are the primary point of contact. Search for relevant plans via plan tags (get_all_plan_tags, list_plans_by_tag); adapt existing plan or create new via workflow; use read_plan_document when given a plan path and structure user questions from the plan's questions array before approval.

#### Workflow Agent

**Location:** `adams/pipeline/workflow_agent.py` | **Model:** `gpt-5.2-pro`

**Role:** Coordinates a single pipeline run (preprocessing → docking → MD). Manages stage sequence, handoffs, paths, logging. Executive provides context (may include preferences from persistent/session memory). Do not assume defaults unless the user, approved plan, or executive context specifies them.

**Tools:**
- `read_reference_file` (universal)
- `create_run_directory` - Creates timestamped run directory
- `setup_pipeline_logger` - Sets up centralized logging
- `create_plan_path` - Creates shared plan JSON file in `agent_data/plans/` (when the wrapper did not already create one)
- `read_plan_document` - Reads plan (returns pretty-printed JSON)
- `append_to_plan_section` - Appends to a plan section: steps (skeleton only, workflow adds first), parameters, questions, answers, additional_notes
- `append_to_step_details` - Appends implementation detail bullets to an existing step by stage and optional step_index (stage agents use this; workflow adds the step skeleton first). When there are multiple steps for the same stage (e.g. two docking steps), step_index (0-based) identifies which step to fill.
- `set_plan_tags` - Assign tag(s) to the current plan so the executive can discover it via list_plans_by_tag
- `file_parser_agent` (sub-agent)
- `preprocessing_agent` (sub-agent)
- `docking_agent` (sub-agent)
- `md_agent` (sub-agent)

**Plan/Execution split:**
- Plan mode (two-phase): (1) Workflow adds the **step skeleton** (ordered stages with descriptions and empty details) via `append_to_plan_section(plan_path, "steps", ...)`. (2) Stage agents use `append_to_step_details(plan_path, stage, content)` to add implementation details for their stage, and `append_to_plan_section` for parameters, questions, additional_notes. Executive presents questions, records user responses in **answers**, then returns final plan.
- Execute mode: run preprocessing -> docking -> MD for one run

---

### Domain/Stage Agents

Agents that orchestrate specific pipeline stages and perform computational workflows.

**Shared Tools:**
- `read_reference_file` (universal)
- `file_parser_agent` (sub-agent) - Used by Docking and MD agents
- `read_plan_document` / `append_to_plan_section` - Used for plan-mode contribution (append to steps, parameters, questions, additional_notes with JSON content; answers filled by executive)

#### Preprocessing Agent

**Location:** `adams/pipeline/data_preprocessing/preprocessing_agent.py` | **Model:** `gpt-5.2`

**Role:** Executes preprocessing stage only (receptor cleaning, protonation, ligand standardization/conformers). Does not coordinate docking or MD.

**Tools:**
- `read_reference_file` (universal)
- `run_clean_pdb` - Cleans receptor PDB file
- `run_data_processing` - Processes compound CSV and generates cleaned/sampled outputs

#### Docking Agent

**Location:** `adams/pipeline/docking/docking_agent.py` | **Model:** `gpt-5.2`

**Role:** Executes docking stage only (search and/or production docking). Does not coordinate preprocessing or MD.

**Tools:**
- `read_reference_file` (universal)
- `file_parser_agent` (sub-agent)
- `run_docking` - Molecular docking with backend selection (vina, vina_gpu, unidock)
- `run_find_pocket` - Clusters search docking results to identify binding pockets

#### MD Agent

**Location:** `adams/pipeline/md_analysis/md_agent.py` | **Model:** `gpt-5.2`

**Role:** Executes MD stage only (prepare, simulate, analyze). Does not coordinate preprocessing or docking.

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

**Role:** Current-run error solving only. Invoked by the controller when something went wrong in the current run. Has session memory (read + write) and is **responsible for session tagging** (e.g. "error", "docking", "incomplete"); the controller does not have session tools.

**Tools:**
- `read_reference_file` (universal)
- **Session memory** (read + write): list sessions, get session info, **set_session_tags**, **set_session_description** — only this agent writes session tags/description
- `read_plan_document` - Read the intended plan when debugging why the current run failed
- `read_trace_file` - Reads trace file and returns raw contents (JSONL)
- `parse_trace_file` (RECOMMENDED) - Parses trace file and extracts structured information
- `parse_log_file` (RECOMMENDED) - Parses log file and extracts structured information
- `search_log_file` - Searches log file for specific patterns
- `list_trace_files` - Lists all available trace files
- `list_log_files` - Lists all available log files and extracts run identifiers

**Used by:** Biophysics Controller Agent (when diagnosing current-run errors)

#### Oversight Agent

**Location:** `adams/helper_agents/oversight/oversight_agent.py` | **Model:** `gpt-5.2`

**Role:** Validates execution plans only (scientific soundness, intent alignment, parameters). Does not execute or suggest pipeline mechanics; provides approve/reject and focused feedback.

**Tools:**
- `submit_review` - Submits structured review of proposed pipeline execution plan
- `read_plan_document` / `append_to_plan_section` - If controller passes plan_path: load plan, optionally append concerns/suggestions to additional_notes

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

**Sequential Stage Execution (single run):**
```
Workflow Agent coordinates:
  1. Preprocessing Agent (parallel if multiple receptors)
  2. Docking Agent (sequential: search → find_pocket → production)
  3. MD Agent (sequential: topology → ligands → NVT → NPT → production → analysis)
```

**Plan Mode (single shared JSON plan):**
```
Executive asks Workflow for per-run plan
  -> Workflow creates plan JSON in agent_data/plans/
  -> Preprocessing reads plan, appends to steps / parameters / questions / additional_notes
  -> Docking appends to same sections
  -> MD appends to same sections
  -> Workflow returns full plan (read_plan_document) to Executive
  -> Executive uses plan.questions (and steps/parameters) to structure user questions, records responses in plan.answers, then requests approval
```

Plan JSON schema (see `adams/user_plan_utils.py`):
- **steps**: list of `{ "stage", "description", "details" }` (pipeline steps per stage)
- **parameters**: dict keyed by stage (e.g. `preprocessing`, `docking`, `md`) with key-value params
- **questions**: list of `{ "id", "stage", "question", "choices"?, "default"? }` for user prompts
- **answers**: dict of question_id -> value (user responses; filled by executive after user replies)
- **additional_notes**: list of strings (freeform notes)

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

---

## Planning Capabilities and Tool Integration

Planning lets the controller get a structured, reviewable plan (steps, parameters, questions) before running the pipeline. Plans are stored as JSON files under `agent_data/plans/` and are filled by the workflow and stage agents, then reviewed by oversight and approved by the user.

### Plan document structure

Defined in `adams/user_plan_utils.py`. Each plan JSON has six sections:

| Section | Who writes it | Purpose |
|--------|----------------|--------|
| **user_prompt** | Workflow (from executive’s verbatim user message) | Original request; used for matching and context. Must be the actual user message. |
| **steps** | Stage agents (preprocessing, docking, md) | Per-stage descriptions and detail lists (e.g. "run_clean_pdb", "run_protonate_receptor"). |
| **parameters** | Stage agents | Key–value params by stage (e.g. `preprocessing.pH`, `docking.engine`). |
| **questions** | Stage agents | User-facing questions (id, stage, question, choices, default) for parameters not fixed by context. |
| **answers** | Executive only | Filled after the user responds; question_id → value (e.g. `{"pocket_choice": "center_1"}`). |
| **additional_notes** | Stage agents or oversight | Freeform notes; oversight can append concerns/suggestions. |

All edits go through **append_to_plan_section(plan_path, section, content)** so the file stays valid and append-only (no merge logic). **content** is JSON for every section except **user_prompt**, which accepts raw text (control chars sanitized).

### Planning tools and who has them

| Tool | Executive | Workflow | Preprocessing | Docking | MD | Oversight |
|------|------------|----------|----------------|----------|-----|-----------|
| **create_plan_path** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **read_plan_document** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **append_to_plan_section** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

- **create_plan_path**: Only the workflow agent creates plan files. It returns the plan path; the executive never creates plans directly.
- **read_plan_document**: All agents that touch plans can read the current JSON (e.g. executive to show the plan, stage agents to see what’s already there, oversight to review).
- **append_to_plan_section**: Executive uses it mainly for **answers** (and **additional_notes** if needed). Stage agents use it for **steps**, **parameters**, **questions**, **additional_notes**. Oversight may append to **additional_notes** (e.g. concerns).

### How planning fits with existing tools

1. **Session memory and plan reuse**  
   - **Executive** discovers plans by **plan tags**: `get_all_plan_tags()` then `list_plans_by_tag(tag)` to get plan_paths; then **read_plan_document(plan_path)** to load and adapt (Principle 5). The executive does not have session memory tools.  
   - **Meta_analysis** uses `get_session_plan_summary(session_id)` to get **plan_paths** for a session, then **read_plan_document(plan_path)** to compare intended vs actual when diagnosing failures.  
   - Plan path association is done only by the **workflow wrapper**: when the executive calls **workflow_agent(message, plan_path=..., session_id=...)**, the wrapper links that plan to the session; when **plan_path** is omitted and **session_id** is set, the wrapper creates a new plan and links it. Sessions store **plan_paths** in `agent_data/memory/sessions.json`.

2. **Oversight**  
   - The executive must submit the **plan document** (content from **read_plan_document(plan_path)**) to the oversight agent, not a hand-written summary.  
   - Oversight has **read_plan_document** and **append_to_plan_section** so it can load the plan and optionally append to **additional_notes**; its main output is still **submit_review** (approve/reject + feedback).

3. **File discovery and reference docs**  
   - Planning does not replace **file_finder_agent** or **list_agent_data_files_tool**. The executive still uses them to find inputs; the workflow and stage agents use that context (and **read_reference_file**) when filling steps/parameters/questions.  
   - Plan **parameters** can reference paths (e.g. from file discovery); at execution time the workflow passes those and run directories to stage agents as before.

4. **Execution after approval**  
   - Once the user approves, the executive calls **workflow_agent(message, plan_path=..., session_id=...)** with the approved **plan_path** and current **session_id** (from the prompt). The wrapper links the plan to the session and injects the path so the workflow runs the same plan.  
   - The workflow uses the approved plan and **answers** (and any preferences from persistent/session memory) to drive preprocessing → docking → MD. All user interaction is through the executive; stage agents do not request user input during execution.

### End-to-end plan flow (summary)

1. **New plan**: Executive calls **workflow_agent** in plan-only mode with the user’s exact message → workflow calls **create_plan_path**, sets **user_prompt** via **append_to_plan_section**, then calls preprocessing, docking, and MD agents with the **plan_path**; each agent **read_plan_document** and **append_to_plan_section** for its stage. Workflow returns **plan_path** to the executive.  
2. **Reuse**: Executive gets **plan_path** from **list_plans_by_tag(tag)** (after **get_all_plan_tags**), then **read_plan_document(plan_path)** to load and optionally adapt.  
3. **Review**: Executive calls **read_plan_document(plan_path)** and passes that content (plus user request and context) to **oversight_agent**.  
4. **User approval**: Executive presents **steps**, **parameters**, and **questions** from the plan; when the user answers, executive calls **append_to_plan_section(plan_path, "answers", {...})**. After approval, executive calls **workflow_agent(message, plan_path=..., session_id=...)** to execute with the approved plan (the wrapper links the plan to the session).  
5. **Execution**: Workflow runs preprocessing → docking → MD using the plan and **answers**. If something material is missing at execution time, stage agents surface it in their response so the executive can address it in the next turn.
