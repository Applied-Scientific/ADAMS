# Release Notes

---

## v2.0 — 2026-03-17

> Major release. All changes are relative to v1.0.

### Planning & Workflow

- **Structured plan-and-approve flow** — ADAMS assembles a reviewable plan (parameters, questions, notes) before any run starts. Users approve, approve with notes, or reject with feedback. Plans can also be cloned from prior runs, so repeat or similar experiments skip re-questioning entirely.
- **Smart questioning** — Only questions that genuinely require user input are surfaced; anything inferable from files, settings, or memory is resolved automatically.
- **Plan management** — Plans are auto-tagged and searchable. Prior plans can be cloned for new runs, preserving scientific settings (pH, pocket strategy, etc.) while clearing run-specific paths.
- **Memory system** — ADAMS maintains persistent memory across sessions: user preferences (GPU, working directory, custom instructions) are stored and applied automatically on future runs, and past session outcomes, plans, and results are recalled to inform new ones.
- **Session continuity** — Plans are linked to sessions. Users can resume a previous session by ID and restore conversation history.

### Model Support

Updated model roster (GPT-5 removed):

| Provider | Models |
|---|---|
| OpenAI | GPT-5.2, GPT-5.2 Pro, GPT-5.2 Codex, GPT-5.3 Codex, **GPT-5.4** *(new default)*, GPT-5.4 Pro |
| Anthropic | Claude Haiku 4.5, Sonnet 4.5, Opus 4.5, Sonnet 4.6, Opus 4.6 |
| Google | Gemini 2.5 Flash Lite/Flash/Pro, Gemini 3.1 Flash Lite Preview, Gemini 3 Pro Preview |

### UI

- **Logs panel** — Sidebar screen with two-pane layout: file list (newest-first) and color-coded log content (errors red, warnings yellow, info blue).
- **Cancel button** — Stops the active task mid-run and posts a notice in chat.
- **Token usage** — Input/output/total token counts displayed beneath each reply.
- **Live activity** — Tool calls and stage transitions shown in real time instead of a generic "thinking" state.
- **Clean shutdown** — Ctrl+C and SIGTERM save session/run state before exit.

### Reliability & Performance

- **Faster responses** — Reference material is loaded on-demand; prompts streamlined.
- **No mid-run prompts** — All user decisions are collected during planning; approved runs complete without interruption.
- **Structured failure reports** — On error: output folder, completed steps, failed step, error detail, and resume guidance are all reported together.
- **Error diagnosis** — A dedicated post-failure step analyzes logs and suggests how to resume or fix the issue.
- **File discovery** — ADAMS classifies files in the working directory to suggest entry points and locate intermediates from prior runs.
- **Structured result parsing** — Docking affinities, pose counts, and MD metrics are extracted and used for decisions and summaries.

### Docking — Interface & Tooling

- **Unified docking entry point** — Single tool replaces separate CPU/GPU tools; backend selected via parameter (`vina-cpu`, `vina-gpu`, `unidock`).
- **Pluggable backend architecture** — Preprocessing and pose collection are shared; new engines can be added without changing the user-facing interface.
- **Path safety** — All file access is sandboxed to the user's project root.
- **Structured run logs** — Per-session/step timing logs power the live UI status display.
- **Multiprocessing fix** — Resolved spawn errors when docking in parallel chunks.

### Scientific & Pipeline

**Protein Preparation**
- Cofactors, metal ions, and prosthetic groups (HEM, FAD, NAD, ATP, etc.) are preserved by default rather than stripped; chain selection, residue range filtering, and structural water retention are all supported.
- Gap detection via sequence analysis + distance-based checks; TER records inserted at chain breaks to prevent artificial bonds.
- pH-aware protonation via PDB2PQR + PROPKA (default pH 7.4) with full pKa-based titration state prediction.

**Ligand State Enumeration**
- Microstate enumeration: tautomers (RDKit, energy-based pruning), protonation states (pH 6.4–8.4), and stereoisomers for unspecified chiral centers — all with per-parameter caps and sensible defaults.
- Conformer generation with MMFF94s/UFF force-field optimization and energy-window filtering.
- Multi-format input: SMILES, CSV, SDF, MOL2, PDB, PDBQT — auto-detected and routed through the appropriate preparation path.

**Docking**
- Multi-backend: AutoDock Vina (CPU), Vina-GPU (NVIDIA), UniDock (Linux); automatic hardware-based selection with manual override.
- Supports both user-defined binding site coordinates and automated pocket discovery (search docking + DBSCAN clustering).
- Results ranked with per-pocket best poses and configurable top-N summaries.

**Molecular Dynamics**
- Full protein-ligand MD pipeline: protein topology generation, ligand parameterization (GAFF2 + AM1-BCC charges), complex assembly, solvation, and ion addition.
- NVT temperature equilibration → 4-stage NPT pressure equilibration with progressive restraint release; V-rescale thermostat, Parrinello-Rahman barostat, PME electrostatics, dispersion correction for AMBER force fields.
- Stability analysis: protein backbone RMSD, ligand RMSD, per-residue RMSF; ranked summary report merged with docking affinity scores.

---
