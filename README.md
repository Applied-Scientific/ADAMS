# ADAMS: Agent-Driven Autonomous Molecular Simulations

https://github.com/user-attachments/assets/c1a7ebcd-be1a-41fc-a5c9-1396e6ca0974

ADAMS is an autonomous orchestration framework designed for computational chemists, biophysicists, and structural biologists. It streamlines the transition from raw structural data to high-confidence lead candidates by integrating structural refinement, global site discovery, and molecular dynamics-based stability validation into a single agentic pipeline.

## Core Scientific Capabilities

ADAMS automates the decision-heavy segments of the drug discovery workflow, allowing researchers to focus on hypothesis generation rather than file format plumbing.

### 1. Structural Refinement & pKa-Aware Preparation

- **Receptor Cleaning**: Automated handling of multi-chain structures, water molecules, and essential heterogens.
- **Precision Protonation**: Leverages **PDB2PQR** and **PROPKA** to assign pKa-dependent protonation states, ensuring realistic electrostatic environments for docking.
- **Missing Atom Recovery**: Intelligent restoration of missing side-chains and loop regions.
- **Ligand microstate enumeration**: Tautomers, protonation states, and stereoisomer enumeration. Conformer generation with energy optimization & filtering.

### 2. Global Site Discovery (Blind Docking)

- **Pocket Mapping**: Surface-wide grid generation for identifying novel allosteric or orthosteric binding sites without prior pocket knowledge.
- **Cluster Analysis**: Autonomous identification of high-affinity centers through density-based clustering of search-mode poses.

### 3. Production-Grade Pose Generation

- **Targeted Refinement**: High-exhaustiveness docking at identified binding centers using flexible-box sampling.
- **Multi-Backend Support**: Seamlessly switch between CPU and GPU-accelerated engines (AutoDock Vina, Vina-GPU 2.1, Uni-Dock) based on library size and hardware availability.

### 4. Dynamic Stability Assessment (MD Analysis)

- **Trajectory Generation**: Automated GROMACS setup including energy minimization, NVT/NPT equilibration, and production MD (10-100+ ns).
- **Binding Persistence**: Differentiates between transient docking artifacts and stable binders through RMSD, RMSF, and hydrogen-bond persistence analysis.
- **Automated Interpretation**: Agent-driven evaluation of trajectories to prioritize stable poses for experimental validation.

---

## Why ADAMS?

Traditional computational workflows often suffer from "human-in-the-loop" bottlenecks and error-prone manual transitions. ADAMS replaces these with:

- **Natural Language Orchestration**: Interact with your simulation environment using scientific intent (e.g., _"Find the most stable binding site for these ligands on kinase X"_).
- **Autonomous Decision Making**: Intelligent agents determine optimal grid spacing, cluster parameters, and MD candidate selection.
- **End-to-End Traceability**: Every decision, Tool execution, and file transition is logged, providing a complete audit trail for your research.

---

## Quick Start

### Installation

**Prerequisites:** Conda or Mamba package manager and an OpenAI API key.

```bash
# Set up your environment (recommended)
curl -fsSL https://raw.githubusercontent.com/Applied-Scientific/ADAMS/main/scripts/install.sh | bash
```

```bash
# Install ADAMS
bash scripts/install.sh
```

For detailed setup instructions, GPU configuration, and troubleshooting, see [TMI.md](TMI.md).

### API key

Set your API key before or when starting ADAMS:

- **Environment variable (recommended):** Export the key for your provider, e.g. `export OPENAI_API_KEY="your-key"` (or `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` for other models). ADAMS will use it automatically.
- **In the UI:** If no key is set, ADAMS will prompt you to enter it when you open the Agent chat. You can optionally save it to the system keychain for future runs.

See [TMI.md](TMI.md) for keychain storage and security notes.

### Run Your Analysis

Start the interactive TUI (default: new session):

```bash
adams
```

Use `adams -h` or `adams --help` to list all options. To resume a previous session by ID (e.g. `YYYYMMDD_HHMMSS` or `tui_...`):

```bash
adams --continue-session SESSION_ID
```

**ADAMS memory commands**:

- `adams instructions get` — print custom instructions
- `adams instructions set` / `append` — set or append custom instructions (text, `--file`, or stdin)
- `adams instructions clear` — clear custom instructions
- `adams preferences clear` — clear stored preferences (GPU, working directory, learned behaviors)

The agents will guide you through pocket discovery, production docking, and MD-based stability analysis.

---

## Documentation

- **[TMI.md](TMI.md)** - Detailed technical documentation on installation, file organization, and pipeline architecture.
- **[docs/AGENTS_DOCUMENTATION.md](docs/AGENTS_DOCUMENTATION.md)** - Deep dive into agent roles (Controller, Preprocessing, Docking, MD).
- **[docs/WORKFLOW_EXECUTION.md](docs/WORKFLOW_EXECUTION.md)** - Execution sequences and file path specifications.
- **[docs/RELEASE_NOTES.md](docs/RELEASE_NOTES.md)** - Version-level docking and preprocessing changes.

---

## Citation

If ADAMS assists in your research, please cite our work:

```bibtex
@software{adams2025,
  title = {ADAMS: Agent-Driven Autonomous Molecular Simulations},
  author = {Rhizome Research},
  year = {2025},
  url = {https://github.com/Applied-Scientific/ADAMS},
  license = {Apache-2.0}
}
```

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
