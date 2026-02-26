# ADAMS: Agent-Driven Autonomous Molecular Simulations

https://github.com/user-attachments/assets/0a92b767-21b2-4e42-9dd1-53801a984627

ADAMS is an autonomous orchestration framework designed for computational chemists, biophysicists, and structural biologists. It streamlines the transition from raw structural data to high-confidence lead candidates by integrating structural refinement and global site discovery into a single agentic pipeline. With safety and security in mind, all inputs, intermediates, and results stay on your machine. Run logs, agent metadata/memories and other ADAMS produced data live under agent_data; nothing is sent to external servers. You provide and control your own LLM API key; ADAMS does not collect or store credentials.

## Core Scientific Capabilities

ADAMS automates the decision-heavy segments of the drug discovery workflow, allowing researchers to focus on hypothesis generation rather than file format plumbing.

### 1. Structural Refinement & pKa-Aware Preparation

- **Receptor Cleaning**: Automated handling of multi-chain structures, water molecules, and essential heterogens.
- **Precision Protonation**: Leverages **PDB2PQR** and **PROPKA** to assign pKa-dependent protonation states, ensuring realistic electrostatic environments for docking.
- **Missing Atom Recovery**: Intelligent restoration of missing side-chains and loop regions.

### 2. Global Site Discovery (Blind Docking)

- **Pocket Mapping**: Surface-wide grid generation for identifying novel allosteric or orthosteric binding sites without prior pocket knowledge.
- **Cluster Analysis**: Autonomous identification of high-affinity centers through density-based clustering of search-mode poses.

### 3. Production-Grade Pose Generation

- **Targeted Refinement**: High-exhaustiveness docking at identified binding centers using flexible-box sampling.
- **Multi-Backend Support**: Seamlessly switch between CPU and GPU-accelerated engines (AutoDock Vina, Vina-GPU 2.1, Uni-Dock) based on library size and hardware availability.

### 4. Dynamic Stability Assessment *(coming soon)*

Molecular dynamics–based stability validation and trajectory analysis in a future release.

---

## Why ADAMS?

Traditional computational workflows often suffer from "human-in-the-loop" bottlenecks and error-prone manual transitions. ADAMS replaces these with:

- **Natural Language Orchestration**: Interact with your simulation environment using scientific intent (e.g., _"Find the most stable binding site for these ligands on kinase X"_).
- **Autonomous Decision Making**: Intelligent agents determine optimal grid spacing, cluster parameters, and pose selection.
- **End-to-End Traceability**: Every decision, tool execution, and file transition is logged, providing a complete audit trail for your research.

---

## Quick Start

### Installation

**Prerequisites:** Conda or Mamba and your favorite LLM provider's API key.

**Option 1:**

```bash
# Set up your environment (recommended)
curl -fsSL https://raw.githubusercontent.com/Applied-Scientific/ADAMS/main/scripts/install.sh | bash
```

**Option 2:**

```bash
# Clone the repo, then install ADAMS
git clone https://github.com/Applied-Scientific/ADAMS.git
cd ADAMS
bash scripts/install.sh
```

For detailed setup instructions, GPU configuration, and troubleshooting, see [TMI.md](TMI.md).

### Run Your Analysis

Start the interactive biophysics controller:

```bash
adams
```

The agents will guide you through pocket discovery and production docking.

---

## Privacy & Security

- **Local data**: All inputs, intermediates, and results stay on your machine. Run logs, agent metadata/memories and other ADAMS produced data live under `agent_data`; nothing is sent to external servers.
- **Your API key**: You provide and control your own LLM API key; ADAMS does not collect or store credentials.

---

## Documentation

- **[TMI.md](TMI.md)** - Detailed technical documentation on installation, file organization, and pipeline architecture.
- **[docs/AGENTS_DOCUMENTATION.md](docs/AGENTS_DOCUMENTATION.md)** - Deep dive into agent roles (Controller, Preprocessing, Docking).
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
