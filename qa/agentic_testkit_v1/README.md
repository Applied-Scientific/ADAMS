# Agentic Test Kit v1 (Docking + Soluble MD)

This test kit validates the ADAMS docking and soluble MD pipelines using BRD4 as the demo target.

## Included Scenario

- **BRD4 holo** (`protein_waters_apo_brd4.pdb` as docking receptor; BRD4 CIFs as redocking references)

## Running Prompts

Each prompt uses a TXT key=value contract from `configs_txt/` as its single source of configuration.
Feed the prompt text to the ADAMS agent (or any compatible agentic runner).

Available prompts:
- `prompts/12_brd4_holo_docking_redocking_txt.md` — docking validation (no MD)
- `prompts/22_brd4_holo_md_short_txt.md` — short MD smoke test (continues from docking outputs)

Contract used:
- `configs_txt/brd4_holo_3mxf.txt`

Prerequisites:
- Active ADAMS env (`conda activate adams`)
- `OPENAI_API_KEY` exported (for agentic runs)
- Protonation binaries (`pdb2pqr` / `pdb2pqr30`, `propka3` / `propka`)
- OpenBabel CLI (`obabel`)
- GROMACS (`gmx`) for MD runs
- Force field `ff99sb_ildn_tip3p` (included in standard GROMACS)

## Runtime Defaults (Current)

Docking:
- `backend=vina_gpu`
- `mode=production`
- `num_poses=100`
- `production_gridsize=20.0`

Short MD smoke:
- `md.tops=1`
- `md.production_nsteps=50000` (100 ps at 2 fs)
- `md.production_dt_fs=2.0`
- `md.last_frames=50`
