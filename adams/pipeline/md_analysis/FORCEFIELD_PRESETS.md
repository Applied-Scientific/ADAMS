# Amber Force Field Presets (MD Analysis)

The pipeline supports named Amber-style force field presets for **protein + water + ligand defaults**. Preset defaults for ligands are **GAFF2 + AM1-BCC**; these can still be overridden in `run_lig_prepare` via `atom_type` and `charge_type`. See [Amber Force Fields](https://ambermd.org/AmberModels.php) and [Protein Force Fields](https://ambermd.org/AmberModels_proteins.php).

## Conda environment (GROMACS 2024.4)

All users use the project conda environment (installed via `./scripts/install.sh`).

- **After install:** **`ff99sb_ildn_tip3p`** is in standard GROMACS. The install script (Step 5.5) adds **Amber14SB** and **a99SB-disp**, so **`amber14sb_tip3p`** and **`a99sb_disp`** are available after a normal install.
- **Not provided by script:** The presets `ff19sb_opc` and `charmm36m` are not installed; their `.ff` or water models are not in the standard build. Use another preset or install manually (see below).
- **Raw force field:** You can also pass `forcefield` and `water_model` directly (no preset). Valid water models in conda: `tip3p`, `tip4p`, `tip5p`, `spc`, `spce`, `tips3p`, `none`, `select`. Valid `.ff` in conda: `amber99sb-ildn`, `amber99sb`, `amber94`, `amber96`, `amber99`, `amberGS`, `charmm27`, `gromos*`, `oplsaa`.

## Presets

| Preset | Protein | Water | Ligands | Notes |
|--------|--------|--------|--------|--------|
| `ff19sb_opc` | ff19SB | OPC | **GAFF2 / AM1-BCC** (preset default; configurable) | No ready GROMACS port; see intbio 14SB+OPC as alternative. |
| `ff14sb_tip3p` | ff14SB (`amber14sb`) | TIP3P | **GAFF2 / AM1-BCC** (preset default; configurable) | Alias for `amber14sb_tip3p`; **added by ADAMS install script** |
| `amber14sb_tip3p` | Amber14SB | TIP3P | **GAFF2 / AM1-BCC** (preset default; configurable) | **Added by ADAMS install script** |
| `ff99sb_ildn_tip3p` | ff99SB-ILDN | TIP3P | **GAFF2 / AM1-BCC** (preset default; configurable) | **Included in standard GROMACS** (conda). |
| `a99sb_disp` | a99SB-disp | a99SBdisp_water (TIP4P-D variant) | **GAFF2 / AM1-BCC** (preset default; configurable) | **Added by ADAMS install script** |
| `charmm36m` | CHARMM36m | TIP3P | **GAFF2 / AM1-BCC** (preset default; configurable) | Manual: charmm2gmx or MacKerell GROMACS port. |

For your minimum baseline, use one of these full stacks:
- `ff19sb_opc` + `atom_type="gaff2"` + `charge_type="bcc"`
- `ff14sb_tip3p` (or `amber14sb_tip3p`) + `atom_type="gaff2"` + `charge_type="bcc"`
- `ff99sb_ildn_tip3p` + `atom_type="gaff2"` + `charge_type="bcc"`

## Usage

- **Agent / API**: Call `run_protein_topology(file_paths, forcefield_preset="ff99sb_ildn_tip3p")` (or any preset name). Omit `forcefield_preset` to use raw `forcefield` and `water_model` instead.
- **Ligands**: `run_lig_prepare(..., charge_type="bcc", atom_type="gaff2")` (default) gives AM1-BCC and GAFF2; use `atom_type="gaff"` for GAFF.

## Installing extra force fields (conda)

The ADAMS install script (`./scripts/install.sh`) installs **Amber14SB** and **a99SB-disp** in Step 5.5 (into GROMACS `share/gromacs/top`). If that step was skipped or failed, run the install script again; it will skip completed steps and retry Step 5.5.

**Not installed by the script:**

- **CHARMM36m**: Convert from CHARMM with [charmm2gmx](https://awacha.gitlab.io/charmm2gmx/) or use a pre-built GROMACS port (e.g. [MacKerell lab](https://mackerell.umaryland.edu/charmm_ff.shtml)), then place a `charmm36m.ff` directory in the same GROMACS top directory.
- **ff19SB + OPC**: No single ready-made GROMACS package; [intbio/gromacs_ff](https://github.com/intbio/gromacs_ff) has Amber14SB+OPC variants (e.g. `amber14sb_parmbsc1_opc_lmi.ff`) if you want OPC with 14SB instead of 19SB.
