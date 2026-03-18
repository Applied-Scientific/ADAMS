from __future__ import annotations

import argparse

from .pipeline.docking.protocolized_docking import run_standard_docking_job


def add_dock_subparser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        'dock',
        help='Run a full production docking job without interactive chat',
        description=(
            'Prepare receptor and ligands, then run production docking in one command. '
            'Supports a ligand CSV/SMILES file or a folder of .smi files.'
        ),
    )
    ligand_group = parser.add_mutually_exclusive_group(required=True)
    ligand_group.add_argument('--ligand-input', help='Ligand file understood by ADAMS preprocessing')
    ligand_group.add_argument('--ligand-folder', help='Folder of .smi/.smiles files (one or more ligands per file)')

    parser.add_argument('--receptor', required=True, help='Receptor structure path (PDB/mmCIF)')
    parser.add_argument('--outdir', required=True, help='Output directory for this docking job')
    parser.add_argument(
        '--docking-centers',
        required=True,
        help='Comma-separated production centers: x,y,z or x1,y1,z1,x2,y2,z2,...',
    )
    parser.add_argument('--backend', default='vina_gpu', choices=['vina', 'vina_gpu', 'unidock'])
    parser.add_argument('--chain-to-keep', default='all')
    parser.add_argument('--keep-heterogens', default='essential')
    parser.add_argument('--required-heterogens', default=None, help='Comma-separated heterogens that must remain after prep, e.g. PO4')
    parser.add_argument('--keep-water', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--pH', type=float, default=7.4)
    parser.add_argument('--warning-strict', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--id-col', default='ID')
    parser.add_argument('--smiles-col', default='SMILES')
    parser.add_argument('--molwt-upper-bound', type=float, default=700.0)
    parser.add_argument('--molwt-lower-bound', type=float, default=0.0)
    parser.add_argument('--enumerate-microstates', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num-confs', type=int, default=8)
    parser.add_argument('--max-confs-to-keep', type=int, default=2)
    parser.add_argument('--conformer-energy-window-kcal', type=float, default=3.0)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--charge-model', default='gasteiger')
    parser.add_argument('--num-pockets', type=int, default=1)
    parser.add_argument('--num-poses', type=int, default=20)
    parser.add_argument('--production-gridsize', type=float, default=20.0)
    parser.add_argument('--lock-grid-center', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num-gpus', type=int, default=1)
    parser.add_argument('--gpu-ids', default=None, help='Comma-separated GPU IDs, e.g. 0 or 0,1')
    parser.add_argument('--num-cores', type=int, default=None, help='CPU cores for backend=vina')
    return parser


def run_dock_command(args) -> None:
    run_standard_docking_job(
        receptor=args.receptor,
        outdir=args.outdir,
        docking_centers=args.docking_centers,
        ligand_input=args.ligand_input,
        ligand_folder=args.ligand_folder,
        backend=args.backend,
        chain_to_keep=args.chain_to_keep,
        keep_heterogens=args.keep_heterogens,
        required_heterogens=args.required_heterogens,
        keep_water=args.keep_water,
        pH=args.pH,
        warning_strict=args.warning_strict,
        id_col=args.id_col,
        smiles_col=args.smiles_col,
        molwt_upper_bound=args.molwt_upper_bound,
        molwt_lower_bound=args.molwt_lower_bound,
        enumerate_microstates=args.enumerate_microstates,
        num_confs=args.num_confs,
        max_confs_to_keep=args.max_confs_to_keep,
        conformer_energy_window_kcal=args.conformer_energy_window_kcal,
        random_seed=args.random_seed,
        charge_model=args.charge_model,
        num_pockets=args.num_pockets,
        num_poses=args.num_poses,
        production_gridsize=args.production_gridsize,
        lock_grid_center=args.lock_grid_center,
        num_gpus=args.num_gpus,
        gpu_ids=args.gpu_ids,
        num_cores=args.num_cores,
    )
