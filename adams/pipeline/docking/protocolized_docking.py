from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from ...logger_utils import setup_logger
from ...path_config import set_agent_data_path
from ...utils.console_transcript import start_console_transcript


def split_csv_values(raw: Optional[str | Iterable[str]]) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return [item.strip() for item in str(raw).split(',') if item.strip()]


def parse_docking_centers(raw: str | Iterable[float]) -> list[float]:
    if isinstance(raw, (list, tuple)):
        values = [float(item) for item in raw]
    else:
        values = [float(item.strip()) for item in str(raw).split(',') if item.strip()]
    if not values or len(values) % 3 != 0:
        raise ValueError(
            "docking_centers must contain 3*N floats, e.g. '1,2,3'"
        )
    return values


def parse_gpu_ids(raw: Optional[str | Iterable[int]]) -> Optional[list[int]]:
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        return [int(value) for value in raw]
    values = split_csv_values(raw)
    if not values:
        return None
    return [int(value) for value in values]


def normalize_keep_heterogens(raw: Optional[str | Iterable[str]]):
    if raw is None:
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return None
        lower = text.lower()
        if lower == 'none':
            return None
        if lower == 'essential':
            return 'essential'
    values = split_csv_values(raw)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def build_smiles_csv_from_folder(folder: str | Path, output_csv: str | Path) -> Path:
    folder_path = Path(folder).expanduser().resolve()
    output_path = Path(output_csv).expanduser().resolve()
    if not folder_path.is_dir():
        raise FileNotFoundError(f'Ligand folder not found: {folder_path}')

    files = sorted(list(folder_path.glob('*.smi')) + list(folder_path.glob('*.smiles')))
    if not files:
        raise FileNotFoundError(f'No .smi or .smiles files found in: {folder_path}')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str]] = []
    for file_path in files:
        line_index = 0
        for raw_line in file_path.read_text(encoding='utf-8').splitlines():
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            smiles = parts[0]
            ligand_id = parts[1] if len(parts) > 1 else file_path.stem
            if line_index > 0 and len(parts) == 1:
                ligand_id = f'{file_path.stem}__{line_index + 1}'
            rows.append((ligand_id, smiles))
            line_index += 1

    if not rows:
        raise ValueError(f'No ligand rows found in folder: {folder_path}')

    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['ID', 'SMILES'])
        writer.writerows(rows)
    return output_path


def count_heterogens(pdb_path: str | Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with Path(pdb_path).open(encoding='utf-8', errors='replace') as handle:
        for line in handle:
            if line.startswith('HETATM') and len(line) >= 20:
                counts[line[17:20].strip().upper()] += 1
    return counts


def verify_required_heterogens(pdb_path: str | Path, required: Iterable[str]) -> Counter[str]:
    required_names = [name.strip().upper() for name in required if str(name).strip()]
    if not required_names:
        return Counter()
    counts = count_heterogens(pdb_path)
    missing = [name for name in required_names if counts.get(name, 0) == 0]
    if missing:
        raise RuntimeError(
            f"Required heterogens missing from {pdb_path}: {', '.join(missing)}. "
            f"Observed heterogens: {dict(counts)}"
        )
    return counts


def write_score_only_csv(ranked_csv: Path, production_csv: Path) -> Optional[Path]:
    rows: list[tuple[str, float]] = []

    if ranked_csv.exists():
        with ranked_csv.open(encoding='utf-8', errors='replace') as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            name_key = next((key for key in ('LigandName', 'Parent_ID', 'ID') if key in fieldnames), None)
            if name_key and 'affinity' in fieldnames:
                for row in reader:
                    try:
                        rows.append((row[name_key], float(row['affinity'])))
                    except (KeyError, TypeError, ValueError):
                        continue
    elif production_csv.exists():
        best_by_name: dict[str, float] = {}
        with production_csv.open(encoding='utf-8', errors='replace') as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames or []
            name_key = next((key for key in ('Parent_ID', 'LigandName', 'ID') if key in fieldnames), None)
            if name_key and 'affinity' in fieldnames:
                for row in reader:
                    try:
                        affinity = float(row['affinity'])
                    except (KeyError, TypeError, ValueError):
                        continue
                    name = row.get(name_key)
                    if not name:
                        continue
                    current = best_by_name.get(name)
                    if current is None or affinity < current:
                        best_by_name[name] = affinity
        rows = sorted(best_by_name.items(), key=lambda item: item[1])

    if not rows:
        return None

    output_path = ranked_csv.parent / 'ligand_score_only.csv'
    with output_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['LigandName', 'affinity'])
        writer.writerows(rows)
    return output_path


def run_standard_docking_job(
    receptor: str,
    outdir: str,
    docking_centers: str | Iterable[float],
    ligand_input: Optional[str] = None,
    ligand_folder: Optional[str] = None,
    backend: str = 'vina_gpu',
    chain_to_keep: str = 'all',
    residue_range_start: Optional[int] = None,
    residue_range_end: Optional[int] = None,
    keep_heterogens: Optional[str | Iterable[str]] = 'essential',
    required_heterogens: Optional[Iterable[str] | str] = None,
    keep_water: bool = False,
    pH: float = 7.4,
    warning_strict: bool = False,
    id_col: str = 'ID',
    smiles_col: str = 'SMILES',
    molwt_upper_bound: float = 700.0,
    molwt_lower_bound: float = 0.0,
    enumerate_microstates: bool = True,
    num_confs: int = 8,
    max_confs_to_keep: int = 2,
    conformer_energy_window_kcal: float = 3.0,
    random_seed: int = 42,
    charge_model: str = 'gasteiger',
    num_pockets: int = 1,
    num_poses: int = 20,
    production_gridsize: float = 20.0,
    lock_grid_center: bool = True,
    num_gpus: Optional[int] = 1,
    gpu_ids: Optional[Iterable[int] | str] = None,
    num_cores: Optional[int] = None,
) -> dict:
    from ..data_preprocessing.clean_pdb import CleanPDB
    from ..data_preprocessing.conformer_generation import generate_conformers_to_pdbqt
    from ..data_preprocessing.ligand_preprocessing import LigandPreprocessor
    from ..data_preprocessing.protonation import run_pdb2pqr
    from ..data_preprocessing.standardize_ligands import (
        convert_3d_to_pdbqt,
        detect_ligand_format,
        standardize_2d_to_csv,
    )
    from ..file_organization import setup_preprocessing_dirs
    from .docking import DockingPipeline

    if bool(ligand_input) == bool(ligand_folder):
        raise ValueError('Provide exactly one of ligand_input or ligand_folder')

    outdir_path = Path(outdir).expanduser().resolve()
    outdir_path.mkdir(parents=True, exist_ok=True)

    agent_data_path = set_agent_data_path(path=outdir_path / 'agent_data')
    transcript_path = start_console_transcript(agent_data_path / 'logs' / 'console_transcript.log')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    pipeline_log = agent_data_path / 'logs' / f'adams_pipeline_dock_{timestamp}.log'
    setup_logger(log_file=str(pipeline_log))

    print(f'[Transcript] {transcript_path}')
    print(f'[Pipeline Log] {pipeline_log}')
    print(f'[Outdir] {outdir_path}')

    if ligand_folder:
        ligand_csv = build_smiles_csv_from_folder(
            ligand_folder,
            outdir_path / 'inputs' / f'ligands_from_folder_{timestamp}.csv',
        )
        ligand_input_path = ligand_csv
        print(f'[Ligands] Built CSV from folder: {ligand_input_path}')
    else:
        ligand_input_path = Path(ligand_input).expanduser().resolve()
        if not ligand_input_path.exists():
            raise FileNotFoundError(f'Ligand input not found: {ligand_input_path}')
        print(f'[Ligands] Using input file: {ligand_input_path}')

    receptor_input = Path(receptor).expanduser().resolve()
    if not receptor_input.exists():
        raise FileNotFoundError(f'Receptor not found: {receptor_input}')
    print(f'[Receptor] {receptor_input}')

    format_info = detect_ligand_format(str(ligand_input_path))
    if format_info['has_3d']:
        mapping_csv = Path(
            convert_3d_to_pdbqt(str(ligand_input_path), str(outdir_path), charge_model=charge_model)
        ).resolve()
        standardized_csv = ligand_input_path
        ligand_csv_for_conformers = ligand_input_path
        print(f'[Ligands] 3D input detected; mapping CSV: {mapping_csv}')
    else:
        standardized_csv = Path(
            standardize_2d_to_csv(
                str(ligand_input_path),
                str(outdir_path),
                id_col=id_col,
                smiles_col=smiles_col,
            )
        ).resolve()
        print(f'[Ligands] Standardized CSV: {standardized_csv}')

        ligand_csv_for_conformers = standardized_csv
        if enumerate_microstates:
            prep_outputs = LigandPreprocessor(
                input_data=str(standardized_csv),
                molwt_upper_bound=molwt_upper_bound,
                molwt_lower_bound=molwt_lower_bound,
                outpath=str(outdir_path),
                enumerate_microstates=True,
                enumerate_tautomers=True,
                enumerate_protonation=True,
                enumerate_stereoisomers=True,
            ).run()
            ligand_csv_for_conformers = Path(prep_outputs['small_mw']).resolve()
            print(f'[Ligands] Microstate-enumerated CSV: {ligand_csv_for_conformers}')

        mapping_csv = Path(
            generate_conformers_to_pdbqt(
                str(ligand_csv_for_conformers),
                str(outdir_path),
                num_confs=num_confs,
                max_confs_to_keep=max_confs_to_keep,
                conformer_energy_window_kcal=conformer_energy_window_kcal,
                random_seed=random_seed,
                charge_model=charge_model,
            )
        ).resolve()
        print(f'[Ligands] Docking-ready mapping CSV: {mapping_csv}')

    normalized_keep_heterogens = normalize_keep_heterogens(keep_heterogens)
    required_heterogen_names = split_csv_values(required_heterogens)

    cleaned_pdb = Path(
        CleanPDB(
            input_pdb=str(receptor_input),
            outpath=str(outdir_path),
            ligand=False,
            chain_to_keep=chain_to_keep,
            residue_range_start=residue_range_start,
            residue_range_end=residue_range_end,
            keep_water=keep_water,
            keep_heterogens=normalized_keep_heterogens,
            pH=pH,
        ).clean()
    ).resolve()
    print(f'[Receptor] Cleaned PDB: {cleaned_pdb}')
    cleaned_heterogen_counts = Counter()
    if required_heterogen_names:
        cleaned_heterogen_counts = verify_required_heterogens(cleaned_pdb, required_heterogen_names)
        print(f'[Receptor] Required heterogens retained after cleaning: {dict(cleaned_heterogen_counts)}')

    dir_structure = setup_preprocessing_dirs(str(outdir_path))
    input_prefix = cleaned_pdb.stem[:-6] if cleaned_pdb.stem.endswith('_clean') else cleaned_pdb.stem
    output_pqr = Path(dir_structure['receptors']) / f'{input_prefix}_protonated.pqr'
    output_pdb = Path(dir_structure['receptors']) / f'{input_prefix}_protonated.pdb'

    try:
        protonated_pdb, _, warnings_csv, warning_summary = run_pdb2pqr(
            input_pdb=str(cleaned_pdb),
            output_pqr=str(output_pqr),
            output_pdb=str(output_pdb),
            pH=pH,
            ff='AMBER',
            ffout='AMBER',
            warning_strict=warning_strict,
        )
    except TypeError as exc:
        if 'warning_strict' not in str(exc):
            raise
        protonated_pdb, _, warnings_csv, warning_summary = run_pdb2pqr(
            input_pdb=str(cleaned_pdb),
            output_pqr=str(output_pqr),
            output_pdb=str(output_pdb),
            pH=pH,
            ff='AMBER',
            ffout='AMBER',
        )
    protonated_pdb = Path(protonated_pdb).resolve()
    print(f'[Receptor] Protonated PDB: {protonated_pdb}')
    if warnings_csv:
        print(f'[Receptor] Protonation warnings CSV: {warnings_csv}')
    if warning_summary:
        print(f'[Receptor] Protonation warning summary: {warning_summary}')

    protonated_heterogen_counts = Counter()
    if required_heterogen_names:
        protonated_heterogen_counts = verify_required_heterogens(protonated_pdb, required_heterogen_names)
        print(f'[Receptor] Required heterogens retained after protonation: {dict(protonated_heterogen_counts)}')

    centers = parse_docking_centers(docking_centers)
    parsed_gpu_ids = parse_gpu_ids(gpu_ids)
    resolved_num_gpus = num_gpus
    if parsed_gpu_ids is not None:
        resolved_num_gpus = len(parsed_gpu_ids)
    if backend in {'vina_gpu', 'unidock'}:
        if parsed_gpu_ids is None:
            parsed_gpu_ids = list(range(resolved_num_gpus)) if resolved_num_gpus is not None else [0]
        if resolved_num_gpus is None:
            resolved_num_gpus = len(parsed_gpu_ids)
        print(f'[Docking] GPU backend={backend}, num_gpus={resolved_num_gpus}, gpu_ids={parsed_gpu_ids}')
    else:
        print(f'[Docking] CPU backend={backend}, num_cores={num_cores}')

    pipeline_kwargs = {
        'input_data': str(mapping_csv),
        'receptor': str(protonated_pdb),
        'mode': 'production',
        'num_pockets': num_pockets,
        'num_poses': num_poses,
        'docking_centers': centers,
        'production_gridsize': production_gridsize,
        'lock_grid_center': lock_grid_center,
        'out_folder': str(outdir_path),
        'pH': pH,
        'charge_model': charge_model,
        'num_cores': num_cores,
        'num_gpus': resolved_num_gpus,
        'gpu_ids': parsed_gpu_ids,
    }
    production_csv = Path(DockingPipeline(backend=backend, **pipeline_kwargs).run()).resolve()

    ranked_csv = outdir_path / 'docking' / 'production' / 'summaries' / 'production_best_entity_per_ligand_ranked.csv'
    failed_csv = outdir_path / 'docking' / 'production' / 'summaries' / 'failed_combinations.csv'
    score_only_csv = write_score_only_csv(ranked_csv, production_csv)

    print('[Done] Outputs:')
    print(str(mapping_csv))
    print(str(protonated_pdb))
    print(str(production_csv))
    if ranked_csv.exists():
        print(str(ranked_csv.resolve()))
    if failed_csv.exists():
        print(str(failed_csv.resolve()))
    if score_only_csv is not None:
        print(str(score_only_csv.resolve()))

    return {
        'outdir': str(outdir_path),
        'agent_data_path': str(agent_data_path),
        'console_transcript': str(transcript_path),
        'pipeline_log': str(pipeline_log),
        'ligand_input_used': str(ligand_input_path),
        'standardized_ligands_csv': str(standardized_csv),
        'ligand_csv_for_conformers': str(ligand_csv_for_conformers),
        'docking_ready_ligands_csv': str(mapping_csv),
        'cleaned_pdb': str(cleaned_pdb),
        'protonated_pdb': str(protonated_pdb),
        'production_docking_results_csv': str(production_csv),
        'production_best_entity_per_ligand_ranked_csv': str(ranked_csv.resolve()) if ranked_csv.exists() else '',
        'failed_combinations_csv': str(failed_csv.resolve()) if failed_csv.exists() else '',
        'ligand_score_only_csv': str(score_only_csv.resolve()) if score_only_csv is not None else '',
        'required_heterogens_after_cleaning': dict(cleaned_heterogen_counts),
        'required_heterogens_after_protonation': dict(protonated_heterogen_counts),
        'warnings_csv': str(warnings_csv) if warnings_csv else '',
        'warning_summary': str(warning_summary) if warning_summary else '',
    }
