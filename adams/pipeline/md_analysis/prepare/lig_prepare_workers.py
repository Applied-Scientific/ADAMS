"""
LigPrepare worker entry points for parallel execution.

These functions are picklable and are invoked by ParallelExecutor / run_phases_or_raise.
They orchestrate ACPYPE, pose system setup, and energy minimization per pose.
"""

import os
import pandas as pd

from ....logger_utils import get_logger
from ....utils.parallel_executor import TaskResult
from ..shared import LIGAND_RESNAME
from .acpype_runner import run_acpype
from .ligand_ops import (
    formal_charge,
    formal_charge_from_smiles,
    get_smiles_by_id,
    restore_from_pdbqt,
    restore_from_pdbqt_structure_only,
    restore_from_pdbqt_with_smiles,
)
from .pose_system_setup import run_energy_minimization, setup_pose_until_min


def acpype_and_setup_single_pose(task):
    """Run PDBQT restore, ACPYPE, and system setup (no energy minimization) for one pose.

    Used by phase 1 of the two-phase LigPrepare flow. Leaves the pose ready for
    run_min_only in phase 2.
    """
    logger = get_logger()
    pose = task["pose"]
    poses_dir = task.get("poses_dir")
    smiles_file = task["smiles_file"]
    pose_smiles_lookup = task.get("pose_smiles_lookup", {})

    out_dir = task.get("out_dir")
    if not out_dir:
        ligand_name, _, grid_rank = pose.partition("_pocket_")
        grid_id, _, rank = grid_rank.partition("_top")
        if poses_dir is None:
            raise ValueError("Task is missing both out_dir and poses_dir.")
        out_dir = os.path.join(poses_dir, f"{ligand_name}_pocket_{grid_id}_top{rank}")
    pose_pdbqt = task.get("pose_pdbqt") or os.path.join(out_dir, "ligand.pdbqt")
    pose_mol2 = os.path.join(out_dir, "ligand.mol2")

    lookup_val = task.get("smiles_lookup_value")
    is_direct_smiles = bool(task.get("smiles_is_direct", False))
    if lookup_val is None:
        fallback_ligand_name = task.get("manifest", {}).get("ID")
        if fallback_ligand_name is None:
            fallback_ligand_name = pose.split("_pocket_")[0]
        lookup_val, is_direct_smiles = pose_smiles_lookup.get(
            pose, (fallback_ligand_name, False)
        )
    net_charge_override = None

    if is_direct_smiles and lookup_val:
        restore_from_pdbqt_with_smiles(lookup_val, pose_pdbqt, pose_mol2)
        try:
            net_charge_override = formal_charge_from_smiles(lookup_val)
        except Exception:
            pass
    else:
        query_id = lookup_val
        if smiles_file:
            try:
                restore_from_pdbqt(smiles_file, query_id, pose_pdbqt, pose_mol2)
                smi = get_smiles_by_id(smiles_file, query_id)
                if smi is not None and not (isinstance(smi, float) and pd.isna(smi)):
                    try:
                        net_charge_override = formal_charge_from_smiles(smi)
                    except Exception:
                        pass
            except Exception as restore_err:
                logger.warning(
                    "SMILES template restore failed for %s (%s), "
                    "using structure-only restore.",
                    pose,
                    restore_err,
                )
                restore_from_pdbqt_structure_only(pose_pdbqt, pose_mol2)
                try:
                    net_charge_override = formal_charge(pose_mol2)
                except Exception:
                    pass
        else:
            restore_from_pdbqt_structure_only(pose_pdbqt, pose_mol2)
            try:
                net_charge_override = formal_charge(pose_mol2)
            except Exception:
                pass

    run_acpype(
        pose_mol2,
        LIGAND_RESNAME,
        charge_type=task["charge_type"],
        atom_type=task["atom_type"],
        net_charge_override=net_charge_override,
        retry_with_gas_on_failure=task["retry_with_gas_on_failure"],
    )

    task["out_dir"] = out_dir
    setup_pose_until_min(task, run_min=False)

    return TaskResult(task_id=pose, success=True)


def run_min_only(task):
    """Run energy minimization for one pose and return TaskResult (for run_phases_or_raise)."""
    run_energy_minimization(task)
    return TaskResult(task_id=task["pose"], success=True)


def prepare_single_pose(task):
    """Prepare one pose end-to-end: PDBQT restore, ACPYPE, solvation, minimization.

    Legacy single-phase worker. Accepts a single task dict, returns a TaskResult.
    """
    logger = get_logger()
    pose = task["pose"]
    poses_dir = task.get("poses_dir")
    smiles_file = task["smiles_file"]
    pose_smiles_lookup = task.get("pose_smiles_lookup", {})

    out_dir = task.get("out_dir")
    if not out_dir:
        ligand_name, _, grid_rank = pose.partition("_pocket_")
        grid_id, _, rank = grid_rank.partition("_top")
        if poses_dir is None:
            raise ValueError("Task is missing both out_dir and poses_dir.")
        out_dir = os.path.join(poses_dir, f"{ligand_name}_pocket_{grid_id}_top{rank}")
    pose_pdbqt = task.get("pose_pdbqt") or os.path.join(out_dir, "ligand.pdbqt")
    pose_mol2 = os.path.join(out_dir, "ligand.mol2")

    lookup_val = task.get("smiles_lookup_value")
    is_direct_smiles = bool(task.get("smiles_is_direct", False))
    if lookup_val is None:
        fallback_ligand_name = task.get("manifest", {}).get("ID")
        if fallback_ligand_name is None:
            fallback_ligand_name = pose.split("_pocket_")[0]
        lookup_val, is_direct_smiles = pose_smiles_lookup.get(
            pose, (fallback_ligand_name, False)
        )
    net_charge_override = None

    if is_direct_smiles and lookup_val:
        restore_from_pdbqt_with_smiles(lookup_val, pose_pdbqt, pose_mol2)
        try:
            net_charge_override = formal_charge_from_smiles(lookup_val)
        except Exception:
            pass
    else:
        query_id = lookup_val
        if smiles_file:
            try:
                restore_from_pdbqt(smiles_file, query_id, pose_pdbqt, pose_mol2)
                smi = get_smiles_by_id(smiles_file, query_id)
                if smi is not None and not (isinstance(smi, float) and pd.isna(smi)):
                    try:
                        net_charge_override = formal_charge_from_smiles(smi)
                    except Exception:
                        pass
            except Exception as restore_err:
                logger.warning(
                    "SMILES template restore failed for %s (%s), "
                    "using structure-only restore.",
                    pose,
                    restore_err,
                )
                restore_from_pdbqt_structure_only(pose_pdbqt, pose_mol2)
                try:
                    net_charge_override = formal_charge(pose_mol2)
                except Exception:
                    pass
        else:
            restore_from_pdbqt_structure_only(pose_pdbqt, pose_mol2)
            try:
                net_charge_override = formal_charge(pose_mol2)
            except Exception:
                pass

    run_acpype(
        pose_mol2,
        LIGAND_RESNAME,
        charge_type=task["charge_type"],
        atom_type=task["atom_type"],
        net_charge_override=net_charge_override,
        retry_with_gas_on_failure=task["retry_with_gas_on_failure"],
    )

    task["out_dir"] = out_dir
    setup_pose_until_min(task, run_min=True)

    return TaskResult(task_id=pose, success=True)
