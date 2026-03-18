"""
Pose-level system setup: solvation, ions, index, restraints, and energy minimization.

Used by LigPrepare (soluble workflow). Each function operates on one pose directory
and uses task dicts for serializable context (safe for parallel workers).
"""

import os
import shutil

from ....logger_utils import get_logger
from ....utils import run_cmd
from .ligand_ops import add_ligand_topology_with_atomtypes, combine_gro
from ..shared import (
    GromacsContext,
    LIGAND_RESNAME,
    LIG_HEAVY_GROUP_NAME,
    clean_gro_file,
    detect_solvent_group_for_genion,
    detect_ligand_resname_from_gro,
    get_mdp_dir,
    get_ndx_group_index,
    make_system_index,
    rename_ndx_last_group,
    resolve_protein_topology_assets,
    run_grompp,
    collect_posre_includes,
    generate_staged_restraint_variants,
    patch_topology_staged_restraints,
    validate_ionic_strength_after_genion,
    validate_system_neutrality_after_genion,
    write_bulk_solvent_index,
)
from ..shared.constants import RESTRAINT_FC_VALUES


def setup_pose_until_min(task, run_min: bool = True):
    """Set up solvation, ions, restraints, and optionally energy minimization for one pose.

    When run_min=False, stops after copying min.mdp (no mdrun). Used by the
    two-phase flow so phase 1 (CPU) does setup only and phase 2 (GPU) runs all EMs.

    All subprocess calls use absolute paths or ``cwd=`` so no ``os.chdir()``
    is needed (safe for parallel workers).
    """
    logger = get_logger()
    pose_name = task["out_dir"]
    ctx = GromacsContext.from_task_params(task["ctx"])
    gmx_machine = ctx.gmx_binary
    file_paths = task["file_paths"]
    grompp_warning_policy = ctx.grompp_warning_policy
    water_margin = task["water_margin"]
    water_model = task["water_model"]
    solvent_box = task["solvent_box"]
    ion_conc = task["ion_conc"]
    pname = task["pname"]
    nname = task["nname"]
    gromacs_binary_type = ctx.binary_type
    num_cores = task["num_cores"]
    num_gpus = task["num_gpus"]
    root_path = task["root_path"]

    protein_gro = file_paths["protein_gro"]
    combine_gro(
        protein_gro,
        f"{pose_name}/{LIGAND_RESNAME}.acpype/{LIGAND_RESNAME}_GMX.gro",
        f"{pose_name}/complex.gro",
    )

    protein_top = file_paths["protein_top"]
    add_ligand_topology_with_atomtypes(
        protein_top,
        f"{pose_name}/{LIGAND_RESNAME}.acpype/{LIGAND_RESNAME}_GMX.itp",
        f"{pose_name}/system.top",
        ligand_name=LIGAND_RESNAME,
    )

    topology_assets = resolve_protein_topology_assets(
        protein_top=protein_top,
        explicit_posre=file_paths.get("posre_itp"),
        protein_dir=file_paths.get("protein_dir"),
        root_path=root_path,
        logger=logger,
    )

    protein_top_dir = os.path.dirname(os.path.abspath(protein_top))
    for include_src_abs in topology_assets["local_include_files"]:
        rel_include = os.path.relpath(include_src_abs, protein_top_dir)
        if rel_include.startswith(".."):
            continue
        include_dst = os.path.join(pose_name, rel_include)
        os.makedirs(os.path.dirname(include_dst), exist_ok=True)
        shutil.copy2(include_src_abs, include_dst)

    editconf_cmd = [
        gmx_machine, "editconf",
        "-f", f"{pose_name}/complex.gro",
        "-o", f"{pose_name}/newbox.gro",
        "-bt", "cubic",
        "-d", str(water_margin),
    ]
    run_cmd(editconf_cmd, check=True)

    solvate_cmd = [
        gmx_machine, "solvate",
        "-cp", f"{pose_name}/newbox.gro",
        "-cs", solvent_box,
        "-p", f"{pose_name}/system.top",
        "-o", f"{pose_name}/solv.gro",
    ]
    logger.debug("Solvating with water model %s (box: %s)", water_model, solvent_box)
    run_cmd(solvate_cmd, check=True)

    mdp_dir = get_mdp_dir()
    ions_mdp_path = os.path.join(mdp_dir, "ions.mdp")

    existing_posre_paths = topology_assets["posre_files"]
    checked_posre_paths = topology_assets["checked_posre_paths"]

    if not existing_posre_paths:
        raise FileNotFoundError(
            "No required protein restraint ITP file was found.\n"
            "Searched for POSRES includes referenced by protein_top and fallback posre.itp paths.\n"
            f"Checked paths:\n  - " + "\n  - ".join(checked_posre_paths)
        )

    shutil.copy(ions_mdp_path, pose_name)
    for posre_path in existing_posre_paths:
        shutil.copy(posre_path, pose_name)

    grompp_cmd = [
        gmx_machine, "grompp",
        "-f", f"{pose_name}/ions.mdp",
        "-c", f"{pose_name}/solv.gro",
        "-p", f"{pose_name}/system.top",
        "-o", f"{pose_name}/ions.tpr",
        "-po", f"{pose_name}/mdout.mdp",
    ]
    run_grompp(grompp_cmd, warning_policy=grompp_warning_policy)

    solvent_group_name = detect_solvent_group_for_genion(f"{pose_name}/system.top")
    genion_index_path = f"{pose_name}/genion_bulk_solvent.ndx"
    solvent_start_atom, solvent_end_atom = write_bulk_solvent_index(
        f"{pose_name}/newbox.gro",
        f"{pose_name}/solv.gro",
        genion_index_path,
        group_name=solvent_group_name,
    )
    logger.info(
        "Prepared bulk-solvent genion group '%s' for %s: atoms %d-%d from appended solvent block.",
        solvent_group_name,
        pose_name,
        solvent_start_atom,
        solvent_end_atom,
    )
    genion_cmd = [
        gmx_machine, "genion",
        "-s", f"{pose_name}/ions.tpr",
        "-o", f"{pose_name}/solv_ions.gro",
        "-n", genion_index_path,
        "-p", f"{pose_name}/system.top",
        "-pname", pname,
        "-nname", nname,
        "-conc", str(ion_conc),
        "-neutral",
    ]
    run_cmd(genion_cmd, input_str="0\n", check=True)

    validate_system_neutrality_after_genion(gmx_machine, mdp_dir, pose_name)
    validate_ionic_strength_after_genion(
        system_top_path=f"{pose_name}/system.top",
        pname=pname, nname=nname,
        target_conc=ion_conc, logger=logger,
    )

    ligand_gro_path = f"{pose_name}/{LIGAND_RESNAME}.acpype/{LIGAND_RESNAME}_GMX.gro"
    ligand_resname_for_index = detect_ligand_resname_from_gro(ligand_gro_path)

    make_ndx_lig_cmd = [
        gmx_machine, "make_ndx",
        "-f", f"{pose_name}/{LIGAND_RESNAME}.acpype/{LIGAND_RESNAME}_GMX.gro",
        "-o", f"{pose_name}/index_{LIGAND_RESNAME}.ndx",
    ]
    run_cmd(
        make_ndx_lig_cmd,
        input_str=f"r {ligand_resname_for_index} & ! a H*\nq\n",
        check=True,
    )
    index_lig_path = f"{pose_name}/index_{LIGAND_RESNAME}.ndx"
    rename_ndx_last_group(index_lig_path, LIG_HEAVY_GROUP_NAME)
    lig_heavy_grp = get_ndx_group_index(index_lig_path, LIG_HEAVY_GROUP_NAME)

    make_system_index(
        gmx_machine,
        f"{pose_name}/solv_ions.gro",
        f"{pose_name}/index.ndx",
        ligand_resname=ligand_resname_for_index,
        top_path=f"{pose_name}/system.top",
        ligand_gro_path=ligand_gro_path,
    )

    genrestr_cmd = [
        gmx_machine, "genrestr",
        "-f", f"{pose_name}/{LIGAND_RESNAME}.acpype/{LIGAND_RESNAME}_GMX.gro",
        "-n", index_lig_path,
        "-o", f"{pose_name}/posre_{LIGAND_RESNAME}.itp",
        "-fc", "1000", "1000", "1000",
    ]
    run_cmd(genrestr_cmd, input_str=f"{lig_heavy_grp}\n", check=True)
    posre_lig_path = f"{pose_name}/posre_{LIGAND_RESNAME}.itp"
    if not os.path.exists(posre_lig_path):
        acpype_posre = f"{pose_name}/{LIGAND_RESNAME}.acpype/posre_{LIGAND_RESNAME}.itp"
        if os.path.exists(acpype_posre):
            shutil.copy2(acpype_posre, posre_lig_path)
            logger.warning(
                "genrestr did not produce %s; recovered from ACPYPE output.",
                os.path.basename(posre_lig_path),
            )
        else:
            raise FileNotFoundError(
                f"Ligand restraint file not found after genrestr: {posre_lig_path}. "
                f"Also not found in ACPYPE output: {acpype_posre}."
            )

    # Prepare staged restraint variants for gradual POSRES release in equilibration.
    system_top_path = f"{pose_name}/system.top"
    patch_topology_staged_restraints(system_top_path, strict=False)
    staged_posre_files = collect_posre_includes(system_top_path, strict=False)
    if not staged_posre_files:
        raise FileNotFoundError(
            "No position restraint includes were found after topology patching. "
            "Cannot generate staged restraint variants for equilibration."
        )
    for base_posre in staged_posre_files:
        if not os.path.exists(base_posre):
            raise FileNotFoundError(
                f"Expected restraint file referenced by topology was not found: {base_posre}"
            )
    generate_staged_restraint_variants(staged_posre_files, fc_values=RESTRAINT_FC_VALUES)

    # Energy minimization — use cwd= instead of os.chdir()
    clean_gro_file(pose_name)

    min_mdp_path = os.path.join(mdp_dir, "min.mdp")
    shutil.copy(min_mdp_path, pose_name)

    if not run_min:
        return

    # Use context with worker-appropriate ntmpi/ntomp for min stage
    if gromacs_binary_type == "cuda" and num_gpus > 0:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible_gpu_count = (
            len([gpu.strip() for gpu in cuda_visible.split(",") if gpu.strip()])
            if cuda_visible
            else num_gpus
        )
        ntmpi = max(1, min(num_gpus, visible_gpu_count))
        ntomp = max(1, num_cores // ntmpi)
        ctx_min = ctx.with_overrides(ntmpi=ntmpi, ntomp=ntomp)
    else:
        ctx_min = ctx
    ctx_min.run_stage("min", "min.mdp", "solv_ions.gro", "index.ndx", "system.top", cwd=pose_name)
    logger.info("Energy minimization completed (pose %s)", pose_name)


def run_energy_minimization(task):
    """Run energy minimization for one pose already set up (solv_ions.gro, system.top, min.mdp in place).

    Used by phase 2 of the two-phase LigPrepare flow. Expects CUDA_VISIBLE_DEVICES
    to be set by ParallelExecutor when using GPUs.
    """
    logger = get_logger()
    pose_name = task["out_dir"]
    ctx = GromacsContext.from_task_params(task["ctx"])
    num_cores = task["num_cores"]
    num_gpus = task["num_gpus"]
    gromacs_binary_type = ctx.binary_type

    if gromacs_binary_type == "cuda" and num_gpus > 0:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        visible_gpu_count = (
            len([gpu.strip() for gpu in cuda_visible.split(",") if gpu.strip()])
            if cuda_visible
            else num_gpus
        )
        ntmpi = max(1, min(num_gpus, visible_gpu_count))
        ntomp = max(1, num_cores // ntmpi)
        ctx_min = ctx.with_overrides(ntmpi=ntmpi, ntomp=ntomp)
    else:
        ctx_min = ctx
    ctx_min.run_stage("min", "min.mdp", "solv_ions.gro", "index.ndx", "system.top", cwd=pose_name)
    logger.info("Energy minimization completed (pose %s)", pose_name)
