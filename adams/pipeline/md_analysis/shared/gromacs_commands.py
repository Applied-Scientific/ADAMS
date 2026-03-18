"""
Shared GROMACS execution helpers (free functions).
"""

import os
import shutil

from ....logger_utils import get_logger
from ....utils import run_cmd
from .constants import get_gromacs_friendly_ranks
from .grompp_warnings import run_grompp
from .index_ops import get_ndx_group_index


def build_mdrun_env(ntomp: int, env_overrides: dict = None) -> dict:
    """Build a stable environment for heavy GROMACS ``mdrun`` commands."""
    ntomp_val = max(1, int(ntomp))
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(ntomp_val)
    if env_overrides:
        env.update(env_overrides)
    return env


def run_mdrun(
    gmx_binary: str,
    deffnm: str,
    *,
    run_on_gpu: bool = False,
    binary_type: str = "standard",
    ntmpi: int = None,
    ntomp: int = 1,
    max_mpi: int = 32,
    mdrun_env_overrides: dict = None,
    cwd: str = None,
):
    """Execute ``gmx mdrun`` with GPU / MPI / CPU dispatch."""
    ntomp_val = max(1, int(ntomp))
    mdrun_env = build_mdrun_env(ntomp_val, env_overrides=mdrun_env_overrides)
    use_mpi = (
        (binary_type == "mpi")
        and (not run_on_gpu)
        and (ntmpi is not None and ntmpi > 1)
    )

    if run_on_gpu or binary_type == "cuda":
        ntmpi_val = ntmpi if (ntmpi and ntmpi > 0) else 1
        cuda_visible = mdrun_env.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible:
            visible_gpu_count = len(
                [gpu.strip() for gpu in cuda_visible.split(",") if gpu.strip()]
            )
            if ntmpi_val > visible_gpu_count:
                get_logger().info(
                    "Clamping ntmpi from %d to %d based on CUDA_VISIBLE_DEVICES=%s",
                    ntmpi_val, visible_gpu_count, cuda_visible,
                )
                ntmpi_val = visible_gpu_count
        cmd = [
            gmx_binary, "mdrun",
            "-ntmpi", str(ntmpi_val),
            "-ntomp", str(ntomp_val),
            "-deffnm", deffnm,
        ]
    elif use_mpi:
        cmd = ["mpirun"]
        if ntmpi is not None and ntmpi >= max_mpi:
            cmd.append("--oversubscribe")
        cmd.extend([
            "-np", str(ntmpi),
            gmx_binary, "mdrun",
            "-ntomp", str(ntomp_val),
            "-deffnm", deffnm,
        ])
    else:
        cmd = [gmx_binary, "mdrun", "-ntomp", str(ntomp_val), "-deffnm", deffnm]

    run_cmd(cmd, check=True, env=mdrun_env, cwd=cwd)


def validate_required_file_paths(file_paths: dict, required_keys: list, context: str = ""):
    """Validate that required keys exist and are non-empty in file_paths."""
    missing = [key for key in required_keys if not file_paths.get(key)]
    if missing:
        context_msg = f"{context}\n" if context else ""
        raise ValueError(
            f"{context_msg}"
            f"Required paths missing from file_paths: {missing}\n"
            f"Available keys: {list(file_paths.keys())}"
        )


def copy_mdp_files(mdp_dir: str, mdp_files: list, destination_dir: str):
    """Copy required MDP files into a destination directory.

    MDP paths are deterministic (package layout under mdp/); no existence checks.
    """
    for mdp_file in mdp_files:
        src = os.path.join(mdp_dir, mdp_file)
        shutil.copy(src, destination_dir)


def run_md_stage(
    gmx_binary: str,
    stage_name: str,
    mdp_file: str,
    input_gro: str,
    index_file: str,
    topology_file: str,
    *,
    run_on_gpu: bool = False,
    binary_type: str = "standard",
    ntmpi: int = None,
    ntomp: int = 1,
    max_mpi: int = 32,
    warning_policy=None,
    logger=None,
    mdrun_env_overrides: dict = None,
    cwd: str = None,
):
    """Execute one MD stage: grompp preparation followed by mdrun."""
    grompp_cmd = [
        gmx_binary, "grompp",
        "-f", mdp_file,
        "-c", input_gro,
        "-r", input_gro,
        "-n", index_file,
        "-p", topology_file,
        "-o", f"{stage_name}.tpr",
        "-po", f"{stage_name}_mdout.mdp",
    ]
    run_grompp(grompp_cmd, warning_policy=warning_policy, cwd=cwd)

    snapped_ntmpi = ntmpi
    if snapped_ntmpi is not None and snapped_ntmpi > 1:
        friendly_ntmpi = get_gromacs_friendly_ranks(snapped_ntmpi)
        if friendly_ntmpi != snapped_ntmpi and logger is not None:
            logger.info(
                "Adjusted MPI ranks from %d to %d for optimal GROMACS domain decomposition",
                snapped_ntmpi, friendly_ntmpi,
            )
        snapped_ntmpi = friendly_ntmpi

    run_mdrun(
        gmx_binary, stage_name,
        run_on_gpu=run_on_gpu,
        binary_type=binary_type,
        ntmpi=snapped_ntmpi,
        ntomp=ntomp,
        max_mpi=max_mpi,
        mdrun_env_overrides=mdrun_env_overrides,
        cwd=cwd,
    )


def pbc_reimage_trajectory(
    gmx_binary: str,
    tpr: str,
    xtc: str,
    ndx: str,
    output_xtc: str,
    center_group: str = "Protein",
    output_group: str = "System",
):
    """Apply PBC re-imaging and centering to produce an analysis-ready trajectory."""
    logger = get_logger()
    try:
        center_idx = get_ndx_group_index(ndx, center_group)
    except ValueError:
        logger.warning(
            "Could not find group '%s' in %s for PBC centering. Falling back to group index 1.",
            center_group, ndx,
        )
        center_idx = 1
    try:
        output_idx = get_ndx_group_index(ndx, output_group)
    except ValueError:
        logger.warning(
            "Could not find group '%s' in %s for PBC output. Falling back to group index 0.",
            output_group, ndx,
        )
        output_idx = 0

    intermediate_xtc = output_xtc.replace(".xtc", "_nopbc_tmp.xtc")
    logger.info("PBC post-processing 1/2: making molecules whole (-pbc mol)")
    trjconv_mol_cmd = [
        gmx_binary, "trjconv",
        "-s", tpr, "-f", xtc, "-n", ndx,
        "-pbc", "mol", "-o", intermediate_xtc,
    ]
    run_cmd(trjconv_mol_cmd, input_str=f"{output_idx}\n", check=True)
    logger.info("PBC post-processing 2/2: centering on group '%s' (idx %d)", center_group, center_idx)
    trjconv_center_cmd = [
        gmx_binary, "trjconv",
        "-s", tpr, "-f", intermediate_xtc, "-n", ndx,
        "-center", "-pbc", "mol", "-o", output_xtc,
    ]
    run_cmd(trjconv_center_cmd, input_str=f"{center_idx}\n{output_idx}\n", check=True)
    if os.path.exists(intermediate_xtc):
        os.remove(intermediate_xtc)
    logger.info("PBC-corrected trajectory written to %s", output_xtc)
    return output_xtc


def resolve_backbone_group(ndx_path):
    """Find the Backbone (preferred) or Protein index group. Returns 0-based index or None."""
    logger = get_logger()
    try:
        return get_ndx_group_index(ndx_path, "Backbone")
    except ValueError:
        pass
    try:
        return get_ndx_group_index(ndx_path, "Protein")
    except ValueError:
        pass
    logger.warning("Could not find Backbone or Protein group in index %s.", ndx_path)
    return None
