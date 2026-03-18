"""
MD analysis agent tools.

Agent-facing execution is unified through workflow-aware tools:
    - run_build_membrane_system_openmm
    - run_md_prepare
    - run_md_simulation
    - run_md_analysis

Each accepts workflow="auto|soluble|membrane". In auto mode, membrane keys in
file_paths trigger membrane routing; otherwise soluble routing is used.
"""

import os
from pathlib import Path
from typing import List, Optional, TypedDict

from agents import Agent, function_tool

from ...helper_agents.file_parser.file_parser_agent import get_file_parser_agent
from ...model_config import get_current_model_name, get_resolved_model
from ...user_plan_utils import (
    append_to_plan_section,
    contribute_stage_to_plan,
    read_plan_document,
)
from ..references.reference_file_reader import read_reference_file
from .agent_utils import build_file_paths, discover_paths
from .prepare.lig_prepare import LigPrepare
from .prepare.membrane_prep import MembranePrep
from .prepare.openmm_membrane_builder import OpenMMMembraneBuilder
from .prepare.protein_topology import ProteinTopology
from .simulate.soluble_md import SolubleMd
from .simulate.membrane_md import MembraneMd
from .analyze.stability_analysis import StabilityAnalysis
from .analyze.membrane_analysis import MembraneAnalysis
from .shared import (
    GromppWarningPolicy,
    get_approved_grompp_warnings_path,
    load_approved_grompp_warnings,
)


def _make_grompp_policy(approved_grompp_warnings: list = None) -> GromppWarningPolicy:
    """Create a ``GromppWarningPolicy`` from agent-supplied fingerprints."""
    return GromppWarningPolicy(
        approved=set(approved_grompp_warnings or []),
        descriptions={},
    )


def _attach_policy_report(result: dict, policy: GromppWarningPolicy) -> dict:
    """If any pre-approved warnings were used, record them in *result*."""
    if policy._approved:
        result["pre_approved_grompp_warnings_used"] = (
            policy.get_approved_with_descriptions()
        )
    return result


def _detect_workflow(file_paths: "FilePathsDict", workflow: str = "auto") -> str:
    """
    Resolve workflow mode for unified agent tools.

    Rules:
      - explicit workflow ("soluble" or "membrane") always wins
      - auto selects membrane when membrane-specific keys are present
      - otherwise defaults to soluble
    """
    normalized = (workflow or "auto").strip().lower()
    if normalized in {"soluble", "membrane"}:
        return normalized
    if normalized != "auto":
        raise ValueError(
            f"Invalid workflow '{workflow}'. Expected one of: auto, soluble, membrane."
        )

    membrane_markers = (
        "membrane_build",
        "membrane_system_gro",
        "membrane_system_top",
        "membrane_dir",
        "membrane_min_gro",
        "membrane_top",
        "membrane_ndx",
        "membrane_md_xtc",
    )
    if any(file_paths.get(key) for key in membrane_markers):
        return "membrane"
    return "soluble"


def _resolve_forcefield_preset(forcefield_preset: Optional[str]) -> Optional[str]:
    """
    Normalize forcefield_preset and map "auto" to a single default preset.

    Explicit presets are passed through unchanged.
    """
    normalized = (
        forcefield_preset.strip().lower()
        if isinstance(forcefield_preset, str) and forcefield_preset.strip()
        else None
    )
    if normalized == "auto":
        return "ff99sb_ildn_tip3p"
    return normalized


def _format_stored_grompp_approvals_for_prompt() -> str:
    """
    Build a system-prompt section listing historically approved grompp warnings
    from agent_data/memory/approved_grompp_warnings.json. The agent uses this
    to decide what to pass in approved_grompp_warnings each call (adaptive to
    current user intent; user may revoke any approval).
    """
    path = get_approved_grompp_warnings_path()
    loaded = load_approved_grompp_warnings(path) if path else {}
    fps = loaded.get("approved_fingerprints", set())
    desc = loaded.get("descriptions", {})
    if not fps:
        return (
            "**STORED GROMPP WARNING APPROVALS (from previous sessions):**\n"
            "- None stored. When the user approves warnings, they are saved to "
            "agent_data/memory/approved_grompp_warnings.json and will appear here on the next run.\n"
        )
    lines = [
        "**STORED GROMPP WARNING APPROVALS (from previous sessions):**",
        "The following warnings were previously approved and are stored in "
        "agent_data/memory/approved_grompp_warnings.json. Pass only the fingerprints "
        "that match the user's *current* intent in approved_grompp_warnings when calling "
        "run_md_prepare or run_md_simulation; the user may revoke approval for any warning.",
        "",
    ]
    for fp in sorted(fps):
        d = desc.get(fp, "(no description)")
        lines.append(f"- Fingerprint: {fp!r}")
        lines.append(f"  Description: {d}")
        lines.append("")
    return "\n".join(lines)


class FilePathsDict(TypedDict, total=False):
    """Type definition for file_paths dictionary used throughout MD pipeline."""

    md_root: str
    protein_dir: str
    poses_dir: str
    reports_dir: str
    protein_file: str
    protein_gro: str
    protein_top: str
    posre_itp: str
    water_model: str
    docking_csv: str
    ligand_input: str
    gromacs_path: str
    ambertools_path: str
    gromacs_binary_type: str
    summary_report: str
    brief_report: str
    workflow_used: str
    pose_manifest: str
    prepared_poses: List[str]
    prepared_pose_count: int
    failed_pose_count: int
    dropped_parent_ids: List[str]
    lig_prepare_summary_path: str
    lig_prepare_failures_path: str
    md_completed_poses: List[str]
    md_failed_poses: List[str]
    md_runtime_summary: str

    membrane_dir: str
    membrane_system_gro: str
    membrane_system_top: str
    membrane_system_gro_work: str
    membrane_min_gro: str
    membrane_top: str
    membrane_ndx: str
    membrane_posre: str
    membrane_posre_files: List[str]
    membrane_posre_variants: List[str]
    membrane_normalization_report: str
    membrane_build: bool
    force_rebuild: bool
    membrane_source: str
    membrane_build_report: str
    membrane_orientation_report: str
    orientation_policy: str
    membrane_lipid_type: str
    membrane_padding_nm: float
    membrane_ionic_strength_m: float
    membrane_positive_ion: str
    membrane_negative_ion: str
    membrane_md_tpr: str
    membrane_md_xtc: str
    membrane_md_gro: str
    membrane_reports_dir: str
    membrane_analysis_report: str
    membrane_rmsd_xvg: str
    membrane_density_xvg: str
    membrane_runtime_summary: str


@function_tool
def run_build_membrane_system_openmm(
    file_paths: FilePathsDict,
    lipid_type: str = "POPC",
    minimum_padding_nm: float = 2.0,
    ionic_strength_m: float = 0.15,
    positive_ion: str = "K+",
    negative_ion: str = "Cl-",
    orientation_policy: str = "warn",
    force_rebuild: bool = False,
) -> FilePathsDict:
    """
    Build membrane GRO/TOP from protein input using OpenMM.

    If valid pre-built membrane_system_gro + membrane_system_top are already
    provided and force_rebuild=False, pre-built files are kept.
    """
    builder = OpenMMMembraneBuilder(
        file_paths=file_paths,
        lipid_type=lipid_type,
        minimum_padding_nm=minimum_padding_nm,
        ionic_strength_m=ionic_strength_m,
        positive_ion=positive_ion,
        negative_ion=negative_ion,
        orientation_policy=orientation_policy,
    )
    return builder.run(force_rebuild=force_rebuild)


def run_md_prepare_impl(
    file_paths: FilePathsDict,
    workflow: str = "auto",
    forcefield: str = "amber99sb-ildn",
    water_model: Optional[str] = None,
    forcefield_preset: str = None,
    ignore_hydrogens: bool = True,
    tops: Optional[int] = 3,
    selection_scope: str = "per_grid",
    num_cores: int = 0,
    num_gpus: int = -1,
    max_jobs: int = 0,
    charge_type: str = "bcc",
    atom_type: str = "gaff2",
    retry_with_gas_on_failure: bool = False,
    water_margin: float = 1.0,
    ion_conc: float = 0.15,
    pname: str = "K",
    nname: str = "CL",
    approved_grompp_warnings: Optional[List[str]] = None,
) -> FilePathsDict:
    """
    Unified preparation interface for both workflows.

    - soluble: runs ProteinTopology (if needed) then LigPrepare
    - membrane: runs MembranePrep
    - selection_scope:
        - "per_grid" (default): keep top `tops` rows per grid/pocket
        - "per_parent_per_grid": keep top `tops` rows per parent ligand per grid/pocket
      Set `tops=None` or `tops<=0` to disable the cap.

    num_gpus: -1 = use all available GPUs in parallel; 0 = CPU-only; N = use N GPUs in parallel.
    max_jobs: maximum concurrent LigPrepare jobs (0 = auto).
    """
    requested_water_model = (
        water_model.strip().lower()
        if isinstance(water_model, str) and water_model.strip()
        else None
    )
    workflow_used = _detect_workflow(file_paths, workflow)
    normalized_forcefield_preset = _resolve_forcefield_preset(forcefield_preset)
    policy = _make_grompp_policy(approved_grompp_warnings)

    if workflow_used == "membrane":
        membrane_gro = file_paths.get("membrane_system_gro")
        membrane_top = file_paths.get("membrane_system_top")
        should_build = bool(file_paths.get("membrane_build"))
        force_rebuild = bool(file_paths.get("force_rebuild"))
        has_prebuilt = bool(
            membrane_gro
            and membrane_top
            and isinstance(membrane_gro, str)
            and isinstance(membrane_top, str)
            and os.path.exists(membrane_gro)
            and os.path.exists(membrane_top)
        )
        if (should_build and not has_prebuilt) or force_rebuild:
            builder = OpenMMMembraneBuilder(
                file_paths=file_paths,
                lipid_type=str(file_paths.get("membrane_lipid_type", "POPC")),
                minimum_padding_nm=float(file_paths.get("membrane_padding_nm", 2.0)),
                ionic_strength_m=float(file_paths.get("membrane_ionic_strength_m", 0.15)),
                positive_ion=str(file_paths.get("membrane_positive_ion", "K+")),
                negative_ion=str(file_paths.get("membrane_negative_ion", "Cl-")),
                orientation_policy=str(file_paths.get("orientation_policy", "warn")),
            )
            file_paths = builder.run(force_rebuild=force_rebuild)

        prep = MembranePrep(
            file_paths=file_paths,
            forcefield=forcefield,
            water_model=requested_water_model or "tip3p",
            forcefield_preset=normalized_forcefield_preset,
            ignore_hydrogens=ignore_hydrogens,
            water_margin=water_margin,
            ion_conc=ion_conc,
            pname=pname,
            nname=nname,
            grompp_warning_policy=policy,
        )
        result = prep.run()
    else:
        current = file_paths
        if not (current.get("protein_gro") and current.get("protein_top")):
            topology_water_model = (
                requested_water_model
                or (
                    current.get("water_model", "").strip().lower()
                    if isinstance(current.get("water_model"), str)
                    else None
                )
                or "tip3p"
            )
            topology = ProteinTopology(
                file_paths=current,
                forcefield=forcefield,
                water_model=topology_water_model,
                forcefield_preset=normalized_forcefield_preset,
                ignore_hydrogens=ignore_hydrogens,
            )
            current = topology.run()

        lig_prepare = LigPrepare(
            file_paths=current,
            tops=tops,
            selection_scope=selection_scope,
            num_cores=num_cores if num_cores > 0 else None,
            num_gpus=num_gpus,
            max_jobs=max_jobs,
            charge_type=charge_type,
            atom_type=atom_type,
            retry_with_gas_on_failure=retry_with_gas_on_failure,
            water_margin=water_margin,
            ion_conc=ion_conc,
            pname=pname,
            nname=nname,
            water_model=requested_water_model,
            grompp_warning_policy=policy,
        )
        result = lig_prepare.run()
        result["water_model"] = lig_prepare.water_model

    result["workflow_used"] = workflow_used
    return _attach_policy_report(result, policy)


@function_tool
def run_md_prepare(
    file_paths: FilePathsDict, workflow: str = "auto",
    forcefield: str = "amber99sb-ildn", water_model: Optional[str] = None,
    forcefield_preset: str = None, ignore_hydrogens: bool = True,
    tops: Optional[int] = 3, selection_scope: str = "per_grid",
    num_cores: int = 0, num_gpus: int = -1, max_jobs: int = 0,
    charge_type: str = "bcc", atom_type: str = "gaff2",
    retry_with_gas_on_failure: bool = False, water_margin: float = 1.0,
    ion_conc: float = 0.15, pname: str = "K", nname: str = "CL",
    approved_grompp_warnings: Optional[List[str]] = None,
) -> FilePathsDict:
    """Unified preparation interface for both workflows (agent-facing wrapper)."""
    return run_md_prepare_impl(**{k: v for k, v in locals().items()})


def run_md_simulation_impl(
    file_paths: FilePathsDict,
    workflow: str = "auto",
    gpu: bool = False,
    num_gpus: int = -1,
    mpi_ranks: int = 0,
    omp_threads: int = 0,
    max_jobs: int = 0,
    production_nsteps: Optional[int] = None,
    production_dt_fs: float = 2.0,
    soluble_eq_nsteps_scale: Optional[float] = None,
    membrane_prod_nsteps: Optional[int] = None,
    membrane_eq_nsteps_scale: Optional[float] = None,
    topol: str = "system.top",
    index: str = "index.ndx",
    approved_grompp_warnings: Optional[List[str]] = None,
) -> FilePathsDict:
    """Unified MD simulation interface for soluble and membrane workflows.

    num_gpus: -1 = use all available GPUs in parallel; 0 = CPU-only; N = use N GPUs (one job per GPU).
    production_dt_fs: soluble workflow production timestep target (default 2.0 fs; values >2.0 fs are advanced/non-default and auto-fallback to 2.0 fs on failure).
    production_nsteps: optional soluble workflow production nsteps override applied at runtime.
    soluble_eq_nsteps_scale: optional multiplier for soluble equilibration nsteps applied at runtime.
    membrane_prod_nsteps: optional membrane production nsteps override applied at runtime.
    membrane_eq_nsteps_scale: optional multiplier for membrane equilibration nsteps (for quick smoke tests).
    """
    workflow_used = _detect_workflow(file_paths, workflow)
    policy = _make_grompp_policy(approved_grompp_warnings)

    if workflow_used == "membrane":
        runner = MembraneMd(
            file_paths=file_paths,
            gpu=gpu,
            num_gpus=num_gpus,
            mpi_ranks=mpi_ranks,
            omp_threads=omp_threads,
            membrane_prod_nsteps=membrane_prod_nsteps,
            membrane_eq_nsteps_scale=membrane_eq_nsteps_scale,
            topol=topol,
            index=index,
            grompp_warning_policy=policy,
        )
    else:
        runner = SolubleMd(
            file_paths=file_paths,
            gpu=gpu,
            num_gpus=num_gpus,
            mpi_ranks=mpi_ranks,
            omp_threads=omp_threads,
            max_jobs=max_jobs,
            production_nsteps=production_nsteps,
            production_dt_fs=production_dt_fs,
            soluble_eq_nsteps_scale=soluble_eq_nsteps_scale,
            topol=topol,
            index=index,
            grompp_warning_policy=policy,
        )

    result = runner.run()
    result["workflow_used"] = workflow_used
    return _attach_policy_report(result, policy)


@function_tool
def run_md_simulation(
    file_paths: FilePathsDict, workflow: str = "auto",
    gpu: bool = False, num_gpus: int = -1, mpi_ranks: int = 0,
    omp_threads: int = 0, max_jobs: int = 0,
    production_nsteps: Optional[int] = None, production_dt_fs: float = 2.0,
    soluble_eq_nsteps_scale: Optional[float] = None,
    membrane_prod_nsteps: Optional[int] = None,
    membrane_eq_nsteps_scale: Optional[float] = None,
    topol: str = "system.top", index: str = "index.ndx",
    approved_grompp_warnings: Optional[List[str]] = None,
) -> FilePathsDict:
    """Unified MD simulation interface (agent-facing wrapper)."""
    return run_md_simulation_impl(**{k: v for k, v in locals().items()})


def run_md_analysis_impl(
    file_paths: FilePathsDict,
    workflow: str = "auto",
    prefix: str = "md",
    analysis_range: str = "all",
    last_frames: int = 100,
    vina_report: str = "",
) -> FilePathsDict:
    """Unified trajectory analysis interface for soluble and membrane workflows."""
    workflow_used = _detect_workflow(file_paths, workflow)
    if workflow_used == "membrane":
        analysis = MembraneAnalysis(
            file_paths=file_paths,
            prefix=prefix,
            analysis_range=analysis_range,
            last_frames=last_frames,
        )
    else:
        analysis = StabilityAnalysis(
            file_paths=file_paths,
            prefix=prefix,
            Range=analysis_range,
            last_frames=last_frames,
            vina_report=vina_report or None,
        )

    result = analysis.run()
    result["workflow_used"] = workflow_used
    return result


@function_tool
def run_md_analysis(
    file_paths: FilePathsDict, workflow: str = "auto",
    prefix: str = "md", analysis_range: str = "all",
    last_frames: int = 100, vina_report: str = "",
) -> FilePathsDict:
    """Unified trajectory analysis interface (agent-facing wrapper)."""
    return run_md_analysis_impl(**{k: v for k, v in locals().items()})


prompt_path = Path(__file__).parent / "md_agent_prompt.md"
system_prompt = (
    prompt_path.read_text()
    + "\n\n"
    + _format_stored_grompp_approvals_for_prompt()
)

_md_agent = None
_md_model = None


def get_md_agent() -> Agent:
    global _md_agent, _md_model
    current_model = get_current_model_name()
    if _md_agent is None or _md_model != current_model:
        _md_agent = Agent(
            model=get_resolved_model(),
            name="Stability MD Agent",
            tools=[
                read_reference_file,
                read_plan_document,
                append_to_plan_section,
                contribute_stage_to_plan,
                get_file_parser_agent().as_tool(
                    tool_name="file_parser_agent",
                    tool_description=(
                        "An agent that extracts structured statistics from MD results to check completion status. "
                        "Use this agent to parse MD results directories to check completion status, identify which poses "
                        "have completed MD simulations, and find analysis reports."
                    ),
                ),
                # Shared setup
                build_file_paths,
                discover_paths,
                # Unified MD workflow interface
                run_md_prepare,
                run_build_membrane_system_openmm,
                run_md_simulation,
                run_md_analysis,
            ],
            instructions=system_prompt,
        )
        _md_model = current_model
    return _md_agent
