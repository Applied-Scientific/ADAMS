"""
Workflow agent wrapper: single place for session-plan linking and mode injection.

The wrapper accepts an optional agent-passed session_id but always falls back to
the current run context session id (set in setup_tracing) if linking with the
passed id fails. This keeps linking compatible across both fresh sessions and
continued sessions, even if an upstream caller provides a stale placeholder.

Modes:
- plan_path omitted -> PLANNING mode (wrapper creates a plan, injects planning prompt)
- plan_path provided -> EXECUTION mode (inject execution prompt, follow approved plan)
"""

import os
from pathlib import Path
from typing import Optional

from agents import Runner, function_tool

from ..memory.session_memory import add_session_plan_path
from ..path_config import get_current_session_id
from ..user_plan_utils import _create_plan_path_impl
from .workflow_agent import get_workflow_agent

_prompt_dir = Path(__file__).parent
_planning_injection = (_prompt_dir / "workflow_planning_prompt.md").read_text()
_execution_injection = (_prompt_dir / "workflow_execution_prompt.md").read_text()
_fill_params_injection = (_prompt_dir / "workflow_fill_params_prompt.md").read_text()


def _link_plan_with_fallback(preferred_session_id: Optional[str], plan_path: str) -> Optional[str]:
    """
    Link plan_path to session memory, preferring explicit session_id but safely
    falling back to the current run session id when needed.

    Returns the session id that succeeded, or None if no link was created.
    """
    current_session_id = get_current_session_id()

    if preferred_session_id:
        if add_session_plan_path(preferred_session_id, plan_path):
            return preferred_session_id
        # Fallback for stale/wrong passed IDs (e.g., UI-generated placeholders).
        if current_session_id and current_session_id != preferred_session_id:
            if add_session_plan_path(current_session_id, plan_path):
                return current_session_id
        return None

    if current_session_id:
        if add_session_plan_path(current_session_id, plan_path):
            return current_session_id
    return None


def _truthy_env(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_routing_policy_injection() -> str:
    """Return internal-only routing policy instructions for the workflow agent."""
    if _truthy_env("ADAMS_WORKFLOW_STRICT_PROTOCOL"):
        return (
            "[INTERNAL ROUTING POLICY]\n"
            "- Developer strict protocol mode is active.\n"
            "- For ordinary known-center production docking, you must use "
            "run_standard_docking_job_tool first.\n"
            "- If that standard protocol fails, stop and report the failure clearly.\n"
            "- Do not retry by switching to the flexible preprocessing_agent -> "
            "docking_agent path.\n"
            "- Search docking and clearly custom workflows may still use the flexible path.\n"
        )
    return (
        "[INTERNAL ROUTING POLICY]\n"
        "- Default v2.0 routing is active.\n"
        "- For ordinary known-center production docking, prefer "
        "run_standard_docking_job_tool first.\n"
        "- If that standard protocol fails for a technical execution reason, you may "
        "retry once through the flexible preprocessing_agent -> docking_agent path.\n"
        "- If you use that flexible fallback, report it explicitly and keep the same "
        "output folder.\n"
        "- Do not use the flexible fallback for search docking, clearly custom "
        "scientific workflows, or user-intent changes.\n"
    )


def _build_unattended_qa_injection(message: str) -> str:
    """Return extra guardrails for authoritative unattended QA prompts."""
    if "UNATTENDED QA EXECUTION POLICY (AUTHORITATIVE)" not in message and (
        "UNATTENDED CONTRACT OVERRIDE (AUTHORITATIVE)" not in message
    ):
        return ""
    return (
        "[INTERNAL UNATTENDED QA POLICY]\n"
        "- This is a fresh isolated unattended QA scenario.\n"
        "- Do not search for, reuse, clone, or adapt plans from other scenarios.\n"
        "- If a plan is needed, keep it isolated to this scenario only.\n"
        "- Preserve authoritative out_folder and md_workdir paths exactly as provided.\n"
        "- Do not replace those paths with agent_data/outputs/run_* bookkeeping directories.\n"
        "- When writing parameters, use only the stage keys preprocessing, docking, and md.\n"
        "- Do not invent extra parameter sections such as runner_qa.\n"
        "- Do not introduce user-approval pauses or follow-up questions for these unattended prompts.\n"
    )


def _make_workflow_agent_wrapper() -> callable:
    """
    Build the workflow agent wrapper tool. Injects mode-specific prompt fragment
    and plan path: PLANNING when plan_path is omitted, EXECUTION when plan_path
    is provided.
    """

    @function_tool
    def workflow_agent(
        message: str,
        plan_path: Optional[str] = None,
        session_id: Optional[str] = None,
        fill_only: Optional[bool] = None,
    ) -> str:
        """
        Coordinate the complete molecular docking workflow. Use for full pipelines
        (preprocessing -> docking -> MD) or individual steps. You can specify
        use_gpu=True or use_gpu=False in the message.

        **Plan path**: When executing an approved plan or reusing one from
        list_plans_by_tag, pass plan_path (wrapper injects EXECUTION mode).
        When creating a new plan (plan-only), omit plan_path so the wrapper
        creates one and links it (wrapper injects PLANNING mode). The wrapper
        links plans to the current run's session automatically (session_id is
        optional; pass it from your prompt for consistency if you have it).

        **Fill only**: When you have a cloned plan (from clone_plan) and need
        the workflow to fill only run directory and input paths (no stage agents,
        no execution), pass plan_path and fill_only=True. The workflow will
        create_run_directory, read the plan, and append_to_plan_section(plan_path,
        "parameters", ...) then return. Use this for "base new on previous".
        """
        wf_agent = get_workflow_agent()
        user_message = message.rstrip()
        do_fill_only = fill_only is True and plan_path
        routing_policy = _build_routing_policy_injection()
        unattended_policy = _build_unattended_qa_injection(user_message)

        # Prefer an explicitly passed session_id, but fall back to current run
        # session_id if the explicit ID cannot be linked in session memory.
        session_id_for_linking = session_id or get_current_session_id()

        if plan_path:
            if session_id_for_linking:
                _link_plan_with_fallback(session_id_for_linking, plan_path)
            if do_fill_only:
                # FILL PARAMETERS ONLY: no stage agents, no execution
                effective_message = (
                    "[WORKFLOW MODE: FILL PARAMETERS ONLY]\n\n"
                    + _fill_params_injection
                    + "\n\n"
                    + routing_policy
                    + ("\n\n" + unattended_policy if unattended_policy else "")
                    + "\n\n---\n\n"
                    + user_message
                    + f"\n\n[Plan path: {plan_path}]"
                )
            else:
                # EXECUTION MODE: follow the approved plan
                effective_message = (
                    "[WORKFLOW MODE: EXECUTION]\n\n"
                    + _execution_injection
                    + "\n\n"
                    + routing_policy
                    + ("\n\n" + unattended_policy if unattended_policy else "")
                    + "\n\n---\n\n"
                    + user_message
                    + f"\n\n[Plan path (execute this approved plan): {plan_path}]"
                )
        else:
            # PLANNING MODE: draft the plan only
            plan_path_to_use = None
            if session_id_for_linking:
                plan_path_to_use = _create_plan_path_impl(prefix="plan")
                _link_plan_with_fallback(session_id_for_linking, plan_path_to_use)
            if not plan_path_to_use:
                plan_path_to_use = _create_plan_path_impl(prefix="plan")
            effective_message = (
                "[WORKFLOW MODE: PLANNING]\n\n"
                + _planning_injection
                + "\n\n"
                + routing_policy
                + ("\n\n" + unattended_policy if unattended_policy else "")
                + "\n\n---\n\n"
                + user_message
                + f"\n\n[Plan path: {plan_path_to_use}]"
            )

        result = Runner.run_sync(
            wf_agent,
            effective_message,
            max_turns=50,
        )
        return result.final_output if result and hasattr(result, "final_output") else str(result)

    return workflow_agent
