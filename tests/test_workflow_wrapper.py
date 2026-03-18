from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_workflow_wrapper_module():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "adams" / "pipeline" / "workflow_wrapper.py"
    sys.path.insert(0, str(repo_root))

    adams_pkg = types.ModuleType("adams")
    adams_pkg.__path__ = [str(repo_root / "adams")]
    sys.modules["adams"] = adams_pkg

    pipeline_pkg = types.ModuleType("adams.pipeline")
    pipeline_pkg.__path__ = [str(repo_root / "adams" / "pipeline")]
    sys.modules["adams.pipeline"] = pipeline_pkg

    agents_mod = types.ModuleType("agents")

    class _Runner:
        @staticmethod
        def run_sync(*args, **kwargs):
            raise RuntimeError("Runner.run_sync should not be used in this unit test")

    agents_mod.Runner = _Runner
    agents_mod.function_tool = lambda fn: fn
    sys.modules["agents"] = agents_mod

    memory_mod = types.ModuleType("adams.memory.session_memory")
    memory_mod.add_session_plan_path = lambda *args, **kwargs: None
    sys.modules["adams.memory.session_memory"] = memory_mod

    path_config_mod = types.ModuleType("adams.path_config")
    path_config_mod.get_current_session_id = lambda: "unit-test-session"
    sys.modules["adams.path_config"] = path_config_mod

    user_plan_mod = types.ModuleType("adams.user_plan_utils")
    user_plan_mod._create_plan_path_impl = lambda prefix="plan": f"/tmp/{prefix}.json"
    sys.modules["adams.user_plan_utils"] = user_plan_mod

    workflow_agent_mod = types.ModuleType("adams.pipeline.workflow_agent")
    workflow_agent_mod.get_workflow_agent = lambda: object()
    sys.modules["adams.pipeline.workflow_agent"] = workflow_agent_mod

    spec = importlib.util.spec_from_file_location("adams.pipeline.workflow_wrapper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_unattended_qa_injection_detects_authoritative_prompt():
    mod = _load_workflow_wrapper_module()
    message = (
        "UNATTENDED QA EXECUTION POLICY (AUTHORITATIVE)\n"
        "UNATTENDED CONTRACT OVERRIDE (AUTHORITATIVE)\n"
        "out_folder=/tmp/out\n"
    )

    injection = mod._build_unattended_qa_injection(message)

    assert "[INTERNAL UNATTENDED QA POLICY]" in injection
    assert "Do not search for, reuse, clone, or adapt plans from other scenarios." in injection
    assert "Do not invent extra parameter sections such as runner_qa." in injection


def test_build_unattended_qa_injection_is_empty_for_normal_messages():
    mod = _load_workflow_wrapper_module()

    injection = mod._build_unattended_qa_injection("Please dock these ligands against this receptor.")

    assert injection == ""
