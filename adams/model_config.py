"""
Single source of truth for the LLM model used by all agents.

create_agent(session_id, model=...) sets the model; all agents (executive and
sub-agents) are created with the same resolved model so one configuration
drives the entire stack.
"""

_current_model: str = "gpt-5.4"


def _resolve_model(model: str):
    """Resolve a model string to an SDK-compatible model object.

    Plain names (e.g. 'gpt-5.4') are passed through as-is for native OpenAI
    routing.  Names containing '/' (e.g. 'gemini/gemini-3-pro',
    'anthropic/claude-3-5-sonnet-20240620') are wrapped in LitellmModel
    so the Agents SDK routes them through LiteLLM.
    """
    if "/" in model:
        from agents.extensions.models.litellm_model import LitellmModel
        return LitellmModel(model=model)
    return model


def set_model(model: str) -> None:
    """Set the model used by all agents. Call at the start of create_agent()."""
    global _current_model
    _current_model = model


def get_current_model_name() -> str:
    """Return the currently configured model name."""
    return _current_model


def get_resolved_model():
    """Return the current model resolved for the SDK (same as executive agent)."""
    return _resolve_model(_current_model)
