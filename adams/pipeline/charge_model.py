"""
Shared helpers for charge-model normalization and validation.
"""

SUPPORTED_CHARGE_MODELS = ("gasteiger", "eem", "mmff94", "qeq", "qtpie")


def normalize_charge_model(charge_model: str) -> str:
    """Return normalized charge model string."""
    if charge_model is None:
        raise ValueError(
            "charge_model cannot be None. Use one of: "
            + ", ".join(SUPPORTED_CHARGE_MODELS)
        )

    model = str(charge_model).strip().lower()
    if not model:
        raise ValueError(
            "charge_model cannot be empty. Use one of: "
            + ", ".join(SUPPORTED_CHARGE_MODELS)
        )
    return model


def validate_charge_model(charge_model: str) -> str:
    """
    Validate charge model and return normalized value.

    The allowed set is chosen so receptor (OpenBabel) and ligand pipeline metadata
    stay explicit and typo-safe. Docking runs should use the same value across all
    stages.
    """
    model = normalize_charge_model(charge_model)
    if model not in SUPPORTED_CHARGE_MODELS:
        raise ValueError(
            f"Unsupported charge_model: {charge_model}. "
            f"Choose one of: {', '.join(SUPPORTED_CHARGE_MODELS)}"
        )
    return model
