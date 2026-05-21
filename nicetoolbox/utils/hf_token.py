"""Resolve Hugging Face Hub token from machine-specific config."""

from nicetoolbox.configs.schemas.machine_specific_paths import MachineSpecificConfig


def effective_hf_hub_token(machine: MachineSpecificConfig | None = None) -> str | None:
    """
    Return a non-empty token string if ``hugging_face_token`` is set in machine_specific_paths.toml.

    Environment variables are not consulted; configure the token only in ``machine_specific_paths.toml``.
    """
    if machine is None:
        return None
    v = getattr(machine, "hugging_face_token", None)
    if v and str(v).strip():
        return str(v).strip()
    return None
