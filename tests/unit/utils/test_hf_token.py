"""Tests for Hugging Face token resolution."""

from pathlib import Path

from nicetoolbox.configs.schemas.machine_specific_paths import MachineSpecificConfig
from nicetoolbox.utils.hf_token import effective_hf_hub_token


def test_effective_token_from_machine(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "ignored_env")
    m = MachineSpecificConfig(conda_path=Path("/tmp"), hugging_face_token="machine_tok")
    assert effective_hf_hub_token(m) == "machine_tok"


def test_effective_token_ignores_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "env_only")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hub_only")
    m = MachineSpecificConfig(conda_path=Path("/tmp"), hugging_face_token="")
    assert effective_hf_hub_token(m) is None


def test_effective_token_from_machine_when_trimmed(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    m = MachineSpecificConfig(conda_path=Path("/tmp"), hugging_face_token="  hf_abc  ")
    assert effective_hf_hub_token(m) == "hf_abc"


def test_effective_token_none_when_machine_empty(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    m = MachineSpecificConfig(conda_path=Path("/tmp"), hugging_face_token="")
    assert effective_hf_hub_token(m) is None


def test_effective_token_none_when_machine_none():
    assert effective_hf_hub_token(None) is None
