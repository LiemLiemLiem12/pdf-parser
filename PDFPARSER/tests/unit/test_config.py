import pytest
from pydantic import ValidationError

from app.core.config import InferenceMode, ModelAlias, Settings


def test_defaults_are_deterministic_and_mock_only() -> None:
    settings = Settings(_env_file=None)

    assert settings.inference_mode is InferenceMode.MOCK
    assert settings.model_alias is ModelAlias.MOCK
    assert settings.resolved_model_id == "mock"


def test_mock_mode_rejects_real_model_alias() -> None:
    with pytest.raises(ValidationError, match="MODEL_ALIAS must be 'mock'"):
        Settings(_env_file=None, inference_mode="mock", model_alias="qwen14b")


def test_transformers_mode_requires_real_model_selection() -> None:
    with pytest.raises(ValidationError, match="MODEL_ALIAS must select a real model"):
        Settings(_env_file=None, inference_mode="transformers", model_alias="mock")


def test_transformers_mode_resolves_known_model() -> None:
    settings = Settings(
        _env_file=None,
        inference_mode="transformers",
        model_alias="qwen14b",
        quantization="nf4-4bit",
    )

    assert settings.resolved_model_id == "Qwen/Qwen2.5-14B-Instruct"


def test_retry_cap_cannot_exceed_one() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, retry_cap=2)
