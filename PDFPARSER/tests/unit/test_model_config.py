import pytest

from app.llm.model_config import ModelConfigurationError, resolve_model_config


def test_allowlisted_model_is_resolved() -> None:
    config = resolve_model_config("qwen14b", "nf4-4bit")

    assert config.model_id == "Qwen/Qwen2.5-14B-Instruct"
    assert config.alias == "qwen14b"


def test_unknown_alias_is_rejected() -> None:
    with pytest.raises(ModelConfigurationError, match="not allowlisted"):
        resolve_model_config("arbitrary", "nf4-4bit")


def test_unreviewed_model_override_is_rejected() -> None:
    with pytest.raises(ModelConfigurationError, match="must match"):
        resolve_model_config(
            "qwen14b",
            "nf4-4bit",
            model_id_override="unreviewed/checkpoint",
        )


def test_unsupported_quantization_is_rejected() -> None:
    with pytest.raises(ModelConfigurationError, match="not allowed"):
        resolve_model_config("qwen14b", "none")
