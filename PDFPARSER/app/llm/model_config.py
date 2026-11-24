"""Allowlisted real-model configuration resolution."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODEL_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "models.yaml"


class ModelConfigurationError(ValueError):
    """Raised when a model selection or configuration file is invalid."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    alias: str
    model_id: str
    quantization: str


def _read_config(path: Path) -> dict[str, Any]:
    """Read the JSON-compatible YAML file without adding a base dependency."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelConfigurationError(f"Unable to read model configuration: {path}") from exc
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ModelConfigurationError("Unsupported model configuration schema")
    return value


def resolve_model_config(
    alias: str,
    quantization: str,
    *,
    model_id_override: str | None = None,
    config_path: Path = DEFAULT_MODEL_CONFIG_PATH,
) -> ModelConfig:
    """Resolve an allowlisted alias and reject unreviewed checkpoint overrides."""

    document = _read_config(config_path)
    models = document.get("models")
    entry = models.get(alias) if isinstance(models, dict) else None
    if not isinstance(entry, dict):
        allowed = ", ".join(sorted(models)) if isinstance(models, dict) else "none"
        raise ModelConfigurationError(
            f"Model alias {alias!r} is not allowlisted; expected one of: {allowed}"
        )

    model_id = entry.get("model_id")
    quantizations = entry.get("quantizations")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ModelConfigurationError(f"Model alias {alias!r} has no valid model ID")
    if not isinstance(quantizations, list) or not all(
        isinstance(item, str) for item in quantizations
    ):
        raise ModelConfigurationError(f"Model alias {alias!r} has invalid quantizations")
    if quantization not in quantizations:
        raise ModelConfigurationError(
            f"Quantization {quantization!r} is not allowed for model alias {alias!r}"
        )
    if model_id_override is not None and model_id_override != model_id:
        raise ModelConfigurationError(
            "MODEL_ID must match the checkpoint registered for the selected MODEL_ALIAS"
        )
    return ModelConfig(alias=alias, model_id=model_id, quantization=quantization)
