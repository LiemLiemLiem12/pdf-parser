"""Environment-backed application settings.

This module deliberately has no ML imports. Importing the application in mock
mode must never initialize CUDA or download a model.
"""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class InferenceMode(StrEnum):
    MOCK = "mock"
    TRANSFORMERS = "transformers"


class ModelAlias(StrEnum):
    MOCK = "mock"
    QWEN_7B = "qwen7b"
    QWEN_14B = "qwen14b"
    QWEN_72B = "qwen72b"


MODEL_IDS: dict[ModelAlias, str | None] = {
    ModelAlias.MOCK: None,
    ModelAlias.QWEN_7B: "Qwen/Qwen2.5-7B-Instruct",
    ModelAlias.QWEN_14B: "Qwen/Qwen2.5-14B-Instruct",
    ModelAlias.QWEN_72B: "Qwen/Qwen2.5-72B-Instruct",
}

PositiveInt = Annotated[int, Field(gt=0)]
RetryCount = Annotated[int, Field(ge=0, le=1)]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Resume Parser"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"

    inference_mode: InferenceMode = InferenceMode.MOCK
    model_alias: ModelAlias = ModelAlias.MOCK
    model_id: str | None = None
    model_revision: str = "main"
    quantization: str = "none"

    input_max_chars: PositiveInt = 50_000
    max_new_tokens: PositiveInt = 2_048
    retry_cap: RetryCount = 1
    max_file_size_bytes: PositiveInt = 10 * 1024 * 1024
    max_pdf_pages: PositiveInt = 30
    max_decompressed_size_bytes: PositiveInt = 50 * 1024 * 1024
    artifact_path: Path = Path("artifacts")
    request_id_header: str = "X-Request-ID"

    @model_validator(mode="after")
    def validate_inference_configuration(self) -> "Settings":
        if self.inference_mode is InferenceMode.MOCK:
            if self.model_alias is not ModelAlias.MOCK:
                raise ValueError("MODEL_ALIAS must be 'mock' when INFERENCE_MODE is 'mock'")
            if self.model_id is not None:
                raise ValueError("MODEL_ID must be unset when INFERENCE_MODE is 'mock'")
        elif self.model_alias is ModelAlias.MOCK and not self.model_id:
            raise ValueError(
                "MODEL_ALIAS must select a real model, or MODEL_ID must be set, "
                "when INFERENCE_MODE is 'transformers'"
            )

        if not self.model_revision.strip():
            raise ValueError("MODEL_REVISION must not be blank")
        if not self.request_id_header.strip():
            raise ValueError("REQUEST_ID_HEADER must not be blank")
        if self.input_max_chars < self.max_new_tokens:
            raise ValueError("INPUT_MAX_CHARS must be greater than or equal to MAX_NEW_TOKENS")
        return self

    @property
    def resolved_model_id(self) -> str:
        """Return the configured checkpoint ID, or the deterministic mock label."""

        return self.model_id or MODEL_IDS[self.model_alias] or "mock"
