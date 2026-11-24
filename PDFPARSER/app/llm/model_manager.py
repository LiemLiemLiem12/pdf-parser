"""Lifecycle manager for one allowlisted Transformers checkpoint."""

from __future__ import annotations

import gc
import logging
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Callable

from app.core.config import Settings
from app.llm.model_config import ModelConfig, resolve_model_config


logger = logging.getLogger(__name__)


class ModelManagerState(StrEnum):
    NEW = "new"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


class ModelLoadError(RuntimeError):
    """Raised when a real model cannot be initialized safely."""


@dataclass(frozen=True, slots=True)
class ModelMetadata:
    alias: str
    model_id: str
    revision: str
    quantization: str
    compute_dtype: str
    device_map: dict[str, str]


def _load_ml_dependencies() -> tuple[Any, Any, Any, Any]:
    """Import heavy ML packages only when real inference is selected."""

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except (ImportError, OSError) as exc:
        raise ModelLoadError("Real-model dependencies are unavailable") from exc
    return torch, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def _normalized_device_map(model: Any) -> dict[str, str]:
    raw_map = getattr(model, "hf_device_map", None)
    if not isinstance(raw_map, dict) or not raw_map:
        device = getattr(model, "device", None)
        if device is None:
            raise ModelLoadError("Loaded model did not expose its device placement")
        raw_map = {"": device}
    return {str(module): str(device) for module, device in raw_map.items()}


def _verify_gpu_only_device_map(device_map: dict[str, str]) -> None:
    invalid = {
        module: device
        for module, device in device_map.items()
        if not (device.isdigit() or device == "cuda" or device.startswith("cuda:"))
    }
    if invalid:
        placements = ", ".join(
            f"{module or '<root>'}={device}" for module, device in sorted(invalid.items())
        )
        raise ModelLoadError(f"GPU-only baseline rejected device placement: {placements}")


class ModelManager:
    """Load, describe, reuse and release exactly one model/tokenizer pair."""

    def __init__(
        self,
        settings: Settings,
        *,
        dependency_loader: Callable[[], tuple[Any, Any, Any, Any]] = _load_ml_dependencies,
    ) -> None:
        self._settings = settings
        self._dependency_loader = dependency_loader
        self._state = ModelManagerState.NEW
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._metadata: ModelMetadata | None = None
        self._torch: Any | None = None
        self._failure_reason: str | None = None

    @property
    def state(self) -> ModelManagerState:
        return self._state

    @property
    def ready(self) -> bool:
        return self._state is ModelManagerState.READY

    @property
    def model(self) -> Any:
        if not self.ready or self._model is None:
            raise ModelLoadError("Model is not ready")
        return self._model

    @property
    def tokenizer(self) -> Any:
        if not self.ready or self._tokenizer is None:
            raise ModelLoadError("Tokenizer is not ready")
        return self._tokenizer

    @property
    def metadata(self) -> dict[str, Any] | None:
        return asdict(self._metadata) if self._metadata is not None else None

    @property
    def failure_reason(self) -> str | None:
        return self._failure_reason

    def load(self) -> None:
        if self.ready:
            return
        if self._state is not ModelManagerState.NEW:
            raise ModelLoadError(f"Model load cannot run from state {self._state.value!r}")

        self._state = ModelManagerState.LOADING
        try:
            config = self._resolve_config()
            torch, model_class, tokenizer_class, quantization_class = (
                self._dependency_loader()
            )
            self._torch = torch
            self._require_cuda(torch)
            compute_dtype, dtype_name = self._select_compute_dtype(torch)
            quantization_config = quantization_class(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            tokenizer = tokenizer_class.from_pretrained(
                config.model_id,
                revision=self._settings.model_revision,
                trust_remote_code=False,
            )
            model = model_class.from_pretrained(
                config.model_id,
                revision=self._settings.model_revision,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=compute_dtype,
                trust_remote_code=False,
            )
            self._tokenizer = tokenizer
            self._model = model
            device_map = _normalized_device_map(model)
            _verify_gpu_only_device_map(device_map)
            model.eval()

            self._metadata = ModelMetadata(
                alias=config.alias,
                model_id=config.model_id,
                revision=self._settings.model_revision,
                quantization=config.quantization,
                compute_dtype=dtype_name,
                device_map=device_map,
            )
            self._state = ModelManagerState.READY
            logger.info("model_loaded", extra=self.metadata or {})
        except Exception as exc:
            self._state = ModelManagerState.FAILED
            self._failure_reason = self._safe_failure_reason(exc)
            self._release_resources()
            if isinstance(exc, ModelLoadError):
                raise
            raise ModelLoadError(self._failure_reason) from exc

    def close(self) -> None:
        self._release_resources()
        self._state = ModelManagerState.CLOSED

    def _resolve_config(self) -> ModelConfig:
        return resolve_model_config(
            self._settings.model_alias.value,
            self._settings.quantization,
            model_id_override=self._settings.model_id,
        )

    @staticmethod
    def _require_cuda(torch: Any) -> None:
        if not torch.cuda.is_available():
            raise ModelLoadError("CUDA is unavailable; refusing real-model startup")

    @staticmethod
    def _select_compute_dtype(torch: Any) -> tuple[Any, str]:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
        return torch.float16, "float16"

    @staticmethod
    def _safe_failure_reason(exc: Exception) -> str:
        name = type(exc).__name__
        if "out of memory" in str(exc).lower():
            return f"Model initialization failed: CUDA out of memory ({name})"
        return f"Model initialization failed ({name})"

    def _release_resources(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        if self._torch is not None:
            try:
                if self._torch.cuda.is_available():
                    self._torch.cuda.empty_cache()
            except (AttributeError, RuntimeError):
                logger.warning("gpu_cache_cleanup_failed")
