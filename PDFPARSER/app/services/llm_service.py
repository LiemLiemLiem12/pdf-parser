"""Mock and Transformers-backed text generation services."""

import json
import threading
import time
from enum import StrEnum
from typing import Any, Sequence

from app.llm.contracts import (
    ChatMessage,
    CompletionStatus,
    GenerationResult,
)
from app.llm.model_manager import ModelManager


def _load_torch() -> Any:
    """Import torch only when real inference is explicitly constructed."""

    import torch

    return torch


def _model_input_device(model: Any) -> Any:
    """Resolve the device used by the model's input embedding layer."""

    get_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_embeddings):
        embeddings = get_embeddings()
        weight = getattr(embeddings, "weight", None)
        device = getattr(weight, "device", None)
        if device is not None:
            return device

    device = getattr(model, "device", None)
    if device is None:
        raise RuntimeError("Loaded model did not expose an input device")
    return device


def _eos_token_ids(model: Any, tokenizer: Any) -> set[int]:
    """Collect tokenizer and generation-config EOS IDs in a normalized set."""

    values = [getattr(tokenizer, "eos_token_id", None)]
    generation_config = getattr(model, "generation_config", None)
    values.append(getattr(generation_config, "eos_token_id", None))

    result: set[int] = set()
    for value in values:
        if isinstance(value, int):
            result.add(value)
        elif isinstance(value, (list, tuple, set)):
            result.update(item for item in value if isinstance(item, int))
    return result


class TransformersLLMService:
    """Run deterministic, serialized generation using one loaded model manager."""

    def __init__(
        self,
        manager: ModelManager,
        *,
        context_window_tokens: int,
        max_new_tokens: int,
        torch_module: Any | None = None,
        generation_lock: threading.Lock | None = None,
    ) -> None:
        if max_new_tokens >= context_window_tokens:
            raise ValueError("max_new_tokens must be less than context_window_tokens")

        self._model = manager.model
        self._tokenizer = manager.tokenizer
        self._torch = torch_module or _load_torch()
        self._context_window_tokens = context_window_tokens
        self._max_new_tokens = max_new_tokens
        self._generation_lock = generation_lock or threading.Lock()

    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        prompt = self._tokenizer.apply_chat_template(
            [
                {"role": message.role.value, "content": message.content}
                for message in messages
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = self._tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
        )
        input_tokens = int(encoded["input_ids"].shape[-1])
        if input_tokens + self._max_new_tokens > self._context_window_tokens:
            return GenerationResult(
                raw_text="",
                completion_status=CompletionStatus.FAILED,
                input_tokens=input_tokens,
                output_tokens=0,
                error_code="CONTEXT_LIMIT_EXCEEDED",
                error_message="Resume prompt exceeds the model context limit",
            )

        device = _model_input_device(self._model)
        model_inputs = {
            name: tensor.to(device)
            for name, tensor in encoded.items()
        }

        with self._generation_lock:
            return self._generate_locked(model_inputs, input_tokens, device)

    def _generate_locked(
        self,
        model_inputs: dict[str, Any],
        input_tokens: int,
        device: Any,
    ) -> GenerationResult:
        cuda = self._torch.cuda
        try:
            cuda.reset_peak_memory_stats(device)
            cuda.synchronize(device)
            started_at = time.perf_counter()
            with self._torch.inference_mode():
                generated = self._model.generate(
                    **model_inputs,
                    do_sample=False,
                    max_new_tokens=self._max_new_tokens,
                )
            cuda.synchronize(device)
            generation_ms = (time.perf_counter() - started_at) * 1_000

            sequences = getattr(generated, "sequences", generated)
            continuation = sequences[:, input_tokens:]
            output_tokens = int(continuation.shape[-1])
            token_ids = continuation[0].tolist()
            eos_ids = _eos_token_ids(self._model, self._tokenizer)
            reached_eos = any(token_id in eos_ids for token_id in token_ids)
            truncated = output_tokens >= self._max_new_tokens and not reached_eos
            status = (
                CompletionStatus.TRUNCATED
                if truncated
                else CompletionStatus.COMPLETED
            )
            raw_text = self._tokenizer.decode(
                continuation[0],
                skip_special_tokens=True,
            )
            return GenerationResult(
                raw_text=raw_text,
                completion_status=status,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                generation_ms=generation_ms,
                peak_allocated_vram_bytes=int(cuda.max_memory_allocated(device)),
                peak_reserved_vram_bytes=int(cuda.max_memory_reserved(device)),
                error_code="MAX_NEW_TOKENS" if truncated else None,
                error_message=(
                    "Generation reached its output token limit" if truncated else None
                ),
            )
        except Exception as exc:
            is_oom = self._is_cuda_oom(exc)
            if is_oom:
                try:
                    cuda.empty_cache()
                except (AttributeError, RuntimeError):
                    pass
            return GenerationResult(
                raw_text="",
                completion_status=CompletionStatus.FAILED,
                input_tokens=input_tokens,
                output_tokens=0,
                error_code="CUDA_OUT_OF_MEMORY" if is_oom else "GENERATION_FAILED",
                error_message=(
                    "Model generation ran out of GPU memory"
                    if is_oom
                    else "Model generation failed"
                ),
            )

    def _is_cuda_oom(self, exc: Exception) -> bool:
        cuda_oom_type = getattr(self._torch.cuda, "OutOfMemoryError", None)
        return (
            isinstance(cuda_oom_type, type)
            and isinstance(exc, cuda_oom_type)
        ) or "out of memory" in str(exc).lower()


class MockScenario(StrEnum):
    VALID = "valid"
    MALFORMED_JSON = "malformed_json"
    INCORRECT_SCHEMA = "incorrect_schema"
    TRUNCATED = "truncated"
    FAILURE = "failure"


_VALID_PROFILE = {
    "personal_info": {
        "full_name": "Nguyen Van A",
        "email": None,
        "phone": None,
        "location": None,
    },
    "summary": None,
    "work_experience": [],
    "education": [],
    "skills": ["Python"],
    "certifications": [],
    "languages": [],
}


class MockLLMService:
    """Stable test double; its output must never be used as accuracy evidence."""

    def __init__(self, scenario: MockScenario = MockScenario.VALID) -> None:
        self.scenario = scenario
        self.call_count = 0
        self.last_messages: tuple[ChatMessage, ...] = ()

    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        self.call_count += 1
        self.last_messages = tuple(messages)

        if self.scenario is MockScenario.VALID:
            return GenerationResult(
                raw_text=json.dumps(_VALID_PROFILE, ensure_ascii=False),
                completion_status=CompletionStatus.COMPLETED,
            )
        if self.scenario is MockScenario.MALFORMED_JSON:
            return GenerationResult(
                raw_text='{"personal_info":',
                completion_status=CompletionStatus.COMPLETED,
            )
        if self.scenario is MockScenario.INCORRECT_SCHEMA:
            return GenerationResult(
                raw_text='{"personal_info": {}}',
                completion_status=CompletionStatus.COMPLETED,
            )
        if self.scenario is MockScenario.TRUNCATED:
            return GenerationResult(
                raw_text='{"personal_info":',
                completion_status=CompletionStatus.TRUNCATED,
                error_code="MAX_NEW_TOKENS",
                error_message="Generation reached its output token limit",
            )
        return GenerationResult(
            raw_text="",
            completion_status=CompletionStatus.FAILED,
            error_code="MOCK_GENERATION_FAILURE",
            error_message="Deterministic mock generation failure",
        )
