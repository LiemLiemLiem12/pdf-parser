import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from app.llm.contracts import ChatMessage, CompletionStatus, MessageRole
from app.services.llm_service import (
    MockLLMService,
    MockScenario,
    TransformersLLMService,
)


class FakeTensor:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows
        self.shape = (len(rows), len(rows[0]) if rows else 0)
        self.device = None

    def to(self, device):  # type: ignore[no-untyped-def]
        self.device = device
        return self

    def __getitem__(self, key):  # type: ignore[no-untyped-def]
        if isinstance(key, tuple):
            row_key, column_key = key
            rows = self.rows[row_key]
            if rows and isinstance(rows[0], int):
                rows = [rows]
            return FakeTensor([row[column_key] for row in rows])
        return FakeVector(self.rows[key])


class FakeVector:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[int]:
        return list(self.values)


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids
        self.template_calls: list[tuple[list[dict[str, str]], dict[str, object]]] = []
        self.tokenizer_kwargs: dict[str, object] = {}
        self.decoded_ids: list[int] = []

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.template_calls.append((messages, kwargs))
        return "rendered prompt"

    def __call__(self, prompt, **kwargs):  # type: ignore[no-untyped-def]
        assert prompt == "rendered prompt"
        self.tokenizer_kwargs = kwargs
        return {
            "input_ids": FakeTensor([list(self.input_ids)]),
            "attention_mask": FakeTensor([[1] * len(self.input_ids)]),
        }

    def decode(self, token_ids: FakeVector, **kwargs) -> str:  # type: ignore[no-untyped-def]
        assert kwargs == {"skip_special_tokens": True}
        self.decoded_ids = token_ids.tolist()
        return '{"skills": ["Python"]}'


class FakeModel:
    def __init__(
        self,
        output_ids: list[int],
        *,
        error: Exception | None = None,
        delay_seconds: float = 0,
    ) -> None:
        self.output_ids = output_ids
        self.error = error
        self.delay_seconds = delay_seconds
        self.generation_config = SimpleNamespace(eos_token_id=[99, 100])
        self.generate_calls = 0
        self.generate_kwargs: dict[str, object] = {}
        self.active_calls = 0
        self.max_active_calls = 0
        self._counter_lock = threading.Lock()

    def get_input_embeddings(self):  # type: ignore[no-untyped-def]
        return SimpleNamespace(weight=SimpleNamespace(device="cuda:0"))

    def generate(self, **kwargs):  # type: ignore[no-untyped-def]
        self.generate_calls += 1
        self.generate_kwargs = kwargs
        with self._counter_lock:
            self.active_calls += 1
            self.max_active_calls = max(self.max_active_calls, self.active_calls)
        try:
            if self.delay_seconds:
                time.sleep(self.delay_seconds)
            if self.error is not None:
                raise self.error
            prompt_ids = kwargs["input_ids"].rows[0]
            return FakeTensor([prompt_ids + self.output_ids])
        finally:
            with self._counter_lock:
                self.active_calls -= 1


class FakeCuda:
    OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})

    def __init__(self) -> None:
        self.empty_cache_calls = 0
        self.reset_devices: list[str] = []
        self.synchronize_devices: list[str] = []

    def reset_peak_memory_stats(self, device: str) -> None:
        self.reset_devices.append(device)

    def synchronize(self, device: str) -> None:
        self.synchronize_devices.append(device)

    @staticmethod
    def max_memory_allocated(device: str) -> int:
        assert device == "cuda:0"
        return 1_000

    @staticmethod
    def max_memory_reserved(device: str) -> int:
        assert device == "cuda:0"
        return 2_000

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


class FakeTorch:
    def __init__(self) -> None:
        self.cuda = FakeCuda()

    @staticmethod
    def inference_mode():  # type: ignore[no-untyped-def]
        return nullcontext()


class FakeManager:
    def __init__(self, model: FakeModel, tokenizer: FakeTokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer


def real_service(
    model: FakeModel,
    tokenizer: FakeTokenizer,
    *,
    context_window_tokens: int = 10,
    max_new_tokens: int = 5,
) -> tuple[TransformersLLMService, FakeTorch]:
    torch = FakeTorch()
    service = TransformersLLMService(
        FakeManager(model, tokenizer),  # type: ignore[arg-type]
        context_window_tokens=context_window_tokens,
        max_new_tokens=max_new_tokens,
        torch_module=torch,
    )
    return service, torch


@pytest.mark.parametrize(
    ("scenario", "status"),
    [
        (MockScenario.VALID, CompletionStatus.COMPLETED),
        (MockScenario.MALFORMED_JSON, CompletionStatus.COMPLETED),
        (MockScenario.INCORRECT_SCHEMA, CompletionStatus.COMPLETED),
        (MockScenario.TRUNCATED, CompletionStatus.TRUNCATED),
        (MockScenario.FAILURE, CompletionStatus.FAILED),
    ],
)
def test_mock_scenarios_are_deterministic(
    scenario: MockScenario, status: CompletionStatus
) -> None:
    service = MockLLMService(scenario)
    messages = (ChatMessage(MessageRole.USER, "synthetic resume"),)

    first = service.generate(messages)
    second = service.generate(messages)

    assert first == second
    assert first.completion_status is status
    assert service.call_count == 2


def test_transformers_generation_uses_chat_template_and_decodes_only_continuation() -> None:
    tokenizer = FakeTokenizer([1, 2, 3])
    model = FakeModel([10, 11, 99])
    service, torch = real_service(model, tokenizer)
    messages = (
        ChatMessage(MessageRole.SYSTEM, "Return JSON"),
        ChatMessage(MessageRole.USER, "Synthetic resume"),
    )

    result = service.generate(messages)

    assert result.completion_status is CompletionStatus.COMPLETED
    assert result.raw_text == '{"skills": ["Python"]}'
    assert result.input_tokens == 3
    assert result.output_tokens == 3
    assert result.peak_allocated_vram_bytes == 1_000
    assert result.peak_reserved_vram_bytes == 2_000
    assert tokenizer.template_calls == [
        (
            [
                {"role": "system", "content": "Return JSON"},
                {"role": "user", "content": "Synthetic resume"},
            ],
            {"tokenize": False, "add_generation_prompt": True},
        )
    ]
    assert tokenizer.tokenizer_kwargs == {
        "return_tensors": "pt",
        "add_special_tokens": False,
    }
    assert tokenizer.decoded_ids == [10, 11, 99]
    assert model.generate_kwargs["do_sample"] is False
    assert model.generate_kwargs["max_new_tokens"] == 5
    assert model.generate_kwargs["input_ids"].device == "cuda:0"
    assert torch.cuda.reset_devices == ["cuda:0"]
    assert torch.cuda.synchronize_devices == ["cuda:0", "cuda:0"]


def test_context_overflow_fails_before_model_generation() -> None:
    tokenizer = FakeTokenizer([1, 2, 3, 4, 5, 6])
    model = FakeModel([99])
    service, _ = real_service(model, tokenizer)

    result = service.generate((ChatMessage(MessageRole.USER, "resume"),))

    assert result.completion_status is CompletionStatus.FAILED
    assert result.error_code == "CONTEXT_LIMIT_EXCEEDED"
    assert result.input_tokens == 6
    assert model.generate_calls == 0


def test_generation_at_token_limit_without_eos_is_truncated() -> None:
    tokenizer = FakeTokenizer([1, 2, 3])
    model = FakeModel([10, 11, 12])
    service, _ = real_service(model, tokenizer, max_new_tokens=3)

    result = service.generate((ChatMessage(MessageRole.USER, "resume"),))

    assert result.completion_status is CompletionStatus.TRUNCATED
    assert result.error_code == "MAX_NEW_TOKENS"
    assert result.output_tokens == 3


def test_cuda_oom_is_sanitized_and_not_retried() -> None:
    tokenizer = FakeTokenizer([1, 2, 3])
    model = FakeModel([], error=RuntimeError("CUDA out of memory: private details"))
    service, torch = real_service(model, tokenizer)

    result = service.generate((ChatMessage(MessageRole.USER, "private resume"),))

    assert result.completion_status is CompletionStatus.FAILED
    assert result.error_code == "CUDA_OUT_OF_MEMORY"
    assert result.error_message == "Model generation ran out of GPU memory"
    assert "private resume" not in result.error_message
    assert model.generate_calls == 1
    assert torch.cuda.empty_cache_calls == 1


def test_generation_lock_serializes_concurrent_requests() -> None:
    tokenizer = FakeTokenizer([1, 2, 3])
    model = FakeModel([99], delay_seconds=0.02)
    service, _ = real_service(model, tokenizer)
    messages = (ChatMessage(MessageRole.USER, "resume"),)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: service.generate(messages), range(2)))

    assert all(result.completion_status is CompletionStatus.COMPLETED for result in results)
    assert model.generate_calls == 2
    assert model.max_active_calls == 1
