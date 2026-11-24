"""Reusable deterministic fakes for parsing-service tests."""

from collections.abc import Sequence

from app.llm.contracts import ChatMessage, GenerationResult


class SequenceGenerator:
    def __init__(self, results: Sequence[GenerationResult]) -> None:
        if not results:
            raise ValueError("At least one generation result is required")
        self._results = list(results)
        self.calls: list[tuple[ChatMessage, ...]] = []

    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        self.calls.append(tuple(messages))
        index = min(len(self.calls) - 1, len(self._results) - 1)
        return self._results[index]
