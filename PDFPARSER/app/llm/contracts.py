"""Small generation interface used by the parsing core."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, Sequence


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: MessageRole
    content: str


class CompletionStatus(StrEnum):
    COMPLETED = "completed"
    TRUNCATED = "truncated"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class GenerationResult:
    raw_text: str
    completion_status: CompletionStatus
    input_tokens: int | None = None
    output_tokens: int | None = None
    generation_ms: float | None = None
    peak_allocated_vram_bytes: int | None = None
    peak_reserved_vram_bytes: int | None = None
    error_code: str | None = None
    error_message: str | None = None


class Generator(Protocol):
    def generate(self, messages: Sequence[ChatMessage]) -> GenerationResult:
        """Generate one response for the supplied ordered chat messages."""
