"""Deterministic construction of trusted instructions and untrusted CV data."""

import json
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from app.llm.contracts import ChatMessage, MessageRole
from app.schemas.resume_schema import ResumeProfile


PROMPT_VERSION = "v1"


class PromptBuilder:
    def __init__(self, template_path: Path | None = None) -> None:
        path = template_path or Path(__file__).with_name("resume_extraction_v1.txt")
        self._instructions = path.read_text(encoding="utf-8").strip()

    @property
    def version(self) -> str:
        return PROMPT_VERSION

    def build(
        self,
        text: str,
        schema: type[BaseModel] = ResumeProfile,
    ) -> tuple[ChatMessage, ...]:
        schema_json = json.dumps(
            schema.model_json_schema(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        system_content = f"{self._instructions}\n\nJSON Schema:\n{schema_json}"
        user_content = f"RESUME_TEXT_BEGIN\n{text}\nRESUME_TEXT_END"
        return (
            ChatMessage(role=MessageRole.SYSTEM, content=system_content),
            ChatMessage(role=MessageRole.USER, content=user_content),
        )

    def build_correction(
        self,
        messages: Sequence[ChatMessage],
        raw_output: str,
        error_code: str,
    ) -> tuple[ChatMessage, ...]:
        """Add one bounded correction turn while preserving the original prompt."""

        correction = (
            f"The previous output was rejected with {error_code}. "
            "Return a corrected single JSON object only. Do not add commentary."
        )
        return (
            *messages,
            ChatMessage(role=MessageRole.ASSISTANT, content=raw_output),
            ChatMessage(role=MessageRole.USER, content=correction),
        )
