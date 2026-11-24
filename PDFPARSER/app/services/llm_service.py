"""Deterministic generator used by unit and API tests."""

import json
from enum import StrEnum
from typing import Sequence

from app.llm.contracts import (
    ChatMessage,
    CompletionStatus,
    GenerationResult,
)


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
