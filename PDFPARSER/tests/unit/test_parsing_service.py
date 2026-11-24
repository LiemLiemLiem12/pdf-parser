import json

import pytest

from app.core.exceptions import GenerationFailedError, InvalidModelOutputError
from app.llm.contracts import CompletionStatus, GenerationResult
from app.services.llm_service import MockLLMService
from app.services.parsing_service import ParsingService
from tests.fakes import SequenceGenerator
from tests.fixtures import sparse_profile


def completed(raw_text: str) -> GenerationResult:
    return GenerationResult(
        raw_text=raw_text,
        completion_status=CompletionStatus.COMPLETED,
        input_tokens=10,
        output_tokens=20,
        generation_ms=5.5,
    )


def test_valid_mock_output_returns_profile_and_metadata() -> None:
    result = ParsingService(MockLLMService()).parse("Kỹ năng: Python")

    assert result.data.skills == ["Python"]
    assert result.metadata.attempt_count == 1
    assert result.metadata.first_attempt_status == "valid"
    assert result.metadata.schema_version == "1.0"


def test_invalid_json_can_receive_one_corrective_retry() -> None:
    generator = SequenceGenerator(
        [completed('{"personal_info":'), completed(json.dumps(sparse_profile()))]
    )

    result = ParsingService(generator, retry_cap=1).parse("Synthetic resume")

    assert result.metadata.attempt_count == 2
    assert result.metadata.first_attempt_status == "invalid_json"
    assert len(generator.calls) == 2
    assert len(generator.calls[1]) == 4


def test_raw_baseline_disables_retry_and_preserves_failure_status() -> None:
    generator = SequenceGenerator([completed('{"personal_info":')])

    with pytest.raises(InvalidModelOutputError) as captured:
        ParsingService(generator).parse("Synthetic resume", allow_retry=False)

    assert captured.value.context == {
        "attempt_count": 1,
        "first_attempt_status": "invalid_json",
    }
    assert len(generator.calls) == 1


@pytest.mark.parametrize("status", [CompletionStatus.TRUNCATED, CompletionStatus.FAILED])
def test_generation_failure_or_truncation_is_never_retried(status: CompletionStatus) -> None:
    generator = SequenceGenerator(
        [GenerationResult(raw_text='{"partial":', completion_status=status)]
    )

    with pytest.raises(GenerationFailedError) as captured:
        ParsingService(generator).parse("Synthetic resume")

    assert captured.value.context["attempt_count"] == 1
    assert len(generator.calls) == 1


def test_failed_validation_never_becomes_empty_profile() -> None:
    generator = SequenceGenerator([completed('{"personal_info": {}}')])

    with pytest.raises(InvalidModelOutputError) as captured:
        ParsingService(generator, retry_cap=0).parse("Synthetic resume")

    assert captured.value.code == "INVALID_RESUME_SCHEMA"
    assert captured.value.context["first_attempt_status"] == "invalid_schema"


def test_second_attempt_failure_preserves_first_attempt_status() -> None:
    generator = SequenceGenerator(
        [
            completed('{"personal_info":'),
            GenerationResult(raw_text="", completion_status=CompletionStatus.FAILED),
        ]
    )

    with pytest.raises(GenerationFailedError) as captured:
        ParsingService(generator).parse("Synthetic resume")

    assert captured.value.context == {
        "attempt_count": 2,
        "first_attempt_status": "invalid_json",
    }
