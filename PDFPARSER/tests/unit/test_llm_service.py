import pytest

from app.llm.contracts import ChatMessage, CompletionStatus, MessageRole
from app.services.llm_service import MockLLMService, MockScenario


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
