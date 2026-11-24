import json

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.llm.contracts import CompletionStatus, GenerationResult
from app.main import create_app
from tests.fakes import SequenceGenerator
from tests.fixtures import sparse_profile


def completed(raw_text: str) -> GenerationResult:
    return GenerationResult(raw_text=raw_text, completion_status=CompletionStatus.COMPLETED)


def test_parse_text_returns_stable_contract_and_passes_text_to_prompt() -> None:
    app = create_app(Settings(_env_file=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/resumes/parse-text",
            json={"text": "Kỹ năng: Python"},
            headers={"X-Request-ID": "req_parse_text"},
        )
        generator = app.state.runtime.generator

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req_parse_text"
    assert body["status"] == "success"
    assert body["model"] == "mock"
    assert body["schema_version"] == "1.0"
    assert body["data"]["skills"] == ["Python"]
    assert body["metadata"]["prompt_version"] == "v1"
    assert body["metadata"]["quantization"] == "none"
    assert "Kỹ năng: Python" in generator.last_messages[1].content


def test_parse_text_uses_bounded_retry_and_maps_model_failure() -> None:
    retrying_generator = SequenceGenerator(
        [completed('{"personal_info":'), completed(json.dumps(sparse_profile()))]
    )
    app = create_app(Settings(_env_file=None), generator=retrying_generator)

    with TestClient(app) as client:
        response = client.post("/api/v1/resumes/parse-text", json={"text": "Synthetic CV"})

    assert response.status_code == 200
    assert response.json()["metadata"]["attempt_count"] == 2
    assert response.json()["metadata"]["first_attempt_status"] == "invalid_json"

    failing_generator = SequenceGenerator(
        [GenerationResult(raw_text="", completion_status=CompletionStatus.FAILED)]
    )
    failed_app = create_app(Settings(_env_file=None), generator=failing_generator)
    with TestClient(failed_app, raise_server_exceptions=False) as client:
        failed = client.post(
            "/api/v1/resumes/parse-text",
            json={"text": "Synthetic CV"},
            headers={"X-Request-ID": "req_failed"},
        )

    assert failed.status_code == 502
    assert failed.json() == {
        "request_id": "req_failed",
        "status": "error",
        "error": {"code": "GENERATION_FAILED", "message": "Model generation failed"},
    }


def test_parse_text_maps_context_overflow_to_422_without_retry() -> None:
    generator = SequenceGenerator(
        [
            GenerationResult(
                raw_text="",
                completion_status=CompletionStatus.FAILED,
                error_code="CONTEXT_LIMIT_EXCEEDED",
            )
        ]
    )
    app = create_app(Settings(_env_file=None), generator=generator)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/resumes/parse-text",
            json={"text": "Synthetic CV"},
            headers={"X-Request-ID": "req_context"},
        )

    assert response.status_code == 422
    assert response.json() == {
        "request_id": "req_context",
        "status": "error",
        "error": {
            "code": "CONTEXT_LIMIT_EXCEEDED",
            "message": "Resume prompt exceeds the model context limit",
        },
    }
    assert len(generator.calls) == 1


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": 123}, {"text": "ok", "extra": 1}])
def test_parse_text_rejects_invalid_request_contract(payload: dict[str, object]) -> None:
    app = create_app(Settings(_env_file=None))

    with TestClient(app) as client:
        response = client.post("/api/v1/resumes/parse-text", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "REQUEST_VALIDATION_ERROR"


def test_upload_rejects_unsupported_document_without_calling_model() -> None:
    app = create_app(Settings(_env_file=None))

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/resumes/parse",
            files={"file": ("resume.txt", b"plain text", "text/plain")},
        )
        generator = app.state.runtime.generator

    assert response.status_code == 415
    assert response.json()["error"]["code"] == "UNSUPPORTED_DOCUMENT_TYPE"
    assert generator.call_count == 0


def test_openapi_describes_both_parse_contracts_and_error_envelopes() -> None:
    schema = create_app(Settings(_env_file=None)).openapi()

    upload = schema["paths"]["/api/v1/resumes/parse"]["post"]
    direct_text = schema["paths"]["/api/v1/resumes/parse-text"]["post"]
    assert "multipart/form-data" in upload["requestBody"]["content"]
    assert "application/json" in direct_text["requestBody"]["content"]
    assert {"200", "400", "413", "415", "422", "502", "503"} <= set(
        upload["responses"]
    )
    error_schema = schema["components"]["schemas"]["ErrorResponse"]
    assert set(error_schema["required"]) == {"request_id", "error"}


def test_docx_upload_passes_extracted_text_to_model() -> None:
    pytest.importorskip("docx")
    from tests.fixtures.documents import make_docx_bytes

    app = create_app(Settings(_env_file=None))
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/resumes/parse",
            files={
                "file": (
                    "resume.docx",
                    make_docx_bytes(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )
        generator = app.state.runtime.generator

    assert response.status_code == 200
    assert "Nguyễn Văn A" in generator.last_messages[1].content
    assert "Kỹ năng\tPython" in generator.last_messages[1].content
    assert response.json()["metadata"]["warnings"]
