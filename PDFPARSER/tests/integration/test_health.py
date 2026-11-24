from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.llm_service import MockLLMService


class LoadedModelManager:
    def __init__(self) -> None:
        self.load_calls = 0
        self.close_calls = 0
        self.failure_reason = None
        self.metadata = {
            "alias": "qwen14b",
            "model_id": "Qwen/Qwen2.5-14B-Instruct",
        }

    def load(self) -> None:
        self.load_calls += 1

    def close(self) -> None:
        self.close_calls += 1


def test_mock_app_is_live_and_ready_with_request_id() -> None:
    app = create_app(Settings(_env_file=None))

    with TestClient(app) as client:
        health = client.get("/api/v1/health", headers={"X-Request-ID": "req_test"})
        ready = client.get("/api/v1/ready")

    assert health.status_code == 200
    assert health.json() == {
        "status": "ok",
        "service": "Resume Parser",
        "version": "0.1.0",
    }
    assert health.headers["X-Request-ID"] == "req_test"
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "model": "mock",
        "inference_mode": "mock",
    }


def test_real_mode_is_live_but_not_ready_when_model_initialization_fails() -> None:
    settings = Settings(
        _env_file=None,
        inference_mode="transformers",
        model_alias="qwen14b",
        quantization="nf4-4bit",
    )
    app = create_app(settings)

    with TestClient(app, raise_server_exceptions=False) as client:
        health = client.get("/api/v1/health")
        ready = client.get("/api/v1/ready", headers={"X-Request-ID": "req_not_ready"})

    assert health.status_code == 200
    assert ready.status_code == 503
    assert ready.json() == {
        "request_id": "req_not_ready",
        "status": "error",
        "error": {
            "code": "MODEL_NOT_READY",
            "message": "Model initialization failed (ModelLoadError)",
        },
    }


def test_loaded_manager_is_created_once_and_released_by_lifespan() -> None:
    settings = Settings(
        _env_file=None,
        inference_mode="transformers",
        model_alias="qwen14b",
        quantization="nf4-4bit",
    )
    manager = LoadedModelManager()
    app = create_app(
        settings,
        model_manager=manager,  # type: ignore[arg-type]
        generator=MockLLMService(),
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        first = client.get("/api/v1/ready")
        second = client.get("/api/v1/ready")
        assert app.state.runtime.model_metadata == manager.metadata

    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json() == {
        "status": "ready",
        "model": "Qwen/Qwen2.5-14B-Instruct",
        "inference_mode": "transformers",
    }
    assert manager.load_calls == 1
    assert manager.close_calls == 1


def test_unknown_route_uses_consistent_error_contract() -> None:
    app = create_app(Settings(_env_file=None))

    with TestClient(app) as client:
        response = client.get("/missing", headers={"X-Request-ID": "req_missing"})

    assert response.status_code == 404
    assert response.json() == {
        "request_id": "req_missing",
        "status": "error",
        "error": {"code": "HTTP_ERROR", "message": "Not Found"},
    }


def test_unsafe_request_id_is_replaced() -> None:
    app = create_app(Settings(_env_file=None))

    with TestClient(app) as client:
        response = client.get("/api/v1/health", headers={"X-Request-ID": "unsafe value"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"].startswith("req_")
    assert response.headers["X-Request-ID"] != "unsafe value"
