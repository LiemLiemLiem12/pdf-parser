"""FastAPI application factory for the Resume Parser."""

import logging
import re
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.controllers.health_controller import router as health_router
from app.controllers.resume_controller import router as resume_router
from app.core.config import Settings
from app.core.dependencies import RuntimeState
from app.core.exceptions import AppError
from app.core.logging import configure_logging
from app.llm.contracts import Generator
from app.llm.model_manager import ModelManager


logger = logging.getLogger(__name__)
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", f"req_{uuid4().hex}")


def _select_request_id(candidate: str | None) -> str:
    if candidate and _SAFE_REQUEST_ID.fullmatch(candidate.strip()):
        return candidate.strip()
    return f"req_{uuid4().hex}"


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "request_id": _request_id(request),
            "status": "error",
            "error": {"code": code, "message": message},
        },
    )


def create_app(
    settings: Settings | None = None,
    *,
    generator: Generator | None = None,
    model_manager: ModelManager | None = None,
) -> FastAPI:
    """Build an isolated application instance.

    Passing settings explicitly is intended for deterministic tests. Normal
    execution loads them from the environment when this factory is called.
    """

    app_settings = settings or Settings()
    configure_logging(app_settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        runtime = RuntimeState()
        app.state.runtime = runtime
        logger.info(
            "application_starting",
            extra={"inference_mode": app_settings.inference_mode.value},
        )
        runtime.start(
            app_settings,
            generator=generator,
            model_manager=model_manager,
        )
        try:
            yield
        finally:
            runtime.stop()
            logger.info("application_stopped")

    app = FastAPI(
        title=app_settings.app_name,
        version=app_settings.app_version,
        lifespan=lifespan,
    )
    app.state.settings = app_settings

    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        configured_header = app_settings.request_id_header
        request.state.request_id = _select_request_id(request.headers.get(configured_header))
        started_at = time.perf_counter()
        response = await call_next(request)
        response.headers[configured_header] = request.state.request_id
        logger.info(
            "request_completed",
            extra={
                "request_id": request.state.request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
        )
        return response

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        logger.warning(
            "application_error",
            extra={"request_id": _request_id(request), "error_code": exc.code},
        )
        return _error_response(
            request,
            status_code=exc.status_code,
            code=exc.code,
            message=exc.message,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        logger.warning(
            "request_validation_error",
            extra={"request_id": _request_id(request), "error_count": len(exc.errors())},
        )
        return _error_response(
            request,
            status_code=422,
            code="REQUEST_VALIDATION_ERROR",
            message="Request validation failed",
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        message = exc.detail if isinstance(exc.detail, str) else "HTTP request failed"
        return _error_response(
            request,
            status_code=exc.status_code,
            code="HTTP_ERROR",
            message=message,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception(
            "unexpected_application_error",
            extra={"request_id": _request_id(request)},
        )
        return _error_response(
            request,
            status_code=500,
            code="INTERNAL_ERROR",
            message="An unexpected internal error occurred",
        )

    app.include_router(health_router)
    app.include_router(resume_router)
    return app


app = create_app()
