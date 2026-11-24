"""Application state and dependency accessors."""

from dataclasses import dataclass
from fastapi import Request

from app.core.config import InferenceMode, Settings
from app.core.exceptions import ModelNotReadyError
from app.llm.contracts import Generator
from app.llm.model_manager import ModelLoadError, ModelManager
from app.services.document_service import DocumentService
from app.services.llm_service import MockLLMService, TransformersLLMService
from app.services.parsing_service import ParsingService

@dataclass(slots=True)
class RuntimeState:
    """Mutable lifecycle state shared by controllers.

    Keeping state explicit makes liveness independent from model readiness and
    lets RP-012 reuse the model manager created during application startup.
    """

    ready: bool = False
    reason: str = "Application is starting"
    model_id: str | None = None
    generator: Generator | None = None
    parsing_service: ParsingService | None = None
    document_service: DocumentService | None = None
    model_manager: ModelManager | None = None
    model_metadata: dict[str, object] | None = None

    def start(
        self,
        settings: Settings,
        generator: Generator | None = None,
        model_manager: ModelManager | None = None,
    ) -> None:
        self.model_id = settings.resolved_model_id
        self.document_service = DocumentService(
            max_file_size_bytes=settings.max_file_size_bytes,
            max_pdf_pages=settings.max_pdf_pages,
            max_decompressed_size_bytes=settings.max_decompressed_size_bytes,
            max_text_chars=settings.input_max_chars,
        )
        if settings.inference_mode is InferenceMode.MOCK:
            self.generator = generator or MockLLMService()
            self.parsing_service = ParsingService(
                self.generator,
                retry_cap=settings.retry_cap,
                max_input_chars=settings.input_max_chars,
            )
            self.ready = True
            self.reason = "Mock inference is ready"
            return

        self.model_manager = model_manager or ModelManager(settings)
        try:
            self.model_manager.load()
        except ModelLoadError:
            self.ready = False
            self.reason = self.model_manager.failure_reason or "Real model initialization failed"
            return

        self.model_metadata = self.model_manager.metadata
        try:
            self.generator = generator or TransformersLLMService(
                self.model_manager,
                context_window_tokens=settings.context_window_tokens,
                max_new_tokens=settings.max_new_tokens,
            )
        except Exception as exc:
            self.ready = False
            self.reason = (
                "Transformers generation service initialization failed "
                f"({type(exc).__name__})"
            )
            return

        self.parsing_service = ParsingService(
            self.generator,
            retry_cap=settings.retry_cap,
            max_input_chars=settings.input_max_chars,
        )
        self.ready = True
        self.reason = "Transformers inference is ready"

    def stop(self) -> None:
        if self.model_manager is not None:
            self.model_manager.close()
        self.ready = False
        self.reason = "Application is shutting down"
        self.model_id = None
        self.generator = None
        self.parsing_service = None
        self.document_service = None
        self.model_manager = None
        self.model_metadata = None


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_runtime_state(request: Request) -> RuntimeState:
    return request.app.state.runtime


def get_parsing_service(request: Request) -> ParsingService:
    runtime = get_runtime_state(request)
    if not runtime.ready or runtime.parsing_service is None:
        raise ModelNotReadyError(runtime.reason)
    return runtime.parsing_service


def get_document_service(request: Request) -> DocumentService:
    runtime = get_runtime_state(request)
    if runtime.document_service is None:
        raise ModelNotReadyError(runtime.reason)
    return runtime.document_service
