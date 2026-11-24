"""Resume upload and direct-text parsing endpoints."""

from fastapi import APIRouter, Depends, File, Request, UploadFile

from app.core.config import Settings
from app.core.dependencies import (
    RuntimeState,
    get_document_service,
    get_parsing_service,
    get_runtime_state,
    get_settings,
)
from app.schemas.response_schema import (
    ApiParseMetadata,
    ErrorResponse,
    ParseResult,
    ParseSuccessResponse,
    ParseTextRequest,
)
from app.services.document_service import DocumentService
from app.services.parsing_service import ParsingService


router = APIRouter(prefix="/api/v1/resumes", tags=["resumes"])

_FILE_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid or corrupt document"},
    413: {"model": ErrorResponse, "description": "Upload exceeds the configured size limit"},
    415: {"model": ErrorResponse, "description": "Unsupported document type"},
    422: {"model": ErrorResponse, "description": "Invalid request or no usable text"},
    502: {"model": ErrorResponse, "description": "Model generation or output failure"},
    503: {"model": ErrorResponse, "description": "Parser or extractor is not ready"},
}
_TEXT_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid resume text"},
    422: {"model": ErrorResponse, "description": "Invalid request or context limit"},
    502: {"model": ErrorResponse, "description": "Model generation or output failure"},
    503: {"model": ErrorResponse, "description": "Parser is not ready"},
}


def _success_response(
    request: Request,
    *,
    settings: Settings,
    runtime: RuntimeState,
    result: ParseResult,
    extraction_warnings: tuple[str, ...] = (),
) -> ParseSuccessResponse:
    metadata = result.metadata
    return ParseSuccessResponse(
        request_id=request.state.request_id,
        model=runtime.model_id or settings.resolved_model_id,
        schema_version=metadata.schema_version,
        data=result.data,
        metadata=ApiParseMetadata(
            prompt_version=metadata.prompt_version,
            quantization=settings.quantization,
            warnings=[*extraction_warnings, *metadata.warnings],
            attempt_count=metadata.attempt_count,
            first_attempt_status=metadata.first_attempt_status,
            generation=metadata.generation,
        ),
    )


@router.post(
    "/parse",
    response_model=ParseSuccessResponse,
    responses=_FILE_ERROR_RESPONSES,
)
def parse_resume_file(
    request: Request,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    runtime: RuntimeState = Depends(get_runtime_state),
    document_service: DocumentService = Depends(get_document_service),
    parsing_service: ParsingService = Depends(get_parsing_service),
) -> ParseSuccessResponse:
    file_bytes = file.file.read(settings.max_file_size_bytes + 1)
    document = document_service.extract(
        file_bytes,
        filename=file.filename,
        content_type=file.content_type,
    )
    result = parsing_service.parse(document.text)
    return _success_response(
        request,
        settings=settings,
        runtime=runtime,
        result=result,
        extraction_warnings=document.warnings,
    )


@router.post(
    "/parse-text",
    response_model=ParseSuccessResponse,
    responses=_TEXT_ERROR_RESPONSES,
)
def parse_resume_text(
    payload: ParseTextRequest,
    request: Request,
    settings: Settings = Depends(get_settings),
    runtime: RuntimeState = Depends(get_runtime_state),
    parsing_service: ParsingService = Depends(get_parsing_service),
) -> ParseSuccessResponse:
    result = parsing_service.parse(payload.text)
    return _success_response(
        request,
        settings=settings,
        runtime=runtime,
        result=result,
    )
