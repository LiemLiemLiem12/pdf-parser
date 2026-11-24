"""Typed application errors and stable API error codes."""

from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any


@dataclass(slots=True)
class AppError(Exception):
    code: str
    message: str
    status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


class ModelNotReadyError(AppError):
    def __init__(self, message: str = "Model is not ready") -> None:
        super().__init__(
            code="MODEL_NOT_READY",
            message=message,
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )


class InvalidInputError(AppError):
    def __init__(self, message: str = "Invalid input") -> None:
        super().__init__(
            code="INVALID_INPUT",
            message=message,
            status_code=HTTPStatus.BAD_REQUEST,
        )


class InvalidDocumentError(AppError):
    def __init__(self, message: str = "The document is invalid or corrupt") -> None:
        super().__init__(
            code="INVALID_DOCUMENT",
            message=message,
            status_code=HTTPStatus.BAD_REQUEST,
        )


class FileTooLargeError(AppError):
    def __init__(self, message: str = "The uploaded file is too large") -> None:
        super().__init__(
            code="FILE_TOO_LARGE",
            message=message,
            status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
        )


class UnsupportedDocumentTypeError(AppError):
    def __init__(self, message: str = "Only PDF and DOCX documents are supported") -> None:
        super().__init__(
            code="UNSUPPORTED_DOCUMENT_TYPE",
            message=message,
            status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
        )


class DocumentLimitError(AppError):
    def __init__(self, message: str = "The document exceeds a processing limit") -> None:
        super().__init__(
            code="DOCUMENT_LIMIT_EXCEEDED",
            message=message,
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )


class NoExtractableTextError(AppError):
    def __init__(self, message: str = "The document contains no extractable text") -> None:
        super().__init__(
            code="NO_EXTRACTABLE_TEXT",
            message=message,
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        )


class DocumentExtractorUnavailableError(AppError):
    def __init__(self) -> None:
        super().__init__(
            code="DOCUMENT_EXTRACTOR_UNAVAILABLE",
            message="Document extraction dependencies are unavailable",
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        )


class InvalidModelOutputError(AppError):
    def __init__(
        self,
        message: str,
        *,
        code: str = "INVALID_MODEL_OUTPUT",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code=code,
            message=message,
            status_code=HTTPStatus.BAD_GATEWAY,
            context=context or {},
        )


class ResumeSchemaValidationError(InvalidModelOutputError):
    def __init__(
        self,
        message: str = "Model output does not match the resume schema",
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            code="INVALID_RESUME_SCHEMA",
            context=context,
        )


class GenerationFailedError(AppError):
    def __init__(
        self,
        message: str = "Model generation failed",
        *,
        code: str = "GENERATION_FAILED",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code=code,
            message=message,
            status_code=HTTPStatus.BAD_GATEWAY,
            context=context or {},
        )
