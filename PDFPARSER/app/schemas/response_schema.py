"""Parsing and HTTP response contracts, separate from resume data."""

from typing import Literal

from pydantic import Field

from app.schemas.resume_schema import ResumeProfile, StrictSchema


FirstAttemptStatus = Literal[
    "valid",
    "invalid_json",
    "invalid_schema",
    "generation_failed",
    "truncated",
]


class GenerationMetadata(StrictSchema):
    input_tokens: int | None
    output_tokens: int | None
    generation_ms: float | None
    completion_status: Literal["completed", "truncated", "failed"]


class ParseMetadata(StrictSchema):
    prompt_version: str
    schema_version: str
    warnings: list[str]
    attempt_count: int
    first_attempt_status: FirstAttemptStatus
    generation: GenerationMetadata


class ParseResult(StrictSchema):
    data: ResumeProfile
    metadata: ParseMetadata


class ParseTextRequest(StrictSchema):
    text: str = Field(min_length=1)


class ApiParseMetadata(StrictSchema):
    prompt_version: str
    quantization: str
    warnings: list[str]
    attempt_count: int
    first_attempt_status: FirstAttemptStatus
    generation: GenerationMetadata


class ParseSuccessResponse(StrictSchema):
    request_id: str
    status: Literal["success"] = "success"
    model: str
    schema_version: str
    data: ResumeProfile
    metadata: ApiParseMetadata


class ErrorDetail(StrictSchema):
    code: str
    message: str


class ErrorResponse(StrictSchema):
    request_id: str
    status: Literal["error"] = "error"
    error: ErrorDetail
