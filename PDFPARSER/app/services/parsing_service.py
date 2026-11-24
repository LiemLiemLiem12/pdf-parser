"""Orchestrate prompt construction, generation, parsing and validation."""

from app.core.exceptions import (
    GenerationFailedError,
    InvalidInputError,
    InvalidModelOutputError,
    ResumeSchemaValidationError,
)
from app.llm.contracts import CompletionStatus, GenerationResult, Generator
from app.processors.output_processor import OutputProcessor
from app.prompts.prompt_builder import PromptBuilder
from app.schemas.response_schema import (
    FirstAttemptStatus,
    GenerationMetadata,
    ParseMetadata,
    ParseResult,
)
from app.schemas.resume_schema import SCHEMA_VERSION
from app.validators.resume_validator import ResumeValidator


class ParsingService:
    def __init__(
        self,
        generator: Generator,
        *,
        prompt_builder: PromptBuilder | None = None,
        output_processor: OutputProcessor | None = None,
        validator: ResumeValidator | None = None,
        retry_cap: int = 1,
        max_input_chars: int | None = None,
    ) -> None:
        if retry_cap not in (0, 1):
            raise ValueError("retry_cap must be 0 or 1")
        self._generator = generator
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._output_processor = output_processor or OutputProcessor()
        self._validator = validator or ResumeValidator()
        self._retry_cap = retry_cap
        self._max_input_chars = max_input_chars

    def parse(self, text: str, *, allow_retry: bool = True) -> ParseResult:
        if not text.strip():
            raise InvalidInputError("Resume text must not be empty")
        if self._max_input_chars is not None and len(text) > self._max_input_chars:
            raise InvalidInputError("Resume text exceeds the configured input limit")

        messages = self._prompt_builder.build(text)
        attempt_count = 0
        first_attempt_status: FirstAttemptStatus | None = None

        while True:
            attempt_count += 1
            generation = self._generate(messages, attempt_count, first_attempt_status)

            if generation.completion_status is CompletionStatus.FAILED:
                raise GenerationFailedError(
                    "Model generation failed",
                    code=generation.error_code or "GENERATION_FAILED",
                    context={
                        "attempt_count": attempt_count,
                        "first_attempt_status": first_attempt_status or "generation_failed",
                    },
                )
            if generation.completion_status is CompletionStatus.TRUNCATED:
                raise GenerationFailedError(
                    "Model generation was truncated",
                    code="GENERATION_TRUNCATED",
                    context={
                        "attempt_count": attempt_count,
                        "first_attempt_status": first_attempt_status or "truncated",
                    },
                )

            try:
                payload = self._output_processor.parse(generation.raw_text)
                validation = self._validator.validate(payload)
            except (InvalidModelOutputError, ResumeSchemaValidationError) as exc:
                status: FirstAttemptStatus = (
                    "invalid_schema"
                    if isinstance(exc, ResumeSchemaValidationError)
                    else "invalid_json"
                )
                if first_attempt_status is None:
                    first_attempt_status = status

                can_retry = allow_retry and attempt_count <= self._retry_cap
                if can_retry:
                    messages = self._prompt_builder.build_correction(
                        messages,
                        generation.raw_text,
                        exc.code,
                    )
                    continue

                exc.context.update(
                    {
                        "attempt_count": attempt_count,
                        "first_attempt_status": first_attempt_status,
                    }
                )
                raise

            if first_attempt_status is None:
                first_attempt_status = "valid"
            return ParseResult(
                data=validation.profile,
                metadata=ParseMetadata(
                    prompt_version=self._prompt_builder.version,
                    schema_version=SCHEMA_VERSION,
                    warnings=list(validation.warnings),
                    attempt_count=attempt_count,
                    first_attempt_status=first_attempt_status,
                    generation=GenerationMetadata(
                        input_tokens=generation.input_tokens,
                        output_tokens=generation.output_tokens,
                        generation_ms=generation.generation_ms,
                        completion_status=generation.completion_status.value,
                    ),
                ),
            )

    def _generate(
        self,
        messages,  # type: ignore[no-untyped-def]
        attempt_count: int,
        first_attempt_status: FirstAttemptStatus | None,
    ) -> GenerationResult:
        try:
            return self._generator.generate(messages)
        except Exception as exc:
            raise GenerationFailedError(
                context={
                    "attempt_count": attempt_count,
                    "first_attempt_status": first_attempt_status or "generation_failed",
                }
            ) from exc
