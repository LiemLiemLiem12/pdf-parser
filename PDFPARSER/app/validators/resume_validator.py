"""Strict schema validation with conservative deterministic warnings."""

import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from typing import Any

from pydantic import ValidationError

from app.core.exceptions import ResumeSchemaValidationError
from app.schemas.resume_schema import ResumeProfile


_EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
_DATE = re.compile(r"^(?P<year>\d{4})(?:-(?P<month>0[1-9]|1[0-2]))?(?:-(?P<day>0[1-9]|[12]\d|3[01]))?$")


@dataclass(frozen=True, slots=True)
class ValidationResult:
    profile: ResumeProfile
    warnings: tuple[str, ...]


def _date_bounds(value: str) -> tuple[date, date] | None:
    match = _DATE.fullmatch(value)
    if not match:
        return None
    year = int(match.group("year"))
    month_text = match.group("month")
    day_text = match.group("day")
    try:
        if month_text is None:
            return date(year, 1, 1), date(year, 12, 31)
        month = int(month_text)
        if day_text is None:
            return date(year, month, 1), date(year, month, monthrange(year, month)[1])
        exact = date(year, month, int(day_text))
        return exact, exact
    except ValueError:
        return None


class ResumeValidator:
    def validate(self, payload: dict[str, Any]) -> ValidationResult:
        try:
            profile = ResumeProfile.model_validate(payload, strict=True)
        except ValidationError as exc:
            raise ResumeSchemaValidationError(
                context={"validation_error_count": exc.error_count()}
            ) from exc

        warnings: list[str] = []
        email = profile.personal_info.email
        if email is not None and not _EMAIL.fullmatch(email):
            warnings.append("personal_info.email has a suspicious format")

        for index, experience in enumerate(profile.work_experience):
            if experience.start_date is None or experience.end_date is None:
                continue
            start_bounds = _date_bounds(experience.start_date)
            end_bounds = _date_bounds(experience.end_date)
            if (
                start_bounds is not None
                and end_bounds is not None
                and start_bounds[0] > end_bounds[1]
            ):
                warnings.append(f"work_experience[{index}] start_date is after end_date")

        return ValidationResult(profile=profile, warnings=tuple(warnings))
