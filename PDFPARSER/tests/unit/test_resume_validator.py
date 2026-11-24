import pytest

from app.core.exceptions import ResumeSchemaValidationError
from app.validators.resume_validator import ResumeValidator
from tests.fixtures import sparse_profile


def test_validator_returns_deterministic_warnings() -> None:
    data = sparse_profile()
    data["personal_info"]["email"] = "not-an-email"
    data["work_experience"] = [
        {
            "company": "Example",
            "job_title": "Engineer",
            "start_date": "2024-02",
            "end_date": "2023-12",
            "description": [],
        }
    ]

    result = ResumeValidator().validate(data)

    assert result.warnings == (
        "personal_info.email has a suspicious format",
        "work_experience[0] start_date is after end_date",
    )


def test_validator_does_not_coerce_wrong_types() -> None:
    data = sparse_profile()
    data["skills"] = "Python"

    with pytest.raises(ResumeSchemaValidationError) as captured:
        ResumeValidator().validate(data)

    assert captured.value.context["validation_error_count"] == 1


def test_date_warning_respects_source_precision() -> None:
    data = sparse_profile()
    data["work_experience"] = [
        {
            "company": None,
            "job_title": None,
            "start_date": "2024-12",
            "end_date": "2024",
            "description": [],
        }
    ]

    result = ResumeValidator().validate(data)

    assert result.warnings == ()
