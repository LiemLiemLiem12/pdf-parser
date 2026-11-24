"""Canonical Resume Profile schema, version 1.0."""

from pydantic import BaseModel, ConfigDict


SCHEMA_VERSION = "1.0"


class StrictSchema(BaseModel):
    """Base contract: no coercion and no undeclared fields."""

    model_config = ConfigDict(extra="forbid", strict=True)


class PersonalInfo(StrictSchema):
    full_name: str | None
    email: str | None
    phone: str | None
    location: str | None


class WorkExperience(StrictSchema):
    company: str | None
    job_title: str | None
    start_date: str | None
    end_date: str | None
    description: list[str]


class Education(StrictSchema):
    institution: str | None
    degree: str | None
    major: str | None
    graduation_year: str | None


class ResumeProfile(StrictSchema):
    personal_info: PersonalInfo
    summary: str | None
    work_experience: list[WorkExperience]
    education: list[Education]
    skills: list[str]
    certifications: list[str]
    languages: list[str]


def resume_json_schema() -> dict[str, object]:
    """Generate the JSON Schema from the one canonical model implementation."""

    return ResumeProfile.model_json_schema()
