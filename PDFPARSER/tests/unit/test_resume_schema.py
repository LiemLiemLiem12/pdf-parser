import pytest
from pydantic import ValidationError

from app.schemas.resume_schema import ResumeProfile, resume_json_schema
from tests.fixtures import complete_profile, sparse_profile


def test_sparse_and_complete_profiles_are_valid() -> None:
    sparse = ResumeProfile.model_validate(sparse_profile(), strict=True)
    complete = ResumeProfile.model_validate(complete_profile(), strict=True)

    assert sparse.skills == []
    assert len(complete.work_experience) == 2
    assert complete.work_experience[1].company == "Công ty B"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda data: data.pop("skills"),
        lambda data: data.update({"skills": "Python"}),
        lambda data: data.update({"unknown": "value"}),
        lambda data: data["personal_info"].update({"unknown": "value"}),
    ],
)
def test_missing_wrong_and_extra_fields_are_rejected(mutation) -> None:  # type: ignore[no-untyped-def]
    data = sparse_profile()
    mutation(data)

    with pytest.raises(ValidationError):
        ResumeProfile.model_validate(data, strict=True)


def test_json_schema_is_generated_from_canonical_model() -> None:
    schema = resume_json_schema()

    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(ResumeProfile.model_fields)
