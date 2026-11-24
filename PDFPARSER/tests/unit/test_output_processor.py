import pytest

from app.core.exceptions import InvalidModelOutputError
from app.processors.output_processor import OutputProcessor


def test_parses_plain_and_one_complete_json_fence() -> None:
    processor = OutputProcessor()

    assert processor.parse('{"value": 1}') == {"value": 1}
    assert processor.parse('```json\n{"value": 1}\n```') == {"value": 1}


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ('{"value": 1} trailing', "INVALID_JSON"),
        ('{"a": 1}{"b": 2}', "INVALID_JSON"),
        ('{"a": 1, "a": 2}', "DUPLICATE_JSON_KEY"),
        ('{"value": NaN}', "NON_STANDARD_JSON"),
        ('{"value":', "INVALID_JSON"),
        ('```json\n{"value": 1}', "INVALID_JSON_FENCE"),
        ('[1, 2]', "JSON_ROOT_NOT_OBJECT"),
    ],
)
def test_rejects_unsupported_or_ambiguous_output(raw: str, code: str) -> None:
    with pytest.raises(InvalidModelOutputError) as captured:
        OutputProcessor().parse(raw)

    assert captured.value.code == code
