"""Strict JSON extraction from raw model output."""

import json
import re
from typing import Any

from app.core.exceptions import InvalidModelOutputError


_JSON_FENCE = re.compile(r"\A```(?:json)?[ \t]*\r?\n(?P<body>.*)\r?\n```\Z", re.DOTALL)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InvalidModelOutputError(
                f"Duplicate JSON key: {key}",
                code="DUPLICATE_JSON_KEY",
            )
        result[key] = value
    return result


def _reject_non_standard_constant(value: str) -> None:
    raise InvalidModelOutputError(
        f"Non-standard JSON constant is not allowed: {value}",
        code="NON_STANDARD_JSON",
    )


class OutputProcessor:
    """Parse exactly one object, with one optional complete JSON fence."""

    def parse(self, raw_text: str) -> dict[str, Any]:
        candidate = raw_text.strip()
        if not candidate:
            raise InvalidModelOutputError("Model output is empty", code="EMPTY_MODEL_OUTPUT")

        fence_match = _JSON_FENCE.fullmatch(candidate)
        if fence_match:
            candidate = fence_match.group("body").strip()
        elif candidate.startswith("```") or candidate.endswith("```"):
            raise InvalidModelOutputError(
                "Model output contains an incomplete or unsupported code fence",
                code="INVALID_JSON_FENCE",
            )

        try:
            parsed = json.loads(
                candidate,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_standard_constant,
            )
        except InvalidModelOutputError:
            raise
        except (json.JSONDecodeError, TypeError) as exc:
            raise InvalidModelOutputError(
                "Model output is not one complete JSON object",
                code="INVALID_JSON",
            ) from exc

        if not isinstance(parsed, dict):
            raise InvalidModelOutputError(
                "Model output root must be a JSON object",
                code="JSON_ROOT_NOT_OBJECT",
            )
        return parsed
