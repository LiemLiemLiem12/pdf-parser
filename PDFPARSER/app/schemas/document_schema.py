"""Contracts returned by document extraction."""

from dataclasses import dataclass
from typing import Literal


DocumentFormat = Literal["pdf", "docx"]


@dataclass(frozen=True, slots=True)
class DocumentText:
    text: str
    format: DocumentFormat
    page_count: int | None
    warnings: tuple[str, ...] = ()
