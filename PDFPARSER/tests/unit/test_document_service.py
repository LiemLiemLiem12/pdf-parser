from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from app.core.exceptions import (
    DocumentLimitError,
    FileTooLargeError,
    InvalidDocumentError,
    NoExtractableTextError,
    UnsupportedDocumentTypeError,
)
from app.services.document_service import DocumentService


def service(**overrides) -> DocumentService:  # type: ignore[no-untyped-def]
    options = {
        "max_file_size_bytes": 10_000,
        "max_pdf_pages": 3,
        "max_decompressed_size_bytes": 20_000,
        "max_text_chars": 5_000,
    }
    options.update(overrides)
    return DocumentService(**options)


def test_rejects_empty_oversized_and_unsupported_files() -> None:
    with pytest.raises(InvalidDocumentError):
        service().extract(b"", filename="resume.pdf", content_type="application/pdf")
    with pytest.raises(FileTooLargeError):
        service(max_file_size_bytes=3).extract(b"%PDF-oversized", filename="resume.pdf")
    with pytest.raises(UnsupportedDocumentTypeError):
        service().extract(b"plain text", filename="resume.txt", content_type="text/plain")


def test_rejects_extension_or_media_type_that_conflicts_with_signature() -> None:
    with pytest.raises(InvalidDocumentError):
        service().extract(b"%PDF-not-valid", filename="resume.docx")
    with pytest.raises(InvalidDocumentError):
        service().extract(b"%PDF-not-valid", filename="resume.pdf", content_type="text/plain")


def test_bounds_docx_expanded_size_before_parsing() -> None:
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", "x")
        archive.writestr("word/document.xml", "x" * 2_000)

    with pytest.raises(DocumentLimitError):
        service(max_decompressed_size_bytes=100).extract(
            output.getvalue(),
            filename="resume.docx",
            content_type=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            ),
        )


def test_extracts_docx_paragraphs_and_tables_in_order() -> None:
    pytest.importorskip("docx")
    from tests.fixtures.documents import make_docx_bytes

    result = service(
        max_file_size_bytes=100_000,
        max_decompressed_size_bytes=2_000_000,
    ).extract(
          make_docx_bytes(),
        filename="resume.docx",
        content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )

    assert result.format == "docx"
    assert result.page_count is None
    assert result.text == "Nguyễn Văn A\nKỹ năng\tPython"
    assert result.warnings


def test_extracts_pdf_text_and_rejects_textless_pdf() -> None:
    pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    from tests.fixtures.documents import make_pdf_bytes

    parsed = service(max_file_size_bytes=100_000).extract(
        make_pdf_bytes(), filename="resume.pdf", content_type="application/pdf"
    )
    assert parsed.format == "pdf"
    assert parsed.page_count == 1
    assert "Python Developer" in parsed.text

    output = BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(output)
    with pytest.raises(NoExtractableTextError):
        service(max_file_size_bytes=100_000).extract(
            output.getvalue(), filename="resume.pdf", content_type="application/pdf"
        )
