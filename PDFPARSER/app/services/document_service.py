"""Bounded PDF and DOCX text extraction."""

from io import BytesIO
from pathlib import Path
from collections.abc import Iterator
from typing import Any
from zipfile import BadZipFile, ZipFile, is_zipfile

from app.core.exceptions import (
    DocumentExtractorUnavailableError,
    DocumentLimitError,
    FileTooLargeError,
    InvalidDocumentError,
    NoExtractableTextError,
    UnsupportedDocumentTypeError,
)
from app.processors.text_processor import TextProcessor
from app.schemas.document_schema import DocumentFormat, DocumentText


_DOCX_REQUIRED_MEMBERS = {"[Content_Types].xml", "word/document.xml"}
_PDF_CONTENT_TYPES = {"application/pdf"}
_DOCX_CONTENT_TYPES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
}


class DocumentService:
    def __init__(
        self,
        *,
        max_file_size_bytes: int,
        max_pdf_pages: int,
        max_decompressed_size_bytes: int,
        max_text_chars: int,
        text_processor: TextProcessor | None = None,
    ) -> None:
        self._max_file_size_bytes = max_file_size_bytes
        self._max_pdf_pages = max_pdf_pages
        self._max_decompressed_size_bytes = max_decompressed_size_bytes
        self._max_text_chars = max_text_chars
        self._text_processor = text_processor or TextProcessor()

    def extract(
        self,
        file_bytes: bytes,
        *,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> DocumentText:
        if not file_bytes:
            raise InvalidDocumentError("The uploaded file is empty")
        if len(file_bytes) > self._max_file_size_bytes:
            raise FileTooLargeError()

        document_format = self._detect_format(file_bytes, filename, content_type)
        if document_format == "pdf":
            result = self._extract_pdf(file_bytes)
        else:
            result = self._extract_docx(file_bytes)

        normalized = self._text_processor.normalize(result.text)
        if not normalized:
            raise NoExtractableTextError()
        if len(normalized) > self._max_text_chars:
            raise DocumentLimitError("Extracted text exceeds the configured input limit")
        return DocumentText(
            text=normalized,
            format=result.format,
            page_count=result.page_count,
            warnings=result.warnings,
        )

    def _detect_format(
        self,
        file_bytes: bytes,
        filename: str | None,
        content_type: str | None,
    ) -> DocumentFormat:
        suffix = Path(filename or "").suffix.lower()
        media_type = (content_type or "").split(";", maxsplit=1)[0].strip().lower()

        if file_bytes.startswith(b"%PDF-"):
            if suffix and suffix != ".pdf":
                raise InvalidDocumentError("The filename does not match the PDF content")
            if media_type and media_type not in _PDF_CONTENT_TYPES | {"application/octet-stream"}:
                raise InvalidDocumentError("The content type does not match the PDF content")
            return "pdf"

        if is_zipfile(BytesIO(file_bytes)):
            try:
                with ZipFile(BytesIO(file_bytes)) as archive:
                    members = set(archive.namelist())
            except BadZipFile as exc:
                raise InvalidDocumentError() from exc
            if _DOCX_REQUIRED_MEMBERS.issubset(members):
                if suffix and suffix != ".docx":
                    raise InvalidDocumentError("The filename does not match the DOCX content")
                if media_type and media_type not in _DOCX_CONTENT_TYPES | {
                    "application/octet-stream"
                }:
                    raise InvalidDocumentError("The content type does not match the DOCX content")
                return "docx"

        if suffix not in {".pdf", ".docx"} and media_type not in (
            _PDF_CONTENT_TYPES | _DOCX_CONTENT_TYPES
        ):
            raise UnsupportedDocumentTypeError()
        raise InvalidDocumentError("The file signature does not match a valid PDF or DOCX document")

    def _extract_pdf(self, file_bytes: bytes) -> DocumentText:
        try:
            from pypdf import PdfReader
            from pypdf.errors import PdfReadError
        except ModuleNotFoundError as exc:
            raise DocumentExtractorUnavailableError() from exc

        try:
            reader = PdfReader(BytesIO(file_bytes), strict=False)
            if reader.is_encrypted:
                raise InvalidDocumentError("Encrypted PDF documents are not supported")
            page_count = len(reader.pages)
            if page_count > self._max_pdf_pages:
                raise DocumentLimitError(
                    f"PDF exceeds the configured limit of {self._max_pdf_pages} pages"
                )
            pages = [(page.extract_text() or "").strip() for page in reader.pages]
        except (InvalidDocumentError, DocumentLimitError):
            raise
        except (PdfReadError, ValueError, TypeError, OSError) as exc:
            raise InvalidDocumentError("The PDF document is invalid or corrupt") from exc

        warnings: list[str] = []
        empty_pages = sum(not page for page in pages)
        if empty_pages and empty_pages < page_count:
            warnings.append(f"{empty_pages} PDF page(s) contained no extractable text")
        return DocumentText(
            text="\n\n\f\n\n".join(pages),
            format="pdf",
            page_count=page_count,
            warnings=tuple(warnings),
        )

    def _extract_docx(self, file_bytes: bytes) -> DocumentText:
        self._validate_docx_archive(file_bytes)
        try:
            from docx import Document
            from docx.opc.exceptions import PackageNotFoundError

            document = Document(BytesIO(file_bytes))
            blocks = list(self._iter_docx_blocks(document))
        except ModuleNotFoundError as exc:
            raise DocumentExtractorUnavailableError() from exc
        except (BadZipFile, KeyError, ValueError, OSError, PackageNotFoundError) as exc:
            raise InvalidDocumentError("The DOCX document is invalid or corrupt") from exc

        return DocumentText(
            text="\n".join(block for block in blocks if block.strip()),
            format="docx",
            page_count=None,
            warnings=(
                "DOCX headers, footers, text boxes and nested content may not preserve document order",
            ),
        )

    def _validate_docx_archive(self, file_bytes: bytes) -> None:
        try:
            with ZipFile(BytesIO(file_bytes)) as archive:
                infos = archive.infolist()
                names = {info.filename for info in infos}
                if not _DOCX_REQUIRED_MEMBERS.issubset(names):
                    raise InvalidDocumentError("The DOCX container is missing required content")
                expanded_size = sum(info.file_size for info in infos)
                if expanded_size > self._max_decompressed_size_bytes:
                    raise DocumentLimitError("DOCX expanded content exceeds the configured limit")
        except (InvalidDocumentError, DocumentLimitError):
            raise
        except BadZipFile as exc:
            raise InvalidDocumentError("The DOCX document is invalid or corrupt") from exc

    @staticmethod
    def _iter_docx_blocks(document: Any) -> Iterator[str]:
        from docx.oxml.table import CT_Tbl
        from docx.oxml.text.paragraph import CT_P
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        for child in document.element.body.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, document).text
            elif isinstance(child, CT_Tbl):
                table = Table(child, document)
                for row in table.rows:
                    yield "\t".join(cell.text for cell in row.cells)
