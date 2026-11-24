"""Synthetic documents containing no private resume data."""

from io import BytesIO


def make_docx_bytes() -> bytes:
    from docx import Document

    document = Document()
    document.add_paragraph("Nguyễn Văn A")
    table = document.add_table(rows=1, cols=2)
    table.cell(0, 0).text = "Kỹ năng"
    table.cell(0, 1).text = "Python"
    output = BytesIO()
    document.save(output)
    return output.getvalue()


def make_pdf_bytes(text: str = "Python Developer") -> bytes:
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_reference = writer._add_object(font)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {NameObject("/F1"): font_reference}
            )
        }
    )
    stream = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)

    output = BytesIO()
    writer.write(output)
    return output.getvalue()
