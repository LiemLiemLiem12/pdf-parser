from app.processors.text_processor import TextProcessor


def test_normalizes_line_endings_and_removes_invalid_controls() -> None:
    text = "Nguyễn Văn A\r\nPython\x00\x07\rKỹ năng\tFastAPI\fTrang 2"

    normalized = TextProcessor().normalize(text)

    assert normalized == "Nguyễn Văn A\nPython\nKỹ năng\tFastAPI\fTrang 2"
