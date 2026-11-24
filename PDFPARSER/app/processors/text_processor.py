"""Conservative normalization for text extracted from resumes."""

import unicodedata


class TextProcessor:
    """Remove unsafe controls without changing resume language or structure."""

    _PRESERVED_CONTROLS = {"\n", "\t", "\f"}

    def normalize(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        cleaned = "".join(
            character
            for character in normalized
            if character in self._PRESERVED_CONTROLS
            or unicodedata.category(character) not in {"Cc", "Cs"}
        )
        return cleaned.strip()
