"""Plain-text parser.

Paragraphs are separated by blank lines. Decoding falls back from UTF-8 to a lenient
codec rather than failing, and records ``encoding_fallback`` so a mangled source is
visible in the document's flags instead of silently indexed.
"""

from __future__ import annotations

import re
from typing import Final

from fasterrag.core.parsing.models import DocumentBuilder, ParsedDocument, ParseFlag

__all__ = ["decode", "parse_plaintext"]

_PARAGRAPH_BREAK: Final = re.compile(r"\n\s*\n")


def decode(data: bytes, builder: DocumentBuilder) -> str:
    """Decode bytes as UTF-8, falling back to a lenient codec and flagging the fallback."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        builder.flag(ParseFlag.ENCODING_FALLBACK)

    return data.decode("utf-8", errors="replace")


def parse_plaintext(data: bytes, *, mime_type: str = "text/plain") -> ParsedDocument:
    """Parse plain text into paragraph blocks."""
    builder = DocumentBuilder(mime_type=mime_type, parser="plaintext")
    text = decode(data, builder)

    for paragraph in _PARAGRAPH_BREAK.split(text):
        builder.add("paragraph", paragraph)

    return builder.build()
