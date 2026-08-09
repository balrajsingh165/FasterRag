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
    """Decode bytes as UTF-8, dropping a byte-order mark and flagging a lenient fallback.

    The codec is ``utf-8-sig`` rather than ``utf-8`` because U+FEFF is not whitespace, so
    ``str.strip`` leaves it in place and a leading BOM reaches the first block. Every parser
    decodes through here, and it broke each of them differently: ``# Title`` became a
    paragraph instead of a heading, the first CSV column became ``﻿name``, and valid
    JSON was rejected outright at line 1 column 1. Windows editors and PowerShell write a BOM
    by default, so this is the ordinary case rather than an exotic one — the same reasoning
    already applied to ``config.yaml`` in the settings loader.

    Only a leading mark is removed. U+FEFF anywhere else is a zero-width no-break space and
    belongs to the content.
    """
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        builder.flag(ParseFlag.ENCODING_FALLBACK)

    return data.decode("utf-8-sig", errors="replace")


def parse_plaintext(data: bytes, *, mime_type: str = "text/plain") -> ParsedDocument:
    """Parse plain text into paragraph blocks."""
    builder = DocumentBuilder(mime_type=mime_type, parser="plaintext")
    text = decode(data, builder)

    for paragraph in _PARAGRAPH_BREAK.split(text):
        builder.add("paragraph", paragraph)

    return builder.build()
