"""Plain-text parser, and the decoding every text format shares.

Paragraphs are separated by blank lines. Decoding tries, in order: the codec a byte-order
mark names (UTF-8, UTF-16 or UTF-32), UTF-8, then Windows-1252 — the superset of Latin-1
that Windows editors write and that browsers substitute for a ``latin-1`` label. A file
read by anything after UTF-8 is flagged ``encoding_fallback`` so the guess is visible in
the document's flags. When even that fails, the bytes are decoded with replacement
characters, and a result that is mostly replacement characters and control codes is
refused with ``ENCODING_UNREADABLE`` rather than indexed: it is a binary file or a codec
this list does not know, and its blocks would embed to vectors that mean nothing
(TASK-0472).
"""

from __future__ import annotations

import codecs
import re
import unicodedata
from typing import Final

from fasterrag.core.parsing.models import DocumentBuilder, ParsedDocument, ParseFlag
from fasterrag.errors import ErrorCode, ParseError

__all__ = ["UNREADABLE_SHARE", "decode", "parse_plaintext"]

_PARAGRAPH_BREAK: Final = re.compile(r"\n\s*\n")

_BYTE_ORDER_MARKS: Final[tuple[tuple[bytes, str], ...]] = (
    (codecs.BOM_UTF32_LE, "utf-32"),
    (codecs.BOM_UTF32_BE, "utf-32"),
    (codecs.BOM_UTF8, "utf-8-sig"),
    (codecs.BOM_UTF16_LE, "utf-16"),
    (codecs.BOM_UTF16_BE, "utf-16"),
)
_LEGACY_CODEC: Final = "cp1252"
_REPLACEMENT: Final = "\ufffd"

UNREADABLE_SHARE: Final = 0.5
"""The share of replacement and control characters above which a lenient decode is refused."""


def _unreadable(character: str) -> bool:
    """Return whether a character is a decoding casualty rather than text."""
    return character == _REPLACEMENT or (
        unicodedata.category(character) == "Cc" and not character.isspace()
    )


def _readable(text: str) -> str:
    """Return a leniently decoded text, refusing one that is mostly unreadable."""
    unreadable = sum(1 for character in text if _unreadable(character))
    if text and unreadable / len(text) > UNREADABLE_SHARE:
        raise ParseError(
            f"{unreadable} of {len(text)} characters could not be decoded as text; the file "
            "is binary or in an encoding other than UTF-8, UTF-16, UTF-32 or Windows-1252",
            code=ErrorCode.ENCODING_UNREADABLE,
            retryable=False,
        )
    return text


def decode(data: bytes, builder: DocumentBuilder) -> str:
    """Decode bytes by byte-order mark, then UTF-8, then Windows-1252, flagging any guess.

    Raises:
        ParseError: ``ENCODING_UNREADABLE`` when no codec reads it and the lenient result
            is mostly replacement characters.
    """
    for mark, codec in _BYTE_ORDER_MARKS:
        if data.startswith(mark):
            try:
                return data.decode(codec)
            except UnicodeDecodeError:
                builder.flag(ParseFlag.ENCODING_FALLBACK)
                return _readable(data.decode(codec, errors="replace"))

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        builder.flag(ParseFlag.ENCODING_FALLBACK)

    try:
        text = data.decode(_LEGACY_CODEC)
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    return _readable(text)


def parse_plaintext(data: bytes, *, mime_type: str = "text/plain") -> ParsedDocument:
    """Parse plain text into paragraph blocks."""
    builder = DocumentBuilder(mime_type=mime_type, parser="plaintext")
    text = decode(data, builder)

    for paragraph in _PARAGRAPH_BREAK.split(text):
        builder.add("paragraph", paragraph)

    return builder.build()
