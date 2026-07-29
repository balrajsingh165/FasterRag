"""Parsing pipeline: turn a source file into a structured :class:`ParsedDocument`.

Dispatch is by file extension, and every parser returns the same structure regardless of
format, so chunking never learns where a document came from. Parsers are pure functions
over bytes, which is what lets the same code path serve a file on disk, a fetched URL,
and an inline payload.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Protocol

from fasterrag.core.parsing.docx import parse_docx
from fasterrag.core.parsing.html import parse_html
from fasterrag.core.parsing.markdown import parse_markdown
from fasterrag.core.parsing.models import (
    Block,
    BlockKind,
    DocumentBuilder,
    ParsedDocument,
    ParseFlag,
)
from fasterrag.core.parsing.pdf import parse_pdf
from fasterrag.core.parsing.plaintext import parse_plaintext
from fasterrag.core.parsing.tabular import parse_csv, parse_json
from fasterrag.errors import ErrorCode, IngestionError, ParseError

__all__ = [
    "SUPPORTED_SUFFIXES",
    "Block",
    "BlockKind",
    "DocumentBuilder",
    "ParseFlag",
    "ParsedDocument",
    "parse_bytes",
    "parse_csv",
    "parse_document",
    "parse_docx",
    "parse_html",
    "parse_json",
    "parse_markdown",
    "parse_pdf",
    "parse_plaintext",
]

_DOCX_MIME: Final = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


class FormatParser(Protocol):
    """The signature every format parser implements."""

    def __call__(self, data: bytes, *, mime_type: str) -> ParsedDocument:
        """Parse ``data`` into a structured document."""
        ...


_PARSERS: Final[dict[str, tuple[FormatParser, str]]] = {
    ".txt": (parse_plaintext, "text/plain"),
    ".text": (parse_plaintext, "text/plain"),
    ".log": (parse_plaintext, "text/plain"),
    ".md": (parse_markdown, "text/markdown"),
    ".markdown": (parse_markdown, "text/markdown"),
    ".html": (parse_html, "text/html"),
    ".htm": (parse_html, "text/html"),
    ".xhtml": (parse_html, "application/xhtml+xml"),
    ".csv": (parse_csv, "text/csv"),
    ".tsv": (parse_csv, "text/tab-separated-values"),
    ".json": (parse_json, "application/json"),
    ".docx": (parse_docx, _DOCX_MIME),
    ".pdf": (parse_pdf, "application/pdf"),
}

SUPPORTED_SUFFIXES: Final[frozenset[str]] = frozenset(_PARSERS)


def parse_bytes(
    data: bytes,
    *,
    filename: str,
    max_bytes: int | None = None,
) -> ParsedDocument:
    """Parse in-memory content, choosing the parser from ``filename``.

    Args:
        data: The raw document bytes.
        filename: Name the content came from; only its extension is used.
        max_bytes: Optional size ceiling, normally ``ingestion.max_document_mb``.

    Returns:
        The structured document.

    Raises:
        IngestionError: With ``PAYLOAD_TOO_LARGE`` when the content exceeds ``max_bytes``.
        ParseError: If the extension has no parser or the content cannot be parsed.
    """
    if max_bytes is not None and len(data) > max_bytes:
        raise IngestionError(
            f"{filename} is {len(data)} bytes, above the configured limit of {max_bytes}",
            code=ErrorCode.PAYLOAD_TOO_LARGE,
        )

    suffix = Path(filename).suffix.lower()
    registered = _PARSERS.get(suffix)
    if registered is None:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ParseError(
            f"no parser is registered for {suffix or filename!r}; supported types: {supported}"
        )

    parser, mime_type = registered
    return parser(data, mime_type=mime_type)


def parse_document(source: str | Path, *, max_bytes: int | None = None) -> ParsedDocument:
    """Read and parse a file from disk.

    Args:
        source: Path to the document.
        max_bytes: Optional size ceiling, normally ``ingestion.max_document_mb``.

    Returns:
        The structured document.

    Raises:
        IngestionError: With ``PAYLOAD_TOO_LARGE`` when the file exceeds ``max_bytes``.
        ParseError: If the file is missing, unreadable, or of an unsupported type.
    """
    path = Path(source)
    try:
        data = path.read_bytes()
    except FileNotFoundError as exc:
        raise ParseError(f"{path} does not exist") from exc
    except OSError as exc:
        raise ParseError(f"{path} could not be read: {exc.strerror}") from exc

    return parse_bytes(data, filename=path.name, max_bytes=max_bytes)
