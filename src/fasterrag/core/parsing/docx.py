"""DOCX parser.

Walks the document body in XML order so paragraphs and tables interleave the way they do
on the page — iterating paragraphs and tables separately, as the obvious implementation
does, silently destroys reading order. Word's outline styles give real heading levels.
"""

from __future__ import annotations

import io
from typing import Final

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from fasterrag.core.parsing.models import DocumentBuilder, ParsedDocument, ParseFlag

__all__ = ["parse_docx"]

_MIME: Final = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_MIN_HEADING_LEVEL: Final = 1
_MAX_HEADING_LEVEL: Final = 6


def _heading_level(style_name: str) -> int | None:
    """Return the outline depth for a Word heading style, or None if it is not a heading.

    The result is clamped into 1-6. A converter emitting a custom ``Heading 0`` style
    otherwise produced depth 0, and the builder keeps only entries strictly shallower than
    the incoming level — nothing is shallower than 0, so every real ``Heading 1`` nested
    underneath it instead of resetting the path, and the whole document came back sectioned
    under one bogus root.
    """
    lowered = style_name.lower()
    if lowered == "title":
        return 1
    if not lowered.startswith("heading"):
        return None

    tail = lowered.removeprefix("heading").strip()
    if not tail.isdigit():
        return None
    return max(_MIN_HEADING_LEVEL, min(int(tail), _MAX_HEADING_LEVEL))


def _table_text(table: Table) -> str:
    """Serialize a table as pipe-delimited rows."""
    rows: list[str] = []
    for row in table.rows:
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_docx(data: bytes, *, mime_type: str = _MIME) -> ParsedDocument:
    """Parse a DOCX file into structural blocks in reading order."""
    builder = DocumentBuilder(mime_type=mime_type, parser="docx")
    document = Document(io.BytesIO(data))

    properties = document.core_properties
    builder.meta(
        title=properties.title,
        author=properties.author,
        subject=properties.subject,
        created=properties.created.isoformat() if properties.created else None,
        modified=properties.modified.isoformat() if properties.modified else None,
    )

    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            paragraph = Paragraph(child, document)
            text = paragraph.text
            if not text.strip():
                continue

            style_name = paragraph.style.name if paragraph.style is not None else ""
            level = _heading_level(style_name or "")
            if level is not None:
                builder.heading(text, level)
            elif style_name and "list" in style_name.lower():
                builder.add("list_item", text)
            else:
                builder.add("paragraph", text)

        elif child.tag == qn("w:tbl"):
            builder.add("table", _table_text(Table(child, document)))
            builder.flag(ParseFlag.TABLES_DETECTED)

    return builder.build()
