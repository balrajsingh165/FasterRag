"""DOCX parser.

Walks the document body in XML order so paragraphs and tables interleave the way they do
on the page — iterating paragraphs and tables separately, as the obvious implementation
does, silently destroys reading order. Word's outline styles give real heading levels.

Three parts of a Word file sit outside the body and were never read until TASK-0459:

* **Headers and footers** — page furniture, emitted once each before the body, so they
  carry no heading path. A section whose header is linked to the previous one repeats it
  and is skipped, and identical text is emitted once however many sections repeat it.
* **Footnotes and endnotes** — emitted after the body in note order. Word's separator
  notes carry no text and are skipped.
* **Nested tables** — a table inside a cell is serialized into that cell as ``a / b``
  cells and ``;``-separated rows, where ``cell.text`` used to drop it.

A DOCX is a zip archive, so a small file can unpack to gigabytes. The archive's declared
uncompressed size is summed before anything is decompressed and a document above
:data:`MAX_UNZIPPED_BYTES` is refused with ``PAYLOAD_TOO_LARGE``. The declared size is a
real bound, not a hint: ``zipfile`` stops inflating a member at its declared size and
fails the CRC check on one that lies.
"""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator
from typing import Final

from docx import Document
from docx.document import Document as WordDocument
from docx.opc.constants import RELATIONSHIP_TYPE
from docx.oxml import parse_xml
from docx.oxml.ns import qn
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph

from fasterrag.core.parsing.models import DocumentBuilder, ParsedDocument, ParseFlag
from fasterrag.errors import ErrorCode, FasterRagError, IngestionError, ParseError

__all__ = ["MAX_UNZIPPED_BYTES", "parse_docx"]

_MIME: Final = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
_MIN_HEADING_LEVEL: Final = 1
_MAX_HEADING_LEVEL: Final = 6
_NOTE_PARTS: Final = (RELATIONSHIP_TYPE.FOOTNOTES, RELATIONSHIP_TYPE.ENDNOTES)
_NOTE_TAGS: Final = (qn("w:footnote"), qn("w:endnote"))
_TEXT_NOTE_TYPES: Final = frozenset({None, "normal"})

MAX_UNZIPPED_BYTES: Final = 256 * 1024 * 1024
"""The most a DOCX may unpack to; a zip bomb is refused before it is inflated."""


def _heading_level(style_name: str) -> int | None:
    """Return the outline depth for a Word heading style, or None if it is not a heading."""
    lowered = style_name.lower()
    if lowered == "title":
        return 1
    if not lowered.startswith("heading"):
        return None

    tail = lowered.removeprefix("heading").strip()
    if not tail.isdigit():
        return None
    return max(_MIN_HEADING_LEVEL, min(int(tail), _MAX_HEADING_LEVEL))


def _nested_text(table: Table) -> str:
    """Serialize a table found inside a cell onto one line."""
    return "; ".join(" / ".join(_cell_text(cell) for cell in row.cells) for row in table.rows)


def _cell_text(cell: _Cell) -> str:
    """Return a cell's text on one line, any table nested inside it included."""
    parts = [
        item.text if isinstance(item, Paragraph) else _nested_text(item)
        for item in cell.iter_inner_content()
    ]
    return "\n".join(parts).strip().replace("\n", " ")


def _table_text(table: Table) -> str:
    """Serialize a table as pipe-delimited rows."""
    rows: list[str] = []
    for row in table.rows:
        cells = [_cell_text(cell) for cell in row.cells]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _add_paragraph(builder: DocumentBuilder, paragraph: Paragraph, *, outline: bool) -> None:
    """Append a body paragraph as a heading, list item or paragraph by its style."""
    text = paragraph.text
    if not text.strip():
        return

    style_name = paragraph.style.name if paragraph.style is not None else ""
    level = _heading_level(style_name or "") if outline else None
    if level is not None:
        builder.heading(text, level)
    elif style_name and "list" in style_name.lower():
        builder.add("list_item", text)
    else:
        builder.add("paragraph", text)


def _add_table(builder: DocumentBuilder, table: Table) -> None:
    """Append a table block and record that the document has one."""
    builder.add("table", _table_text(table))
    builder.flag(ParseFlag.TABLES_DETECTED)


def _furniture(document: WordDocument) -> Iterator[Paragraph | Table]:
    """Yield every header and footer item once, skipping parts linked to a previous section."""
    for section in document.sections:
        for part in (
            section.first_page_header,
            section.header,
            section.even_page_header,
            section.first_page_footer,
            section.footer,
            section.even_page_footer,
        ):
            if not part.is_linked_to_previous:
                yield from part.iter_inner_content()


def _notes(document: WordDocument) -> Iterator[str]:
    """Yield the text of every footnote and endnote, in note order."""
    for relationship in document.part.rels.values():
        if relationship.is_external or relationship.reltype not in _NOTE_PARTS:
            continue
        root = parse_xml(relationship.target_part.blob)
        for note in root:
            if note.tag in _NOTE_TAGS and note.get(qn("w:type")) in _TEXT_NOTE_TYPES:
                yield " ".join(
                    "".join(run.text or "" for run in paragraph.iter(qn("w:t")))
                    for paragraph in note.iter(qn("w:p"))
                )


def _refuse_a_bomb(data: bytes) -> None:
    """Refuse an archive whose members declare more than :data:`MAX_UNZIPPED_BYTES`."""
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        declared = sum(member.file_size for member in archive.infolist())

    if declared > MAX_UNZIPPED_BYTES:
        raise IngestionError(
            f"the DOCX unpacks to {declared} bytes, above the {MAX_UNZIPPED_BYTES}-byte limit "
            "on a decompressed document",
            code=ErrorCode.PAYLOAD_TOO_LARGE,
            retryable=False,
        )


def _parse(data: bytes, builder: DocumentBuilder) -> None:
    """Fill ``builder`` from the document's furniture, body and notes, in that order."""
    document = Document(io.BytesIO(data))

    properties = document.core_properties
    builder.meta(
        title=properties.title,
        author=properties.author,
        subject=properties.subject,
        created=properties.created.isoformat() if properties.created else None,
        modified=properties.modified.isoformat() if properties.modified else None,
    )

    seen: set[str] = set()
    for item in _furniture(document):
        text = item.text if isinstance(item, Paragraph) else _table_text(item)
        if text.strip() and text not in seen:
            seen.add(text)
            if isinstance(item, Paragraph):
                _add_paragraph(builder, item, outline=False)
            else:
                _add_table(builder, item)

    for child in document.element.body.iterchildren():
        if child.tag == qn("w:p"):
            _add_paragraph(builder, Paragraph(child, document), outline=True)
        elif child.tag == qn("w:tbl"):
            _add_table(builder, Table(child, document))

    for note in _notes(document):
        builder.add("paragraph", note)


def parse_docx(data: bytes, *, mime_type: str = _MIME) -> ParsedDocument:
    """Parse a DOCX file into structural blocks in reading order.

    Raises:
        IngestionError: ``PAYLOAD_TOO_LARGE`` when the archive unpacks past the limit.
        ParseError: When the file is not a readable DOCX.
    """
    builder = DocumentBuilder(mime_type=mime_type, parser="docx")
    try:
        _refuse_a_bomb(data)
        _parse(data, builder)
    except FasterRagError:
        raise
    # CRITICAL: broad for the reason the PDF parser's is. A truncated or bit-flipped DOCX
    # fails inside zipfile, zlib, lxml or python-docx with whichever of a dozen exception
    # types the damage reaches first, none of them documented as the API's. Each must
    # become a typed ParseError so the document is dead-lettered and the job continues.
    except Exception as exc:
        raise ParseError(f"the DOCX could not be read: {type(exc).__name__}: {exc}") from exc
    return builder.build()
