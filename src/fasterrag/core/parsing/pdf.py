"""PDF parser with table extraction, layout-based heading inference, and an OCR path.

Three things make PDF the hardest format and each is handled explicitly:

* **Tables.** Table regions are located first and their cells serialized row by row.
  Lines that fall inside a table are then skipped, because a table extracted *and* read
  again as prose duplicates its content and poisons retrieval.
* **Headings.** A PDF has no semantic headings, so they are inferred from type size: a
  short line noticeably larger than the document's body size becomes a heading. This is
  a documented heuristic, not a guarantee, and it only ever affects the ``section`` label.
* **Scanned pages.** A page yielding almost no text is rasterized and passed through OCR
  when the ``ocr`` extra and the tesseract binary are both present. When they are not,
  the document is flagged ``low_text_yield`` rather than quietly indexed as empty, and a
  document with no extractable text at all fails so it lands in the dead-letter queue
  (``docs/failure-modes.md`` row 2).
"""

from __future__ import annotations

import io
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import pdfplumber

from fasterrag.core.parsing.models import DocumentBuilder, ParsedDocument, ParseFlag
from fasterrag.errors import ParseError
from fasterrag.observability.logging import get_logger

__all__ = ["MINIMUM_CHARS_PER_PAGE", "parse_pdf"]

MINIMUM_CHARS_PER_PAGE: Final = 40
HEADING_SIZE_RATIO: Final = 1.15
MAX_HEADING_CHARS: Final = 120
OCR_RESOLUTION: Final = 200
_MAX_HEADING_LEVEL: Final = 6

_logger = get_logger(__name__)


@dataclass
class _Page:
    """Extraction results for one page, gathered before blocks are built."""

    number: int
    tables: list[str] = field(default_factory=list)
    lines: list[tuple[str, float]] = field(default_factory=list)
    ocr_text: str | None = None


def _line_size(line: dict[str, Any]) -> float:
    """Return the largest glyph size on a line, which is what makes headings stand out."""
    sizes = [char.get("size", 0.0) for char in line.get("chars", [])]
    return max(sizes) if sizes else 0.0


def _inside_table(line: dict[str, Any], boxes: Sequence[tuple[float, float, float, float]]) -> bool:
    """Return whether a line sits within an already-extracted table region."""
    centre = (float(line["top"]) + float(line["bottom"])) / 2
    for x0, top, x1, bottom in boxes:
        if top <= centre <= bottom and float(line["x1"]) > x0 and float(line["x0"]) < x1:
            return True
    return False


def _serialize_table(rows: Sequence[Sequence[str | None]]) -> str:
    """Render extracted table cells as pipe-delimited rows."""
    lines: list[str] = []
    for row in rows:
        cells = [(cell or "").strip().replace("\n", " ") for cell in row]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _ocr_page(page: pdfplumber.page.Page) -> str | None:
    """Return OCR text for a page, or None when OCR is unavailable.

    Both the Python extra and the tesseract binary are optional. Their absence is a
    reported capability gap, never a failure: the caller flags ``low_text_yield``.
    """
    try:
        import pytesseract
    except ImportError:
        _logger.info("ocr unavailable: install the 'ocr' extra to read scanned pages")
        return None

    try:
        image = page.to_image(resolution=OCR_RESOLUTION).original
        return str(pytesseract.image_to_string(image))
    except pytesseract.TesseractNotFoundError:
        _logger.info("ocr unavailable: the tesseract binary is not installed")
        return None
    except (OSError, ValueError) as exc:
        _logger.warning("ocr failed for a page", extra={"reason": type(exc).__name__})
        return None


def _extract(pdf: pdfplumber.pdf.PDF) -> list[_Page]:
    """Pull tables, text lines, and any OCR text out of every page."""
    pages: list[_Page] = []

    for number, page in enumerate(pdf.pages, start=1):
        extracted = _Page(number=number)
        boxes: list[tuple[float, float, float, float]] = []

        for table in page.find_tables():
            serialized = _serialize_table(table.extract())
            if serialized:
                extracted.tables.append(serialized)
                boxes.append(
                    (
                        float(table.bbox[0]),
                        float(table.bbox[1]),
                        float(table.bbox[2]),
                        float(table.bbox[3]),
                    )
                )

        for line in page.extract_text_lines():
            text = str(line.get("text", "")).strip()
            if text and not _inside_table(line, boxes):
                extracted.lines.append((text, _line_size(line)))

        page_chars = sum(len(text) for text, _ in extracted.lines)
        page_chars += sum(len(table) for table in extracted.tables)
        if page_chars < MINIMUM_CHARS_PER_PAGE:
            extracted.ocr_text = _ocr_page(page)

        pages.append(extracted)

    return pages


def _body_size(pages: Sequence[_Page]) -> float:
    """Return the document's typical type size, used as the heading baseline."""
    sizes = [size for page in pages for _, size in page.lines if size > 0]
    return statistics.median(sizes) if sizes else 0.0


def _is_heading(text: str, size: float, body: float) -> bool:
    """Return whether a line looks like a heading rather than prose."""
    return bool(body) and size >= body * HEADING_SIZE_RATIO and len(text) <= MAX_HEADING_CHARS


def parse_pdf(data: bytes, *, mime_type: str = "application/pdf") -> ParsedDocument:
    """Parse a PDF into structural blocks.

    Raises:
        ParseError: If the file cannot be opened, or if no page yields any text even
            after the OCR path, which sends the document to the dead-letter queue.
    """
    builder = DocumentBuilder(mime_type=mime_type, parser="pdf-layout")

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            metadata = dict(pdf.metadata or {})
            pages = _extract(pdf)
    except ParseError:
        raise
    # CRITICAL: the catch stays broad because pdfminer raises undocumented exception types
    # on malformed files. A corrupt document must become a typed ParseError so the pipeline
    # dead-letters it and continues; letting an arbitrary exception escape kills the worker.
    except Exception as exc:
        raise ParseError(f"the PDF could not be read: {type(exc).__name__}") from exc

    builder.meta(
        title=str(metadata.get("Title", "")).strip() or None,
        author=str(metadata.get("Author", "")).strip() or None,
        page_count=len(pages),
    )

    body = _body_size(pages)
    heading_sizes = sorted(
        {size for page in pages for _, size in page.lines if size > body}, reverse=True
    )

    for page in pages:
        for table in page.tables:
            builder.add("table", table, page=page.number)
            builder.flag(ParseFlag.TABLES_DETECTED)

        if page.ocr_text:
            builder.flag(ParseFlag.OCR_APPLIED)
            for recognized in page.ocr_text.split("\n\n"):
                builder.add("paragraph", recognized, page=page.number)
            continue

        if not page.lines and not page.tables:
            builder.flag(ParseFlag.LOW_TEXT_YIELD)
            continue

        paragraph: list[str] = []
        for text, size in page.lines:
            if _is_heading(text, size, body):
                builder.add("paragraph", " ".join(paragraph), page=page.number)
                paragraph = []
                level = (
                    min(heading_sizes.index(size) + 1, _MAX_HEADING_LEVEL)
                    if size in heading_sizes
                    else 1
                )
                builder.heading(text, level, page=page.number)
            else:
                paragraph.append(text)

        builder.add("paragraph", " ".join(paragraph), page=page.number)

    document = builder.build()
    if not document.text.strip():
        raise ParseError(
            "the PDF yielded no extractable text; it is likely a scan and OCR is "
            "unavailable (install the 'ocr' extra and the tesseract binary)"
        )
    return document
