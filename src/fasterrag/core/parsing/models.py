"""Parsed-document structures shared by every parser.

A parsed document is a single canonical text plus the blocks that compose it, in reading
order. Blocks carry the page, the heading path, and character offsets into that text,
which is what lets a chunk record ``span``, ``page``, and ``section`` later
(``docs/data-model.md``).

One invariant makes the whole chain trustworthy and is property-tested: for every block,
``document.text[block.start:block.end] == block.text``. Chunk offsets are only meaningful
because parser offsets are.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Literal

__all__ = [
    "BLOCK_SEPARATOR",
    "Block",
    "BlockKind",
    "DocumentBuilder",
    "ParseFlag",
    "ParsedDocument",
]

BlockKind = Literal["heading", "paragraph", "list_item", "table", "code"]

BLOCK_SEPARATOR: Final = "\n\n"
SECTION_SEPARATOR: Final = " > "


class ParseFlag(StrEnum):
    """Quality signals a parser records on a document.

    Stored on the document's ``parse_flags`` so a degraded parse is visible instead of
    silently indexed (``docs/data-model.md``).
    """

    TABLES_DETECTED = "tables_detected"
    OCR_APPLIED = "ocr_applied"
    LOW_TEXT_YIELD = "low_text_yield"
    ENCODING_FALLBACK = "encoding_fallback"


@dataclass(frozen=True, slots=True)
class Block:
    """One structural unit of a document, in reading order."""

    kind: BlockKind
    text: str
    start: int
    end: int
    page: int | None = None
    section: str | None = None
    level: int | None = None


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """The canonical text of a source document plus its structure."""

    text: str
    blocks: tuple[Block, ...]
    mime_type: str
    parser: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    flags: tuple[str, ...] = ()

    def block_at(self, offset: int) -> Block | None:
        """Return the block containing ``offset``, or None if it falls in a separator."""
        for block in self.blocks:
            if block.start <= offset < block.end:
                return block
        return None

    def section_at(self, offset: int) -> str | None:
        """Return the heading path in force at ``offset``."""
        current: str | None = None
        for block in self.blocks:
            if block.start > offset:
                break
            current = block.section
        return current

    def page_at(self, offset: int) -> int | None:
        """Return the page containing ``offset``, for paginated sources."""
        current: int | None = None
        for block in self.blocks:
            if block.start > offset:
                break
            current = block.page
        return current


class DocumentBuilder:
    """Accumulates blocks while keeping offsets consistent with the joined text.

    Every parser builds through this, so the offset invariant and the heading-path
    bookkeeping are implemented once rather than in each format.
    """

    def __init__(self, *, mime_type: str, parser: str) -> None:
        """Start an empty document."""
        self._mime_type = mime_type
        self._parser = parser
        self._parts: list[str] = []
        self._blocks: list[Block] = []
        self._offset = 0
        self._headings: list[tuple[int, str]] = []
        self._flags: list[str] = []
        self._metadata: dict[str, Any] = {}

    @property
    def section(self) -> str | None:
        """Return the current heading path, deepest last."""
        if not self._headings:
            return None
        return SECTION_SEPARATOR.join(text for _, text in self._headings)

    def add(
        self,
        kind: BlockKind,
        text: str,
        *,
        page: int | None = None,
        level: int | None = None,
    ) -> None:
        """Append a block, skipping anything that is empty after stripping."""
        cleaned = text.strip()
        if not cleaned:
            return

        if self._parts:
            self._offset += len(BLOCK_SEPARATOR)
            self._parts.append(BLOCK_SEPARATOR)

        start = self._offset
        self._parts.append(cleaned)
        self._offset += len(cleaned)

        self._blocks.append(
            Block(
                kind=kind,
                text=cleaned,
                start=start,
                end=self._offset,
                page=page,
                section=self.section,
                level=level,
            )
        )

    def heading(self, text: str, level: int, *, page: int | None = None) -> None:
        """Append a heading and update the heading path."""
        cleaned = text.strip()
        if not cleaned:
            return

        self._headings = [entry for entry in self._headings if entry[0] < level]
        self._headings.append((level, cleaned))
        self.add("heading", cleaned, page=page, level=level)

    def flag(self, flag: ParseFlag) -> None:
        """Record a parse-quality signal once."""
        if flag.value not in self._flags:
            self._flags.append(flag.value)

    def meta(self, **values: Any) -> None:
        """Record document metadata, ignoring empty values."""
        for key, value in values.items():
            if value not in (None, "", [], {}):
                self._metadata[key] = value

    @property
    def blocks(self) -> Sequence[Block]:
        """Return the blocks accumulated so far."""
        return self._blocks

    def build(self) -> ParsedDocument:
        """Return the finished document."""
        return ParsedDocument(
            text="".join(self._parts),
            blocks=tuple(self._blocks),
            mime_type=self._mime_type,
            parser=self._parser,
            metadata=dict(self._metadata),
            flags=tuple(self._flags),
        )
