"""Markdown parser.

Uses a real Markdown token stream rather than line matching, so ATX and Setext headings,
fenced code, list items, and tables are classified correctly and the heading path is
accurate. Structure is what layout-aware chunking depends on
(``docs/architecture.md`` §5).
"""

from __future__ import annotations

from typing import Final

from markdown_it import MarkdownIt
from markdown_it.token import Token

from fasterrag.core.parsing.models import DocumentBuilder, ParsedDocument, ParseFlag
from fasterrag.core.parsing.plaintext import decode

__all__ = ["parse_markdown"]

_MAX_HEADING_LEVEL: Final = 6


def _heading_level(token: Token) -> int:
    """Return the heading depth encoded in an ``h1`` to ``h6`` tag."""
    try:
        return min(int(token.tag[1:]), _MAX_HEADING_LEVEL)
    except ValueError:
        return 1


def parse_markdown(data: bytes, *, mime_type: str = "text/markdown") -> ParsedDocument:
    """Parse Markdown into heading, paragraph, list-item, code, and table blocks."""
    builder = DocumentBuilder(mime_type=mime_type, parser="markdown")
    text = decode(data, builder)

    tokens = MarkdownIt("commonmark").enable("table").parse(text)
    first_heading: str | None = None

    index = 0
    while index < len(tokens):
        token = tokens[index]

        if token.type == "heading_open":
            level = _heading_level(token)
            content = tokens[index + 1].content if index + 1 < len(tokens) else ""
            builder.heading(content, level)
            if first_heading is None:
                first_heading = content.strip()
            index += 3
            continue

        if token.type == "fence" or token.type == "code_block":
            builder.add("code", token.content)
            index += 1
            continue

        if token.type == "inline":
            index += 1
            continue

        if token.type == "paragraph_open":
            content = tokens[index + 1].content if index + 1 < len(tokens) else ""
            parent = tokens[index - 1].type if index else ""
            if parent.startswith("list_item"):
                builder.add("list_item", content)
            else:
                builder.add("paragraph", content)
            index += 3
            continue

        if token.type == "table_open":
            builder.add("table", _table_text(tokens, index))
            builder.flag(ParseFlag.TABLES_DETECTED)
            index = _skip_to(tokens, index, "table_close")
            continue

        index += 1

    builder.meta(title=first_heading)
    return builder.build()


def _table_text(tokens: list[Token], start: int) -> str:
    """Serialize a table's cells as pipe-delimited rows, preserving row structure."""
    rows: list[list[str]] = []
    current: list[str] = []

    for token in tokens[start:]:
        if token.type == "table_close":
            break
        if token.type in {"tr_open"}:
            current = []
        elif token.type in {"tr_close"}:
            if current:
                rows.append(current)
        elif token.type == "inline":
            current.append(token.content.strip())

    return "\n".join(" | ".join(row) for row in rows)


def _skip_to(tokens: list[Token], start: int, token_type: str) -> int:
    """Return the index just past the next token of ``token_type``."""
    for offset, token in enumerate(tokens[start:], start=start):
        if token.type == token_type:
            return offset + 1
    return len(tokens)
