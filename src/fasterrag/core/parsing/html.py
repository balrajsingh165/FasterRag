"""HTML parser.

Walks the document in tree order so reading order survives, drops non-content elements
(scripts, styles, navigation chrome), and serializes tables row by row instead of
flattening them into a wall of words.
"""

from __future__ import annotations

from typing import Final

from bs4 import BeautifulSoup, Tag

from fasterrag.core.parsing.models import DocumentBuilder, ParsedDocument, ParseFlag
from fasterrag.core.parsing.plaintext import decode

__all__ = ["parse_html"]

_DROPPED: Final[frozenset[str]] = frozenset(
    {"script", "style", "noscript", "template", "nav", "footer", "aside", "form", "svg"}
)
_HEADINGS: Final[frozenset[str]] = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_PARAGRAPHS: Final[frozenset[str]] = frozenset({"p", "blockquote", "figcaption"})
_CONTENT: Final[frozenset[str]] = _HEADINGS | _PARAGRAPHS | {"li", "pre", "table", "dt", "dd"}


def _table_text(table: Tag) -> str:
    """Serialize a table as pipe-delimited rows."""
    rows: list[str] = []
    for row in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def parse_html(data: bytes, *, mime_type: str = "text/html") -> ParsedDocument:
    """Parse HTML into structural blocks, preserving reading order."""
    builder = DocumentBuilder(mime_type=mime_type, parser="html")
    soup = BeautifulSoup(decode(data, builder), "lxml")

    for element in soup.find_all(list(_DROPPED)):
        element.decompose()

    if soup.title and soup.title.string:
        builder.meta(title=soup.title.string.strip())
    for name in ("author", "description", "keywords"):
        tag = soup.find("meta", attrs={"name": name})
        if isinstance(tag, Tag):
            content = tag.get("content")
            if isinstance(content, str):
                builder.meta(**{name: content.strip()})

    body = soup.body or soup
    for element in body.find_all(list(_CONTENT)):
        name = element.name
        if name in _HEADINGS:
            builder.heading(element.get_text(" ", strip=True), int(name[1:]))
        elif name == "table":
            builder.add("table", _table_text(element))
            builder.flag(ParseFlag.TABLES_DETECTED)
        elif name == "pre":
            builder.add("code", element.get_text("\n", strip=True))
        elif name == "li":
            builder.add("list_item", element.get_text(" ", strip=True))
        else:
            builder.add("paragraph", element.get_text(" ", strip=True))

    return builder.build()
