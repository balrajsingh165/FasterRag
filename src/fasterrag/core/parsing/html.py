"""HTML parser.

Walks the document in tree order so reading order survives, drops non-content elements
(scripts, styles, navigation chrome), and serializes tables row by row instead of
flattening them into a wall of words.

A content element (a heading, paragraph, list item, table…) is emitted whole and its
descendants are not visited again, so ``<li><p>x</p></li>`` is one block rather than two.
Text that sits in no content element — straight inside a ``div``, a ``span`` or the body —
is gathered into a paragraph, closed at every block-level boundary. The walk keeps its own
stack instead of recursing, so nesting depth costs memory, never the interpreter's
recursion limit (TASK-0459, TASK-0489).
"""

from __future__ import annotations

from typing import Final

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString, PageElement, PreformattedString

from fasterrag.core.parsing.models import DocumentBuilder, ParsedDocument, ParseFlag
from fasterrag.core.parsing.plaintext import decode

__all__ = ["parse_html"]

_DROPPED: Final[frozenset[str]] = frozenset(
    {"script", "style", "noscript", "template", "nav", "footer", "aside", "form", "svg"}
)
_HEADINGS: Final[frozenset[str]] = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_PARAGRAPHS: Final[frozenset[str]] = frozenset({"p", "blockquote", "figcaption"})
_CONTENT: Final[frozenset[str]] = _HEADINGS | _PARAGRAPHS | {"li", "pre", "table", "dt", "dd"}
_NOT_WALKED: Final[frozenset[str]] = frozenset({"head", "title"})
_INLINE: Final[frozenset[str]] = frozenset(
    {"a", "abbr", "acronym", "b", "bdi", "bdo", "big", "cite", "code", "data", "del", "dfn"}
    | {"em", "font", "i", "ins", "kbd", "label", "mark", "q", "s", "samp", "small", "span"}
    | {"strike", "strong", "sub", "sup", "time", "tt", "u", "var"}
)
_LINE_BREAKS: Final[frozenset[str]] = frozenset({"br", "wbr"})
_BOUNDARY: Final = object()


def _table_text(table: Tag) -> str:
    """Serialize a table as pipe-delimited rows."""
    rows: list[str] = []
    for row in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if any(cells):
            rows.append(" | ".join(cells))
    return "\n".join(rows)


def _emit(element: Tag, builder: DocumentBuilder) -> None:
    """Append one content element as the block its tag names."""
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


def _walk(root: Tag, builder: DocumentBuilder) -> None:
    """Emit content elements and loose text in document order, without recursion."""
    loose: list[str] = []

    def flush() -> None:
        builder.add("paragraph", " ".join("".join(loose).split()))
        loose.clear()

    stack: list[PageElement | object] = list(reversed(root.contents))
    while stack:
        node = stack.pop()
        if node is _BOUNDARY:
            flush()
        elif isinstance(node, NavigableString):
            if not isinstance(node, PreformattedString):
                loose.append(str(node))
        elif isinstance(node, Tag) and node.name not in _NOT_WALKED:
            if node.name in _CONTENT:
                flush()
                _emit(node, builder)
            elif node.name in _LINE_BREAKS:
                loose.append(" ")
            elif node.name in _INLINE:
                stack.extend(reversed(node.contents))
            else:
                flush()
                stack.append(_BOUNDARY)
                stack.extend(reversed(node.contents))
    flush()


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

    _walk(soup.body or soup, builder)
    return builder.build()
