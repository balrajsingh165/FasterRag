"""CSV and JSON parsers.

Rows are serialized as ``column: value`` pairs rather than bare cells, because a
retrieved chunk has to be self-describing: a naked row of values carries no meaning once
it is separated from its header. Wide files are chunked by row groups so no single block
becomes unretrievable.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Final

from fasterrag.core.parsing.models import DocumentBuilder, ParsedDocument, ParseFlag
from fasterrag.core.parsing.options import DEFAULT_PARSING_OPTIONS, ParsingOptions
from fasterrag.core.parsing.plaintext import decode
from fasterrag.errors import ParseError

__all__ = ["parse_csv", "parse_json"]

_SNIFF_BYTES: Final = 8192


def _describe(row: dict[str, Any]) -> str:
    """Render one record as ``column: value`` pairs, skipping values with no content.

    A blank value is dropped rather than labelled. ``value not in (None, "")`` let a
    whitespace-only cell through, which renders as ``column:    `` — and since the builder
    strips the block as a whole rather than each pair, a file of nothing but spaces and
    commas reached the index as one block of colons and semicolons.
    """
    return "; ".join(
        f"{key}: {value}" for key, value in row.items() if value is not None and str(value).strip()
    )


def _dialect(sample: str) -> type[csv.Dialect] | csv.Dialect:
    """Return the sniffed dialect, or the Excel default when the sniff is unusable.

    The sniffer validates a candidate on its own terms and can still hand back one
    ``csv.reader`` refuses — most easily a delimiter equal to the quote character, which
    ``"a"b"c"`` produces. The reader rejects that with a plain ``ValueError``, not a
    ``csv.Error``, so it escaped the parsing layer untyped and reached the API as a generic
    500 with no problem+json body. A sniff is a guess; an unusable one is discarded exactly
    like a failed one.
    """
    try:
        sniffed = csv.Sniffer().sniff(sample)
    except csv.Error:
        return csv.excel

    try:
        csv.reader([""], sniffed)
    except (csv.Error, ValueError, TypeError):
        return csv.excel

    return sniffed


def parse_csv(
    data: bytes,
    *,
    mime_type: str = "text/csv",
    options: ParsingOptions = DEFAULT_PARSING_OPTIONS,
) -> ParsedDocument:
    """Parse delimited text, detecting the dialect from a sample of the file.

    Args:
        data: The raw delimited-text bytes.
        mime_type: MIME type recorded on the parsed document.
        options: Parser thresholds; ``rows_per_block`` decides how many records are
            serialized into one block.

    Returns:
        The structured document.

    Raises:
        ParseError: If the delimited content cannot be read as rows.
    """
    builder = DocumentBuilder(mime_type=mime_type, parser="csv")
    text = decode(data, builder)

    # CRITICAL: newline="" is what the csv module's own documentation requires. Without it
    # StringIO translates newlines itself, and a lone CR — an old Mac line ending, or a cell
    # a spreadsheet exported with an embedded carriage return — reaches the reader as a
    # newline inside an unquoted field, which it rejects with `csv.Error`. Excel opens those
    # files without complaint, so the failure looked like a fasterRag bug, and the error was
    # untyped besides.
    reader = csv.DictReader(io.StringIO(text, newline=""), dialect=_dialect(text[:_SNIFF_BYTES]))
    if reader.fieldnames:
        builder.meta(columns=[name for name in reader.fieldnames if name])

    group: list[str] = []
    rows = 0
    try:
        for record in reader:
            described = _describe({key: value for key, value in record.items() if key})
            if not described:
                continue
            group.append(described)
            rows += 1
            if len(group) >= options.rows_per_block:
                builder.add("table", "\n".join(group))
                group = []
    except csv.Error as exc:
        raise ParseError(f"the delimited file could not be read: {exc}") from exc

    if group:
        builder.add("table", "\n".join(group))

    builder.meta(row_count=rows)
    builder.flag(ParseFlag.TABLES_DETECTED)
    return builder.build()


def _leaf(path: str, value: Any) -> str:
    """Render one scalar with its key path, or nothing when it carries no content.

    A scalar at the root has no path, and pasting one on anyway produced blocks like
    ``": None"`` for ``null`` and a lone ``":"`` for ``""`` — chunks that get embedded and
    retrieved while saying nothing. Empty values are dropped here for the same reason
    :func:`_describe` drops them inside an object.
    """
    if value is None or value == "":
        return ""
    return f"{path}: {value}" if path else str(value)


def _walk(node: Any, path: str, builder: DocumentBuilder, depth: int = 0) -> None:
    """Emit blocks for a JSON tree, using object keys as the heading path."""
    if isinstance(node, dict):
        scalars = {key: value for key, value in node.items() if not isinstance(value, dict | list)}
        if scalars:
            builder.add("paragraph", _describe(scalars))
        for key, value in node.items():
            if isinstance(value, dict | list):
                builder.heading(f"{path}.{key}" if path else str(key), min(depth + 1, 6))
                _walk(value, f"{path}.{key}" if path else str(key), builder, depth + 1)
        return

    if isinstance(node, list):
        for index, item in enumerate(node):
            if isinstance(item, dict | list):
                _walk(item, f"{path}[{index}]", builder, depth)
            else:
                builder.add("paragraph", _leaf(f"{path}[{index}]", item))
        return

    builder.add("paragraph", _leaf(path, node))


def parse_json(data: bytes, *, mime_type: str = "application/json") -> ParsedDocument:
    """Parse JSON into blocks that keep each value's key path.

    Raises:
        ParseError: If the document is not valid JSON.
    """
    builder = DocumentBuilder(mime_type=mime_type, parser="json")
    text = decode(data, builder)

    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON at line {exc.lineno} column {exc.colno}") from exc

    _walk(document, "", builder)
    return builder.build()
