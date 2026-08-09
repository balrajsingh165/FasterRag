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
    """Render one record as ``column: value`` pairs."""
    return "; ".join(f"{key}: {value}" for key, value in row.items() if value not in (None, ""))


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
    """
    builder = DocumentBuilder(mime_type=mime_type, parser="csv")
    text = decode(data, builder)

    try:
        dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(text[:_SNIFF_BYTES])
    except csv.Error:
        dialect = csv.excel

    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if reader.fieldnames:
        builder.meta(columns=[name for name in reader.fieldnames if name])

    group: list[str] = []
    rows = 0
    for record in reader:
        described = _describe({key: value for key, value in record.items() if key})
        if not described:
            continue
        group.append(described)
        rows += 1
        if len(group) >= options.rows_per_block:
            builder.add("table", "\n".join(group))
            group = []

    if group:
        builder.add("table", "\n".join(group))

    builder.meta(row_count=rows)
    builder.flag(ParseFlag.TABLES_DETECTED)
    return builder.build()


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
                builder.add("paragraph", f"{path}[{index}]: {item}")
        return

    builder.add("paragraph", f"{path}: {node}")


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
