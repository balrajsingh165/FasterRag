"""CSV and JSON parsers.

Rows are serialized as ``column: value`` pairs rather than bare cells, because a
retrieved chunk has to be self-describing: a naked row of values carries no meaning once
it is separated from its header. Wide files are chunked by row groups so no single block
becomes unretrievable.

JSON is walked with an explicit stack and refused past :data:`MAX_NESTING_DEPTH` with
``NESTING_TOO_DEEP``. The recursive walk raised ``RecursionError`` on a few thousand levels,
and how deep ``json.loads`` itself will go differs between interpreters — 3.12 refuses a
10,000-deep array that 3.14 accepts — so the limit is this module's, not the runtime's
(TASK-0489).
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Final, Literal

from fasterrag.core.parsing.models import DocumentBuilder, ParsedDocument, ParseFlag
from fasterrag.core.parsing.options import DEFAULT_PARSING_OPTIONS, ParsingOptions
from fasterrag.core.parsing.plaintext import decode
from fasterrag.errors import ErrorCode, ParseError

__all__ = ["parse_csv", "parse_json"]

_SNIFF_BYTES: Final = 8192

CR: Final = "\r"
CRLF: Final = "\r\n"
NEWLINE: Final = "\n"

MAX_NESTING_DEPTH: Final = 512
"""The deepest a JSON document may nest; deeper is refused with ``NESTING_TOO_DEEP``."""

_Node = tuple[Literal["node"], Any, str, int, int]
_Heading = tuple[Literal["heading"], str, int]
_Leaf = tuple[Literal["leaf"], str]
_Step = _Node | _Heading | _Leaf


def _describe(row: dict[str, Any]) -> str:
    """Render one record as ``column: value`` pairs, skipping values with no content."""
    return "; ".join(
        f"{key}: {value}" for key, value in row.items() if value is not None and str(value).strip()
    )


def _dialect(sample: str) -> type[csv.Dialect] | csv.Dialect:
    """Return the sniffed dialect, or the Excel default when the sniff is unusable."""
    # CRITICAL: the sample is newline-normalised before sniffing, because a dialect is a
    # property of the delimiter and quoting, never of the line terminator. `csv.Sniffer` does
    # not agree: on Python 3.12 it detects a space delimiter for a file with LF endings and
    # then fails outright on the identical file with CRLF, falling back to a comma and
    # splitting the rows differently. The result was that Windows line endings changed what a
    # block *was* rather than only what was inside it, which the parsing invariants assert is
    # impossible. CPython 3.13 changed the sniffer and hid it, so it reproduced only on the
    # interpreter floor this package supports.
    try:
        sniffed = csv.Sniffer().sniff(sample.replace(CRLF, NEWLINE).replace(CR, NEWLINE))
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
    """Parse delimited text, detecting the dialect from a sample of the file."""
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
    """Render one scalar with its key path, or nothing when it carries no content."""
    if value is None or value == "":
        return ""
    return f"{path}: {value}" if path else str(value)


def _too_deep() -> ParseError:
    """Return the refusal for a document nested past :data:`MAX_NESTING_DEPTH`."""
    return ParseError(
        f"the JSON document nests deeper than {MAX_NESTING_DEPTH} levels",
        code=ErrorCode.NESTING_TOO_DEEP,
        retryable=False,
    )


def _children(node: Any, path: str, depth: int, nesting: int) -> list[_Step]:
    """Return the steps a container expands to, in document order."""
    steps: list[_Step] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, dict | list):
                child = f"{path}.{key}" if path else str(key)
                steps.append(("heading", child, min(depth + 1, 6)))
                steps.append(("node", value, child, depth + 1, nesting + 1))
    else:
        for index, item in enumerate(node):
            if isinstance(item, dict | list):
                steps.append(("node", item, f"{path}[{index}]", depth, nesting + 1))
            else:
                steps.append(("leaf", _leaf(f"{path}[{index}]", item)))
    return steps


def _walk(document: Any, builder: DocumentBuilder) -> None:
    """Emit blocks for a JSON tree, keys as the heading path, with a stack instead of recursion."""
    stack: list[_Step] = [("node", document, "", 0, 1)]
    while stack:
        step = stack.pop()
        if step[0] == "heading":
            builder.heading(step[1], step[2])
        elif step[0] == "leaf":
            builder.add("paragraph", step[1])
        else:
            _, node, path, depth, nesting = step
            if nesting > MAX_NESTING_DEPTH:
                raise _too_deep()
            if not isinstance(node, dict | list):
                builder.add("paragraph", _leaf(path, node))
                continue
            if isinstance(node, dict):
                scalars = {k: v for k, v in node.items() if not isinstance(v, dict | list)}
                if scalars:
                    builder.add("paragraph", _describe(scalars))
            stack.extend(reversed(_children(node, path, depth, nesting)))


def parse_json(data: bytes, *, mime_type: str = "application/json") -> ParsedDocument:
    """Parse JSON into blocks that keep each value's key path.

    Raises:
        ParseError: If the document is not valid JSON, or ``NESTING_TOO_DEEP`` past the limit.
    """
    builder = DocumentBuilder(mime_type=mime_type, parser="json")
    text = decode(data, builder)

    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON at line {exc.lineno} column {exc.colno}") from exc
    except RecursionError as exc:
        raise _too_deep() from exc

    _walk(document, builder)
    return builder.build()
