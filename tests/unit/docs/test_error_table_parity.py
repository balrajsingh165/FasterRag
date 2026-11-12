r"""api-reference.md's error code table must document every ``ErrorCode``, with a remedy.

Until TASK-0500 the statuses were checked against a dict typed into ``tests/unit/test_errors.py``,
which compared the code with itself: the document could lose a row, or state a wrong status,
and nothing failed. This file parses the table instead, so the document is the thing checked.

**The format.** The table directly under ``### Error code table`` has exactly the columns in
:data:`HEADER`, one row per code, in any order. The first cell is the code in backticks and
nothing else; the second is the bare HTTP status; the class cell is a class exported by
``fasterrag.errors`` or ``—``; meaning and remedy are prose and may not be empty or ``—``.
A literal pipe inside a cell is written ``\|``. A new code is a new row at the end.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import fasterrag.errors as errors_module
from fasterrag.errors import PROBLEM_SPECS, ErrorCode, ProblemSpec

ROOT: Final = Path(__file__).resolve().parents[3]
API_REFERENCE: Final = ROOT / "docs" / "api-reference.md"
HEADING: Final = "### Error code table"
HEADER: Final = ("`code`", "HTTP", "Taxonomy class", "Meaning", "Remedy")

_CELL_SPLIT: Final = re.compile(r"(?<!\\)\|")
_CODE_CELL: Final = re.compile(r"^`([A-Z][A-Z0-9_]*)`$")
_CLASS_CELL: Final = re.compile(r"^`([A-Za-z]+)`$")
_EMPTY: Final = frozenset({"", "—", "-"})


@dataclass(frozen=True)
class Row:
    """One parsed row of the error code table."""

    line: int
    cells: tuple[str, ...]


def _cells(line: str) -> tuple[str, ...]:
    r"""Split one Markdown table line into stripped cells, honouring ``\|``."""
    return tuple(cell.strip() for cell in _CELL_SPLIT.split(line.strip().strip("|")))


def parse_table(text: str) -> tuple[tuple[str, ...], list[Row]]:
    """Return the header and the rows of the table that follows :data:`HEADING`."""
    lines = text.splitlines()
    start = next(
        (i for i, line in enumerate(lines) if line.startswith(HEADING)),
        None,
    )
    assert start is not None, f"{API_REFERENCE.name} has no heading starting {HEADING!r}"
    table_start = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("|"))
    header = _cells(lines[table_start])
    rows: list[Row] = []
    for index in range(table_start + 2, len(lines)):
        if not lines[index].startswith("|"):
            break
        rows.append(Row(index + 1, _cells(lines[index])))
    return header, rows


def table_problems(rows: list[Row], specs: dict[ErrorCode, ProblemSpec]) -> list[str]:
    """Return every disagreement between the table's rows and ``specs``."""
    problems: list[str] = []
    seen: dict[str, int] = {}
    exported = set(errors_module.__all__)
    for row in rows:
        where = f"line {row.line}"
        if len(row.cells) != len(HEADER):
            problems.append(f"{where}: {len(row.cells)} cells, expected {len(HEADER)}")
            continue
        code_cell, status_cell, class_cell, meaning, remedy = row.cells
        match = _CODE_CELL.match(code_cell)
        if match is None:
            problems.append(f"{where}: first cell {code_cell!r} is not one backticked code")
            continue
        name = match.group(1)
        if name in seen:
            problems.append(f"{where}: {name} already has a row at line {seen[name]}")
        seen[name] = row.line
        if name not in ErrorCode.__members__:
            problems.append(f"{where}: {name} is not an ErrorCode member")
            continue
        documented = int(status_cell) if status_cell.isdigit() else None
        actual = specs[ErrorCode[name]].status
        if documented != actual:
            problems.append(
                f"{where}: {name} documented as {status_cell!r}, PROBLEM_SPECS {actual}"
            )
        class_match = _CLASS_CELL.match(class_cell)
        if class_cell not in _EMPTY and (
            class_match is None or class_match.group(1) not in exported
        ):
            problems.append(f"{where}: {name} names {class_cell!r}, not a fasterrag.errors class")
        for column, value in (("Meaning", meaning), ("Remedy", remedy)):
            if value in _EMPTY:
                problems.append(f"{where}: {name} has no {column}")
    for code in specs:
        if code.name not in seen:
            problems.append(f"{code.name} has no row in the error code table")
    return problems


def test_the_table_has_the_documented_columns() -> None:
    """The parser reads columns by position, so the header is pinned."""
    header, rows = parse_table(API_REFERENCE.read_text(encoding="utf-8"))
    assert header == HEADER
    assert rows


def test_every_error_code_has_exactly_one_correct_row_with_a_remedy() -> None:
    """A code without a row, a row without a code, or a wrong status fails the build."""
    _, rows = parse_table(API_REFERENCE.read_text(encoding="utf-8"))
    assert table_problems(rows, PROBLEM_SPECS) == []


def test_a_code_missing_from_the_table_is_reported() -> None:
    """Pins the direction TASK-0500 exists for: a new code with no row."""
    _, rows = parse_table(API_REFERENCE.read_text(encoding="utf-8"))
    dropped = [row for row in rows if row.cells[0] != f"`{ErrorCode.QUEUE_FULL}`"]
    assert table_problems(dropped, PROBLEM_SPECS) == [
        "QUEUE_FULL has no row in the error code table"
    ]


def test_an_unknown_code_a_wrong_status_and_an_empty_remedy_are_reported() -> None:
    """Each other way the table can drift from the code names its line."""
    specs = {ErrorCode.NOT_FOUND: PROBLEM_SPECS[ErrorCode.NOT_FOUND]}
    rows = [
        Row(1, ("`NOT_FOUND`", "410", "—", "Gone.", "—")),
        Row(2, ("`NO_SUCH_CODE`", "500", "—", "Made up.", "Delete the row.")),
        Row(3, ("`NOT_FOUND`", "404", "`NotAClass`", "Twice.", "Once.")),
    ]
    assert table_problems(rows, specs) == [
        "line 1: NOT_FOUND documented as '410', PROBLEM_SPECS 404",
        "line 1: NOT_FOUND has no Remedy",
        "line 2: NO_SUCH_CODE is not an ErrorCode member",
        "line 3: NOT_FOUND already has a row at line 1",
        "line 3: NOT_FOUND names '`NotAClass`', not a fasterrag.errors class",
    ]
