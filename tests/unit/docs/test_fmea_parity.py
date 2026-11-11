"""failure-modes.md's proof column must name tests that exist.

The FMEA table's last column is the one that makes the rest of the document worth
anything: a failure mode with a mitigation nobody can re-run is a paragraph, not a control.
Four separate rounds found a row naming a suite that was never written — rows 8, 19 and 10's
load-test clause, and then row 18's "Chaos: reranker-down rung" (TASK-0289) — and each was
caught by eye, by somebody who happened to be reading nearby.

`scripts/check_claims.py` cannot catch this class. That gate pairs a *denial* with a ticked
task id, and this is the mirror image: an **affirmation** that a test exists. A denial that
goes stale is a document lagging reality; an affirmation that was never true sends the next
reader looking for a file that is not there, and then persuades them the behaviour is
covered when nothing covers it. This file closes that direction, the way
`test_internals_parity.py` (TASK-0255) closed it for internals.md's spec→module map, whose
design it follows — parse the table, resolve every token, and keep an explicit, minimal
exemption list rather than quietly skipping what does not resolve.

**How a citation is told apart from prose.** The document's own convention, stated in its
preamble: a citation is a backticked token beginning `tests/`, optionally narrowed with
`::` to a class or function inside the file. Every other backtick in the table — a config
key like `chunking.chunk_size`, a problem `code`, a metric name, a symbol like
`create_dashboard` — is prose evidence and is never resolved, so widening the pattern is the
one change that could make this gate fire on an honest row and get it disabled.

**What a row with no proof says.** `**no test**`, in the column itself, plus what *is*
covered. Those rows are listed in :data:`UNPROVEN_ROWS` with the reason, and the two
directions are both checked: a row carrying the marker must be listed, and a listed row must
carry it. A row may do both — cite a test for one half of its claim and mark the other half
missing — which is how row 30 is written. Rows 8 and 10 were written that way until
TASK-0243 built the soak and load suites on 2026-11-11 and both lost their marker.

Every assertion here is made against the parsed rows rather than against the raw text, so
deleting the table turns this file red instead of leaving a sentence in prose to satisfy it.
"""

import re
from pathlib import Path
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[3]
FMEA: Final = ROOT / "docs" / "failure-modes.md"
CHAOS: Final = ROOT / "tests" / "chaos"

_CITATION: Final = re.compile(r"`(tests/[^`]+)`")
_BACKTICK: Final = re.compile(r"`([^`]+)`")
_MARKER: Final = "**no test**"
_COLUMNS: Final = 7

UNPROVEN_ROWS: Final[dict[int, str]] = {
    2: "no scanned-document fixture exists and nothing asserts the low_text_yield flag",
    19: "a load suite exists since TASK-0243, and cannot prove this row because the row's "
    "own per-call rerank timeout and stage=rerank label are unbuilt (TASK-0165)",
    30: "no case injects an unhealthy container into a compose bring-up",
    36: "nothing asserts the query path answering while the dashboard is down",
}
"""Rows whose proof column admits it has none, with why.

Listed here rather than left to the marker alone so that the exemption is reviewable in one
place and cannot grow silently: adding a row to this table is a visible diff, and
:func:`test_the_exemption_list_stays_minimal` deletes the entry the day the row gets a test.
A row absent from both the citations and this set is a genuine gap, which is the whole point.
"""

EXEMPT_CHAOS_MODULES: Final[frozenset[str]] = frozenset({"conftest.py", "disk_quota_probe.py"})
"""Files under tests/chaos/ that are not scenarios, so no section of the document owes them one."""


def _rows() -> list[tuple[int, str]]:
    """Return the (number, proof cell) of every data row in the FMEA table."""
    rows: list[tuple[int, str]] = []
    for line in FMEA.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != _COLUMNS or not cells[0].isdigit():
            continue
        rows.append((int(cells[0]), cells[-1]))
    return rows


def _defines(source: str, name: str) -> bool:
    """Return whether a module defines a function or class of this name."""
    pattern = rf"^\s*(?:async\s+)?(?:def|class)\s+{re.escape(name)}\b"
    return re.search(pattern, source, re.M) is not None


def resolve(citation: str) -> str | None:
    """Return why a citation does not resolve, or ``None`` when it does.

    A citation is a repository-root path — a directory when it ends in ``/``, otherwise a
    file, optionally followed by ``::`` segments naming a class or function it must define.
    """
    path, _, trailing = citation.partition("::")
    target = ROOT / path
    if path.endswith("/"):
        return None if target.is_dir() else f"{path} is not a directory"
    if not target.is_file():
        return f"{path} is not a file"
    source = target.read_text(encoding="utf-8")
    for name in (segment for segment in trailing.split("::") if segment):
        if not _defines(source, name):
            return f"{path} defines no {name}"
    return None


def table_citations() -> list[tuple[int, str]]:
    """Return every (row number, citation) pair the proof column names."""
    return [(number, found) for number, cell in _rows() for found in _CITATION.findall(cell)]


def document_citations() -> list[str]:
    """Return every citation anywhere in the document, prose sections included."""
    return _CITATION.findall(FMEA.read_text(encoding="utf-8"))


def marked_rows() -> set[int]:
    """Return the rows whose proof column admits it has no test."""
    return {number for number, cell in _rows() if _MARKER in cell}


def chaos_modules() -> list[str]:
    """Return every scenario module that ships under tests/chaos/."""
    return sorted(
        path.name
        for path in CHAOS.glob("*.py")
        if path.name not in EXEMPT_CHAOS_MODULES and not path.name.startswith("__")
    )


def test_the_fmea_table_was_parsed() -> None:
    """Trap T12. A parser matching zero rows reports perfect parity over an empty set."""
    assert len(_rows()) >= 30


def test_the_proof_column_names_tests() -> None:
    """The same guard one level down: a citation pattern that stopped matching is vacuous."""
    assert len(table_citations()) >= 30


def test_the_document_names_tests_outside_the_table_too() -> None:
    """The observed-behavior sections cite suites in prose; those must be resolved as well."""
    assert len(document_citations()) > len(table_citations())


def test_every_test_the_proof_column_names_exists() -> None:
    """A row citing a test nobody wrote claims proof that nobody can run."""
    broken = sorted(
        f"row {number}: {citation} ({reason})"
        for number, citation in table_citations()
        if (reason := resolve(citation)) is not None
    )

    assert broken == [], f"failure-modes.md cites tests that are not in the tree: {broken}"


def test_every_test_the_document_names_anywhere_exists() -> None:
    """Prose citations rot the same way; the chaos logs below the table are full of them."""
    broken = sorted(
        f"{citation} ({reason})"
        for citation in set(document_citations())
        if (reason := resolve(citation)) is not None
    )

    assert broken == [], f"failure-modes.md names tests that are not in the tree: {broken}"


def test_every_row_either_cites_a_test_or_says_it_has_none() -> None:
    """The affirmative direction. Silence in this column is how all four gaps survived."""
    cited = {number for number, _ in table_citations()}
    silent = sorted(
        number for number, _ in _rows() if number not in cited and number not in marked_rows()
    )

    assert silent == [], f"FMEA rows neither citing a test nor marked '{_MARKER}': {silent}"


def test_the_marked_rows_are_exactly_the_listed_ones() -> None:
    """Both directions, so neither the document nor this file can widen the hole alone."""
    assert sorted(marked_rows()) == sorted(UNPROVEN_ROWS)


def test_the_exemption_list_stays_minimal() -> None:
    """A row that gets its test must lose its entry here, not keep a standing exemption."""
    numbers = {number for number, _ in _rows()}
    stale = sorted(number for number in UNPROVEN_ROWS if number not in numbers)

    assert stale == [], f"UNPROVEN_ROWS names rows the FMEA table no longer has: {stale}"


def test_every_chaos_module_is_accounted_for() -> None:
    """The tree→doc direction: a landed scenario this document ignores is a silent suite.

    The document says as much of the reindex drill — that it is logged "so that a landed
    chaos suite is not one this table silently ignores" — and nothing held it to that.
    """
    body = FMEA.read_text(encoding="utf-8")
    unmentioned = [name for name in chaos_modules() if f"tests/chaos/{name}" not in body]

    assert unmentioned == [], f"chaos scenarios failure-modes.md never mentions: {unmentioned}"


def test_prose_backticks_are_not_mistaken_for_citations() -> None:
    """The discriminator, pinned: widening it is what would make this gate fire on prose.

    Config keys, problem codes and symbols all appear in backticks in the same cells, and a
    pattern that tried to resolve them would produce false reds on an honest document.
    """
    prose = {"chunking.chunk_size", "RETRIEVAL_FAILED", "create_dashboard", "low_text_yield"}
    everything = set(_BACKTICK.findall(FMEA.read_text(encoding="utf-8")))

    assert prose <= everything
    assert prose.isdisjoint(set(document_citations()))


def test_a_citation_naming_a_missing_file_or_function_does_not_resolve() -> None:
    """The resolver's own red path, so a bug there cannot make the gate pass by accident."""
    assert resolve("tests/chaos/test_reranker_down.py") is None
    assert resolve("tests/chaos/test_no_such_scenario.py") is not None
    assert resolve("tests/chaos/test_reranker_down.py::test_no_such_case") is not None
    assert resolve("tests/chaos/test_reranker_down.py::ScoringModel") is None
    assert resolve("tests/chaos/") is None
    assert resolve("tests/no_such_directory/") is not None
    assert resolve("tests/chaos/test_reranker_down.py/") is not None
