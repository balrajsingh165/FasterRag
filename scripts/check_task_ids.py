"""Fail when source code cites a ``TASK-`` id the ledger cannot support.

Two ways it cannot: the id is not defined in ``docs/todo.md`` at all, or a marker claiming
work is still outstanding cites an id the ledger has already ticked.

Every other id gate in this repository looks at ``docs/``. ``check_claims.py`` scans
``docs/*.md`` for stale denials, ``check_blockers.py`` checks that every blocker names a
real entry, ``check_doc_truth.py`` checks the ledger against itself. None of them opens a
``.py`` file. So a module docstring, or a ``# TODO:`` marker, could cite a task that is
ticked, superseded or simply invented, and every gate stayed green.

That is not hypothetical. ``core/rate_limit.py`` and two rate-limit test modules were left
citing TASK-0216 for the "never run against a real server" gap after the ledger moved that
gap to TASK-0290. Both ids exist, so no spelling check would have helped; the pointer was
simply aimed at finished work, and a reader following it would have concluded the gap was
closed. Both gates passed.

Two rules, and they fail for different reasons:

* **An undefined id is a dangling reference.** A reader who greps the ledger for it finds
  nothing and cannot tell whether the task was renumbered, never filed, or imagined. There
  is no judgement in this one — the id either appears in ``docs/todo.md`` or it does not.
* **A ``# TODO:`` or ``# BLOCKED:`` marker citing a TICKED id is a contradiction between
  two statements this repository treats as authoritative.** The marker says work remains;
  the ledger says it is done. Exactly one of them is wrong and the author of neither knows
  which. This is a failure rather than a warning, deliberately: a warning in a green CI log
  is read by nobody, the condition is mechanically decidable rather than a matter of taste,
  and every resolution is cheap and local — repoint the marker at the open task, delete it
  because the work really did land, or untick the entry. The thing that gets a gate
  disabled is firing on honest code, and honest code cannot hold this pair.

A docstring citing a ticked id is NOT a violation. Docstrings carry history on purpose
("extracted behind a RateLimiter ABC under TASK-0216"), and a completed task is the correct
citation for completed work. Only the markers claim something is still outstanding.

What this gate CANNOT do, stated plainly:

* it reads comments and docstrings only, not arbitrary string literals. Test fixtures hold
  deliberately-absent ids as data — ``tests/unit/scripts/test_check_blockers.py`` feeds the
  blockers gate an id no ledger will ever define, on purpose — and failing those would be
  firing on correct code. The cost is real: an id inside a dict literal or a
  ``pytest.mark.skip`` reason is invisible here;
* it reads ``.py`` files under ``src/``, ``tests/`` and ``scripts/``. Workflow YAML,
  fixtures and Markdown outside ``docs/`` are not covered;
* it checks that an id resolves and that a marker does not point at finished work. It
  cannot tell whether the id cited is the RIGHT one among the open entries.

Exit codes: ``0`` when every citation resolves, ``1`` on any violation or if the scan
parses nothing.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from pathlib import Path
from typing import Final, NamedTuple

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
SCANNED_TREES: Final[tuple[str, ...]] = ("src", "tests", "scripts")
TODO_RELATIVE: Final = "docs/todo.md"

_TASK: Final = re.compile(r"\bTASK-\d{4}\b")
_ENTRY: Final = re.compile(r"^- \[([ x])\] (TASK-\d{4}):", re.MULTILINE)

# CRITICAL: the three marker keywords are the ones CLAUDE.md permits as comments, and the
# split between them is the whole rule. `# TODO:` and `# BLOCKED:` assert outstanding work,
# so they contradict a ticked ledger entry. `# CRITICAL:` asserts an invariant and routinely
# cites the completed task that established it, which is correct and must stay green.
_PENDING_MARKER: Final = re.compile(r"^#\s*(?:TODO|BLOCKED):")
_INVARIANT_MARKER: Final = re.compile(r"^#\s*CRITICAL:")

_DOCSTRING_OWNERS: Final = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


class Citation(NamedTuple):
    """One ``TASK-`` id found in a comment or docstring, with where and how it was written."""

    path: str
    line: int
    identifier: str
    pending: bool


def _ledger(root: Path) -> tuple[set[str], set[str]]:
    """Return the ids ``docs/todo.md`` defines, and the subset it ticks as done."""
    todo = root / TODO_RELATIVE
    if not todo.is_file():
        return set(), set()

    defined: set[str] = set()
    ticked: set[str] = set()
    for box, identifier in _ENTRY.findall(todo.read_text(encoding="utf-8")):
        defined.add(identifier)
        if box == "x":
            ticked.add(identifier)
    return defined, ticked


def _source_files(root: Path) -> list[Path]:
    """Return every Python file under the scanned trees, in a stable order."""
    files: list[Path] = []
    for tree in SCANNED_TREES:
        directory = root / tree
        if not directory.is_dir():
            continue
        files += [
            path for path in sorted(directory.rglob("*.py")) if "__pycache__" not in path.parts
        ]
    return files


def _comments(text: str) -> dict[int, tuple[str, bool]]:
    """Return each comment keyed by its line, flagged when it belongs to a pending block.

    A marker opens a block: every ``#`` line immediately following it continues that marker,
    until a blank line, code, or a different marker ends it.
    """
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return {}

    comments: dict[int, tuple[str, bool]] = {}
    open_block = False
    previous = -2
    for token in tokens:
        if token.type != tokenize.COMMENT:
            continue
        line = token.start[0]
        body = token.string.strip()
        if _PENDING_MARKER.match(body):
            open_block = True
        elif _INVARIANT_MARKER.match(body) or line != previous + 1:
            open_block = False
        comments[line] = (token.string, open_block)
        previous = line
    return comments


def _docstring_lines(text: str) -> set[int]:
    """Return every source line covered by a module, class or function docstring."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()

    covered: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, _DOCSTRING_OWNERS):
            continue
        body = getattr(node, "body", [])
        if not body:
            continue
        first = body[0]
        if not isinstance(first, ast.Expr) or not isinstance(first.value, ast.Constant):
            continue
        if not isinstance(first.value.value, str):
            continue
        covered.update(range(first.value.lineno, (first.value.end_lineno or 0) + 1))
    return covered


def _citations(relative: str, text: str) -> list[Citation]:
    """Return every ``TASK-`` id written in a comment or a docstring of one file."""
    comments = _comments(text)
    docstrings = _docstring_lines(text)

    found: list[Citation] = []
    for number, line in enumerate(text.splitlines(), 1):
        if number in comments:
            source, pending = comments[number]
        elif number in docstrings:
            source, pending = line, False
        else:
            continue
        found += [
            Citation(relative, number, identifier, pending)
            for identifier in sorted(set(_TASK.findall(source)))
        ]
    return found


def check(root: Path = REPOSITORY_ROOT) -> list[str]:
    """Return one message per unresolved or contradicted citation, empty when all resolve."""
    defined, ticked = _ledger(root)
    if not defined:
        return [
            f"{TODO_RELATIVE} defines no task ids, so the scan had nothing to resolve "
            "against; the ledger is missing or no longer written as checklist entries"
        ]

    files = _source_files(root)
    if not files:
        trees = ", ".join(SCANNED_TREES)
        return [f"no Python files found under {trees}; the gate proved nothing"]

    citations: list[Citation] = []
    violations: list[str] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        citations += _citations(relative, path.read_text(encoding="utf-8"))

    for citation in citations:
        if citation.identifier not in defined:
            violations.append(
                f"{citation.path}:{citation.line}: cites {citation.identifier}, which "
                f"{TODO_RELATIVE} does not define"
            )
        elif citation.pending and citation.identifier in ticked:
            violations.append(
                f"{citation.path}:{citation.line}: a pending marker cites "
                f"{citation.identifier}, which {TODO_RELATIVE} ticks as done"
            )

    if not citations:
        return [
            f"scanned {len(files)} Python files and matched no TASK- id at all; either the "
            "code has stopped citing the ledger, or this scan no longer reads comments and "
            "docstrings the way they are written"
        ]
    return violations


def main(argv: list[str] | None = None) -> int:
    """Run the check, reporting every unresolved citation at once."""
    parser = argparse.ArgumentParser(description="Resolve every TASK- id cited in source.")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="repository root to scan (default: the checkout this script lives in)",
    )
    arguments = parser.parse_args(argv)

    violations = check(arguments.root)
    if not violations:
        return 0

    sys.stderr.write("source code cites a task the ledger does not support:\n")
    for violation in violations:
        sys.stderr.write(f"  {violation}\n")

    if any("does not define" in violation for violation in violations):
        sys.stderr.write(
            "\nfor an undefined id: the reference dangles. File the entry, or repoint the "
            "citation at the id that really covers the work. Do not delete the id alone — "
            "the sentence around it then explains nothing.\n"
        )
    if any("ticks as done" in violation for violation in violations):
        sys.stderr.write(
            "\nfor a pending marker on a ticked id: the marker says work remains and the "
            "ledger says it is finished, so one of the two is wrong. Repoint the marker at "
            "the open entry, delete it if the work landed, or untick the entry.\n"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
