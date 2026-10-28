"""Fail when a document calls a capability unbuilt while its cited task is ticked.

Three separate rounds found documents asserting that shipped work did not exist. The D9
runtime cost governor was described in ``internals.md`` as not existing while
``services/budget.py`` had been shipped and wired for two and a half months;
``differentiators.md`` and ``quickstart.md`` called ``--watch`` declared but not implemented
after it was implemented; ``failure-modes.md`` said the partition-injection suite had not
been written after it was. Each was caught by eye, by somebody who happened to be reading
nearby, which is not a control.

The recurring shape is machine-checkable without understanding the prose: a sentence that
denies a capability exists, carrying the id of the task that would build it. If that id is
ticked in ``docs/todo.md``, the sentence is stale by the ledger's own account. The denial
itself is never the failure — ``the cache_only rung is not built (TASK-0159)`` is true, and
stays green for exactly as long as TASK-0159 stays open.

What this gate CANNOT do, stated plainly because a check that oversells its coverage is the
same defect in miniature:

* it cannot see a denial that cites no task id at all — the overwhelming majority of English
  sentences about unbuilt things;
* it cannot see a denial that is false for any reason other than its task being ticked, such
  as work that shipped under a different id, or under no id;
* it only knows the phrasings in :data:`DENIAL_PATTERNS`, derived from the three real
  incidents above. A fourth incident worded differently passes.

It is a gate against the one failure that has actually happened three times, not a truth
detector.

Exit codes: ``0`` when every denial cites open work, ``1`` on any stale denial or if the
scan parses nothing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
DOCS: Final = REPOSITORY_ROOT / "docs"
TODO_NAME: Final = "todo.md"

# CRITICAL: every phrasing here is taken from a sentence somebody actually wrote rather
# than imagined: "The runtime governor does not exist", "declared but not implemented",
# "none of these suites has been written", "Neither exists yet", "the cache_only
# degradation rung is not built". Widening this list is cheap, and widening it carelessly
# is how a gate starts firing on honest prose and gets disabled.
DENIAL_PATTERNS: Final[tuple[str, ...]] = (
    r"\b(?:does|do|did)\s+not\s+(?:yet\s+)?exist",
    r"\bneither\s+exists?(?:\s+yet)?",
    r"\bnot[\s-]+(?:yet[\s-]+)?implemented",
    r"\b(?:is|are|was|were)\s+not\s+(?:yet\s+)?built",
    r"\b(?:has|have)\s+not\s+(?:yet\s+)?been\s+written",
    r"\b(?:is|are|was|were)\s+not\s+(?:yet\s+)?written",
)

_DENIAL: Final = re.compile("|".join(DENIAL_PATTERNS), re.IGNORECASE)
_TASK: Final = re.compile(r"\bTASK-\d{4}\b")
_TICKED: Final = re.compile(r"^- \[x\] (TASK-\d{4}):", re.MULTILINE)
_FENCE: Final = re.compile(r"```.*?```", re.DOTALL)

# CRITICAL: a denial and the id it cites are paired at sentence scope, never wider. One
# markdown line in quickstart.md already carries a ticked "TASK-0242 ✅" sentence next to a
# truthful "the cache_only degradation rung is not built (TASK-0159)", and pairing the two
# would fail an honest document. Colons and semicolons are deliberately NOT breaks — a
# denial routinely introduces its own citation across one ("Layers 1.7, 1.8 and 1.10 do not
# exist: ... (TASK-0243)"), and breaking there would drop the id and the claim with it.
_SENTENCE_BREAK: Final = re.compile(r"(?<=[.!?])\s+|\n|\|")

# CRITICAL: a table denial is paired with its OWN cell's id, and only with the rest of
# the row when its cell cites none. Both halves are load-bearing and pull opposite ways.
# Splitting on the pipe alone is a false negative with a live instance:
# release-readiness.md said an "unbuilt spend cap ... enforcement that does not exist"
# in one cell while naming a ticked TASK-0242 in the cell beside it, and the first
# version of this gate scored it green. Joining rows unconditionally is a false positive:
# a row whose denial cell truthfully cites an OPEN id would be failed by a ticked id that
# merely happens to sit in a neighbouring cell. The cell's own citation wins because a
# writer who put an id next to the claim meant that id.
_TABLE_ROW: Final = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_RULE: Final = re.compile(r"^\s*\|[\s:|-]+\|\s*$")


def _ticked_ids(todo: Path) -> set[str]:
    """Return the ids the ledger marks completed."""
    if not todo.is_file():
        return set()
    return set(_TICKED.findall(todo.read_text(encoding="utf-8")))


def _row_units(line: str) -> list[str]:
    """Return one unit per table cell, lending the row's ids to a cell that cites none."""
    cells = line.strip().strip("|").split("|")
    elsewhere = sorted({task for cell in cells for task in _TASK.findall(cell)})
    units: list[str] = []
    for cell in cells:
        if not cell.strip():
            continue
        if _TASK.search(cell) or not elsewhere:
            units.append(cell)
            continue
        units.append(f"{cell} ({' '.join(elsewhere)})")
    return units


def _split(body: str) -> list[str]:
    """Split into comparison units, resolving table rows cell by cell."""
    units: list[str] = []
    for line in body.split("\n"):
        if _TABLE_RULE.match(line):
            continue
        if _TABLE_ROW.match(line):
            units.extend(_row_units(line))
            continue
        units.extend(_SENTENCE_BREAK.split(line))
    return units


def _sentences(body: str) -> list[tuple[int, str]]:
    """Return each sentence of a document with the line it starts on.

    Fenced blocks are dropped first: a phrase inside one is sample output or a quoted log,
    not the document asserting anything. Table cells are split apart along with sentences,
    since two cells of one row are two unrelated statements.
    """
    stripped = _FENCE.sub(lambda match: "\n" * match.group(0).count("\n"), body)

    units: list[tuple[int, str]] = []
    offset = 0
    for piece in _split(stripped):
        start = stripped.find(piece, offset) if piece else offset
        if start < 0:
            start = offset
        if piece.strip():
            units.append((stripped[:start].count("\n") + 1, piece))
        offset = start + len(piece)
    return units


def check(docs: Path = DOCS) -> list[str]:
    """Return one message per stale denial, empty when every denial cites open work.

    Args:
        docs: Documentation directory to scan. Parameterised so the gate itself can be run
            against a constructed tree — a check nobody has watched fail is not a check.

    A scan that reads no documents, or finds no denials at all, is reported as a failure
    rather than a pass. A parser that silently matches nothing looks identical to a clean
    repository, and the repository is never that clean.
    """
    if not docs.is_dir():
        return [f"{docs} is not a directory; the claim scan read nothing"]

    ticked = _ticked_ids(docs / TODO_NAME)

    violations: list[str] = []
    documents = 0
    denials = 0

    for path in sorted(docs.rglob("*.md")):
        if path.name == TODO_NAME:
            continue
        documents += 1
        for line, sentence in _sentences(path.read_text(encoding="utf-8")):
            denial = _DENIAL.search(sentence)
            if denial is None:
                continue
            denials += 1
            relative = path.relative_to(docs.parent).as_posix()
            violations += [
                f"{relative}:{line}: says {denial.group(0)!r} but cites {identifier}, "
                f"which docs/{TODO_NAME} ticks as done"
                for identifier in sorted(set(_TASK.findall(sentence)) & ticked)
            ]

    if documents == 0:
        return [f"{docs} holds no documents to scan; the gate proved nothing"]
    if denials == 0:
        return [
            f"scanned {documents} documents and matched no denial phrasing at all; "
            "either every unbuilt capability has stopped being documented, or "
            "DENIAL_PATTERNS no longer matches how this repository writes"
        ]
    return violations


def main(argv: list[str] | None = None) -> int:
    """Run the check, reporting every stale denial at once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    violations = check()
    if not violations:
        return 0

    sys.stderr.write("a document calls shipped work unbuilt:\n")
    for violation in violations:
        sys.stderr.write(f"  {violation}\n")
    sys.stderr.write(
        "\nthe ledger ticks the task, so either the sentence is stale and should describe "
        "what shipped, or the tick is wrong. Do not delete the task id to quiet this.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
