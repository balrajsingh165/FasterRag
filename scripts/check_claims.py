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

The 2026-11-07 sweep (TASK-0309) found 24 defects this gate scored green, and two of its
three causes are closed here (TASK-0310). The scan root now covers ``README.md`` and
``CLAUDE.md`` as well as ``docs/`` — it read neither, so the literal sentence this gate
exists to stop ("``--watch`` is declared but not yet implemented", matching its own
pattern) survived on the page PyPI renders. And :data:`DENIAL_PATTERNS` now carries the
denial shapes that contain no denial verb, every one of which needed the word ``not`` to
match before: ``unbuilt``, ``pending (TASK-####)``, ``refuses startup until … ships``,
``not enforced by anything``.

What this gate CANNOT do, stated plainly because a check that oversells its coverage is the
same defect in miniature:

* **an umbrella task id keeps every denial citing it green, and no regex closes this.** The
  gate's only question is whether the cited id is ticked. TASK-0049 is one open id covering
  six vector-DB adapters; Chroma shipped under it on 2026-11-03 and Milvus in 348aa66 on
  2026-11-08, and the id correctly stays open for the rest. So ``README.md`` calling the Milvus
  adapter "pending (TASK-0049)" and ``migration-guide.md`` calling it "specified but
  unbuilt (TASK-0049)" are both false today and both pass — the new ``unbuilt`` and
  ``pending`` patterns match them and change nothing. Closing this needs denials to cite an
  id covering exactly the thing denied, or a per-item sub-ledger; it is not a pattern
  problem;
* it cannot see a denial that cites no task id at all — the overwhelming majority of English
  sentences about unbuilt things;
* it cannot see a denial that is false for any reason other than its task being ticked, such
  as work that shipped under a different id, or under no id;
* it only knows the phrasings in :data:`DENIAL_PATTERNS`, derived from real incidents. A
  fresh incident worded differently passes;
* it says nothing about the affirmative half of the provable-claims policy — an overclaim,
  a stale count, an unmeasured superlative. ``tests/unit/docs/test_counts_parity.py`` and
  ``tests/unit/docs/test_superlative_scan.py`` cover the two slices of that which are
  mechanically decidable.

It is a gate against the failures that have actually happened here, not a truth detector.

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
DOCS_DIRECTORY: Final = "docs"
TODO_NAME: Final = "todo.md"

# CRITICAL: README.md is the page PyPI renders and CLAUDE.md is loaded into every agent
# session as instructions, so a stale denial in either is read more often than one in
# docs/. The gate iterated docs/ only until TASK-0310, and that is how the --watch denial
# it was built to stop survived in README.md. A missing file here is itself a failure:
# losing a scan root silently is the defect, not a tidier scan.
ROOT_DOCUMENTS: Final[tuple[str, ...]] = ("README.md", "CLAUDE.md")

# CRITICAL: every phrasing here is taken from a sentence somebody actually wrote rather
# than imagined: "The runtime governor does not exist", "declared but not implemented",
# "none of these suites has been written", "Neither exists yet", "the cache_only
# degradation rung is not built", "adapters pending (TASK-0049)", "specified but unbuilt",
# "until TASK-0124 ships it". Widening this list is cheap, and widening it carelessly is
# how a gate starts firing on honest prose and gets disabled: "refuses to start" alone was
# rejected because cookbook.md, security.md, differentiators.md and python-api.md all use
# it for fail-fast behaviour that shipped, one of them beside a ticked id. The shape that
# denies something is "refuses to start UNTIL", never "refuses to start".
DENIAL_PATTERNS: Final[tuple[str, ...]] = (
    r"\b(?:does|do|did)\s+not\s+(?:yet\s+)?exist",
    r"\bneither\s+exists?(?:\s+yet)?",
    r"\bnot[\s-]+(?:yet[\s-]+)?implemented",
    r"\b(?:is|are|was|were)\s+not\s+(?:yet\s+)?built",
    r"\b(?:has|have)\s+not\s+(?:yet\s+)?been\s+written",
    r"\b(?:is|are|was|were)\s+not\s+(?:yet\s+)?written",
    r"\b(?:is|are|was|were|remains?|stays?)\s+(?:\w+\s+){0,2}unbuilt\b",
    r"\bspecified[\s-]*but[\s-]*unbuilt\b",
    r"\bpending\s*\(TASK-\d{4}\)",
    r"\brefuses?\s+(?:to\s+)?start(?:\s*up)?\s+until\b",
    r"\bnot\s+enforced\s+by\s+anything\b",
    r"\buntil\s+(?:\S+\s+){0,4}?(?:ships?|lands?)\b",
)

# CRITICAL: a changelog entry describes the repository at the date in its heading and is
# frozen afterwards, exactly as a ticked ledger entry is. "raises a ConfigError until
# TASK-0124 ships it" was true of the 2026-07-31 release and stays in the record; rewriting
# it to match today would destroy the history the file exists to keep. The cost is real and
# accepted: a standing note at the top of CHANGELOG.md is invisible to this gate, which is
# why tests/unit/docs/test_counts_parity.py reads that file instead.
EXEMPT_DOCUMENTS: Final[frozenset[str]] = frozenset({"docs/CHANGELOG.md"})

# CRITICAL: sentences whose ticked id names the task that FOUND the gap, not the task that
# will close it. "Row 19 is unbuilt twice over, found while fixing row 18 (TASK-0289)" is
# true today — TASK-0165 is the open id that builds the rerank timeout — and no pattern
# separates an attribution clause from a citation. This is the umbrella-id hole in
# miniature, recorded the way check_doc_truth.py records its duplicate ids: so the gate
# reports growth rather than history. Never add an entry to quiet a new failure; repoint
# the sentence at the id that covers what it denies, which is what the gate is asking for.
ATTRIBUTED_DENIALS: Final[tuple[tuple[str, str], ...]] = (
    ("docs/failure-modes.md", "Row 19 is unbuilt twice over"),
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

# CRITICAL: this break is used to attribute an id INSIDE one unit and never to form
# the units themselves. It consumes closing markup, because a bolded sentence ends
# `records.**` and the unit break above, wanting whitespace straight after the full
# stop, does not see that as an end at all. Widening the unit break instead split a
# bolded prose denial away from the citation in the sentence after it, which is a
# false negative with a live instance in failure-modes.md.
_CLAUSE_BREAK: Final = re.compile(r"(?<=[.!?])[*_`)\]]*\s+")

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


def _split(line: str) -> list[str]:
    """Split one line into comparison units, resolving a table row cell by cell."""
    if _TABLE_RULE.match(line):
        return []
    if _TABLE_ROW.match(line):
        return _row_units(line)
    return _SENTENCE_BREAK.split(line)


def _sentences(body: str) -> list[tuple[int, str]]:
    """Return each sentence of a document with the line it sits on.

    Counted per line rather than by searching the body for each unit: a reconstructed
    table cell carries ids its source line does not, so the search drifted forward and
    the gate named a line up to 76 off. A gate pointing at the wrong line is ignored.
    """
    stripped = _FENCE.sub(lambda match: "\n" * match.group(0).count("\n"), body)
    return [
        (number, piece)
        for number, line in enumerate(stripped.split("\n"), 1)
        for piece in _split(line)
        if piece.strip()
    ]


def scanned_documents(root: Path) -> list[Path]:
    """Return every document the gate reads: docs/ minus the ledger, plus the root pages."""
    docs = root / DOCS_DIRECTORY
    found = [path for path in sorted(docs.rglob("*.md")) if path.name != TODO_NAME]
    found += [root / name for name in ROOT_DOCUMENTS if (root / name).is_file()]
    return [path for path in found if path.relative_to(root).as_posix() not in EXEMPT_DOCUMENTS]


def _cited_by(denial: re.Match[str], unit: str) -> set[str]:
    """Return the ids the denial's own clause cites, or the unit's when it cites none.

    A table cell is one unit and was never split further, so a cell carrying a truthful
    open denial and, after it, a sentence about shipped work was failed for that
    neighbour's ticked id.
    """
    starts = [0] + [match.end() for match in _CLAUSE_BREAK.finditer(unit)]
    bounds = [(start, end) for start, end in zip(starts, [*starts[1:], len(unit)], strict=True)]
    own = next(
        (unit[start:end] for start, end in bounds if start <= denial.start() < end),
        unit,
    )
    return set(_TASK.findall(own)) or set(_TASK.findall(unit))


def _attributed(relative: str, sentence: str) -> bool:
    """Say whether this sentence is a recorded discovery attribution rather than a denial."""
    return any(
        relative == document and fragment in sentence for document, fragment in ATTRIBUTED_DENIALS
    )


def check(root: Path = REPOSITORY_ROOT) -> list[str]:
    """Return one message per stale denial, empty when every denial cites open work."""
    docs = root / DOCS_DIRECTORY
    if not docs.is_dir():
        return [f"{docs} is not a directory; the claim scan read nothing"]

    ticked = _ticked_ids(docs / TODO_NAME)

    violations: list[str] = []
    paths = scanned_documents(root)
    denials = 0

    for path in paths:
        for line, sentence in _sentences(path.read_text(encoding="utf-8")):
            denial = _DENIAL.search(sentence)
            if denial is None:
                continue
            denials += 1
            relative = path.relative_to(root).as_posix()
            if _attributed(relative, sentence):
                continue
            violations += [
                f"{relative}:{line}: says {denial.group(0)!r} but cites {identifier}, "
                f"which docs/{TODO_NAME} ticks as done"
                for identifier in sorted(_cited_by(denial, sentence) & ticked)
            ]

    unread = [name for name in ROOT_DOCUMENTS if not (root / name).is_file()]
    if unread:
        return [
            f"{', '.join(unread)} absent from {root}, so the scan skipped the documents "
            "read most often; a narrowed scan root is the defect TASK-0310 closed"
        ]
    if not any(path.relative_to(root).parts[0] == DOCS_DIRECTORY for path in paths):
        return [f"{docs} holds no documents to scan; the gate proved nothing"]
    if denials == 0:
        return [
            f"scanned {len(paths)} documents and matched no denial phrasing at all; "
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
