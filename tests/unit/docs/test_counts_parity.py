"""Numbers the documents state about the tree must be the numbers the tree holds.

Four of the 24 findings in the 2026-11-07 provable-claims sweep (TASK-0309) were stale
counts: a benchmark ledger described as holding two entries while it held three, LOC
figures in `CLAUDE.md` quoted from a measurement two weeks old, a `quality` job described
as running four doc gates while it ran five. Counts are the cheapest defect class to gate
and the easiest to let rot, because nobody re-counts prose — the sentence stays grammatical
as the number goes wrong, and every reader downstream treats it as checked.

Only numbers with a mechanical denominator are pinned here. A count the tree can produce in
one expression is checkable; "twelve flagship capabilities" is an editorial decision about
what counts as flagship, and asserting it would invent a denominator the documents do not
define. Every pin below is read out of the prose rather than hard-coded, so correcting the
document is the fix and no number lives in two places.

What this file CANNOT catch, stated because a check that oversells its coverage is the same
defect it exists to stop:

* a count phrased in a shape the patterns here do not know. These are regexes over English,
  derived from the sentences this repository has actually written;
* a count that is wrong in a direction the tree cannot see — "about 230 source modules" is
  checkable, "four embedding providers" is only checkable if provider means one directory;
* whether a number is *meaningful*. `LOC_TOLERANCE` admits growth between measurements, so
  a figure can be stale-but-passing for a while by design;
* anything about the affirmative claims that carry no number at all. That is
  `test_superlative_scan.py`, and between them they cover the mechanical slice of the
  provable-claims policy, not the policy.
"""

import re
from pathlib import Path
from typing import Final

import yaml

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
BENCHMARKS = DOCS / "benchmarks.md"
INSTRUCTIONS = ROOT / "CLAUDE.md"
TESTING_STRATEGY = DOCS / "testing-strategy.md"
WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"

NUMBER_WORDS: Final[dict[str, int]] = {
    "no": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_BENCH_HEADING = re.compile(r"^### (BENCH-\d{4})\b", re.M)
_LEDGER_SECTION = re.compile(r"^## Ledger[ \t]*$.*?(?=^## |\Z)", re.M | re.S)
"""CRITICAL: anchored to end-of-line because `## Ledger rules` is a different section.

Matching `^## Ledger\\b` found the rules section, stopped at `## Entry template`, and
returned a ledger of zero entries — a parser that agrees with no stated count at all.
"""

_COUNT = r"(\d+|" + "|".join(NUMBER_WORDS) + r")"

# CRITICAL: the negative lookahead is the whole rule, not a tidy-up. "the benchmark ledger
# holds no entry from either" and "holds no entry from it" are claims about a *subset* —
# load and soak measurements — and both are true. Dropping the lookahead turns two correct
# sentences in testing-strategy.md into failures, which is how a gate gets disabled.
_LEDGER_CLAIM = re.compile(
    r"holds?\s+\*{0,2}(?:currently\s+)?\*{0,2}" + _COUNT + r"\*{0,2}\s+entr(?:y|ies)(?!\s+from\b)",
    re.I,
)

_LOC_CLAIM = re.compile(r"([\d.]+)k\s+LOC\s+(?:under\s+`src/`|of\s+tests)", re.I)
_CASE_CLAIM = re.compile(r"`(tests/[A-Za-z0-9_/]+\.py)`[^.|]{0,40}?\((\d+)\s+cases")
_TEST_DEF = re.compile(r"^\s*(?:async\s+)?def\s+test_", re.M)
_GATE_CLAIM = re.compile(_COUNT + r"\s+scripts?\s+runs?\s+in\s+the\s+`quality`\s+job", re.I)
_GATE_STEP = re.compile(r"python\s+scripts/(check_[a-z_]+\.py)")

LOC_TOLERANCE: Final = 0.10
"""How far a stated LOC figure may drift from the tree before it is called stale.

A figure introducing "current repository state" is read as describing today. Growth between
measurements is expected, so an exact pin would fire on every merge; 10% is the band at
which "27.5k" stops being a fair description and re-measuring becomes the cheaper fix.
"""


def _documents() -> list[Path]:
    """Return the prose the count scan reads: docs/ minus the ledger, plus the root pages."""
    found = [path for path in sorted(DOCS.rglob("*.md")) if path.name != "todo.md"]
    return [*found, ROOT / "README.md", INSTRUCTIONS]


def _body(path: Path) -> str:
    """Return a document with fenced blocks removed, line numbering preserved."""
    text = path.read_text(encoding="utf-8")
    return _FENCE.sub(lambda match: "\n" * match.group(0).count("\n"), text)


def _as_int(token: str) -> int:
    return int(token) if token.isdigit() else NUMBER_WORDS[token.lower()]


def _sited(path: Path, body: str, pattern: re.Pattern[str]) -> list[tuple[str, re.Match[str]]]:
    """Return every match with the `file:line` it sits on."""
    return [
        (f"{path.relative_to(ROOT).as_posix()}:{body[: match.start()].count(chr(10)) + 1}", match)
        for match in pattern.finditer(body)
    ]


def ledger_entries() -> list[str]:
    """Return the BENCH ids the ledger actually holds.

    Two independent exclusions, because the naive count is wrong by one: the entry template
    is a `### BENCH-0001` heading inside a fenced block, and it sits outside `## Ledger`.
    """
    section = _LEDGER_SECTION.search(_body(BENCHMARKS))
    if section is None:
        raise AssertionError("benchmarks.md has no '## Ledger' section")
    return _BENCH_HEADING.findall(section.group(0))


def ledger_count_claims() -> list[tuple[str, int]]:
    """Return every `file:line` that states how many entries the benchmark ledger holds."""
    claims: list[tuple[str, int]] = []
    for path in _documents():
        body = _body(path)
        claims += [
            (site, _as_int(match.group(1)))
            for site, match in _sited(path, body, _LEDGER_CLAIM)
            if "benchmark" in body[max(0, match.start() - 200) : match.start()].lower()
            or "ledger" in body[max(0, match.start() - 200) : match.start()].lower()
        ]
    return claims


def measured_loc() -> dict[str, int]:
    """Return the line counts CLAUDE.md's figures describe."""
    return {
        area: sum(
            len(path.read_bytes().split(b"\n")) - 1 for path in sorted((ROOT / area).rglob("*.py"))
        )
        for area in ("src", "tests")
    }


def loc_claims() -> list[tuple[str, str, float]]:
    """Return CLAUDE.md's `(site, area, thousands-of-lines)` figures, in document order."""
    body = _body(INSTRUCTIONS)
    return [
        (site, "src" if "src/" in match.group(0) else "tests", float(match.group(1)))
        for site, match in _sited(INSTRUCTIONS, body, _LOC_CLAIM)
    ]


def case_count_claims() -> list[tuple[str, str, int]]:
    """Return every `(site, test file, stated case count)` a document asserts."""
    claims: list[tuple[str, str, int]] = []
    for path in _documents():
        body = _body(path)
        claims += [
            (site, match.group(1), int(match.group(2)))
            for site, match in _sited(path, body, _CASE_CLAIM)
        ]
    return claims


def quality_job_gates() -> list[str]:
    """Return the `check_*.py` scripts the CI quality job actually runs."""
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    steps = workflow["jobs"]["quality"]["steps"]
    return sorted({name for step in steps for name in _GATE_STEP.findall(str(step.get("run", "")))})


def gate_count_claims() -> list[tuple[str, int]]:
    """Return every stated count of the doc gates running in the quality job."""
    body = _body(TESTING_STRATEGY)
    return [
        (site, _as_int(match.group(1)))
        for site, match in _sited(TESTING_STRATEGY, body, _GATE_CLAIM)
    ]


def test_the_documents_were_read() -> None:
    """Vacuity guard: a scan over an empty set reports perfect parity."""
    assert len(_documents()) >= 30


def test_the_ledger_was_parsed() -> None:
    """Vacuity guard: an unparsed ledger would make every count claim agree with nothing."""
    assert len(ledger_entries()) >= 3


def test_the_entry_template_is_not_counted_as_an_entry() -> None:
    """The naive `### BENCH-` count is 4 and the true answer is 3; TASK-0310 made that error."""
    naive = _BENCH_HEADING.findall(BENCHMARKS.read_text(encoding="utf-8"))

    assert len(naive) == len(ledger_entries()) + 1
    assert sorted(set(naive)) == sorted(set(ledger_entries()))


def test_every_stated_ledger_size_matches_the_ledger() -> None:
    """A document that miscounts the ledger miscounts the evidence behind every claim."""
    held = len(ledger_entries())
    wrong = [f"{site} says {stated}" for site, stated in ledger_count_claims() if stated != held]

    assert wrong == [], f"benchmarks.md holds {held} entries; contradicted by: {wrong}"


def test_the_ledger_size_is_stated_somewhere() -> None:
    """Vacuity guard: the pin is worthless if the phrasings stop matching any prose."""
    assert len(ledger_count_claims()) >= 3


def test_the_loc_figures_were_parsed() -> None:
    """Vacuity guard: CLAUDE.md states one figure for src/ and one for tests/."""
    assert sorted(area for _, area, _ in loc_claims()) == ["src", "tests"]


def test_the_stated_loc_figures_still_describe_the_tree() -> None:
    """CLAUDE.md introduces these as current repository state, so they must stay current."""
    measured = measured_loc()
    drifted = [
        f"{site}: {area}/ stated {stated}k, measured {measured[area] / 1000:.1f}k "
        f"({abs(measured[area] - stated * 1000) / (stated * 1000):.0%} off)"
        for site, area, stated in loc_claims()
        if abs(measured[area] - stated * 1000) / (stated * 1000) > LOC_TOLERANCE
    ]

    assert drifted == [], f"CLAUDE.md's LOC figures no longer describe the tree: {drifted}"


def test_the_case_counts_were_parsed() -> None:
    """Vacuity guard: five documents cite a test file with its case count."""
    assert len(case_count_claims()) >= 5


def test_every_named_test_file_exists() -> None:
    """A proof cited with a case count is worth less than none if the file is gone."""
    missing = [
        f"{site} names {name}"
        for site, name, _ in case_count_claims()
        if not (ROOT / name).is_file()
    ]

    assert missing == [], f"documents cite test files absent from the tree: {missing}"


def test_every_stated_case_count_matches_the_file() -> None:
    """`(8 cases)` is a measurement of a file, and the file is the only thing that can say."""
    wrong = [
        f"{site}: {name} stated {stated}, holds {found}"
        for site, name, stated in case_count_claims()
        if (ROOT / name).is_file()
        and (found := len(_TEST_DEF.findall((ROOT / name).read_text(encoding="utf-8")))) != stated
    ]

    assert wrong == [], f"documents miscount the tests they cite as proof: {wrong}"


def test_the_quality_job_gates_were_found() -> None:
    """Vacuity guard: a workflow that parses to nothing would agree with any claim."""
    assert len(quality_job_gates()) >= 4


def test_the_stated_doc_gate_count_matches_the_workflow() -> None:
    """testing-strategy.md enumerates the doc gates; a sixth one added silently is invisible."""
    gates = quality_job_gates()
    wrong = [
        f"{site} says {stated}" for site, stated in gate_count_claims() if stated != len(gates)
    ]

    assert wrong == [], (
        f"the quality job runs {len(gates)} gates ({gates}); contradicted by: {wrong}"
    )


def test_the_doc_gate_count_is_stated_somewhere() -> None:
    """Vacuity guard: the sentence this pins is one rewrite away from matching nothing."""
    assert len(gate_count_claims()) >= 1
