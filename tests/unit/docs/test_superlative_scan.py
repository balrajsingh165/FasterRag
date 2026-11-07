"""A superiority claim must cite a ledger entry or be written as a goal.

The provable-claims policy names its own enforcement: *"periodically grep docs for
superlatives and verify each is ledger-linked"*. That makes enforcement exactly as reliable
as somebody remembering, and the 2026-11-07 sweep (TASK-0309) found 24 defects that had
accumulated in that gap. This file is the grep, run on every push.

The rule: a policy-listed superlative must sit in a sentence that either cites a
`BENCH-####` ledger entry or carries a goal marker (`goal`, `aims to`, `targets`,
`ambition`). A superlative inside quotation marks or backticks is a *mention* — the
documents discuss the word constantly, because the policy is itself documented — and is
never a claim. Everything else is an unmeasured assertion of superiority, which the policy
calls a bug.

What this file CANNOT catch, stated because a check that oversells its coverage is the same
defect it exists to stop:

* **a superiority claim made without a listed word.** "No other framework does this",
  "nothing else comes close", "an order of magnitude" — all unmeasured, all invisible here.
  The scan knows a vocabulary, not a meaning;
* **bare `best` and bare `most <adjective>`, deliberately.** The policy names both, and
  both were measured against this tree before being left out: of 5 occurrences of bare
  `best`, 2 are ordinary English (`presenting the best of an arbitrary subset as the best
  available`), and of 30 occurrences of `most <word>`, roughly 27 are (`at most four
  lines`, `most frameworks`, `the most useful finding`). A gate that fires on ordinary
  English gets disabled, taking the 3 real hits with it. The policy's periodic human grep
  remains the only control over those two words;
* **`cheapest`, left out on the same evidence and with a sharper reason.** Its one
  occurrence in this tree — "it is the cheapest insurance against the next twenty" — is a
  statement about the cost of doing a piece of work, not about fasterRag, and no ledger
  entry could ever support or refute it. `quickest` is kept because its one occurrence is
  in stable reference prose where allowlisting it records the judgement in public;
* **a goal marker used to launder a claim.** "Our goal is to be the fastest, and we are"
  passes. The marker is evidence of intent, not of honesty;
* **anything in `src/` docstrings.** `docs/code-notes.md` mirrors them and is excluded as a
  generated file, so a superlative written in a docstring is seen by nothing;
* **whether a cited `BENCH-####` entry actually supports the claim beside it.** That is a
  human reading, and the ledger's own rules (benchmarks.md §5) are where it belongs.
"""

import re
from pathlib import Path
from typing import Final

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"

# CRITICAL: the policy names "fastest", "most reliable" and "best". This list is those,
# plus the vocabulary this repository has actually used about itself, minus the two words
# measured too noisy to block on (see the module docstring). Adding a word is cheap; adding
# one that fires on ordinary English is how the whole scan gets switched off.
POLICY_SUPERLATIVES: Final[tuple[str, ...]] = (
    r"fastest",
    r"quickest",
    r"most\s+(?:reliable|efficient|optimi[sz]ed|accurate|advanced|complete|powerful|scalable|performant)",
    r"best[\s-]in[\s-]class",
    r"industry[\s-]leading",
    r"state[\s-]of[\s-]the[\s-]art",
    r"world[\s-]class",
    r"unmatched",
    r"unrivall?ed",
    r"outperforms?",
    r"faster\s+than\s+any",
)

# CRITICAL: a generated mirror of src/ docstrings, maintained by scripts/trim_docstrings.py.
# Scanning it would pin allowlist fragments to text that is rewritten whenever a docstring
# is, and would report every finding twice. The cost is in the module docstring: a
# superlative written in a docstring is invisible to this scan.
EXCLUDED_DOCUMENTS: Final[frozenset[str]] = frozenset({"docs/code-notes.md"})

ALLOWLIST: Final[tuple[tuple[str, str], ...]] = (
    ("docs/cli-reference.md", "the quickest way to see what a deployment customised"),
)
"""Sentences where a listed word is English rather than a claim about the product.

One entry, and it should stay close to that. "The quickest way to see what a deployment
customised" ranks two ways of reading a config file, not fasterRag against a competitor;
no ledger entry could support or refute it. An entry here is a judgement recorded in
public, which is the point — a carve-out nobody can see is how the 24 defects accumulated.
"""

_FENCE = re.compile(r"```.*?```", re.DOTALL)
_SUPERLATIVE = re.compile("|".join(POLICY_SUPERLATIVES), re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\|")
_BENCH = re.compile(r"\bBENCH-\d{4}\b")
_GOAL = re.compile(
    r"\bgoals?\b|\baims?\s+to\b|\btargets?\b|\bambitions?\b|\baspires?\b|\bintends?\s+to\b",
    re.IGNORECASE,
)

# CRITICAL: straight single quotes are NOT a quoting form here, because an apostrophe opens
# one that never closes — "the project's central "fastest" ambition" would swallow the rest
# of the sentence. Every mention in this repository is inside double quotes, typographic
# quotes or backticks, including `'fastest'` nested inside a double-quoted policy excerpt.
_QUOTED = re.compile(r"\"[^\"]*\"|“[^”]*”|`[^`]*`")


def _body(path: Path) -> str:
    """Return a document with fenced blocks removed, line numbering preserved."""
    text = path.read_text(encoding="utf-8")
    return _FENCE.sub(lambda match: "\n" * match.group(0).count("\n"), text)


def documents() -> list[tuple[str, str]]:
    """Return `(path, body)` for every prose document the policy governs."""
    paths = [path for path in sorted(DOCS.rglob("*.md")) if path.name != "todo.md"]
    paths += [ROOT / "README.md", ROOT / "CLAUDE.md"]
    return [
        (name, _body(path))
        for path in paths
        if (name := path.relative_to(ROOT).as_posix()) not in EXCLUDED_DOCUMENTS
    ]


def _mentioned(sentence: str, start: int, end: int) -> bool:
    """Say whether the match at this span sits inside quotation marks or backticks."""
    return any(span.start() <= start and end <= span.end() for span in _QUOTED.finditer(sentence))


def _allowed(name: str, sentence: str) -> bool:
    return any(name == document and fragment in sentence for document, fragment in ALLOWLIST)


def candidates(name: str, body: str) -> list[tuple[int, str, str]]:
    """Return every `(line, word, sentence)` holding a listed superlative, exempt or not."""
    found: list[tuple[int, str, str]] = []
    for number, line in enumerate(body.split("\n"), 1):
        for sentence in _SENTENCE_BREAK.split(line):
            found += [
                (number, match.group(0), sentence)
                for match in _SUPERLATIVE.finditer(sentence)
                if not _mentioned(sentence, match.start(), match.end())
            ]
    return found


def findings(name: str, body: str) -> list[str]:
    """Return one message per superlative that cites no ledger entry and states no goal."""
    return [
        f"{name}:{number}: {word!r} claims superiority with no BENCH- citation and no "
        f"goal marker — {sentence.strip()[:120]}"
        for number, word, sentence in candidates(name, body)
        if not _BENCH.search(sentence)
        and not _GOAL.search(sentence)
        and not _allowed(name, sentence)
    ]


def repository_candidates() -> list[tuple[str, int, str]]:
    return [
        (name, number, word)
        for name, body in documents()
        for number, word, _ in candidates(name, body)
    ]


def repository_findings() -> list[str]:
    return [message for name, body in documents() for message in findings(name, body)]


def test_the_documents_were_read() -> None:
    """Vacuity guard: a scan over an empty set reports a clean tree."""
    assert len(documents()) >= 30


def test_the_scan_matches_superlatives_at_all() -> None:
    """Vacuity guard: this repository discusses superlatives constantly, so zero is a bug."""
    assert len(repository_candidates()) >= 10


def test_the_quoting_rule_still_exempts_something() -> None:
    """Vacuity guard: if nothing is exempted as a mention, the quoting rule stopped working."""
    quoted = [
        (name, number)
        for name, body in documents()
        for number, line in enumerate(body.split("\n"), 1)
        for match in _SUPERLATIVE.finditer(line)
        if _mentioned(line, match.start(), match.end())
    ]

    assert len(quoted) >= 5


def test_no_document_claims_superiority_without_a_measurement() -> None:
    """The policy's rule, enforced instead of remembered."""
    violations = repository_findings()

    assert violations == [], "unmeasured superlatives: " + "; ".join(violations)


def test_a_bare_claim_is_caught() -> None:
    """The shape the policy forbids: an assertion with nothing behind it."""
    assert findings("docs/x.md", "fasterRag is the fastest RAG framework available.\n")


def test_a_goal_is_not_a_claim() -> None:
    """README.md, CLAUDE.md and scope.md all state the ambition, which the policy permits."""
    assert findings("docs/x.md", "Its goal is to be the fastest, most efficient framework.\n") == []


def test_a_ledger_citation_licenses_the_claim() -> None:
    """The whole point of the ledger: a measured claim is allowed to be made."""
    body = "Ingest is the fastest path for bulk corpora (BENCH-0001).\n"

    assert findings("docs/x.md", body) == []


def test_a_quoted_superlative_is_a_mention() -> None:
    """Fourteen sentences in this repository discuss the word; none of them claims anything."""
    body = 'Every "fastest" claim must trace to an entry in the ledger.\n'

    assert findings("docs/x.md", body) == []


def test_a_backticked_superlative_is_a_mention() -> None:
    assert findings("docs/x.md", "Grep for `fastest` before every release.\n") == []


def test_an_apostrophe_does_not_open_a_quotation() -> None:
    """Straight single quotes are not a quoting form; treating them as one hid real claims."""
    body = "The project's fastest path is unmeasured.\n"

    assert findings("docs/x.md", body) != []


def test_a_claim_in_a_fenced_block_is_sample_output() -> None:
    """A fence holds a command or its output, not the document asserting anything."""
    assert candidates("docs/x.md", _FENCE.sub("", "```\nthe fastest run\n```\n")) == []


def test_the_allowlist_suppresses_the_sentence_it_names() -> None:
    """The mechanism itself, proven rather than assumed from a green real-tree run."""
    document, fragment = ALLOWLIST[0]
    body = f"Some lead-in. {fragment} and nothing more.\n"

    assert findings("docs/other.md", body) != []
    assert findings(document, body) == []


def test_the_allowlist_still_names_live_prose() -> None:
    """An entry kept after its sentence is rewritten is a standing hole nobody sees."""
    stale = [
        f"{document}: {fragment!r}"
        for document, fragment in ALLOWLIST
        if fragment not in (ROOT / document).read_text(encoding="utf-8")
    ]

    assert stale == [], f"ALLOWLIST names prose that no longer exists: {stale}"


def test_the_allowlist_stays_small() -> None:
    """Every entry is a judgement call; a growing list means the rule is being worked around."""
    assert len(ALLOWLIST) <= 3
    assert all((ROOT / document).is_file() for document, _ in ALLOWLIST)
