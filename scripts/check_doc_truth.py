"""Fail when the always-loaded docs claim a repository state that the tree contradicts.

The first formal audit (2026-08-02) found ``CLAUDE.md`` and ``README.md`` both asserting a
documentation-only repository with no implementation code, while ``src/fasterrag`` held
around two hundred modules and a passing test suite. Nothing caught it, because nothing was
looking: every other gate checks the code against itself, and none checks the prose against
the code.

That inversion is worse than an ordinary stale sentence. ``CLAUDE.md`` is loaded into every
agent session as instructions, so a false "do not write implementation code" either stops
work that should happen or gets ignored — and an instruction file that is routinely ignored
stops being an instruction file.

The check is deliberately narrow. It asserts the two or three facts that are cheap to verify
and expensive to get wrong, and it says nothing about prose quality. A doc gate that fires on
judgement calls gets disabled.

Exit codes: ``0`` when the claims hold, ``1`` when any is contradicted.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]

# Phrases that assert there is no implementation code. Matched case-insensitively against
# the always-loaded docs; any hit while `src/` holds modules is the audit's exact inversion.
NO_CODE_CLAIMS: Final[tuple[str, ...]] = (
    r"documentation[\s-]only",
    r"no implementation code exists",
    r"no code (?:exists|has been written)",
    r"the code does not exist yet",
)

GUARDED_DOCS: Final[tuple[str, ...]] = ("CLAUDE.md", "README.md")

# A source tree this small is a scaffold, not an implementation; the claim would be fair.
IMPLEMENTATION_THRESHOLD: Final = 20

# CRITICAL: collisions already on `main`, recorded so the gate reports growth rather than
# history. Each is two distinct pieces of work that were assigned one id by parallel sessions
# reading the ledger at the same moment. Ticked entries are append-only and frozen, so these
# are not renumbered; TASK-0184's two live citations both mean its dashboard entry. Never add
# to this map to silence a new collision — pick the next free id instead.
KNOWN_DUPLICATE_IDS: Final[dict[str, int]] = {
    "TASK-0184": 2,
    "TASK-0185": 2,
    "TASK-0186": 2,
}


def _python_modules(root: Path) -> int:
    """Return how many Python modules the package actually ships."""
    package = root / "src" / "fasterrag"
    if not package.is_dir():
        return 0
    return sum(1 for path in package.rglob("*.py") if "__pycache__" not in path.parts)


def _strip_code_blocks(text: str) -> str:
    """Remove fenced code blocks.

    A phrase inside a fenced block is an example, a log line, or a quoted historical note —
    not the document asserting anything about the present.
    """
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def _duplicate_task_ids(root: Path) -> list[str]:
    """Return one message per ``TASK-`` id that defines more than one entry.

    An id is how code, docs, and blockers.md point at a piece of work, so two entries sharing
    one is a reference that resolves to two different things. Three collisions
    (TASK-0184/0185/0186, six distinct pieces of work) reached `main` unnoticed because
    nothing looked — parallel work picks "the next free id" by reading a ledger that a
    sibling has already extended.

    Existing collisions are reported but not repaired here: ticked entries are append-only
    and frozen by CLAUDE.md, so renumbering history is not this gate's call. What it does is
    stop the count from growing.
    """
    ledger = root / "docs" / "todo.md"
    if not ledger.is_file():
        return []

    entries = re.findall(r"^- \[[ x]\] (TASK-\d{4}):", ledger.read_text(encoding="utf-8"), re.M)
    seen: dict[str, int] = {}
    for identifier in entries:
        seen[identifier] = seen.get(identifier, 0) + 1

    duplicates = sorted(
        name for name, count in seen.items() if count > KNOWN_DUPLICATE_IDS.get(name, 1)
    )
    return [
        f"docs/todo.md: {name} defines {seen[name]} entries; an id must name one piece of work"
        for name in duplicates
    ]


def check(root: Path = REPOSITORY_ROOT) -> list[str]:
    """Return one message per contradicted claim, empty when the docs are truthful.

    Args:
        root: Repository to inspect. Parameterised so the gate itself can be tested against
            a constructed tree — a check nobody has watched fail is not a check.
    """
    violations: list[str] = _duplicate_task_ids(root)

    modules = _python_modules(root)
    if modules < IMPLEMENTATION_THRESHOLD:
        return violations

    for name in GUARDED_DOCS:
        path = root / name
        if not path.is_file():
            continue

        body = _strip_code_blocks(path.read_text(encoding="utf-8"))
        for pattern in NO_CODE_CLAIMS:
            for match in re.finditer(pattern, body, flags=re.IGNORECASE):
                line = body[: match.start()].count("\n") + 1
                violations.append(
                    f"{name}:{line}: claims {match.group(0)!r}, but src/fasterrag holds "
                    f"{modules} modules"
                )

    return violations


def main(argv: list[str] | None = None) -> int:
    """Run the check, reporting every contradiction at once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)

    violations = check()
    if not violations:
        return 0

    sys.stderr.write("the docs contradict the repository:\n")
    for violation in violations:
        sys.stderr.write(f"  {violation}\n")

    # The two violation classes have nothing to do with each other, and a gate that prints
    # the wrong remedy teaches people to ignore the right one.
    if any("docs/todo.md:" in violation for violation in violations):
        sys.stderr.write(
            "\nfor a duplicate id: give the newer entry the next free TASK- id. Parallel "
            "sessions collide here because each reads the ledger before the other appends.\n"
        )
    if any(name in violation for name in GUARDED_DOCS for violation in violations):
        sys.stderr.write(
            "\nfor a no-code claim: update the document to describe the build phase, or "
            "delete the code. CLAUDE.md is loaded as instructions on every session — a false "
            "one is followed.\n"
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
