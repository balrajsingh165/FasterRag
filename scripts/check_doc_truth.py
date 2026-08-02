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


def check(root: Path = REPOSITORY_ROOT) -> list[str]:
    """Return one message per contradicted claim, empty when the docs are truthful.

    Args:
        root: Repository to inspect. Parameterised so the gate itself can be tested against
            a constructed tree — a check nobody has watched fail is not a check.
    """
    modules = _python_modules(root)
    if modules < IMPLEMENTATION_THRESHOLD:
        return []

    violations: list[str] = []
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

    sys.stderr.write("the always-loaded docs contradict the repository:\n")
    for violation in violations:
        sys.stderr.write(f"  {violation}\n")
    sys.stderr.write(
        "\nupdate the document to describe the build phase, or delete the code. "
        "CLAUDE.md is loaded as instructions on every session — a false one is followed.\n"
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
