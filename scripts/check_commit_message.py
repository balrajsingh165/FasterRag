"""Enforce the fasterRag commit-message rule.

Commit messages must be a single line with no trailers of any kind and no AI
attribution (``docs/CONTRIBUTING.md`` §1, CI gate 8 in ``docs/testing-strategy.md`` §2).
Runs in two modes: against a commit-message file (the ``commit-msg`` pre-commit hook)
or against a revision range (CI), so the same rule is enforced locally and remotely.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

FORBIDDEN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"co-authored-by", re.IGNORECASE), "trailer 'Co-Authored-By' is not allowed"),
    (re.compile(r"signed-off-by", re.IGNORECASE), "trailer 'Signed-off-by' is not allowed"),
    (re.compile(r"generated with", re.IGNORECASE), "AI attribution is not allowed"),
    (re.compile(r"claude code", re.IGNORECASE), "AI attribution is not allowed"),
    (re.compile(r"\U0001f916"), "AI attribution is not allowed"),
)


def significant_lines(message: str) -> list[str]:
    """Return the message lines that git would keep, dropping comments and blanks."""
    return [
        line for line in message.splitlines() if line.strip() and not line.lstrip().startswith("#")
    ]


def check_message(message: str) -> list[str]:
    """Return every rule violation in ``message``; an empty list means the message passes."""
    violations: list[str] = []
    lines = significant_lines(message)

    if not lines:
        violations.append("commit message is empty")
    elif len(lines) > 1:
        violations.append(
            f"commit message must be a single line (found {len(lines)} lines); "
            "no multi-line bodies and no trailers"
        )

    for pattern, reason in FORBIDDEN_PATTERNS:
        if pattern.search(message):
            violations.append(reason)

    return violations


def messages_in_range(rev_range: str) -> list[str]:
    """Return every commit message in ``rev_range``, newest first."""
    result = subprocess.run(
        ["git", "log", "--format=%B%x00", rev_range],
        capture_output=True,
        text=True,
        check=True,
    )
    return [chunk for chunk in result.stdout.split("\0") if chunk.strip()]


def report(violations: list[str], subject: str) -> None:
    """Write violations for one commit message to stderr."""
    sys.stderr.write(f"commit message rejected: {subject}\n")
    for violation in violations:
        sys.stderr.write(f"  - {violation}\n")


def main(argv: list[str] | None = None) -> int:
    """Validate a commit-message file or a revision range; return a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="commit message file (commit-msg hook mode)")
    parser.add_argument("--range", dest="rev_range", help="revision range to check (CI mode)")
    args = parser.parse_args(argv)

    if args.rev_range:
        messages = messages_in_range(args.rev_range)
    elif args.path:
        messages = [Path(args.path).read_text(encoding="utf-8")]
    else:
        parser.error("provide a commit message file or --range")

    failed = False
    for message in messages:
        violations = check_message(message)
        if violations:
            failed = True
            lines = significant_lines(message)
            report(violations, lines[0] if lines else "<empty>")

    if failed:
        sys.stderr.write(
            "\nfasterRag requires single-line commit messages with no trailers "
            "(docs/CONTRIBUTING.md §1).\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
