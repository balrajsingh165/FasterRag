"""Fail when source code carries an explanatory comment CLAUDE.md does not permit.

``CLAUDE.md`` states the rule as an absolute: *"Docstrings only. NO inline comments and NO
explanatory comments in code."* Comments are allowed in exactly two shapes — a
``# CRITICAL:`` flag saying why something must not change, and a ``# TODO:`` / ``# BLOCKED:``
marker saying what remains — plus the continuation ``#`` lines a marker may open. Everything
else belongs in a docstring, or, when it needs more than four lines, in
``docs/code-notes.md``.

Nothing enforced it. ``ruff`` has no rule for "this comment is prose", ``mypy`` never reads
a comment, and the two sibling gates look at other things entirely: ``check_task_ids.py``
resolves the ``TASK-`` ids *inside* comments without caring whether the comment should exist,
and ``check_error_taxonomy.py`` reads comments only to find a ``# type: ignore`` without a
justification beside it. So the rule held exactly as far as whoever was typing remembered it,
and a 2026-11-08 scan of ``src/`` found 130 prose comment lines in 30 of its 151 modules —
roughly one file in five — none of which any gate had ever objected to.

The failure that matters is not tidiness. Prose in a comment is prose nobody greps, nobody
reviews against the docs, and nothing keeps true: the reasoning drifts from the code it sits
beside and there is no second copy to contradict it. The same words in ``docs/code-notes.md``
are addressed by ``module::qualname``, show up in a documentation diff, and survive the
function being moved.

The rules:

* ``prose-comment`` (error) — a comment that is not a marker, not a continuation of one, and
  not a tool directive. Reported once per contiguous block, since a four-line explanation is
  one defect rather than four.
* ``vacuous-scan`` (error) — the scan matched no files, tokenized no comments at all, or
  recognised no markers anywhere. Any of the three means the gate proved nothing, and a gate
  that cannot tell "clean" from "broken" is worse than no gate (rebuild-playbook trap T12).

Permitted without a marker: ``# noqa``, ``# type: ignore``, ``# pragma:``, ``# fmt:``,
``# ruff:``, ``# mypy:``, ``# isort:``. These address a tool rather than a reader, there is
nowhere else to put them, and ``check_error_taxonomy.py`` already governs whether a
``# type: ignore`` carries its justification. A shebang and a PEP 263 encoding declaration
are not comments for this purpose either: the interpreter reads both.

What this gate CANNOT do, said plainly:

* it cannot tell a good ``# CRITICAL:`` from a bad one. Writing ``# CRITICAL:`` in front of
  a paragraph of ordinary prose defeats it completely, and only review catches that;
* it reads ``src/`` and ``scripts/``. ``tests/`` is deliberately out of scope — the comments
  there label fixtures and scenarios for a reader rather than explaining production
  behaviour, and sweeping them in would make the gate's first act a 53-site cleanup that
  improves nothing;
* it works on tokens, so a ``#`` inside a string literal is invisible to it — correct, but
  it also means a comment smuggled into an ``exec`` string is invisible.

Exit codes: ``0`` when no finding exceeds the ratchet, ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import io
import re
import sys
import tokenize
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
SCANNED_TREES: Final[tuple[str, ...]] = ("src", "scripts")

PROSE_COMMENT: Final = "prose-comment"
VACUOUS: Final = "vacuous-scan"

# CRITICAL: these three keywords are the entire permission CLAUDE.md grants, and a marker may
# open a block of plain `#` continuation lines. Widening this set widens the rule itself, so
# it changes only when CLAUDE.md does.
_MARKER: Final = re.compile(r"^#\s*(?:CRITICAL|TODO|BLOCKED):")

_DIRECTIVE: Final = re.compile(
    r"^#\s*(?:noqa\b|type:\s*ignore|pragma:|fmt:\s*(?:on|off|skip)\b|ruff:|mypy:|isort:)"
)
_ENCODING: Final = re.compile(r"^#.*coding[:=]\s*[-\w.]+")

# CRITICAL: keyed by path and never by line number. Nine agents edit this tree in parallel and
# every line number in it moves between a scan and a commit; a line-keyed waiver would go
# stale within the hour and start excusing a different comment than the one it was written
# for. A count is a backlog, not a permission: raising one to make a new violation green is
# the one thing this file must never be used for.
#
# One entry, added 2026-11-08 with the gate: two blocks in `core/chunking/tokenizer.py` — the
# local-provider table and the broad catch around the tokenizer load — were being edited by a
# concurrent slice while this gate was written, so moving them would have collided with that
# work. TASK-0336 covers them; clear this entry when it does, and do not add a second.
# `core/chunking/semantic.py` is named by the same task and needs no entry: it was measured
# the same day and holds no disallowed comment at all.
RATCHET: Final[dict[str, int]] = {
    f"{PROSE_COMMENT}:src/fasterrag/core/chunking/tokenizer.py": 2,
}


@dataclass(frozen=True, slots=True)
class Finding:
    """One disallowed comment block, located precisely enough for a human to judge it."""

    rule: str
    path: str
    line: int
    detail: str

    @property
    def key(self) -> str:
        """Return the ratchet key this finding counts against."""
        return f"{self.rule}:{self.path}"

    def __str__(self) -> str:
        """Return the one-line report form."""
        return f"{self.path}:{self.line}: [{self.rule}] {self.detail}"


@dataclass(frozen=True, slots=True)
class Comment:
    """One comment token, with the two facts the block rules need."""

    line: int
    text: str
    standalone: bool


@dataclass
class Tally:
    """What one file contributed, so the vacuity guard can judge the whole scan."""

    comments: int = 0
    markers: int = 0


def _sources(root: Path, *parts: str) -> list[Path]:
    """Return the Python modules under ``root/parts``, excluding bytecode caches."""
    base = root.joinpath(*parts)
    if not base.is_dir():
        return []
    return sorted(path for path in base.rglob("*.py") if "__pycache__" not in path.parts)


def _comments(source: str) -> list[Comment]:
    """Return the real comment tokens, so a ``#`` inside a string literal never matches."""
    found: list[Comment] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                found.append(
                    Comment(
                        line=token.start[0],
                        text=token.string.strip(),
                        standalone=token.line.strip().startswith("#"),
                    )
                )
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return found
    return found


def _interpreter_line(comment: Comment) -> bool:
    """Return whether this is a shebang or a PEP 263 encoding declaration, not a comment."""
    if comment.line > 2:
        return False
    if comment.line == 1 and comment.text.startswith("#!"):
        return True
    return bool(_ENCODING.match(comment.text))


def _blocks(comments: list[Comment]) -> list[list[Comment]]:
    """Return the contiguous runs of disallowed comments, each reported as one finding."""
    disallowed: list[Comment] = []
    open_block = False
    previous = -2
    for comment in comments:
        continues = open_block and comment.standalone and comment.line == previous + 1
        previous = comment.line
        if _MARKER.match(comment.text):
            open_block = comment.standalone
            continue
        if continues:
            continue
        open_block = False
        if _DIRECTIVE.match(comment.text) or _interpreter_line(comment):
            continue
        disallowed.append(comment)

    blocks: list[list[Comment]] = []
    for comment in disallowed:
        last = blocks[-1][-1] if blocks else None
        if (
            last is not None
            and last.standalone
            and comment.standalone
            and comment.line == last.line + 1
        ):
            blocks[-1].append(comment)
        else:
            blocks.append([comment])
    return blocks


def _scan_file(relative: str, source: str, tally: Tally) -> list[Finding]:
    """Return one finding per disallowed block in a file, updating the scan's tally."""
    comments = _comments(source)
    tally.comments += len(comments)
    tally.markers += sum(1 for comment in comments if _MARKER.match(comment.text))

    findings: list[Finding] = []
    for block in _blocks(comments):
        span = "1 line" if len(block) == 1 else f"{len(block)} lines"
        findings.append(
            Finding(
                PROSE_COMMENT,
                relative,
                block[0].line,
                f"{span} of explanatory comment: {block[0].text[:72]!r}",
            )
        )
    return findings


def check(root: Path = REPOSITORY_ROOT, trees: tuple[str, ...] = SCANNED_TREES) -> list[Finding]:
    """Return every finding in ``root``, including the vacuity guard's own."""
    modules = [path for tree in trees for path in _sources(root, tree)]
    where = "/".join(trees)
    if not modules:
        return [Finding(VACUOUS, where, 0, "the scan matched no Python module; it proved nothing")]

    findings: list[Finding] = []
    tally = Tally()
    for path in modules:
        relative = path.relative_to(root).as_posix()
        findings.extend(_scan_file(relative, path.read_text(encoding="utf-8"), tally))

    if tally.comments == 0:
        return [
            Finding(
                VACUOUS,
                where,
                0,
                f"tokenized {len(modules)} modules and found no comment at all; either every "
                "comment has left this tree, or this scan no longer reads them",
            )
        ]
    if tally.markers == 0:
        return [
            Finding(
                VACUOUS,
                where,
                0,
                f"read {tally.comments} comments across {len(modules)} modules and recognised "
                "no '# CRITICAL:', '# TODO:' or '# BLOCKED:' marker among them; the marker "
                "pattern is the whole rule, so a scan that matches none of them is broken",
            )
        ]
    return findings


def over_ratchet(findings: list[Finding]) -> list[Finding]:
    """Return the findings beyond what the ratchet already records."""
    counted: dict[str, int] = {}
    excess: list[Finding] = []
    for finding in findings:
        counted[finding.key] = counted.get(finding.key, 0) + 1
        if counted[finding.key] > RATCHET.get(finding.key, 0):
            excess.append(finding)
    return excess


def _report(write: Callable[[str], int], heading: str, findings: list[Finding]) -> None:
    """Write one headed block of findings, or nothing when the block is empty."""
    if not findings:
        return
    write(f"{heading}\n")
    for finding in findings:
        write(f"  {finding}\n")
    write("\n")


def main(argv: list[str] | None = None) -> int:
    """Run the gate, separating new violations from the ratcheted backlog."""
    parser = argparse.ArgumentParser(description="Check that only permitted comments exist.")
    parser.add_argument(
        "--root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="repository root to scan (default: the checkout this script lives in)",
    )
    arguments = parser.parse_args(argv)

    findings = check(arguments.root)
    excess = over_ratchet(findings)
    known = [finding for finding in findings if finding not in excess]

    _report(sys.stderr.write, "comments CLAUDE.md does not permit:", excess)
    _report(sys.stdout.write, "known and ratcheted (a backlog, not a permission):", known)

    if excess:
        sys.stderr.write(
            "CLAUDE.md allows a comment only as '# CRITICAL:', '# TODO:' or '# BLOCKED:', or "
            "as a '#' line continuing one. Fold the reasoning into the docstring if it fits "
            "in four lines, otherwise move it to docs/code-notes.md under the enclosing "
            "'module::qualname' and delete the comment. Do not widen the ratchet in "
            "scripts/check_comments.py.\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
