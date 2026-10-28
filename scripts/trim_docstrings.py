"""Cut over-long function, method and class docstrings down, moving the prose into docs/.

See docs/CONTRIBUTING.md for the rule this enforces and where the moved text lands.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

REPOSITORY_ROOT: Final = Path(__file__).resolve().parents[1]
NOTES: Final = REPOSITORY_ROOT / "docs" / "code-notes.md"
MAXIMUM_LINES: Final = 4

# CRITICAL: module docstrings are deliberately exempt. They are the file's documentation
# rather than a function's, they are what `internals.md` points readers at, and gutting them
# would move the "why" nowhere useful. Only def/class docstrings are trimmed.
TRIMMABLE: Final = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


@dataclass
class Note:
    """One docstring's overflow, kept so it can be written to docs/."""

    module: str
    qualname: str
    summary: str
    body: str


@dataclass
class Result:
    """What a run changed."""

    trimmed: int = 0
    files: set[str] = field(default_factory=set)
    notes: list[Note] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _structure(source: str) -> str:
    """Return the AST with every docstring blanked, for before/after comparison."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, *TRIMMABLE)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            first.value.value = ""
    return ast.dump(tree)


def _qualnames(tree: ast.Module) -> dict[int, str]:
    """Map each node id to its dotted qualified name."""
    names: dict[int, str] = {}

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, TRIMMABLE):
                qualname = f"{prefix}.{child.name}" if prefix else child.name
                names[id(child)] = qualname
                walk(child, qualname)
            else:
                walk(child, prefix)

    walk(tree, "")
    return names


def _summary(doc: str) -> tuple[str, str]:
    """Split a docstring into the part that stays and the part that moves."""
    stripped = doc.strip("\n")
    lines = stripped.splitlines()
    kept: list[str] = []
    for line in lines:
        if not line.strip() and kept:
            break
        kept.append(line)
        if len(kept) == MAXIMUM_LINES:
            break
    overflow = lines[len(kept) :]
    return "\n".join(kept).strip(), "\n".join(overflow).strip()


def _rewrite(path: Path, result: Result, *, apply: bool) -> None:
    """Trim one file's over-long def/class docstrings."""
    original = path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(original)
    except SyntaxError as error:
        result.problems.append(f"{path}: unparseable ({error})")
        return

    names = _qualnames(tree)
    module = path.relative_to(REPOSITORY_ROOT).as_posix()
    edits: list[tuple[int, int, str]] = []

    for node in ast.walk(tree):
        if not isinstance(node, TRIMMABLE) or not node.body:
            continue
        first = node.body[0]
        if not (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)):
            continue
        doc = first.value.value
        if not isinstance(doc, str) or len(doc.strip().splitlines()) <= MAXIMUM_LINES:
            continue
        if first.end_lineno is None:
            continue

        summary, overflow = _summary(doc)
        if not summary:
            continue

        indent = " " * first.col_offset
        if len(summary.splitlines()) == 1:
            replacement = f'{indent}"""{summary}"""\n'
        else:
            inner = "\n".join(f"{indent}{line}" if line else "" for line in summary.splitlines())
            replacement = f'{indent}"""{inner.lstrip()}\n{indent}"""\n'

        edits.append((first.lineno, first.end_lineno, replacement))
        if overflow:
            result.notes.append(Note(module, names.get(id(node), "?"), summary, overflow))

    if not edits:
        return

    lines = original.splitlines(keepends=True)
    for start, end, replacement in sorted(edits, reverse=True):
        lines[start - 1 : end] = [replacement]
    updated = "".join(lines)

    # CRITICAL: the whole safety argument for this tool. A previous scripted docstring pass
    # used a regex, split sentences mid-line, and had to be reverted. Comparing the AST with
    # every docstring blanked proves the rewrite changed documentation and nothing else — if
    # a single statement moved, this refuses the file rather than writing it.
    try:
        if _structure(original) != _structure(updated):
            result.problems.append(f"{path}: rewrite altered code structure; refused")
            return
    except SyntaxError as error:
        result.problems.append(f"{path}: rewrite produced invalid python ({error}); refused")
        return

    result.trimmed += len(edits)
    result.files.add(module)
    if apply:
        newline = "\r\n" if b"\r\n" in path.read_bytes() else "\n"
        with path.open("w", encoding="utf-8", newline=newline) as handle:
            handle.write(updated)


def _write_notes(notes: list[Note]) -> None:
    """Append the moved prose to docs/code-notes.md, grouped by module."""
    grouped: dict[str, list[Note]] = {}
    for note in notes:
        grouped.setdefault(note.module, []).append(note)

    lines = [
        "# code-notes.md — Reasoning Moved Out Of Docstrings",
        "",
        "Docstrings state what a thing does in at most four lines "
        "([CONTRIBUTING.md](CONTRIBUTING.md)). The longer reasoning that used to sit in them "
        "lives here, addressed by `module::qualname`, so it is still searchable and still "
        "reviewable without making every function header a page long.",
        "",
        "This file is generated by `scripts/trim_docstrings.py` and appended to, never "
        "rewritten by hand in bulk: an entry is the record of why a piece of code is the way "
        "it is, and losing it costs more than the lines it saves.",
        "",
    ]
    for module in sorted(grouped):
        lines.append(f"## `{module}`")
        lines.append("")
        for note in grouped[module]:
            lines.append(f"### `{note.qualname}`")
            lines.append("")
            lines.append(f"> {note.summary}")
            lines.append("")
            lines.append(note.body)
            lines.append("")

    with NOTES.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def check(paths: list[Path], *, apply: bool = False) -> Result:
    """Trim (or report) over-long docstrings across the given roots."""
    result = Result()
    for root in paths:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            _rewrite(path, result, apply=apply)
    if apply and result.notes:
        _write_notes(result.notes)
    return result


def main(argv: list[str] | None = None) -> int:
    """Report or apply the trim; exit non-zero on a refused file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="*", default=["src", "tests", "scripts"])
    parser.add_argument("--apply", action="store_true", help="write the changes")
    arguments = parser.parse_args(argv)

    roots = [REPOSITORY_ROOT / root for root in arguments.roots]
    result = check(roots, apply=arguments.apply)

    verb = "trimmed" if arguments.apply else "would trim"
    sys.stdout.write(
        f"{verb} {result.trimmed} docstrings across {len(result.files)} files; "
        f"{len(result.notes)} notes moved to docs/code-notes.md\n"
    )
    for problem in result.problems:
        sys.stderr.write(f"trim_docstrings: {problem}\n")
    return 1 if result.problems else 0


if __name__ == "__main__":
    sys.exit(main())
