"""Expansion of directory arguments into the files a job actually ingests.

A source is taken verbatim by the pipeline: ``tasks_for`` derives a document id from each
string and reads its bytes. A directory has no bytes, so passing one produced a job of a
single unreadable document — the most obvious way to use the CLI silently indexing nothing.
Expanding here rather than in the service keeps the API from being handed a server-side
directory walk it never asked for.

Symbolic links (TASK-0504). A link to a directory is never descended — ``glob`` and
``rglob`` do not follow one — so a link back up the tree cannot loop the walk. A link that
does not lead to a readable file, a loop or a dangling target, is listed anyway rather than
dropped, so the job dead-letters it by name instead of losing it without a word. Whether a
link may point out of the corpus is not decided here: with ``ingestion.allowed_source_roots``
set, the parse worker dead-letters one that escapes as ``SOURCE_LINK_ESCAPES``.

Hidden targets (TASK-0545). The hidden-file rule is applied to a link's resolved target as
well as to the link, whether or not allowed roots are set: a visible ``notes.md`` pointing at
``../.env`` used to be read and indexed. The target is judged from the point where its path
leaves the walked root, by the same rule as the link: a hidden or skipped component past that
point excludes it, so a link into ``~/.ssh`` is skipped while a corpus that itself lives under
``~/.local`` keeps its links to visible files beside it. The link is skipped silently, as a
hidden file is. A file named explicitly on the command line is not expanded and not judged.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

__all__ = ["SKIPPED_DIRECTORIES", "expand_sources"]

SKIPPED_DIRECTORIES: frozenset[str] = frozenset(
    {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache"}
)


def _excluded(parts: Sequence[str]) -> bool:
    """Return whether any path component is hidden or names a skipped directory."""
    return any(part.startswith(".") or part in SKIPPED_DIRECTORIES for part in parts)


def _target_visible(path: Path, root: Path) -> bool:
    """Return whether a link's target passes the hidden-file rule from where it leaves ``root``.

    A target that cannot be resolved is kept, so the job dead-letters it by name (TASK-0545).
    """
    try:
        target = path.resolve()
        base = root.resolve()
    except (OSError, RuntimeError):
        return True
    try:
        parts = target.relative_to(os.path.commonpath([target, base])).parts
    except ValueError:
        parts = target.parts[1:]
    return not _excluded(parts)


def _visible(path: Path, root: Path) -> bool:
    """Return whether a path, and a link's target, are neither hidden nor in a skipped directory.

    A directory argument asks for a corpus, and ``.DS_Store`` or ``.env`` are never part of one.
    """
    if _excluded(path.relative_to(root).parts):
        return False
    return not path.is_symlink() or _target_visible(path, root)


def _listed(path: Path) -> bool:
    """Return whether an entry is a file, or a link that is not a directory and so is reported."""
    return path.is_file() or (path.is_symlink() and not path.is_dir())


def expand_sources(sources: Sequence[str], *, recursive: bool = False) -> list[str]:
    """Return the sources with any directory replaced by the files inside it."""
    expanded: list[str] = []

    for source in sources:
        try:
            path = Path(source)
            is_directory = path.is_dir()
        except OSError:
            expanded.append(source)
            continue

        if not is_directory:
            expanded.append(source)
            continue

        found = path.rglob("*") if recursive else path.glob("*")
        expanded.extend(
            sorted(str(child) for child in found if _listed(child) and _visible(child, path))
        )

    return expanded
