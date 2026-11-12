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
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

__all__ = ["SKIPPED_DIRECTORIES", "expand_sources"]

SKIPPED_DIRECTORIES: frozenset[str] = frozenset(
    {".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache"}
)


def _visible(path: Path, root: Path) -> bool:
    """Return whether a path is neither hidden nor inside a skipped directory.

    Hidden files are excluded because a directory argument is a request for a corpus, and
    ``.DS_Store`` or ``.env`` are never part of one — the latter emphatically so.
    """
    return not any(
        part.startswith(".") or part in SKIPPED_DIRECTORIES for part in path.relative_to(root).parts
    )


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
