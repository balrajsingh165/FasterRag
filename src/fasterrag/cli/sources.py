"""Expansion of directory arguments into the files a job actually ingests.

A source is taken verbatim by the pipeline: ``tasks_for`` derives a document id from each
string and reads its bytes. A directory has no bytes, so passing one produced a job of a
single unreadable document — the most obvious way to use the CLI silently indexing nothing.
Expanding here rather than in the service keeps the API from being handed a server-side
directory walk it never asked for.
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
            sorted(str(child) for child in found if child.is_file() and _visible(child, path))
        )

    return expanded
