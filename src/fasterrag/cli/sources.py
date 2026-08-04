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

# Walking these wastes a full parse pass on files no corpus contains. Ingesting a project
# folder should not mean parsing its virtualenv.
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
    """Return the sources with any directory replaced by the files inside it.

    Anything that is not an existing directory passes through untouched, so URLs, inline
    payloads, and plain file paths are unaffected.

    Args:
        sources: The sources as given on the command line.
        recursive: Whether to descend into subdirectories. Without it only a directory's
            immediate files are taken, which is what makes ``--recursive`` mean something.

    Returns:
        The expanded sources. Files from one directory are sorted, because document ids
        and job order derive from this list and an arbitrary filesystem order would make
        the same command produce a different job each run.
    """
    expanded: list[str] = []

    for source in sources:
        try:
            path = Path(source)
            is_directory = path.is_dir()
        except OSError:
            # A source the filesystem cannot even stat is not a directory; leave it for
            # the pipeline to report as unreadable with its own reason code.
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
