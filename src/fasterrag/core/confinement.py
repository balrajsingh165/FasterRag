"""Whether a caller-named path stays inside the directories a deployment allows.

Two layers ask this — ``services/sources.py`` before a job starts, and the parse worker
before it reads a byte — and the layering rule lets neither import the other's layer the
right way round, so the predicates live here, below both, written once.

A path can escape two ways. Written outside a root (``/etc/passwd``, ``root/../x``) it is the
caller's mistake, and the whole request is refused. Written inside a root but resolving
outside it, it is a symbolic link in the corpus pointing away: one bad document among good
ones, which the worker dead-letters as ``SOURCE_LINK_ESCAPES`` while the job goes on
(TASK-0504). Resolution follows every link and collapses ``..``, so neither can step outside
a root. A link loop cannot be resolved at all; Python 3.12 says so with a bare
``RuntimeError``, which is typed here.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

from fasterrag.errors import ParseError

__all__ = ["resolved_within", "written_within"]


def _resolve(value: str | Path) -> Path:
    """Resolve every link in a path, typing the error a link loop raises."""
    try:
        return Path(value).resolve()
    except (OSError, RuntimeError) as exc:
        raise ParseError(
            f"{value} could not be resolved; a symbolic link loop does this: {exc}"
        ) from exc


def resolved_within(value: str, roots: Sequence[str]) -> Path | None:
    """Return the fully resolved path when it sits inside a root, else ``None``."""
    candidate = _resolve(value)
    if any(candidate.is_relative_to(_resolve(root)) for root in roots):
        return candidate
    return None


def written_within(value: str, roots: Sequence[str]) -> bool:
    """Return whether the path as written — ``..`` collapsed, no link followed — is in a root."""
    candidate = Path(os.path.abspath(value))
    return any(
        candidate.is_relative_to(Path(os.path.abspath(root)))
        or candidate.is_relative_to(_resolve(root))
        for root in roots
    )
