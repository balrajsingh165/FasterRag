"""Deprecated location of directory expansion, which now lives in ``services/source_expansion.py``.

Moved on 2026-11-15 (TASK-0529): the facade expands directories as the CLI does, and the
facade may not import the CLI. ``expand_sources`` and ``SKIPPED_DIRECTORIES`` are re-exported
here, as the same objects, for one release so code written against ``0.1.0b1`` or ``0.1.0b2``
keeps working; import from ``fasterrag.services.source_expansion`` instead. This module will be
removed in a later release.
"""

from __future__ import annotations

from fasterrag.services.source_expansion import SKIPPED_DIRECTORIES, expand_sources

__all__ = ["SKIPPED_DIRECTORIES", "expand_sources"]
