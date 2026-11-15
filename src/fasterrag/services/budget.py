"""Deprecated location of the D9 cost governor, which now lives in :mod:`fasterrag.core.budget`.

Moved on 2026-11-15 (TASK-0529): the embedding workers charge the governor, and a worker may
not import a service. Every public name is re-exported here, as the same object, for one
release so code written against ``0.1.0b1`` or ``0.1.0b2`` keeps working; import from
``fasterrag.core.budget`` instead. This module will be removed in a later release.
"""

from __future__ import annotations

from fasterrag.core.budget import (
    BUCKETS_PER_WINDOW,
    ROLLING_WINDOW_SECONDS,
    CostGovernor,
    create_governor,
    reset_shared_governors,
    shared_governor,
)

__all__ = [
    "BUCKETS_PER_WINDOW",
    "ROLLING_WINDOW_SECONDS",
    "CostGovernor",
    "create_governor",
    "reset_shared_governors",
    "shared_governor",
]
