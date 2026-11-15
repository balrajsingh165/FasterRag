"""Names TASK-0529 moved down a layer still import from where 0.1.0b1 and 0.1.0b2 put them.

Each old module re-exports the moved names for one release. The re-export must be the same
object, not a copy: a second ``CostGovernor`` class or a second price table would type-check
and quietly split the shared state the old and new import paths are meant to share.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pytest

from fasterrag.core.journaling import DocumentJournal
from fasterrag.core.tracing import Span, Trace
from fasterrag.observability.trace_view import TraceView
from fasterrag.services.journal import Journal

MOVED: dict[tuple[str, str], tuple[str, ...]] = {
    ("fasterrag.services.budget", "fasterrag.core.budget"): (
        "BUCKETS_PER_WINDOW",
        "ROLLING_WINDOW_SECONDS",
        "CostGovernor",
        "create_governor",
        "reset_shared_governors",
        "shared_governor",
    ),
    ("fasterrag.services.estimation", "fasterrag.core.pricing"): (
        "EMBEDDING_PRICES",
        "GENERATION_PRICES",
        "LOCAL_PROVIDERS",
        "PRICES_DATED",
        "TokenPrice",
        "price_for",
        "price_generation",
    ),
    ("fasterrag.services.journal", "fasterrag.core.journaling"): (
        "DocumentRecord",
        "DocumentStatus",
    ),
    ("fasterrag.cli.sources", "fasterrag.services.source_expansion"): (
        "SKIPPED_DIRECTORIES",
        "expand_sources",
    ),
}


@pytest.mark.parametrize(
    ("old", "new", "name"),
    [(old, new, name) for (old, new), names in MOVED.items() for name in names],
)
def test_a_moved_name_is_the_same_object_at_its_old_path(old: str, new: str, name: str) -> None:
    """Code written against the published betas keeps importing from the old module."""
    previous, current = import_module(old), import_module(new)

    assert getattr(previous, name) is getattr(current, name)
    assert name in previous.__all__
    assert name in current.__all__


def test_the_journal_is_what_a_parse_worker_is_typed_against(tmp_path: Path) -> None:
    """The pool takes a ``DocumentJournal``; mypy checks this assignment, pytest that it runs."""
    journal: DocumentJournal = Journal(tmp_path / "journal")

    assert callable(journal.record_indexed)


def test_a_trace_is_what_an_exporter_reads() -> None:
    """The exporters take a ``TraceView``; mypy checks that ``Trace`` still fits it."""
    trace: TraceView = Trace(trace_id="0" * 32, query="q", spans=[Span("retrieval", 0.0, 1.0)])

    assert [span.name for span in trace.spans] == ["retrieval"]
