"""The shape of a query trace as the trace exporters read it.

The trace model (``core/tracing.py``) belongs to the pipeline: it records retrieval candidates
and the prompt templates a query ran under, so it imports ``core``. The exporters that ship a
trace to Langfuse or an OTLP endpoint sit below ``core`` (``docs/structure.md``, the import
rule), so they cannot name that class. They read a trace through these protocols instead, which
:class:`~fasterrag.core.tracing.Trace` and :class:`~fasterrag.core.tracing.Span` satisfy
structurally — nothing registers them, and mypy checks the fit wherever a trace is exported.

Added on 2026-11-15 (TASK-0529). Every member is read-only: an exporter never changes a trace.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

__all__ = ["SpanView", "TraceView"]


class SpanView(Protocol):
    """One stage of a query, as an exporter reads it."""

    @property
    def name(self) -> str:
        """Return the stage name, one of ``core.tracing.SPAN_NAMES``."""

    @property
    def start_ms(self) -> float:
        """Return the offset the stage began at, in milliseconds from the trace's start."""

    @property
    def end_ms(self) -> float:
        """Return the offset the stage ended at, in milliseconds from the trace's start."""

    @property
    def attributes(self) -> Mapping[str, Any]:
        """Return what the stage observed."""


class TraceView(Protocol):
    """Everything one query did, as an exporter reads it."""

    @property
    def trace_id(self) -> str:
        """Return the 32-hex trace id."""

    @property
    def query(self) -> str:
        """Return the query text."""

    @property
    def collection(self) -> str | None:
        """Return the collection queried, if one was named."""

    @property
    def tenant(self) -> str | None:
        """Return the tenant the query ran for, if any."""

    @property
    def filters(self) -> Mapping[str, Any] | None:
        """Return the metadata filters the query applied, if any."""

    @property
    def config_snapshot(self) -> Mapping[str, Any]:
        """Return the retrieval-affecting configuration the query ran under."""

    @property
    def retrieved(self) -> Sequence[Mapping[str, Any]]:
        """Return the candidate records retrieval produced."""

    @property
    def prompt(self) -> str:
        """Return the prompt sent to the model."""

    @property
    def response(self) -> str:
        """Return the model's response."""

    @property
    def result(self) -> Mapping[str, Any]:
        """Return the answer body the query returned."""

    @property
    def spans(self) -> Sequence[SpanView]:
        """Return the stages, in the order they closed."""

    @property
    def created_at(self) -> str:
        """Return when the query ran, in ISO 8601."""
