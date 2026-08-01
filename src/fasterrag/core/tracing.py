"""Query traces and the four RAG spans (D8).

The ``Trace`` entity of ``docs/data-model.md``: everything a past query did, kept so that
"why did this answer change last week?" is answerable rather than a matter of recollection.

Four span types, per ``docs/observability.md`` §3 — ``retrieval``, ``reranker``,
``context-assembly``, ``generation``. A skipped stage produces no span rather than a
zero-duration one, so the absence of a reranker span means reranking did not run, which is
information rather than noise.

A trace records the **retrieval-affecting configuration** it executed under, not the whole
file. That subset is what replay varies, and recording ``app.port`` alongside it would make
every unrelated edit look like a reason an answer changed.

Nothing here holds a secret: the snapshot carries provider and model names, never keys, in
line with invariant 6 of the data model.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.models import ScoredChunk

__all__ = [
    "SPAN_NAMES",
    "Span",
    "SpanRecorder",
    "Trace",
    "config_snapshot",
]

SPAN_NAMES: tuple[str, ...] = ("retrieval", "reranker", "context-assembly", "generation")


def config_snapshot(settings: Settings) -> dict[str, Any]:
    """Return the retrieval-affecting configuration a trace executed under.

    The same subset the index lockfile anchors on, plus the generation settings that change
    an answer without changing what was retrieved. Replay compares these two snapshots to
    explain a difference, so anything omitted here is a difference replay cannot account for.
    """
    return {
        "chunking": settings.chunking.model_dump(mode="json"),
        "embeddings": {
            "provider": settings.embeddings.provider,
            "model": settings.embeddings.model,
            "dimensions": settings.embeddings.dimensions,
        },
        "retrieval": settings.retrieval.model_dump(mode="json"),
        "generation": settings.generation.model_dump(mode="json"),
        "llm": {
            "provider": settings.llm.provider,
            "model": settings.llm.model,
            "temperature": settings.llm.temperature,
            "max_tokens": settings.llm.max_tokens,
        },
    }


@dataclass(frozen=True, slots=True)
class Span:
    """One stage of a query, and how long it took."""

    name: str
    start_ms: float
    end_ms: float
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Return the span's wall-clock duration."""
        return self.end_ms - self.start_ms

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized form."""
        return {
            "name": self.name,
            "start_ms": round(self.start_ms, 3),
            "end_ms": round(self.end_ms, 3),
            "duration_ms": round(self.duration_ms, 3),
            "attributes": self.attributes,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Span:
        """Rebuild a span from its serialized form."""
        return cls(
            name=str(payload["name"]),
            start_ms=float(payload["start_ms"]),
            end_ms=float(payload["end_ms"]),
            attributes=dict(payload.get("attributes") or {}),
        )


class SpanRecorder:
    """Collects spans for one query, timed against a single origin.

    All spans share one clock origin so their offsets are directly comparable — timing each
    against its own start would produce four durations that cannot be laid on one timeline,
    which is the entire point of nesting them under a root.
    """

    def __init__(self) -> None:
        """Start a recorder at the current instant."""
        self._origin = time.perf_counter()
        self._spans: list[Span] = []

    @property
    def elapsed_ms(self) -> float:
        """Return milliseconds since the recorder started."""
        return (time.perf_counter() - self._origin) * 1000

    def record(self, name: str, start_ms: float, **attributes: Any) -> Span:
        """Close a span that began at ``start_ms``, with the attributes it observed."""
        span = Span(name=name, start_ms=start_ms, end_ms=self.elapsed_ms, attributes=attributes)
        self._spans.append(span)
        return span

    @property
    def spans(self) -> list[Span]:
        """Return the spans recorded so far, in the order they closed."""
        return list(self._spans)


@dataclass(frozen=True, slots=True)
class Trace:
    """Everything one query did, persisted for replay and inspection."""

    trace_id: str
    query: str
    collection: str | None = None
    filters: dict[str, Any] | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    prompt: str = ""
    response: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)
    created_at: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized form, which is also the ``GET /v1/traces/{id}`` body."""
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "collection": self.collection,
            "filters": self.filters,
            "config_snapshot": self.config_snapshot,
            "retrieved": self.retrieved,
            "prompt": self.prompt,
            "response": self.response,
            "result": self.result,
            "spans": [span.as_dict() for span in self.spans],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Trace:
        """Rebuild a trace from its persisted form."""
        return cls(
            trace_id=str(payload["trace_id"]),
            query=str(payload.get("query", "")),
            collection=payload.get("collection"),
            filters=payload.get("filters"),
            config_snapshot=dict(payload.get("config_snapshot") or {}),
            retrieved=list(payload.get("retrieved") or []),
            prompt=str(payload.get("prompt", "")),
            response=str(payload.get("response", "")),
            result=dict(payload.get("result") or {}),
            spans=[Span.from_dict(span) for span in payload.get("spans") or []],
            created_at=str(payload.get("created_at", "")),
        )


def record_chunk(chunk: ScoredChunk) -> dict[str, Any]:
    """Return the full candidate record a trace keeps for one chunk.

    Every leg's rank and score is kept, not just the fused number: replay's whole job is
    explaining why an ordering changed, and a collapsed score cannot distinguish "the dense
    leg moved it" from "the reranker did".
    """
    return {
        "chunk_id": chunk.chunk_id,
        "text": chunk.text,
        "source": chunk.source,
        "document_id": chunk.document_id,
        "dense_rank": chunk.dense_rank,
        "dense_score": chunk.dense_score,
        "bm25_rank": chunk.bm25_rank,
        "bm25_score": chunk.bm25_score,
        "rrf_score": chunk.rrf_score,
        "rerank_score": chunk.rerank_score,
        "final_rank": chunk.final_rank,
    }
