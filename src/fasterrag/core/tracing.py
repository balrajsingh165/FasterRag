"""Query traces and the four RAG spans (D8).

The ``Trace`` entity of ``docs/data-model.md``: everything a past query did, kept so that
"why did this answer change last week?" is answerable rather than a matter of recollection.

Four span types, per ``docs/observability.md`` §3 — ``retrieval``, ``reranker``,
``context-assembly``, ``generation``. A skipped stage produces no span rather than a
zero-duration one, so the absence of a reranker span means reranking did not run, which is
information rather than noise. Since TASK-0487 retrieval is also broken into its legs
(``query-embedding``, ``dense-search``, ``keyword-search``, ``fusion``, ``reranker``), and an
ingest records a trace of its own (``kind="ingest"``) whose pipelined stages — ``parse``,
``chunk``, ``embed``, ``upsert`` — are each one span covering the window that stage was
active, carrying its summed busy time, because those stages overlap by design.

A trace records the **retrieval-affecting configuration** it executed under, not the whole
file. That subset is what replay varies, and recording ``app.port`` alongside it would make
every unrelated edit look like a reason an answer changed. The P1 and P3 template versions
are part of that subset (``prompts.*``): a prompt change alters answers without touching
retrieval, so a replay that could not see it would blame the configuration for it. A trace
stored before they were recorded replays with both keys reported as changed from ``None``.

Nothing here holds a secret: the snapshot carries provider and model names, never keys, in
line with invariant 6 of the data model.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from fasterrag.config.schema import Settings
from fasterrag.core.faithfulness import P3_TEMPLATE_VERSION
from fasterrag.core.generation import P1_TEMPLATE_VERSION
from fasterrag.core.retrieval.models import ScoredChunk

__all__ = [
    "SPAN_NAMES",
    "Span",
    "SpanRecorder",
    "StageClock",
    "Trace",
    "config_snapshot",
]

SPAN_NAMES: tuple[str, ...] = ("retrieval", "reranker", "context-assembly", "generation")


def config_snapshot(settings: Settings) -> dict[str, Any]:
    """Return the retrieval-affecting configuration and prompt templates a trace ran under."""
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
        "prompts": {
            "p1_template_version": P1_TEMPLATE_VERSION,
            "p3_template_version": P3_TEMPLATE_VERSION,
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
    """Collects spans for one query, timed against a single origin."""

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

    def add(self, span: Span) -> None:
        """Append a span timed elsewhere against this recorder's origin."""
        self._spans.append(span)

    @property
    def spans(self) -> list[Span]:
        """Return the spans recorded so far, in the order they closed."""
        return list(self._spans)


@dataclass(slots=True)
class _Window:
    """Where one stage was first and last active, and how long it was busy in all."""

    start_ms: float
    end_ms: float
    busy_ms: float = 0.0
    count: int = 0


class StageClock:
    """Times a stage that runs many times, concurrently, as one span per stage.

    ``observe`` is called with each interval's stage and seconds, which is how every
    interval also reaches a metric. ``close`` adds the spans to the recorder.
    """

    def __init__(
        self,
        recorder: SpanRecorder | None = None,
        observe: Callable[[str, float], None] | None = None,
    ) -> None:
        """Time against ``recorder``'s origin, reporting each interval to ``observe``."""
        self.recorder = recorder or SpanRecorder()
        self._observe = observe
        self._windows: dict[str, _Window] = {}

    @property
    def elapsed_ms(self) -> float:
        """Return milliseconds since the recorder started."""
        return self.recorder.elapsed_ms

    def interval(self, stage: str, start_ms: float, end_ms: float) -> None:
        """Count one interval of ``stage`` that ran from ``start_ms`` to ``end_ms``."""
        window = self._windows.get(stage)
        if window is None:
            window = self._windows[stage] = _Window(start_ms, end_ms)
        window.start_ms = min(window.start_ms, start_ms)
        window.end_ms = max(window.end_ms, end_ms)
        window.busy_ms += end_ms - start_ms
        window.count += 1
        if self._observe is not None:
            self._observe(stage, (end_ms - start_ms) / 1000)

    @contextmanager
    def timing(self, stage: str) -> Iterator[None]:
        """Count the block as one interval of ``stage``, whether or not it raised."""
        started = self.elapsed_ms
        try:
            yield
        finally:
            self.interval(stage, started, self.elapsed_ms)

    def busy_ms(self, stage: str) -> float:
        """Return the summed duration of every interval of ``stage`` so far."""
        window = self._windows.get(stage)
        return window.busy_ms if window is not None else 0.0

    def close(self) -> list[Span]:
        """Add one span per stage to the recorder, in the order the stages began."""
        spans = [
            Span(
                name=stage,
                start_ms=window.start_ms,
                end_ms=window.end_ms,
                attributes={"busy_ms": round(window.busy_ms, 3), "count": window.count},
            )
            for stage, window in sorted(self._windows.items(), key=lambda item: item[1].start_ms)
        ]
        for span in spans:
            self.recorder.add(span)
        self._windows.clear()
        return spans


@dataclass(frozen=True, slots=True)
class Trace:
    """Everything one query did, persisted for replay and inspection."""

    trace_id: str
    query: str
    collection: str | None = None
    tenant: str | None = None
    filters: dict[str, Any] | None = None
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    prompt: str = ""
    response: str = ""
    result: dict[str, Any] = field(default_factory=dict)
    spans: list[Span] = field(default_factory=list)
    created_at: str = ""
    kind: str = "query"

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized form, which is also the ``GET /v1/traces/{id}`` body."""
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "collection": self.collection,
            "tenant": self.tenant,
            "filters": self.filters,
            "config_snapshot": self.config_snapshot,
            "retrieved": self.retrieved,
            "prompt": self.prompt,
            "response": self.response,
            "result": self.result,
            "spans": [span.as_dict() for span in self.spans],
            "created_at": self.created_at,
            "kind": self.kind,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Trace:
        """Rebuild a trace from its persisted form."""
        return cls(
            trace_id=str(payload["trace_id"]),
            query=str(payload.get("query", "")),
            collection=payload.get("collection"),
            # CRITICAL: restored, not defaulted. Without this the tenant survives `as_dict`
            # but is lost on the way back, so every stored trace reads as untenanted — and
            # the tenant check in `TraceStore.load` would then hide every trace from the
            # tenant that owns it.
            tenant=payload.get("tenant"),
            filters=payload.get("filters"),
            config_snapshot=dict(payload.get("config_snapshot") or {}),
            retrieved=list(payload.get("retrieved") or []),
            prompt=str(payload.get("prompt", "")),
            response=str(payload.get("response", "")),
            result=dict(payload.get("result") or {}),
            spans=[Span.from_dict(span) for span in payload.get("spans") or []],
            created_at=str(payload.get("created_at", "")),
            kind=str(payload.get("kind") or "query"),
        )


def record_chunk(chunk: ScoredChunk) -> dict[str, Any]:
    """Return the full candidate record a trace keeps for one chunk."""
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
