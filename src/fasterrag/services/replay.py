"""Time-travel replay (D8).

Re-executes a past query under a candidate configuration and reports what changed. Trace
viewers elsewhere show what happened; the point of this one is answering *why an answer
changed* by running history forward again under a different config.

The diff is structured rather than textual. "The answer is different" is not a finding —
"chunk ``c_9f2`` fell from rank 2 to rank 7 and left the context, and ``rrf_k`` went from 60
to 10" is. Retrieval changes are therefore reported as chunks **added**, **removed**, and
**reordered**, and separately from the configuration keys that differ.

A replay under an identical configuration must reproduce the original retrieval set exactly;
that determinism is the acceptance test for D8, and it is what makes any observed difference
attributable to the config change rather than to noise.

Replay never writes: it does not store its own trace, does not populate the semantic cache,
and does not touch the index. Investigating an incident must not alter the evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from fasterrag.config.schema import Settings
from fasterrag.core.tracing import Trace, config_snapshot, record_chunk
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.generation import GenerationService

__all__ = ["ReplayResult", "RetrievalDiff", "diff_config", "diff_retrieval", "replay_trace"]

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalDiff:
    """How a replayed retrieval set differs from the original."""

    added: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    reordered: list[dict[str, Any]] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        """Return whether the two retrieval sets are the same chunks in the same order."""
        return not (self.added or self.removed or self.reordered)

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized form."""
        return {
            "identical": self.identical,
            "added": self.added,
            "removed": self.removed,
            "reordered": self.reordered,
        }


def diff_retrieval(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> RetrievalDiff:
    """Compare two retrieval sets by chunk id and position.

    Args:
        before: The original trace's candidates, in rank order.
        after: The replayed candidates, in rank order.

    Returns:
        The chunks gained, lost, and moved. A chunk that merely changed score without
        changing position is not reported: the position is what decides whether it reaches
        the context, and a score that moved without moving anything is not a finding.
    """
    original = {str(chunk["chunk_id"]): index for index, chunk in enumerate(before)}
    replayed = {str(chunk["chunk_id"]): index for index, chunk in enumerate(after)}

    added = [chunk_id for chunk_id in replayed if chunk_id not in original]
    removed = [chunk_id for chunk_id in original if chunk_id not in replayed]
    reordered = [
        {"chunk_id": chunk_id, "was": original[chunk_id] + 1, "now": replayed[chunk_id] + 1}
        for chunk_id in replayed
        if chunk_id in original and original[chunk_id] != replayed[chunk_id]
    ]

    return RetrievalDiff(added=added, removed=removed, reordered=reordered)


def _flatten(payload: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested config snapshot into dotted keys, so a diff names the exact key."""
    if not isinstance(payload, dict):
        return {prefix: payload}

    flattened: dict[str, Any] = {}
    for key, value in payload.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flattened.update(_flatten(value, path))
    return flattened


def diff_config(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    """Return every retrieval-affecting key whose value differs, with both values."""
    original = _flatten(before)
    candidate = _flatten(after)

    changes: list[dict[str, Any]] = []
    for key in sorted(set(original) | set(candidate)):
        was = original.get(key)
        now = candidate.get(key)
        if was != now:
            changes.append({"key": key, "was": was, "now": now})
    return changes


@dataclass(frozen=True, slots=True)
class ReplayResult:
    """A past query re-executed, beside what it originally produced."""

    trace_id: str
    query: str
    config_changes: list[dict[str, Any]] = field(default_factory=list)
    retrieval: RetrievalDiff = field(default_factory=RetrievalDiff)
    original_answer: str | None = None
    replayed_answer: str | None = None
    original_citations: list[str] = field(default_factory=list)
    replayed_citations: list[str] = field(default_factory=list)

    @property
    def answer_changed(self) -> bool:
        """Return whether the replayed answer differs from the original."""
        return self.original_answer != self.replayed_answer

    @property
    def deterministic(self) -> bool:
        """Return whether an unchanged config reproduced the original retrieval exactly.

        Only meaningful when nothing changed: with a different config a difference is the
        expected outcome, not a determinism failure.
        """
        return bool(not self.config_changes and self.retrieval.identical)

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized diff, the body of ``POST /v1/replay``."""
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "config_changes": self.config_changes,
            "retrieval": self.retrieval.as_dict(),
            "answer_changed": self.answer_changed,
            "original": {"answer": self.original_answer, "citations": self.original_citations},
            "replayed": {"answer": self.replayed_answer, "citations": self.replayed_citations},
        }


def _citation_ids(result: dict[str, Any]) -> list[str]:
    """Return the chunk ids a stored result cited."""
    return [
        str(citation.get("chunk_id", ""))
        for citation in result.get("citations") or []
        if isinstance(citation, dict)
    ]


async def replay_trace(
    trace: Trace,
    candidate: Settings,
    service: GenerationService,
) -> ReplayResult:
    """Re-execute ``trace``'s query under ``candidate`` and diff the outcome.

    Args:
        trace: The stored trace to re-execute.
        candidate: The configuration to replay under.
        service: A generation service built from ``candidate``. It must have no trace store,
            so the replay leaves no trace of its own — an investigation that adds to the
            evidence it is examining is worse than useless.

    Returns:
        The structured diff.

    Raises:
        FasterRagError: With ``VALIDATION_FAILED`` if the service would record a trace,
            because that would silently corrupt the record being investigated.
    """
    if service.traces is not None and service.traces.enabled:
        raise FasterRagError(
            "a replay must run against a service with tracing disabled, so investigating a "
            "trace cannot add to the traces being investigated",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )

    replayed, retrieved = await service.answer_with_candidates(
        trace.query,
        collection=trace.collection,
        filters=trace.filters,
    )
    candidates = [record_chunk(chunk) for chunk in retrieved]

    result = ReplayResult(
        trace_id=trace.trace_id,
        query=trace.query,
        config_changes=diff_config(trace.config_snapshot, config_snapshot(candidate)),
        retrieval=diff_retrieval(trace.retrieved, candidates),
        original_answer=trace.result.get("answer"),
        replayed_answer=replayed.answer,
        original_citations=_citation_ids(trace.result),
        replayed_citations=[citation.chunk_id for citation in replayed.citations],
    )

    _logger.info(
        "replayed a stored trace",
        extra={
            "trace_id": trace.trace_id,
            "config_changes": len(result.config_changes),
            "retrieval_identical": result.retrieval.identical,
            "answer_changed": result.answer_changed,
        },
    )
    return result
