"""Semantic response cache: answer a paraphrase from the answer already given.

Keyed by the *meaning* of a query rather than its text (``docs/architecture.md`` §7). "what
is the notice period?" and "how much notice do I have to give?" are the same question, and an
exact-match cache serves neither from the other. A stored answer is reused when the cosine
similarity of the two query vectors clears ``cache.similarity_threshold``.

That is also why the feature defaults to ``false`` and why the threshold is bounded to
0.90-0.99: a threshold set too loose answers one question with another question's answer,
which is worse than a cache miss and invisible without the eval probe of FMEA row 25.

**Invalidation is TTL plus event.** Entries expire after ``cache.ttl``, and a corpus change
drops every entry immediately regardless of remaining TTL. A cached answer describes the
corpus at the moment it was generated; once documents are added, removed, or reindexed, that
description may be wrong, and a stale answer must never outlive the data it came from.

Matching is a linear cosine scan over live entries, bounded by ``MAXIMUM_ENTRIES``. A cache
that needed an index to search itself would be a second vector database, and this one is
sized in thousands of entries — a scan is cheaper than the retrieval call it is avoiding.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fasterrag.config.schema import Settings
from fasterrag.core.cache.embedding import decode_vector, encode_vector
from fasterrag.core.cache.stats import CacheStats
from fasterrag.core.cache.store import CacheStore, MemoryStore
from fasterrag.errors import CacheError
from fasterrag.observability.logging import get_logger

__all__ = ["CacheHit", "SemanticCache", "cosine_similarity"]

_logger = get_logger(__name__)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two vectors, or zero if either has no magnitude.

    Vectors of different lengths score zero rather than raising: a length mismatch means the
    entry was written under a different embedding model, which is a miss, not an error.
    """
    if len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


@dataclass(frozen=True, slots=True)
class CacheHit:
    """A stored response and how closely its query matched."""

    response: dict[str, Any]
    similarity: float
    question: str

    def as_dict(self) -> dict[str, Any]:
        """Return the ``cache`` member of a query response."""
        return {"semantic_hit": True, "similarity": round(self.similarity, 6)}


MISS: dict[str, Any] = {"semantic_hit": False, "similarity": None}


def _encode(
    question: str, vector: Sequence[float], response: dict[str, Any], tenant: str | None = None
) -> bytes:
    """Pack an entry as a length-prefixed vector followed by JSON.

    The vector is packed rather than serialized into the JSON because it is the part read on
    every single lookup, and parsing a thousand JSON float arrays per query would make the
    cache slower than the pipeline it is short-circuiting.
    """
    packed = encode_vector(vector)
    header = json.dumps(
        {"question": question, "vector_bytes": len(packed), "tenant": tenant}
    ).encode("utf-8")
    return b"".join(
        (len(header).to_bytes(4, "little"), header, packed, json.dumps(response).encode("utf-8"))
    )


def _decode(raw: bytes) -> tuple[str, list[float], dict[str, Any], str | None] | None:
    """Return the question, vector, response, and tenant in ``raw``, or ``None`` if malformed."""
    try:
        if len(raw) < 4:
            return None
        header_length = int.from_bytes(raw[:4], "little")
        header = json.loads(raw[4 : 4 + header_length])
        start = 4 + header_length
        end = start + int(header["vector_bytes"])
        vector = decode_vector(raw[start:end])
        if vector is None:
            return None
        response = json.loads(raw[end:])
    except (KeyError, ValueError, TypeError, UnicodeDecodeError):
        return None

    if not isinstance(response, dict):
        return None
    tenant = header.get("tenant")
    return (
        str(header.get("question", "")),
        vector,
        response,
        str(tenant) if tenant is not None else None,
    )


class SemanticCache:
    """Serves a previous answer when a new query means the same thing."""

    def __init__(
        self, settings: Settings, store: CacheStore | None = None, *, enabled: bool | None = None
    ) -> None:
        """Build the cache.

        Args:
            settings: Validated configuration; supplies the threshold and TTL.
            store: Backend holding entries. Defaults to in-process memory.
            enabled: Overrides ``cache.semantic``, for callers that decide per instance.
        """
        self.settings = settings
        self.config = settings.cache
        self.store = store or MemoryStore()
        self.enabled = self.config.semantic if enabled is None else enabled
        self.stats = CacheStats(name="semantic")

    @property
    def threshold(self) -> float:
        """Return the similarity a stored query must clear to be reused."""
        return self.config.similarity_threshold

    async def lookup(
        self, vector: Sequence[float], *, tenant: str | None = None
    ) -> CacheHit | None:
        """Return the closest stored response above the threshold, or ``None``.

        A backend failure is a miss, not an error: the query runs the full pipeline and the
        caller never learns the cache was unavailable (FMEA row 23).

        # CRITICAL: entries belonging to another tenant are skipped, and the check is on the
        # entry itself rather than on the key. Lookup compares vectors across every stored
        # entry, so without this a sufficiently similar question from one tenant returns
        # another tenant's answer — a cross-tenant disclosure with a cache hit's latency and
        # no error anywhere. A key prefix alone would not help, because the scan ignores keys.
        """
        if not self.enabled:
            return None

        try:
            entries = await self.store.items()
        except CacheError as exc:
            self.stats.record_error()
            _logger.warning(
                "semantic cache read failed; running the full pipeline",
                extra={"code": exc.code.value, "trace_id": exc.trace_id},
            )
            return None

        best: CacheHit | None = None
        for _, raw in entries:
            decoded = _decode(raw)
            if decoded is None:
                continue
            question, stored, response, owner = decoded
            if owner != tenant:
                continue
            similarity = cosine_similarity(vector, stored)
            if similarity >= self.threshold and (best is None or similarity > best.similarity):
                best = CacheHit(response=response, similarity=similarity, question=question)

        if best is None:
            self.stats.record_miss()
            return None

        self.stats.record_hit()
        _logger.info(
            "semantic cache hit",
            extra={"similarity": round(best.similarity, 4), "threshold": self.threshold},
        )
        return best

    async def store_response(
        self,
        question: str,
        vector: Sequence[float],
        response: dict[str, Any],
        *,
        tenant: str | None = None,
    ) -> None:
        """Remember ``response`` as the answer to anything close to ``vector``.

        A write failure is logged and dropped. The answer has already been produced, and
        failing the query because it could not be filed away would be absurd.
        """
        if not self.enabled:
            return

        try:
            # The tenant is in the key as well as the entry: without it two tenants asking
            # the same question collide and the second silently overwrites the first.
            await self.store.set(
                f"sem:{tenant or '-'}:{question}",
                _encode(question, vector, response, tenant),
                ttl=self.config.ttl,
            )
        except CacheError as exc:
            self.stats.record_error()
            _logger.warning(
                "semantic cache write failed; the answer is still returned",
                extra={"code": exc.code.value, "trace_id": exc.trace_id},
            )

    async def invalidate(self, reason: str) -> None:
        """Drop every entry because the corpus changed.

        Called on ingest, delete, and reindex. Deliberately total rather than selective:
        deciding which cached answers a new document could have changed would mean knowing
        what each answer was derived from *and* what the new document says, and a cache that
        guesses wrong here serves a confidently stale answer.
        """
        if not self.enabled:
            return

        try:
            entries = await self.store.items()
            await self.store.clear()
        except CacheError as exc:
            self.stats.record_error()
            _logger.warning(
                "semantic cache invalidation failed; entries may be stale until they expire",
                extra={"code": exc.code.value, "trace_id": exc.trace_id, "reason": reason},
            )
            return

        self.stats.record_invalidation(len(entries))
        _logger.info(
            "semantic cache invalidated by a corpus change",
            extra={"reason": reason, "entries": len(entries)},
        )

    async def close(self) -> None:
        """Release the backend."""
        await self.store.close()
