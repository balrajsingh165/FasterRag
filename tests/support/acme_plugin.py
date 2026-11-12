"""A third-party provider package, written the way a plugin author would write one.

Everything here subclasses a public base class and nothing here is known to fasterRag: no
factory names these classes and no test hands one to a service. They are reached only
through entry points a fake distribution registers (``tests/support/plugins.py``), which is
the whole point — if a class in this module ends up serving a query, configuration selected
it by name.

The store is faithful rather than minimal. Both retrieval legs read the same points and
apply the same pushed-down filter, because ``services.querying`` requires that of any
backend and a double that returned canned hits would prove nothing about selection
reaching a working adapter.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import ClassVar

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.embeddings.tiering import matches
from fasterrag.adapters.llm.base import Completion, LLMAdapter
from fasterrag.adapters.vectordb.base import (
    CollectionInfo,
    CollectionSpec,
    HealthStatus,
    Point,
    PointSelector,
    PointUpdate,
    ScoredPoint,
    SearchQuery,
    UpsertResult,
    VectorDBAdapter,
)
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError

__all__ = ["ANSWER_PREFIX", "DIMENSIONS", "AcmeEmbedder", "AcmeLLM", "AcmeVectorDB"]

DIMENSIONS = 8
ANSWER_PREFIX = "acme-llm:"


def _vector(text: str) -> list[float]:
    """Return a deterministic vector for a piece of text."""
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [digest[index] / 255.0 for index in range(DIMENSIONS)]


class AcmeVectorDB(VectorDBAdapter):
    """An in-memory vector database honouring the whole adapter contract."""

    adapter_api_version: ClassVar[str] = "1.0"

    def __init__(self, settings: Settings) -> None:
        """Start with no collections, aliases or snapshots."""
        super().__init__(settings)
        self.specs: dict[str, CollectionSpec] = {}
        self.points: dict[str, dict[str, Point]] = {}
        self.aliases: dict[str, str] = {}
        self.snapshots: dict[str, dict[str, dict[str, Point]]] = {}

    def _resolve(self, name: str) -> str:
        """Return the collection a name or alias refers to, refusing an unknown one."""
        target = self.aliases.get(name, name)
        if target not in self.points:
            raise FasterRagError(
                f"collection {name!r} does not exist", code=ErrorCode.NOT_FOUND, retryable=False
            )
        return target

    async def create_collection(self, spec: CollectionSpec) -> None:
        """Create a collection unless it already exists."""
        self.specs.setdefault(spec.name, spec)
        self.points.setdefault(spec.name, {})

    async def list_collections(self) -> list[CollectionInfo]:
        """Return every collection held."""
        return [
            CollectionInfo(
                name=name,
                vectors=len(self.points[name]),
                dimensions=spec.dimensions,
                distance=spec.distance,
                sparse=spec.sparse,
            )
            for name, spec in sorted(self.specs.items())
        ]

    async def drop_collection(self, name: str) -> bool:
        """Delete a collection, reporting whether it existed."""
        self.specs.pop(name, None)
        return self.points.pop(name, None) is not None

    async def snapshot(self, collection: str) -> str:
        """Copy a collection's points under a new snapshot name."""
        held = self.snapshots.setdefault(collection, {})
        name = f"{collection}-{len(held) + 1}"
        held[name] = dict(self.points[self._resolve(collection)])
        return name

    async def list_snapshots(self, collection: str) -> list[str]:
        """Return a collection's snapshots, newest last."""
        return list(self.snapshots.get(collection, {}))

    async def delete_snapshot(self, collection: str, snapshot: str) -> bool:
        """Delete one snapshot, reporting whether it existed."""
        return self.snapshots.get(collection, {}).pop(snapshot, None) is not None

    async def restore_snapshot(self, collection: str, snapshot: str) -> None:
        """Replace a collection's points with a snapshot's."""
        held = self.snapshots.get(collection, {}).get(snapshot)
        if held is None:
            raise FasterRagError(
                f"snapshot {snapshot!r} does not exist", code=ErrorCode.NOT_FOUND, retryable=False
            )
        self.points[self._resolve(collection)] = dict(held)

    async def set_alias(self, alias: str, collection: str) -> None:
        """Point an alias at a collection."""
        self.aliases[alias] = collection

    async def alias_target(self, alias: str) -> str | None:
        """Return the collection an alias resolves to."""
        return self.aliases.get(alias)

    async def delete_alias(self, alias: str) -> bool:
        """Remove an alias, reporting whether one existed."""
        return self.aliases.pop(alias, None) is not None

    async def upsert(self, points: list[Point]) -> UpsertResult:
        """Write points, overwriting any with the same id."""
        for point in points:
            self.points[self._resolve(point.collection)][point.point_id] = point
        return UpsertResult(upserted=len(points))

    async def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        """Yield every point in a collection."""
        for point in list(self.points[self._resolve(collection)].values()):
            yield point if with_vectors else replace(point, vector=(), sparse=None)

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        """Return the best matches on whichever leg the query carries."""
        scored: list[ScoredPoint] = []
        for point in self.points[self._resolve(query.collection)].values():
            if query.filters is not None and not matches(query.filters, point.payload):
                continue
            scored.append(
                ScoredPoint(
                    point_id=point.point_id,
                    score=self._score(query, point),
                    payload=dict(point.payload),
                )
            )
        scored.sort(key=lambda hit: (-hit.score, hit.point_id))
        return scored[: query.limit]

    @staticmethod
    def _score(query: SearchQuery, point: Point) -> float:
        """Score a point on the leg the query asked for."""
        if query.sparse is not None:
            if point.sparse is None:
                return 0.0
            weights = dict(zip(point.sparse.indices, point.sparse.values, strict=True))
            return sum(weights.get(index, 0.0) for index in query.sparse.indices)
        return sum(a * b for a, b in zip(query.vector or (), point.vector, strict=True))

    async def update(self, updates: list[PointUpdate]) -> None:
        """Merge metadata into existing points."""
        for change in updates:
            held = self.points[self._resolve(change.collection)]
            point = held[change.point_id]
            held[change.point_id] = replace(point, payload={**point.payload, **change.payload})

    async def delete(self, selector: PointSelector) -> None:
        """Delete the selected points."""
        held = self.points[self._resolve(selector.collection)]
        if selector.point_ids is not None:
            doomed = set(selector.point_ids)
        else:
            doomed = {
                point_id
                for point_id, point in held.items()
                if matches(selector.filters or {}, point.payload)
            }
        for point_id in doomed:
            held.pop(point_id, None)

    async def health(self) -> HealthStatus:
        """Report healthy; the store is a dictionary."""
        return HealthStatus(healthy=True, detail="acme reachable", latency_ms=0.0)

    async def close(self) -> None:
        """Release nothing."""


class AcmeEmbedder(EmbeddingAdapter):
    """A deterministic embedder that needs no model and no network."""

    provider: ClassVar[str] = "acme"
    adapter_api_version: ClassVar[str] = "1.0"

    @property
    def model(self) -> str:
        """Return the configured model name."""
        return self.config.model

    @property
    def model_version(self) -> str:
        """Return a fixed version."""
        return "acme-1"

    @property
    def dimensions(self) -> int | None:
        """Return the fixed vector size."""
        return DIMENSIONS

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed passages."""
        return EmbeddingResult(
            vectors=[_vector(text) for text in texts],
            model=self.model,
            model_version=self.model_version,
        )

    async def embed_query(self, text: str) -> list[float]:
        """Embed one query."""
        return _vector(text)

    async def health(self) -> HealthStatus:
        """Report healthy."""
        return HealthStatus(healthy=True, detail="acme reachable", latency_ms=0.0)

    async def close(self) -> None:
        """Release nothing."""


class AcmeLLM(LLMAdapter):
    """Answers with the prompt it was given, marked so its origin is visible."""

    provider: ClassVar[str] = "acme"
    adapter_api_version: ClassVar[str] = "1.0"

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        """Return the prompt back as the answer."""
        return Completion(
            text=f"{ANSWER_PREFIX} {prompt}",
            model=self.model,
            prompt_tokens=1,
            completion_tokens=1,
        )

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Yield the answer as one delta."""
        yield f"{ANSWER_PREFIX} {prompt}"

    async def health(self) -> HealthStatus:
        """Report healthy."""
        return HealthStatus(healthy=True, detail="acme reachable", latency_ms=0.0)

    async def close(self) -> None:
        """Release nothing."""
