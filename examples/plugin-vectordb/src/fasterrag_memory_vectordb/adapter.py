"""An in-process vector database, written as a third-party fasterRag plugin would be.

Everything here uses only the public extension surface: the ``VectorDBAdapter`` base class and
the request and result types next to it, ``validate_filter`` for the shared filter grammar, and
the typed errors from ``fasterrag.errors``. The package registers the class under the
``fasterrag.vectordb`` entry-point group (see ``pyproject.toml``), which is all it takes for
``vector_db.provider: memory`` to select it.

The store holds everything in dictionaries, so it lives and dies with its process. It is still
a faithful implementation of the contract — cosine, dot and Euclidean scoring, pushed-down
filters, a BM25 keyword leg whose IDF comes from the live collection, aliases and snapshots —
because the point of the example is that it passes ``tests/contract/vectordb.py`` unmodified.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from fasterrag.adapters.vectordb.base import (
    CollectionInfo,
    CollectionSpec,
    Filter,
    HealthStatus,
    Point,
    PointSelector,
    PointUpdate,
    ScoredPoint,
    SearchQuery,
    SparseVector,
    UpsertResult,
    VectorDBAdapter,
    validate_filter,
)
from fasterrag.config.schema import Settings
from fasterrag.errors import EmbedError, ErrorCode, FasterRagError

__all__ = ["MemoryVectorDB"]


@dataclass
class _Collection:
    """One collection: its shape and its points, keyed by point id."""

    spec: CollectionSpec
    points: dict[str, Point] = field(default_factory=dict)


def _not_found(name: str) -> FasterRagError:
    """Return the typed error a caller branches on to answer 404."""
    return FasterRagError(
        f"collection {name!r} does not exist", code=ErrorCode.NOT_FOUND, retryable=False
    )


def _compare(operator: str, actual: Any, expected: Any) -> bool:
    """Apply one comparison operator from the shared filter grammar."""
    if operator == "$eq":
        return bool(actual == expected)
    if operator == "$ne":
        return bool(actual != expected)
    if operator == "$in":
        return actual in expected
    if operator == "$nin":
        return actual not in expected
    if actual is None:
        return False
    if operator == "$gt":
        return bool(actual > expected)
    if operator == "$gte":
        return bool(actual >= expected)
    if operator == "$lt":
        return bool(actual < expected)
    return bool(actual <= expected)


def _matches(filters: Filter | None, payload: Mapping[str, Any]) -> bool:
    """Return whether a payload satisfies every condition of a filter."""
    for key, condition in (filters or {}).items():
        actual = payload.get(key)
        if isinstance(condition, Mapping):
            operator, expected = next(iter(condition.items()))
            if key not in payload and operator not in ("$ne", "$nin"):
                return False
            if not _compare(operator, actual, expected):
                return False
        elif key not in payload or actual != condition:
            return False
    return True


def _dense_score(distance: str, query: Sequence[float], stored: Sequence[float]) -> float:
    """Score a stored vector against a query, higher meaning closer."""
    dot = sum(a * b for a, b in zip(query, stored, strict=True))
    if distance == "dot":
        return dot
    if distance == "euclid":
        return -math.dist(query, stored)
    norms = math.hypot(*query) * math.hypot(*stored)
    return dot / norms if norms else 0.0


class MemoryVectorDB(VectorDBAdapter):
    """A vector database held in this process's memory."""

    adapter_api_version: ClassVar[str] = "1.0"

    def __init__(self, settings: Settings) -> None:
        """Start empty: no collections, aliases or snapshots."""
        super().__init__(settings)
        self._collections: dict[str, _Collection] = {}
        self._aliases: dict[str, str] = {}
        self._snapshots: dict[str, dict[str, _Collection]] = {}

    def _resolve(self, name: str) -> _Collection:
        """Return the collection a name or alias refers to, refusing an unknown one."""
        held = self._collections.get(self._aliases.get(name, name))
        if held is None:
            raise _not_found(name)
        return held

    async def create_collection(self, spec: CollectionSpec) -> None:
        """Create a collection, or return quietly if one with the same shape exists."""
        held = self._collections.get(spec.name)
        if held is None:
            self._collections[spec.name] = _Collection(spec=spec)
            return
        if held.spec.dimensions != spec.dimensions:
            raise FasterRagError(
                f"collection {spec.name!r} already exists with {held.spec.dimensions} "
                f"dimensions, not {spec.dimensions}",
                code=ErrorCode.CONFLICT,
            )
        if spec.sparse and not held.spec.sparse:
            raise FasterRagError(
                f"collection {spec.name!r} already exists without a keyword leg",
                code=ErrorCode.CONFLICT,
            )

    async def list_collections(self) -> list[CollectionInfo]:
        """Return every collection with its size and shape."""
        return [
            CollectionInfo(
                name=name,
                vectors=len(held.points),
                dimensions=held.spec.dimensions,
                distance=held.spec.distance,
                sparse=held.spec.sparse,
            )
            for name, held in sorted(self._collections.items())
        ]

    async def drop_collection(self, name: str) -> bool:
        """Delete a collection, reporting whether it existed."""
        return self._collections.pop(name, None) is not None

    async def snapshot(self, collection: str) -> str:
        """Copy a collection under a new snapshot name and return the name."""
        held = self._resolve(collection)
        taken = self._snapshots.setdefault(collection, {})
        name = f"{collection}-snapshot-{len(taken) + 1}"
        taken[name] = _Collection(spec=held.spec, points=dict(held.points))
        return name

    async def list_snapshots(self, collection: str) -> list[str]:
        """Return a collection's snapshots, newest last."""
        return list(self._snapshots.get(collection, {}))

    async def delete_snapshot(self, collection: str, snapshot: str) -> bool:
        """Delete one snapshot, reporting whether it existed."""
        return self._snapshots.get(collection, {}).pop(snapshot, None) is not None

    async def restore_snapshot(self, collection: str, snapshot: str) -> None:
        """Replace a collection with a copy of one of its snapshots."""
        taken = self._snapshots.get(collection, {}).get(snapshot)
        if taken is None:
            raise FasterRagError(
                f"collection {collection!r} has no snapshot {snapshot!r}",
                code=ErrorCode.NOT_FOUND,
                retryable=False,
            )
        self._collections[collection] = _Collection(spec=taken.spec, points=dict(taken.points))

    async def set_alias(self, alias: str, collection: str) -> None:
        """Point an alias at a collection, replacing any previous target."""
        self._aliases[alias] = collection

    async def alias_target(self, alias: str) -> str | None:
        """Return the collection an alias resolves to, or None."""
        return self._aliases.get(alias)

    async def delete_alias(self, alias: str) -> bool:
        """Remove an alias, reporting whether one existed."""
        return self._aliases.pop(alias, None) is not None

    async def upsert(self, points: list[Point]) -> UpsertResult:
        """Write points, overwriting any with the same id; refuse a wrong-sized vector."""
        for point in points:
            held = self._resolve(point.collection)
            if len(point.vector) != held.spec.dimensions:
                raise EmbedError(
                    f"point {point.point_id!r} has {len(point.vector)} dimensions but "
                    f"collection {held.spec.name!r} stores {held.spec.dimensions}",
                    retryable=False,
                )
            sparse = point.sparse if held.spec.sparse else None
            if sparse is not None:
                ordered = sorted(zip(sparse.indices, sparse.values, strict=True))
                sparse = SparseVector(
                    indices=[index for index, _ in ordered], values=[value for _, value in ordered]
                )
            held.points[point.point_id] = replace(
                point, collection=held.spec.name, payload=dict(point.payload), sparse=sparse
            )
        return UpsertResult(upserted=len(points))

    async def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        """Yield every point in a collection, with both legs when asked for vectors."""
        held = self._resolve(collection)
        for point in list(held.points.values()):
            yield point if with_vectors else replace(point, vector=(), sparse=None)

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        """Return the best matches on whichever leg the query carries, filters applied."""
        validate_filter(query.filters)
        held = self._resolve(query.collection)
        candidates = [p for p in held.points.values() if _matches(query.filters, p.payload)]
        if query.sparse is not None:
            scored = self._keyword_scores(query.sparse, held, candidates)
        else:
            assert query.vector is not None
            scored = [
                (p, _dense_score(held.spec.distance, query.vector, p.vector)) for p in candidates
            ]
        scored.sort(key=lambda hit: (-hit[1], hit[0].point_id))
        return [
            ScoredPoint(
                point_id=point.point_id,
                score=score,
                payload=dict(point.payload) if query.with_payload else {},
                vector=list(point.vector) if query.with_vectors else None,
            )
            for point, score in scored[: query.limit]
        ]

    @staticmethod
    def _keyword_scores(
        query: SparseVector, held: _Collection, candidates: list[Point]
    ) -> list[tuple[Point, float]]:
        """Score candidates with BM25, taking IDF from the whole live collection."""
        stored = [dict(zip(p.sparse.indices, p.sparse.values, strict=True)) for p in candidates]
        everything = [p.sparse for p in held.points.values() if p.sparse is not None]
        total = len(everything)
        idf: dict[int, float] = {}
        for index in query.indices:
            documents = sum(1 for vector in everything if index in vector.indices)
            if documents:
                idf[index] = math.log(1 + (total - documents + 0.5) / (documents + 0.5))

        scored: list[tuple[Point, float]] = []
        for point, weights in zip(candidates, stored, strict=True):
            score = sum(
                weights[index] * weight * idf[index]
                for index, weight in zip(query.indices, query.values, strict=True)
                if index in weights and index in idf
            )
            if score > 0:
                scored.append((point, score))
        return scored

    async def update(self, updates: list[PointUpdate]) -> None:
        """Merge metadata into existing points without touching their vectors."""
        for change in updates:
            held = self._resolve(change.collection)
            point = held.points.get(change.point_id)
            if point is not None:
                held.points[change.point_id] = replace(
                    point, payload={**point.payload, **change.payload}
                )

    async def delete(self, selector: PointSelector) -> None:
        """Delete the selected points; selecting nothing is not an error."""
        validate_filter(selector.filters)
        held = self._resolve(selector.collection)
        if selector.point_ids is not None:
            doomed = set(selector.point_ids)
        else:
            doomed = {
                point_id
                for point_id, point in held.points.items()
                if _matches(selector.filters, point.payload)
            }
        for point_id in doomed:
            held.points.pop(point_id, None)

    async def health(self) -> HealthStatus:
        """Report healthy: there is no server to be unreachable."""
        return HealthStatus(healthy=True, detail="in-process memory store", latency_ms=0.0)

    async def close(self) -> None:
        """Release nothing; the store is garbage-collected with the adapter."""
