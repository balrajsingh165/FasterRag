"""Vendor-neutral vector database contract.

The interface in ``docs/architecture.md`` §10, plus the request and result types its
methods exchange. These types are the whole vocabulary core code uses to talk to a
vector database, so swapping ``vector_db.provider`` cannot ripple past this package.

Each method takes exactly one request object, and that object names the collection it
targets — adapters are stateless with respect to collections.

**Metadata filters** are vendor-neutral mappings translated by each adapter and pushed
down to the backend. Raw expressions are never passed through to a vendor
(``docs/security.md`` §5). The supported grammar is a field name mapped either to a
scalar (equality) or to a single-operator object:

``{"department": "legal", "year": {"$gte": 2024}, "status": {"$in": ["a", "b"]}}``
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError

__all__ = [
    "COMPARISON_OPERATORS",
    "CollectionInfo",
    "CollectionSpec",
    "Distance",
    "Filter",
    "HealthStatus",
    "Point",
    "PointSelector",
    "PointUpdate",
    "ScoredPoint",
    "SearchQuery",
    "SparseVector",
    "UpsertResult",
    "VectorDBAdapter",
    "validate_filter",
]

Distance = Literal["cosine", "dot", "euclid"]
Filter = Mapping[str, Any]

COMPARISON_OPERATORS: Final[frozenset[str]] = frozenset(
    {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin"}
)

_SET_OPERATORS: Final[frozenset[str]] = frozenset({"$in", "$nin"})


@dataclass(frozen=True, slots=True)
class SparseVector:
    """Term indices paired with their weights, for the keyword retrieval leg."""

    indices: Sequence[int]
    values: Sequence[float]

    def __post_init__(self) -> None:
        """Reject a vector whose indices and values do not line up."""
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"a sparse vector has {len(self.indices)} indices but {len(self.values)} values"
            )

    def __len__(self) -> int:
        """Return the number of terms."""
        return len(self.indices)

    @property
    def empty(self) -> bool:
        """Return whether the vector carries no terms."""
        return not self.indices


@dataclass(frozen=True, slots=True)
class CollectionSpec:
    """The shape of a collection to create."""

    name: str
    dimensions: int
    distance: Distance = "cosine"
    shard_number: int = 1
    replication_factor: int = 1
    sparse: bool = False


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    """What a collection contains, in vendor-neutral terms."""

    name: str
    vectors: int
    dimensions: int | None = None
    distance: Distance | None = None
    sparse: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return the form ``fasterrag index list --json`` prints."""
        return {
            "name": self.name,
            "vectors": self.vectors,
            "dimensions": self.dimensions,
            "distance": self.distance,
            "sparse": self.sparse,
        }


@dataclass(frozen=True, slots=True)
class Point:
    """One indexed vector with its metadata payload."""

    point_id: str
    collection: str
    vector: Sequence[float]
    payload: Mapping[str, Any] = field(default_factory=dict)
    sparse: SparseVector | None = None


@dataclass(frozen=True, slots=True)
class UpsertResult:
    """How many points an upsert wrote."""

    upserted: int


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """One retrieval leg: dense similarity or sparse keyword, with optional filtering."""

    collection: str
    vector: Sequence[float] | None = None
    limit: int = 10
    filters: Filter | None = None
    with_payload: bool = True
    with_vectors: bool = False
    sparse: SparseVector | None = None

    def __post_init__(self) -> None:
        """Require exactly one leg, so an empty query cannot silently return nothing."""
        if (self.vector is None) == (self.sparse is None):
            raise FasterRagError(
                "a search must carry exactly one of a dense vector or a sparse vector",
                code=ErrorCode.VALIDATION_FAILED,
            )


@dataclass(frozen=True, slots=True)
class ScoredPoint:
    """One search hit."""

    point_id: str
    score: float
    payload: Mapping[str, Any] = field(default_factory=dict)
    vector: Sequence[float] | None = None


@dataclass(frozen=True, slots=True)
class PointUpdate:
    """A metadata-only update to an already-indexed point."""

    point_id: str
    collection: str
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class PointSelector:
    """What to delete: explicit ids or everything matching a filter, never both."""

    collection: str
    point_ids: Sequence[str] | None = None
    filters: Filter | None = None

    def __post_init__(self) -> None:
        """Reject an ambiguous or empty selector before it reaches a backend."""
        if (self.point_ids is None) == (self.filters is None):
            raise FasterRagError(
                "a point selector must set exactly one of point_ids or filters",
                code=ErrorCode.VALIDATION_FAILED,
            )


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Whether a backend is answering, and how quickly."""

    healthy: bool
    detail: str | None = None
    latency_ms: float | None = None


def validate_filter(filters: Filter | None) -> None:
    """Reject filter expressions outside the supported grammar."""
    if filters is None:
        return

    for key, condition in filters.items():
        if not isinstance(condition, Mapping):
            continue

        operators = set(condition)
        unsupported = operators - COMPARISON_OPERATORS
        if unsupported:
            supported = ", ".join(sorted(COMPARISON_OPERATORS))
            raise FasterRagError(
                f"filter on {key!r} uses unsupported operators "
                f"{sorted(unsupported)}; supported operators are: {supported}",
                code=ErrorCode.VALIDATION_FAILED,
            )
        if len(operators) != 1:
            raise FasterRagError(
                f"filter on {key!r} must use exactly one operator, got {sorted(operators)}",
                code=ErrorCode.VALIDATION_FAILED,
            )

        operator = next(iter(operators))
        if operator in _SET_OPERATORS and not isinstance(condition[operator], list):
            raise FasterRagError(
                f"filter on {key!r} with {operator} requires a list of values",
                code=ErrorCode.VALIDATION_FAILED,
            )


class VectorDBAdapter(ABC):
    """Vendor-neutral vector database contract."""

    def __init__(self, settings: Settings) -> None:
        """Build the adapter from validated configuration."""
        self.settings = settings

    @abstractmethod
    async def create_collection(self, spec: CollectionSpec) -> None:
        """Create a collection, or return quietly if it already matches ``spec``."""

    @abstractmethod
    async def list_collections(self) -> list[CollectionInfo]:
        """Return every collection the backend holds."""

    @abstractmethod
    async def drop_collection(self, name: str) -> bool:
        """Delete a collection and everything in it."""

    @abstractmethod
    async def snapshot(self, collection: str) -> str:
        """Take a backend-native snapshot of a collection and return its name."""

    @abstractmethod
    async def list_snapshots(self, collection: str) -> list[str]:
        """Return the snapshots the backend holds for a collection, newest last."""

    @abstractmethod
    async def delete_snapshot(self, collection: str, snapshot: str) -> bool:
        """Delete one of a collection's snapshots."""

    @abstractmethod
    async def restore_snapshot(self, collection: str, snapshot: str) -> None:
        """Restore a collection from one of its snapshots."""

    @abstractmethod
    async def set_alias(self, alias: str, collection: str) -> None:
        """Point ``alias`` at ``collection``, atomically replacing any previous target."""

    @abstractmethod
    async def alias_target(self, alias: str) -> str | None:
        """Return the collection ``alias`` resolves to, or ``None`` if it is not an alias."""

    @abstractmethod
    async def delete_alias(self, alias: str) -> bool:
        """Remove an alias, reporting whether one existed. The collection is untouched."""

    @abstractmethod
    async def upsert(self, points: list[Point]) -> UpsertResult:
        """Write points, overwriting any that already exist.

        Upserts are idempotent: the same points written twice leave the same index
        state, which is what makes queue replays and crash recovery safe.
        """

    @abstractmethod
    def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        """Yield every point in a collection, in batches."""

    @abstractmethod
    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        """Return the nearest points, with any metadata filter pushed down."""

    @abstractmethod
    async def update(self, updates: list[PointUpdate]) -> None:
        """Merge metadata into existing points without touching their vectors."""

    @abstractmethod
    async def delete(self, selector: PointSelector) -> None:
        """Delete the selected points."""

    @abstractmethod
    async def health(self) -> HealthStatus:
        """Report whether the backend is reachable and answering."""

    @abstractmethod
    async def close(self) -> None:
        """Release connections held by the adapter."""
