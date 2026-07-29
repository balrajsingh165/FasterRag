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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError

__all__ = [
    "COMPARISON_OPERATORS",
    "CollectionSpec",
    "Distance",
    "Filter",
    "HealthStatus",
    "Point",
    "PointSelector",
    "PointUpdate",
    "ScoredPoint",
    "SearchQuery",
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
class CollectionSpec:
    """The shape of a collection to create."""

    name: str
    dimensions: int
    distance: Distance = "cosine"
    shard_number: int = 1
    replication_factor: int = 1


@dataclass(frozen=True, slots=True)
class Point:
    """One indexed vector with its metadata payload.

    ``point_id`` is the deterministic chunk id from ``docs/data-model.md``. Adapters map
    it onto whatever identifier the backend accepts, deterministically, so re-upserting
    the same chunk overwrites rather than duplicates.
    """

    point_id: str
    collection: str
    vector: Sequence[float]
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UpsertResult:
    """How many points an upsert wrote."""

    upserted: int


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """A dense similarity search with optional metadata filtering."""

    collection: str
    vector: Sequence[float]
    limit: int = 10
    filters: Filter | None = None
    with_payload: bool = True
    with_vectors: bool = False


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
    """Reject filter expressions outside the supported grammar.

    Args:
        filters: The expression to check; ``None`` is valid and means no filtering.

    Raises:
        FasterRagError: With ``VALIDATION_FAILED`` when a field maps to an unsupported
            operator, to more than one operator, or to a set operator without a list.
    """
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
    """Vendor-neutral vector database contract.

    Every implementation — built-in or registered by a third party through the
    ``fasterrag.vectordb`` entry point — must pass the shared adapter contract suite,
    which is what makes "any vector DB" a tested promise rather than a hope
    (``docs/testing-strategy.md`` §1.5).

    Implementations translate vendor errors into the typed taxonomy: transport and
    availability failures become retryable errors, authentication failures do not, and
    no vendor exception type ever escapes the adapter.
    """

    def __init__(self, settings: Settings) -> None:
        """Build the adapter from validated configuration.

        This one-argument constructor is the registration contract: the factory builds
        every provider, built-in or plugin, by calling it with the whole ``Settings``
        object, so an adapter can read both its own section and the shared reliability
        timeouts. Opening connections here is discouraged — construction must stay cheap
        and side-effect-free.

        Args:
            settings: The validated configuration.
        """
        self.settings = settings

    @abstractmethod
    async def create_collection(self, spec: CollectionSpec) -> None:
        """Create a collection, or return quietly if it already matches ``spec``.

        Raises:
            FasterRagError: With ``CONFLICT`` if a collection of that name exists with
                incompatible dimensions or distance.
        """

    @abstractmethod
    async def upsert(self, points: list[Point]) -> UpsertResult:
        """Write points, overwriting any that already exist.

        Upserts are idempotent: the same points written twice leave the same index
        state, which is what makes queue replays and crash recovery safe.
        """

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
