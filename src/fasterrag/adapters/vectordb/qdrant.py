"""Qdrant reference adapter.

The reference implementation of :class:`~fasterrag.adapters.vectordb.base.VectorDBAdapter`
(``docs/adr/ADR-0001``). All three deployment modes connect through this one class:
``docker`` talks to the container the provisioner manages, and ``external`` talks to an
instance the operator runs, whether on localhost or another machine. Only who runs the
server differs; the client path is identical, which is why remote mode needs no special
code — just a reachable host.

Both ports matter. The client defaults are 6333 (REST) and 6334 (gRPC), and
``prefer_grpc`` defaults to false; exposing only 6333 breaks clients that attempt gRPC
(``docs/references.md`` R5), so ``fasterrag doctor`` checks each port separately.

Qdrant point ids must be unsigned integers or UUIDs, while fasterRag chunk ids are
opaque strings such as ``c_9f2``. The adapter derives a UUID from the chunk id
deterministically and keeps the original in the payload, so idempotent upserts and
replay safety survive the translation.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any, Final, cast

from grpc import RpcError, StatusCode
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from fasterrag.adapters.vectordb.base import (
    CollectionSpec,
    Distance,
    Filter,
    HealthStatus,
    Point,
    PointSelector,
    PointUpdate,
    ScoredPoint,
    SearchQuery,
    UpsertResult,
    VectorDBAdapter,
    validate_filter,
)
from fasterrag.config.schema import Settings
from fasterrag.errors import EmbedError, ErrorCode, FasterRagError, ProviderError
from fasterrag.observability.logging import get_logger

__all__ = ["POINT_ID_PAYLOAD_KEY", "QdrantAdapter"]

POINT_ID_PAYLOAD_KEY: Final = "point_id"

# CRITICAL: this namespace is what makes a chunk id map to the same Qdrant point id
# forever. Changing it orphans every previously indexed vector and turns idempotent
# upserts into silent duplicates.
_POINT_ID_NAMESPACE: Final = uuid.uuid5(uuid.NAMESPACE_URL, "https://fasterrag.dev/point-id")

_DISTANCES: Final[dict[Distance, models.Distance]] = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}

_RANGE_OPERATORS: Final[frozenset[str]] = frozenset({"$gt", "$gte", "$lt", "$lte"})

_AUTH_STATUSES: Final[frozenset[int]] = frozenset({401, 403})
_SERVER_ERROR_THRESHOLD: Final = 500

_GRPC_AUTH_CODES: Final[frozenset[StatusCode]] = frozenset(
    {StatusCode.UNAUTHENTICATED, StatusCode.PERMISSION_DENIED}
)
_GRPC_NON_RETRYABLE: Final[frozenset[StatusCode]] = frozenset(
    {StatusCode.INVALID_ARGUMENT, StatusCode.FAILED_PRECONDITION, StatusCode.OUT_OF_RANGE}
)

_VENDOR_ERRORS: Final = (UnexpectedResponse, ResponseHandlingException, RpcError, OSError)

_logger = get_logger(__name__)


def to_point_id(point_id: str) -> str:
    """Return the deterministic Qdrant UUID for a fasterRag point id."""
    return str(uuid.uuid5(_POINT_ID_NAMESPACE, point_id))


def _condition(key: str, operator: str, value: Any) -> models.FieldCondition:
    """Translate one supported filter operator into a Qdrant field condition."""
    if operator in _RANGE_OPERATORS:
        bound = operator.removeprefix("$")
        return models.FieldCondition(key=key, range=models.Range(**{bound: value}))
    if operator == "$in":
        return models.FieldCondition(key=key, match=models.MatchAny(any=value))
    if operator == "$nin":
        return models.FieldCondition(key=key, match=models.MatchExcept(**{"except": value}))
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def to_qdrant_filter(filters: Filter | None) -> models.Filter | None:
    """Translate a vendor-neutral filter expression into a Qdrant filter.

    Args:
        filters: The validated expression, or ``None`` for no filtering.

    Returns:
        The pushed-down Qdrant filter, or ``None``.

    Raises:
        FasterRagError: With ``VALIDATION_FAILED`` if the expression is unsupported.
    """
    validate_filter(filters)
    if not filters:
        return None

    must: list[models.Condition] = []
    must_not: list[models.Condition] = []

    for key, condition in filters.items():
        if not isinstance(condition, Mapping):
            must.append(_condition(key, "$eq", condition))
            continue

        operator = next(iter(condition))
        value = condition[operator]
        if operator == "$ne":
            must_not.append(_condition(key, "$eq", value))
        else:
            must.append(_condition(key, operator, value))

    return models.Filter(must=must or None, must_not=must_not or None)


class QdrantAdapter(VectorDBAdapter):
    """Qdrant implementation of the vector database contract."""

    def __init__(self, settings: Settings) -> None:
        """Build the adapter. No connection is opened until an operation runs.

        Args:
            settings: Validated configuration supplying the connection details, the
                name of the environment variable holding the API key, and the shared
                vector-database timeout.
        """
        self._settings = settings.vector_db
        self._api_key_env = settings.vector_db.api_key_env
        self._timeout_seconds = max(1, round(settings.reliability.timeouts.vector_db_ms / 1000))
        self._client: AsyncQdrantClient | None = None
        self._dimensions: dict[str, int] = {}

    @property
    def client(self) -> AsyncQdrantClient:
        """Return the lazily-built Qdrant client."""
        if self._client is None:
            api_key = os.environ.get(self._api_key_env) if self._api_key_env else None
            self._client = AsyncQdrantClient(
                host=self._settings.host,
                port=self._settings.port,
                grpc_port=self._settings.grpc_port,
                prefer_grpc=self._settings.prefer_grpc,
                # CRITICAL: pass https explicitly. The Qdrant client switches to TLS on its
                # own as soon as an api_key is supplied, which fails with a TLS handshake
                # error against the plain-HTTP listener a container serves by default.
                https=self._settings.https,
                api_key=api_key,
                timeout=self._timeout_seconds,
            )
        return self._client

    def _auth_error(self, operation: str, transport: str) -> ProviderError:
        """Build the non-retryable error for a rejected credential."""
        named = self._api_key_env or "vector_db.api_key_env"
        return ProviderError(
            f"qdrant rejected the credentials during {operation} over {transport}; "
            f"check the key in the {named} environment variable and the server's "
            "QDRANT__SERVICE__API_KEY",
            code=ErrorCode.EMBED_PROVIDER_ERROR,
            retryable=False,
        )

    def _translate_grpc(self, exc: RpcError, operation: str) -> FasterRagError:
        """Map a gRPC failure onto the taxonomy.

        gRPC reports status codes rather than HTTP statuses, so without this an
        authentication rejection over gRPC would be classified as a transport blip and
        retried until the breaker opened.
        """
        code_getter = getattr(exc, "code", None)
        status = code_getter() if callable(code_getter) else None

        if status in _GRPC_AUTH_CODES:
            return self._auth_error(operation, "grpc")
        if status is StatusCode.NOT_FOUND:
            return FasterRagError(
                f"qdrant reported a missing resource during {operation}",
                code=ErrorCode.NOT_FOUND,
            )
        if status is StatusCode.ALREADY_EXISTS:
            return FasterRagError(
                f"qdrant reported a conflicting state during {operation}",
                code=ErrorCode.CONFLICT,
            )

        name = status.name if isinstance(status, StatusCode) else "UNKNOWN"
        return ProviderError(
            f"qdrant failed during {operation} over grpc ({name})",
            code=ErrorCode.EMBED_PROVIDER_ERROR,
            retryable=status not in _GRPC_NON_RETRYABLE,
        )

    def _translate(self, exc: BaseException, operation: str) -> FasterRagError:
        """Map a Qdrant client failure onto the typed error taxonomy.

        Authentication failures are never retried and name the environment variable
        rather than the key, so a credential cannot reach a log line.
        """
        if isinstance(exc, RpcError):
            return self._translate_grpc(exc, operation)

        if isinstance(exc, UnexpectedResponse) and exc.status_code is not None:
            status = exc.status_code
            if status in _AUTH_STATUSES:
                return self._auth_error(operation, f"http {status}")
            if status == 404:
                return FasterRagError(
                    f"qdrant reported a missing resource during {operation}",
                    code=ErrorCode.NOT_FOUND,
                )
            if status == 409:
                return FasterRagError(
                    f"qdrant reported a conflicting state during {operation}",
                    code=ErrorCode.CONFLICT,
                )
            return ProviderError(
                f"qdrant returned HTTP {status} during {operation}",
                code=ErrorCode.EMBED_PROVIDER_ERROR,
                retryable=status >= _SERVER_ERROR_THRESHOLD,
            )

        return ProviderError(
            f"qdrant was unreachable during {operation}: {type(exc).__name__}",
            code=ErrorCode.EMBED_PROVIDER_ERROR,
            retryable=True,
        )

    @asynccontextmanager
    async def _mapped_errors(self, operation: str) -> AsyncIterator[None]:
        """Convert vendor exceptions into typed errors so no vendor type escapes."""
        try:
            yield
        except _VENDOR_ERRORS as exc:
            raise self._translate(exc, operation) from exc

    async def create_collection(self, spec: CollectionSpec) -> None:
        """Create the collection, or verify an existing one is compatible."""
        async with self._mapped_errors("create_collection"):
            if await self.client.collection_exists(spec.name):
                await self._require_compatible(spec)
                return

            await self.client.create_collection(
                collection_name=spec.name,
                vectors_config=models.VectorParams(
                    size=spec.dimensions,
                    distance=_DISTANCES[spec.distance],
                ),
                shard_number=spec.shard_number,
                replication_factor=spec.replication_factor,
            )
        self._dimensions[spec.name] = spec.dimensions

    async def _require_compatible(self, spec: CollectionSpec) -> None:
        """Reject an existing collection whose vectors cannot hold ``spec``."""
        info = await self.client.get_collection(spec.name)
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            raise FasterRagError(
                f"collection {spec.name!r} uses named vectors, which fasterRag does not manage",
                code=ErrorCode.CONFLICT,
            )
        if vectors.size != spec.dimensions:
            raise FasterRagError(
                f"collection {spec.name!r} already exists with {vectors.size} dimensions, "
                f"but the configuration expects {spec.dimensions}; re-embed through a "
                "blue/green reindex rather than writing mixed vectors",
                code=ErrorCode.CONFLICT,
            )
        expected = _DISTANCES[spec.distance]
        if vectors.distance != expected:
            raise FasterRagError(
                f"collection {spec.name!r} already exists with distance "
                f"{vectors.distance}, but the configuration expects {expected}",
                code=ErrorCode.CONFLICT,
            )
        self._dimensions[spec.name] = vectors.size

    async def _collection_dimensions(self, collection: str) -> int | None:
        """Return the collection's vector size, cached after the first lookup."""
        cached = self._dimensions.get(collection)
        if cached is not None:
            return cached

        info = await self.client.get_collection(collection)
        vectors = info.config.params.vectors
        if not isinstance(vectors, models.VectorParams):
            return None

        self._dimensions[collection] = vectors.size
        return vectors.size

    async def upsert(self, points: list[Point]) -> UpsertResult:
        """Write points, overwriting any that already exist."""
        if not points:
            return UpsertResult(upserted=0)

        grouped: dict[str, list[Point]] = {}
        for point in points:
            grouped.setdefault(point.collection, []).append(point)

        async with self._mapped_errors("upsert"):
            for collection, batch in grouped.items():
                await self._require_matching_dimensions(collection, batch)
                await self.client.upsert(
                    collection_name=collection,
                    points=[
                        models.PointStruct(
                            id=to_point_id(point.point_id),
                            vector=list(point.vector),
                            payload={
                                **dict(point.payload),
                                POINT_ID_PAYLOAD_KEY: point.point_id,
                            },
                        )
                        for point in batch
                    ],
                    wait=True,
                )

        return UpsertResult(upserted=len(points))

    async def _require_matching_dimensions(self, collection: str, batch: list[Point]) -> None:
        """Reject a batch whose vectors do not match the collection's vector size."""
        expected = await self._collection_dimensions(collection)
        if expected is None:
            return

        for point in batch:
            actual = len(point.vector)
            if actual != expected:
                raise EmbedError(
                    f"point {point.point_id!r} has {actual} dimensions but collection "
                    f"{collection!r} stores {expected}; the configured embedding model "
                    "does not match the one this collection was built with",
                    retryable=False,
                )

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        """Return the nearest points, pushing any metadata filter into Qdrant."""
        query_filter = to_qdrant_filter(query.filters)
        payload_selector: bool | list[str] = True if query.with_payload else [POINT_ID_PAYLOAD_KEY]

        async with self._mapped_errors("search"):
            response = await self.client.query_points(
                collection_name=query.collection,
                query=list(query.vector),
                limit=query.limit,
                query_filter=query_filter,
                with_payload=payload_selector,
                with_vectors=query.with_vectors,
            )

        return [self._to_scored_point(point, query.with_payload) for point in response.points]

    @staticmethod
    def _to_scored_point(point: models.ScoredPoint, with_payload: bool) -> ScoredPoint:
        """Convert a Qdrant hit into the vendor-neutral result type."""
        payload = dict(point.payload or {})
        point_id = payload.pop(POINT_ID_PAYLOAD_KEY, None)

        raw = point.vector
        vector: Sequence[float] | None = None
        if isinstance(raw, list) and all(isinstance(value, int | float) for value in raw):
            vector = cast("list[float]", raw)

        return ScoredPoint(
            point_id=str(point_id) if point_id is not None else str(point.id),
            score=point.score,
            payload=payload if with_payload else {},
            vector=vector,
        )

    async def update(self, updates: list[PointUpdate]) -> None:
        """Merge metadata into existing points without touching their vectors."""
        if not updates:
            return

        grouped: dict[tuple[str, str], list[models.ExtendedPointId]] = {}
        payloads: dict[tuple[str, str], Mapping[str, Any]] = {}
        for update in updates:
            key = (update.collection, json.dumps(dict(update.payload), sort_keys=True, default=str))
            grouped.setdefault(key, []).append(to_point_id(update.point_id))
            payloads[key] = update.payload

        async with self._mapped_errors("update"):
            for key, point_ids in grouped.items():
                collection, _ = key
                await self.client.set_payload(
                    collection_name=collection,
                    payload=dict(payloads[key]),
                    points=point_ids,
                    wait=True,
                )

    async def delete(self, selector: PointSelector) -> None:
        """Delete the selected points."""
        points_selector: models.PointsSelector
        if selector.point_ids is not None:
            points_selector = models.PointIdsList(
                points=[to_point_id(point_id) for point_id in selector.point_ids]
            )
        else:
            query_filter = to_qdrant_filter(selector.filters)
            if query_filter is None:
                raise FasterRagError(
                    "a delete filter must select something; refusing to delete a whole "
                    "collection through an empty filter",
                    code=ErrorCode.VALIDATION_FAILED,
                )
            points_selector = models.FilterSelector(filter=query_filter)

        async with self._mapped_errors("delete"):
            await self.client.delete(
                collection_name=selector.collection,
                points_selector=points_selector,
                wait=True,
            )

    async def health(self) -> HealthStatus:
        """Report reachability without raising, so probes can render the failure.

        ``/readyz`` and ``fasterrag doctor`` both call this to describe a sick backend
        rather than fail on it, so the error is logged here and returned as state.
        """
        started = time.perf_counter()
        try:
            async with self._mapped_errors("health"):
                await self.client.get_collections()
        except FasterRagError as exc:
            _logger.warning(
                "vector database health check failed",
                extra={"code": exc.code.value, "trace_id": exc.trace_id},
            )
            return HealthStatus(healthy=False, detail=exc.detail)

        elapsed_ms = (time.perf_counter() - started) * 1000
        return HealthStatus(healthy=True, latency_ms=round(elapsed_ms, 3))

    async def close(self) -> None:
        """Close the client's connections."""
        if self._client is not None:
            async with self._mapped_errors("close"):
                await self._client.close()
            self._client = None

    def describe_endpoints(self) -> Sequence[tuple[str, int]]:
        """Return the ``(host, port)`` pairs that must be reachable for this adapter.

        Both the REST and gRPC ports are listed because a deployment exposing only 6333
        fails any client that attempts gRPC (``docs/failure-modes.md`` row 15).
        """
        return (
            (self._settings.host, self._settings.port),
            (self._settings.host, self._settings.grpc_port),
        )
