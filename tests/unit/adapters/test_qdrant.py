from dataclasses import dataclass, field
from typing import Any, cast

import httpx
import pytest
from grpc import RpcError, StatusCode
from qdrant_client import AsyncQdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from fasterrag.adapters.vectordb.base import (
    CollectionSpec,
    Point,
    PointSelector,
    PointUpdate,
    SearchQuery,
)
from fasterrag.adapters.vectordb.qdrant import (
    POINT_ID_PAYLOAD_KEY,
    QdrantAdapter,
    to_point_id,
    to_qdrant_filter,
)
from fasterrag.config.schema import Settings
from fasterrag.errors import EmbedError, ErrorCode, FasterRagError


@dataclass
class Params:
    vectors: models.VectorParams | dict[str, models.VectorParams]


@dataclass
class Config:
    params: Params


@dataclass
class Info:
    config: Config


@dataclass
class QueryResponse:
    points: list[models.ScoredPoint]


@dataclass
class FakeClient:
    """Records calls the adapter makes, standing in for AsyncQdrantClient."""

    exists: bool = False
    info: Info | None = None
    raises: Exception | None = None
    hits: list[models.ScoredPoint] = field(default_factory=list)
    created: list[dict[str, Any]] = field(default_factory=list)
    upserted: list[dict[str, Any]] = field(default_factory=list)
    searched: list[dict[str, Any]] = field(default_factory=list)
    payloads: list[dict[str, Any]] = field(default_factory=list)
    deleted: list[dict[str, Any]] = field(default_factory=list)
    closed: bool = False

    def _maybe_raise(self) -> None:
        if self.raises is not None:
            raise self.raises

    async def collection_exists(self, collection_name: str) -> bool:
        self._maybe_raise()
        return self.exists

    async def get_collection(self, collection_name: str) -> Info:
        self._maybe_raise()
        return self.info if self.info is not None else collection_info()

    async def create_collection(self, **kwargs: Any) -> None:
        self._maybe_raise()
        self.created.append(kwargs)

    async def upsert(self, **kwargs: Any) -> None:
        self._maybe_raise()
        self.upserted.append(kwargs)

    async def query_points(self, **kwargs: Any) -> QueryResponse:
        self._maybe_raise()
        self.searched.append(kwargs)
        return QueryResponse(points=self.hits)

    async def set_payload(self, **kwargs: Any) -> None:
        self._maybe_raise()
        self.payloads.append(kwargs)

    async def delete(self, **kwargs: Any) -> None:
        self._maybe_raise()
        self.deleted.append(kwargs)

    async def get_collections(self) -> object:
        self._maybe_raise()
        return object()

    async def close(self) -> None:
        self.closed = True


def build(client: FakeClient, settings: Settings | None = None) -> QdrantAdapter:
    adapter = QdrantAdapter(settings or Settings())
    adapter._client = cast("AsyncQdrantClient", client)
    return adapter


def collection_info(size: int = 3, distance: models.Distance = models.Distance.COSINE) -> Info:
    return Info(
        config=Config(params=Params(vectors=models.VectorParams(size=size, distance=distance)))
    )


def unexpected(status: int) -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=status, reason_phrase="", content=b"", headers=httpx.Headers()
    )


def test_point_ids_are_deterministic_and_distinct() -> None:
    assert to_point_id("c_9f2") == to_point_id("c_9f2")
    assert to_point_id("c_9f2") != to_point_id("c_9f3")


def test_scalar_filter_becomes_a_match_condition() -> None:
    translated = to_qdrant_filter({"department": "legal"})
    assert translated is not None
    assert translated.must == [
        models.FieldCondition(key="department", match=models.MatchValue(value="legal"))
    ]


def test_range_filter_is_pushed_down() -> None:
    translated = to_qdrant_filter({"year": {"$gte": 2024}})
    assert translated is not None
    assert translated.must == [models.FieldCondition(key="year", range=models.Range(gte=2024))]


def test_set_membership_filter_is_pushed_down() -> None:
    translated = to_qdrant_filter({"tag": {"$in": ["a", "b"]}})
    assert translated is not None
    assert translated.must == [
        models.FieldCondition(key="tag", match=models.MatchAny(any=["a", "b"]))
    ]


def test_negation_becomes_must_not() -> None:
    translated = to_qdrant_filter({"tenant": {"$ne": "acme"}})
    assert translated is not None
    assert translated.must is None
    assert translated.must_not == [
        models.FieldCondition(key="tenant", match=models.MatchValue(value="acme"))
    ]


def test_multiple_fields_are_combined_with_and() -> None:
    translated = to_qdrant_filter({"department": "legal", "year": {"$lt": 2020}})
    assert translated is not None
    assert isinstance(translated.must, list)
    assert len(translated.must) == 2


def test_no_filter_translates_to_none() -> None:
    assert to_qdrant_filter(None) is None
    assert to_qdrant_filter({}) is None


async def test_create_collection_creates_when_absent() -> None:
    client = FakeClient(exists=False)
    await build(client).create_collection(CollectionSpec(name="default", dimensions=384))

    assert client.created[0]["collection_name"] == "default"
    assert client.created[0]["vectors_config"].size == 384
    assert client.created[0]["vectors_config"].distance == models.Distance.COSINE


async def test_create_collection_is_idempotent_when_compatible() -> None:
    client = FakeClient(exists=True, info=collection_info(size=384))
    await build(client).create_collection(CollectionSpec(name="default", dimensions=384))

    assert client.created == []


async def test_create_collection_conflicts_on_dimension_mismatch() -> None:
    client = FakeClient(exists=True, info=collection_info(size=768))

    with pytest.raises(FasterRagError, match="already exists with 768 dimensions") as caught:
        await build(client).create_collection(CollectionSpec(name="default", dimensions=384))
    assert caught.value.code is ErrorCode.CONFLICT


async def test_create_collection_conflicts_on_distance_mismatch() -> None:
    client = FakeClient(exists=True, info=collection_info(distance=models.Distance.DOT))

    with pytest.raises(FasterRagError, match="distance") as caught:
        await build(client).create_collection(
            CollectionSpec(name="default", dimensions=3, distance="cosine")
        )
    assert caught.value.code is ErrorCode.CONFLICT


async def test_named_vector_collections_are_refused() -> None:
    client = FakeClient(
        exists=True,
        info=Info(
            config=Config(
                params=Params(
                    vectors={"text": models.VectorParams(size=3, distance=models.Distance.COSINE)}
                )
            )
        ),
    )

    with pytest.raises(FasterRagError, match="named vectors"):
        await build(client).create_collection(CollectionSpec(name="default", dimensions=3))


async def test_upsert_maps_ids_and_preserves_the_original() -> None:
    client = FakeClient(info=collection_info(size=3))
    result = await build(client).upsert(
        [
            Point(
                point_id="c_9f2",
                collection="default",
                vector=[0.1, 0.2, 0.3],
                payload={"page": 12},
            )
        ]
    )

    assert result.upserted == 1
    written = client.upserted[0]["points"][0]
    assert written.id == to_point_id("c_9f2")
    assert written.payload[POINT_ID_PAYLOAD_KEY] == "c_9f2"
    assert written.payload["page"] == 12


async def test_upsert_is_idempotent_for_the_same_chunk() -> None:
    client = FakeClient(info=collection_info(size=3))
    adapter = build(client)
    point = Point(point_id="c_9f2", collection="default", vector=[0.1, 0.2, 0.3])

    await adapter.upsert([point])
    await adapter.upsert([point])

    assert client.upserted[0]["points"][0].id == client.upserted[1]["points"][0].id


async def test_upsert_rejects_a_dimension_mismatch() -> None:
    client = FakeClient(info=collection_info(size=384))

    with pytest.raises(EmbedError, match="has 3 dimensions but collection") as caught:
        await build(client).upsert(
            [Point(point_id="c_1", collection="default", vector=[0.1, 0.2, 0.3])]
        )
    assert caught.value.retryable is False
    assert client.upserted == []


async def test_upsert_groups_by_collection() -> None:
    client = FakeClient(info=collection_info(size=3))
    await build(client).upsert(
        [
            Point(point_id="c_1", collection="a", vector=[0.1, 0.2, 0.3]),
            Point(point_id="c_2", collection="b", vector=[0.1, 0.2, 0.3]),
            Point(point_id="c_3", collection="a", vector=[0.1, 0.2, 0.3]),
        ]
    )

    assert len(client.upserted) == 2
    assert {call["collection_name"] for call in client.upserted} == {"a", "b"}


async def test_empty_upsert_touches_no_backend() -> None:
    client = FakeClient()
    assert (await build(client).upsert([])).upserted == 0
    assert client.upserted == []


async def test_search_returns_the_original_point_id() -> None:
    client = FakeClient(
        hits=[
            models.ScoredPoint(
                id=to_point_id("c_9f2"),
                version=1,
                score=0.91,
                payload={POINT_ID_PAYLOAD_KEY: "c_9f2", "page": 12},
            )
        ]
    )

    hits = await build(client).search(SearchQuery(collection="default", vector=[0.1, 0.2, 0.3]))

    assert hits[0].point_id == "c_9f2"
    assert hits[0].score == 0.91
    assert hits[0].payload == {"page": 12}


async def test_search_pushes_the_filter_down() -> None:
    client = FakeClient()
    await build(client).search(
        SearchQuery(collection="default", vector=[0.1], filters={"department": "legal"})
    )

    assert client.searched[0]["query_filter"] is not None


async def test_search_without_payload_still_resolves_ids() -> None:
    client = FakeClient(
        hits=[
            models.ScoredPoint(
                id=to_point_id("c_1"), version=1, score=0.5, payload={POINT_ID_PAYLOAD_KEY: "c_1"}
            )
        ]
    )

    hits = await build(client).search(
        SearchQuery(collection="default", vector=[0.1], with_payload=False)
    )

    assert client.searched[0]["with_payload"] == [POINT_ID_PAYLOAD_KEY]
    assert hits[0].point_id == "c_1"
    assert hits[0].payload == {}


async def test_search_falls_back_to_the_backend_id() -> None:
    client = FakeClient(
        hits=[models.ScoredPoint(id=to_point_id("c_1"), version=1, score=0.5, payload={})]
    )

    hits = await build(client).search(SearchQuery(collection="default", vector=[0.1]))

    assert hits[0].point_id == to_point_id("c_1")


async def test_update_batches_identical_payloads() -> None:
    client = FakeClient()
    await build(client).update(
        [
            PointUpdate(point_id="c_1", collection="default", payload={"reviewed": True}),
            PointUpdate(point_id="c_2", collection="default", payload={"reviewed": True}),
            PointUpdate(point_id="c_3", collection="default", payload={"reviewed": False}),
        ]
    )

    assert len(client.payloads) == 2
    batched = next(call for call in client.payloads if call["payload"] == {"reviewed": True})
    assert len(batched["points"]) == 2


async def test_empty_update_touches_no_backend() -> None:
    client = FakeClient()
    await build(client).update([])
    assert client.payloads == []


async def test_delete_by_ids_uses_mapped_identifiers() -> None:
    client = FakeClient()
    await build(client).delete(PointSelector(collection="default", point_ids=["c_1"]))

    selector = client.deleted[0]["points_selector"]
    assert isinstance(selector, models.PointIdsList)
    assert selector.points == [to_point_id("c_1")]


async def test_delete_by_filter_uses_a_filter_selector() -> None:
    client = FakeClient()
    await build(client).delete(PointSelector(collection="default", filters={"tenant": "acme"}))

    assert isinstance(client.deleted[0]["points_selector"], models.FilterSelector)


async def test_delete_refuses_an_empty_filter() -> None:
    client = FakeClient()
    with pytest.raises(FasterRagError, match="refusing to delete a whole collection"):
        await build(client).delete(PointSelector(collection="default", filters={}))
    assert client.deleted == []


async def test_authentication_failure_is_not_retryable_and_hides_the_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_API_KEY", "super-secret-key")
    client = FakeClient(raises=unexpected(401))

    with pytest.raises(FasterRagError) as caught:
        await build(client).search(SearchQuery(collection="default", vector=[0.1]))

    assert caught.value.retryable is False
    assert "QDRANT_API_KEY" in caught.value.detail
    assert "super-secret-key" not in str(caught.value)


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (403, ErrorCode.EMBED_PROVIDER_ERROR, False),
        (404, ErrorCode.NOT_FOUND, False),
        (409, ErrorCode.CONFLICT, False),
        (429, ErrorCode.EMBED_PROVIDER_ERROR, False),
        (500, ErrorCode.EMBED_PROVIDER_ERROR, True),
        (503, ErrorCode.EMBED_PROVIDER_ERROR, True),
    ],
)
async def test_http_failures_map_onto_the_taxonomy(
    status: int, code: ErrorCode, retryable: bool
) -> None:
    client = FakeClient(raises=unexpected(status))

    with pytest.raises(FasterRagError) as caught:
        await build(client).search(SearchQuery(collection="default", vector=[0.1]))

    assert caught.value.code is code
    assert caught.value.retryable is retryable


class FakeRpcError(RpcError):
    """A gRPC failure carrying a status code, as grpc.aio raises."""

    def __init__(self, status: StatusCode) -> None:
        self._status = status

    def code(self) -> StatusCode:
        return self._status


@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [
        (StatusCode.UNAUTHENTICATED, ErrorCode.EMBED_PROVIDER_ERROR, False),
        (StatusCode.PERMISSION_DENIED, ErrorCode.EMBED_PROVIDER_ERROR, False),
        (StatusCode.INVALID_ARGUMENT, ErrorCode.EMBED_PROVIDER_ERROR, False),
        (StatusCode.NOT_FOUND, ErrorCode.NOT_FOUND, False),
        (StatusCode.ALREADY_EXISTS, ErrorCode.CONFLICT, False),
        (StatusCode.UNAVAILABLE, ErrorCode.EMBED_PROVIDER_ERROR, True),
        (StatusCode.DEADLINE_EXCEEDED, ErrorCode.EMBED_PROVIDER_ERROR, True),
    ],
)
async def test_grpc_failures_map_onto_the_taxonomy(
    status: StatusCode, code: ErrorCode, retryable: bool
) -> None:
    client = FakeClient(raises=FakeRpcError(status))

    with pytest.raises(FasterRagError) as caught:
        await build(client).search(SearchQuery(collection="default", vector=[0.1]))

    assert caught.value.code is code
    assert caught.value.retryable is retryable


async def test_a_grpc_auth_rejection_names_the_env_var_and_is_not_retried(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_API_KEY", "super-secret-key")
    client = FakeClient(raises=FakeRpcError(StatusCode.UNAUTHENTICATED))

    with pytest.raises(FasterRagError) as caught:
        await build(client).search(SearchQuery(collection="default", vector=[0.1]))

    assert caught.value.retryable is False
    assert "QDRANT_API_KEY" in caught.value.detail
    assert "super-secret-key" not in str(caught.value)


async def test_transport_failures_are_retryable() -> None:
    client = FakeClient(raises=ResponseHandlingException(OSError("connection refused")))

    with pytest.raises(FasterRagError) as caught:
        await build(client).search(SearchQuery(collection="default", vector=[0.1]))

    assert caught.value.retryable is True
    assert "unreachable" in caught.value.detail


async def test_no_vendor_exception_escapes_the_adapter() -> None:
    client = FakeClient(raises=unexpected(500))

    with pytest.raises(FasterRagError):
        await build(client).search(SearchQuery(collection="default", vector=[0.1]))


async def test_health_reports_latency_when_reachable() -> None:
    status = await build(FakeClient()).health()

    assert status.healthy is True
    assert status.latency_ms is not None


async def test_health_reports_failure_instead_of_raising() -> None:
    status = await build(FakeClient(raises=unexpected(503))).health()

    assert status.healthy is False
    assert status.detail is not None


def test_transport_settings_are_passed_explicitly(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def recording(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("fasterrag.adapters.vectordb.qdrant.AsyncQdrantClient", recording)
    monkeypatch.setenv("QDRANT_API_KEY", "a-key")

    assert QdrantAdapter(Settings()).client is not None

    assert captured["https"] is False
    assert captured["prefer_grpc"] is False
    assert captured["port"] == 6333
    assert captured["grpc_port"] == 6334


def test_tls_is_used_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def recording(**kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("fasterrag.adapters.vectordb.qdrant.AsyncQdrantClient", recording)
    settings = Settings.model_validate({"vector_db": {"https": True}})

    assert QdrantAdapter(settings).client is not None

    assert captured["https"] is True


def test_both_ports_must_be_reachable() -> None:
    endpoints = build(FakeClient()).describe_endpoints()

    assert [port for _, port in endpoints] == [6333, 6334]


async def test_close_releases_the_client() -> None:
    client = FakeClient()
    adapter = build(client)

    await adapter.close()

    assert client.closed is True
    assert adapter._client is None
