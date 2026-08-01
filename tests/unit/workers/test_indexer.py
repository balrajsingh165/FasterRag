import asyncio
from typing import Any

import pytest

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
from fasterrag.core.chunking.models import TextChunk
from fasterrag.workers.indexer import Indexer, chunk_payload
from fasterrag.workers.queues import ChunkPayload, EmbeddedBatch


class RecordingAdapter(VectorDBAdapter):
    """Captures what the indexer writes."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.specs: list[CollectionSpec] = []
        self.points: list[Point] = []
        self.aliases: dict[str, str] = {}

    async def create_collection(self, spec: CollectionSpec) -> None:
        self.specs.append(spec)

    async def list_collections(self) -> list[CollectionInfo]:
        return [CollectionInfo(name=spec.name, vectors=len(self.points)) for spec in self.specs]

    async def drop_collection(self, name: str) -> bool:
        self.specs = [spec for spec in self.specs if spec.name != name]
        return True

    async def set_alias(self, alias: str, collection: str) -> None:
        self.aliases[alias] = collection

    async def alias_target(self, alias: str) -> str | None:
        return self.aliases.get(alias)

    async def delete_alias(self, alias: str) -> bool:
        return self.aliases.pop(alias, None) is not None

    async def upsert(self, points: list[Point]) -> UpsertResult:
        self.points.extend(points)
        return UpsertResult(upserted=len(points))

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        return []

    async def update(self, updates: list[PointUpdate]) -> None:
        return None

    async def delete(self, selector: PointSelector) -> None:
        return None

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        return None


def settings(**overrides: Any) -> Settings:
    return Settings.model_validate(overrides)


def payload(index: int = 0, **extra: Any) -> ChunkPayload:
    return ChunkPayload(
        chunk_id=f"c_{index}",
        document_id="d_1",
        source="contracts/vendor.pdf",
        content_hash="a" * 64,
        chunk=TextChunk(
            text="Either party may terminate with thirty days notice.",
            start=10,
            end=61,
            chunk_index=index,
            token_count=9,
            strategy="recursive",
            page=12,
            section="3. Termination",
        ),
        metadata=extra.pop("metadata", {"department": "legal"}),
        tenant=extra.pop("tenant", None),
    )


def batch(count: int = 1, **extra: Any) -> EmbeddedBatch:
    chunks = [payload(index, **extra) for index in range(count)]
    return EmbeddedBatch(
        chunks=chunks,
        vectors=[[0.1, 0.2, 0.3] for _ in chunks],
        model="bge-small",
        model_version="bge-small-v1",
    )


async def test_the_collection_is_created_with_a_sparse_index_when_hybrid() -> None:
    adapter = RecordingAdapter(settings())
    await Indexer(settings(), adapter).ensure_collection(384)

    assert adapter.specs[0].sparse is True
    assert adapter.specs[0].dimensions == 384


async def test_a_dense_only_collection_is_created_when_hybrid_is_off() -> None:
    configured = settings(retrieval={"hybrid": False})
    adapter = RecordingAdapter(configured)

    await Indexer(configured, adapter).ensure_collection(384)

    assert adapter.specs[0].sparse is False


async def test_collection_settings_are_passed_through() -> None:
    configured = settings(
        vector_db={"collection": {"distance": "dot", "shard_number": 4, "replication_factor": 2}}
    )
    adapter = RecordingAdapter(configured)

    await Indexer(configured, adapter).ensure_collection(8)
    spec = adapter.specs[0]

    assert spec.distance == "dot"
    assert spec.shard_number == 4
    assert spec.replication_factor == 2


async def test_each_chunk_is_written_with_its_own_id() -> None:
    adapter = RecordingAdapter(settings())
    indexer = Indexer(settings(), adapter)

    await indexer.write(batch(count=3))

    assert [point.point_id for point in adapter.points] == ["c_0", "c_1", "c_2"]
    assert indexer.written == 3


async def test_the_payload_carries_everything_a_citation_needs() -> None:
    adapter = RecordingAdapter(settings())
    await Indexer(settings(), adapter).write(batch())

    stored = adapter.points[0].payload
    assert stored["document_id"] == "d_1"
    assert stored["source_uri"] == "contracts/vendor.pdf"
    assert stored["span"] == {"start": 10, "end": 61}
    assert stored["page"] == 12
    assert stored["section"] == "3. Termination"


async def test_the_payload_records_the_model_drift_detection_compares() -> None:
    adapter = RecordingAdapter(settings())
    await Indexer(settings(), adapter).write(batch())

    stored = adapter.points[0].payload
    assert stored["embedding_model"] == "bge-small"
    assert stored["embedding_model_version"] == "bge-small-v1"
    assert stored["chunker_strategy"] == "recursive"


async def test_user_metadata_is_stored_for_filtering() -> None:
    adapter = RecordingAdapter(settings())
    await Indexer(settings(), adapter).write(batch())

    assert adapter.points[0].payload["department"] == "legal"


async def test_a_tenant_is_stored_when_present() -> None:
    adapter = RecordingAdapter(settings())
    await Indexer(settings(), adapter).write(batch(tenant="acme"))

    assert adapter.points[0].payload["tenant"] == "acme"


async def test_no_tenant_key_is_written_for_single_tenant_deployments() -> None:
    adapter = RecordingAdapter(settings())
    await Indexer(settings(), adapter).write(batch())

    assert "tenant" not in adapter.points[0].payload


async def test_a_sparse_vector_is_produced_for_the_keyword_leg() -> None:
    adapter = RecordingAdapter(settings())
    await Indexer(settings(), adapter).write(batch())

    sparse = adapter.points[0].sparse
    assert sparse is not None
    assert len(sparse) > 0


async def test_no_sparse_vector_is_produced_when_hybrid_is_off() -> None:
    configured = settings(retrieval={"hybrid": False})
    adapter = RecordingAdapter(configured)

    await Indexer(configured, adapter).write(batch())

    assert adapter.points[0].sparse is None


async def test_vectors_stay_paired_with_their_chunks() -> None:
    adapter = RecordingAdapter(settings())
    chunks = [payload(index) for index in range(3)]
    written = EmbeddedBatch(
        chunks=chunks,
        vectors=[[float(index)] * 3 for index in range(3)],
        model="m",
        model_version="v",
    )

    await Indexer(settings(), adapter).write(written)

    for index, point in enumerate(adapter.points):
        assert list(point.vector) == [float(index)] * 3
        assert point.point_id == f"c_{index}"


async def test_concurrent_writers_create_the_collection_once() -> None:
    adapter = RecordingAdapter(settings())
    indexer = Indexer(settings(), adapter)

    await asyncio.gather(*(indexer.write(batch()) for _ in range(5)))

    assert len(adapter.specs) == 1


async def test_the_collection_is_created_from_the_first_batch_vectors() -> None:
    adapter = RecordingAdapter(settings())
    written = EmbeddedBatch(chunks=[payload()], vectors=[[0.0] * 12], model="m", model_version="v")

    await Indexer(settings(), adapter).write(written)

    assert adapter.specs[0].dimensions == 12


async def test_an_empty_batch_writes_nothing() -> None:
    adapter = RecordingAdapter(settings())
    await Indexer(settings(), adapter).write(
        EmbeddedBatch(chunks=[], vectors=[], model="m", model_version="v")
    )

    assert adapter.points == []


async def test_writing_the_same_batch_twice_targets_the_same_points() -> None:
    adapter = RecordingAdapter(settings())
    indexer = Indexer(settings(), adapter)

    await indexer.write(batch(count=2))
    await indexer.write(batch(count=2))

    assert [point.point_id for point in adapter.points] == ["c_0", "c_1", "c_0", "c_1"]


async def test_the_configured_collection_is_used_by_default() -> None:
    configured = settings(vector_db={"collection": {"default_name": "contracts"}})
    adapter = RecordingAdapter(configured)

    await Indexer(configured, adapter).write(batch())

    assert adapter.points[0].collection == "contracts"


async def test_an_explicit_collection_overrides_the_default() -> None:
    adapter = RecordingAdapter(settings())

    await Indexer(settings(), adapter, collection="scratch").write(batch())

    assert adapter.points[0].collection == "scratch"


def test_the_payload_never_invents_fields() -> None:
    stored = chunk_payload(payload(), model="m", model_version="v")

    known = {
        "document_id",
        "source_uri",
        "content_hash",
        "text",
        "span",
        "chunk_index",
        "token_count",
        "chunker_strategy",
        "embedding_model",
        "embedding_model_version",
        "page",
        "section",
        "tenant",
        "department",
    }
    assert set(stored) <= known


@pytest.mark.parametrize("field", ["page", "section"])
def test_absent_structure_is_omitted_rather_than_stored_as_null(field: str) -> None:
    bare = ChunkPayload(
        chunk_id="c_0",
        document_id="d_1",
        source="notes.txt",
        content_hash="b" * 64,
        chunk=TextChunk(
            text="body", start=0, end=4, chunk_index=0, token_count=1, strategy="fixed"
        ),
    )

    assert field not in chunk_payload(bare, model="m", model_version="v")
