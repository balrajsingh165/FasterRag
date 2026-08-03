from dataclasses import dataclass, field
from typing import Any, cast

import pytest
from qdrant_client import AsyncQdrantClient, models

from fasterrag.adapters.vectordb.qdrant import POINT_ID_PAYLOAD_KEY, QdrantAdapter
from fasterrag.config.schema import Settings
from fasterrag.errors import FasterRagError


def record(point_id: str, *, vector: Any = None, text: str = "body") -> models.Record:
    return models.Record(
        id=point_id,
        payload={POINT_ID_PAYLOAD_KEY: point_id, "text": text},
        vector=vector,
    )


@dataclass
class ScrollingClient:
    """Serves records in pages, exactly as Qdrant's cursor does."""

    pages: list[tuple[list[models.Record], Any]] = field(default_factory=list)
    named: bool = False
    calls: list[dict[str, Any]] = field(default_factory=list)
    raises: Exception | None = None

    async def scroll(self, **kwargs: Any) -> tuple[list[models.Record], Any]:
        if self.raises is not None:
            raise self.raises
        self.calls.append(kwargs)
        return self.pages[len(self.calls) - 1]

    async def get_collection(self, collection_name: str) -> Any:
        params = models.VectorParams(size=3, distance=models.Distance.COSINE)
        config = {"dense": params} if self.named else params
        return models.CollectionInfo(
            status=models.CollectionStatus.GREEN,
            optimizer_status=models.OptimizersStatusOneOf.OK,
            indexed_vectors_count=1,
            points_count=1,
            segments_count=1,
            payload_schema={},
            config=models.CollectionConfig(
                params=models.CollectionParams(vectors=config),
                hnsw_config=models.HnswConfig(m=16, ef_construct=100, full_scan_threshold=1),
                optimizer_config=models.OptimizersConfig(
                    deleted_threshold=0.2,
                    vacuum_min_vector_number=1,
                    default_segment_number=1,
                    flush_interval_sec=1,
                    max_optimization_threads=1,
                ),
                wal_config=models.WalConfig(wal_capacity_mb=1, wal_segments_ahead=0),
            ),
        )


def build(client: ScrollingClient) -> QdrantAdapter:
    adapter = QdrantAdapter(Settings())
    adapter._client = cast("AsyncQdrantClient", client)
    return adapter


async def collect(adapter: QdrantAdapter, **kwargs: Any) -> list[Any]:
    return [point async for point in adapter.iterate_points("docs", **kwargs)]


async def test_a_single_page_is_yielded() -> None:
    adapter = build(ScrollingClient(pages=[([record("c_1"), record("c_2")], None)]))

    points = await collect(adapter)

    assert [point.point_id for point in points] == ["c_1", "c_2"]


async def test_the_cursor_is_followed_across_pages() -> None:
    """A collection larger than one page must stream, not stop at the first batch."""
    client = ScrollingClient(
        pages=[([record("c_1")], "cursor-1"), ([record("c_2")], "cursor-2"), ([], None)]
    )
    adapter = build(client)

    points = await collect(adapter)

    assert [point.point_id for point in points] == ["c_1", "c_2"]
    assert [call["offset"] for call in client.calls] == [None, "cursor-1", "cursor-2"]


async def test_iteration_stops_when_the_cursor_is_exhausted() -> None:
    """A cursor that never returns None would loop forever against a live backend."""
    client = ScrollingClient(pages=[([record("c_1")], None)])
    adapter = build(client)

    await collect(adapter)

    assert len(client.calls) == 1


async def test_the_original_point_id_is_restored_from_the_payload() -> None:
    """Qdrant stores a derived UUID; the archive needs the id fasterRag minted."""
    adapter = build(ScrollingClient(pages=[([record("c_9f2")], None)]))

    points = await collect(adapter)

    assert points[0].point_id == "c_9f2"
    assert POINT_ID_PAYLOAD_KEY not in points[0].payload


async def test_the_payload_survives() -> None:
    adapter = build(ScrollingClient(pages=[([record("c_1", text="chunk body")], None)]))

    assert (await collect(adapter))[0].payload["text"] == "chunk body"


async def test_vectors_are_omitted_by_default() -> None:
    """An archive without --include-vectors should not pay to transfer them."""
    client = ScrollingClient(pages=[([record("c_1")], None)])
    adapter = build(client)

    await collect(adapter)

    assert client.calls[0]["with_vectors"] is False


async def test_an_unnamed_vector_is_returned() -> None:
    adapter = build(ScrollingClient(pages=[([record("c_1", vector=[0.1, 0.2, 0.3])], None)]))

    assert list((await collect(adapter, with_vectors=True))[0].vector) == [0.1, 0.2, 0.3]


async def test_a_named_vector_is_unwrapped() -> None:
    """Hybrid collections name their dense vector; the archive stores a bare list."""
    client = ScrollingClient(
        pages=[([record("c_1", vector={"dense": [0.4, 0.5, 0.6]})], None)], named=True
    )
    adapter = build(client)

    assert list((await collect(adapter, with_vectors=True))[0].vector) == [0.4, 0.5, 0.6]


async def test_the_batch_size_is_passed_through() -> None:
    client = ScrollingClient(pages=[([], None)])
    adapter = build(client)

    await collect(adapter, batch_size=32)

    assert client.calls[0]["limit"] == 32


async def test_a_backend_failure_becomes_a_typed_error() -> None:
    """No vendor exception escapes the adapter boundary."""
    adapter = build(ScrollingClient(raises=OSError("connection reset")))

    with pytest.raises(FasterRagError):
        await collect(adapter)
