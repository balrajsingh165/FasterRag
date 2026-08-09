"""The D11 acceptance test: a collection exported and re-imported through a real Qdrant.

Every other archive test runs against a fake adapter, which proves the format and the
verification order but never that a real backend's points survive the trip. This is the
check TASK-0079 was left open for — it could not run while no Docker daemon existed.

Vector copy is the path asserted here. Re-embedding is exercised by the unit suite, and
proving it against a live backend would mean loading an embedding model in CI to compare
vectors the archive deliberately does not carry.
"""

import re
import tarfile
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path

import pytest

from fasterrag.adapters.vectordb.base import CollectionSpec, Point
from fasterrag.adapters.vectordb.qdrant import QdrantAdapter
from fasterrag.config.schema import Settings
from fasterrag.errors import FasterRagError
from fasterrag.services.archive import export_archive
from fasterrag.services.archive_import import (
    VerificationError,
    import_archive,
    open_archive,
)

pytestmark = pytest.mark.integration

DIMENSIONS = 8


def vector(seed: int) -> list[float]:
    """Return a unit-ish vector distinct per seed, so a mixed-up point is visible."""
    values = [0.0] * DIMENSIONS
    values[seed % DIMENSIONS] = 1.0
    values[(seed + 3) % DIMENSIONS] = 0.5
    return values


def points(collection: str, count: int) -> list[Point]:
    return [
        Point(
            point_id=f"c_{index:04d}",
            collection=collection,
            vector=vector(index),
            payload={
                "text": f"chunk number {index} about expense policy",
                "document_id": f"d_{index // 3:04d}",
                "source": f"/docs/policy-{index // 3}.md",
                "chunk_index": index % 3,
                "content_hash": f"hash{index:04d}",
            },
        )
        for index in range(count)
    ]


@pytest.fixture
async def populated(
    qdrant: Settings, collection_name: str
) -> AsyncIterator[tuple[QdrantAdapter, str, list[Point]]]:
    """Return an adapter over a collection holding known points."""
    adapter = QdrantAdapter(qdrant)
    written = points(collection_name, 12)

    await adapter.create_collection(
        CollectionSpec(name=collection_name, dimensions=DIMENSIONS, distance="cosine")
    )
    await adapter.upsert(written)

    yield adapter, collection_name, written

    for name in (collection_name, f"{collection_name}-restored"):
        # Teardown must not mask the failure that brought us here.
        with suppress(FasterRagError):
            await adapter.drop_collection(name)
    await adapter.close()


async def test_a_collection_round_trips_through_an_archive(
    populated: tuple[QdrantAdapter, str, list[Point]], qdrant: Settings, tmp_path: Path
) -> None:
    """The D11 acceptance criterion: export, import, and the same vectors come back."""
    adapter, collection, written = populated
    archive = tmp_path / "corpus.fragx"

    exported = await export_archive(
        qdrant, adapter, collection=collection, destination=archive, include_vectors=True
    )

    assert exported.chunks == len(written)
    assert archive.is_file()

    restored = f"{collection}-restored"
    imported = await import_archive(qdrant, adapter, open_archive(archive), collection=restored)

    assert imported.chunks == len(written)
    counts = {info.name: info.vectors for info in await adapter.list_collections()}
    assert counts[restored] == len(written)


async def test_every_vector_survives_the_copy(
    populated: tuple[QdrantAdapter, str, list[Point]], qdrant: Settings, tmp_path: Path
) -> None:
    """A round trip that loses precision silently changes what the corpus retrieves."""
    adapter, collection, written = populated
    archive = tmp_path / "corpus.fragx"
    restored = f"{collection}-restored"

    await export_archive(
        qdrant, adapter, collection=collection, destination=archive, include_vectors=True
    )
    await import_archive(qdrant, adapter, open_archive(archive), collection=restored)

    source = {
        point.point_id: point
        async for point in adapter.iterate_points(collection, with_vectors=True)
    }
    recovered = {
        point.point_id: point async for point in adapter.iterate_points(restored, with_vectors=True)
    }

    # Compared against the *source collection*, not the raw input. A cosine collection holds
    # unit vectors, so Qdrant normalises on write — [1.0, .., 0.5, ..] comes back as
    # [0.894, .., 0.447, ..]. What the round trip must preserve is what the backend holds,
    # which is not what was handed to it.
    assert set(recovered) == set(source) == {point.point_id for point in written}
    for point_id, original in source.items():
        assert list(recovered[point_id].vector) == pytest.approx(list(original.vector), abs=1e-6)


async def test_payloads_survive_the_copy(
    populated: tuple[QdrantAdapter, str, list[Point]], qdrant: Settings, tmp_path: Path
) -> None:
    """The text and provenance are what make a restored chunk citable."""
    adapter, collection, written = populated
    archive = tmp_path / "corpus.fragx"
    restored = f"{collection}-restored"

    await export_archive(
        qdrant, adapter, collection=collection, destination=archive, include_vectors=True
    )
    await import_archive(qdrant, adapter, open_archive(archive), collection=restored)

    recovered = {
        point.point_id: point async for point in adapter.iterate_points(restored, with_vectors=True)
    }

    for original in written:
        payload = recovered[original.point_id].payload
        assert payload["text"] == original.payload["text"]
        assert payload["source"] == original.payload["source"]
        assert payload["document_id"] == original.payload["document_id"]


async def test_two_exports_differ_only_by_the_manifest_timestamp(
    populated: tuple[QdrantAdapter, str, list[Point]], qdrant: Settings, tmp_path: Path
) -> None:
    """Otherwise every backup diff reports changes that are only noise.

    The precise property, established against a live Qdrant: every *data* member is
    byte-identical across two exports of an unchanged collection, and ``manifest.json``
    differs only in ``created_at``. The source comment claimed whole archives were
    identical; they are not, because a wall clock is embedded on purpose (TASK-0214).
    """
    adapter, collection, _ = populated
    archive = tmp_path / "corpus.fragx"

    exports: list[dict[str, bytes]] = []
    for _ in range(2):
        archive.unlink(missing_ok=True)
        await export_archive(
            qdrant, adapter, collection=collection, destination=archive, include_vectors=True
        )
        with tarfile.open(archive) as tar:
            exports.append(
                {
                    member.name: tar.extractfile(member).read()  # type: ignore[union-attr]
                    for member in tar.getmembers()
                }
            )

    first, second = exports
    assert set(first) == set(second)

    for name in sorted(first):
        if name == "manifest.json":
            continue
        assert first[name] == second[name], f"{name} is not reproducible"

    stamped = re.compile(rb'"created_at": "[^"]+"')
    assert stamped.sub(b"X", first["manifest.json"]) == stamped.sub(b"X", second["manifest.json"])
    assert first["manifest.json"] != second["manifest.json"]


async def test_a_corrupted_archive_is_refused_before_anything_is_written(
    populated: tuple[QdrantAdapter, str, list[Point]], qdrant: Settings, tmp_path: Path
) -> None:
    """A half-imported collection looks like a corpus rather than an error."""
    adapter, collection, _ = populated
    archive = tmp_path / "corpus.fragx"

    await export_archive(
        qdrant, adapter, collection=collection, destination=archive, include_vectors=True
    )
    raw = bytearray(archive.read_bytes())
    raw[len(raw) // 2] ^= 0xFF
    archive.write_bytes(bytes(raw))

    with pytest.raises(VerificationError):
        open_archive(archive)

    assert f"{collection}-restored" not in [info.name for info in await adapter.list_collections()]


async def test_a_dimension_mismatch_refuses_the_vector_copy(
    populated: tuple[QdrantAdapter, str, list[Point]], qdrant: Settings, tmp_path: Path
) -> None:
    """Copying vectors into a collection of another width is the one unrecoverable import."""
    adapter, collection, _ = populated
    archive = tmp_path / "corpus.fragx"
    restored = f"{collection}-restored"

    await export_archive(
        qdrant, adapter, collection=collection, destination=archive, include_vectors=True
    )
    await adapter.create_collection(
        CollectionSpec(name=restored, dimensions=DIMENSIONS * 2, distance="cosine")
    )

    with pytest.raises(FasterRagError) as caught:
        await import_archive(qdrant, adapter, open_archive(archive), collection=restored)

    assert "dimension" in caught.value.detail.lower()
