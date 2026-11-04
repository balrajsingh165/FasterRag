"""The D11 acceptance test: a collection exported and re-imported through a real Qdrant.

Every other archive test runs against a fake adapter, which proves the format and the
verification order but never that a real backend's points survive the trip. This is the
check TASK-0079 was left open for — it could not run while no Docker daemon existed.

Vector copy is the path asserted here. Re-embedding is exercised by the unit suite, and
proving it against a live backend would mean loading an embedding model in CI to compare
vectors the archive deliberately does not carry.
"""

import json
import re
import tarfile
from collections.abc import AsyncIterator
from contextlib import suppress
from pathlib import Path

import pytest

from fasterrag.adapters.vectordb.base import CollectionSpec, Point, SearchQuery
from fasterrag.adapters.vectordb.qdrant import QdrantAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.bm25 import encode_document, encode_query
from fasterrag.errors import FasterRagError
from fasterrag.services.archive import FORMAT_VERSION, export_archive
from fasterrag.services.archive_import import (
    VerificationError,
    import_archive,
    open_archive,
)
from fasterrag.services.lockfile import LockStore

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
                "source_uri": f"/docs/policy-{index // 3}.md",
                "chunk_index": index % 3,
                "content_hash": f"hash{index // 3:04d}",
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
        assert payload["document_id"] == original.payload["document_id"]
        assert payload["chunk_index"] == original.payload["chunk_index"]
        # CRITICAL: source_uri and content_hash are the two fields the exporter lifts out
        # of the chunk payload into documents.jsonl. Nothing but this assertion notices
        # when import stops joining them back on, and a chunk without a source_uri cannot
        # be cited. The fixture used the key `source` until 2026-11-02, which is exactly
        # why the live run passed while the loss was real.
        assert payload["source_uri"] == original.payload["source_uri"]
        assert payload["content_hash"] == original.payload["content_hash"]


async def test_two_exports_differ_only_by_the_manifest_timestamp(
    populated: tuple[QdrantAdapter, str, list[Point]], qdrant: Settings, tmp_path: Path
) -> None:
    """Otherwise every backup diff reports changes that are only noise."""
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


async def test_a_differently_configured_model_refuses_the_copy_before_writing(
    populated: tuple[QdrantAdapter, str, list[Point]], qdrant: Settings, tmp_path: Path
) -> None:
    """Vectors from one model in a deployment querying with another answer wrongly."""
    adapter, collection, _ = populated
    archive = tmp_path / "corpus.fragx"
    restored = f"{collection}-restored"

    await export_archive(
        qdrant, adapter, collection=collection, destination=archive, include_vectors=True
    )
    elsewhere = qdrant.model_copy(
        update={"embeddings": qdrant.embeddings.model_copy(update={"model": "another/model"})}
    )

    with pytest.raises(FasterRagError, match="different embedding configuration"):
        await import_archive(
            elsewhere,
            adapter,
            open_archive(archive),
            collection=restored,
            lock_root=tmp_path,
        )

    assert restored not in [info.name for info in await adapter.list_collections()]


async def test_the_import_leaves_a_lockfile_describing_the_restored_collection(
    populated: tuple[QdrantAdapter, str, list[Point]], qdrant: Settings, tmp_path: Path
) -> None:
    """Without one, reindex and drift detection have nothing to compare the copy against."""
    adapter, collection, _ = populated
    archive = tmp_path / "corpus.fragx"
    restored = f"{collection}-restored"

    await export_archive(
        qdrant, adapter, collection=collection, destination=archive, include_vectors=True
    )
    await import_archive(
        qdrant, adapter, open_archive(archive), collection=restored, lock_root=tmp_path
    )

    lock = LockStore(tmp_path).read(restored)
    assert lock is not None
    assert lock.dimensions == DIMENSIONS
    assert set(lock.document_hashes) == {f"d_{index:04d}" for index in range(4)}


async def test_a_live_archive_imports_without_materialising_or_indexing_it(
    populated: tuple[QdrantAdapter, str, list[Point]], qdrant: Settings, tmp_path: Path
) -> None:
    """TASK-0293: the streaming import, against an archive a real backend produced.

    The unit suite proves the reader streams a synthetic archive. What only a live export
    can show is that the archive Qdrant's own scroll order produces is read in lockstep —
    no offset index, so import memory does not grow with the collection — and that the
    staging directory the reader writes is gone afterwards.
    """
    adapter, collection, written = populated
    archive = tmp_path / "corpus.fragx"
    restored = f"{collection}-restored"

    await export_archive(
        qdrant, adapter, collection=collection, destination=archive, include_vectors=True
    )
    reader = open_archive(archive)
    staging = reader.staging

    assert reader.vector_offsets is None
    assert (staging / "chunks.jsonl").is_file()

    counts = await import_archive(qdrant, adapter, reader, collection=restored, lock_root=tmp_path)
    reader.close()

    assert counts.chunks == len(written)
    assert not staging.exists()


KEYWORD_PASSAGES = {
    "c_notice": "Either party may terminate this agreement with written notice.",
    "c_payment": "Invoices are payable within thirty days of the notice date.",
    "c_expense": "Expense claims need a receipt and a manager approval.",
}

# CRITICAL: deliberately not the deployment's configured k1/b. An import that recomputed the
# keyword leg from chunk text would produce weights under 1.2/0.75 instead, which scores
# differently — and that difference is the only thing separating a carried leg from a
# re-derived one. Aligning these parameters with the defaults would make this test pass
# against the behaviour it exists to forbid.
SOURCE_K1 = 0.4
SOURCE_B = 0.1


@pytest.fixture
async def hybrid(
    qdrant: Settings, collection_name: str
) -> AsyncIterator[tuple[QdrantAdapter, str]]:
    """Return an adapter over a hybrid collection whose keyword leg is already indexed."""
    adapter = QdrantAdapter(qdrant)

    await adapter.create_collection(
        CollectionSpec(name=collection_name, dimensions=DIMENSIONS, distance="cosine", sparse=True)
    )
    await adapter.upsert(
        [
            Point(
                point_id=chunk_id,
                collection=collection_name,
                vector=vector(index),
                payload={
                    "text": text,
                    "document_id": "d_0000",
                    "source_uri": "/docs/policy-0.md",
                    "content_hash": "hash0000",
                },
                sparse=encode_document(text, k1=SOURCE_K1, b=SOURCE_B),
            )
            for index, (chunk_id, text) in enumerate(KEYWORD_PASSAGES.items())
        ]
    )

    yield adapter, collection_name

    for name in (collection_name, f"{collection_name}-restored"):
        with suppress(FasterRagError):
            await adapter.drop_collection(name)
    await adapter.close()


async def test_the_keyword_ranking_and_scores_survive_the_round_trip(
    hybrid: tuple[QdrantAdapter, str], qdrant: Settings, tmp_path: Path
) -> None:
    """TASK-0302 fidelity is the ranking, not the bytes: a leg that rescores is a leg lost."""
    adapter, collection = hybrid
    archive = tmp_path / "corpus.fragx"
    restored = f"{collection}-restored"
    query = encode_query("terminate notice")

    before = await adapter.search(SearchQuery(collection=collection, sparse=query, limit=5))
    await export_archive(
        qdrant, adapter, collection=collection, destination=archive, include_vectors=True
    )
    await import_archive(
        qdrant, adapter, open_archive(archive), collection=restored, lock_root=tmp_path
    )
    after = await adapter.search(SearchQuery(collection=restored, sparse=query, limit=5))

    assert [hit.point_id for hit in before] == ["c_notice", "c_payment"]
    assert [hit.point_id for hit in after] == [hit.point_id for hit in before]
    assert [hit.score for hit in after] == pytest.approx([hit.score for hit in before])


async def test_the_stored_postings_come_back_term_for_term(
    hybrid: tuple[QdrantAdapter, str], qdrant: Settings, tmp_path: Path
) -> None:
    """What a backend holds is what the archive must carry, weights and all."""
    adapter, collection = hybrid
    archive = tmp_path / "corpus.fragx"
    restored = f"{collection}-restored"

    await export_archive(
        qdrant, adapter, collection=collection, destination=archive, include_vectors=True
    )
    await import_archive(
        qdrant, adapter, open_archive(archive), collection=restored, lock_root=tmp_path
    )

    source = {
        point.point_id: point
        async for point in adapter.iterate_points(collection, with_vectors=True)
    }
    recovered = {
        point.point_id: point async for point in adapter.iterate_points(restored, with_vectors=True)
    }

    assert set(recovered) == set(source) == set(KEYWORD_PASSAGES)
    for point_id, original in source.items():
        carried, held = recovered[point_id].sparse, original.sparse
        assert carried is not None
        assert held is not None
        assert dict(zip(carried.indices, carried.values, strict=True)) == pytest.approx(
            dict(zip(held.indices, held.values, strict=True))
        )


async def test_a_live_hybrid_export_declares_the_newer_major(
    hybrid: tuple[QdrantAdapter, str], qdrant: Settings, tmp_path: Path
) -> None:
    """An older reader must refuse this archive rather than drop its keyword leg quietly."""
    adapter, collection = hybrid
    archive = tmp_path / "corpus.fragx"

    await export_archive(
        qdrant, adapter, collection=collection, destination=archive, include_vectors=True
    )
    with tarfile.open(archive) as tar:
        manifest = json.loads(tar.extractfile("manifest.json").read())  # type: ignore[union-attr]

    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["includes_sparse"] is True
