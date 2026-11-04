import json
import tarfile
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from fasterrag.adapters.vectordb.base import Point, SparseVector
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.archive import (
    ARCHIVE_SUFFIX,
    CHECKSUMS_NAME,
    CHUNKS_NAME,
    DENSE_ONLY_FORMAT_VERSION,
    DOCUMENTS_NAME,
    FORMAT_VERSION,
    LOCK_NAME,
    MANIFEST_NAME,
    VECTORS_NAME,
    export_archive,
)


def point(
    chunk_id: str,
    document_id: str = "d_1",
    *,
    sparse: SparseVector | None = None,
    **extra: object,
) -> Point:
    payload = {
        "document_id": document_id,
        "source_uri": "docs/a.md",
        "content_hash": "abc",
        "text": f"body of {chunk_id}",
        "span": {"start": 0, "end": 10},
        "chunk_index": 0,
        "embedding_model": "m",
        **extra,
    }
    return Point(
        point_id=chunk_id, collection="docs", vector=[0.1, 0.2], payload=payload, sparse=sparse
    )


class FakeAdapter:
    """Serves a fixed set of points through the iteration contract."""

    def __init__(self, points: list[Point]) -> None:
        self.points = points
        self.asked_for_vectors: bool | None = None

    async def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        self.asked_for_vectors = with_vectors
        for item in self.points:
            yield item


async def write(tmp_path: Path, points: list[Point], **kwargs: object) -> Path:
    adapter = FakeAdapter(points)
    destination = tmp_path / "backup"
    await export_archive(
        Settings.model_validate({}),
        adapter,  # type: ignore[arg-type]
        collection="docs",
        destination=destination,
        **kwargs,  # type: ignore[arg-type]
    )
    return destination.with_name(f"backup{ARCHIVE_SUFFIX}")


def members(archive: Path) -> dict[str, bytes]:
    with tarfile.open(archive, "r:gz") as tar:
        return {
            member.name: tar.extractfile(member).read()  # type: ignore[union-attr]
            for member in tar.getmembers()
        }


async def test_the_archive_carries_every_required_member(tmp_path: Path) -> None:
    archive = await write(tmp_path, [point("c_1")])

    assert set(members(archive)) == {
        MANIFEST_NAME,
        CHECKSUMS_NAME,
        DOCUMENTS_NAME,
        CHUNKS_NAME,
        LOCK_NAME,
    }


async def test_vectors_are_omitted_unless_requested(tmp_path: Path) -> None:
    archive = await write(tmp_path, [point("c_1")])

    assert VECTORS_NAME not in members(archive)


async def test_vectors_are_included_on_request(tmp_path: Path) -> None:
    archive = await write(tmp_path, [point("c_1")], include_vectors=True)
    body = members(archive)[VECTORS_NAME].decode("utf-8").strip()

    assert json.loads(body) == {"chunk_id": "c_1", "vector": [0.1, 0.2]}


async def test_the_suffix_is_appended_when_absent(tmp_path: Path) -> None:
    archive = await write(tmp_path, [point("c_1")])

    assert archive.suffix == ARCHIVE_SUFFIX
    assert archive.is_file()


async def test_one_document_row_per_document_not_per_chunk(tmp_path: Path) -> None:
    """Three chunks of one document must not produce three document rows."""
    archive = await write(tmp_path, [point("c_1"), point("c_2"), point("c_3")])
    body = members(archive)

    assert len(body[DOCUMENTS_NAME].decode("utf-8").strip().splitlines()) == 1
    assert len(body[CHUNKS_NAME].decode("utf-8").strip().splitlines()) == 3


async def test_the_manifest_counts_match_the_rows(tmp_path: Path) -> None:
    """Import verifies these against actual line counts, so a lie here defeats the check."""
    archive = await write(
        tmp_path, [point("c_1"), point("c_2", document_id="d_2")], include_vectors=True
    )
    manifest = json.loads(members(archive)[MANIFEST_NAME])

    assert manifest["counts"] == {"documents": 2, "chunks": 2, "vectors": 2}


async def test_a_dense_only_export_declares_the_original_format_version(tmp_path: Path) -> None:
    """An archive that gained no new data must not be stranded over a feature it does not use."""
    manifest = json.loads(members(await write(tmp_path, [point("c_1")]))[MANIFEST_NAME])

    assert manifest["format_version"] == DENSE_ONLY_FORMAT_VERSION
    assert "includes_sparse" not in manifest


async def test_the_manifest_records_whether_vectors_are_present(tmp_path: Path) -> None:
    with_vectors = await write(tmp_path / "a", [point("c_1")], include_vectors=True)
    without = await write(tmp_path / "b", [point("c_1")])

    assert json.loads(members(with_vectors)[MANIFEST_NAME])["includes_vectors"] is True
    assert json.loads(members(without)[MANIFEST_NAME])["includes_vectors"] is False


async def test_the_checksums_cover_every_data_member(tmp_path: Path) -> None:
    archive = await write(tmp_path, [point("c_1")], include_vectors=True)
    body = members(archive)
    listed = {
        line.split("  ", 1)[1] for line in body[CHECKSUMS_NAME].decode("utf-8").strip().splitlines()
    }

    assert listed == {DOCUMENTS_NAME, CHUNKS_NAME, VECTORS_NAME, LOCK_NAME}


async def test_the_checksums_are_correct(tmp_path: Path) -> None:
    import hashlib

    archive = await write(tmp_path, [point("c_1")])
    body = members(archive)
    digests = {
        line.split("  ", 1)[1]: line.split("  ", 1)[0]
        for line in body[CHECKSUMS_NAME].decode("utf-8").strip().splitlines()
    }

    assert digests[CHUNKS_NAME] == hashlib.sha256(body[CHUNKS_NAME]).hexdigest()


async def test_two_exports_of_an_unchanged_collection_are_byte_identical(
    tmp_path: Path,
) -> None:
    """Otherwise every backup diff reports a timestamp and hides the real changes."""
    first = members(await write(tmp_path / "a", [point("c_1")]))
    second = members(await write(tmp_path / "b", [point("c_1")]))

    del first[MANIFEST_NAME], second[MANIFEST_NAME]
    assert first == second


async def test_an_empty_collection_refuses_to_export(tmp_path: Path) -> None:
    """An empty archive imports cleanly and silently produces an empty collection."""
    with pytest.raises(FasterRagError) as caught:
        await write(tmp_path, [])

    assert caught.value.code is ErrorCode.NOT_FOUND


async def test_unnamed_payload_fields_survive_in_metadata(tmp_path: Path) -> None:
    """A round trip must not drop what the backend was holding."""
    archive = await write(tmp_path, [point("c_1", department="legal")])
    row = json.loads(members(archive)[CHUNKS_NAME].decode("utf-8").strip())

    assert row["metadata"]["department"] == "legal"
    assert row["metadata"]["chunk_index"] == 0


async def test_the_named_fields_are_not_duplicated_into_metadata(tmp_path: Path) -> None:
    archive = await write(tmp_path, [point("c_1")])
    row = json.loads(members(archive)[CHUNKS_NAME].decode("utf-8").strip())

    assert "text" not in row["metadata"]
    assert "span" not in row["metadata"]


async def test_vectors_are_not_fetched_when_not_requested(tmp_path: Path) -> None:
    """Transferring vectors nobody asked for is the expensive part of an export."""
    adapter = FakeAdapter([point("c_1")])
    await export_archive(
        Settings.model_validate({}),
        adapter,  # type: ignore[arg-type]
        collection="docs",
        destination=tmp_path / "backup",
    )

    assert adapter.asked_for_vectors is False


async def test_the_manifest_records_the_observed_vector_width(tmp_path: Path) -> None:
    """A deployment with no lockfile and no configured dimension recorded 0.

    An archive declaring zero dimensions cannot be imported at all — the export succeeds and
    produces something unusable, which is worse than failing.
    """
    archive = await write(tmp_path, [point("c_1")], include_vectors=True)

    manifest = json.loads(members(archive)[MANIFEST_NAME])

    assert manifest["embedding"]["dimensions"] == 2


async def test_the_width_is_observed_even_without_exported_vectors(tmp_path: Path) -> None:
    """The collection's width is knowable whether or not the vectors are carried."""
    archive = await write(tmp_path, [point("c_1")])

    assert json.loads(members(archive)[MANIFEST_NAME])["embedding"]["dimensions"] == 2


async def test_the_keyword_leg_is_carried_beside_the_dense_vector(tmp_path: Path) -> None:
    """TASK-0302: a hybrid collection stores two vectors per chunk, so an archive carries two."""
    leg = SparseVector(indices=[3, 11], values=[0.5, 0.25])
    archive = await write(tmp_path, [point("c_1", sparse=leg)], include_vectors=True)
    row = json.loads(members(archive)[VECTORS_NAME].decode("utf-8").strip())

    assert row["sparse"] == {"indices": [3, 11], "values": [0.5, 0.25]}


async def test_sparse_postings_are_written_ascending_by_term_index(tmp_path: Path) -> None:
    """Ascending order is what lets a reader verify the postings and a diff stay quiet."""
    leg = SparseVector(indices=[11, 3], values=[0.25, 0.5])
    archive = await write(tmp_path, [point("c_1", sparse=leg)], include_vectors=True)
    row = json.loads(members(archive)[VECTORS_NAME].decode("utf-8").strip())

    assert row["sparse"] == {"indices": [3, 11], "values": [0.5, 0.25]}


async def test_the_same_postings_in_another_order_export_identically(tmp_path: Path) -> None:
    """Byte-identity must not depend on a vendor walk happening to come back sorted."""
    first = await write(
        tmp_path / "a",
        [point("c_1", sparse=SparseVector(indices=[11, 3], values=[0.25, 0.5]))],
        include_vectors=True,
    )
    second = await write(
        tmp_path / "b",
        [point("c_1", sparse=SparseVector(indices=[3, 11], values=[0.5, 0.25]))],
        include_vectors=True,
    )

    assert members(first)[VECTORS_NAME] == members(second)[VECTORS_NAME]


async def test_a_dense_only_row_carries_no_sparse_key_at_all(tmp_path: Path) -> None:
    """Presence is the signal: `sparse` means the backend was holding a keyword leg."""
    archive = await write(tmp_path, [point("c_1")], include_vectors=True)

    assert "sparse" not in json.loads(members(archive)[VECTORS_NAME].decode("utf-8").strip())


async def test_an_archive_carrying_the_keyword_leg_declares_the_newer_major(
    tmp_path: Path,
) -> None:
    """A 1.x reader accepts a 1.x archive and then drops the postings without a word."""
    leg = SparseVector(indices=[1], values=[1.0])
    archive = await write(tmp_path, [point("c_1", sparse=leg)], include_vectors=True)
    manifest = json.loads(members(archive)[MANIFEST_NAME])

    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["format_version"].split(".")[0] != DENSE_ONLY_FORMAT_VERSION.split(".")[0]
    assert manifest["includes_sparse"] is True


async def test_an_empty_keyword_leg_is_still_carried(tmp_path: Path) -> None:
    """A chunk of nothing but stopwords held an empty leg; recomputing one is not the same."""
    archive = await write(
        tmp_path, [point("c_1", sparse=SparseVector(indices=[], values=[]))], include_vectors=True
    )
    row = json.loads(members(archive)[VECTORS_NAME].decode("utf-8").strip())

    assert row["sparse"] == {"indices": [], "values": []}
    assert json.loads(members(archive)[MANIFEST_NAME])["includes_sparse"] is True
