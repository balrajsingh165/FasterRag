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
    ArchiveCounts,
    build_manifest,
    export_archive,
)
from fasterrag.services.lockfile import build_lock
from tests.support.archive import archive_members


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
        return archive_members(tar)


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


async def test_the_checksums_list_the_data_members_in_the_order_the_spec_lays_them_out(
    tmp_path: Path,
) -> None:
    """The set test above holds for any order, so the vectors member can drift anywhere.

    The container listing puts `vectors.jsonl` between `chunks.jsonl` and `index.lock`,
    and a fixed order is what keeps the member deterministic rather than incidental.
    """
    archive = await write(tmp_path, [point("c_1")], include_vectors=True)
    listed = [
        line.split("  ", 1)[1]
        for line in members(archive)[CHECKSUMS_NAME].decode("utf-8").splitlines()
        if line.strip()
    ]

    assert listed == [DOCUMENTS_NAME, CHUNKS_NAME, VECTORS_NAME, LOCK_NAME]


async def test_a_chunk_with_no_recorded_span_exports_a_zero_span(tmp_path: Path) -> None:
    """`span` is a character offset into the parsed document, and one is where text starts.

    A default of anything else cites the wrong characters on every chunk that lost its
    span, and the citation still looks perfectly well formed.
    """
    spanless = Point(
        point_id="c_1",
        collection="docs",
        vector=[0.1, 0.2],
        payload={"document_id": "d_1", "source_uri": "docs/a.md", "text": "body"},
    )
    archive = await write(tmp_path, [spanless])
    row = json.loads(members(archive)[CHUNKS_NAME].decode("utf-8").strip())

    assert row["span"] == {"start": 0, "end": 0}


async def test_a_recorded_span_is_carried_rather_than_defaulted(tmp_path: Path) -> None:
    """Guards the test above: a writer that hard-coded zeroes would satisfy it too."""
    archive = await write(tmp_path, [point("c_1", span={"start": 17, "end": 42})])
    row = json.loads(members(archive)[CHUNKS_NAME].decode("utf-8").strip())

    assert row["span"] == {"start": 17, "end": 42}


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


async def test_every_tar_header_is_normalised_so_the_archive_file_is_reproducible(
    tmp_path: Path,
) -> None:
    """Equal member bytes are not byte-identity: a real mtime makes every backup file differ.

    The identity test above compares extracted members, so it passes whatever the headers
    say — and the headers are the half a `cmp` or an object-store dedup actually sees.
    """
    archive = await write(tmp_path, [point("c_1")], include_vectors=True)

    with tarfile.open(archive, "r:gz") as tar:
        headers = [
            (member.mtime, member.uid, member.gid, member.uname, member.gname)
            for member in tar.getmembers()
        ]

    assert len(headers) == 6
    assert headers == [(0, 0, 0, "", "")] * 6


async def test_a_document_with_no_recorded_version_exports_as_version_one(
    tmp_path: Path,
) -> None:
    """The spec makes `version` a required int, and zero claims a document before its first."""
    archive = await write(tmp_path, [point("c_1")])
    row = json.loads(members(archive)[DOCUMENTS_NAME].decode("utf-8").strip())

    assert row["version"] == 1


async def test_a_recorded_document_version_is_carried_rather_than_defaulted(
    tmp_path: Path,
) -> None:
    """A re-ingested document is on its second version, and an archive that forgets says one."""
    archive = await write(tmp_path, [point("c_1", version=4)])
    row = json.loads(members(archive)[DOCUMENTS_NAME].decode("utf-8").strip())

    assert row["version"] == 4


async def test_an_empty_collection_refuses_to_export(tmp_path: Path) -> None:
    """An empty archive imports cleanly and silently produces an empty collection."""
    with pytest.raises(FasterRagError) as caught:
        await write(tmp_path, [])

    assert caught.value.code is ErrorCode.NOT_FOUND
    assert caught.value.retryable is False


async def test_the_document_metadata_the_backend_held_survives_the_export(
    tmp_path: Path,
) -> None:
    """Document metadata lives only here, and import rejoins it onto every chunk.

    Losing it leaves every restored chunk with the right text and no idea which team,
    matter or tenant it belonged to, which no count or checksum would notice.
    """
    archive = await write(
        tmp_path, [point("c_1", document_metadata={"team": "legal", "matter": "a-1"})]
    )
    row = json.loads(members(archive)[DOCUMENTS_NAME].decode("utf-8").strip())

    assert row["metadata"] == {"team": "legal", "matter": "a-1"}


async def test_a_dense_only_export_declares_no_vectors_in_its_counts(tmp_path: Path) -> None:
    """Import verifies this against the rows, so a count of vectors nobody carried refuses.

    The counts test above only exercises `--include-vectors`, where the two agree anyway.
    """
    manifest = json.loads(members(await write(tmp_path, [point("c_1")]))[MANIFEST_NAME])

    assert manifest["counts"]["vectors"] == 0


async def test_a_row_is_key_sorted_whatever_order_the_payload_arrived_in(
    tmp_path: Path,
) -> None:
    """Byte-identity across two exports is the point, and a backend's key order is not ours.

    The identity test feeds both runs the same dict, so insertion order is identical and
    sorting is never exercised.
    """
    first = await write(tmp_path / "a", [point("c_1", zebra=1, alpha=2)])
    second = await write(tmp_path / "b", [point("c_1", alpha=2, zebra=1)])

    assert members(first)[CHUNKS_NAME] == members(second)[CHUNKS_NAME]


async def test_non_ascii_text_is_written_as_utf8_rather_than_escaped(tmp_path: Path) -> None:
    """The spec says UTF-8, and an escaped archive is one no other tool greps or diffs."""
    archive = await write(tmp_path, [point("c_1", text="café ☕ cœur")])

    assert "café ☕ cœur".encode() in members(archive)[CHUNKS_NAME]


async def test_a_destination_whose_directory_does_not_exist_yet_is_created(
    tmp_path: Path,
) -> None:
    """An operator naming a backup directory that is not there should get one, not a trace."""
    destination = tmp_path / "backups" / "2026" / "11" / "docs"
    await export_archive(
        Settings.model_validate({}),
        FakeAdapter([point("c_1")]),  # type: ignore[arg-type]
        collection="docs",
        destination=destination,
    )

    assert destination.with_name(f"docs{ARCHIVE_SUFFIX}").is_file()


async def test_the_first_observed_width_is_the_one_recorded(tmp_path: Path) -> None:
    """`observed_dimensions` is a fallback for a collection that declares none.

    A later row must not silently redefine it, because the recorded width is what import
    creates the target collection at.
    """
    narrow = Point(point_id="c_1", collection="docs", vector=[0.1, 0.2], payload={})
    wide = Point(point_id="c_2", collection="docs", vector=[0.1, 0.2, 0.3], payload={})
    archive = await write(tmp_path, [narrow, wide])

    assert json.loads(members(archive)[MANIFEST_NAME])["embedding"]["dimensions"] == 2


async def test_a_staging_directory_that_cannot_be_removed_does_not_fail_the_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The archive is already written when cleanup runs; failing there discards a good backup.

    On Windows an open handle makes the removal fail, and that must not turn a completed
    export into an error an operator has to interpret.
    """

    def refuse(path: object, ignore_errors: bool = False) -> None:
        if not ignore_errors:
            raise OSError("the staging directory is in use")

    monkeypatch.setattr("fasterrag.services.archive.shutil.rmtree", refuse)
    archive = await write(tmp_path, [point("c_1")])

    assert archive.is_file()


def test_a_manifest_built_with_no_keyword_leg_declares_the_older_major() -> None:
    """The default decides for every caller that does not pass the flag at all."""
    manifest = build_manifest(
        Settings.model_validate({}),
        collection="docs",
        lock=None,
        counts=ArchiveCounts(chunks=1),
        includes_vectors=False,
        tenant=None,
    )

    assert manifest["format_version"] == DENSE_ONLY_FORMAT_VERSION
    assert "includes_sparse" not in manifest


def test_the_lockfiles_width_outranks_the_configured_one() -> None:
    """The lock records what actually built the index; config records what is wanted next.

    Recording the configured width would declare a collection at a size none of its own
    vectors are, and import creates the target at whatever the manifest says.
    """
    settings = Settings.model_validate({"embeddings": {"dimensions": 768}})
    lock = build_lock(
        "docs",
        settings,
        embedding_model="m",
        embedding_model_version="1",
        dimensions=16,
    )

    manifest = build_manifest(
        settings,
        collection="docs",
        lock=lock,
        counts=ArchiveCounts(chunks=1),
        includes_vectors=False,
        tenant=None,
        observed_dimensions=32,
    )

    assert manifest["embedding"]["dimensions"] == 16


def test_a_width_nothing_knows_is_recorded_as_zero() -> None:
    """Zero is the sentinel import reads as undeclared and infers from the rows themselves.

    Any other number is a claim, and a wrong claim builds a collection no row fits into.
    """
    manifest = build_manifest(
        Settings.model_validate({}),
        collection="docs",
        lock=None,
        counts=ArchiveCounts(chunks=1),
        includes_vectors=False,
        tenant=None,
        observed_dimensions=None,
    )

    assert manifest["embedding"]["dimensions"] == 0


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
