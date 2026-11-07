import hashlib
import io
import json
import tarfile
import tempfile
import tracemalloc
from pathlib import Path
from typing import Any

import pytest

from fasterrag.adapters.vectordb.base import Point
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.bm25 import encode_document
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services import archive_import
from fasterrag.services.archive import (
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
from fasterrag.services.archive_import import (
    MAXIMUM_MEMBER_BYTES,
    MAXIMUM_TOTAL_BYTES,
    VerificationError,
    import_archive,
    open_archive,
    supported_major,
)
from fasterrag.services.lockfile import LockStore, build_lock


def line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")


def make(
    tmp_path: Path,
    *,
    documents: list[dict[str, Any]] | None = None,
    chunks: list[dict[str, Any]] | None = None,
    vectors: list[dict[str, Any]] | None = None,
    manifest: dict[str, Any] | None = None,
    corrupt: str | None = None,
    drop: str | None = None,
    sparse: bool = False,
) -> Path:
    documents = documents if documents is not None else [{"document_id": "d_1"}]
    chunks = chunks if chunks is not None else [{"chunk_id": "c_1", "document_id": "d_1"}]

    members: dict[str, bytes] = {
        DOCUMENTS_NAME: b"".join(line(row) for row in documents),
        CHUNKS_NAME: b"".join(line(row) for row in chunks),
        LOCK_NAME: b"{}\n",
    }
    if vectors is not None:
        members[VECTORS_NAME] = b"".join(line(row) for row in vectors)

    body = manifest or {
        "format_version": FORMAT_VERSION if sparse else DENSE_ONLY_FORMAT_VERSION,
        "collection": {"name": "docs"},
        "embedding": {"model": "m", "model_version": "1", "dimensions": 2},
        "counts": {
            "documents": len(documents),
            "chunks": len(chunks),
            "vectors": len(vectors or []),
        },
        "includes_vectors": vectors is not None,
        **({"includes_sparse": True} if sparse else {}),
    }

    digests = {name: hashlib.sha256(data).hexdigest() for name, data in members.items()}
    members[CHECKSUMS_NAME] = "".join(
        f"{digest}  {name}\n" for name, digest in digests.items()
    ).encode("utf-8")
    members[MANIFEST_NAME] = line(body)

    if corrupt:
        members[corrupt] = members[corrupt] + b'{"tampered": true}\n'
    if drop:
        del members[drop]

    archive = tmp_path / "a.fragx"
    with tarfile.open(archive, "w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mtime = 0
            import io

            tar.addfile(info, io.BytesIO(data))
    return archive


def test_a_well_formed_archive_verifies(tmp_path: Path) -> None:
    reader = open_archive(make(tmp_path))

    assert reader.collection == "docs"
    assert [row["chunk_id"] for row in reader.chunks()] == ["c_1"]


def test_a_tampered_member_fails_its_checksum(tmp_path: Path) -> None:
    """The archive was modified after export, and nothing else would notice."""
    with pytest.raises(VerificationError, match="checksum"):
        open_archive(make(tmp_path, corrupt=CHUNKS_NAME))


def test_a_missing_required_member_is_refused(tmp_path: Path) -> None:
    with pytest.raises(VerificationError, match="missing required member"):
        open_archive(make(tmp_path, drop=DOCUMENTS_NAME))


def test_a_truncated_archive_is_caught_by_the_counts(tmp_path: Path) -> None:
    """A partial transfer would otherwise import as a smaller corpus and report success."""
    manifest = {
        "format_version": "1.0.0",
        "collection": {"name": "docs"},
        "embedding": {"model": "m", "model_version": "1", "dimensions": 2},
        "counts": {"documents": 1, "chunks": 99, "vectors": 0},
        "includes_vectors": False,
    }

    with pytest.raises(VerificationError, match="declares 99 chunks but contains 1"):
        open_archive(make(tmp_path, manifest=manifest))


def test_a_chunk_with_no_document_is_refused(tmp_path: Path) -> None:
    chunks = [{"chunk_id": "c_1", "document_id": "d_missing"}]

    with pytest.raises(VerificationError, match="references document"):
        open_archive(make(tmp_path, chunks=chunks))


def test_a_vector_with_no_chunk_is_refused(tmp_path: Path) -> None:
    vectors = [{"chunk_id": "c_absent", "vector": [0.1, 0.2]}]

    with pytest.raises(VerificationError, match="vector for chunk"):
        open_archive(make(tmp_path, vectors=vectors))


def test_an_unknown_major_version_is_refused(tmp_path: Path) -> None:
    """A major bump means a field changed meaning; reading it anyway imports wrong data."""
    manifest = {
        "format_version": "3.0.0",
        "collection": {"name": "docs"},
        "embedding": {},
        "counts": {"documents": 1, "chunks": 1, "vectors": 0},
        "includes_vectors": False,
    }

    with pytest.raises(VerificationError, match="format_version"):
        open_archive(make(tmp_path, manifest=manifest))


def test_both_shipped_majors_are_readable_and_nothing_else_is() -> None:
    """1.x predates the keyword leg and 2.x carries it; refusing either strands archives."""
    assert supported_major("1.7.0")
    assert supported_major("2.0.0")
    assert supported_major("2.4.1")
    assert not supported_major("3.0.0")
    assert not supported_major("")


def test_a_file_that_is_not_an_archive_is_refused(tmp_path: Path) -> None:
    plain = tmp_path / "notes.fragx"
    plain.write_text("this is not a tar", encoding="utf-8")

    with pytest.raises(VerificationError, match="not a readable"):
        open_archive(plain)


def test_a_traversing_member_is_refused(tmp_path: Path) -> None:
    """The classic tar traversal, rejected at the boundary rather than trusted inward."""
    archive = tmp_path / "evil.fragx"
    with tarfile.open(archive, "w:gz") as tar:
        import io

        data = b"{}\n"
        info = tarfile.TarInfo(name="../escaped.json")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    with pytest.raises(VerificationError, match="plain file"):
        open_archive(archive)


def test_a_vector_copy_needs_vectors(tmp_path: Path) -> None:
    reader = open_archive(make(tmp_path))

    with pytest.raises(FasterRagError) as caught:
        reader.require_vector_copy_compatible(model="m", model_version="1", dimensions=2)

    assert caught.value.code is ErrorCode.CONFLICT
    assert caught.value.retryable is False
    assert "carries no vectors" in caught.value.detail


def test_a_matching_configuration_permits_a_vector_copy(tmp_path: Path) -> None:
    archive = make(tmp_path, vectors=[{"chunk_id": "c_1", "vector": [0.1, 0.2]}])

    open_archive(archive).require_vector_copy_compatible(model="m", model_version="1", dimensions=2)


def test_a_different_model_refuses_a_vector_copy(tmp_path: Path) -> None:
    """Copying vectors from another model answers every query confidently and wrongly."""
    archive = make(tmp_path, vectors=[{"chunk_id": "c_1", "vector": [0.1, 0.2]}])

    with pytest.raises(FasterRagError, match="model 'm' != 'other'"):
        open_archive(archive).require_vector_copy_compatible(
            model="other", model_version="1", dimensions=2
        )


def test_every_mismatch_is_named_at_once(tmp_path: Path) -> None:
    """A summary alone leaves an operator guessing which of three things to change."""
    archive = make(tmp_path, vectors=[{"chunk_id": "c_1", "vector": [0.1, 0.2]}])

    with pytest.raises(FasterRagError) as caught:
        open_archive(archive).require_vector_copy_compatible(
            model="other", model_version="9", dimensions=768
        )

    detail = caught.value.detail
    assert caught.value.retryable is False
    assert "model" in detail
    assert "model_version" in detail
    assert "dimensions" in detail


class RecordingAdapter:
    """Captures what an import writes."""

    def __init__(self, *, existing: list[str] | None = None, fail_after: int | None = None) -> None:
        self.created: list[Any] = []
        self.points: list[Any] = []
        self.batches: list[int] = []
        self.dropped: list[str] = []
        self.existing = existing or []
        self.fail_after = fail_after

    async def list_collections(self) -> list[Any]:
        from fasterrag.adapters.vectordb.base import CollectionInfo

        return [CollectionInfo(name=name, vectors=0) for name in self.existing]

    async def create_collection(self, spec: Any) -> None:
        self.created.append(spec)

    async def drop_collection(self, name: str) -> bool:
        self.dropped.append(name)
        return True

    async def upsert(self, points: list[Any]) -> Any:
        from fasterrag.adapters.vectordb.base import UpsertResult

        self.batches.append(len(points))
        if self.fail_after is not None and len(self.points) >= self.fail_after:
            raise FasterRagError(
                "the backend went away mid-import",
                code=ErrorCode.INTERNAL,
                retryable=True,
            )
        self.points.extend(points)
        return UpsertResult(upserted=len(points))


def copy_settings() -> Settings:
    """Settings whose embedding identity matches the archives `make` writes."""
    return Settings.model_validate({"embeddings": {"model": "m"}})


class StubEmbedder:
    model = "new-model"
    model_version = "9"
    dimensions = 2

    async def embed_documents(self, texts: list[str]) -> Any:
        from fasterrag.adapters.embeddings.base import EmbeddingResult

        return EmbeddingResult(
            vectors=[[0.9, 0.9] for _ in texts], model=self.model, model_version="9"
        )


class StubRouter:
    def __init__(self) -> None:
        self.default = StubEmbedder()


def vectored(tmp_path: Path) -> Path:
    return make(
        tmp_path,
        chunks=[
            {"chunk_id": "c_1", "document_id": "d_1", "text": "one", "metadata": {"a": 1}},
            {"chunk_id": "c_2", "document_id": "d_1", "text": "two", "metadata": {}},
        ],
        vectors=[
            {"chunk_id": "c_1", "vector": [0.1, 0.2]},
            {"chunk_id": "c_2", "vector": [0.3, 0.4]},
        ],
    )


async def test_a_vector_copy_writes_the_archived_vectors(tmp_path: Path) -> None:
    adapter = RecordingAdapter()
    reader = open_archive(vectored(tmp_path))

    counts = await import_archive(
        copy_settings(),
        adapter,  # type: ignore[arg-type]
        reader,
        collection="restored",
        lock_root=tmp_path,
    )

    assert counts.chunks == 2
    assert [list(point.vector) for point in adapter.points] == [[0.1, 0.2], [0.3, 0.4]]


async def test_re_embedding_ignores_the_archived_vectors(tmp_path: Path) -> None:
    """The whole point of --reembed: the target's model decides, not the archive's."""
    adapter = RecordingAdapter()
    reader = open_archive(vectored(tmp_path))

    await import_archive(
        Settings.model_validate({}),
        adapter,  # type: ignore[arg-type]
        reader,
        collection="restored",
        reembed=True,
        router=StubRouter(),  # type: ignore[arg-type]
        lock_root=tmp_path,
    )

    assert [list(point.vector) for point in adapter.points] == [[0.9, 0.9], [0.9, 0.9]]


async def test_re_embedding_records_the_new_model(tmp_path: Path) -> None:
    adapter = RecordingAdapter()

    await import_archive(
        Settings.model_validate({}),
        adapter,  # type: ignore[arg-type]
        open_archive(vectored(tmp_path)),
        collection="restored",
        reembed=True,
        router=StubRouter(),  # type: ignore[arg-type]
        lock_root=tmp_path,
    )

    assert adapter.points[0].payload["embedding_model"] == "new-model"


async def test_re_embedding_without_a_router_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FasterRagError, match="needs an embedding router") as caught:
        await import_archive(
            Settings.model_validate({}),
            RecordingAdapter(),  # type: ignore[arg-type]
            open_archive(vectored(tmp_path)),
            collection="restored",
            reembed=True,
            lock_root=tmp_path,
        )

    assert caught.value.code is ErrorCode.VALIDATION_FAILED
    assert caught.value.retryable is False


async def test_a_vector_copy_from_a_vectorless_archive_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FasterRagError) as caught:
        await import_archive(
            copy_settings(),
            RecordingAdapter(),  # type: ignore[arg-type]
            open_archive(make(tmp_path)),
            collection="restored",
            lock_root=tmp_path,
        )

    assert caught.value.code is ErrorCode.CONFLICT


async def test_archived_metadata_survives_the_round_trip(tmp_path: Path) -> None:
    adapter = RecordingAdapter()

    await import_archive(
        copy_settings(),
        adapter,  # type: ignore[arg-type]
        open_archive(vectored(tmp_path)),
        collection="restored",
        lock_root=tmp_path,
    )

    assert adapter.points[0].payload["a"] == 1


async def test_metadata_cannot_shadow_a_named_field(tmp_path: Path) -> None:
    """A stray `text` key in metadata must not overwrite the real chunk text."""
    archive = make(
        tmp_path,
        chunks=[
            {
                "chunk_id": "c_1",
                "document_id": "d_1",
                "text": "real body",
                "metadata": {"text": "impostor"},
            }
        ],
        vectors=[{"chunk_id": "c_1", "vector": [0.1, 0.2]}],
    )
    adapter = RecordingAdapter()

    await import_archive(
        copy_settings(),
        adapter,  # type: ignore[arg-type]
        open_archive(archive),
        collection="restored",
        lock_root=tmp_path,
    )

    assert adapter.points[0].payload["text"] == "real body"


async def test_the_collection_is_created_at_the_archived_width(tmp_path: Path) -> None:
    """A collection created at the wrong width fails on the first upsert."""
    adapter = RecordingAdapter()

    await import_archive(
        copy_settings(),
        adapter,  # type: ignore[arg-type]
        open_archive(vectored(tmp_path)),
        collection="restored",
        lock_root=tmp_path,
    )

    assert adapter.created[0].dimensions == 2


async def test_chunk_ids_are_preserved_so_import_is_idempotent(tmp_path: Path) -> None:
    """Deterministic ids are what make a second import upsert rather than duplicate."""
    adapter = RecordingAdapter()

    await import_archive(
        copy_settings(),
        adapter,  # type: ignore[arg-type]
        open_archive(vectored(tmp_path)),
        collection="restored",
        lock_root=tmp_path,
    )

    assert [point.point_id for point in adapter.points] == ["c_1", "c_2"]


async def test_a_vector_copy_into_a_differently_configured_deployment_is_refused(
    tmp_path: Path,
) -> None:
    """The guard compared the manifest against itself, so it could never fire."""
    adapter = RecordingAdapter()

    with pytest.raises(FasterRagError) as caught:
        await import_archive(
            Settings.model_validate({"embeddings": {"model": "other-model"}}),
            adapter,  # type: ignore[arg-type]
            open_archive(vectored(tmp_path)),
            collection="restored",
            lock_root=tmp_path,
        )

    assert caught.value.code is ErrorCode.CONFLICT
    assert "model 'm' != 'other-model'" in caught.value.detail
    assert adapter.points == []


async def test_the_targets_own_lockfile_decides_what_a_vector_copy_must_match(
    tmp_path: Path,
) -> None:
    """A collection already built by one model is the strongest statement of the target."""
    store = LockStore(tmp_path)
    store.write(
        build_lock(
            "restored",
            copy_settings(),
            embedding_model="m",
            embedding_model_version="7",
            dimensions=2,
        )
    )

    with pytest.raises(FasterRagError, match="model_version '1' != '7'"):
        await import_archive(
            copy_settings(),
            RecordingAdapter(),  # type: ignore[arg-type]
            open_archive(vectored(tmp_path)),
            collection="restored",
            lock_root=tmp_path,
        )


def test_a_manifest_that_lies_about_its_width_is_refused(tmp_path: Path) -> None:
    """The manifest's width is what the collection is created at; a lie builds it wrong."""
    archive = make(tmp_path, vectors=[{"chunk_id": "c_1", "vector": [0.1, 0.2, 0.3]}])

    with pytest.raises(VerificationError, match="declares 2 dimensions"):
        open_archive(archive)


def test_vectors_of_differing_widths_are_refused(tmp_path: Path) -> None:
    """A hand-edited vectors member need not declare a width to be inconsistent."""
    manifest = {
        "format_version": "1.0.0",
        "collection": {"name": "docs"},
        "embedding": {"model": "m", "model_version": "1", "dimensions": 0},
        "counts": {"documents": 1, "chunks": 2, "vectors": 2},
        "includes_vectors": True,
    }
    archive = make(
        tmp_path,
        chunks=[
            {"chunk_id": "c_1", "document_id": "d_1"},
            {"chunk_id": "c_2", "document_id": "d_1"},
        ],
        vectors=[
            {"chunk_id": "c_1", "vector": [0.1, 0.2]},
            {"chunk_id": "c_2", "vector": [0.1, 0.2, 0.3]},
        ],
        manifest=manifest,
    )

    with pytest.raises(VerificationError, match="the manifest and the data disagree"):
        open_archive(archive)


async def test_document_provenance_is_restored_onto_every_chunk(tmp_path: Path) -> None:
    """`source_uri` lives only in documents.jsonl; without the join a chunk cannot be cited."""
    archive = make(
        tmp_path,
        documents=[
            {
                "document_id": "d_1",
                "source_uri": "docs/policy.md",
                "content_hash": "sha-abc",
                "version": 1,
                "metadata": {"team": "legal"},
            }
        ],
        chunks=[{"chunk_id": "c_1", "document_id": "d_1", "text": "one", "metadata": {}}],
        vectors=[{"chunk_id": "c_1", "vector": [0.1, 0.2]}],
    )
    adapter = RecordingAdapter()

    await import_archive(
        copy_settings(),
        adapter,  # type: ignore[arg-type]
        open_archive(archive),
        collection="restored",
        lock_root=tmp_path,
    )

    payload = adapter.points[0].payload
    assert payload["source_uri"] == "docs/policy.md"
    assert payload["content_hash"] == "sha-abc"
    assert payload["document_metadata"] == {"team": "legal"}


async def test_a_failed_import_drops_the_collection_it_created(tmp_path: Path) -> None:
    """A half-written collection looks like a corpus rather than like an error."""
    adapter = RecordingAdapter(fail_after=0)

    with pytest.raises(FasterRagError, match="went away mid-import"):
        await import_archive(
            copy_settings(),
            adapter,  # type: ignore[arg-type]
            open_archive(vectored(tmp_path)),
            collection="restored",
            lock_root=tmp_path,
        )

    assert adapter.dropped == ["restored"]


async def test_a_failed_import_never_drops_a_collection_it_found(tmp_path: Path) -> None:
    """Dropping it would destroy rows the archive never carried."""
    adapter = RecordingAdapter(existing=["restored"], fail_after=0)

    with pytest.raises(FasterRagError) as caught:
        await import_archive(
            copy_settings(),
            adapter,  # type: ignore[arg-type]
            open_archive(vectored(tmp_path)),
            collection="restored",
            lock_root=tmp_path,
        )

    assert adapter.dropped == []
    assert caught.value.code is ErrorCode.CONFLICT
    assert caught.value.retryable is False
    assert "re-running the same import is safe" in caught.value.detail


async def test_a_failed_import_writes_no_lockfile(tmp_path: Path) -> None:
    """A lock for a collection that was rolled back would describe nothing."""
    with pytest.raises(FasterRagError):
        await import_archive(
            copy_settings(),
            RecordingAdapter(fail_after=0),  # type: ignore[arg-type]
            open_archive(vectored(tmp_path)),
            collection="restored",
            lock_root=tmp_path,
        )

    assert not (tmp_path / "restored.lock.json").exists()


async def test_the_import_writes_a_fresh_lockfile(tmp_path: Path) -> None:
    """Without one the restored collection reports drift against whatever lock is lying around."""
    await import_archive(
        copy_settings(),
        RecordingAdapter(),  # type: ignore[arg-type]
        open_archive(vectored(tmp_path)),
        collection="restored",
        lock_root=tmp_path,
    )

    written = LockStore(tmp_path).read("restored")
    assert written is not None
    assert written.embedding_model == "m"
    assert written.dimensions == 2
    assert written.document_hashes == {"d_1": ""}


async def test_the_archived_lockfile_is_kept_beside_the_fresh_one(tmp_path: Path) -> None:
    """The archive's own lock is the provenance of where the collection came from."""
    await import_archive(
        copy_settings(),
        RecordingAdapter(),  # type: ignore[arg-type]
        open_archive(vectored(tmp_path)),
        collection="restored",
        lock_root=tmp_path,
    )

    assert (tmp_path / "restored.source.lock.json").read_bytes() == b"{}\n"


def test_a_member_declaring_more_than_the_limit_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An over-large member is refused on its header, before any of it is read."""
    monkeypatch.setattr("fasterrag.services.archive_import.MAXIMUM_MEMBER_BYTES", 1_000)

    archive = tmp_path / "bomb.fragx"
    payload = b"\0" * 2_000
    with tarfile.open(archive, "w:gz") as handle:
        info = tarfile.TarInfo(MANIFEST_NAME)
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))

    with pytest.raises(VerificationError, match="above the"):
        open_archive(archive)


def test_a_decompression_bomb_is_refused_before_it_exhausts_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Members each under the per-member ceiling must still be refused in aggregate.

    The limits are lowered rather than writing a real multi-gigabyte bomb; the expansion
    ratio under test is the one gzip gives an attacker for free.
    """
    monkeypatch.setattr("fasterrag.services.archive_import.MAXIMUM_MEMBER_BYTES", 1_000_000)
    monkeypatch.setattr("fasterrag.services.archive_import.MAXIMUM_TOTAL_BYTES", 300_000)

    archive = tmp_path / "bomb.fragx"
    payload = b"\0" * 200_000
    with tarfile.open(archive, "w:gz") as handle:
        for name in (MANIFEST_NAME, CHUNKS_NAME, DOCUMENTS_NAME):
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            handle.addfile(info, io.BytesIO(payload))

    assert archive.stat().st_size < 4_096

    with pytest.raises(VerificationError, match="expands beyond"):
        open_archive(archive)


def test_the_decompression_ceilings_are_the_ones_the_security_doc_names() -> None:
    """`docs/security.md` §5.2 states 1 GiB per member, 4 GiB per archive, 256 KiB blocks.

    Both bomb tests lower the ceilings before they run, so every one of them passes at
    whatever the shipped values happen to be — including values that bound nothing.
    """
    assert MAXIMUM_MEMBER_BYTES == 1024**3
    assert MAXIMUM_TOTAL_BYTES == 4 * 1024**3
    assert archive_import._BLOCK_BYTES == 256 * 1024


def test_a_member_of_exactly_the_member_ceiling_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ceiling is the largest member allowed, not the smallest one refused.

    Refusing it turns a legal archive into a corrupt-archive error message, and the
    refusal test above cannot tell the two readings apart.
    """
    archive = make(tmp_path)
    with tarfile.open(archive, "r:gz") as tar:
        widest = max(member.size for member in tar.getmembers())
    monkeypatch.setattr("fasterrag.services.archive_import.MAXIMUM_MEMBER_BYTES", widest)

    with open_archive(archive) as reader:
        assert reader.collection == "docs"


def test_an_archive_that_exactly_fills_the_byte_budget_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The budget is what an archive may spend, so spending all of it is not a breach."""
    archive = make(tmp_path)
    with tarfile.open(archive, "r:gz") as tar:
        total = sum(member.size for member in tar.getmembers())
    monkeypatch.setattr("fasterrag.services.archive_import.MAXIMUM_TOTAL_BYTES", total)

    with open_archive(archive) as reader:
        assert reader.collection == "docs"


def test_a_manifest_that_omits_a_row_count_is_refused(tmp_path: Path) -> None:
    """A count nobody declared is not a count that happens to match.

    The sentinel is what stops an absent `counts.vectors` reading as zero and agreeing
    with an archive whose vectors member was lost in transfer.
    """
    manifest = {
        "format_version": DENSE_ONLY_FORMAT_VERSION,
        "collection": {"name": "docs"},
        "embedding": {"model": "m", "model_version": "1", "dimensions": 2},
        "counts": {"documents": 1, "chunks": 1},
        "includes_vectors": False,
    }

    with pytest.raises(VerificationError, match="declares -1 vectors but contains 0"):
        open_archive(make(tmp_path, manifest=manifest))


def test_an_archive_that_declares_no_width_is_read_at_the_width_of_its_own_rows(
    tmp_path: Path,
) -> None:
    """Zero means undeclared, and the first row is then what the rest must agree with.

    Only the refusal is pinned above, and a reader that treated an undeclared width as
    one would still refuse — for the wrong reason, on every archive of this shape.
    """
    manifest = {
        "format_version": DENSE_ONLY_FORMAT_VERSION,
        "collection": {"name": "docs"},
        "embedding": {"model": "m", "model_version": "1", "dimensions": 0},
        "counts": {"documents": 1, "chunks": 2, "vectors": 2},
        "includes_vectors": True,
    }
    archive = make(
        tmp_path,
        chunks=[
            {"chunk_id": "c_1", "document_id": "d_1"},
            {"chunk_id": "c_2", "document_id": "d_1"},
        ],
        vectors=[
            {"chunk_id": "c_1", "vector": [0.1, 0.2]},
            {"chunk_id": "c_2", "vector": [0.3, 0.4]},
        ],
        manifest=manifest,
    )

    with open_archive(archive) as reader:
        assert reader.embedding["dimensions"] == 0


async def test_a_chunk_row_with_no_span_is_restored_at_a_zero_span(tmp_path: Path) -> None:
    """A span is a character offset into the parsed document, and text starts at zero.

    Any other default cites the wrong characters on every chunk that lost its span, and
    the citation it produces still looks perfectly well formed.
    """
    archive = make(
        tmp_path,
        chunks=[{"chunk_id": "c_1", "document_id": "d_1", "text": "one", "metadata": {}}],
        vectors=[{"chunk_id": "c_1", "vector": [0.1, 0.2]}],
    )
    adapter = RecordingAdapter()

    await import_archive(
        copy_settings(),
        adapter,  # type: ignore[arg-type]
        open_archive(archive),
        collection="restored",
        lock_root=tmp_path,
    )

    assert adapter.points[0].payload["span"] == {"start": 0, "end": 0}


async def test_an_archived_span_is_restored_rather_than_defaulted(tmp_path: Path) -> None:
    """Guards the test above: a writer that hard-coded zeroes would satisfy it too."""
    archive = make(
        tmp_path,
        chunks=[
            {
                "chunk_id": "c_1",
                "document_id": "d_1",
                "text": "one",
                "span": {"start": 17, "end": 42},
                "metadata": {},
            }
        ],
        vectors=[{"chunk_id": "c_1", "vector": [0.1, 0.2]}],
    )
    adapter = RecordingAdapter()

    await import_archive(
        copy_settings(),
        adapter,  # type: ignore[arg-type]
        open_archive(archive),
        collection="restored",
        lock_root=tmp_path,
    )

    assert adapter.points[0].payload["span"] == {"start": 17, "end": 42}


async def test_a_full_batch_is_flushed_rather_than_allowed_to_overflow(tmp_path: Path) -> None:
    """Batching bounds the set of points in flight, and a batch of `n + 1` is not bounded.

    Counting chunks cannot see this: every row still arrives, in batches of the wrong size.
    """
    archive = make(
        tmp_path,
        chunks=[
            {"chunk_id": f"c_{index}", "document_id": "d_1", "text": "t", "metadata": {}}
            for index in range(4)
        ],
        vectors=[{"chunk_id": f"c_{index}", "vector": [0.1, 0.2]} for index in range(4)],
    )
    adapter = RecordingAdapter()

    await import_archive(
        copy_settings(),
        adapter,  # type: ignore[arg-type]
        open_archive(archive),
        collection="restored",
        batch_size=2,
        lock_root=tmp_path,
    )

    assert adapter.batches == [2, 2]


def test_the_readers_repr_does_not_dump_what_it_staged(tmp_path: Path) -> None:
    """A reader reaches log lines and exception messages; the index it holds must not.

    `vector_offsets` carries one entry per vector row, so a repr that included it turns
    a single line about a large collection into megabytes of it.
    """
    with open_archive(make(tmp_path)) as reader:
        reader.vector_offsets = {"c_1": 0}
        shown = repr(reader)

    assert "vector_offsets" not in shown
    assert "workspace" not in shown
    assert "handles" not in shown


def test_a_sparse_entry_with_only_one_of_the_two_lists_is_refused(tmp_path: Path) -> None:
    """Both halves are required, and a check that wanted both missing accepts a half one.

    The entry below reaches the length comparison, which asks a float for its length.
    """
    with pytest.raises(VerificationError, match="indices and a values list"):
        open_archive(keyworded(tmp_path, sparse={"indices": [7], "values": 0.5}, declared=True))


def test_a_manifest_that_declares_no_width_matches_no_target_width(tmp_path: Path) -> None:
    """A value nobody declared is not a value that happens to equal the target's.

    An archive silent about its width must refuse every vector copy, including into the
    one target width a default could accidentally agree with.
    """
    manifest = {
        "format_version": DENSE_ONLY_FORMAT_VERSION,
        "collection": {"name": "docs"},
        "embedding": {"model": "m", "model_version": "1"},
        "counts": {"documents": 1, "chunks": 1, "vectors": 1},
        "includes_vectors": True,
    }
    archive = make(
        tmp_path,
        chunks=[{"chunk_id": "c_1", "document_id": "d_1"}],
        vectors=[{"chunk_id": "c_1", "vector": [0.1]}],
        manifest=manifest,
    )

    with pytest.raises(FasterRagError, match="dimensions None != 1"):
        open_archive(archive).require_vector_copy_compatible(
            model="m", model_version="1", dimensions=1
        )


async def test_an_archive_that_declares_no_width_is_imported_at_the_width_of_its_rows(
    tmp_path: Path,
) -> None:
    """The collection is created at whatever width is settled on, so it must be the rows'.

    Nothing here declares one — not the manifest, not the lockfile, not the settings — and
    the only truthful answer is the vector that is actually about to be written.
    """
    manifest = {
        "format_version": DENSE_ONLY_FORMAT_VERSION,
        "collection": {"name": "docs"},
        "embedding": {"model": "m", "model_version": "1"},
        "counts": {"documents": 1, "chunks": 1, "vectors": 1},
        "includes_vectors": True,
    }
    archive = make(
        tmp_path,
        chunks=[{"chunk_id": "c_1", "document_id": "d_1", "text": "one", "metadata": {}}],
        vectors=[{"chunk_id": "c_1", "vector": [0.1, 0.2]}],
        manifest=manifest,
    )
    adapter = RecordingAdapter()

    await import_archive(
        copy_settings(),
        adapter,  # type: ignore[arg-type]
        open_archive(archive),
        collection="restored",
        lock_root=tmp_path,
    )

    assert adapter.created[0].dimensions == 2


async def test_the_archived_width_outranks_the_configured_one(tmp_path: Path) -> None:
    """The archive says how wide its own vectors are; the config says what is wanted next.

    A vector copy writes the archived vectors, so creating the collection at the configured
    width builds one that no row about to be written fits into.
    """
    store = LockStore(tmp_path)
    store.write(
        build_lock(
            "restored",
            copy_settings(),
            embedding_model="m",
            embedding_model_version="1",
            dimensions=2,
        )
    )
    adapter = RecordingAdapter()

    await import_archive(
        Settings.model_validate({"embeddings": {"model": "m", "dimensions": 768}}),
        adapter,  # type: ignore[arg-type]
        open_archive(vectored(tmp_path)),
        collection="restored",
        lock_root=tmp_path,
    )

    assert adapter.created[0].dimensions == 2


class WidthlessEmbedder:
    """An embedder that declares no dimensionality, as a local model may not."""

    model = "new-model"
    model_version = "9"
    dimensions: int | None = None


class WidthlessRouter:
    """A router whose default embedder declares no width."""

    def __init__(self) -> None:
        self.default = WidthlessEmbedder()


async def test_an_unknown_target_width_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    """Nothing declares a width on the re-embed path, and a guess creates the wrong index.

    Retrying cannot help: the configuration has to change, so the refusal says so.
    """
    with pytest.raises(FasterRagError, match="target vector size is unknown") as caught:
        await import_archive(
            Settings.model_validate({}),
            RecordingAdapter(),  # type: ignore[arg-type]
            open_archive(vectored(tmp_path)),
            collection="restored",
            reembed=True,
            router=WidthlessRouter(),  # type: ignore[arg-type]
            lock_root=tmp_path,
        )

    assert caught.value.code is ErrorCode.VALIDATION_FAILED
    assert caught.value.retryable is False


class UndroppableAdapter(RecordingAdapter):
    """An adapter whose collections cannot be dropped, as a backend mid-outage cannot."""

    async def drop_collection(self, name: str) -> bool:
        raise FasterRagError(
            "the backend refused the drop", code=ErrorCode.INTERNAL, retryable=True
        )


async def test_a_collection_that_cannot_be_dropped_is_named_for_a_human(tmp_path: Path) -> None:
    """The rollback is the only thing standing between a failed import and a fake corpus.

    When it fails too, the one useful act is to say which collection is now incomplete —
    and to say that retrying will not clear it, because the drop is what failed.
    """
    adapter = UndroppableAdapter(fail_after=0)

    with pytest.raises(FasterRagError, match="must be removed by hand") as caught:
        await import_archive(
            copy_settings(),
            adapter,  # type: ignore[arg-type]
            open_archive(vectored(tmp_path)),
            collection="restored",
            lock_root=tmp_path,
        )

    assert caught.value.code is ErrorCode.CONFLICT
    assert caught.value.retryable is False
    assert "restored" in caught.value.detail


def test_closing_the_reader_releases_the_staging_directory(tmp_path: Path) -> None:
    """The members live on disk now, so something has to delete them."""
    reader = open_archive(make(tmp_path))
    staging = reader.staging

    assert (staging / CHUNKS_NAME).is_file()

    reader.close()
    reader.close()

    assert not staging.exists()


def test_leaving_the_reader_block_releases_the_staging_directory(tmp_path: Path) -> None:
    """A caller that scopes the reader should not have to remember the directory."""
    with open_archive(make(tmp_path)) as reader:
        staging = reader.staging
        assert staging.is_dir()

    assert not staging.exists()


def test_a_staging_directory_that_cannot_be_removed_does_not_break_the_caller(
    tmp_path: Path,
) -> None:
    """On Windows an open handle makes cleanup fail, and an import must not die of it."""
    reader = open_archive(make(tmp_path))

    def refuse() -> None:
        raise OSError("the staging directory is in use")

    assert reader.workspace is not None
    reader.workspace.cleanup = refuse  # type: ignore[method-assign]

    reader.close()

    assert reader.workspace is None


def test_a_refused_archive_leaves_no_staging_directory_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verification now writes before it refuses, and a refusal must not leak what it wrote."""
    root = tmp_path / "staging-root"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))

    with pytest.raises(VerificationError, match="checksum"):
        open_archive(make(tmp_path, corrupt=CHUNKS_NAME))

    assert list(root.iterdir()) == []


async def test_a_chunk_missing_its_vector_is_refused_not_handed_its_neighbours(
    tmp_path: Path,
) -> None:
    """Reading the vectors in step with the chunks has to match on id, not on position."""
    archive = make(
        tmp_path,
        chunks=[
            {"chunk_id": "c_1", "document_id": "d_1", "text": "one", "metadata": {}},
            {"chunk_id": "c_2", "document_id": "d_1", "text": "two", "metadata": {}},
        ],
        vectors=[{"chunk_id": "c_2", "vector": [0.3, 0.4]}],
    )
    adapter = RecordingAdapter()

    with pytest.raises(FasterRagError) as caught:
        await import_archive(
            copy_settings(),
            adapter,  # type: ignore[arg-type]
            open_archive(archive),
            collection="restored",
            lock_root=tmp_path,
        )

    assert caught.value.code is ErrorCode.CONFLICT
    assert caught.value.retryable is False
    assert "no vector for chunk 'c_1'" in caught.value.detail
    assert adapter.points == []


async def test_vectors_out_of_chunk_order_still_land_on_their_own_chunks(
    tmp_path: Path,
) -> None:
    """Nothing in the format promises vector order, so a lockstep read alone would mispair them."""
    archive = make(
        tmp_path,
        chunks=[
            {"chunk_id": "c_1", "document_id": "d_1", "text": "one", "metadata": {}},
            {"chunk_id": "c_2", "document_id": "d_1", "text": "two", "metadata": {}},
        ],
        vectors=[
            {"chunk_id": "c_2", "vector": [0.3, 0.4]},
            {"chunk_id": "c_1", "vector": [0.1, 0.2]},
        ],
    )
    adapter = RecordingAdapter()

    await import_archive(
        copy_settings(),
        adapter,  # type: ignore[arg-type]
        open_archive(archive),
        collection="restored",
        lock_root=tmp_path,
    )

    assert [(point.point_id, list(point.vector)) for point in adapter.points] == [
        ("c_1", [0.1, 0.2]),
        ("c_2", [0.3, 0.4]),
    ]


class CountingAdapter:
    """Accepts points and forgets them, so a memory measurement sees the import, not the test."""

    def __init__(self) -> None:
        self.written = 0
        self.created: list[Any] = []

    async def list_collections(self) -> list[Any]:
        return []

    async def create_collection(self, spec: Any) -> None:
        self.created.append(spec)

    async def drop_collection(self, name: str) -> bool:
        return True

    async def upsert(self, points: list[Any]) -> Any:
        from fasterrag.adapters.vectordb.base import UpsertResult

        self.written += len(points)
        return UpsertResult(upserted=len(points))


BULK_CHUNKS = 2_000
BULK_DIMENSIONS = 128


def bulky(tmp_path: Path, *, with_sparse: bool = False) -> tuple[Path, int]:
    """Write an archive far larger than a streaming reader's working set.

    Returns the archive and the uncompressed size of its row members, which is what an
    implementation that materialises them has to hold.
    """
    postings = {
        "indices": [index * 7 for index in range(120)],
        "values": [round(0.5 + index / 1000, 6) for index in range(120)],
    }
    text = "expense policy section " * 350
    vector = [round(0.001 * index, 6) for index in range(BULK_DIMENSIONS)]
    documents: list[dict[str, Any]] = [
        {
            "document_id": "d_1",
            "source_uri": "/docs/policy.md",
            "content_hash": "sha-abc",
            "version": 1,
            "metadata": {},
        }
    ]
    chunks: list[dict[str, Any]] = [
        {"chunk_id": f"c_{index:05d}", "document_id": "d_1", "text": text, "metadata": {}}
        for index in range(BULK_CHUNKS)
    ]
    vectors: list[dict[str, Any]] = [
        {
            "chunk_id": f"c_{index:05d}",
            "vector": vector,
            **({"sparse": postings} if with_sparse else {}),
        }
        for index in range(BULK_CHUNKS)
    ]
    manifest: dict[str, Any] = {
        "format_version": FORMAT_VERSION if with_sparse else DENSE_ONLY_FORMAT_VERSION,
        "collection": {"name": "docs"},
        "embedding": {"model": "m", "model_version": "1", "dimensions": BULK_DIMENSIONS},
        "counts": {"documents": 1, "chunks": BULK_CHUNKS, "vectors": BULK_CHUNKS},
        "includes_vectors": True,
        **({"includes_sparse": True} if with_sparse else {}),
    }

    archive = make(tmp_path, documents=documents, chunks=chunks, vectors=vectors, manifest=manifest)
    payload = sum(len(line(row)) for row in [*documents, *chunks, *vectors])
    return archive, payload


def bulk_settings() -> Settings:
    """Copy settings with the keyword leg off, so the measurement is of the archive stream."""
    return Settings.model_validate({"embeddings": {"model": "m"}, "retrieval": {"hybrid": False}})


async def test_the_import_never_holds_the_archive_it_is_reading(tmp_path: Path) -> None:
    """TASK-0293: a collection is exactly the thing that does not fit in memory.

    The bomb ceilings bound a hostile archive at 4 GiB; they do nothing for an honest
    corpus larger than RAM. The reader that materialised every member before verifying a
    checksum could not pass this, and no other assertion in this file would have noticed.
    """
    archive, payload = bulky(tmp_path)
    adapter = CountingAdapter()

    assert payload > 8 * 1024 * 1024

    tracemalloc.start()
    try:
        counts = await import_archive(
            bulk_settings(),
            adapter,  # type: ignore[arg-type]
            open_archive(archive),
            collection="restored",
            lock_root=tmp_path,
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert counts.chunks == BULK_CHUNKS
    assert adapter.written == BULK_CHUNKS
    assert peak < payload // 4, f"held {peak} bytes while reading a {payload} byte archive"


class SourceAdapter:
    """Yields points the way a backend's cursor does, so the real exporter can run."""

    def __init__(self, points: list[Any]) -> None:
        self.points = points

    async def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> Any:
        for point in self.points:
            yield point


async def test_an_archive_from_the_real_exporter_needs_no_offset_index(tmp_path: Path) -> None:
    """The exporter writes vectors in chunk order, so the import holds one row, not an index.

    The index is the fallback for an archive nobody here produced. If the normal path ever
    takes it, import memory grows with the collection again and nothing else would say so.
    """
    source = SourceAdapter(
        [
            Point(
                point_id=f"c_{index}",
                collection="docs",
                vector=[0.1 * index, 0.2],
                payload={
                    "document_id": "d_1",
                    "source_uri": "/docs/policy.md",
                    "content_hash": "sha-abc",
                    "text": f"chunk {index}",
                },
            )
            for index in range(5)
        ]
    )
    archive = tmp_path / "exported.fragx"

    await export_archive(
        copy_settings(),
        source,  # type: ignore[arg-type]
        collection="docs",
        destination=archive,
        include_vectors=True,
    )
    reader = open_archive(archive)
    adapter = RecordingAdapter()

    counts = await import_archive(
        copy_settings(),
        adapter,  # type: ignore[arg-type]
        reader,
        collection="restored",
        lock_root=tmp_path,
    )

    assert reader.vector_offsets is None
    assert counts.chunks == 5
    assert [point.point_id for point in adapter.points] == [f"c_{index}" for index in range(5)]
    assert adapter.points[3].payload["source_uri"] == "/docs/policy.md"


async def test_re_embedding_an_archive_that_carries_no_vectors_imports_it(
    tmp_path: Path,
) -> None:
    """An export without `--include-vectors` can only come back this way, so it must work.

    There is no vectors member to stream at all, and the reader has to say so rather than
    open a file that was never written.
    """
    archive = make(
        tmp_path,
        chunks=[
            {"chunk_id": "c_1", "document_id": "d_1", "text": "one", "metadata": {}},
            {"chunk_id": "c_2", "document_id": "d_1", "text": "two", "metadata": {}},
        ],
    )
    adapter = RecordingAdapter()

    counts = await import_archive(
        Settings.model_validate({}),
        adapter,  # type: ignore[arg-type]
        open_archive(archive),
        collection="restored",
        reembed=True,
        router=StubRouter(),  # type: ignore[arg-type]
        lock_root=tmp_path,
    )

    assert counts.chunks == 2
    assert [list(point.vector) for point in adapter.points] == [[0.9, 0.9], [0.9, 0.9]]


def test_a_refusal_raised_mid_stream_still_releases_the_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A refusal leaves its row iterator suspended, and an open file cannot be deleted.

    Found in the wild while building this: 61 abandoned staging directories, each holding
    the one `vectors.jsonl` that the reference check still had open when it refused.
    """
    root = tmp_path / "staging-root"
    root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(root))

    with pytest.raises(VerificationError, match="vector for chunk"):
        open_archive(make(tmp_path, vectors=[{"chunk_id": "c_absent", "vector": [0.1, 0.2]}]))

    assert list(root.iterdir()) == []


KEYWORD_TEXT = "termination notice period for either party"
CARRIED = {"indices": [7, 9, 4242], "values": [0.75, 0.5, 0.25]}


def keyworded(tmp_path: Path, *, sparse: dict[str, Any] | None = None, declared: bool) -> Path:
    """Return an archive of one chunk, optionally carrying a keyword leg for it."""
    row: dict[str, Any] = {"chunk_id": "c_1", "vector": [0.1, 0.2]}
    if sparse is not None:
        row["sparse"] = sparse
    return make(
        tmp_path,
        chunks=[{"chunk_id": "c_1", "document_id": "d_1", "text": KEYWORD_TEXT, "metadata": {}}],
        vectors=[row],
        sparse=declared,
    )


async def imported(tmp_path: Path, archive: Path, settings: Settings) -> RecordingAdapter:
    """Import one archive through a recording adapter and return what it was handed."""
    adapter = RecordingAdapter()
    await import_archive(
        settings,
        adapter,  # type: ignore[arg-type]
        open_archive(archive),
        collection="restored",
        lock_root=tmp_path,
    )
    return adapter


async def test_the_carried_keyword_leg_is_written_rather_than_recomputed(tmp_path: Path) -> None:
    """TASK-0302: a restored leg keeps the `k1`/`b` the corpus was actually indexed with.

    This is the one that fails against the pre-change reader, which dropped the archived
    postings and re-derived them from the chunk text under the *target's* parameters.
    """
    adapter = await imported(
        tmp_path, keyworded(tmp_path, sparse=CARRIED, declared=True), copy_settings()
    )
    stored = adapter.points[0].sparse

    assert stored is not None
    assert list(stored.indices) == CARRIED["indices"]
    assert list(stored.values) == pytest.approx(CARRIED["values"])
    assert list(stored.indices) != list(encode_document(KEYWORD_TEXT).indices)


async def test_an_archive_without_a_keyword_leg_has_one_recomputed(tmp_path: Path) -> None:
    """A 1.0.0 archive predates the leg, and a collection with none answers on one leg."""
    settings = copy_settings()
    adapter = await imported(tmp_path, keyworded(tmp_path, declared=False), settings)
    stored = adapter.points[0].sparse
    expected = encode_document(
        KEYWORD_TEXT, k1=settings.retrieval.bm25_k1, b=settings.retrieval.bm25_b
    )

    assert stored is not None
    assert list(stored.indices) == list(expected.indices)
    assert list(stored.values) == pytest.approx(list(expected.values))


async def test_a_target_with_no_keyword_leg_stores_none(tmp_path: Path) -> None:
    """`retrieval.hybrid` off creates a dense-only collection; postings have nowhere to go."""
    settings = Settings.model_validate(
        {"embeddings": {"model": "m"}, "retrieval": {"hybrid": False}}
    )
    adapter = await imported(tmp_path, keyworded(tmp_path, sparse=CARRIED, declared=True), settings)

    assert adapter.points[0].sparse is None


async def test_re_embedding_recomputes_the_keyword_leg_rather_than_carrying_it(
    tmp_path: Path,
) -> None:
    """`--reembed` rebuilds the corpus under the target's config, and BM25 is part of it."""
    settings = copy_settings()
    adapter = RecordingAdapter()

    await import_archive(
        settings,
        adapter,  # type: ignore[arg-type]
        open_archive(keyworded(tmp_path, sparse=CARRIED, declared=True)),
        collection="restored",
        reembed=True,
        router=StubRouter(),  # type: ignore[arg-type]
        lock_root=tmp_path,
    )
    stored = adapter.points[0].sparse

    assert stored is not None
    assert list(stored.indices) == list(encode_document(KEYWORD_TEXT).indices)


def test_a_keyword_leg_the_manifest_never_declared_is_refused(tmp_path: Path) -> None:
    """Such an archive declares a major an older reader accepts, which then drops the leg."""
    with pytest.raises(VerificationError, match="does not declare includes_sparse"):
        open_archive(keyworded(tmp_path, sparse=CARRIED, declared=False))


def test_a_declared_keyword_leg_that_is_absent_is_refused(tmp_path: Path) -> None:
    """The manifest promising postings nothing carries is the truncation check again."""
    with pytest.raises(VerificationError, match="declares includes_sparse"):
        open_archive(keyworded(tmp_path, declared=True))


def test_sparse_indices_that_are_not_ascending_are_refused(tmp_path: Path) -> None:
    """A duplicate or unordered index has no defined meaning; backends disagree on it."""
    unordered = {"indices": [9, 7], "values": [0.5, 0.75]}

    with pytest.raises(VerificationError, match="strictly ascending"):
        open_archive(keyworded(tmp_path, sparse=unordered, declared=True))


def test_sparse_indices_repeated_in_one_posting_list_are_refused(tmp_path: Path) -> None:
    """The backend keeps whichever value it sees last, so the same leg scores differently."""
    repeated = {"indices": [7, 7], "values": [0.5, 0.75]}

    with pytest.raises(VerificationError, match="strictly ascending"):
        open_archive(keyworded(tmp_path, sparse=repeated, declared=True))


def test_sparse_indices_and_values_of_different_lengths_are_refused(tmp_path: Path) -> None:
    """Nothing downstream can pair them, and `SparseVector` would raise mid-write."""
    ragged = {"indices": [7, 9], "values": [0.5]}

    with pytest.raises(VerificationError, match="2 sparse indices but 1 values"):
        open_archive(keyworded(tmp_path, sparse=ragged, declared=True))


def test_a_sparse_entry_that_is_not_two_lists_is_refused(tmp_path: Path) -> None:
    """A hand-edited archive reaches the writer otherwise, and fails halfway through it."""
    with pytest.raises(VerificationError, match="indices and a values list"):
        open_archive(keyworded(tmp_path, sparse={"indices": 7, "values": 0.5}, declared=True))


async def test_the_keyword_leg_survives_a_real_export_and_import(tmp_path: Path) -> None:
    """Both halves against each other: what the exporter wrote is what the importer stores.

    The source postings are deliberately not what this target would compute from the text,
    so a reader that quietly recomputed would come back with different weights.
    """
    source = SourceAdapter(
        [
            Point(
                point_id=f"c_{index}",
                collection="docs",
                vector=[0.1 * index, 0.2],
                payload={
                    "document_id": "d_1",
                    "source_uri": "/docs/policy.md",
                    "content_hash": "sha-abc",
                    "text": KEYWORD_TEXT,
                },
                sparse=encode_document(KEYWORD_TEXT, k1=0.4, b=0.1),
            )
            for index in range(3)
        ]
    )
    archive = tmp_path / "exported.fragx"
    await export_archive(
        copy_settings(),
        source,  # type: ignore[arg-type]
        collection="docs",
        destination=archive,
        include_vectors=True,
    )
    adapter = await imported(tmp_path, archive, copy_settings())

    written = encode_document(KEYWORD_TEXT, k1=0.4, b=0.1)
    expected = dict(zip(written.indices, written.values, strict=True))
    for point in adapter.points:
        assert point.sparse is not None
        assert dict(zip(point.sparse.indices, point.sparse.values, strict=True)) == pytest.approx(
            expected
        )


def test_a_keyword_leg_declared_under_the_older_major_is_refused(tmp_path: Path) -> None:
    """The declaration is not the guard an old reader sees; the major is, so it must agree."""
    manifest = {
        "format_version": DENSE_ONLY_FORMAT_VERSION,
        "collection": {"name": "docs"},
        "embedding": {"model": "m", "model_version": "1", "dimensions": 2},
        "counts": {"documents": 1, "chunks": 1, "vectors": 1},
        "includes_vectors": True,
        "includes_sparse": True,
    }
    archive = make(
        tmp_path,
        chunks=[{"chunk_id": "c_1", "document_id": "d_1", "text": KEYWORD_TEXT}],
        vectors=[{"chunk_id": "c_1", "vector": [0.1, 0.2], "sparse": CARRIED}],
        manifest=manifest,
    )

    with pytest.raises(VerificationError, match="drops the postings in silence"):
        open_archive(archive)


async def test_a_keyword_leg_does_not_cost_the_streaming_import(tmp_path: Path) -> None:
    """TASK-0302 must not undo TASK-0293: postings are verified and written a row at a time.

    The keyword leg adds a whole pass over `vectors.jsonl` and a second value per chunk, and
    an implementation that indexed either of them would grow with the collection again.
    """
    archive, payload = bulky(tmp_path, with_sparse=True)
    adapter = CountingAdapter()
    settings = copy_settings()

    tracemalloc.start()
    try:
        counts = await import_archive(
            settings,
            adapter,  # type: ignore[arg-type]
            open_archive(archive),
            collection="restored",
            lock_root=tmp_path,
        )
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert settings.retrieval.hybrid is True
    assert counts.chunks == BULK_CHUNKS
    assert peak < payload // 4, f"held {peak} bytes while reading a {payload} byte archive"
