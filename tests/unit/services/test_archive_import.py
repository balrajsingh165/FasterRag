import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.archive import (
    CHECKSUMS_NAME,
    CHUNKS_NAME,
    DOCUMENTS_NAME,
    LOCK_NAME,
    MANIFEST_NAME,
    VECTORS_NAME,
)
from fasterrag.services.archive_import import (
    VerificationError,
    import_archive,
    open_archive,
    supported_major,
)


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
        "format_version": "1.0.0",
        "collection": {"name": "docs"},
        "embedding": {"model": "m", "model_version": "1", "dimensions": 2},
        "counts": {
            "documents": len(documents),
            "chunks": len(chunks),
            "vectors": len(vectors or []),
        },
        "includes_vectors": vectors is not None,
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
        "format_version": "2.0.0",
        "collection": {"name": "docs"},
        "embedding": {},
        "counts": {"documents": 1, "chunks": 1, "vectors": 0},
        "includes_vectors": False,
    }

    with pytest.raises(VerificationError, match="format_version"):
        open_archive(make(tmp_path, manifest=manifest))


def test_a_minor_version_ahead_is_accepted() -> None:
    """Minor additions are additive-only, so a newer minor stays readable."""
    assert supported_major("1.7.0")
    assert not supported_major("2.0.0")
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
    assert "model" in detail
    assert "model_version" in detail
    assert "dimensions" in detail


class RecordingAdapter:
    """Captures what an import writes."""

    def __init__(self) -> None:
        self.created: list[Any] = []
        self.points: list[Any] = []

    async def create_collection(self, spec: Any) -> None:
        self.created.append(spec)

    async def upsert(self, points: list[Any]) -> Any:
        self.points.extend(points)
        from fasterrag.adapters.vectordb.base import UpsertResult

        return UpsertResult(upserted=len(points))


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
        Settings.model_validate({}),
        adapter,  # type: ignore[arg-type]
        reader,
        collection="restored",
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
    )

    assert adapter.points[0].payload["embedding_model"] == "new-model"


async def test_re_embedding_without_a_router_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FasterRagError, match="needs an embedding router"):
        await import_archive(
            Settings.model_validate({}),
            RecordingAdapter(),  # type: ignore[arg-type]
            open_archive(vectored(tmp_path)),
            collection="restored",
            reembed=True,
        )


async def test_a_vector_copy_from_a_vectorless_archive_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FasterRagError) as caught:
        await import_archive(
            Settings.model_validate({}),
            RecordingAdapter(),  # type: ignore[arg-type]
            open_archive(make(tmp_path)),
            collection="restored",
        )

    assert caught.value.code is ErrorCode.CONFLICT


async def test_archived_metadata_survives_the_round_trip(tmp_path: Path) -> None:
    adapter = RecordingAdapter()

    await import_archive(
        Settings.model_validate({}),
        adapter,  # type: ignore[arg-type]
        open_archive(vectored(tmp_path)),
        collection="restored",
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
        Settings.model_validate({}),
        adapter,  # type: ignore[arg-type]
        open_archive(archive),
        collection="restored",
    )

    assert adapter.points[0].payload["text"] == "real body"


async def test_the_collection_is_created_at_the_archived_width(tmp_path: Path) -> None:
    """A collection created at the wrong width fails on the first upsert."""
    adapter = RecordingAdapter()

    await import_archive(
        Settings.model_validate({}),
        adapter,  # type: ignore[arg-type]
        open_archive(vectored(tmp_path)),
        collection="restored",
    )

    assert adapter.created[0].dimensions == 2


async def test_chunk_ids_are_preserved_so_import_is_idempotent(tmp_path: Path) -> None:
    """Deterministic ids are what make a second import upsert rather than duplicate."""
    adapter = RecordingAdapter()

    await import_archive(
        Settings.model_validate({}),
        adapter,  # type: ignore[arg-type]
        open_archive(vectored(tmp_path)),
        collection="restored",
    )

    assert [point.point_id for point in adapter.points] == ["c_1", "c_2"]
