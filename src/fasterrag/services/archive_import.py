"""Portable archive import (D11, ``docs/archive-format.md`` §Import semantics).

Reading an archive is mostly *refusing* to read one. The spec's first rule is that nothing is
written until checksums, manifest counts, and referential integrity all pass, and that
ordering is the whole safety property: a half-imported collection is worse than a failed
import, because it looks like a corpus rather than like an error.

Two write paths, and which one is legal is not the caller's choice alone:

* **Vector copy** reuses the archived vectors. Permitted only when the archive carries them
  *and* the target's embedding model, model version, and dimensions all match the manifest.
  Copying vectors from a different model produces a collection that answers every query
  confidently and wrongly, with nothing in the index to indicate why. The target's identity
  comes from its existing lockfile when it has one and from ``embeddings`` otherwise —
  comparing the manifest against itself, as this did until 2026-11-02, is a guard that can
  never fire.
* **Re-embed** ignores the archived vectors and sends chunk text through the current
  embedding configuration. Always legal, because it derives the vectors it stores.

Import is idempotent by the same mechanism ingestion uses: chunk ids are deterministic, so
importing an archive twice upserts over itself rather than duplicating. That is also the
recovery path when a write fails partway: a collection this import created is dropped, and
one that already existed is named in the error, because re-running completes it.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final, NoReturn

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import CollectionSpec, Point, VectorDBAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.bm25 import encode_document
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.archive import (
    CHECKSUMS_NAME,
    CHUNKS_NAME,
    DOCUMENTS_NAME,
    FORMAT_VERSION,
    LOCK_NAME,
    MANIFEST_NAME,
    VECTORS_NAME,
)
from fasterrag.services.lockfile import LockStore, build_lock, create_lock_store

__all__ = [
    "ArchiveReader",
    "ImportCounts",
    "VerificationError",
    "import_archive",
    "open_archive",
    "supported_major",
]


@dataclass(slots=True)
class ImportCounts:
    """What an import wrote."""

    documents: int = 0
    chunks: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return the machine-readable form."""
        return {"documents": self.documents, "chunks": self.chunks}


_GIBIBYTE: Final = 1024 * 1024 * 1024
MAXIMUM_MEMBER_BYTES: Final = _GIBIBYTE
MAXIMUM_TOTAL_BYTES: Final = 4 * _GIBIBYTE

_REQUIRED_MEMBERS: Final[frozenset[str]] = frozenset(
    {MANIFEST_NAME, CHECKSUMS_NAME, DOCUMENTS_NAME, CHUNKS_NAME, LOCK_NAME}
)

_logger = get_logger(__name__)


class VerificationError(FasterRagError):
    """An archive failed a pre-import check; nothing was written."""

    default_code = ErrorCode.VALIDATION_FAILED


def supported_major(version: str) -> bool:
    """Return whether this build can read an archive of ``version``."""
    try:
        return version.split(".", 1)[0] == FORMAT_VERSION.split(".", 1)[0]
    except (AttributeError, IndexError):
        return False


@dataclass(slots=True)
class ArchiveReader:
    """A verified archive, ready to import."""

    path: Path
    manifest: dict[str, Any]
    members: dict[str, bytes] = field(repr=False, default_factory=dict)

    @property
    def includes_vectors(self) -> bool:
        """Return whether the archive carries vectors."""
        return bool(self.manifest.get("includes_vectors"))

    @property
    def collection(self) -> str:
        """Return the collection name the archive was taken from."""
        collection = self.manifest.get("collection") or {}
        return str(collection.get("name", ""))

    @property
    def embedding(self) -> dict[str, Any]:
        """Return the manifest's embedding block."""
        block = self.manifest.get("embedding") or {}
        return dict(block)

    def documents(self) -> Iterator[dict[str, Any]]:
        """Yield every document row."""
        yield from _rows(self.members.get(DOCUMENTS_NAME, b""))

    def chunks(self) -> Iterator[dict[str, Any]]:
        """Yield every chunk row."""
        yield from _rows(self.members.get(CHUNKS_NAME, b""))

    def vectors(self) -> Iterator[dict[str, Any]]:
        """Yield every vector row, or nothing when the archive carries none."""
        yield from _rows(self.members.get(VECTORS_NAME, b""))

    def require_vector_copy_compatible(
        self, *, model: str, model_version: str | None, dimensions: int | None
    ) -> None:
        """Refuse a vector copy that would land vectors from a different embedding space.

        ``None`` means the target declares no such value, so there is nothing to compare.
        """
        if not self.includes_vectors:
            raise FasterRagError(
                "this archive carries no vectors, so a vector copy is impossible; "
                "re-import with re-embedding enabled",
                code=ErrorCode.CONFLICT,
                retryable=False,
            )

        archived = self.embedding
        mismatches: list[str] = []
        if str(archived.get("model", "")) != model:
            mismatches.append(f"model {archived.get('model')!r} != {model!r}")
        if model_version is not None and str(archived.get("model_version", "")) != model_version:
            mismatches.append(
                f"model_version {archived.get('model_version')!r} != {model_version!r}"
            )
        if dimensions is not None and int(archived.get("dimensions", 0)) != dimensions:
            mismatches.append(f"dimensions {archived.get('dimensions')} != {dimensions}")

        if mismatches:
            raise FasterRagError(
                "the archived vectors were produced by a different embedding configuration: "
                + "; ".join(mismatches)
                + ". Copying them would build a collection that answers confidently and "
                "wrongly; re-import with re-embedding enabled",
                code=ErrorCode.CONFLICT,
                retryable=False,
            )


def _rows(body: bytes) -> Iterator[dict[str, Any]]:
    """Yield each JSONL row, skipping blank lines."""
    for line in body.decode("utf-8").splitlines():
        if line.strip():
            yield json.loads(line)


def _read_members(path: Path) -> dict[str, bytes]:
    """Extract the archive's members.

    Raises:
        VerificationError: If the file is not a readable gzipped tar.
    """
    try:
        with tarfile.open(path, "r:gz") as archive:
            members: dict[str, bytes] = {}
            budget = 0
            for member in archive.getmembers():
                # CRITICAL: refuse any member that is not a plain file with a plain name.
                # A tar may carry `../` paths, absolute paths, symlinks, and devices, and
                # extracting one outside the destination is the classic tar traversal. This
                # reader never writes members to disk, but a name check here means a
                # malicious archive is rejected at the boundary rather than trusted inward.
                if not member.isfile() or Path(member.name).name != member.name:
                    raise VerificationError(
                        f"archive member {member.name!r} is not a plain file at the archive "
                        "root; refusing to read it"
                    )
                if member.size > MAXIMUM_MEMBER_BYTES:
                    raise VerificationError(
                        f"archive member {member.name!r} declares {member.size} bytes, above "
                        f"the {MAXIMUM_MEMBER_BYTES} byte limit for a single member"
                    )
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                # CRITICAL: the running total is what bounds a decompression bomb. gzip
                # reaches ratios past 1000:1, so a small .fragx can name many members that
                # are individually under the per-member ceiling and together exhaust memory
                # long before any checksum is verified. `tarfile` does bound a member read
                # to the declared size, so the header cannot understate the real length —
                # the bounded read below is belt-and-braces and the budget is the guard.
                # This is a ceiling, not a design: import materialises every member at once,
                # and the real fix is to stream it (TASK-0293).
                remaining = MAXIMUM_TOTAL_BYTES - budget
                body = extracted.read(remaining + 1)
                if len(body) > remaining:
                    raise VerificationError(
                        f"{path} expands beyond the {MAXIMUM_TOTAL_BYTES} byte import limit; "
                        "refusing to read further"
                    )
                budget += len(body)
                members[member.name] = body
            return members
    except tarfile.TarError as exc:
        raise VerificationError(f"{path} is not a readable .fragx archive: {exc}") from exc
    except OSError as exc:
        raise VerificationError(f"{path} could not be read: {exc}") from exc


def open_archive(path: Path) -> ArchiveReader:
    """Verify an archive completely and return a reader over it."""
    members = _read_members(path)

    missing = _REQUIRED_MEMBERS - set(members)
    if missing:
        raise VerificationError(
            f"{path} is missing required member(s): {', '.join(sorted(missing))}"
        )

    try:
        manifest = json.loads(members[MANIFEST_NAME])
    except json.JSONDecodeError as exc:
        raise VerificationError(f"{path} has an unreadable {MANIFEST_NAME}: {exc}") from exc

    version = str(manifest.get("format_version", ""))
    if not supported_major(version):
        raise VerificationError(
            f"{path} declares format_version {version!r}; this build reads "
            f"{FORMAT_VERSION.split('.', 1)[0]}.x archives only"
        )

    _verify_checksums(path, members)
    reader = ArchiveReader(path=path, manifest=manifest, members=members)
    _verify_counts(path, reader)
    _verify_references(path, reader)
    _verify_vector_widths(path, reader)

    _logger.info(
        "verified a portable archive",
        extra={
            "path": str(path),
            "collection": reader.collection,
            "includes_vectors": reader.includes_vectors,
            **dict(manifest.get("counts") or {}),
        },
    )
    return reader


def _verify_checksums(path: Path, members: dict[str, bytes]) -> None:
    """Confirm every listed member hashes to its recorded digest."""
    for line in members[CHECKSUMS_NAME].decode("utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        body = members.get(name.strip())
        if body is None:
            raise VerificationError(
                f"{path} lists a checksum for {name.strip()!r}, which the archive does not contain"
            )
        actual = hashlib.sha256(body).hexdigest()
        if actual != digest.strip():
            raise VerificationError(
                f"{path} member {name.strip()!r} failed its checksum; the archive is "
                "corrupt or was modified after export"
            )


def _verify_counts(path: Path, reader: ArchiveReader) -> None:
    """Confirm the manifest's counts match the rows actually present.

    This is what catches a truncated archive. Without it a partial transfer imports as a
    smaller corpus and reports success, and the loss is only visible as worse answers later.
    """
    declared = reader.manifest.get("counts") or {}
    actual = {
        "documents": sum(1 for _ in reader.documents()),
        "chunks": sum(1 for _ in reader.chunks()),
        "vectors": sum(1 for _ in reader.vectors()),
    }

    for name, count in actual.items():
        expected = int(declared.get(name, -1))
        if expected != count:
            raise VerificationError(
                f"{path} declares {expected} {name} but contains {count}; the archive is "
                "truncated or its manifest is wrong"
            )


def _verify_references(path: Path, reader: ArchiveReader) -> None:
    """Confirm every chunk resolves to a document and every vector to a chunk."""
    documents = {str(row.get("document_id", "")) for row in reader.documents()}
    chunks: set[str] = set()

    for row in reader.chunks():
        document_id = str(row.get("document_id", ""))
        if document_id not in documents:
            raise VerificationError(
                f"{path} chunk {row.get('chunk_id')!r} references document "
                f"{document_id!r}, which the archive does not contain"
            )
        chunks.add(str(row.get("chunk_id", "")))

    for row in reader.vectors():
        chunk_id = str(row.get("chunk_id", ""))
        if chunk_id not in chunks:
            raise VerificationError(
                f"{path} has a vector for chunk {chunk_id!r}, which the archive does not contain"
            )


def _verify_vector_widths(path: Path, reader: ArchiveReader) -> None:
    """Confirm every archived vector is as wide as the manifest says.

    The manifest's dimension is what the target collection is created at, so a manifest
    that disagrees with its own rows builds a collection no row fits into.
    """
    declared = int(reader.embedding.get("dimensions", 0) or 0)

    for row in reader.vectors():
        width = len(row.get("vector") or [])
        if declared and width != declared:
            raise VerificationError(
                f"{path} declares {declared} dimensions but the vector for chunk "
                f"{row.get('chunk_id')!r} has {width}; the manifest and the data disagree"
            )
        declared = declared or width


async def import_archive(
    settings: Settings,
    adapter: VectorDBAdapter,
    reader: ArchiveReader,
    *,
    collection: str,
    reembed: bool = False,
    router: TieringRouter | None = None,
    batch_size: int = 64,
    lock_root: Path | None = None,
) -> ImportCounts:
    """Write a verified archive into a collection."""
    if reembed and router is None:
        raise FasterRagError(
            "re-embedding needs an embedding router; none was supplied",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )

    store = create_lock_store(settings, lock_root)
    dimensions = settings.embeddings.dimensions
    if reembed:
        assert router is not None
        model, model_version = router.default.model, router.default.model_version
        dimensions = router.default.dimensions or dimensions
    else:
        archived = reader.embedding
        target_model, target_version, target_width = _target_identity(settings, store, collection)
        reader.require_vector_copy_compatible(
            model=target_model, model_version=target_version, dimensions=target_width
        )
        model = str(archived.get("model", ""))
        model_version = str(archived.get("model_version", ""))
        dimensions = int(archived.get("dimensions", 0)) or dimensions
        if not dimensions:
            first = next(iter(reader.vectors()), None)
            dimensions = len(first["vector"]) if first else None

    if not dimensions:
        raise FasterRagError(
            "the target vector size is unknown, so the collection cannot be created; set "
            "embeddings.dimensions or import with re-embedding enabled",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )

    existing = {info.name for info in await adapter.list_collections()}
    await adapter.create_collection(
        CollectionSpec(
            name=collection,
            dimensions=dimensions,
            distance=settings.vector_db.collection.distance,
            sparse=settings.retrieval.hybrid,
        )
    )

    try:
        counts = await _write_chunks(
            settings,
            adapter,
            reader,
            collection=collection,
            model=model,
            model_version=model_version,
            reembed=reembed,
            router=router,
            batch_size=batch_size,
        )
    except Exception as exc:
        await _abandon(adapter, collection, created=collection not in existing, cause=exc)

    _write_lockfiles(
        settings,
        reader,
        store,
        collection=collection,
        model=model,
        model_version=model_version,
        dimensions=dimensions,
    )
    _logger.info(
        "imported a portable archive",
        extra={
            "path": str(reader.path),
            "collection": collection,
            "reembed": reembed,
            **counts.as_dict(),
        },
    )
    return counts


def _target_identity(
    settings: Settings, store: LockStore, collection: str
) -> tuple[str, str | None, int | None]:
    """Return the model, model version, and width the target will be queried with.

    Its own lockfile when it has one, because that is what actually built it; otherwise
    the configured embedding model, because that is what will query it next.
    """
    lock = store.read(collection)
    if lock is not None:
        return lock.embedding_model, lock.embedding_model_version, lock.dimensions
    return settings.embeddings.model, None, settings.embeddings.dimensions


async def _write_chunks(
    settings: Settings,
    adapter: VectorDBAdapter,
    reader: ArchiveReader,
    *,
    collection: str,
    model: str,
    model_version: str,
    reembed: bool,
    router: TieringRouter | None,
    batch_size: int,
) -> ImportCounts:
    """Upsert every archived chunk, in batches."""
    vectors = {str(row["chunk_id"]): row["vector"] for row in reader.vectors()}
    documents = {str(row.get("document_id", "")): row for row in reader.documents()}
    counts = ImportCounts()
    batch: list[Point] = []

    for row in reader.chunks():
        chunk_id = str(row.get("chunk_id", ""))
        text = str(row.get("text", ""))

        if reembed:
            vector: list[float] = []
        else:
            stored = vectors.get(chunk_id)
            if stored is None:
                raise FasterRagError(
                    f"the archive has no vector for chunk {chunk_id!r}; import with "
                    "re-embedding enabled",
                    code=ErrorCode.CONFLICT,
                    retryable=False,
                )
            vector = [float(value) for value in stored]

        batch.append(
            Point(
                point_id=chunk_id,
                collection=collection,
                vector=vector,
                payload=_restored_payload(
                    row,
                    documents.get(str(row.get("document_id", ""))) or {},
                    model=model,
                    model_version=model_version,
                ),
                sparse=(
                    encode_document(
                        text,
                        k1=settings.retrieval.bm25_k1,
                        b=settings.retrieval.bm25_b,
                    )
                    if settings.retrieval.hybrid
                    else None
                ),
            )
        )

        if len(batch) >= batch_size:
            counts.chunks += await _flush(adapter, batch, router if reembed else None)
            batch = []

    if batch:
        counts.chunks += await _flush(adapter, batch, router if reembed else None)

    counts.documents = len(documents)
    return counts


async def _abandon(
    adapter: VectorDBAdapter, collection: str, *, created: bool, cause: Exception
) -> NoReturn:
    """Undo a half-written import, or say plainly that it could not be undone.

    A collection this import created is dropped; one that already existed is left alone,
    because dropping it would destroy data the archive never carried.
    """
    detail = cause.detail if isinstance(cause, FasterRagError) else str(cause)

    if not created:
        raise FasterRagError(
            f"the import of {collection!r} failed ({detail}) after writing part of the "
            "archive into a collection that already existed, so it now holds a mixture of "
            "old and imported rows; re-running the same import is safe and completes it, "
            "because chunk ids are deterministic and upserts overwrite",
            code=ErrorCode.CONFLICT,
            retryable=False,
        ) from cause

    try:
        await adapter.drop_collection(collection)
    except FasterRagError as drop_failure:
        _logger.error(
            "could not drop the collection a failed import created",
            extra={"collection": collection, "cause": detail, "error": drop_failure.detail},
        )
        raise FasterRagError(
            f"the import of {collection!r} failed ({detail}) and the partly written "
            f"collection could not be dropped ({drop_failure.detail}); it holds an "
            "incomplete corpus and must be removed by hand",
            code=ErrorCode.CONFLICT,
            retryable=False,
        ) from cause

    _logger.warning(
        "dropped the collection a failed import created",
        extra={"collection": collection, "cause": detail},
    )
    raise cause


def _write_lockfiles(
    settings: Settings,
    reader: ArchiveReader,
    store: LockStore,
    *,
    collection: str,
    model: str,
    model_version: str,
    dimensions: int,
) -> None:
    """Record what the import built, and keep the archive's own lock for provenance.

    Without a fresh lock the restored collection reports drift against whatever lock
    happened to be lying around, or none at all, and reindex cannot reason about it.
    """
    if not store.enabled:
        return

    store.write(
        build_lock(
            collection,
            settings,
            embedding_model=model,
            embedding_model_version=model_version,
            dimensions=dimensions,
            document_hashes={
                str(row.get("document_id", "")): str(row.get("content_hash", ""))
                for row in reader.documents()
            },
        )
    )

    archived = reader.members.get(LOCK_NAME)
    if archived is None:
        return
    try:
        store.root.mkdir(parents=True, exist_ok=True)
        (store.root / f"{collection}.source.lock.json").write_bytes(archived)
    except OSError as exc:
        _logger.warning(
            "could not retain the archived lockfile; the import itself is unaffected",
            extra={"collection": collection, "error": str(exc)},
        )


def _restored_payload(
    row: dict[str, Any], document: dict[str, Any], *, model: str, model_version: str
) -> dict[str, Any]:
    """Rebuild the stored payload from an archived chunk row and its document row.

    Provenance lives in ``documents.jsonl``, not on the chunk, so without the join a
    restored chunk has no ``source_uri`` to cite and no ``content_hash`` to dedup on.
    """
    payload: dict[str, Any] = dict(row.get("metadata") or {})
    payload.update(
        {
            "document_id": row.get("document_id", ""),
            "text": row.get("text", ""),
            "span": row.get("span") or {"start": 0, "end": 0},
            "embedding_model": model,
            "embedding_model_version": model_version,
        }
    )
    for name in ("source_uri", "content_hash"):
        if document.get(name):
            payload[name] = document[name]
    if document.get("metadata"):
        payload["document_metadata"] = document["metadata"]
    if row.get("page") is not None:
        payload["page"] = row["page"]
    if row.get("context_prefix") is not None:
        payload["context_prefix"] = row["context_prefix"]
    return payload


async def _flush(adapter: VectorDBAdapter, batch: list[Point], router: TieringRouter | None) -> int:
    """Embed when required, then write one batch.

    Embedding happens per batch rather than per chunk: a provider call for every chunk turns
    an import of a large corpus into a rate-limit incident.
    """
    if router is not None:
        result = await router.default.embed_documents(
            [str(point.payload["text"]) for point in batch]
        )
        batch = [
            replace(point, vector=vector)
            for point, vector in zip(batch, result.vectors, strict=True)
        ]

    written = await adapter.upsert(batch)
    return written.upserted
