"""Portable archive export (D11, ``docs/archive-format.md``).

Writes a ``.fragx`` — a gzipped tar carrying a collection's documents, chunks, optionally its
vectors, its lockfile, and a manifest describing all of it. Vendor-neutral by construction:
nothing in the archive names the backend that produced it, and ``source_provider`` is
recorded as information only, never as something import may branch on.

**Streamed through a staging directory, not assembled in memory.** Rows are appended to files
on disk as they arrive from the point iterator, and the tar is built from those files. A
collection is exactly the thing that does not fit in memory, and an export that only worked on
corpora small enough to materialise would be useless on the ones portability matters for.

The manifest is written *last*, because it carries the row counts and those are only known
once the rows exist. Import verifies the counts against actual line counts, so a truncated
export is caught rather than imported as a smaller corpus.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from fasterrag import __version__
from fasterrag.adapters.vectordb.base import Point, VectorDBAdapter
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.lockfile import IndexLock

__all__ = [
    "ARCHIVE_SUFFIX",
    "CHECKSUMS_NAME",
    "CHUNKS_NAME",
    "DOCUMENTS_NAME",
    "FORMAT_VERSION",
    "LOCK_NAME",
    "MANIFEST_NAME",
    "VECTORS_NAME",
    "ArchiveCounts",
    "build_manifest",
    "chunk_record",
    "document_record",
    "export_archive",
    "vector_record",
]

FORMAT_VERSION: Final = "1.0.0"
ARCHIVE_SUFFIX: Final = ".fragx"

MANIFEST_NAME: Final = "manifest.json"
CHECKSUMS_NAME: Final = "checksums.sha256"
DOCUMENTS_NAME: Final = "documents.jsonl"
CHUNKS_NAME: Final = "chunks.jsonl"
VECTORS_NAME: Final = "vectors.jsonl"
LOCK_NAME: Final = "index.lock"

_CHUNK_RESERVED: Final[frozenset[str]] = frozenset(
    {
        "document_id",
        "source_uri",
        "content_hash",
        "text",
        "span",
        "page",
        "context_prefix",
        "document_metadata",
    }
)

_logger = get_logger(__name__)


@dataclass(slots=True)
class ArchiveCounts:
    """Row counts the manifest declares and import verifies."""

    documents: int = 0
    chunks: int = 0
    vectors: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return the manifest form."""
        return {"documents": self.documents, "chunks": self.chunks, "vectors": self.vectors}


def document_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Return one ``documents.jsonl`` row from a chunk's stored payload.

    Documents are reconstructed from chunk payloads rather than read from the journal: the
    journal describes an *ingestion run*, while the archive must describe the collection as
    it stands. A collection built by three separate jobs, or restored from a snapshot, has no
    single journal to read — but every chunk in it carries its document's identity.
    """
    return {
        "document_id": str(payload.get("document_id", "")),
        "source_uri": str(payload.get("source_uri", "")),
        "content_hash": str(payload.get("content_hash", "")),
        "version": int(payload.get("version", 1)),
        "metadata": payload.get("document_metadata") or {},
    }


def chunk_record(point: Point) -> dict[str, Any]:
    """Return one ``chunks.jsonl`` row."""
    payload = dict(point.payload)
    span = payload.get("span") or {}

    return {
        "chunk_id": point.point_id,
        "document_id": str(payload.get("document_id", "")),
        "text": str(payload.get("text", "")),
        "span": {"start": int(span.get("start", 0)), "end": int(span.get("end", 0))},
        "page": payload.get("page"),
        # Everything the indexer stored that is not promoted to a named field travels in
        # metadata, so a round trip loses nothing the backend was holding.
        "metadata": {key: value for key, value in payload.items() if key not in _CHUNK_RESERVED},
        "context_prefix": payload.get("context_prefix"),
    }


def vector_record(point: Point) -> dict[str, Any]:
    """Return one ``vectors.jsonl`` row."""
    return {"chunk_id": point.point_id, "vector": [float(value) for value in point.vector]}


def build_manifest(
    settings: Settings,
    *,
    collection: str,
    lock: IndexLock | None,
    counts: ArchiveCounts,
    includes_vectors: bool,
    tenant: str | None,
) -> dict[str, Any]:
    """Return the archive's self-description.

    Embedding and chunking settings come from the *lockfile* when one exists, and from live
    configuration only as a fallback. The lock records what actually built the index; live
    configuration records what would build it today, and importing under the second while
    believing the first is how a vector-copy import lands vectors from the wrong model.
    """
    chunking = settings.chunking
    embeddings = settings.embeddings

    return {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(tz=UTC).isoformat(),
        "fasterrag_version": __version__,
        "collection": {
            "name": collection,
            "distance": settings.vector_db.collection.distance,
        },
        "embedding": {
            "provider": embeddings.provider,
            "model": lock.embedding_model if lock else embeddings.model,
            "model_version": lock.embedding_model_version if lock else "",
            "dimensions": (lock.dimensions if lock else embeddings.dimensions) or 0,
        },
        "chunking": {
            "strategy": lock.chunker_strategy if lock else chunking.strategy,
            "chunk_size": lock.chunk_size if lock else chunking.chunk_size,
            "overlap": lock.overlap if lock else chunking.overlap,
            "contextual_enrichment": (
                lock.contextual_enrichment if lock else chunking.contextual_enrichment
            ),
        },
        "counts": counts.as_dict(),
        "includes_vectors": includes_vectors,
        "source_provider": settings.vector_db.provider,
        "tenant": tenant,
    }


def _line(payload: dict[str, Any]) -> bytes:
    """Serialize one JSONL row, key-sorted so an unchanged collection exports identically."""
    return (json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


async def export_archive(
    settings: Settings,
    adapter: VectorDBAdapter,
    *,
    collection: str,
    destination: Path,
    include_vectors: bool = False,
    lock: IndexLock | None = None,
    tenant: str | None = None,
) -> ArchiveCounts:
    """Write a collection to a portable archive.

    Args:
        settings: Validated configuration, supplying the manifest's fallback values.
        adapter: The backend to read from.
        collection: Collection to export.
        destination: Archive path; ``.fragx`` is appended when absent.
        include_vectors: Write ``vectors.jsonl``, enabling a vector-copy import.
        lock: The collection's lockfile, carried for provenance and manifest accuracy.
        tenant: Recorded in the manifest when multi-tenancy is on.

    Returns:
        The row counts written.

    Raises:
        FasterRagError: With ``NOT_FOUND`` if the collection yields no points. An empty
            archive imports cleanly and produces an empty collection, which is a silent
            data-loss path rather than a portable backup.
    """
    target = (
        destination
        if destination.suffix == ARCHIVE_SUFFIX
        else destination.with_name(f"{destination.name}{ARCHIVE_SUFFIX}")
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix="fasterrag-export-"))
    counts = ArchiveCounts()
    seen_documents: set[str] = set()

    try:
        with (
            (staging / DOCUMENTS_NAME).open("wb") as documents,
            (staging / CHUNKS_NAME).open("wb") as chunks,
            (staging / VECTORS_NAME).open("wb") as vectors,
        ):
            async for point in adapter.iterate_points(collection, with_vectors=include_vectors):
                payload = dict(point.payload)
                document_id = str(payload.get("document_id", ""))
                if document_id and document_id not in seen_documents:
                    seen_documents.add(document_id)
                    documents.write(_line(document_record(payload)))
                    counts.documents += 1

                chunks.write(_line(chunk_record(point)))
                counts.chunks += 1

                if include_vectors and point.vector:
                    vectors.write(_line(vector_record(point)))
                    counts.vectors += 1

        if not counts.chunks:
            raise FasterRagError(
                f"collection {collection!r} yielded no points, so there is nothing to "
                "export; an empty archive would import cleanly and produce an empty "
                "collection",
                code=ErrorCode.NOT_FOUND,
                retryable=False,
            )

        (staging / LOCK_NAME).write_bytes(_line(lock.as_dict()) if lock else b"{}\n")
        if not include_vectors:
            (staging / VECTORS_NAME).unlink()

        manifest = build_manifest(
            settings,
            collection=collection,
            lock=lock,
            counts=counts,
            includes_vectors=include_vectors,
            tenant=tenant,
        )

        names = [DOCUMENTS_NAME, CHUNKS_NAME, LOCK_NAME]
        if include_vectors:
            names.insert(2, VECTORS_NAME)

        digests = {name: _digest(staging / name) for name in names}
        (staging / CHECKSUMS_NAME).write_text(
            "".join(f"{digests[name]}  {name}\n" for name in names), encoding="utf-8"
        )
        (staging / MANIFEST_NAME).write_bytes(_line(manifest))

        _write_tar(target, staging, [*names, CHECKSUMS_NAME, MANIFEST_NAME])
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    _logger.info(
        "wrote a portable archive",
        extra={"path": str(target), "collection": collection, **counts.as_dict()},
    )
    return counts


def _digest(path: Path) -> str:
    """Return a file's SHA-256, read in blocks so a large member is not loaded whole."""
    running = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            running.update(block)
    return running.hexdigest()


def _write_tar(target: Path, staging: Path, names: list[str]) -> None:
    """Pack the staged members into the archive.

    # CRITICAL: every member gets a fixed mtime and owner. Two exports of an unchanged
    # collection must produce byte-identical archives, or every backup diff reports a change
    # that is only a timestamp and nobody can tell a real one from noise.
    """
    with tarfile.open(target, "w:gz") as archive:
        for name in names:
            path = staging / name
            info = archive.gettarinfo(str(path), arcname=name)
            info.mtime = 0
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with path.open("rb") as handle:
                archive.addfile(info, handle)
