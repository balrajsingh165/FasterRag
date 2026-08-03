"""Portable archive import (D11, ``docs/archive-format.md`` §Import semantics).

Reading an archive is mostly *refusing* to read one. The spec's first rule is that nothing is
written until checksums, manifest counts, and referential integrity all pass, and that
ordering is the whole safety property: a half-imported collection is worse than a failed
import, because it looks like a corpus rather than like an error.

Two write paths, and which one is legal is not the caller's choice alone:

* **Vector copy** reuses the archived vectors. Permitted only when the archive carries them
  *and* the target's embedding model, model version, and dimensions all match the manifest.
  Copying vectors from a different model produces a collection that answers every query
  confidently and wrongly, with nothing in the index to indicate why.
* **Re-embed** ignores the archived vectors and sends chunk text through the current
  embedding configuration. Always legal, because it derives the vectors it stores.

Import is idempotent by the same mechanism ingestion uses: chunk ids are deterministic, so
importing an archive twice upserts over itself rather than duplicating.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

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

__all__ = [
    "ArchiveReader",
    "VerificationError",
    "open_archive",
    "supported_major",
]

_REQUIRED_MEMBERS: Final[frozenset[str]] = frozenset(
    {MANIFEST_NAME, CHECKSUMS_NAME, DOCUMENTS_NAME, CHUNKS_NAME, LOCK_NAME}
)

_logger = get_logger(__name__)


class VerificationError(FasterRagError):
    """An archive failed a pre-import check; nothing was written."""

    default_code = ErrorCode.VALIDATION_FAILED


def supported_major(version: str) -> bool:
    """Return whether this build can read an archive of ``version``.

    Majors are refused rather than attempted. A major bump means a field changed meaning, and
    reading it under the old interpretation imports plausible-looking wrong data — which is
    strictly worse than refusing, because nothing reports it.
    """
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
        self, *, model: str, model_version: str, dimensions: int | None
    ) -> None:
        """Refuse a vector copy that would land vectors from a different embedding space.

        Raises:
            FasterRagError: With ``CONFLICT`` naming every field that differs. Named rather
                than summarised, because "incompatible" leaves an operator guessing which of
                three things to change.
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
        if str(archived.get("model_version", "")) != model_version:
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
                extracted = archive.extractfile(member)
                if extracted is not None:
                    members[member.name] = extracted.read()
            return members
    except tarfile.TarError as exc:
        raise VerificationError(f"{path} is not a readable .fragx archive: {exc}") from exc
    except OSError as exc:
        raise VerificationError(f"{path} could not be read: {exc}") from exc


def open_archive(path: Path) -> ArchiveReader:
    """Verify an archive completely and return a reader over it.

    Every check runs before any caller can write: format version, required members, checksums,
    manifest counts, and referential integrity. A failure raises and nothing has been touched.

    Raises:
        VerificationError: With ``VALIDATION_FAILED`` on any failed check, naming which.
    """
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
