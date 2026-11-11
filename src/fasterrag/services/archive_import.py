"""Portable archive import (D11, ``docs/archive-format.md`` §Import semantics).

Reading an archive is mostly *refusing* to read one. The spec's first rule is that nothing is
written until checksums, manifest counts, and referential integrity all pass, and that
ordering is the whole safety property: a half-imported collection is worse than a failed
import, because it looks like a corpus rather than like an error.

**Streamed through a staging directory, not materialised in memory** (TASK-0293). Members are
copied out of the tar in blocks — hashed and charged against the archive's byte budget as they
land — and every later pass reads rows back off disk one line at a time. A collection is
exactly the thing that does not fit in memory, so an import that only worked on corpora small
enough to materialise would be useless on the ones portability exists for. What is still held
is bounded by *row counts*, not archive bytes: the document rows the chunk join needs, which
the lockfile's per-document hashes already require anyway.

Verify-first survives the change because staging is a pass of its own. The digest of each
member is computed as it streams to disk, and it is the staged file — the one that was
hashed — that every verification pass and the writer then read.

**The manifest must be checksummed, and must agree with ``index.lock``** (TASK-0329). Until
this build the manifest was the one member ``checksums.sha256`` never listed, so rewriting
its ``model``/``model_version`` and repacking passed verification and then passed the
vector-copy guard, which reads exactly those fields. Two rules close it: an archive that
does not list ``manifest.json`` is refused outright — unconditionally, because a manifest
cannot be trusted to declare whether it is verified — and a manifest that contradicts the
lockfile it was built from is refused as well, which gives the embedding fields the row to
disagree with that the counts and widths already have. Neither is authenticity: a repacker
who recomputes the checksums and edits the lock to match still passes, and only a signature
would stop that.

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

**The keyword leg is transplanted when the archive carries one** (TASK-0302). A 2.x archive
stores BM25 postings beside each dense vector, and those are written through unchanged, so a
restored collection ranks keyword queries exactly as the source did — the weights keep the
``k1``/``b`` they were indexed with instead of silently adopting the target's. A 1.x archive
carries none, and the old behaviour is what it falls back to: recompute from chunk text under
the target's parameters, because a collection with no keyword leg at all is worse than one
re-derived. ``--reembed`` recomputes on purpose, since that path exists to rebuild a corpus
under the target's configuration and ``bm25_k1``/``bm25_b`` are part of it.

Import is idempotent by the same mechanism ingestion uses: chunk ids are deterministic, so
importing an archive twice upserts over itself rather than duplicating. That is also the
recovery path when a write fails partway: a collection this import created is dropped, and
one that already existed is named in the error, because re-running completes it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import IO, Any, Final, NoReturn

from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import (
    CollectionSpec,
    Point,
    SparseVector,
    VectorDBAdapter,
)
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.bm25 import encode_document
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.archive import (
    CHECKSUMS_NAME,
    CHUNKS_NAME,
    DENSE_ONLY_FORMAT_VERSION,
    DOCUMENTS_NAME,
    LOCK_NAME,
    MANIFEST_NAME,
    READABLE_MAJORS,
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

# CRITICAL: the block size is the import's working set, not a tuning knob. Peak memory
# while staging is a small multiple of it — gzip and the buffered reader each hold a copy —
# and it is what makes the peak independent of how large the archive is.
_BLOCK_BYTES: Final = 256 * 1024

_REQUIRED_MEMBERS: Final[frozenset[str]] = frozenset(
    {MANIFEST_NAME, CHECKSUMS_NAME, DOCUMENTS_NAME, CHUNKS_NAME, LOCK_NAME}
)

_logger = get_logger(__name__)


class VerificationError(FasterRagError):
    """An archive failed a pre-import check; nothing was written."""

    default_code = ErrorCode.VALIDATION_FAILED


def supported_major(version: str) -> bool:
    """Return whether this build can read an archive of ``version``.

    Two majors are readable, not one: 2.x carries the keyword leg and 1.x predates it, and
    refusing the older would strand every archive written before TASK-0302.
    """
    try:
        return version.split(".", 1)[0] in READABLE_MAJORS
    except (AttributeError, IndexError):
        return False


@dataclass(frozen=True, slots=True)
class _ArchivedVector:
    """One vector row's two legs: the dense vector, and the keyword leg if it carried one."""

    vector: list[float]
    sparse: SparseVector | None


@dataclass(slots=True)
class _StagedMember:
    """One member on disk, with the digest computed while it streamed there."""

    path: Path
    digest: str


@dataclass(slots=True)
class ArchiveReader:
    """A verified archive, staged on disk and read one row at a time.

    Owns the staging directory: ``close`` releases it, and so does garbage collection, so a
    caller that forgets leaves a temporary directory rather than a permanent one.
    """

    path: Path
    manifest: dict[str, Any]
    staging: Path
    # CRITICAL: set only for an archive whose vectors are not in chunk order. An aligned
    # archive — everything our own exporter writes — is read in lockstep with the chunk
    # stream and holds nothing, which is the point of TASK-0293. The index is the fallback
    # that keeps an out-of-order archive readable, and it costs memory per vector row.
    vector_offsets: dict[str, int] | None = field(default=None, repr=False)
    workspace: tempfile.TemporaryDirectory[str] | None = field(default=None, repr=False)
    handles: set[IO[bytes]] = field(default_factory=set, repr=False)

    @property
    def includes_vectors(self) -> bool:
        """Return whether the archive carries vectors."""
        return bool(self.manifest.get("includes_vectors"))

    @property
    def includes_sparse(self) -> bool:
        """Return whether the archive carries the keyword leg beside the dense vectors."""
        return bool(self.manifest.get("includes_sparse"))

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

    def member(self, name: str) -> Path | None:
        """Return the staged path of one member, or ``None`` when the archive has none."""
        staged = self.staging / name
        return staged if staged.is_file() else None

    def rows(self, name: str) -> Iterator[dict[str, Any]]:
        """Yield each JSONL row of one staged member, a line at a time, skipping blanks.

        The handle is registered on the reader, because a stream abandoned mid-iteration
        keeps the file open and Windows will not delete a staged member that is.
        """
        path = self.member(name)
        if path is None:
            return
        handle = path.open("rb")
        self.handles.add(handle)
        try:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        finally:
            self.handles.discard(handle)
            handle.close()

    def documents(self) -> Iterator[dict[str, Any]]:
        """Return a stream of every document row, read off disk rather than held."""
        return self.rows(DOCUMENTS_NAME)

    def chunks(self) -> Iterator[dict[str, Any]]:
        """Return a stream of every chunk row, read off disk rather than held."""
        return self.rows(CHUNKS_NAME)

    def vectors(self) -> Iterator[dict[str, Any]]:
        """Return a stream of every vector row, empty when the archive carries none."""
        return self.rows(VECTORS_NAME)

    @contextmanager
    def vector_lookup(self) -> Iterator[Callable[[str], _ArchivedVector | None]]:
        """Open the vectors member once and yield a per-chunk lookup over it.

        The lookup must be called in chunk order: an aligned archive is walked in lockstep
        and holds one row, and only an out-of-order one pays for the offset index.
        """
        staged = self.member(VECTORS_NAME)
        if staged is None:
            yield _no_vectors
            return

        with staged.open("rb") as handle:
            if self.vector_offsets is None:
                yield _lockstep_lookup(handle)
            else:
                yield _seeking_lookup(handle, self.vector_offsets)

    def close(self) -> None:
        """Release the staged members and the directory holding them. Safe to call twice.

        A refusal raised mid-stream leaves its row iterator suspended and its handle open,
        and an open handle is exactly what stops the staging directory being removed.
        """
        for handle in list(self.handles):
            handle.close()
        self.handles.clear()

        workspace, self.workspace = self.workspace, None
        if workspace is None:
            return
        try:
            workspace.cleanup()
        except OSError as exc:
            _logger.warning(
                "could not remove the import staging directory",
                extra={"path": str(self.staging), "error": str(exc)},
            )

    def __enter__(self) -> ArchiveReader:
        """Return the reader, so a caller can scope its staging directory."""
        return self

    def __exit__(self, *details: object) -> None:
        """Release the staging directory however the block ended."""
        self.close()

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


def _no_vectors(_chunk_id: str) -> _ArchivedVector | None:
    """Return nothing, for an archive that carries no vectors at all."""
    return None


def _archived_vector(row: dict[str, Any]) -> _ArchivedVector:
    """Split one staged vector row into its dense and keyword legs."""
    posting = row.get("sparse")
    sparse = (
        SparseVector(
            indices=[int(index) for index in posting.get("indices") or []],
            values=[float(value) for value in posting.get("values") or []],
        )
        if isinstance(posting, dict)
        else None
    )
    dense = [float(value) for value in row.get("vector") or []]
    return _ArchivedVector(vector=dense, sparse=sparse)


def _lockstep_lookup(handle: IO[bytes]) -> Callable[[str], _ArchivedVector | None]:
    """Return a lookup that walks the vectors member in step with the chunk stream.

    Legal because verification established that the vector rows are a subsequence of the
    chunk rows in order; one row is held at a time, whatever the archive's size.
    """
    pending: list[dict[str, Any]] = []

    def lookup(chunk_id: str) -> _ArchivedVector | None:
        if not pending:
            row = _next_row(handle)
            if row is None:
                return None
            pending.append(row)
        if str(pending[0].get("chunk_id", "")) != chunk_id:
            return None
        return _archived_vector(pending.pop())

    return lookup


def _seeking_lookup(
    handle: IO[bytes], offsets: dict[str, int]
) -> Callable[[str], _ArchivedVector | None]:
    """Return a lookup that seeks to a chunk's vector, for an archive out of chunk order."""

    def lookup(chunk_id: str) -> _ArchivedVector | None:
        offset = offsets.get(chunk_id)
        if offset is None:
            return None
        handle.seek(offset)
        row: dict[str, Any] = json.loads(handle.readline())
        return _archived_vector(row)

    return lookup


def _next_row(handle: IO[bytes]) -> dict[str, Any] | None:
    """Return the next non-blank JSONL row, or ``None`` at end of member."""
    for raw in handle:
        if raw.strip():
            row: dict[str, Any] = json.loads(raw)
            return row
    return None


def _stage_members(path: Path, staging: Path) -> dict[str, _StagedMember]:
    """Stream every member of the archive to the staging directory.

    Raises:
        VerificationError: If the file is not a readable gzipped tar, or breaches a ceiling.
    """
    staged: dict[str, _StagedMember] = {}
    budget = MAXIMUM_TOTAL_BYTES

    try:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive:
                # CRITICAL: refuse any member that is not a plain file with a plain name.
                # A tar may carry `../` paths, absolute paths, symlinks, and devices, and
                # extracting one outside the destination is the classic tar traversal. This
                # reader does write members to disk, into a private staging directory, so
                # the name check is what keeps them inside it.
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

                target = staging / member.name
                digest, budget = _drain(path, extracted, target, budget)
                staged[member.name] = _StagedMember(path=target, digest=digest)
    except tarfile.TarError as exc:
        raise VerificationError(f"{path} is not a readable .fragx archive: {exc}") from exc
    except OSError as exc:
        raise VerificationError(f"{path} could not be read: {exc}") from exc

    return staged


def _drain(path: Path, source: IO[bytes], target: Path, budget: int) -> tuple[str, int]:
    """Copy one member to disk in blocks, hashing it and spending the archive's budget.

    Returns the member's digest and what is left of the budget.
    """
    running = hashlib.sha256()

    with target.open("wb") as handle:
        while True:
            # CRITICAL: the running budget is what bounds a decompression bomb, and it is
            # charged against bytes actually read rather than against declared header
            # sizes. gzip reaches ratios past 1000:1, so a small .fragx can name many
            # members that are individually under the per-member ceiling and together
            # exhaust the machine. Reading `budget + 1` is what makes the breach visible
            # on the block that crosses it, before anything larger is ever allocated.
            block = source.read(min(_BLOCK_BYTES, budget + 1))
            if not block:
                break
            if len(block) > budget:
                raise VerificationError(
                    f"{path} expands beyond the {MAXIMUM_TOTAL_BYTES} byte import limit; "
                    "refusing to read further"
                )
            budget -= len(block)
            running.update(block)
            handle.write(block)

    return running.hexdigest(), budget


def open_archive(path: Path) -> ArchiveReader:
    """Verify an archive completely and return a reader streaming over its staged members."""
    workspace = tempfile.TemporaryDirectory(prefix="fasterrag-import-")
    reader = ArchiveReader(
        path=path, manifest={}, staging=Path(workspace.name), workspace=workspace
    )

    try:
        staged = _stage_members(path, reader.staging)
        # CRITICAL: the checksums are verified before the manifest is parsed, not after.
        # Now that the manifest is a checksummed member (TASK-0329), reading it first would
        # report a tampered manifest as a syntax error and never reach the digest that
        # names it as tampering.
        _require_members(path, staged)
        _verify_checksums(path, staged)
        reader.manifest = _verified_manifest(path, staged)
        _verify_counts(path, reader)
        _verify_references(path, reader)
        _verify_vector_widths(path, reader)
        _verify_keyword_leg(path, reader)
        _verify_manifest_matches_lock(path, reader)
    except Exception:
        reader.close()
        raise

    _logger.info(
        "verified a portable archive",
        extra={
            "path": str(path),
            "collection": reader.collection,
            "includes_vectors": reader.includes_vectors,
            "includes_sparse": reader.includes_sparse,
            **dict(reader.manifest.get("counts") or {}),
        },
    )
    return reader


def _require_members(path: Path, staged: dict[str, _StagedMember]) -> None:
    """Confirm the archive carries every member the format requires."""
    missing = _REQUIRED_MEMBERS - set(staged)
    if missing:
        raise VerificationError(
            f"{path} is missing required member(s): {', '.join(sorted(missing))}"
        )


def _verified_manifest(path: Path, staged: dict[str, _StagedMember]) -> dict[str, Any]:
    """Confirm the manifest parses and declares a format major this build can read."""
    try:
        manifest: dict[str, Any] = json.loads(staged[MANIFEST_NAME].path.read_bytes())
    except json.JSONDecodeError as exc:
        raise VerificationError(f"{path} has an unreadable {MANIFEST_NAME}: {exc}") from exc

    version = str(manifest.get("format_version", ""))
    if not supported_major(version):
        raise VerificationError(
            f"{path} declares format_version {version!r}; this build reads format majors "
            f"{', '.join(sorted(READABLE_MAJORS))} only"
        )
    return manifest


def _verify_checksums(path: Path, staged: dict[str, _StagedMember]) -> None:
    """Confirm every listed member hashes to its recorded digest.

    The digests were computed as the members streamed to staging, so this compares the
    bytes that landed on disk — the ones every later pass reads — not a second copy.
    """
    listed = staged[CHECKSUMS_NAME].path.read_text(encoding="utf-8")
    named: set[str] = set()

    for line in listed.splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition("  ")
        member = staged.get(name.strip())
        if member is None:
            raise VerificationError(
                f"{path} lists a checksum for {name.strip()!r}, which the archive does not contain"
            )
        if member.digest != digest.strip():
            raise VerificationError(
                f"{path} member {name.strip()!r} failed its checksum; the archive is "
                "corrupt or was modified after export"
            )
        named.add(name.strip())

    # CRITICAL: the requirement is unconditional, not gated on the version the manifest
    # declares. The manifest is the member being authenticated, so a reader that trusted it
    # to say whether it must be checksummed would accept a stripped line beside a lowered
    # format_version and verify nothing (TASK-0329).
    if MANIFEST_NAME not in named:
        raise VerificationError(
            f"{path} does not list {MANIFEST_NAME} in {CHECKSUMS_NAME}, so its manifest is "
            "unverified; the manifest is what decides whether archived vectors may be "
            "copied, and an unverified one permits a copy into a different embedding "
            "space. Re-export the collection, or append the manifest's digest to "
            f"{CHECKSUMS_NAME}"
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
    """Confirm every chunk resolves to a document and every vector to a chunk.

    The vector check runs in lockstep with the chunk stream and holds one row; an archive
    whose vectors are not in chunk order falls back to an index, built here so that the
    refusal still happens before a single row is written.
    """
    documents = {str(row.get("document_id", "")) for row in reader.documents()}
    stream = reader.vectors()
    pending = next(stream, None)

    for row in reader.chunks():
        chunk_id = str(row.get("chunk_id", ""))
        document_id = str(row.get("document_id", ""))
        if document_id not in documents:
            raise VerificationError(
                f"{path} chunk {row.get('chunk_id')!r} references document "
                f"{document_id!r}, which the archive does not contain"
            )
        if pending is not None and str(pending.get("chunk_id", "")) == chunk_id:
            pending = next(stream, None)

    if pending is None:
        return

    chunks = {str(row.get("chunk_id", "")) for row in reader.chunks()}
    offsets = _vector_offsets(reader)
    for chunk_id in offsets:
        if chunk_id not in chunks:
            raise VerificationError(
                f"{path} has a vector for chunk {chunk_id!r}, which the archive does not contain"
            )
    reader.vector_offsets = offsets


def _vector_offsets(reader: ArchiveReader) -> dict[str, int]:
    """Index each vector row by the byte offset of the line holding it."""
    offsets: dict[str, int] = {}
    staged = reader.member(VECTORS_NAME)
    if staged is None:
        return offsets

    with staged.open("rb") as handle:
        offset = 0
        for raw in handle:
            if raw.strip():
                row: dict[str, Any] = json.loads(raw)
                offsets[str(row.get("chunk_id", ""))] = offset
            offset += len(raw)
    return offsets


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


def _verify_keyword_leg(path: Path, reader: ArchiveReader) -> None:
    """Confirm the postings are well formed and that the manifest owns up to carrying them.

    An undeclared keyword leg is the dangerous shape, because such an archive still claims a
    major that an older reader accepts, and that reader then drops the leg without a word.
    """
    declared = reader.includes_sparse
    major = str(reader.manifest.get("format_version", "")).split(".", 1)[0]
    found = 0

    for row in reader.vectors():
        posting = row.get("sparse")
        if posting is None:
            continue
        found += 1
        _verify_postings(path, str(row.get("chunk_id", "")), posting)

    if found and not declared:
        raise VerificationError(
            f"{path} carries a keyword leg for {found} chunk(s) but its manifest does not "
            "declare includes_sparse; the archive describes itself wrongly"
        )
    if found and major == DENSE_ONLY_FORMAT_VERSION.split(".", 1)[0]:
        raise VerificationError(
            f"{path} carries a keyword leg but declares format_version "
            f"{reader.manifest.get('format_version')!r}; a reader of that major accepts the "
            "archive and drops the postings in silence, so it is refused here"
        )
    if declared and not found:
        raise VerificationError(
            f"{path} declares includes_sparse but no vector row carries a keyword leg; the "
            "archive is truncated or its manifest is wrong"
        )


_LOCKED_FIELDS: Final[tuple[tuple[str, str, str], ...]] = (
    ("embedding", "model", "embedding_model"),
    ("embedding", "model_version", "embedding_model_version"),
    ("embedding", "dimensions", "dimensions"),
    ("chunking", "strategy", "chunker_strategy"),
    ("chunking", "chunk_size", "chunk_size"),
    ("chunking", "overlap", "overlap"),
)


def _verify_manifest_matches_lock(path: Path, reader: ArchiveReader) -> None:
    """Confirm the manifest agrees with the lockfile it was built from.

    ``model`` and ``model_version`` are the fields the vector-copy guard reads and the only
    ones with no row to contradict them; ``index.lock`` is that row (TASK-0329).
    """
    staged = reader.member(LOCK_NAME)
    if staged is None:
        return
    try:
        lock: dict[str, Any] = json.loads(staged.read_bytes() or b"{}")
    except json.JSONDecodeError as exc:
        raise VerificationError(f"{path} has an unreadable {LOCK_NAME}: {exc}") from exc
    if not isinstance(lock, dict) or not lock:
        return

    disagreements = [
        f"{block}.{field} {(reader.manifest.get(block) or {}).get(field)!r} != {lock[key]!r}"
        for block, field, key in _LOCKED_FIELDS
        if lock.get(key) not in (None, "", 0)
        and str((reader.manifest.get(block) or {}).get(field, "")) != str(lock[key])
    ]
    if disagreements:
        raise VerificationError(
            f"{path} has a manifest that contradicts its own {LOCK_NAME}: "
            + "; ".join(disagreements)
            + ". The manifest describes the embedding space a vector copy is checked "
            "against, so an archive whose two self-descriptions disagree is refused "
            "rather than resolved in favour of either"
        )


def _verify_postings(path: Path, chunk_id: str, posting: Any) -> None:
    """Confirm one keyword leg is two equal-length lists with strictly ascending indices.

    A repeated or unordered index has no defined meaning: a backend keeps whichever value it
    happens to see last, so the same postings would score differently on each (TASK-0228).
    """
    if not isinstance(posting, dict):
        raise VerificationError(
            f"{path} chunk {chunk_id!r} has a sparse entry that is not an object"
        )

    indices, values = posting.get("indices"), posting.get("values")
    if not isinstance(indices, list) or not isinstance(values, list):
        raise VerificationError(
            f"{path} chunk {chunk_id!r} has a sparse entry without both an indices and a "
            "values list"
        )
    if len(indices) != len(values):
        raise VerificationError(
            f"{path} chunk {chunk_id!r} has {len(indices)} sparse indices but "
            f"{len(values)} values; the keyword leg is malformed"
        )
    if any(later <= earlier for earlier, later in pairwise(indices)):
        raise VerificationError(
            f"{path} chunk {chunk_id!r} has sparse indices that are not strictly ascending; "
            "a duplicate or unordered term has no defined meaning in a sparse vector"
        )


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
    """Upsert every archived chunk, streaming the rows and batching the writes.

    Document rows are the one thing held: the payload join needs a chunk's document, and
    the chunk stream does not carry it. There are far fewer of them than chunks, and the
    lockfile's per-document hashes already require every one of them in memory.
    """
    documents = {str(row.get("document_id", "")): row for row in reader.documents()}
    counts = ImportCounts()
    batch: list[Point] = []

    with reader.vector_lookup() as vector_for:
        for row in reader.chunks():
            chunk_id = str(row.get("chunk_id", ""))
            text = str(row.get("text", ""))
            carried: SparseVector | None = None

            if reembed:
                vector: list[float] = []
            else:
                stored = vector_for(chunk_id)
                if stored is None:
                    raise FasterRagError(
                        f"the archive has no vector for chunk {chunk_id!r}; import with "
                        "re-embedding enabled",
                        code=ErrorCode.CONFLICT,
                        retryable=False,
                    )
                vector, carried = stored.vector, stored.sparse

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
                    sparse=_keyword_leg(settings, text, carried),
                )
            )

            if len(batch) >= batch_size:
                counts.chunks += await _flush(adapter, batch, router if reembed else None)
                batch = []

        if batch:
            counts.chunks += await _flush(adapter, batch, router if reembed else None)

    counts.documents = len(documents)
    return counts


def _keyword_leg(
    settings: Settings, text: str, carried: SparseVector | None
) -> SparseVector | None:
    """Return the keyword leg to store: the archived postings, else recomputed from the text.

    Carrying them keeps the ``k1``/``b`` the corpus was indexed with; recomputing is what a
    pre-TASK-0302 archive leaves available, and is better than a collection with no leg.
    """
    if not settings.retrieval.hybrid:
        return None
    if carried is not None:
        return carried
    return encode_document(text, k1=settings.retrieval.bm25_k1, b=settings.retrieval.bm25_b)


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

    archived = reader.member(LOCK_NAME)
    if archived is None:
        return
    try:
        store.root.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(archived, store.root / f"{collection}.source.lock.json")
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
