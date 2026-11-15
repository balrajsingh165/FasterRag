"""Checkpointed, idempotent ingestion state (D3).

Three capabilities live here, and together they turn ingestion from a job that must
succeed into a resumable, auditable batch system:

* **Checkpoints.** A job record is rewritten every ``ingestion.journal.checkpoint_every``
  documents. A crash mid-ingest of millions of documents resumes from the last checkpoint
  instead of restarting, which is the difference between minutes and days.
* **Deduplication.** Content hashes are remembered per collection, so re-running an
  ingest is a no-op rather than a second copy of the corpus. The file is append-only, and a
  deleted document is a **tombstone** appended to it rather than a line removed: every hash
  remembered for that document *before* the tombstone stops counting, and one remembered
  after it counts again, so removed content can be ingested a second time (TASK-0443).
  A document remembered again under a new hash is the same reading: the hash it held
  before stops counting, because the index no longer holds that version — otherwise a
  document edited and then restored would be skipped as a duplicate of a version that is
  gone.
* **Dead-lettering.** A document that fails its retries is recorded with a machine-readable
  reason code and stays queryable, so a bad file never silently disappears and never stops
  the pipeline.
* **Retention.** With ``ingestion.journal.retention_days`` above zero, :meth:`Journal.prune`
  deletes settled jobs past the window — never one that is queued, running or interrupted,
  which can still be run — and compacts each dedup log to the lines that still count. A hash
  a live document holds is never dropped, however old: it is all that tells a re-ingest the
  document is already indexed (TASK-0474).
* **Per backend.** A dedup log answers for one vector database, named by
  :func:`backend_identity` — the provider and where it lives. Keyed by the collection name
  alone, switching ``vector_db.provider`` made a re-ingest skip every file the old backend
  held, and the new one never received them (TASK-0550). A log written before that change
  sits at the collection's root and is claimed, once, by the first backend that reads it:
  the one the configuration names when the upgraded release first runs.
* **Deletes against running jobs.** A delete appends the document id to the collection's
  deletion log and removes the chunks while holding that log's lock; a job takes a
  :class:`DeletionWatch` under the same lock before it loads its dedup map, and each commit
  reads the log's new lines under it again. A document deleted after the watch was taken is
  not remembered and is recorded ``skipped`` with reason ``DELETED_DURING_INGEST``, and the
  job removes the chunks it wrote, so a delete is never undone by a job that was already
  running (TASK-0519).

Every rewrite of a record is atomic and durable — written to a uniquely named temporary
file, flushed, renamed over the live file, and the directory flushed where the platform
allows — and every job record carries a checksum. The live record is replaced, never moved
away, so a reader or a crash at any step finds a whole record; a torn one is detectable, and
the previous good copy is kept so a damaged record degrades to resuming slightly earlier
rather than to losing the job (``docs/reliability.md`` §3, ``docs/failure-modes.md`` rows 32
and 33, ``docs/adr/ADR-0009-journal-stays-on-files.md``).

``DocumentRecord`` and ``DocumentStatus`` are defined in :mod:`fasterrag.core.journaling` since
2026-11-15 (TASK-0529), beside the ``DocumentJournal`` protocol the parse workers are typed
against, and are re-exported here as the same objects.
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import re
import secrets
import shutil
import sys
import time
from collections.abc import Callable, Collection, Coroutine, Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager, suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any, Final

from fasterrag.config.schema import Settings
from fasterrag.core.identity import job_id as new_job_id
from fasterrag.core.identity import text_hash
from fasterrag.core.journaling import (
    DeletionWatch,
    DocumentRecord,
    DocumentStatus,
    Extent,
    IndexOutcome,
)
from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.observability.logging import get_logger

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

__all__ = [
    "DEFAULT_JOURNAL_ROOT",
    "DELETED_DURING_INGEST",
    "Checkpoint",
    "DeletionWatch",
    "DocumentRecord",
    "DocumentStatus",
    "Extent",
    "IndexOutcome",
    "JobRecord",
    "JobStatus",
    "Journal",
    "PruneReport",
    "QuietJobs",
    "backend_identity",
    "create_journal",
    "envelope",
    "open_envelope",
]

DEFAULT_JOURNAL_ROOT: Final = Path(".fasterrag") / "journal"
DELETED_DURING_INGEST: Final = "DELETED_DURING_INGEST"

_JOBS_DIR: Final = "jobs"
_COLLECTIONS_DIR: Final = "collections"
_DOCUMENTS_FILE: Final = "documents.jsonl"
_DEDUP_FILE: Final = "content-hashes.jsonl"
_BACKENDS_DIR: Final = "backends"
_EXTENTS_FILE: Final = "chunk-extents.jsonl"
_BACKEND_SLUG: Final = re.compile(r"[^a-z0-9_-]+")
_DELETIONS_FILE: Final = "deletions.jsonl"
_PREVIOUS_SUFFIX: Final = ".prev"
_TOMBSTONE_KEY: Final = "tombstone"
_TEMPORARY_SUFFIX: Final = ".tmp"
_BUSY_DELAYS: Final = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
_BUSY_IS_TRANSIENT: Final = sys.platform == "win32"
_DISK_FULL: Final = frozenset({errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)})
_LOCK_SUFFIX: Final = ".lock"
_KEY_INDEX_DIR: Final = "job-keys"
_KEY_INDEX_BUILT: Final = "built.json"
_LOCK_TIMEOUT_SECONDS: Final = 30.0
_LOCK_POLL_SECONDS: Final = 0.002
_SETTLED: Final = frozenset({"completed", "partial", "failed"})
_INTERRUPTED: Final = "INTERRUPTED"
_PRUNE_INTERVAL_SECONDS: Final = 3600.0
_SECONDS_PER_DAY: Final = 86400

JobStatus = str

_logger = get_logger(__name__)


def _now() -> str:
    """Return the current UTC timestamp in ISO 8601."""
    return datetime.now(tz=UTC).isoformat()


@dataclass(frozen=True, slots=True)
class Checkpoint:
    """How far a job got, and when that was recorded."""

    last_document_index: int
    written_at: str


@dataclass
class JobRecord:
    """An ingestion job, as persisted."""

    job_id: str
    collection: str
    status: JobStatus = "queued"
    counts: dict[str, int] = field(default_factory=dict)
    sources: list[dict[str, str]] = field(default_factory=list)
    checkpoint: Checkpoint | None = None
    started_at: str | None = None
    finished_at: str | None = None
    idempotency_key: str | None = None
    tenant: str | None = None
    reason_code: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form."""
        payload = asdict(self)
        payload["checkpoint"] = asdict(self.checkpoint) if self.checkpoint else None
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> JobRecord:
        """Rebuild a job record from its persisted form."""
        checkpoint = payload.get("checkpoint")
        return cls(
            job_id=payload["job_id"],
            collection=payload["collection"],
            status=payload.get("status", "queued"),
            counts=dict(payload.get("counts") or {}),
            sources=list(payload.get("sources") or []),
            checkpoint=Checkpoint(**checkpoint) if checkpoint else None,
            started_at=payload.get("started_at"),
            finished_at=payload.get("finished_at"),
            idempotency_key=payload.get("idempotency_key"),
            tenant=payload.get("tenant"),
            reason_code=payload.get("reason_code"),
        )


@dataclass(frozen=True, slots=True)
class QuietJobs:
    """The unsettled jobs nothing has written to lately, and what the look could not judge."""

    running: tuple[JobRecord, ...] = ()
    recent: int = 0
    unreadable: int = 0
    queued: tuple[JobRecord, ...] = ()


@dataclass(frozen=True, slots=True)
class PruneReport:
    """What one retention pass removed, and how much it had to leave for a later pass."""

    jobs: tuple[str, ...] = ()
    dedup_lines: int = 0
    failed: int = 0


def _modified(path: Path, *, missing: float) -> float:
    """Return when ``path`` was last written, or ``missing`` if it cannot be asked."""
    try:
        return path.stat().st_mtime
    except OSError:
        return missing


def _patiently[T](operation: Callable[[], T]) -> T:
    """Run a file operation, retrying it briefly while the file is reported busy.

    Windows raises ``PermissionError`` for a file another handle is replacing at that
    instant; elsewhere it is a real refusal, which the last attempt raises (TASK-0437).
    """
    for delay in _BUSY_DELAYS:
        try:
            return operation()
        except PermissionError:
            time.sleep(delay)
    return operation()


def _read_text(path: Path) -> str | None:
    """Return a file's text, or None when no such file is on disk.

    Bytes that are not UTF-8 are replaced, so a damaged record fails its checksum like any
    torn write; a file still busy after the retries raises ``PermissionError``.
    """
    try:
        return _patiently(lambda: path.read_text(encoding="utf-8", errors="replace"))
    except (FileNotFoundError, NotADirectoryError):
        return None


def _busy(job: str, exc: PermissionError) -> IngestionError:
    """Return the retryable error for a job record no read could open."""
    return IngestionError(
        f"the journal record for job {job!r} is being rewritten and could not be read: "
        f"{exc}. Retry the request",
        code=ErrorCode.NOT_READY,
    )


def _sync_file(path: Path) -> None:
    """Flush a file's contents to the disk; Windows needs a writable handle to do it."""
    descriptor = os.open(path, os.O_RDWR | getattr(os, "O_BINARY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory(directory: Path) -> None:
    """Make a rename in ``directory`` survive power loss, where the platform allows it.

    Windows cannot open a directory to ``fsync`` it — NTFS journals the rename itself — and
    a POSIX filesystem that refuses (``EINVAL``) is logged, since the rename already happened.
    """
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    except OSError as exc:
        if exc.errno not in (errno.EINVAL, errno.ENOTSUP):
            raise
        _logger.warning(
            "this filesystem cannot flush a directory; a rename may not survive power loss",
            extra={"directory": str(directory), "detail": str(exc)},
        )
    finally:
        os.close(descriptor)


def _stage(path: Path, payload: str, staged: list[Path], *, durable: bool = True) -> Path:
    """Write ``payload`` to a uniquely named temporary file beside ``path``, flushed if durable."""
    temporary = path.with_name(f"{path.name}.{secrets.token_hex(8)}{_TEMPORARY_SUFFIX}")
    staged.append(temporary)
    temporary.write_text(payload, encoding="utf-8")
    if durable:
        _sync_file(temporary)
    return temporary


def _keep_previous(path: Path, staged: list[Path]) -> None:
    """Copy the live record to its previous-good slot, when it is whole.

    A copy rather than a rename, so the live record never leaves; a torn one is not copied.
    Not flushed: the copy is a fallback for a damaged live record, which is flushed itself.
    """
    current = _read_text(path)
    if current is None or open_envelope(current) is None:
        return
    previous = path.with_suffix(path.suffix + _PREVIOUS_SUFFIX)
    copy = _stage(previous, current, staged, durable=False)
    _patiently(lambda: os.replace(copy, previous))


def _make_room(path: Path, staged: list[Path], exc: OSError) -> None:
    """Free what a rewrite on a full disk can spare: its staged files and the previous copy."""
    _discard(staged)
    previous = path.with_suffix(path.suffix + _PREVIOUS_SUFFIX)
    previous.unlink(missing_ok=True)
    _logger.warning(
        "the disk is full; the previous copy of a journal record was dropped to make room",
        extra={"path": str(path), "detail": str(exc)},
    )


def _discard(staged: list[Path]) -> None:
    """Remove temporary files a failed write left behind."""
    for temporary in staged:
        # CRITICAL: the cleanup is itself suppressed, for the reason recorded in
        # TraceStore.save and IndexLockStore.write — on Linux a parent that is a file rather
        # than a directory makes `unlink` raise from inside this handler, replacing the error
        # being reported with a second one that escapes.
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
    staged.clear()


def _write_atomically(
    path: Path, payload: str, *, keep_previous: bool = True, durable: bool = True
) -> None:
    """Replace ``path`` so a reader, or a crash at any step, finds a whole record.

    Stage a flushed, uniquely named file, ``os.replace`` it over the live one and flush the
    directory: the live record is never renamed away, so it is never missing (TASK-0437).
    """
    staged: list[Path] = []
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if keep_previous:
            try:
                _keep_previous(path, staged)
            except OSError as exc:
                if exc.errno not in _DISK_FULL:
                    raise
                _make_room(path, staged, exc)
        fresh = _stage(path, payload, staged, durable=durable)
        _patiently(lambda: os.replace(fresh, path))
        if durable:
            _sync_directory(path.parent)
    except OSError as exc:
        _discard(staged)
        if isinstance(exc, PermissionError) and _BUSY_IS_TRANSIENT:
            raise IngestionError(
                f"the journal record {path.name} stayed busy through a short retry while "
                f"another task replaced it: {exc}. Nothing was lost; retry the request",
                code=ErrorCode.NOT_READY,
            ) from exc
        raise IngestionError(
            f"the ingestion journal could not be written to {path}: {exc}. "
            "The last checkpoint is still readable, so the job can be resumed once the "
            "condition clears",
            code=ErrorCode.INTERNAL,
        ) from exc


def _try_lock(descriptor: int) -> None:
    """Take the exclusive lock on an open lock file, raising at once if it is held."""
    if sys.platform == "win32":
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(descriptor: int) -> None:
    """Release the lock :func:`_try_lock` took."""
    if sys.platform == "win32":
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Hold the cross-process advisory lock for ``path``, kept on a ``.lock`` sidecar.

    A sidecar, because a Windows byte-range lock is mandatory: locking the log itself would
    make every reader of it fail while a writer appended.
    """
    lock = path.with_name(path.name + _LOCK_SUFFIX)
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0), 0o644)
    try:
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                _try_lock(descriptor)
                break
            except (BlockingIOError, PermissionError) as exc:
                if time.monotonic() >= deadline:
                    raise IngestionError(
                        f"the ingestion journal lock {lock} was held by another writer for "
                        f"{_LOCK_TIMEOUT_SECONDS:.0f}s: {exc}. Retry once it is released",
                        code=ErrorCode.NOT_READY,
                    ) from exc
                time.sleep(_LOCK_POLL_SECONDS)
        try:
            yield
        finally:
            _unlock(descriptor)
    finally:
        os.close(descriptor)


def _ends_a_line(path: Path) -> bool:
    """Return whether a non-empty log's last byte is a newline."""
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        return handle.read(1) == b"\n"


def _append_line(path: Path, line: str, tails: dict[Path, int]) -> None:
    """Append one line under the file's lock, translating any OS failure (TASK-0475).

    A fragment a crash left is ended before the line is written; ``tails`` remembers where
    this process's last append ended, so the tail is read only when someone else wrote it.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive(path):
            _append_held(path, line, tails)
    except OSError as exc:
        raise _append_failed(path, exc) from exc


def _append_held(path: Path, line: str, tails: dict[Path, int]) -> None:
    """Append one line to a log whose lock the caller already holds."""
    with path.open("ab") as handle:
        size = os.fstat(handle.fileno()).st_size
        whole = size == 0 or tails.get(path) == size or _ends_a_line(path)
        data = (b"" if whole else b"\n") + line.encode("utf-8") + b"\n"
        handle.write(data)
        handle.flush()
        tails[path] = size + len(data)


def _append_failed(path: Path, exc: OSError) -> IngestionError:
    """Return the error for an append the operating system refused."""
    return IngestionError(
        f"the ingestion journal could not append to {path}: {exc}. The documents after "
        "the last checkpoint are processed again once the condition clears",
        code=ErrorCode.INTERNAL,
    )


def _catch_up(path: Path, watch: DeletionWatch) -> None:
    """Add the documents deleted since ``watch`` was last advanced, reading only new lines."""
    try:
        size = path.stat().st_size
    except FileNotFoundError:
        size = 0
    if size < watch.offset:
        watch.offset = 0
    if size == watch.offset:
        return
    with path.open("rb") as handle:
        handle.seek(watch.offset)
        data = handle.read(size - watch.offset)
    complete = data.rfind(b"\n") + 1
    for line in data[:complete].splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            _logger.warning(
                "skipping a deletion log line a crash left unfinished", extra={"path": str(path)}
            )
            continue
        if isinstance(payload, dict) and isinstance(payload.get("document_id"), str):
            watch.deleted.add(payload["document_id"])
    watch.offset += complete


def _logged_lines(path: Path) -> list[dict[str, Any]]:
    """Return the parsed lines of an append-only log, skipping a fragment a crash left."""
    try:
        text = _patiently(lambda: path.read_text(encoding="utf-8", errors="replace"))
    except (FileNotFoundError, NotADirectoryError):
        return []
    parsed: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError as exc:
            _logger.warning(
                "skipping a journal line a crash left unfinished",
                extra={"path": str(path), "line": number, "detail": str(exc)},
            )
    return parsed


def _sync_if_present(path: Path) -> None:
    """Flush a log to the disk if it exists, so a checkpoint never outruns what it covers."""
    if path.is_file():
        _sync_file(path)


def _live_content(lines: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Return the hash to document-id map an ordered dedup log means (tombstones applied)."""
    known: dict[str, str] = {}
    _apply_dedup(lines, known, {})
    return known


def _apply_dedup(
    lines: Sequence[dict[str, Any]], known: dict[str, str], held: dict[str, str]
) -> None:
    """Fold dedup log lines, in order, into a hash map and its document-to-hash inverse."""
    for payload in lines:
        document = payload["document_id"]
        superseded = held.pop(document, None)
        if superseded is not None and known.get(superseded) == document:
            del known[superseded]
        if payload.get(_TOMBSTONE_KEY):
            continue
        held[document] = payload["content_hash"]
        known[held[document]] = document


@dataclass(slots=True)
class _DedupView:
    """What one dedup log meant up to ``offset`` bytes, for the file ``identity`` names."""

    identity: tuple[int, int]
    offset: int = 0
    known: dict[str, str] = field(default_factory=dict)
    held: dict[str, str] = field(default_factory=dict)


def _new_lines(path: Path, view: _DedupView, size: int) -> list[dict[str, Any]]:
    """Return the whole lines appended past ``view.offset``, advancing it over them."""
    with path.open("rb") as handle:
        handle.seek(view.offset)
        data = handle.read(size - view.offset)
    complete = data.rfind(b"\n") + 1
    parsed: list[dict[str, Any]] = []
    for line in data[:complete].splitlines():
        if not line.strip():
            continue
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            _logger.warning(
                "skipping a journal line a crash left unfinished", extra={"path": str(path)}
            )
    view.offset += complete
    return parsed


def _live_extents(lines: Sequence[dict[str, Any]]) -> dict[str, list[Extent]]:
    """Return each document's chunk extents since its last settled one, that one first."""
    extents: dict[str, list[Extent]] = {}
    for payload in lines:
        extent = Extent(str(payload["chunker"]), int(payload["chunks"]), bool(payload["settled"]))
        if extent.settled:
            extents[payload["document_id"]] = [extent]
        else:
            extents.setdefault(payload["document_id"], []).append(extent)
    return extents


def _extent_line(document: str, chunker: str, chunks: int, *, settled: bool) -> str:
    """Return the log line recording that a document has, or may have, these chunks."""
    return json.dumps(
        {"chunker": chunker, "chunks": chunks, "document_id": document, "settled": settled},
        sort_keys=True,
    )


def _dedup_line(content: str, document: str) -> str:
    """Return the dedup log line that remembers ``content`` for ``document``."""
    return json.dumps({"content_hash": content, "document_id": document})


def backend_identity(settings: Settings) -> str:
    """Return the name a dedup log is kept under: the vector database's provider and location.

    Only names, never a secret: pgvector is identified by the variable holding its DSN.
    """
    vector_db = settings.vector_db
    location: list[object]
    if vector_db.provider == "chroma":
        location = [vector_db.chroma.path]
    elif vector_db.provider == "pgvector":
        location = [vector_db.pgvector.dsn_env, vector_db.pgvector.db_schema]
    else:
        location = [vector_db.host, vector_db.port]
    slug = _BACKEND_SLUG.sub("-", vector_db.provider.lower()).strip("-")[:32] or "backend"
    return f"{slug}-{text_hash(json.dumps([vector_db.provider, location]))[:12]}"


def _compacted(path: Path, lines: Sequence[dict[str, Any]]) -> list[str]:
    """Return the lines of a dedup or chunk-extent log that still mean something."""
    if path.name == _EXTENTS_FILE:
        return [
            _extent_line(document, extent.chunker, extent.chunks, settled=extent.settled)
            for document, extents in _live_extents(lines).items()
            for extent in extents
        ]
    return [_dedup_line(c, d) for c, d in _live_content(lines).items()]


def _finished_before(record: JobRecord, cutoff: float) -> bool:
    """Return whether a settled job finished before ``cutoff``; an unreadable time is old."""
    if record.finished_at is None:
        return True
    try:
        return datetime.fromisoformat(record.finished_at).timestamp() < cutoff
    except ValueError as exc:
        _logger.warning(
            "a job's finish time cannot be read; the age of its record decides its retention",
            extra={"job_id": record.job_id, "detail": str(exc)},
        )
        return True


def _key_entry_name(key: str, tenant: str | None) -> str:
    """Return the index file name for a tenant's key, hashed so no key escapes the directory."""
    return f"{text_hash(json.dumps([tenant, key]))}.json"


def envelope(payload: dict[str, Any]) -> str:
    """Wrap a payload with a checksum so a torn write is detectable."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return json.dumps({"checksum": text_hash(body), "body": body}, separators=(",", ":"))


def open_envelope(text: str) -> dict[str, Any] | None:
    """Return the payload if its checksum matches, otherwise None."""
    try:
        wrapper = json.loads(text)
        body = wrapper["body"]
        if text_hash(body) != wrapper["checksum"]:
            return None
        loaded = json.loads(body)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

    return loaded if isinstance(loaded, dict) else None


class Journal:
    """Durable ingestion state: jobs, per-document outcomes, and dedup hashes.

    Every operation that touches a file has two faces: an ``async`` one, which is what
    callers use, and the ``*_blocking`` twin it runs in a worker thread (TASK-0399).
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_JOURNAL_ROOT,
        *,
        checkpoint_every: int = 100,
        enabled: bool = True,
        retention_days: int = 0,
        backend: str | None = None,
    ) -> None:
        """Open a journal rooted at ``root``; ``retention_days`` of 0 keeps every job.

        ``backend`` (see :func:`backend_identity`) scopes dedup logs; None keeps one per collection.
        """
        self.root = Path(root)
        self.checkpoint_every = checkpoint_every
        self.enabled = enabled
        self.retention_days = retention_days
        self.backend = backend
        self._claimed: set[str] = set()
        self._views: dict[Path, _DedupView] = {}
        self._tails: dict[Path, int] = {}
        self._last_pruned: float | None = None

    def _job_path(self, job: str) -> Path:
        """Return the path of a job record."""
        return self.root / _JOBS_DIR / f"{job}.json"

    def _documents_path(self, job: str) -> Path:
        """Return the path of a job's per-document log."""
        return self.root / _JOBS_DIR / job / _DOCUMENTS_FILE

    def _deletions_path(self, collection: str) -> Path:
        """Return the path of a collection's deletion log (TASK-0519)."""
        return self.root / _COLLECTIONS_DIR / collection / _DELETIONS_FILE

    def _dedup_path(self, collection: str) -> Path:
        """Return the path of a collection's content-hash index for this journal's backend."""
        directory = self.root / _COLLECTIONS_DIR / collection
        if self.backend is None:
            return directory / _DEDUP_FILE
        path = directory / _BACKENDS_DIR / self.backend / _DEDUP_FILE
        if collection not in self._claimed:
            self._claim_legacy(directory / _DEDUP_FILE, path)
            self._claimed.add(collection)
        return path

    def _claim_legacy(self, legacy: Path, path: Path) -> None:
        """Move a dedup log written before logs were kept per backend to this backend's place.

        Under the old log's lock, so an append from a process still on that path is not lost.
        """
        if not legacy.is_file() or path.exists():
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with _exclusive(legacy):
                if not legacy.is_file() or path.exists():
                    return
                _patiently(partial(os.replace, legacy, path))
        except OSError as exc:
            raise IngestionError(
                f"the dedup log {legacy} could not be moved to {path}: {exc}",
                code=ErrorCode.INTERNAL,
            ) from exc
        _logger.info(
            "a dedup log written before logs were kept per backend now belongs to the "
            "configured vector database",
            extra={"path": str(path), "backend": self.backend},
        )

    def _deletion_lock(self, collection: str) -> AbstractContextManager[None]:
        """Return the lock that orders a collection's deletes against its commits (TASK-0519)."""
        path = self._deletions_path(collection)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IngestionError(
                f"the ingestion journal could not create {path.parent}: {exc}",
                code=ErrorCode.INTERNAL,
            ) from exc
        return _exclusive(path)

    async def create_job(
        self,
        collection: str,
        sources: Sequence[dict[str, str]],
        *,
        idempotency_key: str | None = None,
        tenant: str | None = None,
    ) -> JobRecord:
        """Create and persist a job, off the event loop.

        One thread hop covers the key lookup and the write together, so the deprecated
        body-field path costs a request its scan but costs the worker nothing.
        """
        return await asyncio.to_thread(
            self.create_job_blocking,
            collection,
            sources,
            idempotency_key=idempotency_key,
            tenant=tenant,
        )

    def create_job_blocking(
        self,
        collection: str,
        sources: Sequence[dict[str, str]],
        *,
        idempotency_key: str | None = None,
        tenant: str | None = None,
    ) -> JobRecord:
        """Create and persist a job; :meth:`create_job` is this off the event loop.

        Replaying an idempotency key returns the original job and creates nothing, so a
        retried submission never starts a second ingest of the same corpus.
        """
        record = JobRecord(
            job_id=new_job_id(),
            collection=collection,
            status="queued",
            sources=[dict(source) for source in sources],
            started_at=_now(),
            idempotency_key=idempotency_key,
            tenant=tenant,
        )
        if idempotency_key is None:
            self.save_job_blocking(record)
            return record

        with self._key_index_lock():
            existing = self._find_indexed(idempotency_key, tenant)
            if existing is not None:
                _logger.info(
                    "returning the existing job for a replayed idempotency key",
                    extra={"job_id": existing.job_id},
                )
                return existing
            self._index_key(record, idempotency_key)
            self.save_job_blocking(record)
        return record

    async def save_job(self, record: JobRecord) -> None:
        """Persist a job record atomically, off the event loop."""
        await asyncio.to_thread(self.save_job_blocking, record)

    def save_job_blocking(self, record: JobRecord) -> None:
        """Persist a job record atomically; :meth:`save_job` is this off the event loop."""
        _write_atomically(self._job_path(record.job_id), envelope(record.as_dict()))

    async def load_job(self, job: str, *, tenant: str | None = None) -> JobRecord:
        """Load a job record off the event loop."""
        return await asyncio.to_thread(self.load_job_blocking, job, tenant=tenant)

    def load_job_blocking(self, job: str, *, tenant: str | None = None) -> JobRecord:
        """Load a job record, falling back to the previous good copy.

        A live record that is missing, busy or torn is read from its previous copy; a job
        is unknown only when neither is on disk, and busy when neither opens (TASK-0437).
        """
        path = self._job_path(job)
        busy: PermissionError | None = None
        try:
            live = _read_text(path)
        except PermissionError as exc:
            live, busy = None, exc
        payload = open_envelope(live) if live is not None else None
        if payload is None:
            try:
                kept = _read_text(path.with_suffix(path.suffix + _PREVIOUS_SUFFIX))
            except PermissionError as exc:
                raise _busy(job, exc) from exc
            if live is None and kept is None:
                if busy is not None:
                    raise _busy(job, busy) from busy
                raise IngestionError(f"unknown ingest job {job!r}", code=ErrorCode.NOT_FOUND)

            payload = open_envelope(kept) if kept is not None else None
            if payload is None:
                raise IngestionError(
                    f"the journal record for job {job!r} is corrupt and has no readable "
                    "previous copy",
                    code=ErrorCode.CHUNK_FAILED,
                    retryable=False,
                )
            _logger.warning(
                "journal record is missing or failed its checksum; resuming from the "
                "previous checkpoint",
                extra={
                    "job_id": job,
                    "live_record": "busy" if busy else "missing" if live is None else "torn",
                },
            )

        record = JobRecord.from_dict(payload)
        if tenant is not None and record.tenant != tenant:
            raise IngestionError(f"unknown ingest job {job!r}", code=ErrorCode.NOT_FOUND)
        return record

    async def find_by_idempotency_key(
        self, key: str, *, tenant: str | None = None
    ) -> JobRecord | None:
        """Return the job ``tenant`` created with ``key``, off the event loop."""
        return await asyncio.to_thread(self.find_by_idempotency_key_blocking, key, tenant=tenant)

    def find_by_idempotency_key_blocking(
        self, key: str, *, tenant: str | None = None
    ) -> JobRecord | None:
        """Return the job ``tenant`` created with ``key``, if one exists.

        Scoped to the tenant (TASK-0318), and answered from the ``job-keys/`` index: one
        entry and one record, whatever the number of jobs (TASK-0437).
        """
        if not (self.root / _JOBS_DIR).is_dir():
            return None
        with self._key_index_lock():
            return self._find_indexed(key, tenant)

    def _key_index_lock(self) -> AbstractContextManager[None]:
        """Return the lock that makes a keyed lookup and the job it creates one step."""
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IngestionError(
                f"the ingestion journal root {self.root} cannot be created: {exc}",
                code=ErrorCode.INTERNAL,
            ) from exc
        return _exclusive(self.root / _KEY_INDEX_DIR)

    def _find_indexed(self, key: str, tenant: str | None) -> JobRecord | None:
        """Return the job the index names for ``(tenant, key)``, under the index lock.

        An index that was never built, or an entry that names the wrong record, is rebuilt
        from a full scan once, and the lookup asked again.
        """
        if not (self.root / _KEY_INDEX_DIR / _KEY_INDEX_BUILT).is_file():
            self._rebuild_key_index()
        found, stale = self._indexed_job(key, tenant)
        if stale:
            self._rebuild_key_index()
            found, _ = self._indexed_job(key, tenant)
        return found

    def _indexed_job(self, key: str, tenant: str | None) -> tuple[JobRecord | None, bool]:
        """Return the job the index entry names and whether the entry was stale."""
        try:
            text = _read_text(self.root / _KEY_INDEX_DIR / _key_entry_name(key, tenant))
        except PermissionError as exc:
            raise _busy(key, exc) from exc
        if text is None:
            return None, False
        entry = open_envelope(text)
        job = entry.get("job_id") if entry is not None else None
        if not isinstance(job, str):
            return None, True
        try:
            record = self.load_job_blocking(job)
        except IngestionError as exc:
            if exc.code is ErrorCode.NOT_READY:
                raise
            _logger.warning(
                "an idempotency index entry names a job that cannot be read; rebuilding",
                extra={"job_id": job, "detail": exc.detail},
            )
            return None, True
        if record.idempotency_key != key or record.tenant != tenant:
            return None, True
        return record, False

    def _index_key(self, record: JobRecord, key: str, *, durable: bool = True) -> None:
        """Point the index at ``record``, before the record exists.

        In that order an interruption leaves an entry naming no job, which a lookup sees
        as stale and rebuilds; the other order would leave a job a replay cannot find.
        """
        entry = {"idempotency_key": key, "job_id": record.job_id, "tenant": record.tenant}
        name = _key_entry_name(key, record.tenant)
        _write_atomically(
            self.root / _KEY_INDEX_DIR / name, envelope(entry), keep_previous=False, durable=durable
        )

    def _rebuild_key_index(self) -> None:
        """Rebuild the idempotency index from every job record — the slow path, taken once.

        The first record holding a key in name order wins, as in the scan this replaced.
        Entries are not flushed one by one: an fsync each made 10,000 jobs take minutes.
        """
        started = time.monotonic()
        wanted: dict[str, tuple[JobRecord, str]] = {}
        for path in sorted((self.root / _JOBS_DIR).glob("*.json")):
            record = self._record_for_index(path.stem)
            if record is not None and record.idempotency_key is not None:
                key = record.idempotency_key
                wanted.setdefault(_key_entry_name(key, record.tenant), (record, key))
        index = self.root / _KEY_INDEX_DIR
        for entry in index.glob("*.json"):
            if entry.name not in wanted and entry.name != _KEY_INDEX_BUILT:
                _patiently(partial(entry.unlink, missing_ok=True))
        for record, key in wanted.values():
            self._index_key(record, key, durable=False)
        built = {"built_at": _now(), "entries": len(wanted)}
        _write_atomically(index / _KEY_INDEX_BUILT, envelope(built), keep_previous=False)
        _logger.info(
            "rebuilt the idempotency key index from a full scan of the job records",
            extra={"entries": len(wanted), "seconds": round(time.monotonic() - started, 3)},
        )

    def _record_for_index(self, job: str) -> JobRecord | None:
        """Return a job record for the index rebuild, or None when it cannot be read."""
        try:
            return self.load_job_blocking(job)
        except IngestionError as exc:
            if exc.code is ErrorCode.NOT_READY:
                raise
            _logger.warning(
                "a job record could not be read while rebuilding the idempotency index",
                extra={"job_id": job, "detail": exc.detail},
            )
            return None

    async def quiet_running_jobs(self, quiet_seconds: float) -> QuietJobs:
        """Return the unsettled jobs not written to for ``quiet_seconds``, off the loop."""
        return await asyncio.to_thread(self.quiet_running_jobs_blocking, quiet_seconds)

    def quiet_running_jobs_blocking(self, quiet_seconds: float) -> QuietJobs:
        """Find quiet ``running`` and ``queued`` jobs; :meth:`quiet_running_jobs` off the loop.

        A record written inside the window is counted and never opened: something is
        still writing it, and reading it could collide with that write (TASK-0526).
        """
        directory = self.root / _JOBS_DIR
        if not directory.is_dir():
            return QuietJobs()

        cutoff = time.time() - quiet_seconds
        running: list[JobRecord] = []
        queued: list[JobRecord] = []
        recent = unreadable = 0
        for path in sorted(directory.glob("*.json")):
            if self._last_written(path.stem) > cutoff:
                recent += 1
                continue

            payload = self._read_quiet_record(path)
            if payload is None:
                unreadable += 1
            elif payload.get("status") == "running":
                running.append(JobRecord.from_dict(payload))
            elif payload.get("status") == "queued":
                queued.append(JobRecord.from_dict(payload))

        return QuietJobs(
            running=tuple(running), recent=recent, unreadable=unreadable, queued=tuple(queued)
        )

    def _last_written(self, job: str) -> float:
        """Return when a job's record or its document log was last written.

        A record that cannot be asked is being replaced at this instant, so it counts as
        written now; a job with no document log yet simply has no second clock.
        """
        return max(
            _modified(self._job_path(job), missing=time.time()),
            _modified(self._documents_path(job), missing=0.0),
        )

    def _read_quiet_record(self, path: Path) -> dict[str, Any] | None:
        """Return a job record's payload, or None when it is torn or cannot be opened."""
        try:
            text = _read_text(path)
        except OSError as exc:
            _logger.warning(
                "a job record could not be read while looking for interrupted jobs",
                extra={"job_id": path.stem, "detail": str(exc)},
            )
            return None
        return open_envelope(text) if text is not None else None

    async def checkpoint(self, record: JobRecord, document_index: int) -> JobRecord:
        """Record progress off the event loop, writing only at the configured interval.

        An interval that is not due is answered from the record alone, so the documents
        between checkpoints pay no thread hop at all.
        """
        if not self._checkpoint_due(document_index):
            return record
        return await asyncio.to_thread(self.checkpoint_blocking, record, document_index)

    def checkpoint_blocking(self, record: JobRecord, document_index: int) -> JobRecord:
        """Record progress; :meth:`checkpoint` is this off the event loop."""
        if not self._checkpoint_due(document_index):
            return record

        updated = replace(
            record,
            checkpoint=Checkpoint(last_document_index=document_index, written_at=_now()),
        )
        self._sync_logs(record)
        self.save_job_blocking(updated)
        return updated

    def _sync_logs(self, record: JobRecord) -> None:
        """Flush the job's document log and its collection's dedup file before a checkpoint.

        Appends are not flushed one by one; this makes everything a checkpoint covers durable
        before the checkpoint claims it (ADR-0009).
        """
        try:
            _sync_if_present(self._documents_path(record.job_id))
            _sync_if_present(self._dedup_path(record.collection))
        except OSError as exc:
            raise IngestionError(
                f"the ingestion journal could not flush job {record.job_id!r} to disk: {exc}",
                code=ErrorCode.INTERNAL,
            ) from exc

    def _checkpoint_due(self, document_index: int) -> bool:
        """Return whether this document index is one a checkpoint is written at."""
        if not self.enabled:
            return False
        return (document_index + 1) % self.checkpoint_every == 0

    def resume_index(self, record: JobRecord) -> int:
        """Return the document index to resume from after a crash."""
        if record.checkpoint is None:
            return 0
        return record.checkpoint.last_document_index + 1

    async def record_document(self, job: str, record: DocumentRecord) -> None:
        """Append a document outcome to the job's log, off the event loop."""
        await asyncio.to_thread(self.record_document_blocking, job, record)

    def record_document_blocking(self, job: str, record: DocumentRecord) -> None:
        """Append a document outcome; :meth:`record_document` is this off the loop."""
        line = json.dumps(asdict(record), sort_keys=True, separators=(",", ":"))
        _append_line(self._documents_path(job), line, self._tails)

    async def record_indexed(
        self,
        job: str,
        collection: str,
        record: DocumentRecord,
        *,
        dedup: bool,
        watch: DeletionWatch | None = None,
    ) -> IndexOutcome:
        """Claim a parsed document's hash and log it indexed, in one hop off the event loop.

        One hop, because the thread handoff costs more than either append (BENCH-0006). The
        outcome says whether it was deleted since ``watch`` (TASK-0519) or a late duplicate.
        """
        return await asyncio.to_thread(
            self.record_indexed_blocking, job, collection, record, dedup=dedup, watch=watch
        )

    def record_indexed_blocking(
        self,
        job: str,
        collection: str,
        record: DocumentRecord,
        *,
        dedup: bool,
        watch: DeletionWatch | None = None,
    ) -> IndexOutcome:
        """Claim a hash and log the document; :meth:`record_indexed` is this off the loop.

        A document deleted after ``watch`` was taken is logged skipped, not remembered, and
        its chunks are the caller's to remove; so are a late duplicate's (TASK-0471).
        """
        if watch is None:
            return self._record_indexed(job, collection, record, dedup=dedup)
        with self._deletion_lock(collection):
            _catch_up(self._deletions_path(collection), watch)
            if record.document_id not in watch.deleted:
                return self._record_indexed(job, collection, record, dedup=dedup)
            self.record_document_blocking(
                job,
                replace(
                    record,
                    status="skipped",
                    reason_code=DELETED_DURING_INGEST,
                    detail=f"{record.source} was deleted from {collection} while this job ran",
                ),
            )
            self.forget_document_blocking(collection, record.document_id)
        return IndexOutcome(deleted=True)

    def _record_indexed(
        self, job: str, collection: str, record: DocumentRecord, *, dedup: bool
    ) -> IndexOutcome:
        """Claim the hash under the dedup lock, then log the document; no deletion check."""
        if dedup and record.content_hash is not None:
            holder = self.claim_content_blocking(
                collection, record.content_hash, record.document_id
            )
            if holder is not None:
                return IndexOutcome(holder=holder)
        self.record_document_blocking(job, record)
        return IndexOutcome()

    async def dead_letter(
        self,
        job: str,
        *,
        document: str,
        source: str,
        reason_code: ErrorCode,
        detail: str,
        attempts: int,
        trace_id: str | None = None,
        content: str | None = None,
    ) -> DocumentRecord:
        """Record a document that failed its retries, off the event loop."""
        return await asyncio.to_thread(
            self.dead_letter_blocking,
            job,
            document=document,
            source=source,
            reason_code=reason_code,
            detail=detail,
            attempts=attempts,
            trace_id=trace_id,
            content=content,
        )

    def dead_letter_blocking(
        self,
        job: str,
        *,
        document: str,
        source: str,
        reason_code: ErrorCode,
        detail: str,
        attempts: int,
        trace_id: str | None = None,
        content: str | None = None,
    ) -> DocumentRecord:
        """Record a failed document; :meth:`dead_letter` is this off the event loop."""
        timestamp = _now()
        record = DocumentRecord(
            document_id=document,
            source=source,
            status="dead_lettered",
            content_hash=content,
            reason_code=reason_code.value,
            detail=detail,
            attempts=attempts,
            trace_id=trace_id,
            first_failed_at=timestamp,
            last_failed_at=timestamp,
        )
        self.record_document_blocking(job, record)
        _logger.warning(
            "document dead-lettered",
            extra={
                "job_id": job,
                "document_id": document,
                "reason_code": reason_code.value,
                "attempts": attempts,
            },
        )
        return record

    async def documents(self, job: str, *, status: str | None = None) -> list[DocumentRecord]:
        """Return a job's document outcomes off the event loop, optionally filtered."""
        return await asyncio.to_thread(self.documents_blocking, job, status=status)

    def documents_blocking(self, job: str, *, status: str | None = None) -> list[DocumentRecord]:
        """Return a job's document outcomes; :meth:`documents` is this off the loop.

        A list rather than a generator: a lazy read would do its file I/O in whichever
        thread consumed it, which is the event loop again.
        """
        found: list[DocumentRecord] = []
        for payload in _logged_lines(self._documents_path(job)):
            record = DocumentRecord(**payload)
            if status is None or record.status == status:
                found.append(record)
        return found

    async def dead_lettered(self, job: str) -> list[DocumentRecord]:
        """Return the job's dead-letter entries, off the event loop."""
        return await asyncio.to_thread(self.dead_lettered_blocking, job)

    def dead_lettered_blocking(self, job: str) -> list[DocumentRecord]:
        """Return the job's dead-letter entries; :meth:`dead_lettered` is this off the loop."""
        return self.documents_blocking(job, status="dead_lettered")

    async def remember_content(self, collection: str, content: str, document: str) -> None:
        """Record that a content hash is indexed in a collection, off the event loop."""
        await asyncio.to_thread(self.remember_content_blocking, collection, content, document)

    def remember_content_blocking(self, collection: str, content: str, document: str) -> None:
        """Record an indexed content hash; :meth:`remember_content` is this off the loop."""
        _append_line(self._dedup_path(collection), _dedup_line(content, document), self._tails)

    async def claim_content(self, collection: str, content: str, document: str) -> str | None:
        """Remember a hash unless another document holds it, off the event loop."""
        return await asyncio.to_thread(self.claim_content_blocking, collection, content, document)

    def claim_content_blocking(self, collection: str, content: str, document: str) -> str | None:
        """Remember a hash, or return the other document holding it; checked under the lock.

        Check and append are one step, so of two jobs committing the same content at once
        exactly one claims it, whichever process each runs in (TASK-0471).
        """
        path = self._dedup_path(collection)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with _exclusive(path):
                holder = self._held_content(path).get(content)
                if holder is not None and holder != document:
                    return holder
                _append_held(path, _dedup_line(content, document), self._tails)
        except OSError as exc:
            raise _append_failed(path, exc) from exc
        return None

    def _held_content(self, path: Path) -> dict[str, str]:
        """Return a dedup log's hash map, reading only lines added since this journal last did.

        The caller holds the log's lock. A commit per document must not re-read the whole
        file (TASK-0534); a log rewritten or removed since is read again from the start.
        """
        try:
            stat = path.stat()
        except FileNotFoundError:
            self._views.pop(path, None)
            return {}
        identity = (stat.st_dev, stat.st_ino)
        view = self._views.get(path)
        if view is None or view.identity != identity or stat.st_size < view.offset:
            view = self._views[path] = _DedupView(identity)
        if stat.st_size > view.offset:
            _apply_dedup(_new_lines(path, view, stat.st_size), view.known, view.held)
        return view.known

    def _extents_path(self, collection: str) -> Path:
        """Return the path of a collection's chunk-extent log, beside its dedup log."""
        return self._dedup_path(collection).with_name(_EXTENTS_FILE)

    async def chunk_extents(self, collection: str) -> dict[str, list[Extent]]:
        """Return each document's possible ``(chunker, chunks)`` extents, off the event loop."""
        return await asyncio.to_thread(self.chunk_extents_blocking, collection)

    def chunk_extents_blocking(self, collection: str) -> dict[str, list[Extent]]:
        """Return the extents; :meth:`chunk_extents` is this off the loop (TASK-0518).

        The first entry is the last settled one, if any; the rest are attempts since.
        """
        return _live_extents(_logged_lines(self._extents_path(collection)))

    async def record_extent(
        self, collection: str, document: str, chunker: str, chunks: int, *, settled: bool
    ) -> None:
        """Record a document's chunk extent, written whatever ``ingestion.dedup`` says."""
        await asyncio.to_thread(
            self.record_extent_blocking, collection, document, chunker, chunks, settled=settled
        )

    def record_extent_blocking(
        self, collection: str, document: str, chunker: str, chunks: int, *, settled: bool
    ) -> None:
        """Record an extent: an attempt before chunks are written, settled once they all are.

        :meth:`record_extent` is this off the event loop.
        """
        line = _extent_line(document, chunker, chunks, settled=settled)
        _append_line(self._extents_path(collection), line, self._tails)

    async def remember_documents(self, collection: str, hashes: dict[str, str]) -> None:
        """Record many indexed documents in one append, off the event loop (TASK-0538)."""
        await asyncio.to_thread(self.remember_documents_blocking, collection, hashes)

    def remember_documents_blocking(self, collection: str, hashes: dict[str, str]) -> None:
        """Record ``document_id → content_hash`` pairs under one lock; the async twin is above."""
        if hashes:
            lines = "\n".join(_dedup_line(digest, doc) for doc, digest in sorted(hashes.items()))
            _append_line(self._dedup_path(collection), lines, self._tails)

    async def forget_document(
        self, collection: str, document: str, *, known: Collection[str] | None = None
    ) -> bool:
        """Tombstone a document in a collection's dedup index, off the event loop."""
        return await asyncio.to_thread(
            self.forget_document_blocking, collection, document, known=known
        )

    def forget_document_blocking(
        self, collection: str, document: str, *, known: Collection[str] | None = None
    ) -> bool:
        """Tombstone a live document; returns whether it was live, writing nothing if not.

        ``known`` is the live document ids a caller already loaded, which spares reading
        the whole dedup file once per document (TASK-0534); without it the file is read.
        """
        held = known if known is not None else self.known_content_blocking(collection).values()
        if document not in held:
            return False

        line = json.dumps(
            {"deleted_at": _now(), "document_id": document, _TOMBSTONE_KEY: True}, sort_keys=True
        )
        _append_line(self._dedup_path(collection), line, self._tails)
        return True

    async def delete_document(
        self,
        collection: str,
        document: str,
        remove: Callable[[], Coroutine[Any, Any, None]],
    ) -> bool:
        """Log a delete, tombstone the document and await ``remove``, under the deletion lock.

        Returns whether the dedup record held the document. A job's commit waits for the
        whole delete, so the two never interleave (TASK-0519).
        """
        loop = asyncio.get_running_loop()

        def removing() -> None:
            asyncio.run_coroutine_threadsafe(remove(), loop).result()

        return await asyncio.to_thread(
            self.delete_document_blocking, collection, document, removing
        )

    def delete_document_blocking(
        self, collection: str, document: str, remove: Callable[[], None]
    ) -> bool:
        """Log, tombstone and remove a document; :meth:`delete_document` is this off the loop."""
        line = json.dumps({"deleted_at": _now(), "document_id": document}, sort_keys=True)
        with self._deletion_lock(collection):
            try:
                _append_held(self._deletions_path(collection), line, self._tails)
            except OSError as exc:
                raise IngestionError(
                    f"the ingestion journal could not record the delete of {document}: {exc}",
                    code=ErrorCode.INTERNAL,
                ) from exc
            known = self.forget_document_blocking(collection, document)
            remove()
        return known

    async def watch_deletions(self, collection: str) -> DeletionWatch:
        """Mark where a job starts watching a collection's deletes, off the event loop."""
        return await asyncio.to_thread(self.watch_deletions_blocking, collection)

    def watch_deletions_blocking(self, collection: str) -> DeletionWatch:
        """Mark where a job starts watching deletes; :meth:`watch_deletions` off the loop.

        Taken under the deletion lock, so a delete in progress is wholly before it or after.
        """
        with self._deletion_lock(collection):
            path = self._deletions_path(collection)
            try:
                return DeletionWatch(offset=path.stat().st_size)
            except FileNotFoundError:
                return DeletionWatch()

    async def forget_collection(self, collection: str) -> bool:
        """Drop a collection's whole dedup record, off the event loop (TASK-0519)."""
        return await asyncio.to_thread(self.forget_collection_blocking, collection)

    def forget_collection_blocking(self, collection: str) -> bool:
        """Drop a collection's dedup record; :meth:`forget_collection` is this off the loop.

        Returns whether there was one. Held under the append lock, so no line lands in a
        file that is being removed.
        """
        path = self._dedup_path(collection)
        if not path.parent.is_dir():
            return False
        try:
            with _exclusive(path):
                existed = path.is_file()
                _patiently(partial(path.unlink, missing_ok=True))
                self._tails.pop(path, None)
            extents = self._extents_path(collection)
            with _exclusive(extents):
                _patiently(partial(extents.unlink, missing_ok=True))
                self._tails.pop(extents, None)
        except OSError as exc:
            raise IngestionError(
                f"the dedup record {path} could not be removed: {exc}", code=ErrorCode.INTERNAL
            ) from exc
        return existed

    async def known_content(self, collection: str) -> dict[str, str]:
        """Return the collection's content-hash to document-id map, off the event loop."""
        return await asyncio.to_thread(self.known_content_blocking, collection)

    def known_content_blocking(self, collection: str) -> dict[str, str]:
        """Return the hash to document-id map; :meth:`known_content` is this off the loop."""
        return _live_content(_logged_lines(self._dedup_path(collection)))

    async def document_hashes(self, collection: str) -> dict[str, str]:
        """Return the collection's ``document_id → content_hash`` map, off the event loop."""
        return await asyncio.to_thread(self.document_hashes_blocking, collection)

    def document_hashes_blocking(self, collection: str) -> dict[str, str]:
        """Return the document-id to hash map; :meth:`document_hashes` is this off the loop."""
        return {
            document: digest for digest, document in self.known_content_blocking(collection).items()
        }

    async def is_duplicate(self, collection: str, content: str) -> bool:
        """Return whether a collection already holds this content, off the event loop."""
        return await asyncio.to_thread(self.is_duplicate_blocking, collection, content)

    def is_duplicate_blocking(self, collection: str, content: str) -> bool:
        """Return whether the content is already held; :meth:`is_duplicate` is this off the loop."""
        return content in self.known_content_blocking(collection)

    def prune_due(self) -> bool:
        """Return whether retention is on and this journal has not pruned in the last hour."""
        if self.retention_days <= 0:
            return False
        last = self._last_pruned
        return last is None or time.monotonic() - last >= _PRUNE_INTERVAL_SECONDS

    async def prune(self) -> PruneReport:
        """Apply ``ingestion.journal.retention_days``, off the event loop (TASK-0474)."""
        return await asyncio.to_thread(self.prune_blocking)

    def prune_blocking(self) -> PruneReport:
        """Remove settled jobs past retention and compact dedup logs; :meth:`prune` off the loop.

        Never a queued, running or interrupted job, and never a hash a live document holds:
        dedup lines go only when superseded or tombstoned, whatever their age.
        """
        if self.retention_days <= 0:
            return PruneReport()
        self._last_pruned = time.monotonic()
        cutoff = time.time() - self.retention_days * _SECONDS_PER_DAY
        pruned, failed = self._prune_jobs(cutoff)
        dropped = 0
        collections = self.root / _COLLECTIONS_DIR
        logs = [
            *collections.glob(f"*/{_DEDUP_FILE}"),
            *collections.glob(f"*/{_BACKENDS_DIR}/*/{_DEDUP_FILE}"),
            *collections.glob(f"*/{_EXTENTS_FILE}"),
            *collections.glob(f"*/{_BACKENDS_DIR}/*/{_EXTENTS_FILE}"),
        ]
        for path in sorted(logs):
            try:
                dropped += self._compact_dedup(path)
            except IngestionError as exc:
                failed += 1
                _logger.warning(
                    "a dedup log could not be compacted; it is left whole and tried again later",
                    extra={"path": str(path), "code": exc.code.value, "detail": exc.detail},
                )
        report = PruneReport(jobs=tuple(pruned), dedup_lines=dropped, failed=failed)
        if pruned or dropped or failed:
            _logger.info(
                "pruned the ingestion journal past its retention window",
                extra={
                    "jobs": len(pruned),
                    "dedup_lines": dropped,
                    "failed": failed,
                    "retention_days": self.retention_days,
                },
            )
        return report

    def _prune_jobs(self, cutoff: float) -> tuple[list[str], int]:
        """Remove every settled, uninterrupted job that finished before ``cutoff``."""
        directory = self.root / _JOBS_DIR
        if not directory.is_dir():
            return [], 0
        pruned: list[str] = []
        failed = 0
        for path in sorted(directory.glob("*.json")):
            if _modified(path, missing=time.time()) >= cutoff:
                continue
            payload = self._read_quiet_record(path)
            if payload is None:
                continue
            record = JobRecord.from_dict(payload)
            if record.status not in _SETTLED or record.reason_code == _INTERRUPTED:
                continue
            if not _finished_before(record, cutoff):
                continue
            try:
                self._remove_job(record)
            except (OSError, IngestionError) as exc:
                failed += 1
                _logger.warning(
                    "a job past retention could not be removed; it is tried again next pass",
                    extra={"job_id": record.job_id, "detail": str(exc)},
                )
                continue
            pruned.append(record.job_id)
        return pruned, failed

    def _remove_job(self, record: JobRecord) -> None:
        """Delete a job's document log, then its record and its key entry under the index lock.

        The previous copy goes before the live record, which a read would otherwise fall back
        to; an entry is removed only when it names this job.
        """
        logs = self._documents_path(record.job_id).parent
        if logs.is_dir():
            _patiently(partial(shutil.rmtree, logs))
        path = self._job_path(record.job_id)
        with self._key_index_lock():
            previous = path.with_suffix(path.suffix + _PREVIOUS_SUFFIX)
            _patiently(partial(previous.unlink, missing_ok=True))
            _patiently(partial(path.unlink, missing_ok=True))
            if record.idempotency_key is not None:
                self._forget_key(record, record.idempotency_key)

    def _forget_key(self, record: JobRecord, key: str) -> None:
        """Remove the index entry for ``key`` if it names ``record``, under the index lock."""
        entry = self.root / _KEY_INDEX_DIR / _key_entry_name(key, record.tenant)
        text = _read_text(entry)
        named = open_envelope(text) if text is not None else None
        if named is not None and named.get("job_id") == record.job_id:
            _patiently(partial(entry.unlink, missing_ok=True))

    def _compact_dedup(self, path: Path) -> int:
        """Rewrite a dedup log as the lines that still count, returning how many were dropped.

        Read once without the lock to skip a log with nothing to drop; rewritten under the
        append lock, so no line appended meanwhile is lost.
        """
        lines = _logged_lines(path)
        if len(lines) <= len(_compacted(path, lines)):
            return 0
        try:
            with _exclusive(path):
                lines = _logged_lines(path)
                live = _compacted(path, lines)
                dropped = len(lines) - len(live)
                if dropped <= 0:
                    return 0
                text = "".join(line + "\n" for line in live)
                _write_atomically(path, text, keep_previous=False)
                self._tails.pop(path, None)
        except OSError as exc:
            raise IngestionError(
                f"the dedup log {path} could not be compacted: {exc}", code=ErrorCode.INTERNAL
            ) from exc
        return dropped

    async def counts(self, job: str) -> dict[str, int]:
        """Return per-status document counts for a job, off the event loop."""
        return await asyncio.to_thread(self.counts_blocking, job)

    def counts_blocking(self, job: str) -> dict[str, int]:
        """Return per-status document counts; :meth:`counts` is this off the event loop."""
        tally: dict[str, int] = {}
        for record in self.documents_blocking(job):
            tally[record.status] = tally.get(record.status, 0) + 1
        tally["total"] = sum(count for status, count in tally.items() if status != "total")
        return tally


def create_journal(settings: Settings, root: str | Path = DEFAULT_JOURNAL_ROOT) -> Journal:
    """Build a journal from validated configuration."""
    journal = settings.ingestion.journal
    return Journal(
        root,
        checkpoint_every=journal.checkpoint_every,
        enabled=journal.enabled,
        retention_days=journal.retention_days,
        backend=backend_identity(settings),
    )
