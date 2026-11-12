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

Every rewrite of a record is atomic and durable — written to a uniquely named temporary
file, flushed, renamed over the live file, and the directory flushed where the platform
allows — and every job record carries a checksum. The live record is replaced, never moved
away, so a reader or a crash at any step finds a whole record; a torn one is detectable, and
the previous good copy is kept so a damaged record degrades to resuming slightly earlier
rather than to losing the job (``docs/reliability.md`` §3, ``docs/failure-modes.md`` rows 32
and 33, ``docs/adr/ADR-0009-journal-stays-on-files.md``).
"""

from __future__ import annotations

import asyncio
import errno
import json
import os
import secrets
import sys
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from fasterrag.config.schema import Settings
from fasterrag.core.identity import job_id as new_job_id
from fasterrag.core.identity import text_hash
from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.observability.logging import get_logger

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

__all__ = [
    "DEFAULT_JOURNAL_ROOT",
    "Checkpoint",
    "DocumentRecord",
    "DocumentStatus",
    "JobRecord",
    "JobStatus",
    "Journal",
    "QuietJobs",
    "create_journal",
    "envelope",
    "open_envelope",
]

DEFAULT_JOURNAL_ROOT: Final = Path(".fasterrag") / "journal"

_JOBS_DIR: Final = "jobs"
_COLLECTIONS_DIR: Final = "collections"
_DOCUMENTS_FILE: Final = "documents.jsonl"
_DEDUP_FILE: Final = "content-hashes.jsonl"
_PREVIOUS_SUFFIX: Final = ".prev"
_TOMBSTONE_KEY: Final = "tombstone"
_TEMPORARY_SUFFIX: Final = ".tmp"
_BUSY_DELAYS: Final = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
_DISK_FULL: Final = frozenset({errno.ENOSPC, getattr(errno, "EDQUOT", errno.ENOSPC)})
_LOCK_SUFFIX: Final = ".lock"
_LOCK_TIMEOUT_SECONDS: Final = 30.0
_LOCK_POLL_SECONDS: Final = 0.002

DocumentStatus = str
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


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """The outcome for one document, and the dead-letter entry when it failed.

    ``detail`` is human-readable and never contains a secret value; ``reason_code`` is
    drawn from the stable error-code table so a client can branch on it.
    """

    document_id: str
    source: str
    status: DocumentStatus
    content_hash: str | None = None
    reason_code: str | None = None
    detail: str | None = None
    attempts: int = 0
    trace_id: str | None = None
    first_failed_at: str | None = None
    last_failed_at: str | None = None

    @property
    def dead_lettered(self) -> bool:
        """Return whether this document was dead-lettered."""
        return self.status == "dead_lettered"


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
    """The ``running`` jobs nothing has written to lately, and what the look could not judge."""

    running: tuple[JobRecord, ...] = ()
    recent: int = 0
    unreadable: int = 0


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


def _stage(path: Path, payload: str, staged: list[Path]) -> Path:
    """Write ``payload`` to a uniquely named, flushed temporary file beside ``path``."""
    temporary = path.with_name(f"{path.name}.{secrets.token_hex(8)}{_TEMPORARY_SUFFIX}")
    staged.append(temporary)
    temporary.write_text(payload, encoding="utf-8")
    _sync_file(temporary)
    return temporary


def _keep_previous(path: Path, staged: list[Path]) -> None:
    """Copy the live record to its previous-good slot, when it is whole.

    A copy rather than a rename, so the live record never leaves; a torn one is not copied,
    so the last good copy is never replaced by a damaged one.
    """
    current = _read_text(path)
    if current is None or open_envelope(current) is None:
        return
    previous = path.with_suffix(path.suffix + _PREVIOUS_SUFFIX)
    copy = _stage(previous, current, staged)
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


def _write_atomically(path: Path, payload: str, *, keep_previous: bool = True) -> None:
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
        fresh = _stage(path, payload, staged)
        _patiently(lambda: os.replace(fresh, path))
        _sync_directory(path.parent)
    except OSError as exc:
        _discard(staged)
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


def _append_line(path: Path, line: str) -> None:
    """Append one line under the file's lock, translating any OS failure (TASK-0475).

    A crash mid-append leaves a line without its newline; the next append ends it first, so
    the fragment stays a line of its own that readers skip rather than a corrupt neighbour.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive(path), path.open("a+b") as handle:
            prefix = b""
            if handle.seek(0, os.SEEK_END) > 0:
                handle.seek(-1, os.SEEK_END)
                prefix = b"" if handle.read(1) == b"\n" else b"\n"
            handle.write(prefix + line.encode("utf-8") + b"\n")
    except OSError as exc:
        raise IngestionError(
            f"the ingestion journal could not append to {path}: {exc}. The documents after "
            "the last checkpoint are processed again once the condition clears",
            code=ErrorCode.INTERNAL,
        ) from exc


def _logged_lines(path: Path) -> list[dict[str, Any]]:
    """Return the parsed lines of an append-only log, skipping a fragment a crash left."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
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
    ) -> None:
        """Open a journal rooted at ``root``."""
        self.root = Path(root)
        self.checkpoint_every = checkpoint_every
        self.enabled = enabled

    def _job_path(self, job: str) -> Path:
        """Return the path of a job record."""
        return self.root / _JOBS_DIR / f"{job}.json"

    def _documents_path(self, job: str) -> Path:
        """Return the path of a job's per-document log."""
        return self.root / _JOBS_DIR / job / _DOCUMENTS_FILE

    def _dedup_path(self, collection: str) -> Path:
        """Return the path of a collection's content-hash index."""
        return self.root / _COLLECTIONS_DIR / collection / _DEDUP_FILE

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
        if idempotency_key is not None:
            existing = self.find_by_idempotency_key_blocking(idempotency_key, tenant=tenant)
            if existing is not None:
                _logger.info(
                    "returning the existing job for a replayed idempotency key",
                    extra={"job_id": existing.job_id},
                )
                return existing

        record = JobRecord(
            job_id=new_job_id(),
            collection=collection,
            status="queued",
            sources=[dict(source) for source in sources],
            started_at=_now(),
            idempotency_key=idempotency_key,
            tenant=tenant,
        )
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

        Scoped to the tenant, because an unscoped match hands one tenant another's job id
        and swallows the ingest it asked for (TASK-0318).
        """
        directory = self.root / _JOBS_DIR
        if not directory.is_dir():
            return None

        for path in sorted(directory.glob("*.json")):
            try:
                text = _read_text(path)
            except PermissionError as exc:
                raise _busy(path.stem, exc) from exc
            payload = open_envelope(text) if text is not None else None
            if payload is None or payload.get("idempotency_key") != key:
                continue
            if payload.get("tenant") == tenant:
                return JobRecord.from_dict(payload)
        return None

    async def quiet_running_jobs(self, quiet_seconds: float) -> QuietJobs:
        """Return the ``running`` jobs not written to for ``quiet_seconds``, off the loop."""
        return await asyncio.to_thread(self.quiet_running_jobs_blocking, quiet_seconds)

    def quiet_running_jobs_blocking(self, quiet_seconds: float) -> QuietJobs:
        """Find quiet ``running`` jobs; :meth:`quiet_running_jobs` is this off the loop.

        A record written inside the window is counted and never opened: something is
        still writing it, and reading it could collide with that write.
        """
        directory = self.root / _JOBS_DIR
        if not directory.is_dir():
            return QuietJobs()

        cutoff = time.time() - quiet_seconds
        running: list[JobRecord] = []
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

        return QuietJobs(running=tuple(running), recent=recent, unreadable=unreadable)

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
        _append_line(self._documents_path(job), line)

    async def record_indexed(
        self, job: str, collection: str, record: DocumentRecord, *, dedup: bool
    ) -> None:
        """Log a parsed document and remember its hash in one hop off the event loop.

        One hop rather than two: the per-document path runs millions of times, and the
        thread handoff costs more than either append (``docs/benchmarks.md`` BENCH-0006).
        """
        await asyncio.to_thread(self.record_indexed_blocking, job, collection, record, dedup=dedup)

    def record_indexed_blocking(
        self, job: str, collection: str, record: DocumentRecord, *, dedup: bool
    ) -> None:
        """Log a document and its hash; :meth:`record_indexed` is this off the loop."""
        self.record_document_blocking(job, record)
        if dedup and record.content_hash is not None:
            self.remember_content_blocking(collection, record.content_hash, record.document_id)

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
        line = json.dumps({"content_hash": content, "document_id": document})
        _append_line(self._dedup_path(collection), line)

    async def forget_document(self, collection: str, document: str) -> bool:
        """Tombstone a document in a collection's dedup index, off the event loop."""
        return await asyncio.to_thread(self.forget_document_blocking, collection, document)

    def forget_document_blocking(self, collection: str, document: str) -> bool:
        """Tombstone a document; :meth:`forget_document` is this off the event loop.

        Returns whether a live record was there. Nothing is written otherwise, so a
        retried delete does not grow the file (TASK-0443).
        """
        if document not in self.known_content_blocking(collection).values():
            return False

        line = json.dumps(
            {"deleted_at": _now(), "document_id": document, _TOMBSTONE_KEY: True}, sort_keys=True
        )
        _append_line(self._dedup_path(collection), line)
        return True

    async def known_content(self, collection: str) -> dict[str, str]:
        """Return the collection's content-hash to document-id map, off the event loop."""
        return await asyncio.to_thread(self.known_content_blocking, collection)

    def known_content_blocking(self, collection: str) -> dict[str, str]:
        """Return the hash to document-id map; :meth:`known_content` is this off the loop."""
        known: dict[str, str] = {}
        held: dict[str, str] = {}
        for payload in _logged_lines(self._dedup_path(collection)):
            document = payload["document_id"]
            superseded = held.pop(document, None)
            if superseded is not None and known.get(superseded) == document:
                del known[superseded]
            if payload.get(_TOMBSTONE_KEY):
                continue
            held[document] = payload["content_hash"]
            known[held[document]] = document
        return known

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
    return Journal(root, checkpoint_every=journal.checkpoint_every, enabled=journal.enabled)
