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

Every write is atomic — written to a temporary file and renamed — and every job record
carries a checksum. A torn write is therefore detectable, and the previous good record is
kept so a damaged one degrades to resuming slightly earlier rather than to losing the job
(``docs/reliability.md`` §3, ``docs/failure-modes.md`` rows 31 and 32).
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from fasterrag.config.schema import Settings
from fasterrag.core.identity import job_id as new_job_id
from fasterrag.core.identity import text_hash
from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.observability.logging import get_logger

__all__ = [
    "DEFAULT_JOURNAL_ROOT",
    "Checkpoint",
    "DocumentRecord",
    "DocumentStatus",
    "JobRecord",
    "JobStatus",
    "Journal",
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
        )


def _write_atomically(path: Path, payload: str) -> None:
    """Write a file atomically, keeping the previous version as a fallback."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            os.replace(path, path.with_suffix(path.suffix + _PREVIOUS_SUFFIX))

        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, path)
    except OSError as exc:
        # CRITICAL: the cleanup is itself suppressed, for the reason recorded in
        # TraceStore.save and IndexLockStore.write — on Linux a parent that is a file rather
        # than a directory makes `unlink` raise from inside this handler, replacing the error
        # being reported with a second one that escapes.
        with suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise IngestionError(
            f"the ingestion journal could not be written to {path}: {exc}. "
            "The last checkpoint is still readable, so the job can be resumed once the "
            "condition clears",
            code=ErrorCode.INTERNAL,
        ) from exc


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
        """Load a job record, falling back to the previous good copy."""
        path = self._job_path(job)
        if not path.exists():
            raise IngestionError(f"unknown ingest job {job!r}", code=ErrorCode.NOT_FOUND)

        payload = open_envelope(path.read_text(encoding="utf-8"))
        if payload is None:
            previous = path.with_suffix(path.suffix + _PREVIOUS_SUFFIX)
            payload = (
                open_envelope(previous.read_text(encoding="utf-8")) if previous.exists() else None
            )
            if payload is None:
                raise IngestionError(
                    f"the journal record for job {job!r} is corrupt and has no readable "
                    "previous copy",
                    code=ErrorCode.CHUNK_FAILED,
                    retryable=False,
                )
            _logger.warning(
                "journal record failed its checksum; resuming from the previous checkpoint",
                extra={"job_id": job},
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
            payload = open_envelope(path.read_text(encoding="utf-8"))
            if payload is None or payload.get("idempotency_key") != key:
                continue
            if payload.get("tenant") == tenant:
                return JobRecord.from_dict(payload)
        return None

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
        self.save_job_blocking(updated)
        return updated

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
        path = self._documents_path(job)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), sort_keys=True, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

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
        path = self._documents_path(job)
        if not path.exists():
            return []

        found: list[DocumentRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = DocumentRecord(**json.loads(line))
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
        path = self._dedup_path(collection)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"content_hash": content, "document_id": document})
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

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
        with self._dedup_path(collection).open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")
        return True

    async def known_content(self, collection: str) -> dict[str, str]:
        """Return the collection's content-hash to document-id map, off the event loop."""
        return await asyncio.to_thread(self.known_content_blocking, collection)

    def known_content_blocking(self, collection: str) -> dict[str, str]:
        """Return the hash to document-id map; :meth:`known_content` is this off the loop."""
        path = self._dedup_path(collection)
        if not path.exists():
            return {}

        known: dict[str, str] = {}
        held: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
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
