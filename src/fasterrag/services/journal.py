"""Checkpointed, idempotent ingestion state (D3).

Three capabilities live here, and together they turn ingestion from a job that must
succeed into a resumable, auditable batch system:

* **Checkpoints.** A job record is rewritten every ``ingestion.journal.checkpoint_every``
  documents. A crash mid-ingest of millions of documents resumes from the last checkpoint
  instead of restarting, which is the difference between minutes and days.
* **Deduplication.** Content hashes are remembered per collection, so re-running an
  ingest is a no-op rather than a second copy of the corpus.
* **Dead-lettering.** A document that fails its retries is recorded with a machine-readable
  reason code and stays queryable, so a bad file never silently disappears and never stops
  the pipeline.

Every write is atomic — written to a temporary file and renamed — and every job record
carries a checksum. A torn write is therefore detectable, and the previous good record is
kept so a damaged one degrades to resuming slightly earlier rather than to losing the job
(``docs/reliability.md`` §3, ``docs/failure-modes.md`` rows 31 and 32).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Sequence
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
]

DEFAULT_JOURNAL_ROOT: Final = Path(".fasterrag") / "journal"

_JOBS_DIR: Final = "jobs"
_COLLECTIONS_DIR: Final = "collections"
_DOCUMENTS_FILE: Final = "documents.jsonl"
_DEDUP_FILE: Final = "content-hashes.jsonl"
_PREVIOUS_SUFFIX: Final = ".prev"

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
    """Write a file atomically, keeping the previous version as a fallback.

    The rename is the only step that publishes the new content, so a crash leaves either
    the old file or the new one — never a half-written record.

    # CRITICAL: the `.prev` rename happens *before* the write, and that ordering is
    # load-bearing rather than incidental. On a filesystem with zero free bytes it frees a
    # file the size of the one about to be written, which is why an already-running job can
    # still checkpoint on the very disk that halted it while a new job cannot start
    # (docs/failure-modes.md row 33, measured against a real ENOSPC). Writing first and
    # renaming after would make "resume from the last checkpoint" unreachable exactly when
    # it is needed.

    Raises:
        IngestionError: If the record cannot be written. The journal is what crash-resume
            reads, so a failure here is reported rather than warned about and swallowed the
            way a lost trace or lockfile is — losing it silently would mean losing the
            ability to resume without anything saying so.
    """
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


def _envelope(payload: dict[str, Any]) -> str:
    """Wrap a payload with a checksum so a torn write is detectable."""
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return json.dumps({"checksum": text_hash(body), "body": body}, separators=(",", ":"))


def _open_envelope(text: str) -> dict[str, Any] | None:
    """Return the payload if its checksum matches, otherwise None."""
    try:
        envelope = json.loads(text)
        body = envelope["body"]
        if text_hash(body) != envelope["checksum"]:
            return None
        loaded = json.loads(body)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

    return loaded if isinstance(loaded, dict) else None


class Journal:
    """Durable ingestion state: jobs, per-document outcomes, and dedup hashes."""

    def __init__(
        self,
        root: str | Path = DEFAULT_JOURNAL_ROOT,
        *,
        checkpoint_every: int = 100,
        enabled: bool = True,
    ) -> None:
        """Open a journal rooted at ``root``.

        Args:
            root: Directory holding journal state. There is no configuration key for this
                path; callers pass the deployment's data directory.
            checkpoint_every: Documents between checkpoints, from
                ``ingestion.journal.checkpoint_every``.
            enabled: When false, checkpoints are skipped but job and document records are
                still written, so status stays queryable with ``journal.enabled: false``.
        """
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

    def create_job(
        self,
        collection: str,
        sources: Sequence[dict[str, str]],
        *,
        idempotency_key: str | None = None,
        tenant: str | None = None,
    ) -> JobRecord:
        """Create and persist a job.

        Replaying an idempotency key returns the original job and creates nothing, so a
        retried submission never starts a second ingest of the same corpus.
        """
        if idempotency_key is not None:
            existing = self.find_by_idempotency_key(idempotency_key)
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
        self.save_job(record)
        return record

    def save_job(self, record: JobRecord) -> None:
        """Persist a job record atomically."""
        _write_atomically(self._job_path(record.job_id), _envelope(record.as_dict()))

    def load_job(self, job: str, *, tenant: str | None = None) -> JobRecord:
        """Load a job record, falling back to the previous good copy.

        # CRITICAL: another tenant's job raises the same `NOT_FOUND` an unknown id does, and
        # deliberately so. A job record carries the source paths of the corpus it ingested,
        # and a distinct "forbidden" would confirm the id is real — job ids sort
        # chronologically, so a caller who has one of their own can guess at neighbours.

        Raises:
            IngestionError: If the job is unknown, belongs to another tenant, or both copies
                are unreadable.
        """
        path = self._job_path(job)
        if not path.exists():
            raise IngestionError(f"unknown ingest job {job!r}", code=ErrorCode.NOT_FOUND)

        payload = _open_envelope(path.read_text(encoding="utf-8"))
        if payload is None:
            previous = path.with_suffix(path.suffix + _PREVIOUS_SUFFIX)
            payload = (
                _open_envelope(previous.read_text(encoding="utf-8")) if previous.exists() else None
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

    def find_by_idempotency_key(self, key: str) -> JobRecord | None:
        """Return the job created with ``key``, if one exists."""
        directory = self.root / _JOBS_DIR
        if not directory.is_dir():
            return None

        for path in sorted(directory.glob("*.json")):
            payload = _open_envelope(path.read_text(encoding="utf-8"))
            if payload is not None and payload.get("idempotency_key") == key:
                return JobRecord.from_dict(payload)
        return None

    def checkpoint(self, record: JobRecord, document_index: int) -> JobRecord:
        """Record progress, writing only at the configured interval.

        Returns the record with its checkpoint updated. Writing on every document would
        make the journal the pipeline's bottleneck; writing never would make a crash
        unrecoverable. The interval is the dial between those.
        """
        if not self.enabled:
            return record

        reached_interval = (document_index + 1) % self.checkpoint_every == 0
        if not reached_interval:
            return record

        updated = replace(
            record,
            checkpoint=Checkpoint(last_document_index=document_index, written_at=_now()),
        )
        self.save_job(updated)
        return updated

    def resume_index(self, record: JobRecord) -> int:
        """Return the document index to resume from after a crash."""
        if record.checkpoint is None:
            return 0
        return record.checkpoint.last_document_index + 1

    def record_document(self, job: str, record: DocumentRecord) -> None:
        """Append a document outcome to the job's log."""
        path = self._documents_path(job)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), sort_keys=True, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

    def dead_letter(
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
        """Record a document that failed its retries, and return the entry."""
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
        self.record_document(job, record)
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

    def documents(self, job: str, *, status: str | None = None) -> Iterator[DocumentRecord]:
        """Yield a job's document outcomes, optionally filtered by status."""
        path = self._documents_path(job)
        if not path.exists():
            return

        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            record = DocumentRecord(**payload)
            if status is None or record.status == status:
                yield record

    def dead_lettered(self, job: str) -> list[DocumentRecord]:
        """Return the job's dead-letter entries."""
        return list(self.documents(job, status="dead_lettered"))

    def remember_content(self, collection: str, content: str, document: str) -> None:
        """Record that a content hash is indexed in a collection."""
        path = self._dedup_path(collection)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"content_hash": content, "document_id": document})
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{line}\n")

    def known_content(self, collection: str) -> dict[str, str]:
        """Return the collection's content-hash to document-id map."""
        path = self._dedup_path(collection)
        if not path.exists():
            return {}

        known: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                payload = json.loads(line)
                known[payload["content_hash"]] = payload["document_id"]
        return known

    def document_hashes(self, collection: str) -> dict[str, str]:
        """Return the collection's ``document_id → content_hash`` map.

        The direction the index lockfile records (D1). The dedup index is stored the other
        way round because its own question is "have I seen this content", and inverting it
        is lossless: a content hash maps to exactly one document by construction.
        """
        return {document: digest for digest, document in self.known_content(collection).items()}

    def is_duplicate(self, collection: str, content: str) -> bool:
        """Return whether a collection already holds this exact content."""
        return content in self.known_content(collection)

    def counts(self, job: str) -> dict[str, int]:
        """Return per-status document counts for a job."""
        tally: dict[str, int] = {}
        for record in self.documents(job):
            tally[record.status] = tally.get(record.status, 0) + 1
        tally["total"] = sum(count for status, count in tally.items() if status != "total")
        return tally


def create_journal(settings: Settings, root: str | Path = DEFAULT_JOURNAL_ROOT) -> Journal:
    """Build a journal from validated configuration.

    One place translates ``ingestion.journal.*`` into constructor arguments, so the CLI, the
    API, and the library cannot disagree about the checkpoint interval a deployment runs
    with — a journal built two ways is a journal that resumes two ways.
    """
    journal = settings.ingestion.journal
    return Journal(root, checkpoint_every=journal.checkpoint_every, enabled=journal.enabled)
