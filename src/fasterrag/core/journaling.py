"""What the parse workers need from the ingestion journal, stated where a worker may import it.

The journal itself (``services/journal.py``) is a service: it owns files, locks and job
records. The CPU pool only writes per-document outcomes into it and reads the dedup index, and
a worker may not import a service (``docs/structure.md``, the import rule). So the record a
worker writes and the five calls it makes live here, below both, and the pool is typed against
:class:`DocumentJournal`. :class:`~fasterrag.services.journal.Journal` satisfies the protocol
structurally; nothing registers it.

Moved here on 2026-11-15 (TASK-0529). ``DocumentRecord`` and ``DocumentStatus`` are still
re-exported from ``services.journal``, as the same objects, so imports from there keep working.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Protocol

from fasterrag.errors import ErrorCode

__all__ = ["DeletionWatch", "DocumentJournal", "DocumentRecord", "DocumentStatus"]

DocumentStatus = str


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


class DocumentJournal(Protocol):
    """The journal calls a parse worker makes: outcomes, dead letters and the dedup index."""

    async def known_content(self, collection: str) -> dict[str, str]:
        """Return the collection's content-hash to document-id map."""

    async def record_document(self, job: str, record: DocumentRecord) -> None:
        """Append a document outcome to the job's log."""

    async def record_indexed(
        self,
        job: str,
        collection: str,
        record: DocumentRecord,
        *,
        dedup: bool,
        watch: DeletionWatch | None = None,
    ) -> bool:
        """Log a parsed document unless deleted since ``watch``; return whether it was kept."""

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
        """Record a document that failed its retries, and return the entry written."""

    async def forget_document(
        self, collection: str, document: str, *, known: Collection[str] | None = None
    ) -> bool:
        """Tombstone a document in the collection's dedup index; return whether it was live."""

    async def watch_deletions(self, collection: str) -> DeletionWatch:
        """Mark where a job starts watching the collection's deletes."""


@dataclass(slots=True)
class DeletionWatch:
    """How far a job has read a collection's deletion log, and what it has seen deleted.

    Taken before the job loads its dedup map, so every delete after that is seen at commit.
    """

    offset: int = 0
    deleted: set[str] = field(default_factory=set)
