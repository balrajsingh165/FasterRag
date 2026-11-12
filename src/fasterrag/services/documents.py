"""Document lifecycle: taking one document out of a collection (TASK-0443).

Ingestion could add a document and nothing could remove one. ``VectorDBAdapter.delete`` has
existed and been contract-tested since the first adapter, but no service called it, so the
only way to retract a document was to drop the collection and re-ingest everything else.

A delete is four effects, each of which leaves a different lie behind if it is skipped:

* the **chunks** — every point whose ``document_id`` payload matches, removed by one
  pushed-down filter rather than by listing ids, because a point's id embeds the chunker
  configuration (``core.identity.chunk_id``) and a filter also catches chunks written under
  a configuration that has since changed;
* the **dedup record** — a tombstone in the journal, without which the content hash stays
  remembered and the same document can never be ingested again;
* the **lockfile** — which would otherwise list a document the collection no longer holds,
  and report the delete as corpus drift on the next ``index lock verify``;
* the **semantic cache** — which may hold an answer citing a chunk that is now gone.

**The order is chosen for the failure it leaves, not for the success.** The tombstone is
written before the vectors are deleted. If the backend then refuses, the journal has
forgotten a document the index still holds: the caller is told the delete failed, repeating
it removes the chunks, and a re-ingest in the meantime re-writes the same deterministic
point ids. The other order strands the opposite state — chunks gone, hash still remembered
— in which a re-ingest is reported ``deduplicated`` and indexes nothing, silently.

**A delete is idempotent, and "not found" is not something it can know.** The adapter
contract's ``delete`` returns nothing and no adapter can count what a filter matched without
a scan, so whether any chunk existed is not observable here. The journal is the only record,
and it is not the index: with ``ingestion.dedup`` off, or after an archive import, it holds
nothing for documents that are indexed. The vector delete is therefore always issued, and
:attr:`DocumentDeletion.known` reports the one thing that *is* known — whether the journal
held a live record. Deleting a document nobody ingested succeeds and says ``known: false``.

**Tenancy.** A document id is a digest of the source path and the tenant
(``core.identity.document_id``), so it is derivable by anyone who can guess a path; holding
the id is not ownership. When a tenant is given, the filter also requires the chunk's
``tenant`` payload tag — the field ``workers.indexer`` writes from the credential the job
was created with — so a caller can remove only chunks carrying its own tag, whichever
collection they are in and whichever of the two tenancy models TASK-0339 settles on. The
tombstone is *not* tenant-checked, because the dedup record carries no tenant to check
against; under collection-name scoping the journal is already per tenant, and the worst a
foreign tombstone does is make its owner's next ingest re-embed one document.

**Listing reads the same record** (TASK-0469). A collection's documents are listed from the
journal's dedup index, the record a delete tombstones, so a deleted document disappears from
the listing in the same step that lets it be ingested again. The cost of that choice is the
same caveat ``known`` carries: the listing is what the journal remembers, not a scan of the
index, so with ``ingestion.dedup`` off, or after an archive import, it is empty for documents
that are indexed. A scan would be honest about those and would read every point of a
collection per page, which is the wrong price for a listing on a very large corpus.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from fasterrag.adapters.vectordb.base import PointSelector, VectorDBAdapter
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.journal import Journal
from fasterrag.services.lockfile import LockStore
from fasterrag.services.tenancy import TENANT_PAYLOAD_KEY

__all__ = [
    "DEFAULT_PAGE_SIZE",
    "DOCUMENT_ID_KEY",
    "MAXIMUM_PAGE_SIZE",
    "DocumentDeletion",
    "DocumentListing",
    "DocumentService",
    "ListedDocument",
    "document_filter",
]

DOCUMENT_ID_KEY = "document_id"
DEFAULT_PAGE_SIZE = 100
MAXIMUM_PAGE_SIZE = 1000

_logger = get_logger(__name__)


class _Invalidates(Protocol):
    """The one thing a delete needs from the semantic cache."""

    async def invalidate(self, reason: str) -> None:
        """Drop every entry because the corpus changed."""


@dataclass(frozen=True, slots=True)
class DocumentDeletion:
    """What a delete did, as far as it can be known.

    ``known`` is whether the journal held a live record of the document; it is not a
    count of chunks removed, which the adapter contract does not report.
    """

    collection: str
    document_id: str
    known: bool
    deleted_at: str

    def as_dict(self) -> dict[str, Any]:
        """Return the form every surface reports."""
        return {
            "collection": self.collection,
            "document_id": self.document_id,
            "known": self.known,
            "deleted_at": self.deleted_at,
        }


@dataclass(frozen=True, slots=True)
class ListedDocument:
    """One document the journal records as indexed in a collection."""

    document_id: str
    content_hash: str

    def as_dict(self) -> dict[str, Any]:
        """Return the form every surface reports."""
        return {"document_id": self.document_id, "content_hash": self.content_hash}


@dataclass(frozen=True, slots=True)
class DocumentListing:
    """One page of a collection's documents, ordered by ``document_id``.

    ``total`` counts every document on every page; ``next_offset`` is ``None`` on the last.
    """

    collection: str
    documents: list[ListedDocument]
    total: int
    offset: int
    limit: int

    @property
    def next_offset(self) -> int | None:
        """Return the offset of the next page, or ``None`` when this page is the last."""
        following = self.offset + len(self.documents)
        return following if following < self.total else None

    def as_dict(self) -> dict[str, Any]:
        """Return the form every surface reports."""
        return {
            "collection": self.collection,
            "documents": [document.as_dict() for document in self.documents],
            "total": self.total,
            "offset": self.offset,
            "limit": self.limit,
            "next_offset": self.next_offset,
        }


def document_filter(document_id: str, tenant: str | None = None) -> dict[str, Any]:
    """Return the payload filter selecting one document's chunks, within one tenant."""
    selected: dict[str, Any] = {DOCUMENT_ID_KEY: document_id}
    if tenant is not None:
        selected[TENANT_PAYLOAD_KEY] = tenant
    return selected


class DocumentService:
    """Removes documents from a collection, and from every record that claims them."""

    def __init__(
        self,
        adapter: VectorDBAdapter,
        journal: Journal,
        *,
        cache: _Invalidates | None = None,
        locks: LockStore | None = None,
    ) -> None:
        """Build the service over the application's adapter and journal."""
        self.adapter = adapter
        self.journal = journal
        self.cache = cache
        self.locks = locks

    async def delete(
        self,
        collection: str,
        document_id: str,
        *,
        tenant: str | None = None,
        display: str | None = None,
    ) -> DocumentDeletion:
        """Delete one document's chunks and every record that it was indexed.

        ``display`` is the collection name the caller used, for error text; ``collection``
        may carry a tenant prefix that must not reach a response.
        """
        if not document_id.strip():
            raise FasterRagError(
                "a document id is required to delete a document",
                code=ErrorCode.VALIDATION_FAILED,
                retryable=False,
            )

        shown = display or collection
        known = await self.journal.forget_document(collection, document_id)
        try:
            await self.adapter.delete(
                PointSelector(collection=collection, filters=document_filter(document_id, tenant))
            )
        except FasterRagError as exc:
            if exc.code is not ErrorCode.NOT_FOUND:
                raise
            raise FasterRagError(
                f"no collection named {shown!r}", code=ErrorCode.NOT_FOUND, retryable=False
            ) from exc

        await self._drop_from_lock(collection, document_id)
        if self.cache is not None:
            await self.cache.invalidate(f"document {document_id} deleted from {shown}")

        _logger.info(
            "document deleted",
            extra={
                "collection": collection,
                "document_id": document_id,
                "known": known,
                "tenant": tenant,
            },
        )
        return DocumentDeletion(
            collection=shown,
            document_id=document_id,
            known=known,
            deleted_at=datetime.now(tz=UTC).isoformat(),
        )

    async def list_documents(
        self,
        collection: str,
        *,
        offset: int = 0,
        limit: int = DEFAULT_PAGE_SIZE,
        display: str | None = None,
    ) -> DocumentListing:
        """Return one page of the documents the journal records in ``collection``.

        Read from the dedup record, so a deleted document's tombstone hides it (TASK-0469).
        A collection the backend does not hold is ``NOT_FOUND``, as on every sibling route.
        """
        if offset < 0 or not 1 <= limit <= MAXIMUM_PAGE_SIZE:
            raise FasterRagError(
                f"offset must be 0 or more and limit between 1 and {MAXIMUM_PAGE_SIZE}; "
                f"got offset={offset}, limit={limit}",
                code=ErrorCode.VALIDATION_FAILED,
                retryable=False,
            )

        shown = display or collection
        if not await self._exists(collection):
            raise FasterRagError(
                f"no collection named {shown!r}", code=ErrorCode.NOT_FOUND, retryable=False
            )

        held = await self.journal.document_hashes(collection)
        ordered = sorted(held)
        return DocumentListing(
            collection=shown,
            documents=[
                ListedDocument(document_id=document, content_hash=held[document])
                for document in ordered[offset : offset + limit]
            ],
            total=len(ordered),
            offset=offset,
            limit=limit,
        )

    async def _exists(self, collection: str) -> bool:
        """Return whether the backend holds ``collection``, as a collection or an alias."""
        if any(info.name == collection for info in await self.adapter.list_collections()):
            return True
        return await self.adapter.alias_target(collection) is not None

    async def _drop_from_lock(self, collection: str, document_id: str) -> None:
        """Remove the document from the collection's lockfile, off the event loop."""
        if self.locks is None or not self.locks.enabled:
            return
        await asyncio.to_thread(_rewrite_lock, self.locks, collection, document_id)


def _rewrite_lock(locks: LockStore, collection: str, document_id: str) -> None:
    """Rewrite a lockfile without one document, leaving an absent or unaffected one alone."""
    lock = locks.read(collection)
    if lock is None or document_id not in lock.document_hashes:
        return

    remaining = {
        document: digest
        for document, digest in lock.document_hashes.items()
        if document != document_id
    }
    locks.write(replace(lock, document_hashes=remaining))
