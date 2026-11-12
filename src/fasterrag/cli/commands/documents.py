"""The ``documents`` command: per-document operations on a collection.

``documents delete`` is ``DELETE /v1/collections/{name}/documents/{document_id}`` from a
terminal, and ``documents list`` is ``GET /v1/collections/{name}/documents``. Each pair calls
the same ``DocumentService`` method, so what is removed, what is listed, what is recorded
and what "not found" means are the same on either surface (TASK-0443, TASK-0469).

The delete builds the configured semantic cache and invalidates it, which reaches a
``disk`` or ``redis`` cache a running server shares. A server holding an in-``memory`` cache
is another process and cannot be reached from here; its cached answers expire on
``cache.ttl``, and the REST route is the way to clear them at once.
"""

from __future__ import annotations

import argparse

from fasterrag.adapters.vectordb.factory import create_vector_db_adapter
from fasterrag.cli.commands.pipeline import _settings_or_none
from fasterrag.cli.output import Console, ExitCode
from fasterrag.core.cache import create_semantic_store
from fasterrag.core.cache.semantic import SemanticCache
from fasterrag.errors import FasterRagError
from fasterrag.services.documents import DocumentDeletion, DocumentListing, DocumentService
from fasterrag.services.journal import create_journal
from fasterrag.services.lockfile import create_lock_store

__all__ = ["run_documents"]


async def run_documents(args: argparse.Namespace, console: Console) -> ExitCode:
    """Dispatch a ``documents`` subcommand: ``list`` or ``delete``."""
    settings = _settings_or_none(args, console)
    if settings is None:
        return ExitCode.USAGE

    collection = args.collection or settings.vector_db.collection.default_name
    adapter = create_vector_db_adapter(settings)
    cache = SemanticCache(settings, create_semantic_store(settings))
    service = DocumentService(
        adapter, create_journal(settings), cache=cache, locks=create_lock_store(settings)
    )
    try:
        if args.action == "list":
            _report_listing(
                await service.list_documents(collection, offset=args.offset, limit=args.limit),
                console,
            )
        else:
            _report_deletion(
                await service.delete(collection, args.document_id, tenant=args.tenant), console
            )
    except FasterRagError as exc:
        console.problem(exc.code.value, exc.detail)
        console.document({"error": exc.code.value, "detail": exc.detail})
        return ExitCode.UNREACHABLE if exc.retryable else ExitCode.FAILURE
    finally:
        await cache.close()
        await adapter.close()
    return ExitCode.SUCCESS


def _report_listing(listing: DocumentListing, console: Console) -> None:
    """Print one line per document, then where this page sits in the whole."""
    for document in listing.documents:
        console.emit(f"{document.document_id}  {document.content_hash}")
    shown = len(listing.documents)
    first = listing.offset + 1 if shown else listing.offset
    console.emit(
        f"{listing.collection}: documents {first}-{listing.offset + shown} "
        f"({shown} of {listing.total})"
    )
    if listing.next_offset is not None:
        console.emit(f"next page: fasterrag documents list --offset {listing.next_offset}")
    console.document(listing.as_dict())


def _report_deletion(deletion: DocumentDeletion, console: Console) -> None:
    """Say what a delete removed, or that the journal had no record of the document."""
    if deletion.known:
        console.emit(f"deleted {deletion.document_id} from {deletion.collection}")
    else:
        console.emit(
            f"{deletion.document_id}: the journal holds no record of it in "
            f"{deletion.collection}; any chunks carrying that id were removed"
        )
    console.document(deletion.as_dict())
