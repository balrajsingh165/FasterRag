"""Payload backfill for the tenant tag, and the audit that must precede it.

``security.tenant_scoped_retrieval`` filters both retrieval legs on the ``tenant`` payload
field. A chunk indexed before ``security.multi_tenancy`` was enabled carries no such field,
and an absent field never matches an equality filter — so turning the flag on over an
unbackfilled corpus returns empty results rather than wrong ones. Fail-closed, but silent.
This module is step 1 of ``docs/security.md`` §3's pre-enable checklist: it makes the
untagged population *visible*, and then writes the tag.

**The audit always runs first, even under ``apply``.** A foreign tenant discovered halfway
through a write would leave half a collection relabelled, so the whole collection is counted
before the first ``PointUpdate`` is sent. That costs a second traversal of a large corpus;
the alternative costs correctness on exactly the corpora large enough to matter.

**Idempotent by predicate, not by bookkeeping.** A point is written only when it lacks the
tag, which is the negation of the state the write produces, so a completed run is a fixpoint
— the next run selects nothing. That is also what makes it resumable: a run killed halfway
needs no checkpoint file to resume, because the surviving untagged points *are* the
remaining work, and no cursor can be stale or corrupt. Order of traversal is irrelevant for
the same reason. ``--overwrite`` keeps the property for a different reason: the value
written is a constant, so re-writing it changes nothing.

**It refuses to relabel.** Points already carrying a different tenant make the untagged ones
unattributable — the shared unscoped collection of ``docs/security.md`` §3 holds several
tenants' chunks, and assigning all of its untagged points to one tenant would hand that
tenant another's data, which is the defect TASK-0339 fixed once in ``chunk_payload``. Such a
collection is refused with ``CONFLICT`` unless the operator passes ``overwrite``, which says
in one word that the whole collection is that tenant's.

**Two escapes, and each one names what it asserts.** ``overwrite`` says the whole collection
is one tenant's and relabels every point in it. ``assume_untagged`` says only the untagged
residue is, and leaves every existing tag alone — the mixed collection's case, which
``overwrite`` would otherwise answer by handing one tenant another's chunks. Passing both is
refused, because they assert different things about the same collection, and the report
records which was used, so ownership that was *asserted* can never be read back as ownership
the system derived. ``inventory_untagged`` is what lets that assertion rest on evidence: it
lists the untagged residue grouped by document, a document being the granularity an operator
can actually recognise. Neither escape is a default and neither is inferred.

**Relocation is deliberately not here.** Splitting a mixed collection into
``<tenant>__<name>`` collections is the remedy TASK-0385 sketched, and it is the wrong one
today: the ingest and query paths address the raw collection name (TASK-0381), so points
moved under a scoped name stop being addressable by the queries the move was meant to fix,
while the filter ``tenant_scoped_retrieval`` installs needs a correct *tag* and does not care
which collection holds it. The reasoning is recorded in ``docs/security.md`` §3.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from fasterrag.adapters.vectordb.base import PointUpdate, VectorDBAdapter
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.tenancy import TENANT_PAYLOAD_KEY, validated_tenant

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "BackfillProgress",
    "BackfillReport",
    "TenantAudit",
    "UntaggedDocument",
    "audit_collection",
    "audit_collections",
    "backfill_tenant",
    "inventory_untagged",
]

DEFAULT_BATCH_SIZE: Final = 512

_logger = get_logger(__name__)


def _tag_of(payload: Mapping[str, Any]) -> str | None:
    """Return the point's tenant tag, or ``None`` when it has none to filter on.

    An empty string counts as untagged: no valid tenant id is empty, so such a point is
    invisible to the filter exactly as an absent field is.
    """
    value = payload.get(TENANT_PAYLOAD_KEY)
    if value is None or value == "":
        return None
    return value if isinstance(value, str) else repr(value)


@dataclass(frozen=True, slots=True)
class TenantAudit:
    """What one collection holds, counted by tenant tag."""

    collection: str
    scanned: int
    untagged: int
    tagged: Mapping[str, int] = field(default_factory=dict)

    @property
    def tenants(self) -> list[str]:
        """Return the distinct tenant tags present, in descending count order."""
        return [
            name for name, _ in sorted(self.tagged.items(), key=lambda item: (-item[1], item[0]))
        ]

    @property
    def backfillable(self) -> bool:
        """Return whether anything in this collection is invisible to a tenant filter."""
        return self.untagged > 0

    def as_dict(self) -> dict[str, Any]:
        """Return the form ``fasterrag index tenants --json`` prints."""
        return {
            "collection": self.collection,
            "scanned": self.scanned,
            "untagged": self.untagged,
            "tagged": dict(self.tagged),
        }


@dataclass(frozen=True, slots=True)
class UntaggedDocument:
    """One document whose chunks carry no tenant tag, as an operator has to judge it."""

    collection: str
    document_id: str
    source_uri: str
    chunks: int

    def as_dict(self) -> dict[str, Any]:
        """Return the row ``fasterrag index tenants --untagged`` writes."""
        return {
            "collection": self.collection,
            "document_id": self.document_id,
            "source_uri": self.source_uri,
            "chunks": self.chunks,
        }


@dataclass(frozen=True, slots=True)
class BackfillProgress:
    """Cumulative counts after one flushed batch, so a killed run still reported progress."""

    updated: int
    batches: int

    def as_dict(self) -> dict[str, Any]:
        """Return the counts a partial run reports."""
        return {"updated": self.updated, "batches": self.batches}


@dataclass(frozen=True, slots=True)
class BackfillReport:
    """What a backfill did, or what it would do — the numbers a human reads before enabling."""

    collection: str
    tenant: str
    applied: bool
    overwrite: bool
    scanned: int
    untagged: int
    already: int
    foreign: int
    updated: int
    batches: int
    assumed: bool = False

    @property
    def pending(self) -> int:
        """Return how many points this run would still leave for the next one.

        Another tenant's points count only under ``overwrite``, which claims them; under
        ``assumed`` they are left where they are on purpose and are not outstanding work.
        """
        claimed = self.foreign if self.overwrite else 0
        return self.untagged + claimed - self.updated

    def as_dict(self) -> dict[str, Any]:
        """Return the form ``fasterrag index backfill-tenant --json`` prints."""
        return {
            "collection": self.collection,
            "tenant": self.tenant,
            "applied": self.applied,
            "overwrite": self.overwrite,
            "assumed": self.assumed,
            "scanned": self.scanned,
            "untagged": self.untagged,
            "already": self.already,
            "foreign": self.foreign,
            "updated": self.updated,
            "batches": self.batches,
            "pending": self.pending,
        }


async def audit_collection(
    adapter: VectorDBAdapter, collection: str, *, batch_size: int = DEFAULT_BATCH_SIZE
) -> TenantAudit:
    """Count a collection's points by tenant tag, reading vectors for none of them."""
    scanned = 0
    untagged = 0
    tagged: dict[str, int] = {}

    async for point in adapter.iterate_points(
        collection, with_vectors=False, batch_size=batch_size
    ):
        scanned += 1
        tag = _tag_of(point.payload)
        if tag is None:
            untagged += 1
        else:
            tagged[tag] = tagged.get(tag, 0) + 1

    return TenantAudit(collection=collection, scanned=scanned, untagged=untagged, tagged=tagged)


async def audit_collections(
    adapter: VectorDBAdapter, *, batch_size: int = DEFAULT_BATCH_SIZE
) -> list[TenantAudit]:
    """Audit every collection the backend holds, so the untagged ones are discoverable."""
    return [
        await audit_collection(adapter, info.name, batch_size=batch_size)
        for info in await adapter.list_collections()
    ]


async def inventory_untagged(
    adapter: VectorDBAdapter, collection: str, *, batch_size: int = DEFAULT_BATCH_SIZE
) -> list[UntaggedDocument]:
    """List a collection's untagged points grouped by the document they were chunked from."""
    counts: dict[str, int] = {}
    sources: dict[str, str] = {}

    async for point in adapter.iterate_points(
        collection, with_vectors=False, batch_size=batch_size
    ):
        if _tag_of(point.payload) is not None:
            continue
        document_id = str(point.payload.get("document_id", ""))
        counts[document_id] = counts.get(document_id, 0) + 1
        sources.setdefault(document_id, str(point.payload.get("source_uri", "")))

    return [
        UntaggedDocument(
            collection=collection,
            document_id=document_id,
            source_uri=sources[document_id],
            chunks=chunks,
        )
        for document_id, chunks in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _refuse_foreign(audit: TenantAudit, tenant: str) -> None:
    """Refuse a collection holding another tenant's points, naming them."""
    foreign = sorted(name for name in audit.tagged if name != tenant)
    if not foreign:
        return
    raise FasterRagError(
        f"collection {audit.collection!r} already holds points tagged for "
        f"{foreign!r}, so its {audit.untagged} untagged points cannot be attributed to "
        f"{tenant!r}; inspect them with `index tenants --untagged`, then pass "
        f"assume_untagged to tag only those for {tenant!r}, or overwrite to claim the "
        "whole collection",
        code=ErrorCode.CONFLICT,
        retryable=False,
    )


def _refuse_contradictory_assertions(overwrite: bool, assume_untagged: bool) -> None:
    """Refuse the two ownership assertions together, since they disagree."""
    if overwrite and assume_untagged:
        raise FasterRagError(
            "overwrite and assume_untagged assert different things about the same "
            "collection — that all of it belongs to this tenant, or that only its "
            "untagged points do; pass exactly one",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )


async def backfill_tenant(
    adapter: VectorDBAdapter,
    collection: str,
    tenant: str,
    *,
    apply: bool = False,
    overwrite: bool = False,
    assume_untagged: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: Callable[[BackfillProgress], None] | None = None,
) -> BackfillReport:
    """Write the tenant tag onto a collection's points, or report what that would write.

    Refuses a collection holding another tenant unless ``overwrite`` or ``assume_untagged``
    asserts who owns it; ``apply`` is what turns the dry run into a write.
    """
    validated_tenant(tenant)
    _refuse_contradictory_assertions(overwrite, assume_untagged)
    audit = await audit_collection(adapter, collection, batch_size=batch_size)

    if not audit.scanned:
        raise FasterRagError(
            f"collection {collection!r} yielded no points; there is nothing to backfill, and "
            "a collection that does not exist reads the same way as an empty one here",
            code=ErrorCode.NOT_FOUND,
            retryable=False,
        )

    if not overwrite and not assume_untagged:
        _refuse_foreign(audit, tenant)

    already = audit.tagged.get(tenant, 0)
    foreign = audit.scanned - already - audit.untagged
    report = BackfillReport(
        collection=collection,
        tenant=tenant,
        applied=apply,
        overwrite=overwrite,
        assumed=assume_untagged,
        scanned=audit.scanned,
        untagged=audit.untagged,
        already=already,
        foreign=foreign,
        updated=0,
        batches=0,
    )

    if not apply:
        return report

    updated, batches = await _write(
        adapter,
        collection,
        tenant,
        overwrite=overwrite,
        batch_size=batch_size,
        progress=progress,
    )
    _logger.info(
        "tenant backfill applied",
        extra={
            "collection": collection,
            "tenant": tenant,
            "updated": updated,
            "batches": batches,
            "overwrite": overwrite,
            "assumed": assume_untagged,
        },
    )
    return BackfillReport(
        collection=report.collection,
        tenant=report.tenant,
        applied=True,
        overwrite=overwrite,
        assumed=assume_untagged,
        scanned=report.scanned,
        untagged=report.untagged,
        already=report.already,
        foreign=report.foreign,
        updated=updated,
        batches=batches,
    )


async def _write(
    adapter: VectorDBAdapter,
    collection: str,
    tenant: str,
    *,
    overwrite: bool,
    batch_size: int,
    progress: Callable[[BackfillProgress], None] | None,
) -> tuple[int, int]:
    """Send one metadata-only update per point that needs the tag, in flushed batches."""
    pending: list[PointUpdate] = []
    updated = 0
    batches = 0

    async def flush() -> None:
        """Write the accumulated batch and report cumulative progress."""
        nonlocal updated, batches
        if not pending:
            return
        await adapter.update(list(pending))
        updated += len(pending)
        batches += 1
        pending.clear()
        if progress is not None:
            progress(BackfillProgress(updated=updated, batches=batches))

    async for point in adapter.iterate_points(
        collection, with_vectors=False, batch_size=batch_size
    ):
        tag = _tag_of(point.payload)
        if tag == tenant or (tag is not None and not overwrite):
            continue
        pending.append(
            PointUpdate(
                point_id=point.point_id,
                collection=collection,
                payload={TENANT_PAYLOAD_KEY: tenant},
            )
        )
        if len(pending) >= batch_size:
            await flush()

    await flush()
    return updated, batches
