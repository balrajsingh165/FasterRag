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
    "audit_collection",
    "audit_collections",
    "backfill_tenant",
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

    @property
    def pending(self) -> int:
        """Return how many points would still be invisible to the filter after this run."""
        return self.untagged + self.foreign - self.updated

    def as_dict(self) -> dict[str, Any]:
        """Return the form ``fasterrag index backfill-tenant --json`` prints."""
        return {
            "collection": self.collection,
            "tenant": self.tenant,
            "applied": self.applied,
            "overwrite": self.overwrite,
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


def _refuse_foreign(audit: TenantAudit, tenant: str) -> None:
    """Refuse a collection holding another tenant's points, naming them."""
    foreign = sorted(name for name in audit.tagged if name != tenant)
    if not foreign:
        return
    raise FasterRagError(
        f"collection {audit.collection!r} already holds points tagged for "
        f"{foreign!r}, so its {audit.untagged} untagged points cannot be attributed to "
        f"{tenant!r}; split the collection, or pass overwrite to relabel every point in it",
        code=ErrorCode.CONFLICT,
        retryable=False,
    )


async def backfill_tenant(
    adapter: VectorDBAdapter,
    collection: str,
    tenant: str,
    *,
    apply: bool = False,
    overwrite: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress: Callable[[BackfillProgress], None] | None = None,
) -> BackfillReport:
    """Write the tenant tag onto a collection's points, or report what that would write.

    Audits first and refuses a collection holding another tenant unless ``overwrite``.
    ``apply`` is what turns the dry run into a write; the counts are identical either way.
    """
    validated_tenant(tenant)
    audit = await audit_collection(adapter, collection, batch_size=batch_size)

    if not audit.scanned:
        raise FasterRagError(
            f"collection {collection!r} yielded no points; there is nothing to backfill, and "
            "a collection that does not exist reads the same way as an empty one here",
            code=ErrorCode.NOT_FOUND,
            retryable=False,
        )

    if not overwrite:
        _refuse_foreign(audit, tenant)

    already = audit.tagged.get(tenant, 0)
    foreign = audit.scanned - already - audit.untagged
    report = BackfillReport(
        collection=collection,
        tenant=tenant,
        applied=apply,
        overwrite=overwrite,
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
        },
    )
    return BackfillReport(
        collection=report.collection,
        tenant=report.tenant,
        applied=True,
        overwrite=overwrite,
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
