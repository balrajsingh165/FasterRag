"""Zero-downtime reindexing (D2).

Re-embedding a corpus must never take queries down. The green collection is built alongside
the live one, validated against the eval set, and then made live by pointing an alias at it —
one atomic operation. Queries in flight during the build keep hitting blue and neither know
nor care that a rebuild is happening.

**The alias is the whole trick.** Callers query a *name*; that name resolves to whichever
physical collection is current. Without the indirection the only way to replace an index is
to drop and re-ingest, which is the hours of downtime this exists to remove.

**The eval gate runs before the swap, not after.** A gate that ran after would detect a
regression by serving it. When ``index.reindex.eval_gate`` is on and the new index scores
worse than the committed baseline, the swap does not happen and blue stays live.

The previous collection is retained for ``index.reindex.rollback_retention_hours`` so a
rollback is an alias flip rather than another rebuild — the difference between seconds and
hours when an index turns out to be wrong in a way the eval set did not catch.

**A keyword leg can only be added by rebuilding** (TASK-0119). Qdrant will not mix a named
sparse vector with an unnamed dense one, so a dense-only collection cannot gain one; the
pgvector and Chroma adapters refuse the same mismatch rather than bolt a keyword leg onto a
populated collection. Turning ``retrieval.hybrid`` on over an indexed corpus is therefore a
*reindex*, not a restart, and :func:`audit_keyword_leg` is the preflight that says so before
a write discovers it.

**The gate sits on the write path, not the query path.** ``POST /v1/ingest`` and its
retry-DLQ sibling call :func:`require_keyword_leg` before a job record exists, so a
misconfigured run is refused with a 409 that names the rebuild rather than accepted with a
``202`` and then dying document by document inside a background task the caller never
watches. ``fasterrag index create`` calls it after creating, which is the only way the CLI
can tell that a backend accepted a create whose keyword leg it did not actually add. The
cost is one metadata round trip per job — noise beside a corpus that takes minutes to
hours, where the same check per query would charge every question for a mistake made once
and report it at 3am to somebody who cannot act on it.

It reads live state through ``CollectionInfo.sparse``, which every adapter reports from the
server rather than from what creation was *asked* for, so the verdict is right even on a
backend whose ``create_collection`` waved the mismatch through (TASK-0305). Detection needs
no new adapter method and no new config key. It is a live-backend call and may raise, like
any other: a caller that treats "unreachable" as "misconfigured" turns a transient outage
into a refusal, so a caller that must not fail closed — a startup probe, a dashboard panel
— uses :func:`audit_keyword_leg` and carries on with the verdict unknown.

**The verdict names the collection the way its caller does** (TASK-0324). A multi-tenant
``POST /v1/collections`` must audit the tenant-scoped backend name — that is the collection
the create touched — while its refusal is an API response body, where the scope prefix is
the one thing `api/collections` promises never appears. The lookup name and the shown name
are therefore separate arguments rather than one, and ``display`` supplies the second.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final

from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.lockfile import LockStore

__all__ = [
    "GREEN_SUFFIX",
    "KeywordLegAudit",
    "ReindexPlan",
    "ReindexResult",
    "RollbackResult",
    "audit_keyword_leg",
    "green_name",
    "plan_reindex",
    "require_keyword_leg",
    "retire",
    "rollback",
    "swap",
]

# CRITICAL: the physical name must be derivable and unique per build, because a rebuild that
# reused one name could not exist alongside the live collection — which is the entire point
# of blue/green. The timestamp makes successive rebuilds distinguishable and orderable.
GREEN_SUFFIX: Final = "-v"

_logger = get_logger(__name__)


def green_name(alias: str, stamp: str | None = None) -> str:
    """Return the physical collection name for a new build behind ``alias``."""
    moment = stamp or datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    return f"{alias}{GREEN_SUFFIX}{moment}"


def _named_for_caller(backend: str, lookup: str, display: str) -> str:
    """Return a backend collection name under the naming its caller used.

    The tenant prefix is a backend detail, so a derived name keeps whatever suffix
    distinguishes it while the part the caller never chose is replaced (TASK-0324).
    """
    if backend.startswith(lookup):
        return f"{display}{backend[len(lookup) :]}"
    return backend if lookup == display else display


@dataclass(frozen=True, slots=True)
class KeywordLegAudit:
    """Whether the live index behind a served name can answer the configured keyword leg.

    ``name`` and ``live`` carry the caller's own naming rather than the backend's, because
    :attr:`detail` reaches an API response body and a scope prefix must not (TASK-0324).
    """

    name: str
    live: str | None
    required: bool
    present: bool

    @property
    def missing(self) -> bool:
        """Return whether hybrid retrieval is configured over an index that cannot serve it."""
        return self.required and self.live is not None and not self.present

    @property
    def detail(self) -> str:
        """Return the operator-facing explanation, naming the collection and the way out."""
        if not self.missing:
            return f"{self.name!r} matches the configured retrieval mode"
        served = "" if self.live == self.name else f" (served as {self.name!r})"
        return (
            f"retrieval.hybrid is on, but collection {self.live!r}{served} "
            "was indexed without a keyword leg, and a sparse vector cannot be attached to an "
            "existing collection. Rebuild it with "
            f"`fasterrag index reembed {self.name} <sources>`, which builds a hybrid "
            "collection alongside the live one and swaps the alias, or set retrieval.hybrid "
            "to false to keep serving dense-only results knowingly"
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized verdict."""
        return {
            "name": self.name,
            "live": self.live,
            "required": self.required,
            "present": self.present,
            "missing": self.missing,
            "detail": self.detail,
        }


async def audit_keyword_leg(
    name: str, settings: Settings, adapter: VectorDBAdapter, *, display: str | None = None
) -> KeywordLegAudit:
    """Report whether the index behind ``name`` carries the configured keyword leg.

    Resolves ``name`` through any alias; ``display`` is the caller's own name for a scoped
    ``name`` and is what the verdict shows. A name nothing has indexed yet is no mismatch.
    """
    shown = display or name
    target = await adapter.alias_target(name) or name
    live = next((info for info in await adapter.list_collections() if info.name == target), None)
    audit = KeywordLegAudit(
        name=shown,
        live=_named_for_caller(live.name, name, shown) if live is not None else None,
        required=settings.retrieval.hybrid,
        present=live is not None and live.sparse,
    )
    if audit.missing:
        _logger.warning(
            "the live index has no keyword leg but hybrid retrieval is configured",
            extra={"alias": name, "live": live.name if live is not None else None},
        )
    return audit


async def require_keyword_leg(
    name: str, settings: Settings, adapter: VectorDBAdapter, *, display: str | None = None
) -> KeywordLegAudit:
    """Refuse a configuration whose keyword leg the live index cannot serve.

    For ``fasterrag doctor`` and the CLI, where a human is waiting and can act on it.
    Startup calls :func:`audit_keyword_leg` instead, which warns without refusing to boot.
    """
    audit = await audit_keyword_leg(name, settings, adapter, display=display)
    if audit.missing:
        raise FasterRagError(audit.detail, code=ErrorCode.CONFLICT, retryable=False)
    return audit


@dataclass(frozen=True, slots=True)
class ReindexPlan:
    """What a reindex intends to do, before it does any of it."""

    alias: str
    blue: str | None
    green: str
    strategy: str
    eval_gate: bool
    rollback_retention_hours: int

    @property
    def first_build(self) -> bool:
        """Return whether there is no live collection to fall back to."""
        return self.blue is None

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized plan."""
        return {
            "alias": self.alias,
            "blue": self.blue,
            "green": self.green,
            "strategy": self.strategy,
            "eval_gate": self.eval_gate,
            "rollback_retention_hours": self.rollback_retention_hours,
            "first_build": self.first_build,
        }


@dataclass(frozen=True, slots=True)
class ReindexResult:
    """The outcome of a reindex."""

    plan: ReindexPlan
    swapped: bool
    reason: str = ""
    gate_ran: bool = False
    eval_report: dict[str, Any] = field(default_factory=dict)
    swap_ms: int = 0

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized outcome."""
        return {
            **self.plan.as_dict(),
            "swapped": self.swapped,
            "reason": self.reason,
            "gate_ran": self.gate_ran,
            "eval": self.eval_report,
            "swap_ms": self.swap_ms,
        }


@dataclass(frozen=True, slots=True)
class RollbackResult:
    """The outcome of a rollback."""

    alias: str
    restored: str
    replaced: str | None
    as_of: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return the serialized outcome."""
        return {
            "alias": self.alias,
            "restored": self.restored,
            "replaced": self.replaced,
            "as_of": self.as_of,
        }


async def plan_reindex(
    alias: str, settings: Settings, adapter: VectorDBAdapter, *, stamp: str | None = None
) -> ReindexPlan:
    """Decide what a reindex of ``alias`` would build and replace."""
    blue = await adapter.alias_target(alias)
    if blue is None:
        existing = {info.name for info in await adapter.list_collections()}
        if alias in existing:
            raise FasterRagError(
                f"{alias!r} is a physical collection, not an alias; zero-downtime reindexing "
                "needs the served name to be an alias, so migrate it with an explicit "
                "reindex into a new name first",
                code=ErrorCode.CONFLICT,
                retryable=False,
            )

    reindex = settings.index.reindex
    return ReindexPlan(
        alias=alias,
        blue=blue,
        green=green_name(alias, stamp),
        strategy=reindex.strategy,
        eval_gate=reindex.eval_gate,
        rollback_retention_hours=reindex.rollback_retention_hours,
    )


async def swap(
    plan: ReindexPlan,
    adapter: VectorDBAdapter,
    *,
    eval_passed: bool | None = True,
    eval_report: dict[str, Any] | None = None,
) -> ReindexResult:
    """Point the alias at the newly built collection, if the gate allows it."""
    report = eval_report or {}
    gate_ran = plan.eval_gate and eval_passed is not None

    if plan.eval_gate and eval_passed is False:
        _logger.warning(
            "the eval gate blocked the alias swap; the previous index is still live",
            extra={"alias": plan.alias, "blue": plan.blue, "green": plan.green},
        )
        return ReindexResult(
            plan=plan,
            swapped=False,
            reason="the eval gate blocked the swap; the new index scored worse than the baseline",
            eval_report=report,
        )

    started = time.perf_counter()
    await adapter.set_alias(plan.alias, plan.green)
    swap_ms = int((time.perf_counter() - started) * 1000)

    reason = (
        ""
        if gate_ran or not plan.eval_gate
        else "the eval gate could not run; the swap proceeded ungated"
    )
    _logger.info(
        "alias swapped to the new index",
        extra={
            "alias": plan.alias,
            "blue": plan.blue,
            "green": plan.green,
            "swap_ms": swap_ms,
            "gate_ran": gate_ran,
            "retained_hours": plan.rollback_retention_hours,
        },
    )
    return ReindexResult(
        plan=plan,
        swapped=True,
        reason=reason,
        gate_ran=gate_ran,
        eval_report=report,
        swap_ms=swap_ms,
    )


async def rollback(
    alias: str, adapter: VectorDBAdapter, *, to: str | None = None
) -> RollbackResult:
    """Point the alias back at the previous build."""
    current = await adapter.alias_target(alias)
    target = to

    if target is None:
        prefix = f"{alias}{GREEN_SUFFIX}"
        builds = sorted(
            info.name
            for info in await adapter.list_collections()
            if info.name.startswith(prefix) and info.name != current
        )
        target = builds[-1] if builds else None

    if target is None:
        raise FasterRagError(
            f"no retained build to roll {alias!r} back to; the retention window may have "
            "expired, or this may be the first build",
            code=ErrorCode.NOT_FOUND,
            retryable=False,
        )

    await adapter.set_alias(alias, target)
    _logger.info(
        "rolled the alias back to a retained build",
        extra={"alias": alias, "restored": target, "replaced": current},
    )
    return RollbackResult(
        alias=alias,
        restored=target,
        replaced=current,
        as_of=datetime.now(tz=UTC).isoformat(),
    )


async def retire(
    alias: str,
    adapter: VectorDBAdapter,
    settings: Settings,
    *,
    locks: LockStore | None = None,
    now: float | None = None,
) -> list[str]:
    """Drop retained builds past the rollback window, returning what went."""
    retention_hours = settings.index.reindex.rollback_retention_hours
    current = await adapter.alias_target(alias)
    prefix = f"{alias}{GREEN_SUFFIX}"
    cutoff = (now or time.time()) - retention_hours * 3600

    dropped: list[str] = []
    for info in await adapter.list_collections():
        if not info.name.startswith(prefix) or info.name == current:
            continue

        stamp = info.name[len(prefix) :]
        try:
            built = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=UTC).timestamp()
        except ValueError:
            _logger.warning(
                "a collection carrying the reindex prefix has no parseable build stamp, so "
                "it is left in place rather than deleted on a guess at its age; the "
                "rollback window is not being enforced over it",
                extra={
                    "alias": alias,
                    "collection": info.name,
                    "stamp": stamp,
                    "expected_format": "%Y%m%d%H%M%S",
                },
            )
            continue

        if built < cutoff and await adapter.drop_collection(info.name):
            dropped.append(info.name)
            if locks is not None:
                locks.delete(info.name)

    if dropped:
        _logger.info(
            "retired builds past the rollback window",
            extra={"alias": alias, "dropped": dropped, "retention_hours": retention_hours},
        )
    return dropped
