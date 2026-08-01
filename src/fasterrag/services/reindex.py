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
    "ReindexPlan",
    "ReindexResult",
    "RollbackResult",
    "green_name",
    "plan_reindex",
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
    """Decide what a reindex of ``alias`` would build and replace.

    Raises:
        FasterRagError: With ``CONFLICT`` when a physical collection already occupies the
            alias name. Blue/green needs the served name to be an alias; if it is a real
            collection, pointing an alias at it later would be ambiguous, and the operator
            has to migrate deliberately rather than have it done implicitly.
    """
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
    """Point the alias at the newly built collection, if the gate allows it.

    Args:
        plan: What the reindex set out to do.
        adapter: The vector database holding both collections.
        eval_passed: ``True`` when the gate approved the build, ``False`` when it rejected
            it, and ``None`` when the gate could not run at all. The three are deliberately
            distinct: a gate that could not run has established nothing, and recording that
            as a pass would let an operator believe a build was validated when it was not.
        eval_report: The gate's own report, carried into the result for the record.

    Returns:
        The outcome. A blocked swap is a *result*, not an exception: the reindex ran
        correctly and the gate did its job, which is a successful outcome of a different
        shape, and blue is still live.
    """
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
    """Point the alias back at the previous build.

    Args:
        alias: The served name.
        adapter: The vector database.
        to: The collection to restore. When omitted, the most recent retained build older
            than the current one is chosen.

    Returns:
        What was restored and what it replaced.

    Raises:
        FasterRagError: With ``NOT_FOUND`` when there is nothing to roll back to, which is
            the honest answer for a first build or a retention window that has expired —
            far better than silently pointing the alias at an arbitrary collection.
    """
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
    """Drop retained builds past the rollback window, returning what went.

    The live collection is never a candidate, whatever its age: a retention window is about
    how long a *replaced* build is kept, and dropping the one currently serving traffic
    would be an outage rather than a cleanup.
    """
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
            # A name that does not carry a parseable stamp was not created by this code,
            # and guessing its age in order to delete it would be reckless.
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
