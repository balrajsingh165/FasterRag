"""The D9 runtime cost governor: token budgets enforced before a provider is called.

The estimator half of D9 answers "what would ingesting this cost?" before any of it runs.
This is the other half, and it answers a different question at a different moment: "is this
query allowed to spend what it is about to spend?" — asked per request, against a cap an
operator set.

**Budgets are checked against the worst case, not the likely one.** A completion's real token
count is only known once the provider has produced it, and by then the money is spent. The
check therefore uses the prompt's actual size plus ``llm.max_tokens``, which is the ceiling
the call can reach. That over-counts a query that finishes early, and that is the intended
direction: a cap that admits a request it cannot pay for is not a cap.

Two budgets, with deliberately different characters:

* ``cost.per_query_token_budget`` bounds one request. It is exact, stateless, and identical
  on every replica — nothing accumulates, so nothing can disagree.
* ``cost.per_tenant_token_budget`` bounds a tenant's spend over a rolling window. That needs
  memory of what has already been spent, and this implementation keeps it **in this process**.

**The window is bucketed, and the approximation runs one way only** (TASK-0366). Until
TASK-0366 a tenant's window held one ``(timestamp, tokens)`` tuple per charge, bounded only
by the traffic inside 24 hours. Measured rather than estimated (BENCH-0005): **122.5 bytes
per charge**, so one tenant at 1M charges a day held **116.8 MiB** and every ``check``
summed the lot — **128 ms** at that size, on the request path, against **7.5 µs** bucketed.
Worse, the old ``record`` pruned by calling ``_spent_since``, which also summed and threw
the total away, so one charge cost O(entries) and a day of traffic was quadratic. Spend is
now accumulated into buckets of ``window_seconds / 24`` — one hour at the shipped 24-hour
window — so a tenant holds at most 26 of them, **4,862 bytes measured**, whatever its
volume.

The exactness that buys: a bucket is evicted when the **newest** charge in it has aged out,
so no charge is ever forgotten early and the counted spend is never lower than the true
spend. The error is retention of up to one bucket too long, so a tenant can stay refused up
to an hour after an exact window would have released its budget, and can never be admitted
past the cap. For a control whose purpose is preventing runaway
spend that is the direction to err in, and it is stated in config-reference.md because an
operator who set a budget expecting exactness has to be able to read it.

# CRITICAL: the per-tenant budget is per replica, exactly like the per-key rate limiter
# (TASK-0216). N replicas therefore grant N times the configured budget, and for a spend cap
# that error runs in the dangerous direction — it permits more than was asked for, quietly.
# It is shipped this way rather than withheld because a single-replica deployment gets
# precisely what it configured, and because refusing to start was the previous behaviour and
# taught operators nothing. Anyone running replicas must size the value accordingly until a
# shared counter lands. The per-query budget has no such caveat.
#
# CRITICAL: "per replica" only holds while every spender inside one replica shares one
# instance, which is what `shared_governor` is for. Until TASK-0337 the generation service
# built its own through `create_governor` on every request, so the real multiplier was one
# window per request and the ceiling could not be reached at all — a cap that cannot be
# reached is not a weak cap, it is an absent one. `create_governor` therefore stays an
# explicitly unshared constructor for tests and injection; production paths call
# `shared_governor`, and the embedding pool and the generation service both do.

A refused request raises ``BUDGET_EXCEEDED``, which the problem table renders as 402 and
marks **non-retryable**. That last part is load-bearing: the degradation ladder in
``GenerationService`` absorbs *retryable* failures into an extractive answer, so a retryable
budget error would silently serve a degraded answer instead of reporting the cap — turning a
spend control into a quality regression nobody could see.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, fields
from typing import Final

from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError

__all__ = [
    "BUCKETS_PER_WINDOW",
    "ROLLING_WINDOW_SECONDS",
    "CostGovernor",
    "create_governor",
    "reset_shared_governors",
    "shared_governor",
]

ROLLING_WINDOW_SECONDS: Final = 86_400.0

BUCKETS_PER_WINDOW: Final = 24

_UNLIMITED: Final = 0

# CRITICAL: a floor, so a governor configured with a zero or negative window divides by
# something. Such a window expires every charge immediately, which is what the old
# per-timestamp code did with it; raising ZeroDivisionError inside a spend check would turn
# a nonsensical budget into a 500 on the query path.
_SMALLEST_BUCKET_SECONDS: Final = 1e-6

# CRITICAL: the one field that is accumulated spend rather than configuration. Keying a
# shared governor on it would make every instance unique and share nothing.
_STATE_FIELDS: Final[frozenset[str]] = frozenset({"_spent"})


@dataclass
class CostGovernor:
    """Enforces the per-query and per-tenant token budgets."""

    per_query: int = _UNLIMITED
    per_tenant: int = _UNLIMITED
    window_seconds: float = ROLLING_WINDOW_SECONDS
    _spent: dict[str, dict[int, tuple[float, int]]] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        """Return whether any budget is set, so callers can skip the work entirely."""
        return self.per_query > _UNLIMITED or self.per_tenant > _UNLIMITED

    @property
    def bucket_seconds(self) -> float:
        """Return the span one bucket covers: an hour at the shipped 24-hour window.

        Derived rather than configured, so it cannot drift from ``window_seconds`` and
        cannot become a fourth field in ``identity``.
        """
        return max(self.window_seconds / BUCKETS_PER_WINDOW, _SMALLEST_BUCKET_SECONDS)

    @property
    def identity(self) -> tuple[object, ...]:
        """Return every configured field, which is what a shared instance is keyed on.

        Read off the dataclass rather than listed by hand, so a field added here joins the
        key instead of letting two differently-configured governors quietly become one.
        """
        return tuple(
            getattr(self, declared.name)
            for declared in fields(self)
            if declared.name not in _STATE_FIELDS
        )

    def check(self, tokens: int, *, tenant: str | None = None, now: float | None = None) -> None:
        """Refuse a request that would exceed either budget."""
        if self.per_query > _UNLIMITED and tokens > self.per_query:
            raise FasterRagError(
                f"this query needs up to {tokens} tokens, above the "
                f"cost.per_query_token_budget of {self.per_query}; shorten the question, "
                "lower retrieval.top_k, or lower llm.max_tokens",
                code=ErrorCode.BUDGET_EXCEEDED,
            )

        if self.per_tenant <= _UNLIMITED:
            return

        moment = time.monotonic() if now is None else now
        already = self._spent_since(_key(tenant), moment)
        if already + tokens > self.per_tenant:
            raise FasterRagError(
                f"{self._describe(tenant)} has spent {already} tokens of the "
                f"cost.per_tenant_token_budget of {self.per_tenant} in the last "
                f"{int(self.window_seconds / 3600)}h, and this query needs up to {tokens} "
                "more; the budget is a rolling window, so it recovers without intervention",
                code=ErrorCode.BUDGET_EXCEEDED,
            )

    def record(self, tokens: int, *, tenant: str | None = None, now: float | None = None) -> None:
        """Charge a completed request's actual token usage to its tenant."""
        if self.per_tenant <= _UNLIMITED or tokens <= 0:
            return

        moment = time.monotonic() if now is None else now
        key = _key(tenant)
        buckets = self._alive(key, moment)
        if buckets is None:
            buckets = self._spent.setdefault(key, {})

        # CRITICAL: the bucket keeps the NEWEST moment charged into it, and `_alive` evicts
        # on that. Keeping the oldest, or the bucket's own start, would forget a charge
        # before its window ended and let a tenant spend past the cap — the one direction a
        # spend control may not be wrong in.
        index = int(moment // self.bucket_seconds)
        newest, total = buckets.get(index, (moment, 0))
        buckets[index] = (moment if moment > newest else newest, total + tokens)

    def spent(self, tenant: str | None = None, *, now: float | None = None) -> int:
        """Return a tenant's spend inside the current window, for tests and diagnostics."""
        moment = time.monotonic() if now is None else now
        return self._spent_since(_key(tenant), moment)

    def _spent_since(self, key: str, moment: float) -> int:
        """Drop everything outside the window and return the tokens left in it."""
        buckets = self._alive(key, moment)
        if buckets is None:
            return 0
        return sum(tokens for _, tokens in buckets.values())

    def _alive(self, key: str, moment: float) -> dict[int, tuple[float, int]] | None:
        """Evict every bucket wholly outside the window; ``None`` once none is left."""
        buckets = self._spent.get(key)
        if buckets is None:
            return None

        for index in [
            index
            for index, (newest, _) in buckets.items()
            if moment - newest >= self.window_seconds
        ]:
            del buckets[index]

        if not buckets:
            del self._spent[key]
            return None
        return buckets

    @staticmethod
    def _describe(tenant: str | None) -> str:
        """Name the charged party without quoting an unset tenant as an empty string."""
        return f"tenant {tenant!r}" if tenant else "this deployment"


def _key(tenant: str | None) -> str:
    """Return the name a tenant's spend is recorded under; unset traffic shares one."""
    return tenant or ""


def create_governor(settings: Settings) -> CostGovernor:
    """Build an **unshared** governor from ``cost.*``, for tests and explicit injection."""
    return CostGovernor(
        per_query=settings.cost.per_query_token_budget,
        per_tenant=settings.cost.per_tenant_token_budget,
    )


_SHARED: Final[dict[tuple[object, ...], CostGovernor]] = {}


def shared_governor(settings: Settings) -> CostGovernor:
    """Return the process-wide governor for these budgets, creating it on first use.

    A rolling per-tenant window is memory, and memory a caller rebuilds per unit of work
    remembers nothing; sharing one instance is what makes spend accumulate at all.
    """
    # CRITICAL: the candidate is built first and keyed on its own `identity`, so the cache
    # key can never cover less than the governor's configuration. Keying on a hand-written
    # subset of `cost.*` was correct only by coincidence, and keying on the whole `cost`
    # section would split the window over `cost.estimator` — two windows, twice the budget,
    # in the direction that permits more than was configured.
    candidate = create_governor(settings)
    governor = _SHARED.get(candidate.identity)
    if governor is None:
        governor = candidate
        _SHARED[candidate.identity] = governor
    return governor


def reset_shared_governors() -> None:
    """Drop every shared governor, so a test or a reload starts from an unspent window."""
    _SHARED.clear()
