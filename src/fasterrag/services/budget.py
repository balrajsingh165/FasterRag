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

# CRITICAL: the per-tenant budget is per replica, exactly like the per-key rate limiter
# (TASK-0216). N replicas therefore grant N times the configured budget, and for a spend cap
# that error runs in the dangerous direction — it permits more than was asked for, quietly.
# It is shipped this way rather than withheld because a single-replica deployment gets
# precisely what it configured, and because refusing to start was the previous behaviour and
# taught operators nothing. Anyone running replicas must size the value accordingly until a
# shared counter lands. The per-query budget has no such caveat.

A refused request raises ``BUDGET_EXCEEDED``, which the problem table renders as 402 and
marks **non-retryable**. That last part is load-bearing: the degradation ladder in
``GenerationService`` absorbs *retryable* failures into an extractive answer, so a retryable
budget error would silently serve a degraded answer instead of reporting the cap — turning a
spend control into a quality regression nobody could see.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Final

from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError

__all__ = ["ROLLING_WINDOW_SECONDS", "CostGovernor", "create_governor"]

# The documented budget is "rolling" without naming a period, and a rolling window means
# nothing without one. A day is the shortest period over which a spend cap is normally
# reasoned about, and it is stated in config-reference.md rather than left to be inferred.
ROLLING_WINDOW_SECONDS: Final = 86_400.0

_UNLIMITED: Final = 0


@dataclass
class CostGovernor:
    """Enforces the per-query and per-tenant token budgets."""

    per_query: int = _UNLIMITED
    per_tenant: int = _UNLIMITED
    window_seconds: float = ROLLING_WINDOW_SECONDS
    _spent: dict[str, deque[tuple[float, int]]] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        """Return whether any budget is set, so callers can skip the work entirely."""
        return self.per_query > _UNLIMITED or self.per_tenant > _UNLIMITED

    def check(self, tokens: int, *, tenant: str | None = None, now: float | None = None) -> None:
        """Refuse a request that would exceed either budget.

        Args:
            tokens: The worst case this request can cost — prompt plus ``llm.max_tokens``.
            tenant: The tenant to charge, or ``None`` for a single-tenant deployment. All
                untenanted traffic shares one bucket, which is what a deployment with
                ``security.multi_tenancy: false`` means by a per-tenant budget.
            now: Monotonic clock override, for tests.

        Raises:
            FasterRagError: ``BUDGET_EXCEEDED`` when the request does not fit. The message
                names which budget refused it and by how much, because "budget exceeded"
                alone leaves an operator unable to tell a too-large query from an exhausted
                account.
        """
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
        already = self._spent_since(self._bucket(tenant), moment)
        if already + tokens > self.per_tenant:
            raise FasterRagError(
                f"{self._describe(tenant)} has spent {already} tokens of the "
                f"cost.per_tenant_token_budget of {self.per_tenant} in the last "
                f"{int(self.window_seconds / 3600)}h, and this query needs up to {tokens} "
                "more; the budget is a rolling window, so it recovers without intervention",
                code=ErrorCode.BUDGET_EXCEEDED,
            )

    def record(self, tokens: int, *, tenant: str | None = None, now: float | None = None) -> None:
        """Charge a completed request's actual token usage to its tenant.

        Recorded after the call rather than at the check, so a tenant is charged what the
        query really cost rather than the ceiling it was admitted against. Admitting on the
        worst case and charging the real one is the combination that neither overspends nor
        over-refuses.
        """
        if self.per_tenant <= _UNLIMITED or tokens <= 0:
            return

        moment = time.monotonic() if now is None else now
        window = self._bucket(tenant)
        self._spent_since(window, moment)
        window.append((moment, tokens))

    def spent(self, tenant: str | None = None, *, now: float | None = None) -> int:
        """Return a tenant's spend inside the current window, for tests and diagnostics."""
        moment = time.monotonic() if now is None else now
        return self._spent_since(self._bucket(tenant), moment)

    def _bucket(self, tenant: str | None) -> deque[tuple[float, int]]:
        """Return the recorded spend for one tenant, creating it on first use."""
        return self._spent.setdefault(tenant or "", deque())

    def _spent_since(self, window: deque[tuple[float, int]], moment: float) -> int:
        """Drop everything older than the window and return what is left."""
        while window and moment - window[0][0] >= self.window_seconds:
            window.popleft()
        return sum(tokens for _, tokens in window)

    @staticmethod
    def _describe(tenant: str | None) -> str:
        """Name the charged party without quoting an unset tenant as an empty string."""
        return f"tenant {tenant!r}" if tenant else "this deployment"


def create_governor(settings: Settings) -> CostGovernor:
    """Build the governor from ``cost.*``.

    Always returned, never ``None``: a governor with both budgets at their ``0`` default
    permits everything, and a caller that has to decide whether it has one is a caller that
    can forget to ask.
    """
    return CostGovernor(
        per_query=settings.cost.per_query_token_budget,
        per_tenant=settings.cost.per_tenant_token_budget,
    )
