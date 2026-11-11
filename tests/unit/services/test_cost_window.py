"""The shape of the per-tenant spend window, and the direction its approximation errs in.

TASK-0337 made ``CostGovernor`` process-lifetime, which is what let the rolling window
actually accumulate — and turned a structure holding one tuple per charge into one bounded
only by a day's traffic (measured at 122.5 bytes per charge and 128 ms per ``check`` at 1M
charges, BENCH-0005). TASK-0366 replaced it with hourly buckets.

Bucketing trades exactness at the window edge for bounded memory, so the cases below pin
*which* exactness was traded. Two things must stay exact: the total counted inside the
window, and the guarantee that no charge is dropped before its window ends. What is
approximate is only how long a charge is remembered *after* that — up to one bucket too
long, which refuses a tenant early rather than letting it overspend.

``test_budget.py`` and ``test_budget_sharing.py`` own the budget's behaviour and the shared
instance; this file owns the window's structure.
"""

from hypothesis import given
from hypothesis import strategies as st

from fasterrag.services.budget import (
    BUCKETS_PER_WINDOW,
    ROLLING_WINDOW_SECONDS,
    CostGovernor,
)

HOUR = ROLLING_WINDOW_SECONDS / BUCKETS_PER_WINDOW


def exact_spend(charges: list[tuple[float, int]], moment: float) -> int:
    """Return what an unbucketed window would have counted, as the baseline to compare to."""
    return sum(
        tokens
        for when, tokens in charges
        if moment - when < ROLLING_WINDOW_SECONDS and when <= moment
    )


def test_a_bucket_spans_an_hour_at_the_shipped_window() -> None:
    """The figure config-reference.md promises operators, read off the code rather than it."""
    assert CostGovernor(per_tenant=1).bucket_seconds == 3600.0
    assert BUCKETS_PER_WINDOW == 24


def test_the_bucket_span_follows_the_window_rather_than_being_configured_beside_it() -> None:
    """A separately configured span could drift from the window and silently widen it."""
    assert CostGovernor(per_tenant=1, window_seconds=2400.0).bucket_seconds == 100.0


def test_a_days_traffic_is_held_in_a_bounded_number_of_buckets() -> None:
    """The whole point of TASK-0366: memory is O(tenants), not O(charges in the window).

    The bound is the window's bucket count plus the two partial buckets its ends can
    straddle; what matters is that it is a constant and not a function of ``charges``.
    """
    governor = CostGovernor(per_tenant=10**12)
    charges = 20_000
    step = ROLLING_WINDOW_SECONDS / charges

    for index in range(charges):
        governor.record(1, tenant="acme", now=index * step)

    assert len(governor._spent["acme"]) <= BUCKETS_PER_WINDOW + 2


def test_the_total_inside_the_window_stays_exact() -> None:
    """Bucketing approximates when spend is forgotten, never how much of it is counted."""
    governor = CostGovernor(per_tenant=10**12)
    charges = 5_000
    step = ROLLING_WINDOW_SECONDS / charges
    expected = 0

    for index in range(charges):
        tokens = 1 + index % 97
        governor.record(tokens, tenant="acme", now=index * step)
        expected += tokens

    assert governor.spent("acme", now=(charges - 1) * step) == expected


def test_no_charge_is_forgotten_before_its_window_ends() -> None:
    """The safety property: under-counting is what would let a tenant spend past the cap."""
    governor = CostGovernor(per_tenant=1000)
    governor.record(100, tenant="acme", now=0.0)
    governor.record(100, tenant="acme", now=HOUR - 1)

    assert governor.spent("acme", now=HOUR - 1 + ROLLING_WINDOW_SECONDS - 1) == 200


def test_spend_is_remembered_up_to_one_bucket_too_long_and_no_longer() -> None:
    """The exactness that was traded, stated as the bound it is held to.

    The first charge's own window ends at ``ROLLING_WINDOW_SECONDS`` and it is still
    counted there; everything in the bucket is gone one window after the bucket's newest
    charge, which bounds the over-retention at one bucket.
    """
    governor = CostGovernor(per_tenant=1000)
    governor.record(100, tenant="acme", now=0.0)
    governor.record(100, tenant="acme", now=HOUR - 1)

    assert governor.spent("acme", now=ROLLING_WINDOW_SECONDS) == 200
    assert governor.spent("acme", now=HOUR - 1 + ROLLING_WINDOW_SECONDS) == 0


def test_an_hour_old_burst_does_not_hold_a_later_hour_hostage() -> None:
    """Buckets expire independently, so a quiet tenant's window still drains hour by hour."""
    governor = CostGovernor(per_tenant=10_000)
    governor.record(900, tenant="acme", now=0.0)
    governor.record(900, tenant="acme", now=10 * HOUR)

    assert governor.spent("acme", now=ROLLING_WINDOW_SECONDS + HOUR) == 900
    assert governor.spent("acme", now=ROLLING_WINDOW_SECONDS + 11 * HOUR) == 0


def test_a_charge_backdated_into_a_bucket_does_not_shorten_its_life() -> None:
    """``now`` is an override, so out-of-order moments must not move a bucket's newest back."""
    governor = CostGovernor(per_tenant=1000)
    governor.record(100, tenant="acme", now=HOUR - 1)
    governor.record(100, tenant="acme", now=0.0)

    assert governor.spent("acme", now=ROLLING_WINDOW_SECONDS) == 200


def test_one_tenants_days_ingest_cannot_refuse_another_tenants_query() -> None:
    """Per-tenant separation, re-pinned at the volume bucketing exists to survive."""
    governor = CostGovernor(per_tenant=10_000)
    step = ROLLING_WINDOW_SECONDS / 5_000

    for index in range(5_000):
        governor.record(1_000, tenant="acme", now=index * step)

    governor.check(9_000, tenant="globex", now=ROLLING_WINDOW_SECONDS)
    assert governor.spent("globex", now=ROLLING_WINDOW_SECONDS) == 0


def test_every_tenant_name_is_dropped_once_its_buckets_age_out() -> None:
    """The TASK-0337 leak fix, re-pinned against the new structure.

    The map is keyed on client-chosen input, so a bucketed window that kept an empty
    per-tenant dictionary would leak exactly as the per-charge deque did.
    """
    governor = CostGovernor(per_tenant=1000)
    for index in range(500):
        governor.record(1, tenant=f"tenant-{index}", now=float(index))

    for index in range(500):
        governor.spent(f"tenant-{index}", now=index + ROLLING_WINDOW_SECONDS)

    assert governor._spent == {}


def test_a_nonsensical_window_expires_everything_instead_of_dividing_by_zero() -> None:
    """A zero window is a misconfiguration; raising inside a spend check would be a 500."""
    governor = CostGovernor(per_tenant=1000, window_seconds=0.0)

    governor.record(100, tenant="acme", now=10.0)

    assert governor.spent("acme", now=10.0) == 0
    assert governor._spent == {}


@given(
    charges=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=2 * ROLLING_WINDOW_SECONDS, allow_nan=False),
            st.integers(min_value=1, max_value=10_000),
        ),
        min_size=1,
        max_size=40,
    ),
    offset=st.floats(min_value=0.0, max_value=2 * ROLLING_WINDOW_SECONDS, allow_nan=False),
)
def test_the_bucketed_window_never_counts_less_than_an_exact_one(
    charges: list[tuple[float, int]], offset: float
) -> None:
    """The property the whole trade rests on: the error is over-counting, never under.

    Under-counting would let a tenant spend past a cap an operator set, which is the one
    direction a spend control may not be wrong in. Over-counting refuses early, visibly.
    """
    governor = CostGovernor(per_tenant=10**12)
    for when, tokens in sorted(charges):
        governor.record(tokens, tenant="acme", now=when)

    moment = max(when for when, _ in charges) + offset

    assert governor.spent("acme", now=moment) >= exact_spend(charges, moment)
