"""Charge accounting and window arithmetic in the cost governor's own terms.

``test_budget.py`` states the contract and ``test_budget_sharing.py`` proves one instance
is shared. Both leave the governor at 100% line and branch coverage while a third of its
mutants survive (TASK-0367), because every case there charges one tenant once, at time
zero, against a default window. These are the cases that distinguish the arithmetic.
"""

import pytest

from fasterrag.errors import FasterRagError
from fasterrag.services.budget import ROLLING_WINDOW_SECONDS, CostGovernor


def test_the_rolling_window_is_twenty_four_hours_and_the_refusal_says_so() -> None:
    """The number is published to the operator, so it is part of the contract, not a detail.

    The refusal tells a caller how long to wait; a window length that drifted from the
    sentence describing it would send them back at the wrong time.
    """
    governor = CostGovernor(per_tenant=1000)
    governor.record(900, tenant="acme")

    with pytest.raises(FasterRagError) as caught:
        governor.check(900, tenant="acme")

    assert ROLLING_WINDOW_SECONDS == 24 * 3600
    assert "in the last 24h" in str(caught.value)


def test_a_budget_of_one_token_is_a_budget_rather_than_unlimited() -> None:
    """Only 0 means unlimited. A sentinel that crept up to 1 would exempt the tightest cap."""
    per_query = CostGovernor(per_query=1)
    per_tenant = CostGovernor(per_tenant=1)

    assert per_query.enabled is True
    assert per_tenant.enabled is True
    with pytest.raises(FasterRagError):
        per_query.check(2)
    with pytest.raises(FasterRagError):
        per_tenant.check(2, tenant="acme")


def test_a_tenant_exactly_on_its_budget_is_served_and_one_token_more_is_not() -> None:
    """The ceiling is reachable, like the per-query one; off by one here is money."""
    governor = CostGovernor(per_tenant=1000)
    governor.record(400, tenant="acme")

    governor.check(600, tenant="acme")
    with pytest.raises(FasterRagError):
        governor.check(601, tenant="acme")


def test_nothing_is_recorded_when_no_tenant_budget_is_configured() -> None:
    """The window exists to enforce a cap; keeping one nobody reads is a leak with no reader."""
    governor = CostGovernor(per_query=500)

    governor.record(400, tenant="acme")

    assert governor._spent == {}
    assert governor.spent("acme") == 0


def test_a_charge_of_zero_or_less_is_not_recorded() -> None:
    """A zero charge would create the tenant's entry, which is the leak the window avoids."""
    governor = CostGovernor(per_tenant=1000)

    governor.record(0, tenant="acme")
    governor.record(-50, tenant="acme")

    assert governor._spent == {}


def test_the_smallest_real_charge_is_still_charged() -> None:
    """The control for the case above: the guard must drop nothing a tenant actually spent."""
    governor = CostGovernor(per_tenant=1000)

    governor.record(1, tenant="acme")

    assert governor.spent("acme") == 1


def test_the_window_is_measured_forward_from_when_the_charge_landed() -> None:
    """Age is ``now - charged_at``. Recording at zero hides a sign error in that subtraction."""
    governor = CostGovernor(per_tenant=1000)
    governor.record(100, tenant="acme", now=1_000.0)

    assert governor.spent("acme", now=1_000.0 + ROLLING_WINDOW_SECONDS - 1) == 100
    assert governor.spent("acme", now=1_000.0 + ROLLING_WINDOW_SECONDS + 1) == 0


def test_a_charge_exactly_as_old_as_the_window_has_already_left_it() -> None:
    """The window is half-open, and the edge is the one moment nobody tests by accident."""
    governor = CostGovernor(per_tenant=1000)
    governor.record(100, tenant="acme", now=0.0)

    assert governor.spent("acme", now=ROLLING_WINDOW_SECONDS) == 0


def test_only_the_aged_part_of_a_window_is_dropped() -> None:
    """Expiring the whole window because its oldest charge aged out would refund the rest.

    Every other case charges a tenant once, so an implementation that dropped the tenant's
    whole record the moment any part of it expired passes all of them.
    """
    governor = CostGovernor(per_tenant=1000)
    governor.record(100, tenant="acme", now=0.0)
    governor.record(250, tenant="acme", now=ROLLING_WINDOW_SECONDS / 2)

    assert governor.spent("acme", now=ROLLING_WINDOW_SECONDS + 1) == 250
    assert "acme" in governor._spent


def test_the_window_length_is_the_one_configured_on_the_instance() -> None:
    """``window_seconds`` is keyed into the shared instance, so it has to be read from it."""
    governor = CostGovernor(per_tenant=1000, window_seconds=60.0)
    governor.record(900, tenant="acme", now=0.0)

    assert governor.spent("acme", now=59.0) == 900
    assert governor.spent("acme", now=61.0) == 0
    governor.check(900, tenant="acme", now=61.0)


def test_the_tenant_refusal_names_the_charged_party() -> None:
    """An operator reading the log needs to know whose budget ran out, not that one did."""
    governor = CostGovernor(per_tenant=1000)
    governor.record(900, tenant="acme")

    with pytest.raises(FasterRagError, match="acme") as caught:
        governor.check(900, tenant="acme")

    assert "this deployment" not in str(caught.value)


def test_the_untenanted_refusal_names_the_deployment_rather_than_an_empty_name() -> None:
    """``tenant ''`` reads as a bug report; the deployment is the charged party here."""
    governor = CostGovernor(per_tenant=1000)
    governor.record(900)

    with pytest.raises(FasterRagError, match="this deployment") as caught:
        governor.check(900)

    assert "''" not in str(caught.value)


def test_a_preflight_check_is_not_a_charge() -> None:
    """The ceiling is what a request is admitted against; the actual usage is what it pays.

    A check that charged would bill every short answer for ``llm.max_tokens`` and exhaust
    a budget nobody spent.
    """
    governor = CostGovernor(per_tenant=1000)

    governor.check(900, tenant="acme")

    assert governor.spent("acme") == 0
    assert governor._spent == {}
