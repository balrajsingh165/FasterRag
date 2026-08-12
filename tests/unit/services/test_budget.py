"""The D9 runtime cost governor.

The estimator half of D9 answers what an ingest would cost before it runs. This is the other
half and asks a different question per request: may this query spend what it is about to?
"""

import pytest

from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.budget import ROLLING_WINDOW_SECONDS, CostGovernor, create_governor


def test_no_budget_permits_everything() -> None:
    """0 means unlimited, and it is the default, so an unconfigured deployment is unaffected."""
    governor = create_governor(Settings.model_validate({}))

    assert governor.enabled is False
    governor.check(10_000_000)


def test_a_query_above_the_per_query_budget_is_refused() -> None:
    governor = CostGovernor(per_query=500)

    with pytest.raises(FasterRagError) as caught:
        governor.check(501)

    assert caught.value.code is ErrorCode.BUDGET_EXCEEDED


def test_the_refusal_is_not_retryable() -> None:
    """The degradation ladder absorbs retryable failures into an extractive answer.

    A retryable budget error would therefore serve a degraded answer instead of reporting the
    cap — turning a spend control into a quality regression nobody could see.
    """
    governor = CostGovernor(per_query=10)

    with pytest.raises(FasterRagError) as caught:
        governor.check(11)

    assert caught.value.retryable is False


def test_the_refusal_names_the_budget_and_what_to_do() -> None:
    """A bare 'budget exceeded' cannot distinguish a huge query from an exhausted account."""
    governor = CostGovernor(per_query=500)

    with pytest.raises(FasterRagError, match=r"cost\.per_query_token_budget") as caught:
        governor.check(900)

    assert "900" in str(caught.value)


def test_a_query_exactly_on_the_budget_is_allowed() -> None:
    """The budget is a ceiling the query may reach, not one it must stay under."""
    CostGovernor(per_query=500).check(500)


def test_spend_accumulates_against_the_tenant_budget() -> None:
    governor = CostGovernor(per_tenant=1000)
    governor.record(600, tenant="acme")

    with pytest.raises(FasterRagError):
        governor.check(500, tenant="acme")


def test_one_tenant_cannot_exhaust_another_s_budget() -> None:
    """The isolation the setting's name promises; sharing one bucket would be the defect."""
    governor = CostGovernor(per_tenant=1000)
    governor.record(900, tenant="acme")

    governor.check(900, tenant="globex")
    assert governor.spent("globex") == 0


def test_spend_leaves_the_window_as_it_rolls() -> None:
    """A rolling budget that never forgets is a lifetime budget under another name."""
    governor = CostGovernor(per_tenant=1000)
    governor.record(900, tenant="acme", now=0.0)

    assert governor.spent("acme", now=ROLLING_WINDOW_SECONDS - 1) == 900
    assert governor.spent("acme", now=ROLLING_WINDOW_SECONDS + 1) == 0
    governor.check(900, tenant="acme", now=ROLLING_WINDOW_SECONDS + 1)


def test_untenanted_traffic_shares_one_bucket() -> None:
    """What a per-tenant budget means without multi-tenancy: the deployment is the tenant."""
    governor = CostGovernor(per_tenant=1000)
    governor.record(600)

    with pytest.raises(FasterRagError):
        governor.check(500)


def test_both_budgets_apply_together() -> None:
    """Neither is a fallback for the other; a query has to satisfy both."""
    governor = CostGovernor(per_query=100, per_tenant=1000)
    governor.record(950, tenant="acme")

    with pytest.raises(FasterRagError, match="per_query"):
        governor.check(200, tenant="acme")
    with pytest.raises(FasterRagError, match="per_tenant"):
        governor.check(60, tenant="acme")


def test_the_governor_reads_its_budgets_from_configuration() -> None:
    settings = Settings.model_validate(
        {"cost": {"per_query_token_budget": 7, "per_tenant_token_budget": 9}}
    )

    governor = create_governor(settings)

    assert (governor.per_query, governor.per_tenant) == (7, 9)
    assert governor.enabled is True
