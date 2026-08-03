import pytest

from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.tenancy import SEPARATOR, scoped_name, unscoped_name, visible_to


def test_a_tenants_collection_is_prefixed() -> None:
    assert scoped_name("docs", "acme") == f"acme{SEPARATOR}docs"


def test_an_untenanted_deployment_is_untouched() -> None:
    """Enabling multi-tenancy must be the only thing that ever moves a collection."""
    assert scoped_name("docs", None) == "docs"
    assert unscoped_name("docs", None) == "docs"


def test_a_tenant_sees_the_name_it_chose() -> None:
    """A tenant should never learn that prefixing exists."""
    assert unscoped_name(f"acme{SEPARATOR}docs", "acme") == "docs"


def test_a_tenant_cannot_see_another_tenants_collection() -> None:
    assert not visible_to(f"globex{SEPARATOR}docs", "acme")
    assert visible_to(f"acme{SEPARATOR}docs", "acme")


def test_a_tenant_cannot_see_untenanted_collections() -> None:
    """Those predate tenancy and belong to nobody who is asking."""
    assert not visible_to("legacy", "acme")


def test_an_operator_sees_everything() -> None:
    assert visible_to("legacy", None)
    assert visible_to(f"acme{SEPARATOR}docs", None)


def test_a_tenant_id_cannot_forge_another_prefix() -> None:
    """`acme__x` would otherwise address `acme`'s collection `x`."""
    with pytest.raises(FasterRagError) as caught:
        scoped_name("docs", f"acme{SEPARATOR}x")

    assert caught.value.code is ErrorCode.TENANT_FORBIDDEN


@pytest.mark.parametrize("tenant", ["", "../etc", "a b", "acme/x"])
def test_a_malformed_tenant_id_is_refused(tenant: str) -> None:
    with pytest.raises(FasterRagError):
        scoped_name("docs", tenant)


def test_ordinary_tenant_ids_are_accepted() -> None:
    for tenant in ("acme", "acme-eu", "acme.eu", "acme_1", "Acme2"):
        assert scoped_name("docs", tenant).endswith("docs")


def test_a_name_that_looks_prefixed_is_not_stripped_for_another_tenant() -> None:
    """Removing the wrong prefix would rename a collection into someone else's namespace."""
    assert unscoped_name(f"globex{SEPARATOR}docs", "acme") == f"globex{SEPARATOR}docs"
