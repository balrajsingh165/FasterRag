import pytest
from fastapi.testclient import TestClient

from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.core.cache.semantic import SemanticCache
from fasterrag.errors import ErrorCode
from tests.unit.api.conftest import StubVectorDB

SECRET = "acme-key"
VECTOR = [1.0, 0.0, 0.0, 0.0]
NEAR = [0.999, 0.01, 0.0, 0.0]


def tenanted(monkeypatch: pytest.MonkeyPatch, keys: str = f"{SECRET}:admin:acme") -> TestClient:
    monkeypatch.setenv("FASTERRAG_API_KEY", keys)
    settings = Settings.model_validate({"security": {"auth": True, "multi_tenancy": True}})
    app = create_app(settings)
    app.state.vector_db = StubVectorDB()
    return TestClient(app)


def headers(tenant: str | None = "acme", secret: str = SECRET) -> dict[str, str]:
    sent = {"Authorization": f"Bearer {secret}"}
    if tenant is not None:
        sent["X-Tenant-ID"] = tenant
    return sent


def cache() -> SemanticCache:
    settings = Settings.model_validate({"cache": {"semantic": True}})
    return SemanticCache(settings, enabled=True)


async def test_a_hit_never_crosses_tenants() -> None:
    """The disclosure this whole feature exists to prevent.

    Lookup compares vectors across every stored entry, so without an ownership check a
    sufficiently similar question from one tenant returns another tenant's answer — at a
    cache hit's latency, with no error anywhere.
    """
    store = cache()
    await store.store_response("salary policy", VECTOR, {"answer": "acme secret"}, tenant="acme")

    leaked = await store.lookup(NEAR, tenant="globex")

    assert leaked is None


async def test_a_tenant_still_hits_its_own_entry() -> None:
    store = cache()
    await store.store_response("salary policy", VECTOR, {"answer": "acme secret"}, tenant="acme")

    hit = await store.lookup(NEAR, tenant="acme")

    assert hit is not None
    assert hit.response["answer"] == "acme secret"


async def test_two_tenants_asking_the_same_question_do_not_collide() -> None:
    """One key for both would let the second write silently overwrite the first."""
    store = cache()
    await store.store_response("same question", VECTOR, {"answer": "acme"}, tenant="acme")
    await store.store_response("same question", VECTOR, {"answer": "globex"}, tenant="globex")

    first = await store.lookup(VECTOR, tenant="acme")
    second = await store.lookup(VECTOR, tenant="globex")

    assert first is not None
    assert second is not None
    assert first.response["answer"] == "acme"
    assert second.response["answer"] == "globex"


async def test_untenanted_entries_are_not_returned_to_a_tenant() -> None:
    """An entry written before tenancy was enabled must not leak into a tenant's results."""
    store = cache()
    await store.store_response("legacy", VECTOR, {"answer": "shared"})

    assert await store.lookup(NEAR, tenant="acme") is None
    assert await store.lookup(NEAR) is not None


def test_a_request_without_a_tenant_header_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unstated tenant would silently act on the untenanted namespace."""
    with tenanted(monkeypatch) as client:
        response = client.get("/metrics", headers=headers(tenant=None))

    assert response.status_code == 403
    assert response.json()["code"] == ErrorCode.TENANT_FORBIDDEN.value


def test_a_key_cannot_act_for_another_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    with tenanted(monkeypatch) as client:
        response = client.get("/metrics", headers=headers(tenant="globex"))

    assert response.status_code == 403
    assert response.json()["code"] == ErrorCode.TENANT_FORBIDDEN.value


def test_the_refusal_does_not_reveal_the_keys_own_tenant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Echoing it would let a caller enumerate tenants by reading the error."""
    with tenanted(monkeypatch) as client:
        detail = client.get("/metrics", headers=headers(tenant="globex")).json()["detail"]

    assert "acme" not in detail


def test_a_matching_tenant_is_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    with tenanted(monkeypatch) as client:
        assert client.get("/metrics", headers=headers()).status_code == 200


def test_an_operator_key_may_act_for_any_tenant_but_must_say_which(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key with no tenant is an operator credential, not a wildcard that skips the header."""
    with tenanted(monkeypatch, keys=f"{SECRET}:admin") as client:
        assert client.get("/metrics", headers=headers(tenant="anything")).status_code == 200
        assert client.get("/metrics", headers=headers(tenant=None)).status_code == 403


def test_the_header_name_follows_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{SECRET}:admin:acme")
    settings = Settings.model_validate(
        {
            "security": {
                "auth": True,
                "multi_tenancy": True,
                "tenant_header": "X-Org",
            }
        }
    )
    app = create_app(settings)
    app.state.vector_db = StubVectorDB()

    with TestClient(app) as client:
        allowed = client.get(
            "/metrics", headers={"Authorization": f"Bearer {SECRET}", "X-Org": "acme"}
        )
        refused = client.get(
            "/metrics", headers={"Authorization": f"Bearer {SECRET}", "X-Tenant-ID": "acme"}
        )

    assert allowed.status_code == 200
    assert refused.status_code == 403


def test_tenancy_off_needs_no_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """The single-tenant deployment must not have to carry a header it has no use for."""
    monkeypatch.setenv("FASTERRAG_API_KEY", SECRET)
    app = create_app(Settings.model_validate({"security": {"auth": True}}))
    app.state.vector_db = StubVectorDB()

    with TestClient(app) as client:
        assert (
            client.get("/metrics", headers={"Authorization": f"Bearer {SECRET}"}).status_code == 200
        )
