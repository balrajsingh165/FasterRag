import pytest
from fastapi.testclient import TestClient

from fasterrag.api.auth import (
    ALL_SCOPES,
    PUBLIC_PATHS,
    ApiKey,
    KeyRegistry,
    RateLimiter,
    load_keys,
    required_scope,
)
from fasterrag.api.main import create_app
from fasterrag.api.problems import PROBLEM_MEDIA_TYPE
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, ErrorCode
from tests.unit.api.conftest import StubVectorDB

SECRET = "secret-key-value"


def secured(monkeypatch: pytest.MonkeyPatch, keys: str = SECRET) -> TestClient:
    monkeypatch.setenv("FASTERRAG_API_KEY", keys)
    settings = Settings.model_validate({"security": {"auth": True}})
    app = create_app(settings)
    app.state.vector_db = StubVectorDB()
    return TestClient(app)


def bearer(secret: str = SECRET) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


def test_a_request_without_a_key_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with secured(monkeypatch) as client:
        response = client.post("/v1/query", json={"query": "hello"})

    assert response.status_code == 401
    assert response.headers["content-type"].startswith(PROBLEM_MEDIA_TYPE)
    assert response.json()["code"] == ErrorCode.AUTH_MISSING.value


def test_a_wrong_key_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with secured(monkeypatch) as client:
        response = client.post("/v1/query", json={"query": "hello"}, headers=bearer("wrong"))

    assert response.status_code == 401
    assert response.json()["code"] == ErrorCode.AUTH_INVALID.value


def test_the_refusal_does_not_reveal_whether_a_key_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unknown key and a revoked one must be indistinguishable to a caller."""
    with secured(monkeypatch) as client:
        unknown = client.post("/v1/query", json={"query": "x"}, headers=bearer("nope"))
        revoked = client.post("/v1/query", json={"query": "x"}, headers=bearer("also-nope"))

    assert unknown.json()["detail"] == revoked.json()["detail"]
    assert unknown.status_code == revoked.status_code


def test_a_valid_key_reaches_the_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """`/metrics` is used because it needs no live backend, so only the gate is under test."""
    with secured(monkeypatch) as client:
        response = client.get("/metrics", headers=bearer())

    assert response.status_code == 200


def test_a_key_without_the_scope_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    with secured(monkeypatch, keys=f"{SECRET}:query") as client:
        response = client.post("/v1/ingest", json={"sources": []}, headers=bearer())

    assert response.status_code == 403
    assert response.json()["code"] == ErrorCode.AUTH_SCOPE.value


def test_health_stays_reachable_without_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A probe that needs a credential turns a key mistake into an outage."""
    with secured(monkeypatch) as client:
        assert client.get("/healthz").status_code == 200


def test_metrics_is_not_public(monkeypatch: pytest.MonkeyPatch) -> None:
    """It exposes per-endpoint volumes and costs, so it is protected like anything else."""
    assert "/metrics" not in PUBLIC_PATHS

    with secured(monkeypatch) as client:
        assert client.get("/metrics").status_code == 401


def test_auth_off_leaves_every_endpoint_open() -> None:
    app = create_app(Settings.model_validate({}))
    app.state.vector_db = StubVectorDB()

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/metrics").status_code == 200


def test_an_unmapped_path_requires_admin() -> None:
    """A new router nobody added to the table is refused, not exposed."""
    assert required_scope("/v1/something-new") == "admin"


def test_public_paths_need_no_scope() -> None:
    for path in PUBLIC_PATHS:
        assert required_scope(path) is None


def test_admin_implies_every_other_scope() -> None:
    key = ApiKey(secret="s", scopes=frozenset({"admin"}))

    for scope in ALL_SCOPES:
        assert key.permits(scope)


def test_a_bare_key_gets_every_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """The single-operator case must not require spelling out four scopes."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "just-a-secret")

    keys = load_keys(Settings.model_validate({"security": {"auth": True}}))

    assert keys[0].scopes == ALL_SCOPES


def test_several_keys_with_distinct_scopes_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", "reader:query; writer:ingest,collections")

    keys = load_keys(Settings.model_validate({"security": {"auth": True}}))

    assert [key.secret for key in keys] == ["reader", "writer"]
    assert keys[0].scopes == frozenset({"query"})
    assert keys[1].scopes == frozenset({"ingest", "collections"})


def test_scopes_and_keys_use_different_separators(monkeypatch: pytest.MonkeyPatch) -> None:
    """One separator for both makes `a:query,ingest` ambiguous, and the parser must guess."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "solo:query,ingest")

    keys = load_keys(Settings.model_validate({"security": {"auth": True}}))

    assert len(keys) == 1
    assert keys[0].scopes == frozenset({"query", "ingest"})


def test_a_tenant_can_be_attached_to_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", "k:query:acme")

    assert load_keys(Settings.model_validate({"security": {"auth": True}}))[0].tenant == "acme"


def test_a_misspelled_scope_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """`collection` would otherwise produce a key that silently permits nothing."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "k:collection")

    with pytest.raises(ConfigError, match="unknown scope"):
        load_keys(Settings.model_validate({"security": {"auth": True}}))


def test_auth_without_any_key_refuses_to_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """Starting would refuse every request, which reads as broken rather than misconfigured."""
    monkeypatch.delenv("FASTERRAG_API_KEY", raising=False)

    with pytest.raises(ConfigError, match=r"security.auth is true"):
        load_keys(Settings.model_validate({"security": {"auth": True}}))


def test_the_registry_matches_the_right_key() -> None:
    first = ApiKey(secret="aaa", scopes=ALL_SCOPES)
    second = ApiKey(secret="bbb", scopes=frozenset({"query"}))
    registry = KeyRegistry([first, second])

    assert registry.resolve("bbb") is second
    assert registry.resolve("ccc") is None


def test_the_rate_limit_refuses_past_the_ceiling() -> None:
    limiter = RateLimiter(limit=2)

    assert limiter.allow("k", now=0.0)[0]
    assert limiter.allow("k", now=0.1)[0]
    allowed, retry_after = limiter.allow("k", now=0.2)

    assert not allowed
    assert retry_after >= 1


def test_the_window_rolls_forward() -> None:
    limiter = RateLimiter(limit=1)
    limiter.allow("k", now=0.0)

    assert not limiter.allow("k", now=30.0)[0]
    assert limiter.allow("k", now=61.0)[0]


def test_the_limit_is_per_key_not_global() -> None:
    """One noisy caller must not exhaust everyone else's budget."""
    limiter = RateLimiter(limit=1)
    limiter.allow("first", now=0.0)

    assert limiter.allow("second", now=0.0)[0]


def test_exceeding_the_limit_returns_429_with_retry_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings.model_validate({"security": {"auth": True, "rate_limit_per_minute": 1}})
    monkeypatch.setenv("FASTERRAG_API_KEY", SECRET)
    app = create_app(settings)
    app.state.vector_db = StubVectorDB()

    with TestClient(app) as client:
        client.get("/metrics", headers=bearer())
        response = client.get("/metrics", headers=bearer())

    assert response.status_code == 429
    assert int(response.headers["Retry-After"]) >= 1
    assert response.json()["code"] == ErrorCode.RATE_LIMITED.value


def test_a_malformed_authorization_header_is_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with secured(monkeypatch) as client:
        response = client.get("/metrics", headers={"Authorization": SECRET})

    assert response.json()["code"] == ErrorCode.AUTH_MISSING.value
