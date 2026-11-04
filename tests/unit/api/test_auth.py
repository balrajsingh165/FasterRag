import logging

import pytest
from fastapi.testclient import TestClient
from starlette.types import Receive, Scope, Send

from fasterrag.api.auth import (
    ALL_SCOPES,
    PUBLIC_PATHS,
    ApiKey,
    AuthMiddleware,
    KeyRegistry,
    load_keys,
    required_scope,
)
from fasterrag.api.main import create_app
from fasterrag.api.problems import PROBLEM_MEDIA_TYPE
from fasterrag.config.schema import Settings
from fasterrag.core.rate_limit import (
    InProcessRateLimiter,
    RateLimitDecision,
    RateLimiter,
    RedisRateLimiter,
)
from fasterrag.errors import ConfigError, ErrorCode
from tests.unit.api.conftest import StubVectorDB

SECRET = "secret-key-value"


async def _unreached(scope: Scope, receive: Receive, send: Send) -> None:
    raise AssertionError("the wrapped application should not be called")


class _RecordingLimiter(RateLimiter):
    def __init__(self) -> None:
        self.closed = False

    async def allow(self, identity: str) -> RateLimitDecision:
        return RateLimitDecision(True, 0)

    async def close(self) -> None:
        self.closed = True


class _RefusingLimiter(RateLimiter):
    """Refuses every request with a wait no clock in the test could have produced."""

    async def allow(self, identity: str) -> RateLimitDecision:
        return RateLimitDecision(False, 37)

    async def close(self) -> None:
        return None


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


def test_the_retry_after_header_is_the_limiter_s_own_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The header is the only place a refused caller learns how long to wait."""
    monkeypatch.setattr(
        "fasterrag.api.auth.create_rate_limiter", lambda _settings: _RefusingLimiter()
    )
    monkeypatch.setenv("FASTERRAG_API_KEY", SECRET)
    app = create_app(Settings.model_validate({"security": {"auth": True}}))
    app.state.vector_db = StubVectorDB()

    with TestClient(app) as client:
        response = client.get("/metrics", headers=bearer())

    assert response.status_code == 429
    assert response.headers["Retry-After"] == "37"


def test_a_malformed_authorization_header_is_treated_as_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with secured(monkeypatch) as client:
        response = client.get("/metrics", headers={"Authorization": SECRET})

    assert response.json()["code"] == ErrorCode.AUTH_MISSING.value


def test_the_default_backend_is_still_the_in_process_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single-process deployment must not start paying for a network round trip."""
    monkeypatch.setenv("FASTERRAG_API_KEY", SECRET)
    settings = Settings.model_validate({"security": {"auth": True}})

    middleware = AuthMiddleware(_unreached, settings)

    assert isinstance(middleware._limiter, InProcessRateLimiter)
    assert middleware._limiter.limit == settings.security.rate_limit_per_minute


def test_selecting_redis_builds_the_shared_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole point of TASK-0216: the backend key has to reach the middleware."""
    monkeypatch.setenv("FASTERRAG_API_KEY", SECRET)
    settings = Settings.model_validate({"security": {"auth": True, "rate_limit_backend": "redis"}})

    middleware = AuthMiddleware(_unreached, settings)

    assert isinstance(middleware._limiter, RedisRateLimiter)


def test_shutdown_releases_the_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    """A limiter holding a connection pool is leaked unless something closes it."""
    recorder = _RecordingLimiter()
    monkeypatch.setattr("fasterrag.api.auth.create_rate_limiter", lambda _settings: recorder)
    monkeypatch.setenv("FASTERRAG_API_KEY", SECRET)
    app = create_app(Settings.model_validate({"security": {"auth": True}}))
    app.state.vector_db = StubVectorDB()

    with TestClient(app) as client:
        client.get("/metrics", headers=bearer())
        assert not recorder.closed

    assert recorder.closed


def test_an_unbound_key_is_announced_when_multi_tenancy_is_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A key that omitted its tenant is an operator credential; say so at startup."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "alpha-secret:query;beta-secret:query:acme")
    settings = Settings.model_validate(
        {"security": {"auth": True, "multi_tenancy": True, "api_key_env": "FASTERRAG_API_KEY"}}
    )

    with caplog.at_level(logging.WARNING):
        AuthMiddleware(_unreached, settings)

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert any("may act for any tenant" in record.getMessage() for record in warnings)
    assert not any("alpha-secret" in record.getMessage() for record in warnings)


def test_a_fully_bound_key_set_is_announced_quietly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The warning must discriminate, or it is noise everyone learns to ignore."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "alpha-secret:query:acme")
    settings = Settings.model_validate(
        {"security": {"auth": True, "multi_tenancy": True, "api_key_env": "FASTERRAG_API_KEY"}}
    )

    with caplog.at_level(logging.WARNING):
        AuthMiddleware(_unreached, settings)

    assert not any("may act for any tenant" in record.getMessage() for record in caplog.records)
