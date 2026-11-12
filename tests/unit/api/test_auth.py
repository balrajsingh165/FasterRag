import hashlib
import logging
import secrets

import pytest
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from fasterrag.api.auth import (
    ALL_SCOPES,
    ALL_SCOPES_TOKEN,
    CROSS_TENANT_SCOPE,
    MINIMUM_KEY_LENGTH,
    PUBLIC_PATHS,
    ApiKey,
    AuthMiddleware,
    KeyRegistry,
    _parse_scopes,
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


def sha256(value: bytes) -> bytes:
    """Return the digest a key is stored and compared as."""
    return hashlib.sha256(value).digest()


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
    key = ApiKey.from_secret("s", frozenset({"admin"}))

    for scope in ALL_SCOPES:
        assert key.permits(scope)


def test_a_bare_key_gets_every_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    """The single-operator case must not require spelling out four scopes."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "just-a-secret-value")

    keys = load_keys(Settings.model_validate({"security": {"auth": True}}))

    assert keys[0].scopes == ALL_SCOPES


def test_several_keys_with_distinct_scopes_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "FASTERRAG_API_KEY", "reader-key-00001:query;writer-key-00001:ingest,collections"
    )

    keys = load_keys(Settings.model_validate({"security": {"auth": True}}))

    registry = KeyRegistry(keys)
    assert [registry.resolve(secret) for secret in ("reader-key-00001", "writer-key-00001")] == keys
    assert keys[0].scopes == frozenset({"query"})
    assert keys[1].scopes == frozenset({"ingest", "collections"})


def test_scopes_and_keys_use_different_separators(monkeypatch: pytest.MonkeyPatch) -> None:
    """One separator for both makes `a:query,ingest` ambiguous, and the parser must guess."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "solo-key-00000001:query,ingest")

    keys = load_keys(Settings.model_validate({"security": {"auth": True}}))

    assert len(keys) == 1
    assert keys[0].scopes == frozenset({"query", "ingest"})


def test_a_tenant_can_be_attached_to_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", "operator-key-0001:query:acme")

    assert load_keys(Settings.model_validate({"security": {"auth": True}}))[0].tenant == "acme"


@pytest.mark.parametrize("raw", ["", " ", " , ", ","])
def test_an_empty_scope_field_is_refused_instead_of_granting_everything(raw: str) -> None:
    """`frozenset(scopes or ALL_SCOPES)` turned a blank field into `admin` (TASK-0327)."""
    with pytest.raises(ConfigError, match="empty scope field"):
        _parse_scopes(raw)


@pytest.mark.parametrize("entry", ["k:", "k::acme", "k: , ", "k:::"])
def test_a_key_with_a_blank_scope_field_refuses_to_start(
    monkeypatch: pytest.MonkeyPatch, entry: str
) -> None:
    """Each of these loaded as a full admin key before TASK-0327, `k::acme` silently."""
    monkeypatch.setenv("FASTERRAG_API_KEY", entry)

    with pytest.raises(ConfigError, match="empty scope field"):
        load_keys(Settings.model_validate({"security": {"auth": True}}))


def test_the_blank_scope_refusal_names_the_entry_and_the_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing closed with no documented fix is just a different outage."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "good-key-00000001:query;bad::acme")

    with pytest.raises(ConfigError) as refusal:
        load_keys(Settings.model_validate({"security": {"auth": True}}))

    assert "entry 2" in str(refusal.value)
    assert f"'{ALL_SCOPES_TOKEN}'" in str(refusal.value)


def test_the_all_scopes_token_asks_for_every_scope_in_writing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", f"operator-key-0001:{ALL_SCOPES_TOKEN}")

    keys = load_keys(Settings.model_validate({"security": {"auth": True}}))

    assert keys[0].scopes == ALL_SCOPES


def test_a_tenant_can_be_named_without_naming_individual_scopes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The replacement spelling for `k::acme`, which no longer parses."""
    monkeypatch.setenv("FASTERRAG_API_KEY", f"operator-key-0001:{ALL_SCOPES_TOKEN}:acme")

    keys = load_keys(Settings.model_validate({"security": {"auth": True}}))

    assert keys[0].scopes == ALL_SCOPES
    assert keys[0].tenant == "acme"


def test_a_bare_secret_still_needs_no_scope_field_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An omitted field and a blank one are different things; only the blank one is refused."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "just-a-secret-value")

    keys = load_keys(Settings.model_validate({"security": {"auth": True}}))

    assert keys[0].scopes == ALL_SCOPES
    assert keys[0].tenant is None


def test_a_trailing_colon_after_the_scopes_leaves_the_key_unbound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`k:query:` is an operator key for any tenant, which TASK-0295 weighs, not this fix."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "operator-key-0001:query:")

    keys = load_keys(Settings.model_validate({"security": {"auth": True}}))

    assert keys[0].scopes == frozenset({"query"})
    assert keys[0].tenant is None


def test_a_third_colon_lands_inside_the_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Documented here as observed behaviour: the remainder is not split again."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "operator-key-0001:query:acme:extra")

    keys = load_keys(Settings.model_validate({"security": {"auth": True}}))

    assert keys[0].tenant == "acme:extra"


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
    first = ApiKey.from_secret("aaa", ALL_SCOPES)
    second = ApiKey.from_secret("bbb", frozenset({"query"}))
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
    """A key that omitted its tenant is refused under tenancy; say so at startup."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "alpha-secret-0001:query;beta-secret-00001:query:acme")
    settings = Settings.model_validate(
        {"security": {"auth": True, "multi_tenancy": True, "api_key_env": "FASTERRAG_API_KEY"}}
    )

    with caplog.at_level(logging.WARNING):
        AuthMiddleware(_unreached, settings)

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert any("refused on every request" in record.getMessage() for record in warnings)
    assert not any("alpha-secret" in record.getMessage() for record in warnings)


def test_a_fully_bound_key_set_is_announced_quietly(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The warning must discriminate, or it is noise everyone learns to ignore."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "alpha-secret-0001:query:acme")
    settings = Settings.model_validate(
        {"security": {"auth": True, "multi_tenancy": True, "api_key_env": "FASTERRAG_API_KEY"}}
    )

    with caplog.at_level(logging.WARNING):
        AuthMiddleware(_unreached, settings)

    messages = [record.getMessage() for record in caplog.records]
    assert not any("refused on every request" in message for message in messages)
    assert not any("may act for any tenant" in message for message in messages)


class _Recorder:
    """Captures the authenticated identity the middleware puts on the scope."""

    def __init__(self) -> None:
        self.state: dict[str, object] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        self.state = dict(scope.get("state", {}))
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})


async def drive(middleware: AuthMiddleware, headers: list[tuple[bytes, bytes]]) -> None:
    """Send one `GET /metrics` through the middleware with the given raw headers."""

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        return None

    await middleware(
        {"type": "http", "path": "/metrics", "method": "GET", "headers": headers},
        receive,
        send,
    )


def test_a_key_cannot_be_rewritten_after_it_is_loaded() -> None:
    """The registry's key objects are handed downstream on every request.

    A mutable one would let any later code widen `scopes` for the whole process, so the
    key is frozen and slotted and neither an assignment nor a new attribute can land.
    """
    key = ApiKey.from_secret("s", frozenset({"query"}))
    field = "scopes"

    with pytest.raises(AttributeError):
        setattr(key, field, ALL_SCOPES)
    assert not hasattr(key, "__dict__")


def test_a_narrow_scope_is_honoured_as_well_as_refused() -> None:
    """`permits` must not collapse to admin-only; a query key has to be able to query."""
    key = ApiKey.from_secret("s", frozenset({"query"}))

    assert key.permits("query")
    assert not key.permits("ingest")


def test_the_scope_table_matches_both_exactly_and_by_prefix() -> None:
    """Both halves of the match stay live, or every endpoint silently requires admin."""
    assert required_scope("/v1/query") == "query"
    assert required_scope("/v1/retrieve") == "query"
    assert required_scope("/v1/collections") == "collections"
    assert required_scope("/v1/ingest/jobs/abc") == "ingest"
    assert required_scope("/v1/traces/t-1") == "admin"
    assert required_scope("/metrics") == "admin"


def test_a_truncated_or_extended_key_does_not_resolve() -> None:
    """The comparison covers the whole secret, not a prefix of it."""
    registry = KeyRegistry([ApiKey.from_secret("aaaa", ALL_SCOPES)])

    assert registry.resolve("aaa") is None
    assert registry.resolve("aaaaa") is None
    assert registry.resolve("aaaa") is not None


def test_the_empty_key_error_counts_entries_from_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The operator has to be told which entry to fix, and the list they read is 1-based."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "good-key-00000001:query;:ingest")

    with pytest.raises(ConfigError, match="entry 2 has an empty key"):
        load_keys(Settings.model_validate({"security": {"auth": True}}))


def test_each_key_is_named_by_its_position_from_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """The unbound-key warning names keys positionally; an off-by-one blames the wrong entry."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "alpha-key-0000001:query;beta-key-00000001:ingest")

    keys = load_keys(Settings.model_validate({"security": {"auth": True}}))

    assert [key.name for key in keys] == ["key-1", "key-2"]


def test_a_non_bearer_credential_reads_as_no_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    """`Basic <blob>` must not be lifted out and tried as a bearer token."""
    with secured(monkeypatch) as client:
        basic = client.get("/metrics", headers={"Authorization": "Basic c2VjcmV0"})
        bare = client.get("/metrics", headers={"Authorization": "Bearer"})

    assert basic.json()["code"] == ErrorCode.AUTH_MISSING.value
    assert bare.json()["code"] == ErrorCode.AUTH_MISSING.value


async def test_an_operator_key_acts_as_the_tenant_it_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The declared tenant has to reach downstream, or the operator acts untenanted instead."""
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{SECRET}:*,{CROSS_TENANT_SCOPE}")
    settings = Settings.model_validate({"security": {"auth": True, "multi_tenancy": True}})
    recorder = _Recorder()

    await drive(
        AuthMiddleware(recorder, settings),
        [
            (b"authorization", f"Bearer {SECRET}".encode("latin-1")),
            (b"x-tenant-id", b"globex"),
        ],
    )

    assert recorder.state["tenant"] == "globex"


async def test_a_bound_key_acts_as_its_own_tenant(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ordinary case: the header matched, so the key's own tenant is what travels."""
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{SECRET}:admin:acme")
    settings = Settings.model_validate({"security": {"auth": True, "multi_tenancy": True}})
    recorder = _Recorder()

    await drive(
        AuthMiddleware(recorder, settings),
        [
            (b"authorization", f"Bearer {SECRET}".encode("latin-1")),
            (b"x-tenant-id", b"acme"),
        ],
    )

    assert recorder.state["tenant"] == "acme"


async def test_a_bound_key_may_not_act_for_a_tenant_that_merely_starts_the_same(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tenant comparison is equality, and a weaker one is a cross-tenant read.

    Relaxing it to a prefix or containment test hands the key bound to `acme` every
    collection belonging to `acme-corp`, with nothing in the response to show for it.
    """
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{SECRET}:admin:acme")
    settings = Settings.model_validate({"security": {"auth": True, "multi_tenancy": True}})
    recorder = _Recorder()

    await drive(
        AuthMiddleware(recorder, settings),
        [
            (b"authorization", f"Bearer {SECRET}".encode("latin-1")),
            (b"x-tenant-id", b"acme-corp"),
        ],
    )

    assert recorder.state == {}


class _CountingSecrets:
    """Stands in for the `secrets` module so the comparison itself can be observed."""

    def __init__(self) -> None:
        """Start with nothing compared."""
        self.compared: list[tuple[bytes, bytes]] = []

    def compare_digest(self, stored: bytes, presented: bytes) -> bool:
        """Record the pair, then answer exactly as the real function would."""
        self.compared.append((stored, presented))
        return secrets.compare_digest(stored, presented)


def test_the_registry_compares_keys_in_constant_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """`==` stops at the first differing byte, which turns the refusal into a key oracle.

    No functional assertion can see the difference, so the call itself is what gets
    pinned: swapping it for `==` leaves every other test in this file green.
    """
    counting = _CountingSecrets()
    monkeypatch.setattr("fasterrag.api.auth.secrets", counting)
    registry = KeyRegistry([ApiKey.from_secret("aaa", ALL_SCOPES)])

    assert registry.resolve("aaa") is not None
    assert counting.compared == [(sha256(b"aaa"), sha256(b"aaa"))]


def test_every_key_is_compared_even_after_one_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returning on the first hit makes how long a lookup takes depend on key order.

    That is the same oracle `compare_digest` closes, one level up, so the scan runs
    to the end and keeps the match rather than short-circuiting on it.
    """
    counting = _CountingSecrets()
    monkeypatch.setattr("fasterrag.api.auth.secrets", counting)
    registry = KeyRegistry(
        [ApiKey.from_secret("aaa", ALL_SCOPES), ApiKey.from_secret("bbb", ALL_SCOPES)]
    )

    assert registry.resolve("aaa") is not None
    assert [stored for stored, _ in counting.compared] == [sha256(b"aaa"), sha256(b"bbb")]


def test_a_bearer_token_with_a_non_ascii_byte_is_refused_not_crashed_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`compare_digest` raises on such a `str`, which turned a 401 into an unhandled 500."""
    with secured(monkeypatch) as client:
        response = client.post(
            "/v1/query",
            json={"query": "hello"},
            headers={b"Authorization": b"Bearer caf\xe9-key"},
        )

    assert response.status_code == 401
    assert response.json()["code"] == ErrorCode.AUTH_INVALID.value


def test_a_collections_key_cannot_drop_a_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """The prefix table gave `DELETE /v1/collections/{name}` to `collections` (TASK-0520)."""
    with secured(monkeypatch, keys=f"{SECRET}:collections") as client:
        dropped = client.delete("/v1/collections/docs?force=true", headers=bearer())
        listed = client.get("/v1/collections", headers=bearer())

    assert dropped.status_code == 403
    assert dropped.json()["code"] == ErrorCode.AUTH_SCOPE.value
    assert "'admin'" in dropped.json()["detail"]
    assert listed.status_code == 200


@pytest.mark.parametrize("placeholder", ["change-me", "CHANGE-ME", "changeme", "change_me"])
def test_the_env_example_placeholder_is_refused_at_startup(
    monkeypatch: pytest.MonkeyPatch, placeholder: str
) -> None:
    """`.env.example` ships `change-me`, and it was accepted as a live key (TASK-0460)."""
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{SECRET};{placeholder}:query")

    with pytest.raises(ConfigError, match="entry 2 is the placeholder") as refusal:
        load_keys(Settings.model_validate({"security": {"auth": True}}))

    assert placeholder not in str(refusal.value)


def test_a_short_key_is_refused_without_printing_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """A guessable key protected nothing; the refusal must not become the leak (TASK-0460)."""
    short = "x" * (MINIMUM_KEY_LENGTH - 1)
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{short}:query:acme")

    with pytest.raises(ConfigError, match=f"shorter than {MINIMUM_KEY_LENGTH}") as refusal:
        load_keys(Settings.model_validate({"security": {"auth": True}}))

    assert short not in str(refusal.value)


def test_a_key_of_exactly_the_minimum_length_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    """The boundary is inclusive, so the documented minimum is a length that works."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "y" * MINIMUM_KEY_LENGTH)

    assert load_keys(Settings.model_validate({"security": {"auth": True}}))[0].name == "key-1"


def test_the_server_refuses_to_start_on_the_placeholder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The refusal is at startup, not at the first request that happens to present it."""
    monkeypatch.setenv("FASTERRAG_API_KEY", "change-me")

    app = create_app(Settings.model_validate({"security": {"auth": True}}))
    app.state.vector_db = StubVectorDB()

    with pytest.raises(ConfigError, match="placeholder"), TestClient(app):
        pass


async def test_an_unbound_key_is_refused_for_every_tenant_without_the_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A key that omitted its tenant used to act for any tenant it named (TASK-0460)."""
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{SECRET}:admin")
    settings = Settings.model_validate({"security": {"auth": True, "multi_tenancy": True}})
    recorder = _Recorder()

    await drive(
        AuthMiddleware(recorder, settings),
        [
            (b"authorization", f"Bearer {SECRET}".encode("latin-1")),
            (b"x-tenant-id", b"globex"),
        ],
    )

    assert recorder.state == {}


def test_the_unbound_refusal_names_the_scope_that_fixes_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", SECRET)
    app = create_app(Settings.model_validate({"security": {"auth": True, "multi_tenancy": True}}))
    app.state.vector_db = StubVectorDB()

    with TestClient(app) as client:
        response = client.get("/metrics", headers={**bearer(), "X-Tenant-ID": "acme"})

    assert response.status_code == 403
    assert response.json()["code"] == ErrorCode.TENANT_FORBIDDEN.value
    assert f"'{CROSS_TENANT_SCOPE}'" in response.json()["detail"]


def test_the_cross_tenant_scope_is_never_implied(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neither a bare secret, `*`, nor `admin` grants it: it must be written by name."""
    monkeypatch.setenv(
        "FASTERRAG_API_KEY",
        f"bare-operator-key-01;star-operator-key-1:*;admin-operator-key-1:admin;"
        f"named-operator-key-1:*,{CROSS_TENANT_SCOPE}",
    )

    keys = load_keys(Settings.model_validate({"security": {"auth": True}}))

    assert [CROSS_TENANT_SCOPE in key.scopes for key in keys] == [False, False, False, True]
    assert keys[3].scopes == ALL_SCOPES | {CROSS_TENANT_SCOPE}


def test_the_startup_warning_names_the_fix(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The warning used to say the key may act for any tenant; now it is refused instead."""
    monkeypatch.setenv(
        "FASTERRAG_API_KEY",
        f"unbound-key-000001:query;crossing-key-00001:query,{CROSS_TENANT_SCOPE}",
    )
    settings = Settings.model_validate({"security": {"auth": True, "multi_tenancy": True}})

    with caplog.at_level(logging.WARNING):
        AuthMiddleware(_unreached, settings)

    messages = [record.getMessage() for record in caplog.records]
    refused = [message for message in messages if "refused on every request" in message]
    assert refused
    assert f"'{CROSS_TENANT_SCOPE}'" in refused[0]
    assert "secret:scopes:tenant" in refused[0]
    assert any("may act for any tenant" in message for message in messages)
    assert not any("unbound-key-000001" in message for message in messages)
