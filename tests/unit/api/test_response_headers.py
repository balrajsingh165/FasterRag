"""Security response headers and the CORS policy (TASK-0494).

The API set no CORS policy and no security headers; `vercel.json` covered only the
documentation site. Every response now carries the three headers, and a cross-origin
preflight is refused unless its origin is named in `security.cors_allowed_origins`.
"""

import pytest
from fastapi.testclient import TestClient

from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from tests.unit.api.conftest import StubVectorDB

SECRET = "headers-test-key-0001"
EXPECTED = {
    "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer",
    "cache-control": "no-store",
}


def client_for(**security: object) -> TestClient:
    """Return a client for an application with the given security settings."""
    app = create_app(Settings.model_validate({"security": security}))
    app.state.vector_db = StubVectorDB()
    return TestClient(app)


def preflight(origin: str) -> dict[str, str]:
    """Return the headers a browser sends before a cross-origin POST with a bearer token."""
    return {
        "Origin": origin,
        "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "authorization,content-type",
    }


def test_every_response_carries_the_security_headers() -> None:
    """A success, a problem document and the readiness probe alike."""
    with client_for() as client:
        responses = [client.get("/healthz"), client.get("/readyz"), client.get("/no-such-route")]

    for response in responses:
        for name, value in EXPECTED.items():
            assert response.headers[name] == value, (response.request.url, name)


def test_a_refusal_at_the_auth_boundary_carries_them_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The headers are added outside authentication, so a 401 is not the one response without."""
    monkeypatch.setenv("FASTERRAG_API_KEY", SECRET)

    with client_for(auth=True) as client:
        response = client.get("/metrics")

    assert response.status_code == 401
    assert all(response.headers[name] == value for name, value in EXPECTED.items())


def test_a_cross_origin_preflight_is_refused_by_default() -> None:
    """The default list is empty, so no origin may call the API from a browser."""
    with client_for() as client:
        response = client.options("/v1/query", headers=preflight("https://evil.example"))

    assert response.status_code == 400
    assert "access-control-allow-origin" not in response.headers


def test_a_simple_cross_origin_request_is_not_granted_by_default() -> None:
    """Without the header a browser withholds the response from the calling page."""
    with client_for() as client:
        response = client.get("/healthz", headers={"Origin": "https://evil.example"})

    assert "access-control-allow-origin" not in response.headers


def test_a_configured_origin_is_allowed_and_no_other(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", SECRET)

    with client_for(auth=True, cors_allowed_origins=["https://app.example"]) as client:
        allowed = client.options("/v1/query", headers=preflight("https://app.example"))
        refused = client.options("/v1/query", headers=preflight("https://evil.example"))

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == "https://app.example"
    assert "authorization" in allowed.headers["access-control-allow-headers"].lower()
    assert "access-control-allow-credentials" not in allowed.headers
    assert refused.status_code == 400
    assert "access-control-allow-origin" not in refused.headers


@pytest.mark.parametrize(
    "origin", ["*", "app.example", "https://app.example/", "https://app.example/path", ""]
)
def test_an_origin_that_is_not_scheme_and_host_is_refused_at_load(origin: str) -> None:
    """`*` would open the API to every page; a path never matches an Origin header."""
    with pytest.raises(ValueError, match="cors_allowed_origins"):
        Settings.model_validate({"security": {"cors_allowed_origins": [origin]}})
