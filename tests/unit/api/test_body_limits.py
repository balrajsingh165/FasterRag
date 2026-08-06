"""Request body size limiting.

`security.max_request_mb` was declared, documented in `security.md` as enforced on every
endpoint with a `413`, and read by nothing: an 8 MB body against a 1 MB limit answered 200.
An unbounded body is a memory exhaustion vector, and a size limit that reads as configured
while doing nothing is worse than an absent one.
"""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from fasterrag.api.limits import BodyLimitMiddleware
from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from tests.unit.api.conftest import StubVectorDB

MEGABYTE = 1024 * 1024


def client(limit_mb: int = 1) -> TestClient:
    app = create_app(Settings.model_validate({"security": {"max_request_mb": limit_mb}}))
    app.state.vector_db = StubVectorDB()
    return TestClient(app, raise_server_exceptions=False)


def body(megabytes: float) -> dict[str, Any]:
    return {"query": "what is the allowance", "filters": {"pad": "x" * int(megabytes * MEGABYTE)}}


def test_a_body_within_the_limit_is_served() -> None:
    assert client().post("/v1/query", json=body(0.25)).status_code == 200


def test_a_body_over_the_limit_is_refused() -> None:
    assert client().post("/v1/query", json=body(8)).status_code == 413


def test_the_refusal_is_a_problem_document() -> None:
    response = client().post("/v1/query", json=body(8))

    assert response.json()["code"] == ErrorCode.PAYLOAD_TOO_LARGE.value


def test_the_refusal_names_the_setting() -> None:
    """An operator has to know which knob produced the 413."""
    response = client().post("/v1/query", json=body(8))

    assert "security.max_request_mb" in response.json()["detail"]


def test_the_refusal_does_not_echo_the_size_sent() -> None:
    """The size is attacker-controlled and would land in logs and problem bodies verbatim."""
    response = client().post("/v1/query", json=body(8))

    assert "8388608" not in response.text


def test_the_limit_follows_configuration() -> None:
    """A limit that ignored its setting would be a constant with a knob next to it."""
    assert client(limit_mb=16).post("/v1/query", json=body(8)).status_code == 200


def test_a_get_is_not_bounded() -> None:
    """A GET with a body is legal and meaningless; refusing one would surprise."""
    assert client().get("/healthz").status_code == 200


async def test_a_chunked_body_is_counted_as_it_arrives() -> None:
    """`Content-Length` is optional, so a header-only check is a limit clients opt into."""
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        chunk = {"type": "http.request", "body": b"x" * MEGABYTE, "more_body": True}
        sent.append(chunk)
        return chunk

    async def app(scope: Any, receive_: Any, send_: Any) -> None:
        while True:
            await receive_()

    middleware = BodyLimitMiddleware(
        app, Settings.model_validate({"security": {"max_request_mb": 2}})
    )
    scope = {"type": "http", "method": "POST", "path": "/v1/query", "headers": []}

    with pytest.raises(FasterRagError) as caught:
        await middleware(scope, receive, _swallow)

    assert caught.value.code is ErrorCode.PAYLOAD_TOO_LARGE
    assert len(sent) <= 3


async def test_an_unparseable_content_length_falls_back_to_counting() -> None:
    """A garbage header must not disable the limit; it just skips the early exit."""
    reached = False

    async def downstream(scope: Any, receive_: Any, send_: Any) -> None:
        nonlocal reached
        reached = True

    middleware = BodyLimitMiddleware(
        downstream, Settings.model_validate({"security": {"max_request_mb": 1}})
    )
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/query",
        "headers": [(b"content-length", b"not-a-number")],
    }

    await middleware(scope, _no_body, _swallow)

    assert reached is True


async def _swallow(message: Any) -> None:
    return None


async def _no_body() -> dict[str, Any]:
    return {"type": "http.request", "body": b"", "more_body": False}


def test_the_declared_length_is_refused_before_the_body_is_read() -> None:
    """The case worth optimising: the server allocates nothing for an oversized upload."""
    response = client().post(
        "/v1/query",
        content=b"{}",
        headers={"content-length": str(64 * MEGABYTE), "content-type": "application/json"},
    )

    assert response.status_code == 413
