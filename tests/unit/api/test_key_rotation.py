"""API keys stored as digests, read from a file that is re-read while serving, and expiring.

TASK-0492: ``load_keys`` held every key in plaintext, a restart was the only way to change
one, and none ever expired. The plaintext form keeps working; these pin the three additions.
"""

import hashlib
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from starlette.types import Message, Receive, Scope, Send

from fasterrag.api import auth
from fasterrag.api.auth import (
    ALL_SCOPES,
    HASHED_PREFIX,
    ApiKey,
    AuthMiddleware,
    KeyRegistry,
    load_keys,
)
from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.core.rate_limit import RateLimitDecision, RateLimiter
from fasterrag.errors import ConfigError, ErrorCode
from tests.unit.api.conftest import StubVectorDB

OLD = "old-key-that-is-rotated-out"
NEW = "new-key-that-is-rotated-in"


def hashed(secret: str) -> str:
    """Return the `sha256:<hex>` form an operator writes for ``secret``."""
    return HASHED_PREFIX + hashlib.sha256(secret.encode("utf-8")).hexdigest()


def settings(**security: object) -> Settings:
    """Return settings with auth on and the given security keys."""
    return Settings.model_validate({"security": {"auth": True, **security}})


def write_atomically(path: Path, text: str) -> None:
    """Replace the key file the way the documents tell an operator to: write, then rename."""
    staging = path.with_suffix(".tmp")
    staging.write_text(text, encoding="utf-8")
    os.replace(staging, path)


def bearer(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


def test_a_hashed_entry_authenticates_the_key_it_is_the_digest_of(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The environment holds only the digest; the caller still presents the key."""
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{hashed(NEW)}:query:acme")

    keys = load_keys(settings())

    assert keys[0].scopes == frozenset({"query"})
    assert keys[0].tenant == "acme"
    assert KeyRegistry(keys).resolve(NEW) is keys[0]
    assert KeyRegistry(keys).resolve(hashed(NEW)) is None


def test_a_hashed_bare_entry_gets_every_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", hashed(NEW))

    assert load_keys(settings())[0].scopes == ALL_SCOPES


@pytest.mark.parametrize("digest", ["abc", "z" * 64, "a" * 63])
def test_a_malformed_digest_is_refused(monkeypatch: pytest.MonkeyPatch, digest: str) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{HASHED_PREFIX}{digest}:query")

    with pytest.raises(ConfigError, match="not 64 hexadecimal digits"):
        load_keys(settings())


def test_the_digest_of_the_placeholder_is_refused_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hashing `change-me` does not make it a secret."""
    monkeypatch.setenv("FASTERRAG_API_KEY", hashed("change-me"))

    with pytest.raises(ConfigError, match="digest of the placeholder"):
        load_keys(settings())


def test_no_key_is_held_in_plaintext(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plaintext entry is reduced to its digest as it is loaded."""
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{NEW}:query")

    key = load_keys(settings())[0]

    assert NEW not in repr(key)
    assert key.digest == hashlib.sha256(NEW.encode()).digest()
    assert key.identity == key.digest.hex()


def test_a_key_past_its_expiry_is_refused_like_an_unknown_one(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    past = (datetime.now(UTC) - timedelta(days=1)).isoformat()
    future = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{OLD} expires={past};{NEW}:admin expires={future}")
    app = create_app(settings())
    app.state.vector_db = StubVectorDB()

    with caplog.at_level(logging.WARNING), TestClient(app) as client:
        expired = client.get("/metrics", headers=bearer(OLD))
        unknown = client.get("/metrics", headers=bearer("never-issued-key-0001"))
        current = client.get("/metrics", headers=bearer(NEW))

    assert expired.status_code == 401
    assert expired.json()["code"] == ErrorCode.AUTH_INVALID.value
    assert expired.json()["detail"] == unknown.json()["detail"]
    assert current.status_code == 200
    assert any("expired api key" in record.getMessage() for record in caplog.records)
    assert not any(OLD in record.getMessage() for record in caplog.records)


def test_an_expiry_with_no_zone_is_read_as_utc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{NEW}:query expires=2030-01-01")

    key = load_keys(settings())[0]

    assert key.expires == datetime(2030, 1, 1, tzinfo=UTC)
    assert key.scopes == frozenset({"query"})


def test_an_unreadable_expiry_is_refused_naming_the_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{OLD};{NEW} expires=next-tuesday")

    with pytest.raises(ConfigError, match="entry 2 has an expires= value") as refusal:
        load_keys(settings())

    assert NEW not in str(refusal.value)


def test_keys_load_from_a_file_beside_the_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key_file = tmp_path / "keys"
    key_file.write_text(f"# rotated weekly\n\n{hashed(NEW)}:query:acme\n", encoding="utf-8")
    monkeypatch.setenv("FASTERRAG_API_KEY", OLD)

    keys = load_keys(settings(api_key_file=str(key_file)))

    assert [key.name for key in keys] == ["key-1", "file-key-3"]
    assert keys[1].tenant == "acme"


def test_a_file_alone_is_enough_when_the_variable_is_switched_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    key_file = tmp_path / "keys"
    key_file.write_text(NEW, encoding="utf-8")
    monkeypatch.delenv("FASTERRAG_API_KEY", raising=False)

    keys = load_keys(settings(api_key_env=None, api_key_file=str(key_file)))

    assert KeyRegistry(keys).resolve(NEW) is not None


def test_auth_with_no_key_source_at_all_is_refused_at_load() -> None:
    with pytest.raises(ValueError, match=r"api_key_env and security.api_key_file"):
        settings(api_key_env=None)


def test_a_missing_key_file_refuses_to_start(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="could not be read"):
        load_keys(settings(api_key_env=None, api_key_file=str(tmp_path / "absent")))


def test_a_key_is_rotated_in_a_running_server(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The done-when of TASK-0492: no restart between the old key working and the new one."""
    monkeypatch.setattr(auth, "KEY_FILE_CHECK_SECONDS", 0.0)
    key_file = tmp_path / "keys"
    key_file.write_text(f"{hashed(OLD)}:admin\n", encoding="utf-8")
    app = create_app(settings(api_key_env=None, api_key_file=str(key_file)))
    app.state.vector_db = StubVectorDB()

    with TestClient(app) as client:
        before = client.get("/metrics", headers=bearer(OLD))
        write_atomically(key_file, f"{hashed(NEW)}:admin\n# old key removed\n")
        old_after = client.get("/metrics", headers=bearer(OLD))
        new_after = client.get("/metrics", headers=bearer(NEW))

    assert before.status_code == 200
    assert old_after.status_code == 401
    assert new_after.status_code == 200


def test_a_key_file_that_breaks_keeps_the_keys_in_force(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Locking every caller out because one line was mistyped would be an outage."""
    monkeypatch.setattr(auth, "KEY_FILE_CHECK_SECONDS", 0.0)
    key_file = tmp_path / "keys"
    key_file.write_text(f"{OLD}:admin\n", encoding="utf-8")
    app = create_app(settings(api_key_env=None, api_key_file=str(key_file)))
    app.state.vector_db = StubVectorDB()

    with caplog.at_level(logging.ERROR), TestClient(app) as client:
        write_atomically(key_file, f"{OLD}:admin\nshort:query\n")
        response = client.get("/metrics", headers=bearer(OLD))

    assert response.status_code == 200
    assert any("could not be loaded" in record.getMessage() for record in caplog.records)


class _Recorder:
    """Answers 200 and records nothing; the status is what is under test."""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})


async def status_of(middleware: AuthMiddleware, secret: str) -> int:
    """Send one `GET /metrics` with ``secret`` and return the status it was answered with."""
    seen: list[int] = []

    async def receive() -> dict[str, object]:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        if message["type"] == "http.response.start":
            seen.append(int(message["status"]))

    headers = [(b"authorization", f"Bearer {secret}".encode("latin-1"))]
    await middleware(
        {"type": "http", "path": "/metrics", "method": "GET", "headers": headers}, receive, send
    )
    return seen[0]


async def test_a_requested_reload_rereads_a_file_whose_stamp_did_not_move(
    tmp_path: Path,
) -> None:
    """SIGHUP's handler: an operator who restored a file's mtime can still force a re-read."""
    key_file = tmp_path / "keys"
    key_file.write_text(f"{OLD}:admin\n", encoding="utf-8")
    middleware = AuthMiddleware(_Recorder(), settings(api_key_env=None, api_key_file=str(key_file)))
    stamp = key_file.stat()
    key_file.write_text(f"{NEW}:admin\n", encoding="utf-8")
    os.utime(key_file, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    middleware._next_key_check = float("inf")

    assert await status_of(middleware, NEW) == 401

    middleware.request_reload()

    assert await status_of(middleware, NEW) == 200
    assert await status_of(middleware, OLD) == 401


def test_the_rate_limit_counts_a_key_by_its_digest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The limiter is handed the digest, so no limiter backend ever sees the key."""
    seen: list[str] = []

    class Recording(RateLimiter):
        async def allow(self, identity: str) -> RateLimitDecision:
            seen.append(identity)
            return RateLimitDecision(True, 0)

        async def close(self) -> None:
            return None

    monkeypatch.setattr(auth, "create_rate_limiter", lambda _settings: Recording())
    monkeypatch.setenv("FASTERRAG_API_KEY", NEW)
    app = create_app(settings())
    app.state.vector_db = StubVectorDB()

    with TestClient(app) as client:
        client.get("/metrics", headers=bearer(NEW))

    assert seen == [hashlib.sha256(NEW.encode()).hexdigest()]


def test_an_api_key_built_from_a_secret_matches_one_loaded_from_it() -> None:
    built = ApiKey.from_secret(NEW, frozenset({"query"}), name="x")

    assert KeyRegistry([built]).resolve(NEW) is built
