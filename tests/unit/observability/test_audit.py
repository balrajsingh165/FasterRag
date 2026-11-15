"""The audit trail's own guarantees: one record per action, a file sink, no secret (TASK-0493)."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.errors import ConfigError, ErrorCode, FasterRagError
from fasterrag.observability.audit import (
    AUDIT_LOGGER,
    Actor,
    audited,
    configure_audit_file,
)
from tests.unit.api.conftest import StubVectorDB

SECRET = "audit-file-secret-0123456789"
ACTOR = Actor(via="api", key="key-1", key_id="0123456789ab", tenant="acme")


@pytest.fixture(autouse=True)
def no_audit_file() -> Iterator[None]:
    """Close any audit file a test opened, so its directory can be removed on Windows."""
    yield
    configure_audit_file(None)


def written(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_file_receives_every_record_whatever_the_log_level(tmp_path: Path) -> None:
    """The trail kept for auditors must not vanish when the log level is turned down."""
    path = tmp_path / "audit" / "trail.jsonl"
    configure_audit_file(str(path))
    logging.getLogger(AUDIT_LOGGER).setLevel(logging.ERROR)
    try:
        with audited("collection.drop", "docs", ACTOR) as entry:
            entry.detail["dropped"] = True
    finally:
        logging.getLogger(AUDIT_LOGGER).setLevel(logging.NOTSET)

    [record] = written(path)
    assert record["action"] == "collection.drop"
    assert record["target"] == "docs"
    assert record["outcome"] == "succeeded"
    assert record["actor_key"] == "key-1"
    assert record["actor_key_id"] == "0123456789ab"
    assert record["tenant"] == "acme"
    assert record["detail"] == {"dropped": True}
    assert len(str(record["trace_id"])) == 32


def test_a_failure_is_recorded_once_with_its_code_and_re_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO", logger=AUDIT_LOGGER)

    with (
        pytest.raises(FasterRagError),
        audited("document.delete", "docs/d_1", ACTOR),
    ):
        raise FasterRagError("gone", code=ErrorCode.NOT_FOUND)

    records = [record for record in caplog.records if record.name == AUDIT_LOGGER]
    assert [(vars(r)["outcome"], vars(r)["error_code"]) for r in records] == [
        ("failed", "NOT_FOUND")
    ]


def test_an_unexpected_error_is_recorded_as_internal(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO", logger=AUDIT_LOGGER)

    with pytest.raises(RuntimeError), audited("doctor.fix", "deployment", ACTOR):
        raise RuntimeError("driver bug")

    records = [record for record in caplog.records if record.name == AUDIT_LOGGER]
    assert [(vars(r)["outcome"], vars(r)["error_code"]) for r in records] == [
        ("failed", "INTERNAL")
    ]


def test_a_cancelled_action_is_recorded_as_cancelled(caplog: pytest.LogCaptureFixture) -> None:
    """A client disconnect cancels the handler; the trail must not claim it succeeded."""
    caplog.set_level("INFO", logger=AUDIT_LOGGER)

    async def cancelled() -> None:
        with audited("archive.import", "a.fragx", ACTOR):
            raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancelled())

    records = [record for record in caplog.records if record.name == AUDIT_LOGGER]
    assert [vars(r)["outcome"] for r in records] == ["cancelled"]


def test_an_unwritable_audit_file_stops_startup_naming_the_key(tmp_path: Path) -> None:
    blocker = tmp_path / "a-file"
    blocker.write_text("", encoding="utf-8")

    with pytest.raises(ConfigError, match=r"security\.audit_log_file"):
        configure_audit_file(str(blocker / "trail.jsonl"))


def test_the_api_writes_its_trail_to_the_configured_file_without_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key is named by its name and digest prefix; its value appears nowhere."""
    monkeypatch.setenv("FASTERRAG_API_KEY", SECRET)
    path = tmp_path / "trail.jsonl"
    app = create_app(
        Settings.model_validate({"security": {"auth": True, "audit_log_file": str(path)}})
    )
    app.state.vector_db = StubVectorDB()

    with TestClient(app) as client:
        client.delete(
            "/v1/collections/docs?force=true", headers={"Authorization": f"Bearer {SECRET}"}
        )

    text = path.read_text(encoding="utf-8")
    assert SECRET not in text
    [record] = written(path)
    assert (record["action"], record["outcome"], record["actor_key"]) == (
        "collection.drop",
        "failed",
        "key-1",
    )
