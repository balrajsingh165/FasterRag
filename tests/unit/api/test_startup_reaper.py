"""A starting application settles the jobs an earlier process left ``running`` (TASK-0436).

The journal was built lazily by the first request that asked for it, so nothing could look
at it during startup, and nothing did: a job interrupted by the process exiting kept its
``running`` record through every later restart.
"""

import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.services.jobs import INTERRUPTED
from fasterrag.services.journal import Journal, QuietJobs
from tests.unit.api.conftest import StubVectorDB
from tests.unit.services.test_job_reaper import left_behind

WAIT_SECONDS = 10


def wait_for_status(client: TestClient, job_id: str, status: str) -> dict[str, object]:
    """Poll a job until it reports ``status``, failing the test if it never does."""
    deadline = time.monotonic() + WAIT_SECONDS
    while True:
        body: dict[str, object] = client.get(f"/v1/ingest/{job_id}").json()
        if body.get("status") == status:
            return body
        assert time.monotonic() < deadline, f"job {job_id} never became {status}: {body}"
        time.sleep(0.01)


def test_startup_settles_a_stale_running_job_and_leaves_a_fresh_one_alone(app: FastAPI) -> None:
    """Nothing looked for interrupted jobs at startup, so they stayed ``running`` for good."""
    journal: Journal = app.state.journal
    stale = left_behind(journal, quiet_for=3600)
    fresh = left_behind(journal, quiet_for=1)

    began = time.monotonic()
    with TestClient(app) as client:
        settled = wait_for_status(client, stale.job_id, "failed")
        untouched = client.get(f"/v1/ingest/{fresh.job_id}").json()
    took = time.monotonic() - began

    assert settled["reason_code"] == INTERRUPTED
    assert (untouched["status"], untouched["reason_code"]) == ("running", None)
    assert took < WAIT_SECONDS, "shutdown waited for the second look instead of cancelling it"


def test_the_journal_is_built_at_startup_rather_than_by_the_first_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A journal that exists only once a request asks for it cannot be read at startup."""
    monkeypatch.chdir(tmp_path)
    application = create_app(Settings())
    application.state.vector_db = StubVectorDB()

    with TestClient(application):
        journal = getattr(application.state, "journal", None)

    assert isinstance(journal, Journal)


def test_a_journal_the_test_supplied_is_kept(app: FastAPI) -> None:
    supplied = app.state.journal

    with TestClient(app):
        assert app.state.journal is supplied


class UnreadableJournal(Journal):
    """A journal whose directory cannot be listed."""

    def quiet_running_jobs_blocking(self, quiet_seconds: float) -> QuietJobs:
        raise PermissionError("the journal directory cannot be read")


def test_an_application_whose_journal_cannot_be_scanned_still_serves(
    app: FastAPI, tmp_path: Path
) -> None:
    """Tidying up after an earlier process must not be able to stop this one starting."""
    app.state.journal = UnreadableJournal(tmp_path / "unreadable")

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
