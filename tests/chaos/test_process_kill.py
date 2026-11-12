"""Kill a real process mid-job, start another, and find no job left ``running`` (TASK-0436).

Every other test of this behaviour cancels a task or prepares a journal directory by hand.
This one does what an operator's worst day does: a real server, over a real socket, is
killed with no chance to settle anything while a job is part-way through a corpus, and a
second process is started on the same journal. ``Popen.kill`` is ``SIGKILL`` on POSIX and
``TerminateProcess`` on Windows, so neither platform runs a handler.

The kill is taken while the job is parked inside the embedding call rather than at a random
instant, so no journal write is in flight. That is deliberate: a kill landing between the
two renames of an atomic write is a different failure (a missing record, TASK-0437), and a
drill that sometimes hit it would be reporting on that task, not this one.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import httpx
import pytest

from fasterrag.services.jobs import INTERRUPTED
from fasterrag.services.journal import DEFAULT_JOURNAL_ROOT, Journal

pytestmark = [pytest.mark.chaos, pytest.mark.must_run]

REPO_ROOT: Final = Path(__file__).resolve().parents[2]
SERVER: Final = Path(__file__).with_name("interrupted_job_server.py")
STARTUP_SECONDS: Final = 120
SETTLE_SECONDS: Final = 60

DOCUMENTS: Final = ("alpha", "bravo", "charlie", "delta")
BODY: Final[dict[str, Any]] = {
    "sources": [
        {"type": "inline", "value": f"# {word}\n\nThe {word} policy allows {index} days.\n"}
        for index, word in enumerate(DOCUMENTS)
    ],
    "idempotency_key": "kill-and-restart-drill",
}


def wait_for(path: Path, process: subprocess.Popen[bytes], seconds: float) -> str:
    """Return a file's text once the server has written it, failing if the server died."""
    deadline = time.monotonic() + seconds
    while not path.exists():
        assert process.poll() is None, f"the server exited with {process.returncode}"
        assert time.monotonic() < deadline, f"{path.name} was not written within {seconds}s"
        time.sleep(0.05)
    return path.read_text(encoding="utf-8")


@contextmanager
def server(
    workspace: Path, mode: str
) -> Iterator[tuple[httpx.Client, Path, subprocess.Popen[bytes]]]:
    """Run the drill's API process in ``workspace`` and always leave it dead afterwards."""
    port_file = workspace / f"port-{mode}"
    hung_file = workspace / f"hung-{mode}"
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(REPO_ROOT / "src"), str(REPO_ROOT)]),
    }
    process = subprocess.Popen(
        [sys.executable, str(SERVER), mode, str(port_file), str(hung_file)],
        cwd=workspace,
        env=environment,
    )
    try:
        port = wait_for(port_file, process, STARTUP_SECONDS)
        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=30) as client:
            yield client, hung_file, process
            assert process.poll() is None, "the server died on its own during the drill"
    finally:
        process.kill()
        process.wait(timeout=30)


def poll_job(client: httpx.Client, job_id: str, status: str) -> dict[str, Any]:
    """Return the job's status body once it reports ``status``."""
    deadline = time.monotonic() + SETTLE_SECONDS
    while True:
        body: dict[str, Any] = client.get(f"/v1/ingest/{job_id}").json()
        if body.get("status") == status:
            return body
        assert time.monotonic() < deadline, f"job {job_id} never became {status}: {body}"
        time.sleep(0.05)


def test_a_killed_process_leaves_no_job_running_after_a_restart(tmp_path: Path) -> None:
    """A job left ``running`` by a process exit was never settled and never resumed."""
    journal = Journal(tmp_path / DEFAULT_JOURNAL_ROOT)

    with server(tmp_path, "hang") as (client, hung_file, process):
        accepted = client.post("/v1/ingest", json=BODY)
        assert accepted.status_code == 202, accepted.text
        job_id = str(accepted.json()["job_id"])
        wait_for(hung_file, process, SETTLE_SECONDS)

    abandoned = journal.load_job_blocking(job_id)
    assert abandoned.status == "running", "the kill was meant to leave the job unsettled"
    assert journal.resume_index(abandoned) == DOCUMENTS.index("charlie")

    with server(tmp_path, "run") as (client, _, _):
        settled = poll_job(client, job_id, "failed")
        replayed = client.post("/v1/ingest", json=BODY)
        finished = poll_job(client, job_id, "completed")

    assert settled["reason_code"] == INTERRUPTED
    assert replayed.status_code == 202
    assert replayed.json()["job_id"] == job_id
    assert finished["reason_code"] is None
    assert finished["counts"]["parsed"] == len(DOCUMENTS) - DOCUMENTS.index("charlie")
    statuses = {
        path.stem: journal.load_job_blocking(path.stem).status
        for path in (journal.root / "jobs").glob("*.json")
    }
    assert statuses == {job_id: "completed"}
