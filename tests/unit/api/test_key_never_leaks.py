"""An API key never reaches a log, a stored trace, a problem body or the metrics (TASK-0491).

Authentication logs a key by its positional name and ``config show`` redacts, but nothing
asserted that the bearer value itself is absent from what the service writes. This module
sends three distinctive values — a full-scope key, an ingest-only key and a refused one —
through the success, 401, 403, 422 and 500 paths of a real application, with a real
generation pipeline writing to a real trace store, and then searches every surface a value
could escape to: the raw log records, the JSON lines the configured handler wrote, every
stored trace file, every response body and header, and the ``/metrics`` exposition.

Each search also proves it searched something: an empty log, an empty trace directory or a
missing response would make "absent" true and worthless.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from fasterrag.adapters.llm.base import LLMAdapter
from fasterrag.api import query as query_router
from fasterrag.api.dependencies import shared_cache, shared_embedding_router, shared_llm
from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.observability.logging import REDACTED, JsonFormatter
from fasterrag.services.generation import GenerationService
from fasterrag.services.journal import Journal
from fasterrag.services.querying import FULL_MODE, Retrieval
from fasterrag.services.traces import TraceStore
from tests.unit.api.conftest import StubVectorDB
from tests.unit.services.test_generation import ScriptedLLM

FULL_KEY: Final = "frk-task0491-full-3c9a51e7d2b84f60"
INGEST_KEY: Final = "frk-task0491-ingest-8e2f47b1a9c6d305"
REFUSED_KEY: Final = "frk-task0491-refused-5b7d9e1f3a2c4068"
VALUES: Final = (FULL_KEY, INGEST_KEY, REFUSED_KEY)


class SwitchableRetrieval:
    """Returns one chunk, or raises an unclassified error to drive the 500 path."""

    def __init__(self) -> None:
        self.fail = False

    async def search(self, text: str, **kwargs: Any) -> Retrieval:
        if self.fail:
            raise RuntimeError("the retrieval backend fell over")
        chunk = ScoredChunk(chunk_id="c_a", text="Notice is thirty days.", rrf_score=0.5)
        return Retrieval(chunks=[chunk], mode=FULL_MODE)


class StubRouter:
    """Stands in for the tiered embedding router."""

    default = object()

    async def close(self) -> None:
        return None


@pytest.fixture
def retrieval() -> SwitchableRetrieval:
    return SwitchableRetrieval()


@pytest.fixture
def app(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, retrieval: SwitchableRetrieval) -> FastAPI:
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{FULL_KEY}:*;{INGEST_KEY}:ingest")
    settings = Settings.model_validate({"security": {"auth": True}})
    application = create_app(settings)
    application.state.vector_db = StubVectorDB()
    application.state.journal = Journal(tmp_path / "journal")
    application.state.traces = TraceStore(tmp_path / "traces")

    llm = ScriptedLLM(settings, text="The notice period is thirty days [^c_a].")
    application.dependency_overrides[shared_embedding_router] = StubRouter
    application.dependency_overrides[shared_cache] = lambda: None
    application.dependency_overrides[shared_llm] = lambda: llm

    def build_generation(
        settings: Settings,
        adapter: Any,
        router: Any,
        cache: Any = None,
        traces: TraceStore | None = None,
        tenant: str | None = None,
        llm: LLMAdapter | None = None,
    ) -> GenerationService:
        assert llm is not None
        return GenerationService(
            settings,
            retrieval,  # type: ignore[arg-type]  # CRITICAL: a duck-typed retrieval double.
            llm,
            traces=traces,
            tenant=tenant,
            owns_llm=False,
        )

    monkeypatch.setattr(query_router, "build_generation", build_generation)
    return application


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def exposed(text: str) -> list[str]:
    """Return every distinctive value that occurs in ``text``."""
    return [value for value in VALUES if value in text]


def test_no_key_value_reaches_a_log_a_trace_a_problem_body_or_the_metrics(
    client: TestClient,
    retrieval: SwitchableRetrieval,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """Nothing asserted that a bearer value stays out of logs, traces and error bodies."""
    caplog.set_level(logging.DEBUG)
    question = {"query": "what is the notice period?", "stream": False}

    responses = {
        "success": client.post("/v1/query", json=question, headers=bearer(FULL_KEY)),
        "missing": client.post("/v1/query", json=question),
        "refused": client.post("/v1/query", json=question, headers=bearer(REFUSED_KEY)),
        "scope": client.post("/v1/query", json=question, headers=bearer(INGEST_KEY)),
        "invalid": client.post("/v1/query", json={"query": ""}, headers=bearer(FULL_KEY)),
    }
    retrieval.fail = True
    responses["internal"] = client.post("/v1/query", json=question, headers=bearer(FULL_KEY))
    responses["metrics"] = client.get("/metrics", headers=bearer(FULL_KEY))

    assert {name: response.status_code for name, response in responses.items()} == {
        "success": 200,
        "missing": 401,
        "refused": 401,
        "scope": 403,
        "invalid": 422,
        "internal": 500,
        "metrics": 200,
    }
    for name, response in responses.items():
        headers = "\n".join(f"{key}: {value}" for key, value in response.headers.items())
        assert exposed(response.text + headers) == [], name

    formatter = JsonFormatter()
    assert caplog.records, "no log record was captured, so absence proves nothing"
    for record in caplog.records:
        assert exposed(formatter.format(record)) == [], record.getMessage()
    assert exposed(capsys.readouterr().err) == []

    stored = [path for path in (tmp_path / "traces").rglob("*") if path.is_file()]
    assert stored, "no trace was stored, so absence proves nothing"
    for path in stored:
        assert exposed(path.read_text(encoding="utf-8")) == [], path.name


def test_the_api_masks_a_key_held_in_a_variable_of_any_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A mask keyed on credential-like names alone would miss a renamed ``api_key_env``."""
    monkeypatch.setenv("CONTROL_PLANE_CREDENTIALS", f"{FULL_KEY}:*")
    settings = Settings.model_validate(
        {"security": {"auth": True, "api_key_env": "CONTROL_PLANE_CREDENTIALS"}}
    )
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        create_app(settings)
        logging.getLogger("fasterrag.probe").warning("logged by mistake: %s", FULL_KEY)
    finally:
        root.handlers = handlers
        root.setLevel(level)

    written = capsys.readouterr().err
    assert f"logged by mistake: {REDACTED}" in written
    assert FULL_KEY not in written
