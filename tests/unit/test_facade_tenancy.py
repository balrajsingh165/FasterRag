"""Both facades querying as a tenant, against a backend that honours the filter grammar.

``FasterRag.ingest`` took ``tenant=`` and the three read methods did not, so with
``security.tenant_scoped_retrieval`` on every facade retrieval was refused, and the one
generation service the facade cached carried no tenant at all (TASK-0461). These cases
start the facades for real — ``async with`` and ``with`` — over the in-memory doubles the
REST tenancy tests use, so which tenant's chunks come back is observed, not asserted.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, ClassVar

import pytest

from fasterrag import facade, sync
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.journal import Journal
from tests.unit.api.test_query_tenancy import (
    ACME_TEXT,
    GLOBEX_TEXT,
    EchoLLM,
    MemoryRouter,
    MemoryVectorDB,
)


class CountingLLM(EchoLLM):
    """Counts closes, so a test can see who owns the provider client."""

    built: ClassVar[list[CountingLLM]] = []

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.closed = 0
        CountingLLM.built.append(self)

    async def close(self) -> None:
        self.closed += 1


def scoped_settings() -> Settings:
    """Return settings with tenant-scoped retrieval on and no model downloads."""
    return Settings.model_validate(
        {
            "security": {"auth": True, "multi_tenancy": True, "tenant_scoped_retrieval": True},
            "retrieval": {"rerank": False},
            "cache": {"semantic": False},
        }
    )


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> MemoryVectorDB:
    """Wire both facades' factories to an in-memory backend holding one chunk per tenant."""
    backend = MemoryVectorDB()
    backend.seed("default", "c_acme", ACME_TEXT, "acme")
    backend.seed("default", "c_globex", GLOBEX_TEXT, "globex")

    CountingLLM.built = []
    monkeypatch.setattr(facade, "create_vector_db_adapter", lambda settings: backend)
    monkeypatch.setattr(facade, "create_embedding_router", lambda settings: MemoryRouter())
    monkeypatch.setattr(facade, "create_journal", lambda settings: Journal(tmp_path / "journal"))
    monkeypatch.setattr(facade, "create_llm_adapter", CountingLLM)
    return backend


def texts(chunks: list[Any]) -> set[str]:
    """Return the chunk texts a retrieval returned."""
    return {chunk.text for chunk in chunks}


async def test_each_tenant_retrieves_only_its_own_chunks(store: MemoryVectorDB) -> None:
    """With scoping on, the facade could not retrieve as anyone: it refused every call."""
    async with facade.FasterRag.from_settings(scoped_settings()) as rag:
        acme = await rag.retrieve("thirty days", tenant="acme")
        globex = await rag.retrieve("thirty days", tenant="globex")

    assert texts(acme) == {ACME_TEXT}
    assert texts(globex) == {GLOBEX_TEXT}


async def test_retrieving_without_a_tenant_is_still_refused_under_scoping(
    store: MemoryVectorDB,
) -> None:
    async with facade.FasterRag.from_settings(scoped_settings()) as rag:
        with pytest.raises(FasterRagError) as caught:
            await rag.retrieve("thirty days")

    assert caught.value.code is ErrorCode.TENANT_FORBIDDEN


async def test_two_tenants_querying_at_once_never_see_each_others_context(
    store: MemoryVectorDB,
) -> None:
    """One cached service per facade answered every tenant with the first one's state."""
    async with facade.FasterRag.from_settings(scoped_settings()) as rag:
        answers = await asyncio.gather(
            *(
                rag.query("thirty days", tenant=tenant)
                for tenant in ("acme", "globex", "acme", "globex")
            )
        )

    for answer, (mine, theirs) in zip(
        answers,
        [(ACME_TEXT, GLOBEX_TEXT), (GLOBEX_TEXT, ACME_TEXT)] * 2,
        strict=True,
    ):
        assert mine in (answer.answer or "")
        assert theirs not in (answer.answer or "")


async def test_a_streamed_query_runs_as_the_tenant_it_names(store: MemoryVectorDB) -> None:
    async with facade.FasterRag.from_settings(scoped_settings()) as rag:
        tokens = [
            str(event.data.get("text", ""))
            async for event in rag.query_stream("thirty days", tenant="globex")
            if event.type == "token"
        ]

    streamed = "".join(tokens)
    assert GLOBEX_TEXT in streamed
    assert ACME_TEXT not in streamed


async def test_one_provider_client_serves_every_query_and_is_closed_once(
    store: MemoryVectorDB,
) -> None:
    """A service per query must not mean a connection pool per query (TASK-0466)."""
    async with facade.FasterRag.from_settings(scoped_settings()) as rag:
        await rag.query("thirty days", tenant="acme")
        await rag.query("thirty days", tenant="globex")
        assert [llm.closed for llm in CountingLLM.built] == [0]

    assert [llm.closed for llm in CountingLLM.built] == [1]


async def test_using_the_async_facade_unstarted_is_a_usage_error() -> None:
    """Calling the library the wrong way was reported as a 500-class CONFIG_INVALID."""
    rag = facade.FasterRag.from_settings(Settings())

    with pytest.raises(FasterRagError) as caught:
        await rag.retrieve("anything", tenant="acme")

    assert caught.value.code is ErrorCode.USAGE_ERROR
    assert caught.value.retryable is False


def test_using_the_blocking_facade_unstarted_is_a_usage_error() -> None:
    rag = sync.FasterRag.from_settings(Settings())

    with pytest.raises(FasterRagError) as caught:
        rag.retrieve("anything")

    assert caught.value.code is ErrorCode.USAGE_ERROR


def test_the_blocking_facade_passes_the_tenant_through(store: MemoryVectorDB) -> None:
    with sync.FasterRag.from_settings(scoped_settings()) as rag:
        acme = rag.retrieve("thirty days", tenant="acme")
        answer = rag.query("thirty days", tenant="globex")
        streamed = "".join(
            str(event.data.get("text", ""))
            for event in rag.query_stream("thirty days", tenant="acme")
            if event.type == "token"
        )

    assert texts(acme) == {ACME_TEXT}
    assert GLOBEX_TEXT in (answer.answer or "")
    assert ACME_TEXT not in (answer.answer or "")
    assert ACME_TEXT in streamed
    assert GLOBEX_TEXT not in streamed


def loop_threads() -> list[threading.Thread]:
    """Return the live threads the blocking facade runs its private loop on."""
    return [thread for thread in threading.enumerate() if thread.name == sync._LOOP_THREAD_NAME]


async def test_the_blocking_facade_works_inside_a_running_event_loop(
    store: MemoryVectorDB,
) -> None:
    """A notebook always has a running loop, and the blocking facade refused to start in one."""
    with sync.FasterRag.from_settings(scoped_settings()) as rag:
        assert len(loop_threads()) == 1
        chunks = rag.retrieve("thirty days", tenant="acme")

    assert texts(chunks) == {ACME_TEXT}
    assert loop_threads() == []
    assert asyncio.get_running_loop().is_running()


def test_the_blocking_facade_works_when_started_from_asyncio_run(store: MemoryVectorDB) -> None:
    """The acceptance case TASK-0497 names: start it inside asyncio.run, complete a retrieve."""

    async def notebook_cell() -> set[str]:
        with sync.FasterRag.from_settings(scoped_settings()) as rag:
            return texts(rag.retrieve("thirty days", tenant="globex"))

    assert asyncio.run(notebook_cell()) == {GLOBEX_TEXT}
    assert loop_threads() == []


async def test_a_call_from_the_facades_own_loop_thread_is_refused(
    store: MemoryVectorDB,
) -> None:
    """That thread would block on a future only it can resolve, and hang with no message."""
    with sync.FasterRag.from_settings(scoped_settings()) as rag:

        async def reenter() -> list[Any]:
            return rag.retrieve("thirty days", tenant="acme")

        with pytest.raises(FasterRagError) as caught:
            rag._run(reenter())

        assert caught.value.code is ErrorCode.USAGE_ERROR
        assert "async with" in caught.value.detail
        assert texts(rag.retrieve("thirty days", tenant="acme")) == {ACME_TEXT}

    assert loop_threads() == []


async def test_a_failed_start_inside_a_running_loop_leaves_no_thread_behind(
    store: MemoryVectorDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(settings: Settings) -> MemoryRouter:
        raise FasterRagError("no embedder", code=ErrorCode.CONFIG_INVALID)

    monkeypatch.setattr(facade, "create_embedding_router", refuse)
    rag = sync.FasterRag.from_settings(scoped_settings())

    with pytest.raises(FasterRagError, match="no embedder"):
        rag.__enter__()

    assert loop_threads() == []
    assert rag._loop is None


async def test_doctor_runs_unentered_inside_a_running_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """asyncio.run inside a running loop raises, so doctor needs a thread there too."""
    from fasterrag.services.doctor import DoctorReport

    async def report(self: facade.FasterRag) -> DoctorReport:
        return DoctorReport(checks=[])

    monkeypatch.setattr(facade.FasterRag, "doctor", report)

    assert sync.FasterRag.from_settings(Settings()).doctor().checks == []
