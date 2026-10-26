"""The network-partition chaos scenario FMEA rows 16 and 35 name (TASK-0241).

Both rows cite "chaos: partition injection" as their proving test and neither had one. A
partition is not the stop-the-container fault ``test_chaos.py`` and ``test_real_faults.py``
already inject, and the difference is the whole point: a stopped container takes its
listening socket with it, so the kernel *refuses* the connection and the failure arrives in
milliseconds. A partitioned peer still answers the SYN — or silently drops the packets —
so the connection is established or left pending and nothing ever comes back. The only
thing that ends the call is our own timeout, and a system without one hangs until the
client gives up, which is the failure mode these rows exist to rule out.

The fault is injected at the socket, not at a code seam: a real TCP listener accepts the
connection on a real ephemeral port and then never writes a byte, so the client library and
the operating system behave exactly as they do against an unreachable peer. Nothing here
needs Docker or a live backend, which is why the whole module runs on any machine — a
partition is one of the few environmental faults that can be reproduced honestly without
one. The companion to row 35's "DNS" half uses a name in the ``.invalid`` TLD, which RFC
2606 reserves precisely so that it cannot resolve.

**What is asserted is the built behavior, not the specified one.** Row 16's mitigation
column names the ``cache_only`` degradation rung; that rung is not implemented (TASK-0159),
so a partitioned vector database surfaces a retryable ``RETRIEVAL_FAILED`` and, once the
breaker opens, ``CIRCUIT_OPEN``. Asserting the rung here would be asserting a plan.

The latency assertions are bounds in both directions, never a figure: the upper bound is
the configured timeout plus slack, and the lower bound is what separates a partition from a
refused connection. Neither is a measurement, so neither belongs in the benchmark ledger.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any, Final

import pytest

from fasterrag.adapters.vectordb.base import SearchQuery
from fasterrag.adapters.vectordb.qdrant import QdrantAdapter
from fasterrag.api.problems import build_problem
from fasterrag.config.schema import Settings
from fasterrag.core.breaker import CircuitBreaker, CircuitOpenError, CircuitState
from fasterrag.errors import ErrorCode, FasterRagError, GenerationError, RetrievalError
from fasterrag.services.querying import RetrievalService

pytestmark = pytest.mark.chaos

COLLECTION: Final = "docs"
VECTOR: Final[list[float]] = [0.1, 0.2, 0.3]
UNRESOLVABLE_HOST: Final = "vector-db.partitioned.invalid"
SERVICE_UNAVAILABLE: Final = 503

PARTITION_TIMEOUT_MS: Final = 1000
"""The vector-database budget every partitioned call in this module waits out.

One second is the floor, not a tuning choice: the adapter rounds the budget to whole
seconds and floors it at one, so a smaller configured value would change nothing the
client does and would only misdescribe what the scenario proves.
"""

TIMEOUT_SLACK_SECONDS: Final = 5.0
HANG_BUDGET_SECONDS: Final = 30.0
SHED_BUDGET_SECONDS: Final = 0.5
RESET_TIMEOUT_MS: Final = 200


class Blackhole:
    """Two TCP listeners that accept a connection and then never answer.

    This is what a partition looks like to a client: the socket is open, the request is
    written, and no response ever arrives. The handler holds each connection open rather
    than closing it, because a close would be read as a reset and the client would fail
    fast on a transport error instead of waiting for its own deadline.
    """

    def __init__(self) -> None:
        """Start with no servers; :meth:`open` binds them."""
        self._servers: list[asyncio.Server] = []
        self._held: list[asyncio.StreamWriter] = []
        self._handlers: list[asyncio.Task[None]] = []
        self.rest_port = 0
        self.grpc_port = 0

    async def _swallow(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Accept the connection and never reply to it.

        The handler parks on an event nothing sets, so :meth:`close` has to cancel it. Its
        task is recorded here rather than tracked by the server, because an abandoned
        handler outlives the fixture and is reported as a destroyed pending task.
        """
        handler = asyncio.current_task()
        if handler is not None:
            self._handlers.append(handler)
        self._held.append(writer)
        await asyncio.Event().wait()

    async def open(self) -> None:
        """Bind both listeners on ephemeral ports.

        Two are bound rather than one because the schema refuses a REST port equal to the
        gRPC port, and a partition has to be injected on whichever of the two the client
        chooses.
        """
        for _ in range(2):
            server = await asyncio.start_server(self._swallow, "127.0.0.1", 0)
            self._servers.append(server)
        self.rest_port = int(self._servers[0].sockets[0].getsockname()[1])
        self.grpc_port = int(self._servers[1].sockets[0].getsockname()[1])

    async def close(self) -> None:
        """Cancel every handler, drop every held connection, and stop listening.

        The handlers are gathered after cancellation so the loop sees each one finish;
        ``return_exceptions`` keeps the ``CancelledError`` each of them raises from
        propagating into the fixture teardown that asked for the cancellation.
        """
        for handler in self._handlers:
            handler.cancel()
        if self._handlers:
            await asyncio.gather(*self._handlers, return_exceptions=True)
        self._handlers.clear()
        for writer in self._held:
            writer.close()
        self._held.clear()
        for server in self._servers:
            server.close()
            await server.wait_closed()
        self._servers.clear()


def handler_tasks() -> list[asyncio.Task[Any]]:
    """Return every blackhole connection handler still running in this loop."""
    return [
        task
        for task in asyncio.all_tasks()
        if "Blackhole._swallow" in str(getattr(task.get_coro(), "__qualname__", ""))
    ]


def settings_for(host: str, rest_port: int, grpc_port: int) -> Settings:
    """Return settings pointing the vector database at a partitioned peer."""
    return Settings.model_validate(
        {
            "vector_db": {
                "provider": "qdrant",
                "mode": "external",
                "host": host,
                "port": rest_port,
                "grpc_port": grpc_port,
                "api_key_env": None,
            },
            "embeddings": {"provider": "huggingface"},
            "llm": {"provider": "ollama", "api_key_env": None},
            "reliability": {
                "timeouts": {"vector_db_ms": PARTITION_TIMEOUT_MS, "llm_ms": 1000},
                "retries": {"max_attempts": 1, "backoff_base_ms": 1},
            },
        }
    )


class StubRouter:
    """Supplies a fixed query vector so no embedding model is loaded.

    The query embedding happens deliberately *outside* the vector-database breaker, so a
    real embedder here would add a second failure domain to a scenario about one.
    """

    def __init__(self) -> None:
        """Expose itself as its own default adapter, as the real router does."""
        self.default = self

    async def embed_query(self, text: str) -> Sequence[float]:
        """Return the fixed vector."""
        return VECTOR


def query() -> SearchQuery:
    """Return the one search every case in this module runs."""
    return SearchQuery(collection=COLLECTION, vector=VECTOR, limit=1)


@pytest.fixture
async def blackhole() -> AsyncIterator[Blackhole]:
    """Return two listeners that swallow everything written to them."""
    partition = Blackhole()
    await partition.open()
    yield partition
    await partition.close()


@pytest.fixture
def partitioned(blackhole: Blackhole) -> Settings:
    """Return settings whose vector database is reachable but silent."""
    return settings_for("127.0.0.1", blackhole.rest_port, blackhole.grpc_port)


@pytest.fixture
async def adapter(partitioned: Settings) -> AsyncIterator[QdrantAdapter]:
    """Return an adapter pointed at the partitioned peer."""
    built = QdrantAdapter(partitioned)
    yield built
    await built.close()


async def failing_search(adapter: QdrantAdapter) -> tuple[FasterRagError, float]:
    """Run the search that cannot succeed, returning the typed failure and its duration.

    ``asyncio.wait_for`` is the outer guard rather than the thing under test: if the budget
    the adapter configures were not honoured, this case must fail on an assertion rather
    than hang the suite for as long as the client is willing to wait.
    """
    started = time.perf_counter()
    with pytest.raises(FasterRagError) as failure:
        await asyncio.wait_for(adapter.search(query()), timeout=HANG_BUDGET_SECONDS)
    return failure.value, time.perf_counter() - started


async def test_a_closed_blackhole_leaves_no_handler_task_behind() -> None:
    """Teardown has to reclaim the parked handlers rather than abandon them.

    ``_swallow`` waits on an event nothing ever sets, so closing the held writers and the
    listeners does not end it. An abandoned handler stays pending for the lifetime of the
    loop and surfaces later as a destroyed-pending-task report in whichever unrelated test
    happens to be running when the loop shuts down.
    """
    partition = Blackhole()
    await partition.open()
    _, writer = await asyncio.open_connection("127.0.0.1", partition.rest_port)
    try:
        writer.write(b"GET /readyz HTTP/1.1\r\n\r\n")
        await writer.drain()
        await asyncio.sleep(0.05)
        assert handler_tasks(), "no handler ever ran, so a clean teardown would prove nothing"
    finally:
        writer.close()
        await partition.close()

    assert handler_tasks() == []


async def test_a_partitioned_vector_database_fails_typed_and_retryable(
    adapter: QdrantAdapter,
) -> None:
    """Row 16: the failure is ``RETRIEVAL_FAILED`` and retryable, never a vendor exception.

    Retryable is the load-bearing half. A partition is by definition temporary, and a
    failure classified permanent would neither be retried nor counted by the breaker, so
    the restored network would never be noticed.
    """
    failure, _ = await failing_search(adapter)

    assert isinstance(failure, RetrievalError)
    assert failure.code is ErrorCode.RETRIEVAL_FAILED
    assert failure.retryable is True
    assert failure.status == SERVICE_UNAVAILABLE


async def test_a_partition_fails_on_our_timeout_rather_than_hanging(
    adapter: QdrantAdapter, partitioned: Settings
) -> None:
    """Row 16: ``reliability.timeouts.vector_db_ms`` is what ends the call.

    Both bounds are asserted and both mean something. The upper bound says the query is
    bounded at all — the guarantee. The lower bound says the fault injected was really a
    partition: a refused connection returns in single-digit milliseconds, so a run that
    came back well inside the budget would have proved the stop-the-container case over
    again under a different name.
    """
    budget = partitioned.reliability.timeouts.vector_db_ms / 1000

    _, elapsed = await failing_search(adapter)

    assert elapsed < budget + TIMEOUT_SLACK_SECONDS
    assert elapsed >= budget / 2


async def test_a_partition_never_yields_a_generic_500(adapter: QdrantAdapter) -> None:
    """The partition renders as a problem document a client can branch on."""
    failure, _ = await failing_search(adapter)

    problem = build_problem(failure)

    assert problem.status == SERVICE_UNAVAILABLE
    assert problem.retryable is True
    assert problem.code is not ErrorCode.INTERNAL
    assert problem.trace_id


async def test_a_partitioned_vector_database_reports_itself_unhealthy(
    adapter: QdrantAdapter,
) -> None:
    """The readiness probe describes the partition instead of raising through it."""
    status = await adapter.health()

    assert status.healthy is False
    assert status.detail


async def test_a_host_that_does_not_resolve_fails_typed_and_retryable() -> None:
    """Row 35's DNS half: a name that cannot resolve is a retryable provider failure.

    ``.invalid`` is reserved by RFC 2606 so that it never resolves. A resolver that
    hijacks NXDOMAIN and answers with its own address would turn this into a connection
    failure against that address instead, which is the same classification by a different
    route — so the case asserts the taxonomy and the bound, not which syscall failed.
    """
    settings = settings_for(UNRESOLVABLE_HOST, 6333, 6334)
    adapter = QdrantAdapter(settings)
    try:
        failure, elapsed = await failing_search(adapter)
    finally:
        await adapter.close()

    assert isinstance(failure, RetrievalError)
    assert failure.code is ErrorCode.RETRIEVAL_FAILED
    assert failure.retryable is True
    assert elapsed < settings.reliability.timeouts.vector_db_ms / 1000 + TIMEOUT_SLACK_SECONDS


async def test_a_partitioned_generation_provider_fails_typed_and_retryable(
    blackhole: Blackhole, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Row 35 per provider: the same partition against the LLM leg is classified the same.

    Row 35 says "any provider", and a taxonomy that holds only where it was first written
    is not a taxonomy. The ``openai_compatible`` provider is used because it is the one
    whose endpoint is configurable, which is what lets a real client be pointed at the
    blackhole. Every generation provider sits behind an optional extra, and stubbing the
    SDK would assert the stub's exception hierarchy rather than the one the taxonomy maps,
    so the case skips where the extra is absent — which means it gates row 35 only in an
    environment that installs ``.[openai]``.
    """
    pytest.importorskip("openai", reason="the openai extra is not installed in this environment")

    monkeypatch.setenv("FASTERRAG_CHAOS_LLM_KEY", "not-a-real-key")
    settings = Settings.model_validate(
        {
            "vector_db": {"provider": "qdrant", "mode": "external", "api_key_env": None},
            "embeddings": {"provider": "huggingface"},
            "llm": {
                "provider": "openai_compatible",
                "api_key_env": "FASTERRAG_CHAOS_LLM_KEY",
                "base_url": f"http://127.0.0.1:{blackhole.rest_port}/v1",
            },
            "reliability": {"timeouts": {"llm_ms": 1000}},
        }
    )
    from fasterrag.adapters.llm.openai import OpenAICompatibleGenerator

    generator = OpenAICompatibleGenerator(settings)
    started = time.perf_counter()
    try:
        with pytest.raises(GenerationError) as failure:
            await asyncio.wait_for(
                generator.complete("what is the notice period?"), HANG_BUDGET_SECONDS
            )
    finally:
        await generator.close()
    elapsed = time.perf_counter() - started

    assert failure.value.retryable is True
    assert elapsed < settings.reliability.timeouts.llm_ms / 1000 + TIMEOUT_SLACK_SECONDS


def service_for(partitioned: Settings, breaker: CircuitBreaker) -> RetrievalService:
    """Build a retrieval service over the partitioned backend with a breaker injected.

    The breaker is injected rather than configured because ``reliability.circuit_breaker``
    floors its reset timeout at a second and defaults the threshold to five — correct for
    production and five partitioned queries of waiting for a scenario.
    """
    return RetrievalService(
        partitioned,
        QdrantAdapter(partitioned),
        StubRouter(),  # type: ignore[arg-type]
        reranker=None,
        breaker=breaker,
    )


def chaos_breaker(**overrides: Any) -> CircuitBreaker:
    """Return a breaker that opens on the first failure unless told otherwise."""
    return CircuitBreaker(
        provider="vector_db",
        failure_threshold=overrides.pop("failure_threshold", 1),
        reset_timeout_ms=overrides.pop("reset_timeout_ms", RESET_TIMEOUT_MS),
    )


async def test_a_partition_opens_the_vector_database_breaker(partitioned: Settings) -> None:
    """Row 16: retries first, then the breaker opens — the second query is shed, not sent.

    Shedding is the behavior that matters during a partition. Without it every query pays
    the full timeout, and those stacked-up waits are themselves the load that keeps a
    recovering backend down. The elapsed bound is what proves the second query was refused
    rather than attempted; without it this case would pass against a breaker that never
    opened at all.
    """
    service = service_for(partitioned, chaos_breaker())
    try:
        with pytest.raises(RetrievalError):
            await asyncio.wait_for(service.search("q"), HANG_BUDGET_SECONDS)

        started = time.perf_counter()
        with pytest.raises(CircuitOpenError) as shed:
            await asyncio.wait_for(service.search("q"), HANG_BUDGET_SECONDS)
        elapsed = time.perf_counter() - started
    finally:
        await service.adapter.close()

    assert shed.value.code is ErrorCode.CIRCUIT_OPEN
    assert shed.value.retryable is True
    assert service.breaker.is_open is True
    assert elapsed < SHED_BUDGET_SECONDS


async def test_the_breaker_admits_one_probe_once_its_reset_timeout_passes(
    partitioned: Settings,
) -> None:
    """Recovery costs one request: after the timeout the next query is tried, not shed.

    The partition is still in place here, which is the honest half of the recovery story —
    what can be proved without a live backend is that the breaker stops shedding and lets a
    probe through, and that a failed probe re-opens it rather than granting a fresh
    threshold's worth of doomed calls. That the probe *succeeds* once the network returns is
    the vector-database half, covered against a real container in
    ``test_real_faults.py::test_a_restarted_container_serves_its_indexed_data_again`` and,
    at the breaker itself, by
    ``tests/unit/core/test_breaker.py::test_a_successful_probe_closes_it``.
    """
    service = service_for(partitioned, chaos_breaker())
    try:
        with pytest.raises(RetrievalError):
            await asyncio.wait_for(service.search("q"), HANG_BUDGET_SECONDS)

        assert service.breaker.is_open is True
        await asyncio.sleep(RESET_TIMEOUT_MS / 1000)

        assert service.breaker.state is CircuitState.HALF_OPEN

        with pytest.raises(RetrievalError):
            await asyncio.wait_for(service.search("q"), HANG_BUDGET_SECONDS)
    finally:
        await service.adapter.close()

    assert service.breaker.is_open is True
