"""The load layer that runs without a rig: what saturation must do, never how fast.

``pytest -m load`` runs this. It is deliberately not the Locust harness next door — that
one points a generator at a deployed instance, reports SLIs and grades nothing, which
makes it a benchmark and not a test (docs/testing-strategy.md §1.7). What a benchmark
cannot do is fail when the system misbehaves under concurrency, and that is the half the
FMEA rows cite. So this module drives the real code in-process at volume and asserts the
properties saturation has to preserve.

**No assertion here is a time, and none will be.** TASK-0350 measured this project's
hosted runner at **2.07x on an identical commit**, so an absolute latency bound would
measure the runner's neighbours and flap. What survives that variance is behaviour:

* **Boundedness** — the queue's occupancy is set by ``workers.queue_depth`` and not by how
  much work is offered, which is the "backpressure, never OOM" rung of FMEA row 10.
* **Conservation** — every item offered is accounted for exactly once, accepted or
  refused. A storm that silently drops work looks identical to a fast one.
* **Typed refusal** — a request a saturated system cannot serve comes back as a problem
  document with a machine-readable code, never a bare 500 and never a hang.

**What this layer cannot reach, stated so nobody reads more into a green run.** The
process is one interpreter with doubles at the vendor edges, so it says nothing about
cross-process contention, connection-pool exhaustion, or a real vector store under
concurrent writers. Those need the reference hardware of TASK-0084 and the Locust harness
pointed at it. This module is the floor, not the measurement.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, ClassVar

import pytest
from fastapi.testclient import TestClient

from fasterrag.api.main import create_app
from fasterrag.api.problems import PROBLEM_MEDIA_TYPE
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.workers.queues import BoundedQueue
from tests.soak.workload import build_service, soak_settings, write_corpus
from tests.unit.api.conftest import StubVectorDB

pytestmark = [pytest.mark.load, pytest.mark.timeout(0)]

CAPACITY = 10
STORM_ITEMS = 2_000
STORM_DOCUMENTS = 60
STORM_REQUESTS = 60
STORM_CLIENTS = 8


class PeakRecordingQueue(BoundedQueue[Any]):
    """A bounded queue that remembers the deepest it ever got."""

    peaks: ClassVar[dict[str, int]] = {}

    def _publish(self) -> None:
        """Record the occupancy alongside publishing it."""
        super()._publish()
        current = self._queue.qsize()
        PeakRecordingQueue.peaks[self.name] = max(
            PeakRecordingQueue.peaks.get(self.name, 0), current
        )


async def _drain_slowly(queue: BoundedQueue[int], expected: int) -> list[int]:
    """Consume at a rate the producer will outrun, which is what makes the queue fill."""
    received: list[int] = []
    while len(received) < expected:
        item = await queue.get()
        queue.task_done()
        if item is None:
            break
        received.append(item)
        await asyncio.sleep(0)
    return received


async def _storm_through_a_full_queue() -> tuple[list[int], int]:
    """Push far more work than the queue holds and return what arrived and what waited."""
    queue: BoundedQueue[int] = BoundedQueue(CAPACITY, name="storm")
    depths: list[int] = []

    async def produce() -> None:
        for item in range(STORM_ITEMS):
            await queue.put(item)
            depths.append(queue.depth)

    consumer = asyncio.create_task(_drain_slowly(queue, STORM_ITEMS))
    await produce()
    received = await consumer
    return received, max(depths)


def test_a_producer_storm_waits_rather_than_growing_the_queue() -> None:
    """FMEA row 10's built rung: occupancy is capped by configuration, not by offered load."""
    received, peak = asyncio.run(_storm_through_a_full_queue())

    assert peak <= CAPACITY, f"the queue reached {peak} with a capacity of {CAPACITY}"
    assert peak == CAPACITY, "the producer never outran the consumer, so nothing was saturated"
    assert received == list(range(STORM_ITEMS))


def test_every_item_a_full_queue_refuses_is_typed_and_none_is_dropped() -> None:
    """Conservation under refusal, and the one path that produces ``QUEUE_FULL``."""
    queue: BoundedQueue[int] = BoundedQueue(CAPACITY, name="refusals")
    accepted = 0
    refusals: list[ErrorCode] = []

    for item in range(STORM_ITEMS):
        try:
            queue.offer(item)
        except IngestionError as error:
            refusals.append(error.code)
        else:
            accepted += 1

    assert accepted == CAPACITY
    assert accepted + len(refusals) == STORM_ITEMS
    assert set(refusals) == {ErrorCode.QUEUE_FULL}
    assert queue.depth == CAPACITY


def test_the_pipeline_queue_is_bounded_by_configuration_not_by_the_corpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same ceiling through the real ingestion service over a corpus that overfills it."""
    PeakRecordingQueue.peaks.clear()
    monkeypatch.setattr("fasterrag.services.ingestion.BoundedQueue", PeakRecordingQueue)
    settings = soak_settings(workers={"embedding_pool_size": 1, "queue_depth": CAPACITY})
    service = build_service(tmp_path / "journal", settings)
    sources = write_corpus(tmp_path / "corpus", documents=STORM_DOCUMENTS)

    async def run() -> None:
        record = await service.accept(sources)
        settled = await service.run(record)
        assert settled.status == "completed", settled.status

    asyncio.run(run())

    peak = max(PeakRecordingQueue.peaks.values(), default=0)
    assert peak > 0, "the recording queue was never constructed, so nothing was measured"
    assert peak <= CAPACITY, f"the pipeline queue reached {peak} against a cap of {CAPACITY}"


def test_a_concurrent_request_storm_never_answers_with_an_untyped_server_error() -> None:
    """Saturation of the request path must stay typed: no bare 500, no unlabelled body."""
    app = create_app(Settings())
    app.state.vector_db = StubVectorDB()
    statuses: list[int] = []

    with TestClient(app, raise_server_exceptions=False) as client:

        def fire(index: int) -> tuple[int, str, dict[str, Any]]:
            response = client.post("/v1/query", json={"query": f"allowance {index}"})
            body = response.json() if response.content else {}
            return response.status_code, response.headers.get("content-type", ""), body

        with ThreadPoolExecutor(max_workers=STORM_CLIENTS) as pool:
            results = list(pool.map(fire, range(STORM_REQUESTS)))

    for status, content_type, body in results:
        statuses.append(status)
        assert status < 500, f"the storm produced a {status}: {body}"
        if status >= 400:
            assert PROBLEM_MEDIA_TYPE in content_type, content_type
            assert body.get("code"), body

    assert len(statuses) == STORM_REQUESTS
    assert any(status < 400 for status in statuses), "every request failed; nothing was served"
