import pytest

from fasterrag.observability import metrics
from fasterrag.workers.queues import BoundedQueue


def test_a_queue_publishes_its_depth_before_anything_is_enqueued() -> None:
    """A gauge that only appears on first use is absent exactly when nothing is flowing."""
    BoundedQueue[int](4, name="declared")

    assert metrics.QUEUE_DEPTH.value(queue="declared") == 0.0


@pytest.mark.asyncio
async def test_depth_tracks_enqueue_and_dequeue() -> None:
    queue: BoundedQueue[int] = BoundedQueue(4, name="tracked")

    await queue.put(1)
    await queue.put(2)
    assert metrics.QUEUE_DEPTH.value(queue="tracked") == 2.0

    await queue.get()
    assert metrics.QUEUE_DEPTH.value(queue="tracked") == 1.0


@pytest.mark.asyncio
async def test_draining_returns_the_gauge_to_zero() -> None:
    queue: BoundedQueue[int] = BoundedQueue(4, name="drained")
    await queue.put(1)

    await queue.drain()

    assert metrics.QUEUE_DEPTH.value(queue="drained") == 0.0


def test_a_rejected_offer_does_not_inflate_the_depth() -> None:
    """QUEUE_FULL is a rejection, not an enqueue; counting it would overstate the backlog."""
    queue: BoundedQueue[int] = BoundedQueue(1, name="rejecting")
    queue.offer(1)

    with pytest.raises(Exception, match="capacity"):
        queue.offer(2)

    assert metrics.QUEUE_DEPTH.value(queue="rejecting") == 1.0


@pytest.mark.asyncio
async def test_queues_are_published_under_separate_labels() -> None:
    """One number for "the queue depth" cannot say which stage is falling behind."""
    first: BoundedQueue[int] = BoundedQueue(4, name="stage_one")
    BoundedQueue[int](4, name="stage_two")

    await first.put(1)

    assert metrics.QUEUE_DEPTH.value(queue="stage_one") == 1.0
    assert metrics.QUEUE_DEPTH.value(queue="stage_two") == 0.0
