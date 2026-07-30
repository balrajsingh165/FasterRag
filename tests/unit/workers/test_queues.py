import asyncio

import pytest

from fasterrag.core.chunking.models import TextChunk
from fasterrag.errors import ErrorCode, IngestionError
from fasterrag.workers.queues import BoundedQueue, ChunkPayload, EmbeddedBatch


def payload(chunk_id: str = "c_1") -> ChunkPayload:
    return ChunkPayload(
        chunk_id=chunk_id,
        document_id="d_1",
        source="a.md",
        content_hash="hash",
        chunk=TextChunk(
            text="body", start=0, end=4, chunk_index=0, token_count=1, strategy="recursive"
        ),
    )


async def test_capacity_is_reported() -> None:
    queue: BoundedQueue[int] = BoundedQueue(2)

    assert queue.capacity == 2
    assert queue.depth == 0
    assert queue.full is False


async def test_depth_tracks_occupancy() -> None:
    queue: BoundedQueue[int] = BoundedQueue(2)

    await queue.put(1)

    assert queue.depth == 1
    assert queue.full is False

    await queue.put(2)
    assert queue.full is True


async def test_a_producer_waits_when_the_queue_is_full() -> None:
    queue: BoundedQueue[int] = BoundedQueue(1)
    await queue.put(1)

    pending = asyncio.create_task(queue.put(2))
    await asyncio.sleep(0)

    assert pending.done() is False

    await queue.get()
    queue.task_done()
    await asyncio.wait_for(pending, timeout=1)


async def test_offering_to_a_full_queue_is_rejected_not_queued() -> None:
    queue: BoundedQueue[int] = BoundedQueue(1)
    queue.offer(1)

    with pytest.raises(IngestionError, match="at its capacity") as caught:
        queue.offer(2)

    assert caught.value.code is ErrorCode.QUEUE_FULL
    assert caught.value.status == 429
    assert caught.value.retryable is True


async def test_offering_succeeds_while_there_is_room() -> None:
    queue: BoundedQueue[int] = BoundedQueue(2)

    queue.offer(1)
    queue.offer(2)

    assert queue.depth == 2


async def test_closing_sends_one_sentinel_per_consumer() -> None:
    queue: BoundedQueue[int] = BoundedQueue(5)

    await queue.close(consumers=3)

    assert [await queue.get() for _ in range(3)] == [None, None, None]


async def test_draining_returns_items_and_skips_sentinels() -> None:
    queue: BoundedQueue[int] = BoundedQueue(5)
    await queue.put(1)
    await queue.put(2)
    await queue.close(consumers=1)

    assert list(await queue.drain()) == [1, 2]
    assert queue.depth == 0


async def test_a_misaligned_batch_is_refused() -> None:
    with pytest.raises(IngestionError, match="2 chunks but 1 vectors") as caught:
        EmbeddedBatch(
            chunks=[payload("c_1"), payload("c_2")],
            vectors=[[0.1]],
            model="m",
            model_version="v",
        )

    assert caught.value.retryable is False


async def test_an_aligned_batch_is_accepted() -> None:
    batch = EmbeddedBatch(chunks=[payload()], vectors=[[0.1, 0.2]], model="m", model_version="v")

    assert len(batch.chunks) == len(batch.vectors)


def test_a_payload_exposes_the_text_to_embed() -> None:
    assert payload().text == "body"
