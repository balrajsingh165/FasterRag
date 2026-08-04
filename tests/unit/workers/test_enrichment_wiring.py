from collections.abc import Sequence
from concurrent.futures import Executor, Future
from pathlib import Path
from typing import Any

from fasterrag.adapters.llm.base import Completion
from fasterrag.config.schema import Settings
from fasterrag.workers.cpu_pool import CpuWorkerPool
from fasterrag.workers.queues import BoundedQueue, ChunkPayload


class Inline(Executor):
    """Runs work in this process, so a test never spawns a worker."""

    def submit(self, fn: Any, /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        future.set_result(fn(*args, **kwargs))
        return future


class StubLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        self.calls += 1
        return Completion(text="From the handbook, on leave.", model="stub")


def corpus(tmp_path: Path) -> str:
    source = tmp_path / "leave.md"
    source.write_text("# Leave policy\n\nStaff accrue 1.75 days a month.\n", encoding="utf-8")
    return str(source)


def settings(enabled: bool) -> Settings:
    return Settings.model_validate({"chunking": {"contextual_enrichment": enabled}})


async def drain(queue: BoundedQueue[ChunkPayload], expected: int) -> list[ChunkPayload]:
    items: list[ChunkPayload] = []
    for _ in range(expected):
        item = await queue.get()
        if item is not None:
            items.append(item)
    return items


async def run(tmp_path: Path, enabled: bool, llm: Any) -> Sequence[ChunkPayload]:
    queue: BoundedQueue[ChunkPayload] = BoundedQueue(50)
    async with CpuWorkerPool(
        settings(enabled), executor_factory=lambda _: Inline(), llm=llm
    ) as pool:
        report = await pool.process(CpuWorkerPool.tasks_for([corpus(tmp_path)]), queue, job="job_1")
    return await drain(queue, report.chunked)


async def test_the_toggle_off_leaves_chunks_untouched(tmp_path: Path) -> None:
    llm = StubLLM()

    chunks = await run(tmp_path, enabled=False, llm=llm)

    assert llm.calls == 0
    assert "context_prefix" not in chunks[0].chunk.metadata


async def test_the_toggle_on_prepends_a_context(tmp_path: Path) -> None:
    """The whole point of TASK-0185: the flag must actually change what is indexed."""
    llm = StubLLM()

    chunks = await run(tmp_path, enabled=True, llm=llm)

    assert llm.calls >= 1
    assert chunks[0].chunk.text.startswith("From the handbook, on leave.")
    assert chunks[0].chunk.metadata["context_prefix"] == "From the handbook, on leave."


async def test_the_toggle_cannot_conjure_a_provider(tmp_path: Path) -> None:
    """With no model supplied, the flag is inert rather than an error."""
    chunks = await run(tmp_path, enabled=True, llm=None)

    assert "context_prefix" not in chunks[0].chunk.metadata


async def test_the_parent_text_is_not_shipped_when_enrichment_is_off(tmp_path: Path) -> None:
    """Carrying it always would double what every parsed document sends through IPC."""
    from fasterrag.workers.cpu_pool import parse_and_chunk

    task = CpuWorkerPool.tasks_for([corpus(tmp_path)])[0]

    assert parse_and_chunk(task, settings(False)).document_text == ""
    assert parse_and_chunk(task, settings(True)).document_text != ""
