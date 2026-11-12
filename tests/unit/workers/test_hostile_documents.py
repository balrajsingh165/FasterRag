"""Hostile and unusual documents cost one dead-letter entry, with a reason code, never the job.

The parser tests in ``tests/unit/core/parsing/`` prove each refusal is typed; these prove
the pool turns it into a dead-letter entry whose ``reason_code`` names the cause, and that
the document beside it is still indexed (TASK-0472, TASK-0489, TASK-0504).
"""

from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.services.journal import Journal
from fasterrag.workers.cpu_pool import CpuWorkerPool
from fasterrag.workers.queues import BoundedQueue, ChunkPayload
from tests.unit.core.parsing.conftest import encrypted_pdf_bytes

GOOD = "# Good\n\n" + "sentence body text. " * 30


def settings(**overrides: Any) -> Settings:
    payload: dict[str, Any] = {"chunking": {"chunk_size": 64, "overlap": 8}}
    payload.update(overrides)
    return Settings.model_validate(payload)


def threads(workers: int) -> Executor:
    """Run parse tasks in threads so tests stay fast and deterministic."""
    return ThreadPoolExecutor(max_workers=workers)


@pytest.fixture
def journal(tmp_path: Path) -> Journal:
    return Journal(tmp_path / "journal", checkpoint_every=2)


async def ingest(
    sources: list[str], journal: Journal, configured: Settings | None = None
) -> tuple[int, int, dict[str, str]]:
    """Run the sources through the pool; return parsed, dead-lettered and reason by source."""
    async with CpuWorkerPool(
        configured or settings(), journal=journal, executor_factory=threads
    ) as pool:
        report = await pool.process(
            CpuWorkerPool.tasks_for(sources), BoundedQueue[ChunkPayload](400), job="job_1"
        )

    reasons = {
        Path(entry.source).name: str(entry.reason_code)
        for entry in journal.dead_lettered_blocking("job_1")
    }
    return report.parsed, report.dead_lettered, reasons


async def test_an_encrypted_pdf_is_dead_lettered_as_pdf_encrypted(
    tmp_path: Path, journal: Journal
) -> None:
    """It used to arrive as ``PARSE_FAILED``, indistinguishable from a corrupt file."""
    (tmp_path / "locked.pdf").write_bytes(encrypted_pdf_bytes(user_password="secret"))
    (tmp_path / "good.md").write_text(GOOD, encoding="utf-8")

    parsed, dead, reasons = await ingest(
        [str(tmp_path / "locked.pdf"), str(tmp_path / "good.md")], journal
    )

    assert (parsed, dead) == (1, 1)
    assert reasons == {"locked.pdf": "PDF_ENCRYPTED"}


async def test_a_binary_file_named_as_text_is_dead_lettered_not_indexed(
    tmp_path: Path, journal: Journal
) -> None:
    """It used to be indexed as a block of replacement characters."""
    (tmp_path / "image.txt").write_bytes(bytes(range(0x80, 0x100)) * 8)
    (tmp_path / "good.md").write_text(GOOD, encoding="utf-8")

    parsed, dead, reasons = await ingest(
        [str(tmp_path / "image.txt"), str(tmp_path / "good.md")], journal
    )

    assert (parsed, dead) == (1, 1)
    assert reasons == {"image.txt": "ENCODING_UNREADABLE"}


async def test_a_latin_1_file_is_indexed_and_flagged(tmp_path: Path, journal: Journal) -> None:
    """The control case: a legacy encoding is a guess to flag, not a document to refuse."""
    (tmp_path / "legacy.txt").write_bytes(("Caf\xe9 cr\xe8me. " * 40).encode("latin-1"))

    parsed, dead, _ = await ingest([str(tmp_path / "legacy.txt")], journal)

    assert (parsed, dead) == (1, 0)
