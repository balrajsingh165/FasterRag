"""Hostile and unusual documents cost one dead-letter entry, with a reason code, never the job.

The parser tests in ``tests/unit/core/parsing/`` prove each refusal is typed; these prove
the pool turns it into a dead-letter entry whose ``reason_code`` names the cause, and that
the document beside it is still indexed (TASK-0472, TASK-0489, TASK-0504).
"""

import sys
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Final

import pytest

from fasterrag.cli.sources import expand_sources
from fasterrag.config.schema import Settings
from fasterrag.services.journal import Journal
from fasterrag.services.sources import resolve_sources, typed_source
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


async def test_a_ten_thousand_deep_json_document_is_dead_lettered_as_nesting_too_deep(
    tmp_path: Path, journal: Journal
) -> None:
    """The TASK-0489 acceptance case: one reason code, and the document beside it indexed."""
    depth = 10_000
    (tmp_path / "deep.json").write_bytes(b"[" * depth + b"1" + b"]" * depth)
    (tmp_path / "good.md").write_text(GOOD, encoding="utf-8")

    parsed, dead, reasons = await ingest(
        [str(tmp_path / "deep.json"), str(tmp_path / "good.md")], journal
    )

    assert (parsed, dead) == (1, 1)
    assert reasons == {"deep.json": "NESTING_TOO_DEEP"}


_LINKS_ARE_PRIVILEGED: Final = sys.platform == "win32"
"""Whether refusing to create a symlink is a privilege the account lacks rather than a defect."""


def link(target: Path, name: Path) -> None:
    """Create a symlink, skipping only where Windows withholds the privilege (TASK-0328)."""
    try:
        name.symlink_to(target)
    except (NotImplementedError, OSError):
        if not _LINKS_ARE_PRIVILEGED:
            raise
        pytest.skip(
            "this Windows account lacks SeCreateSymbolicLinkPrivilege; the Ubuntu CI lane "
            "creates the link and runs this case"
        )


def confined(root: Path) -> Settings:
    """A served deployment that may read only ``root``."""
    return settings(security={"auth": True}, ingestion={"allowed_source_roots": [str(root)]})


async def run_directory(
    root: Path, journal: Journal, configured: Settings
) -> tuple[int, int, dict[str, str], str]:
    """Expand, resolve and parse a directory as ``fasterrag ingest <dir>`` would."""
    resolved = await resolve_sources(
        [typed_source(source) for source in expand_sources([str(root)])], configured
    )
    queue = BoundedQueue[ChunkPayload](400)
    async with CpuWorkerPool(configured, journal=journal, executor_factory=threads) as pool:
        report = await pool.process(CpuWorkerPool.tasks_for(resolved.sources), queue, job="job_1")

    reasons = {
        Path(entry.source).name: str(entry.reason_code)
        for entry in journal.dead_lettered_blocking("job_1")
    }
    indexed = " ".join(payload.chunk.text for payload in await queue.drain())
    return report.parsed, report.dead_lettered, reasons, indexed


@pytest.mark.must_run
async def test_an_escaping_link_in_a_directory_is_one_dead_lettered_document(
    tmp_path: Path, journal: Journal
) -> None:
    """The escape refused the whole job, so one planted link stopped every good document."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "good.md").write_text(GOOD, encoding="utf-8")
    (tmp_path / "secret.env").write_text("TOKEN=hunter2", encoding="utf-8")
    link(tmp_path / "secret.env", corpus / "innocent.md")

    parsed, dead, reasons, indexed = await run_directory(corpus, journal, confined(corpus))

    assert (parsed, dead) == (1, 1)
    assert reasons == {"innocent.md": "SOURCE_LINK_ESCAPES"}
    assert "hunter2" not in indexed


@pytest.mark.must_run
async def test_a_link_loop_in_a_directory_is_dead_lettered_and_the_job_continues(
    tmp_path: Path, journal: Journal
) -> None:
    """A loop was dropped from expansion unseen, and on 3.12 resolving it raised RuntimeError."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "good.md").write_text(GOOD, encoding="utf-8")
    link(corpus / "loop.md", corpus / "loop.md")

    parsed, dead, reasons, _ = await run_directory(corpus, journal, confined(corpus))

    assert (parsed, dead) == (1, 1)
    assert reasons == {"loop.md": "PARSE_FAILED"}
