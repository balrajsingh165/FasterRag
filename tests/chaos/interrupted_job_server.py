"""A real API process for the kill-and-restart drill in ``test_process_kill.py``.

Not a test module. It serves the real application over a real socket, with the real
ingestion service and journal, and replaces only what a laptop cannot supply: the vector
database is an in-memory adapter and the embedding model is a stub. In ``hang`` mode the
stub stops forever at the document containing :data:`HANG_MARKER`, which leaves a job
``running`` with a checkpoint behind it and nothing in flight — the state a process is in
when it is killed mid-corpus.

The journal lands under the working directory, so the drill starts this process with a
temporary directory as its ``cwd`` and reads the same journal afterwards.

Usage: ``python interrupted_job_server.py <hang|run> <port-file> <hung-file>``
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest import mock

import uvicorn
from tests.unit.services.test_ingestion import KeywordLegAdapter, StubEmbedder

from fasterrag.adapters.embeddings.base import EmbeddingResult
from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.api import ingest as ingest_router
from fasterrag.api.main import create_app
from fasterrag.config.schema import Settings
from fasterrag.services.ingestion import IngestionService

HANG_MARKER = "charlie"
STALE_JOB_SECONDS = 1


class HangingEmbedder(StubEmbedder):
    """Embeds normally, or never returns from the document carrying the marker."""

    def __init__(self, settings: Settings, hung_file: Path | None) -> None:
        """Hang at the marker when ``hung_file`` is given, announcing it there first."""
        super().__init__(settings)
        self.hung_file = hung_file

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        """Embed ``texts``, stopping for good at the marked document in hang mode."""
        if self.hung_file is not None and any(HANG_MARKER in text for text in texts):
            self.hung_file.write_text("hung", encoding="utf-8")
            await asyncio.Event().wait()
        return await super().embed_documents(texts)


def _threads(workers: int) -> Executor:
    """Parse in threads, so killing this process leaves no worker process behind."""
    return ThreadPoolExecutor(max_workers=workers)


def _settings() -> Settings:
    """Return settings that checkpoint every document and call a job stale after a second."""
    return Settings.model_validate(
        {
            "embeddings": {"batch_size": 1},
            "workers": {"embedding_pool_size": 1},
            "chunking": {"chunk_size": 128, "overlap": 16},
            "ingestion": {"journal": {"checkpoint_every": 1}},
            "reliability": {
                "retries": {"max_attempts": 0, "backoff_base_ms": 1, "jitter": False},
                "stale_job_seconds": STALE_JOB_SECONDS,
                "shutdown_grace_seconds": 0,
            },
        }
    )


async def _serve(mode: str, port_file: Path, hung_file: Path) -> None:
    """Serve the application on a free port and publish the port once it is listening."""
    settings = _settings()
    embedder = HangingEmbedder(settings, hung_file if mode == "hang" else None)
    router = TieringRouter(embedder)

    def build_ingestion(
        settings: Settings, adapter: Any, journal: Any, cache: Any, **kwargs: Any
    ) -> IngestionService:
        return IngestionService(
            settings, journal=journal, adapter=adapter, router=router, executor_factory=_threads
        )

    mock.patch.object(ingest_router, "build_ingestion", build_ingestion).start()
    mock.patch.object(ingest_router, "build_embedding_router", return_value=router).start()

    app = create_app(settings)
    app.state.vector_db = KeywordLegAdapter(settings)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    serving = asyncio.ensure_future(server.serve())
    while not server.started:
        if serving.done():
            await serving
        await asyncio.sleep(0.02)

    port = server.servers[0].sockets[0].getsockname()[1]
    pending = port_file.with_suffix(".tmp")
    pending.write_text(str(port), encoding="utf-8")
    pending.replace(port_file)
    await serving


if __name__ == "__main__":
    asyncio.run(_serve(sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])))
