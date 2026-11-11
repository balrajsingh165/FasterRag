"""The repeatable unit of work a soak run performs, and the resource sampler watching it.

The workload is the **real ingestion pipeline** — source resolution, the CPU pool, the
bounded queue, the embedding pool, the indexer, journal checkpointing — driven in-process
over a generated corpus. Only the two vendor edges are doubles, and each is a double for a
stated reason rather than for convenience.

**The vector database discards.** A recording double that keeps every point it is handed
makes RSS rise linearly with the corpus by construction, which would turn a leak detector
into a corpus-size detector and fail every run that worked. ``DiscardingAdapter`` counts
and forgets, so anything left in the series is fasterRag's own retention. A real backend
would be better still and is what TASK-0084's reference rig is for; it is not available
here and a container's own memory is not this process's RSS anyway.

**The embedder is deterministic and stateless**, for the same reason and one more: a real
model holds hundreds of megabytes whose allocator behaviour would dominate every sample.

What this costs, stated rather than hidden: a soak over doubles cannot see a leak that
lives in a vendor client, and the two leaks most worth finding — an HTTP connection pool
that never recycles, and a model reloaded per batch (FMEA row 8) — live exactly there. A
run on the reference rig against real Qdrant and a real embedding provider is the one that
closes that, and this harness is what will drive it.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections.abc import AsyncIterator, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import psutil

from fasterrag.adapters.embeddings.base import EmbeddingAdapter, EmbeddingResult
from fasterrag.adapters.embeddings.tiering import TieringRouter
from fasterrag.adapters.vectordb.base import (
    CollectionInfo,
    CollectionSpec,
    HealthStatus,
    Point,
    PointSelector,
    PointUpdate,
    ScoredPoint,
    SearchQuery,
    UpsertResult,
    VectorDBAdapter,
)
from fasterrag.config.schema import Settings
from fasterrag.services.ingestion import IngestionService
from fasterrag.services.journal import Journal

DIMENSIONS: Final = 8
DOCUMENTS_PER_CYCLE: Final = 6
SECONDS_VARIABLE: Final = "FASTERRAG_SOAK_SECONDS"
INTERVAL_VARIABLE: Final = "FASTERRAG_SOAK_SAMPLE_INTERVAL_S"
DEFAULT_SECONDS: Final = 120.0
DEFAULT_INTERVAL_S: Final = 1.0

VERDICTS: Final[list[str]] = []
"""Each finished run's verdict lines, printed by this package's terminal summary."""


@dataclass(frozen=True, slots=True)
class Sample:
    """One reading of the resources a leak would show up in."""

    at_s: float
    rss_bytes: float
    handles: float
    cycles: int


def open_handles(process: psutil.Process) -> float:
    """Return the process's open handle count, under whichever name the platform uses.

    Windows has no file descriptors; the equivalent counter is the handle table, which
    covers files, sockets, events and threads, so it is if anything the stricter series.
    """
    counter = getattr(process, "num_handles", None) or getattr(process, "num_fds", None)
    if counter is None:
        raise RuntimeError("psutil exposes neither num_handles nor num_fds on this platform")
    return float(counter())


def sample(process: psutil.Process, *, started: float, cycles: int) -> Sample:
    """Return one reading of RSS, handles and work completed so far."""
    return Sample(
        at_s=time.monotonic() - started,
        rss_bytes=float(process.memory_info().rss),
        handles=open_handles(process),
        cycles=cycles,
    )


def duration_s() -> float:
    """Return the configured run length; the 24 h run is this value and nothing else."""
    return float(os.environ.get(SECONDS_VARIABLE, DEFAULT_SECONDS))


def interval_s() -> float:
    """Return the configured sampling interval."""
    return float(os.environ.get(INTERVAL_VARIABLE, DEFAULT_INTERVAL_S))


class DiscardingAdapter(VectorDBAdapter):
    """Accepts every write and keeps none of it, so retention in the series is fasterRag's."""

    def __init__(self, settings: Settings) -> None:
        """Build an adapter that counts rather than stores."""
        super().__init__(settings)
        self.upserted = 0
        self.collections: set[str] = set()

    async def create_collection(self, spec: CollectionSpec) -> None:
        self.collections.add(spec.name)

    async def list_collections(self) -> list[CollectionInfo]:
        return [
            CollectionInfo(name=name, vectors=self.upserted, dimensions=DIMENSIONS, sparse=True)
            for name in sorted(self.collections)
        ]

    async def drop_collection(self, name: str) -> bool:
        existed = name in self.collections
        self.collections.discard(name)
        return existed

    async def snapshot(self, collection: str) -> str:
        return f"{collection}-snap"

    async def list_snapshots(self, collection: str) -> list[str]:
        return []

    async def delete_snapshot(self, collection: str, snapshot: str) -> bool:
        return True

    async def restore_snapshot(self, collection: str, snapshot: str) -> None:
        return None

    async def set_alias(self, alias: str, collection: str) -> None:
        return None

    async def alias_target(self, alias: str) -> str | None:
        return None

    async def delete_alias(self, alias: str) -> bool:
        return False

    async def upsert(self, points: list[Point]) -> UpsertResult:
        self.upserted += len(points)
        return UpsertResult(upserted=len(points))

    async def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        """Yield nothing; this adapter keeps no points to scan."""
        empty: list[Point] = []
        for point in empty:
            yield point

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        return []

    async def update(self, updates: list[PointUpdate]) -> None:
        return None

    async def delete(self, selector: PointSelector) -> None:
        return None

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True, detail="discarding", latency_ms=0.0)

    async def close(self) -> None:
        return None


class HashEmbedder(EmbeddingAdapter):
    """Turns text into a deterministic vector, holding no model and no history."""

    provider = "soak-hash"

    @property
    def model(self) -> str:
        return "soak-hash"

    @property
    def model_version(self) -> str:
        return "soak-hash-v1"

    @property
    def dimensions(self) -> int:
        return DIMENSIONS

    async def embed_documents(self, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(
            vectors=[self._vector(text) for text in texts],
            model=self.model,
            model_version=self.model_version,
        )

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True, detail="hash", latency_ms=0.0)

    async def close(self) -> None:
        return None

    @staticmethod
    def _vector(text: str) -> list[float]:
        """Return a unit-ish vector derived from the text's digest."""
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=DIMENSIONS).digest()
        return [byte / 255.0 for byte in digest]


def soak_settings(**overrides: Any) -> Settings:
    """Return settings sized so one cycle is short and the queue is small enough to fill."""
    payload: dict[str, Any] = {
        "embeddings": {"batch_size": 8},
        "workers": {"embedding_pool_size": 1, "queue_depth": 32},
        "chunking": {"chunk_size": 128, "overlap": 16},
        "reliability": {"retries": {"max_attempts": 0, "backoff_base_ms": 1, "jitter": False}},
    }
    payload.update(overrides)
    return Settings.model_validate(payload)


def write_corpus(root: Path, documents: int = DOCUMENTS_PER_CYCLE) -> list[str]:
    """Write a small corpus and return its paths as ingestion sources."""
    root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index in range(documents):
        path = root / f"policy-{index}.md"
        path.write_text(
            f"# Policy {index}\n\n"
            f"Clause {index} sets the allowance at {index * 7} pounds per day.\n\n"
            f"Clause {index} b states that approval is required above {index * 70} pounds.\n",
            encoding="utf-8",
        )
        paths.append(str(path))
    return paths


def threads(workers: int) -> Executor:
    """Run the CPU pool in threads, so a soak spawns no process per cycle."""
    return ThreadPoolExecutor(max_workers=workers)


def build_service(journal_root: Path, settings: Settings | None = None) -> IngestionService:
    """Return an ingestion service over the discarding backend and the hash embedder."""
    resolved = settings or soak_settings()
    return IngestionService(
        resolved,
        journal=Journal(journal_root, checkpoint_every=1),
        adapter=DiscardingAdapter(resolved),
        router=TieringRouter(HashEmbedder(resolved)),
        executor_factory=threads,
    )
