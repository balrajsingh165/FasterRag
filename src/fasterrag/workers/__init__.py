"""Parallel execution: the CPU pool, the embedding pool, bounded queues, and the indexer.

The pipeline is decoupled into two pools joined by a bounded queue. Because the stages
communicate only through that queue, a failed embedding batch retries without re-parsing
its document, a crashed CPU worker costs only its in-flight item, and a slow provider slows
ingestion instead of exhausting memory (``docs/architecture.md`` §2).

Ingestion and query paths use separate pools and queues. That bulkhead is structural, not
configurable: an ingestion storm must never starve live queries.
"""

from fasterrag.workers.cpu_pool import CpuWorkerPool, PoolReport, resolve_pool_size
from fasterrag.workers.embed_pool import ChunkSink, EmbeddingWorkerPool, EmbedReport
from fasterrag.workers.queues import (
    BoundedQueue,
    ChunkPayload,
    DocumentTask,
    EmbeddedBatch,
    ParseOutcome,
)

__all__ = [
    "BoundedQueue",
    "ChunkPayload",
    "ChunkSink",
    "CpuWorkerPool",
    "DocumentTask",
    "EmbedReport",
    "EmbeddedBatch",
    "EmbeddingWorkerPool",
    "ParseOutcome",
    "PoolReport",
    "resolve_pool_size",
]
