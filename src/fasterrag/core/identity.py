"""Deterministic identifiers and content hashes.

The ID scheme of ``docs/data-model.md``. Two of these are deterministic on purpose, and
the guarantees they carry are what make the pipeline replay-safe:

* A **document id** derives from its source URI and tenant, so re-ingesting the same file
  addresses the same document instead of creating a duplicate.
* A **chunk id** derives from the document id, the chunk index, and a hash of the chunking
  configuration. Re-running an identical ingest therefore produces identical ids and the
  index upserts over itself — exactly-once effects from an at-least-once pipeline (D3).
  Changing the chunker changes the ids, which is what stops old and new chunks from being
  silently mixed in one collection.

Job and trace ids are random but time-ordered, so listing them sorts chronologically.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from secrets import token_hex
from typing import Any, Final

from fasterrag.config.schema import Settings

__all__ = [
    "chunk_id",
    "chunker_config_hash",
    "collection_id",
    "content_hash",
    "document_id",
    "job_id",
    "retrieval_config_hash",
    "text_hash",
]

_DOCUMENT_PREFIX: Final = "d_"
_CHUNK_PREFIX: Final = "c_"
_JOB_PREFIX: Final = "job_"
_COLLECTION_PREFIX: Final = "col_"

_ID_LENGTH: Final = 16
_RANDOM_BYTES: Final = 5

_SEQUENCE_LOCK: Final = threading.Lock()
_last_millis = 0
_sequence = 0


def _digest(*parts: str) -> str:
    """Return a stable hex digest over the parts, separated so they cannot collide."""
    joined = "\x00".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def content_hash(data: bytes) -> str:
    """Return the SHA-256 hex digest of raw bytes.

    Drives deduplication and lockfile drift detection, so it hashes the bytes as ingested
    rather than anything normalized.
    """
    return hashlib.sha256(data).hexdigest()


def text_hash(text: str) -> str:
    """Return the SHA-256 hex digest of a string, encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def document_id(source_uri: str, tenant: str | None = None) -> str:
    """Return the deterministic id for a source document."""
    return f"{_DOCUMENT_PREFIX}{_digest(source_uri, tenant or '')[:_ID_LENGTH]}"


def chunk_id(document: str, chunk_index: int, chunker_hash: str) -> str:
    """Return the deterministic id for a chunk.

    A pure function of its three inputs: the same document chunked the same way always
    yields the same ids, which is what makes indexing idempotent.
    """
    return f"{_CHUNK_PREFIX}{_digest(document, str(chunk_index), chunker_hash)[:_ID_LENGTH]}"


def job_id() -> str:
    """Return a fresh job id that sorts chronologically.

    A timestamp alone is not enough to order ids: several jobs can be created inside one
    clock tick, and on Windows that tick can be tens of milliseconds. A per-process
    sequence number breaks those ties, so ids created by one process always sort in
    creation order. Across processes, ordering falls back to the timestamp.
    """
    with _SEQUENCE_LOCK:
        global _last_millis, _sequence
        millis = int(time.time() * 1000)
        if millis <= _last_millis:
            _sequence += 1
        else:
            _last_millis = millis
            _sequence = 0
        stamp, tie_break = _last_millis, _sequence

    return f"{_JOB_PREFIX}{stamp:013d}{tie_break:04d}{token_hex(_RANDOM_BYTES)}"


def collection_id() -> str:
    """Return a fresh collection id; the human-facing key remains the name."""
    return f"{_COLLECTION_PREFIX}{token_hex(_RANDOM_BYTES)}"


def _stable(payload: dict[str, Any]) -> str:
    """Serialize a mapping so equal content always hashes equally."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def chunker_config_hash(settings: Settings) -> str:
    """Return a hash of the settings that change chunk boundaries.

    Only boundary-affecting keys are included. Contextual enrichment is *not* one of them:
    it changes a chunk's embedded text but not where the chunk starts and ends, so
    toggling it must not renumber every chunk in the corpus.
    """
    chunking = settings.chunking
    return _digest(
        _stable(
            {
                "strategy": chunking.strategy,
                "chunk_size": chunking.chunk_size,
                "overlap": chunking.overlap,
            }
        )
    )


def retrieval_config_hash(settings: Settings) -> str:
    """Return a hash of the retrieval-affecting configuration subset.

    The value recorded in ``index.lock`` (D1). Deliberately *not* a hash of the whole
    file: editing an unrelated key such as ``app.port`` must not report index drift.
    """
    return _digest(
        _stable(
            {
                "chunking": settings.chunking.model_dump(mode="json"),
                "embeddings": {
                    "provider": settings.embeddings.provider,
                    "model": settings.embeddings.model,
                    "dimensions": settings.embeddings.dimensions,
                    "tiering": settings.embeddings.tiering.model_dump(mode="json"),
                },
                "retrieval": settings.retrieval.model_dump(mode="json"),
            }
        )
    )
