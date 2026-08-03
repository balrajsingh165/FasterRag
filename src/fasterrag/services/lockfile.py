"""The index lockfile and drift detection (D1).

An index is a build artifact, and like any build artifact it should be reproducible. The
lockfile records what produced it — the retrieval-affecting config, the embedding model and
its version, the chunker and its parameters, and a content hash per indexed document — so
that "is this index still the one my config describes?" has an answer rather than requiring
archaeology.

**Drift is reported, never silent.** Every other framework will happily let you change
``embeddings.model`` and keep serving vectors produced by the old one, mixing two embedding
spaces in one collection where similarity between them is meaningless. Here that state is
detectable and named: verify exits non-zero and says which field moved.

The lockfile is written atomically, because a half-written lockfile is worse than none — it
would report drift against a truncated record and send someone chasing a change nobody made.
"""

from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from fasterrag import __version__
from fasterrag.config.schema import Settings
from fasterrag.core.identity import (
    IDENTITY_VERSION,
    chunker_config_hash,
    retrieval_config_hash,
)
from fasterrag.observability.logging import get_logger

__all__ = [
    "DEFAULT_LOCK_ROOT",
    "LOCK_VERSION",
    "DriftReport",
    "IndexLock",
    "LockStore",
    "build_lock",
    "create_lock_store",
    "detect_drift",
]

LOCK_VERSION: Final = "1.0.0"

DEFAULT_LOCK_ROOT: Final = Path(".fasterrag") / "locks"

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class IndexLock:
    """The reproducibility contract of one collection's index."""

    collection: str
    config_hash: str
    embedding_model: str
    embedding_model_version: str
    dimensions: int | None
    chunker_strategy: str
    chunker_version: str
    chunk_size: int
    overlap: int
    contextual_enrichment: bool
    identity_version: int = IDENTITY_VERSION
    document_hashes: dict[str, str] = field(default_factory=dict)
    built_at: str = ""
    built_by: str = ""
    lock_version: str = LOCK_VERSION

    def as_dict(self) -> dict[str, Any]:
        """Return the persisted form."""
        return {
            "lock_version": self.lock_version,
            "identity_version": self.identity_version,
            "collection": self.collection,
            "config_hash": self.config_hash,
            "embedding_model": self.embedding_model,
            "embedding_model_version": self.embedding_model_version,
            "dimensions": self.dimensions,
            "chunker_strategy": self.chunker_strategy,
            "chunker_version": self.chunker_version,
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "contextual_enrichment": self.contextual_enrichment,
            "document_hashes": self.document_hashes,
            "built_at": self.built_at,
            "built_by": self.built_by,
        }

    def summary(self) -> dict[str, Any]:
        """Return the lock without its per-document hashes.

        What ``GET /v1/collections/{name}`` embeds. A million-document corpus has a million
        hashes, and returning them from a status endpoint would make an inspection call
        heavier than the query it is inspecting.
        """
        payload = self.as_dict()
        payload["documents"] = len(self.document_hashes)
        del payload["document_hashes"]
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> IndexLock:
        """Rebuild a lock from its persisted form."""
        return cls(
            collection=str(payload["collection"]),
            config_hash=str(payload["config_hash"]),
            embedding_model=str(payload.get("embedding_model", "")),
            embedding_model_version=str(payload.get("embedding_model_version", "")),
            dimensions=payload.get("dimensions"),
            chunker_strategy=str(payload.get("chunker_strategy", "")),
            chunker_version=str(payload.get("chunker_version", "")),
            chunk_size=int(payload.get("chunk_size", 0)),
            overlap=int(payload.get("overlap", 0)),
            contextual_enrichment=bool(payload.get("contextual_enrichment", False)),
            document_hashes=dict(payload.get("document_hashes") or {}),
            built_at=str(payload.get("built_at", "")),
            built_by=str(payload.get("built_by", "")),
            lock_version=str(payload.get("lock_version", LOCK_VERSION)),
            # CRITICAL: defaults to 1, not to the current version. A lockfile written before
            # this field existed was built under the original id scheme, and defaulting to
            # "whatever is current" would silently declare it up to date — hiding exactly the
            # mismatch this field was added to surface.
            identity_version=int(payload.get("identity_version", 1)),
        )


@dataclass(frozen=True, slots=True)
class DriftReport:
    """What, if anything, no longer matches the lockfile."""

    collection: str
    fields: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)
    documents_added: list[str] = field(default_factory=list)
    documents_removed: list[str] = field(default_factory=list)
    documents_changed: list[str] = field(default_factory=list)
    missing_lock: bool = False

    @property
    def detected(self) -> bool:
        """Return whether anything drifted."""
        return bool(
            self.fields or self.documents_added or self.documents_removed or self.documents_changed
        )

    def as_dict(self) -> dict[str, Any]:
        """Return the form ``index list`` and the collections endpoint report."""
        return {
            "collection": self.collection,
            "detected": self.detected,
            "missing_lock": self.missing_lock,
            "fields": self.fields,
            "details": self.details,
            "documents": {
                "added": self.documents_added,
                "removed": self.documents_removed,
                "changed": self.documents_changed,
            },
        }


def build_lock(
    collection: str,
    settings: Settings,
    *,
    embedding_model: str,
    embedding_model_version: str,
    dimensions: int | None,
    document_hashes: dict[str, str] | None = None,
) -> IndexLock:
    """Capture what produced an index, ready to be written.

    The model *version* is recorded alongside the name because a provider can change the
    weights behind a stable name; without the version, that change is undetectable and every
    vector written afterwards silently belongs to a different embedding space.
    """
    chunking = settings.chunking
    return IndexLock(
        collection=collection,
        config_hash=retrieval_config_hash(settings),
        embedding_model=embedding_model,
        embedding_model_version=embedding_model_version,
        dimensions=dimensions,
        chunker_strategy=chunking.strategy,
        chunker_version=chunker_config_hash(settings),
        chunk_size=chunking.chunk_size,
        overlap=chunking.overlap,
        contextual_enrichment=chunking.contextual_enrichment,
        document_hashes=dict(document_hashes or {}),
        built_at=datetime.now(tz=UTC).isoformat(),
        built_by=__version__,
    )


def _compare(
    name: str, locked: Any, live: Any, fields: list[str], details: list[dict[str, Any]]
) -> None:
    """Record a field as drifted when the live value no longer matches the lock."""
    if locked != live:
        fields.append(name)
        details.append({"field": name, "locked": locked, "live": live})


def detect_drift(
    lock: IndexLock | None,
    settings: Settings,
    *,
    collection: str,
    embedding_model: str | None = None,
    embedding_model_version: str | None = None,
    document_hashes: dict[str, str] | None = None,
) -> DriftReport:
    """Compare a lockfile against live configuration and corpus.

    Args:
        lock: The stored lockfile, or ``None`` when none was written.
        settings: The live configuration.
        collection: The collection being checked.
        embedding_model: The live model name, when it is known without loading a model.
        embedding_model_version: The live model version.
        document_hashes: Live ``document_id → content_hash`` for the collection.

    Returns:
        The drift report. A missing lockfile is reported as ``missing_lock`` rather than as
        drift: nothing has changed, there is simply nothing to compare against, and calling
        that "drift" would make an un-locked index indistinguishable from a corrupted one.
    """
    if lock is None:
        return DriftReport(collection=collection, missing_lock=True)

    fields: list[str] = []
    details: list[dict[str, Any]] = []

    # Checked first and reported like any other drift, because an id-scheme change makes
    # every document in the collection unaddressable at once. Without it the symptom is a
    # recall of 0.0 that looks like a broken retriever rather than a renamed corpus.
    _compare("identity_version", lock.identity_version, IDENTITY_VERSION, fields, details)
    _compare("config_hash", lock.config_hash, retrieval_config_hash(settings), fields, details)
    _compare("chunker_strategy", lock.chunker_strategy, settings.chunking.strategy, fields, details)
    _compare("chunk_size", lock.chunk_size, settings.chunking.chunk_size, fields, details)
    _compare("overlap", lock.overlap, settings.chunking.overlap, fields, details)
    _compare(
        "contextual_enrichment",
        lock.contextual_enrichment,
        settings.chunking.contextual_enrichment,
        fields,
        details,
    )

    if embedding_model is not None:
        _compare("embedding_model", lock.embedding_model, embedding_model, fields, details)
    if embedding_model_version is not None:
        _compare(
            "embedding_model_version",
            lock.embedding_model_version,
            embedding_model_version,
            fields,
            details,
        )

    added: list[str] = []
    removed: list[str] = []
    changed: list[str] = []
    if document_hashes is not None:
        locked_documents = lock.document_hashes
        added = sorted(set(document_hashes) - set(locked_documents))
        removed = sorted(set(locked_documents) - set(document_hashes))
        changed = sorted(
            document
            for document, digest in document_hashes.items()
            if document in locked_documents and locked_documents[document] != digest
        )

    report = DriftReport(
        collection=collection,
        fields=fields,
        details=details,
        documents_added=added,
        documents_removed=removed,
        documents_changed=changed,
    )

    if report.detected:
        _logger.warning(
            "index drift detected",
            extra={
                "collection": collection,
                "fields": fields,
                "documents_added": len(added),
                "documents_removed": len(removed),
                "documents_changed": len(changed),
            },
        )
    return report


class LockStore:
    """Reads and writes ``index.lock`` files, one per collection."""

    def __init__(self, root: Path | None = None, *, enabled: bool = True) -> None:
        """Build a store rooted at ``root``, defaulting to ``.fasterrag/locks``."""
        self.root = root or DEFAULT_LOCK_ROOT
        self.enabled = enabled

    def _path(self, collection: str) -> Path:
        """Return the lockfile path for a collection."""
        return self.root / f"{collection}.lock.json"

    def write(self, lock: IndexLock) -> None:
        """Persist a lockfile atomically.

        Raises:
            OSError: Never — a failure is logged. An index build that succeeded must not be
                reported as failed because its lockfile could not be written; the drift
                check will report the missing lock, which is the accurate description.
        """
        if not self.enabled:
            return

        path = self._path(lock.collection)
        temporary = path.with_suffix(".tmp")
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            temporary.write_text(json.dumps(lock.as_dict(), indent=2), encoding="utf-8")
            os.replace(temporary, path)
        except OSError as exc:
            # CRITICAL: the cleanup is itself suppressed. On Linux, when the parent is a
            # file rather than a directory, `unlink` raises NotADirectoryError from inside
            # this handler — replacing the error being handled with a second one that
            # escapes, so a degradation path becomes a crash. Windows does not raise here,
            # which is why this only ever failed on the Linux CI leg.
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
            _logger.warning(
                "could not write the index lockfile; the index itself is unaffected",
                extra={"collection": lock.collection, "error": str(exc)},
            )

    def read(self, collection: str) -> IndexLock | None:
        """Return a collection's lockfile, or ``None`` when absent or unreadable."""
        try:
            payload = json.loads(self._path(collection).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        if not isinstance(payload, dict):
            return None

        try:
            return IndexLock.from_dict(payload)
        except (KeyError, TypeError, ValueError):
            _logger.warning("the index lockfile is malformed", extra={"collection": collection})
            return None

    def delete(self, collection: str) -> bool:
        """Remove a collection's lockfile, reporting whether one was there."""
        path = self._path(collection)
        existed = path.exists()
        path.unlink(missing_ok=True)
        return existed


def create_lock_store(settings: Settings, root: Path | None = None) -> LockStore:
    """Build a lock store from validated configuration."""
    return LockStore(root, enabled=settings.index.lockfile)
