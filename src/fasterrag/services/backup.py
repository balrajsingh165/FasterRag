"""Backup and restore of a whole deployment.

The artifact inventory of ``docs/disaster-recovery.md`` §1, in one place, because a backup
that captures the vectors but not the lockfile restores an index nobody can prove anything
about — drift detection has nothing to compare against, and the index stops being a
reproducible build artifact the moment its lockfile is gone.

**``.env`` is never touched.** Secrets belong in the operator's secret manager, and a backup
routine that swept them into a tarball would put every credential wherever backups are
stored. Its absence from a restore is expected and documented, not an omission.

A backup records a *manifest* naming what it captured and when. Restoring from a directory
whose manifest is missing or unreadable is refused: a restore is run during an incident,
which is exactly when nobody should be guessing whether a directory is a complete backup.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from fasterrag import __version__
from fasterrag.adapters.vectordb.base import VectorDBAdapter
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.observability.logging import get_logger
from fasterrag.services.journal import DEFAULT_JOURNAL_ROOT
from fasterrag.services.lockfile import DEFAULT_LOCK_ROOT
from fasterrag.services.traces import DEFAULT_TRACE_ROOT

__all__ = [
    "MANIFEST_NAME",
    "BackupManifest",
    "back_up",
    "read_manifest",
    "restore",
]

MANIFEST_NAME: Final = "manifest.json"

# CRITICAL: `.env` is deliberately absent. Every other artifact is safe to copy; secrets are
# not, and a backup that quietly included them would put credentials wherever backups live.
_FILE_ARTIFACTS: Final[tuple[tuple[str, Path], ...]] = (
    ("locks", DEFAULT_LOCK_ROOT),
    ("journal", DEFAULT_JOURNAL_ROOT),
    ("traces", DEFAULT_TRACE_ROOT),
)

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class BackupManifest:
    """What one backup captured, and when."""

    created_at: str
    fasterrag: str
    collections: dict[str, str] = field(default_factory=dict)
    vector_counts: dict[str, int] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)
    config: str | None = None
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return the persisted manifest."""
        return {
            "created_at": self.created_at,
            "fasterrag": self.fasterrag,
            "collections": self.collections,
            "vector_counts": self.vector_counts,
            "artifacts": self.artifacts,
            "config": self.config,
            "notes": self.notes,
            "excludes": [".env — secrets are operator-owned and never backed up"],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BackupManifest:
        """Rebuild a manifest from its persisted form."""
        return cls(
            created_at=str(payload["created_at"]),
            fasterrag=str(payload.get("fasterrag", "")),
            collections=dict(payload.get("collections") or {}),
            vector_counts={
                name: int(count) for name, count in (payload.get("vector_counts") or {}).items()
            },
            artifacts=list(payload.get("artifacts") or []),
            config=payload.get("config"),
            notes=str(payload.get("notes", "")),
        )


def read_manifest(destination: Path) -> BackupManifest:
    """Return a backup's manifest.

    Raises:
        FasterRagError: With ``NOT_FOUND`` when the manifest is absent or unreadable. A
            restore runs during an incident, which is the worst possible moment to guess
            whether a directory is a complete backup.
    """
    path = destination / MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return BackupManifest.from_dict(payload)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise FasterRagError(
            f"{destination} holds no readable backup manifest, so it cannot be restored "
            "from; a directory without one is not a backup",
            code=ErrorCode.NOT_FOUND,
            retryable=False,
        ) from exc


async def back_up(
    destination: Path,
    adapter: VectorDBAdapter,
    *,
    config_path: Path | None = None,
    collections: list[str] | None = None,
) -> BackupManifest:
    """Capture every documented artifact into ``destination``.

    Args:
        destination: Directory the backup is written to; created if absent.
        adapter: The vector database, which takes its own native snapshots.
        config_path: The ``config.yaml`` in force, copied alongside. It carries no secrets,
            which is precisely why it is safe to include.
        collections: Which collections to snapshot; every one by default.

    Returns:
        The manifest describing what was captured.
    """
    destination.mkdir(parents=True, exist_ok=True)

    listing = await adapter.list_collections()
    wanted = collections or [info.name for info in listing]
    counts = {info.name: info.vectors for info in listing if info.name in wanted}

    snapshots: dict[str, str] = {}
    for name in wanted:
        snapshots[name] = await adapter.snapshot(name)
        _logger.info(
            "snapshotted collection", extra={"collection": name, "snapshot": snapshots[name]}
        )

    captured: list[str] = []
    for label, source in _FILE_ARTIFACTS:
        if source.exists():
            shutil.copytree(source, destination / label, dirs_exist_ok=True)
            captured.append(label)

    stored_config: str | None = None
    if config_path is not None and config_path.is_file():
        shutil.copy2(config_path, destination / config_path.name)
        stored_config = config_path.name

    manifest = BackupManifest(
        created_at=datetime.now(tz=UTC).isoformat(),
        fasterrag=__version__,
        collections=snapshots,
        vector_counts=counts,
        artifacts=captured,
        config=stored_config,
        notes=(
            "snapshots are held by the vector database itself; this directory records which "
            "snapshot belongs to which collection, plus the control files that prove what "
            "the index is"
        ),
    )
    (destination / MANIFEST_NAME).write_text(
        json.dumps(manifest.as_dict(), indent=2), encoding="utf-8"
    )

    _logger.info(
        "backup complete",
        extra={
            "destination": str(destination),
            "collections": len(snapshots),
            "artifacts": captured,
        },
    )
    return manifest


async def restore(
    source: Path,
    adapter: VectorDBAdapter,
    *,
    collections: list[str] | None = None,
    restore_files: bool = True,
) -> dict[str, Any]:
    """Restore a deployment from a backup directory.

    Args:
        source: A directory holding a manifest and the control files.
        adapter: The vector database to restore into.
        collections: Which collections to restore; every one in the manifest by default.
        restore_files: Whether to put the control files back. Off when restoring only a
            single corrupted collection, which is the cheaper shortcut of
            ``docs/disaster-recovery.md`` §4.

    Returns:
        What was restored and what was verified, including any collection whose restored
        vector count differs from what the manifest recorded — a mismatch is reported rather
        than raised, because a partial restore an operator can see beats one that aborts
        halfway with no report of how far it got.
    """
    manifest = read_manifest(source)
    wanted = collections or list(manifest.collections)

    restored: list[str] = []
    mismatches: list[dict[str, Any]] = []
    for name in wanted:
        snapshot = manifest.collections.get(name)
        if snapshot is None:
            mismatches.append({"collection": name, "problem": "absent from the manifest"})
            continue

        await adapter.restore_snapshot(name, snapshot)
        restored.append(name)

    live = {info.name: info.vectors for info in await adapter.list_collections()}
    for name in restored:
        expected = manifest.vector_counts.get(name)
        actual = live.get(name)
        if expected is not None and actual is not None and expected != actual:
            mismatches.append(
                {"collection": name, "expected_vectors": expected, "restored_vectors": actual}
            )

    replaced: list[str] = []
    if restore_files:
        for label, target in _FILE_ARTIFACTS:
            staged = source / label
            if staged.is_dir():
                shutil.copytree(staged, target, dirs_exist_ok=True)
                replaced.append(label)

    _logger.info(
        "restore complete",
        extra={
            "source": str(source),
            "collections": restored,
            "artifacts": replaced,
            "mismatches": len(mismatches),
        },
    )
    return {
        "created_at": manifest.created_at,
        "collections": restored,
        "artifacts": replaced,
        "mismatches": mismatches,
        "verified": not mismatches,
    }
