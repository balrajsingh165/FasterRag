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

**Each run writes its own timestamped set** under the destination, and ``retain`` prunes the
oldest. Writing every run into one directory would mean each backup overwrote the last, so a
deployment running daily backups would hold exactly one recovery point rather than a history
— and would discover that only when the newest backup turned out to be the corrupt one.

Pruning deletes the backend snapshots a set references, not just its directory. A snapshot
lives inside the vector database; removing only the record of it leaves it orphaned, invisible
to every manifest, and consuming disk until somebody finds it by hand.

**Cadence is not fasterRag's job.** cron, systemd timers, and Task Scheduler already run
things on a schedule, do it better, and are what an operator already monitors. This module
makes one run correct and bounded; ``docs/disaster-recovery.md`` §3 shows how to schedule it.
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
    "DEFAULT_RETAIN",
    "MANIFEST_NAME",
    "SET_PREFIX",
    "BackupManifest",
    "back_up",
    "backup_sets",
    "latest_set",
    "prune",
    "read_manifest",
    "restore",
]

MANIFEST_NAME: Final = "manifest.json"

# Every backup set directory starts with this, so a destination can hold sets alongside
# whatever else an operator keeps there without retention ever considering those for deletion.
SET_PREFIX: Final = "set-"

# Fourteen days of daily backups, matching the documented default in
# docs/disaster-recovery.md. Counted in *sets*, not days: fasterRag runs when it is invoked
# and cannot know the cadence a scheduler was configured with, so counting days would mean
# guessing at a number only the operator holds.
DEFAULT_RETAIN: Final = 14

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


def backup_sets(destination: Path) -> list[Path]:
    """Return the backup sets under ``destination``, oldest first.

    Ordered by directory name, which sorts chronologically because the names are UTC
    timestamps in a fixed-width format. Reading each manifest for its ``created_at`` would
    be more direct and would also make the ordering depend on files that may be exactly
    what is corrupt when someone comes looking.
    """
    if not destination.is_dir():
        return []
    return sorted(
        path for path in destination.iterdir() if path.is_dir() and path.name.startswith(SET_PREFIX)
    )


def latest_set(destination: Path) -> Path | None:
    """Return the newest backup set under ``destination``, or ``None`` if there is none."""
    sets = backup_sets(destination)
    return sets[-1] if sets else None


async def prune(destination: Path, adapter: VectorDBAdapter, *, retain: int) -> list[str]:
    """Delete all but the newest ``retain`` backup sets, and their backend snapshots.

    Args:
        destination: The directory holding the sets.
        adapter: The vector database whose snapshots the pruned sets reference.
        retain: How many sets to keep. Values below one are refused rather than treated as
            "keep nothing" — a retention policy that deletes every backup is never what
            somebody meant to type, and the moment to find out is not the next restore.

    Returns:
        The names of the sets removed.

    Raises:
        FasterRagError: If ``retain`` is less than one.
    """
    if retain < 1:
        raise FasterRagError(
            f"backup retention must keep at least one set, got {retain}",
            code=ErrorCode.VALIDATION_FAILED,
            retryable=False,
        )

    sets = backup_sets(destination)
    doomed = sets[:-retain] if len(sets) > retain else []

    removed: list[str] = []
    for path in doomed:
        try:
            manifest = read_manifest(path)
        except FasterRagError:
            # A set whose manifest is unreadable still has to be prunable, or one corrupt
            # directory would pin retention forever and the destination would grow without
            # bound. Its snapshots cannot be identified, so they are left and logged.
            _logger.warning(
                "pruning a backup set with an unreadable manifest; any backend snapshots it "
                "referenced cannot be identified and are left in place",
                extra={"set": path.name},
            )
        else:
            for collection, snapshot in manifest.collections.items():
                try:
                    await adapter.delete_snapshot(collection, snapshot)
                except FasterRagError as exc:
                    # A snapshot that will not delete must not stop the prune. The local set
                    # still goes, and the leak is named rather than silent.
                    _logger.warning(
                        "could not delete a snapshot while pruning; it may be orphaned",
                        extra={
                            "collection": collection,
                            "snapshot": snapshot,
                            "detail": exc.detail,
                        },
                    )

        shutil.rmtree(path, ignore_errors=True)
        removed.append(path.name)

    if removed:
        _logger.info(
            "pruned backup sets", extra={"removed": removed, "retained": len(sets) - len(removed)}
        )
    return removed


async def back_up(
    destination: Path,
    adapter: VectorDBAdapter,
    *,
    config_path: Path | None = None,
    collections: list[str] | None = None,
    retain: int = DEFAULT_RETAIN,
) -> BackupManifest:
    """Capture every documented artifact into a new timestamped set under ``destination``.

    Args:
        destination: Directory holding the backup sets; created if absent. The set itself
            goes in a timestamped subdirectory, so repeated runs accumulate a history
            instead of each one overwriting the last.
        adapter: The vector database, which takes its own native snapshots.
        config_path: The ``config.yaml`` in force, copied alongside. It carries no secrets,
            which is precisely why it is safe to include.
        collections: Which collections to snapshot; every one by default.
        retain: How many sets to keep, pruning the oldest along with the backend snapshots
            they reference.

    Returns:
        The manifest describing what was captured.
    """
    destination.mkdir(parents=True, exist_ok=True)

    started = datetime.now(tz=UTC)
    # Colons are legal in a POSIX filename and illegal in a Windows one, so an ISO timestamp
    # used verbatim would make every backup directory unopenable on Windows.
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    set_root = destination / f"{SET_PREFIX}{stamp}"

    # A second run inside the same second must not merge into the first set, which would
    # produce one directory holding two runs' files and a manifest describing only the later.
    suffix = 1
    while set_root.exists():
        suffix += 1
        set_root = destination / f"{SET_PREFIX}{stamp}-{suffix}"
    set_root.mkdir(parents=True)

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
            shutil.copytree(source, set_root / label, dirs_exist_ok=True)
            captured.append(label)

    stored_config: str | None = None
    if config_path is not None and config_path.is_file():
        shutil.copy2(config_path, set_root / config_path.name)
        stored_config = config_path.name

    manifest = BackupManifest(
        created_at=started.isoformat(),
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
    (set_root / MANIFEST_NAME).write_text(
        json.dumps(manifest.as_dict(), indent=2), encoding="utf-8"
    )

    # CRITICAL: pruned after the manifest is written, never before. Pruning first would
    # delete an old set to make room for one that then failed to complete, trading a full
    # history for a shorter one plus a broken newest entry.
    await prune(destination, adapter, retain=retain)

    _logger.info(
        "backup complete",
        extra={
            "destination": str(set_root),
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
        source: Either one backup set, or the destination holding several — in which case
            the newest is used. During an incident the thing an operator has to hand is the
            path they backed up to, and making them list its subdirectories first to find
            the newest is an avoidable step at the worst possible moment.
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
    resolved = source if (source / MANIFEST_NAME).is_file() else (latest_set(source) or source)
    if resolved != source:
        _logger.info(
            "restoring the newest backup set in the destination",
            extra={"destination": str(source), "set": resolved.name},
        )

    manifest = read_manifest(resolved)
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
            staged = resolved / label
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
