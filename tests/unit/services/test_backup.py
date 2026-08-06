"""Backup sets, retention, and restore resolution.

This module had no tests, which is how it shipped writing every run into one directory:
each backup silently overwrote the last, so a deployment running daily backups held one
recovery point rather than a history, and the snapshots the overwritten manifests referenced
were orphaned inside the vector database.
"""

from pathlib import Path
from typing import Any

import pytest

from fasterrag.adapters.vectordb.base import CollectionInfo
from fasterrag.errors import FasterRagError, ProviderError
from fasterrag.services.backup import (
    MANIFEST_NAME,
    SET_PREFIX,
    back_up,
    backup_sets,
    latest_set,
    prune,
    read_manifest,
    restore,
)


class FakeAdapter:
    """Hands out a fresh snapshot per call and records what was deleted."""

    def __init__(self, *, refuse_delete: bool = False) -> None:
        self.live: set[str] = set()
        self.deleted: list[str] = []
        self.restored: list[tuple[str, str]] = []
        self.refuse_delete = refuse_delete
        self._issued = 0

    async def list_collections(self) -> list[CollectionInfo]:
        return [CollectionInfo(name="policies", vectors=12, dimensions=384, distance="cosine")]

    async def snapshot(self, collection: str) -> str:
        self._issued += 1
        name = f"{collection}-snap-{self._issued}"
        self.live.add(name)
        return name

    async def delete_snapshot(self, collection: str, snapshot: str) -> bool:
        if self.refuse_delete:
            raise ProviderError("the backend refused", retryable=True)
        self.deleted.append(snapshot)
        self.live.discard(snapshot)
        return True

    async def restore_snapshot(self, collection: str, snapshot: str) -> None:
        self.restored.append((collection, snapshot))

    async def count(self, collection: str) -> int:
        return 12

    async def close(self) -> None:
        return None


def adapter(**kwargs: Any) -> Any:
    return FakeAdapter(**kwargs)


async def test_a_backup_writes_its_own_set(tmp_path: Path) -> None:
    await back_up(tmp_path, adapter())

    sets = backup_sets(tmp_path)
    assert len(sets) == 1
    assert sets[0].name.startswith(SET_PREFIX)


async def test_the_manifest_lives_in_the_set(tmp_path: Path) -> None:
    await back_up(tmp_path, adapter())

    assert (backup_sets(tmp_path)[0] / MANIFEST_NAME).is_file()


async def test_repeated_backups_accumulate_history(tmp_path: Path) -> None:
    """The defect this module existed without a test for: each run overwrote the last."""
    backend = adapter()

    for _ in range(3):
        await back_up(tmp_path, backend, retain=10)

    assert len(backup_sets(tmp_path)) == 3


async def test_every_set_keeps_its_own_snapshot(tmp_path: Path) -> None:
    """One manifest overwriting another orphans the snapshot the first referenced."""
    backend = adapter()

    for _ in range(3):
        await back_up(tmp_path, backend, retain=10)

    referenced = {read_manifest(path).collections["policies"] for path in backup_sets(tmp_path)}
    assert len(referenced) == 3
    assert referenced == backend.live


async def test_two_backups_in_one_second_do_not_merge(tmp_path: Path) -> None:
    """A shared directory would hold two runs' files under one manifest describing one."""
    backend = adapter()

    await back_up(tmp_path, backend, retain=10)
    await back_up(tmp_path, backend, retain=10)

    assert len(backup_sets(tmp_path)) == 2


async def test_retention_keeps_the_newest(tmp_path: Path) -> None:
    backend = adapter()

    for _ in range(6):
        await back_up(tmp_path, backend, retain=3)

    assert len(backup_sets(tmp_path)) == 3


async def test_retention_deletes_the_backend_snapshots(tmp_path: Path) -> None:
    """Pruning only the directory leaves the snapshot consuming disk, referenced by nothing."""
    backend = adapter()

    for _ in range(6):
        await back_up(tmp_path, backend, retain=3)

    assert len(backend.deleted) == 3
    assert len(backend.live) == 3


async def test_retention_leaves_no_orphans(tmp_path: Path) -> None:
    """Every surviving snapshot must be reachable from a surviving manifest."""
    backend = adapter()

    for _ in range(6):
        await back_up(tmp_path, backend, retain=2)

    referenced = {read_manifest(path).collections["policies"] for path in backup_sets(tmp_path)}
    assert backend.live == referenced


async def test_pruning_happens_after_the_new_set_is_written(tmp_path: Path) -> None:
    """Pruning first would trade a full history for a shorter one plus a failed newest."""
    backend = adapter()
    for _ in range(3):
        await back_up(tmp_path, backend, retain=3)

    await back_up(tmp_path, backend, retain=3)

    assert len(backup_sets(tmp_path)) == 3


async def test_retention_below_one_is_refused(tmp_path: Path) -> None:
    """A policy that deletes every backup is never what somebody meant to type."""
    with pytest.raises(FasterRagError):
        await prune(tmp_path, adapter(), retain=0)


async def test_a_set_with_an_unreadable_manifest_is_still_prunable(tmp_path: Path) -> None:
    """One corrupt directory must not pin retention and grow the destination forever."""
    backend = adapter()
    for _ in range(3):
        await back_up(tmp_path, backend, retain=10)
    (backup_sets(tmp_path)[0] / MANIFEST_NAME).write_text("not json", encoding="utf-8")

    removed = await prune(tmp_path, backend, retain=1)

    assert len(removed) == 2
    assert len(backup_sets(tmp_path)) == 1


async def test_a_snapshot_that_will_not_delete_does_not_stop_the_prune(tmp_path: Path) -> None:
    """The local set still goes; the leak is named in the log rather than left silent."""
    backend = adapter(refuse_delete=True)
    for _ in range(3):
        await back_up(tmp_path, backend, retain=10)

    removed = await prune(tmp_path, backend, retain=1)

    assert len(removed) == 2
    assert len(backup_sets(tmp_path)) == 1


async def test_unrelated_directories_are_never_pruned(tmp_path: Path) -> None:
    """A destination is an operator's directory; retention only owns what it created."""
    keep = tmp_path / "notes"
    keep.mkdir()
    backend = adapter()
    for _ in range(3):
        await back_up(tmp_path, backend, retain=1)

    assert keep.is_dir()


async def test_the_newest_set_is_reported(tmp_path: Path) -> None:
    backend = adapter()
    for _ in range(3):
        await back_up(tmp_path, backend, retain=10)

    assert latest_set(tmp_path) == backup_sets(tmp_path)[-1]


async def test_no_sets_reports_none(tmp_path: Path) -> None:
    assert latest_set(tmp_path) is None
    assert backup_sets(tmp_path) == []


async def test_restore_accepts_the_destination_and_picks_the_newest(tmp_path: Path) -> None:
    """During an incident the path to hand is the one that was backed up to."""
    backend = adapter()
    for _ in range(3):
        await back_up(tmp_path, backend, retain=10)
    newest = read_manifest(backup_sets(tmp_path)[-1]).collections["policies"]

    await restore(tmp_path, backend, restore_files=False)

    assert backend.restored == [("policies", newest)]


async def test_restore_accepts_one_set_directly(tmp_path: Path) -> None:
    """Recovering to a point that is not the newest is the whole reason to keep a history."""
    backend = adapter()
    for _ in range(3):
        await back_up(tmp_path, backend, retain=10)
    oldest = backup_sets(tmp_path)[0]
    wanted = read_manifest(oldest).collections["policies"]

    await restore(oldest, backend, restore_files=False)

    assert backend.restored == [("policies", wanted)]
