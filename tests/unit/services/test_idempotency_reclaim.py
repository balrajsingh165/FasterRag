"""Reclaiming idempotency records whose keys never come back (TASK-0365).

The bug these cover is a disk leak in a feature that is on by default: expiry was lazy, so
only a key that was *used again* was ever cleaned up, and the recommended usage — a fresh
UUID per request — never uses one again. The cases that matter are therefore the ones a
naive sweeper gets wrong: deleting a record that is still replayable, deleting one the
instant it expires while a concurrent request is taking it over, letting N workers all walk
the directory at once, and failing a request because the sweep failed.
"""

import os
import threading
from pathlib import Path
from typing import Any

import pytest

from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.idempotency import (
    RECLAIM_GRACE_SECONDS,
    FileIdempotencyStore,
    IdempotencyStore,
    Reclamation,
    Reservation,
    StoredResponse,
)
from fasterrag.services.journal import envelope

KEY = "11111111-2222-3333-4444-555555555555"
TENANT = "acme"
DIGEST = "d" * 64
RESPONSE = StoredResponse(status=201, media_type="application/json", body=b'{"name":"handbook"}')


def sweeping_store(root: Path, *, ttl_seconds: int = 1) -> FileIdempotencyStore:
    """Return a store already due a pass, so a test need not wait out an interval."""
    store = FileIdempotencyStore(root, ttl_seconds=ttl_seconds, lease_seconds=60)
    store._next_sweep = 0.0
    return store


def age(store: FileIdempotencyStore, key: str, tenant: str | None, seconds: float) -> Path:
    """Rewind a record's recorded age *and* its modification time, as real time would."""
    path = store.path_for(key, tenant)
    record = store.read_record(path)
    assert record is not None
    record["created_at"] = float(record["created_at"]) - seconds
    store.write_record(path, record)
    moment = os.stat(path).st_mtime - seconds
    os.utime(path, (moment, moment))
    return path


def completed(store: FileIdempotencyStore, key: str = KEY, tenant: str | None = TENANT) -> None:
    """Drive one key all the way to a stored response, as a real request would."""
    reservation = store.claim(key, tenant=tenant, fingerprint=DIGEST)
    assert isinstance(reservation, Reservation)
    store.keep(reservation, RESPONSE)


def test_a_record_whose_key_never_returns_is_reclaimed(tmp_path: Path) -> None:
    """The leak itself: nothing used to remove a record no client ever asks for again."""
    store = sweeping_store(tmp_path)
    completed(store)
    path = age(store, KEY, TENANT, store.ttl_seconds + RECLAIM_GRACE_SECONDS + 1)

    outcome = store.reclaim()

    assert not path.exists()
    assert (outcome.opened, outcome.removed) == (1, 1)


def test_a_replayable_record_survives_a_pass(tmp_path: Path) -> None:
    """A sweep that shortens the published replay window would be worse than the leak."""
    store = sweeping_store(tmp_path, ttl_seconds=3600)
    completed(store)

    outcome = store.reclaim()

    assert outcome.removed == 0
    assert store.claim(KEY, tenant=TENANT, fingerprint=DIGEST) == RESPONSE


def test_a_record_is_kept_until_the_grace_margin_has_also_elapsed(tmp_path: Path) -> None:
    """Expiry alone is not enough: a takeover may be rewriting the record right now."""
    store = sweeping_store(tmp_path)
    completed(store)
    path = age(store, KEY, TENANT, store.ttl_seconds + RECLAIM_GRACE_SECONDS - 10)

    assert store.reclaim().removed == 0
    assert path.exists()


def test_an_abandoned_reservation_is_reclaimed_on_the_record_lifetime(tmp_path: Path) -> None:
    """A worker that died mid-request frees the key in minutes and the file on the lifetime."""
    store = sweeping_store(tmp_path, ttl_seconds=3600)
    assert isinstance(store.claim(KEY, tenant=TENANT, fingerprint=DIGEST), Reservation)
    path = age(store, KEY, TENANT, store.ttl_seconds + RECLAIM_GRACE_SECONDS + 1)

    assert store.reclaim().removed == 1
    assert not path.exists()


def test_a_record_that_fails_its_checksum_is_reclaimed(tmp_path: Path) -> None:
    """A torn record holds nothing: ``claim`` already overwrites it, so a sweep may drop it."""
    store = sweeping_store(tmp_path)
    completed(store)
    path = age(store, KEY, TENANT, store.ttl_seconds + RECLAIM_GRACE_SECONDS + 1)
    path.write_text('{"checksum": "0", "body": "{}"}', encoding="utf-8")
    moment = os.stat(path).st_mtime - 10_000
    os.utime(path, (moment, moment))

    assert store.reclaim().removed == 1


def test_an_orphaned_temporary_file_is_reclaimed_but_a_fresh_one_is_not(tmp_path: Path) -> None:
    """``write_record`` holds a temporary for microseconds, so an old one is a crashed write."""
    store = sweeping_store(tmp_path)
    completed(store)
    bucket = store.path_for(KEY, TENANT).parent
    orphan = bucket / "abandoned.json.deadbeef.tmp"
    orphan.write_text(envelope({"state": "in_progress"}), encoding="utf-8")
    moment = os.stat(orphan).st_mtime - RECLAIM_GRACE_SECONDS - 1
    os.utime(orphan, (moment, moment))
    fresh = bucket / "running.json.feedface.tmp"
    fresh.write_text(envelope({"state": "in_progress"}), encoding="utf-8")

    store.reclaim()

    assert not orphan.exists()
    assert fresh.exists()


def test_a_pass_over_an_empty_store_is_harmless(tmp_path: Path) -> None:
    """Reclamation runs before anything has been written, on every fresh deployment."""
    assert sweeping_store(tmp_path).reclaim().scanned == 0


def test_only_one_process_of_several_sweeps_per_interval(tmp_path: Path) -> None:
    """N Gunicorn workers share one directory; N walks of it per interval is N-1 wasted."""
    first = FileIdempotencyStore(tmp_path, ttl_seconds=3600, lease_seconds=3600)
    second = FileIdempotencyStore(tmp_path, ttl_seconds=3600, lease_seconds=3600)

    assert first._take_sweep_lease() is True
    assert second._take_sweep_lease() is False


def test_the_lease_is_retaken_once_its_interval_has_passed(tmp_path: Path) -> None:
    """A worker that died holding the lease must not stop the host sweeping for good."""
    store = FileIdempotencyStore(tmp_path, ttl_seconds=3600, lease_seconds=3600)
    assert store._take_sweep_lease() is True

    lease = tmp_path / "idempotency" / ".sweep.json"
    held = store.read_record(lease)
    assert held is not None
    held["created_at"] = float(held["created_at"]) - 7200
    store.write_record(lease, held)

    assert store._take_sweep_lease() is True


def test_the_sweep_lease_is_never_mistaken_for_a_record(tmp_path: Path) -> None:
    """The lease lives beside the tenant buckets, so a pass must not try to reclaim it."""
    store = sweeping_store(tmp_path)
    completed(store)
    assert store._take_sweep_lease() is True
    lease = tmp_path / "idempotency" / ".sweep.json"

    store.reclaim()

    assert lease.exists()


async def test_a_request_starts_a_pass_beside_itself_rather_than_inside_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pass costs a cold open per removal; charging that to one caller is what to avoid."""
    store = sweeping_store(tmp_path)
    released = threading.Event()
    real = store.reclaim

    def blocking() -> Reclamation:
        released.wait(timeout=10)
        return real()

    monkeypatch.setattr(store, "reclaim", blocking)
    completed(store, key="stale")
    stale = age(store, "stale", TENANT, store.ttl_seconds + RECLAIM_GRACE_SECONDS + 1)

    outcome = await store.begin(KEY, tenant=TENANT, fingerprint=DIGEST)

    assert isinstance(outcome, Reservation)
    assert stale.exists()
    released.set()
    await store.close()
    assert not stale.exists()


async def test_a_second_request_never_starts_a_second_pass_while_one_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One pass at a time per process, whatever concurrency reaches ``begin``."""
    store = sweeping_store(tmp_path, ttl_seconds=3600)
    store.sweep_interval_seconds = 0
    released = threading.Event()

    def blocking() -> Reclamation:
        released.wait(timeout=10)
        return Reclamation()

    monkeypatch.setattr(store, "reclaim", blocking)

    await store.begin("first", tenant=TENANT, fingerprint=DIGEST)
    running = store._sweeping
    await store.begin("second", tenant=TENANT, fingerprint=DIGEST)

    assert store._sweeping is running
    released.set()
    await store.close()


async def test_a_store_that_is_not_due_starts_nothing(tmp_path: Path) -> None:
    """The due check is an in-process float, so an ordinary request pays no syscall."""
    store = FileIdempotencyStore(tmp_path, ttl_seconds=3600, lease_seconds=3600)

    assert isinstance(await store.begin(KEY, tenant=TENANT, fingerprint=DIGEST), Reservation)

    await store.close()
    assert not (tmp_path / "idempotency" / ".sweep.json").exists()


async def test_a_failing_pass_never_fails_the_request_that_started_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller's own request succeeded; a directory we could not walk is ours to log."""
    store = sweeping_store(tmp_path)

    def unreadable() -> Any:
        raise OSError("the record store is gone")

    monkeypatch.setattr(store, "reclaim", unreadable)

    outcome = await store.begin(KEY, tenant=TENANT, fingerprint=DIGEST)
    await store.close()

    assert isinstance(outcome, Reservation)


async def test_a_refused_request_still_schedules_the_pass_it_was_due(tmp_path: Path) -> None:
    """Reclamation must not depend on requests succeeding, or a hot key would stop it."""
    store = sweeping_store(tmp_path, ttl_seconds=3600)
    assert isinstance(store.claim(KEY, tenant=TENANT, fingerprint=DIGEST), Reservation)

    with pytest.raises(FasterRagError) as refusal:
        await store.begin(KEY, tenant=TENANT, fingerprint="other")
    await store.close()

    assert refusal.value.code is ErrorCode.IDEMPOTENCY_KEY_REUSED
    assert (tmp_path / "idempotency" / ".sweep.json").exists()


def test_the_directory_returns_to_empty_once_its_traffic_has_aged_out(tmp_path: Path) -> None:
    """The invariant the leak broke: records are bounded by a lifetime, not by all traffic."""
    store = sweeping_store(tmp_path)
    for index in range(25):
        completed(store, key=f"key-{index}")
        age(store, f"key-{index}", TENANT, store.ttl_seconds + RECLAIM_GRACE_SECONDS + 1)

    outcome = store.reclaim()

    assert outcome.removed == 25
    assert list(store.path_for(KEY, TENANT).parent.iterdir()) == []


def test_reclamation_is_not_part_of_the_backend_interface() -> None:
    """A store that expires keys natively has nothing to sweep (TASK-0357's boundary)."""
    assert not hasattr(IdempotencyStore, "reclaim")
    assert hasattr(FileIdempotencyStore, "reclaim")
