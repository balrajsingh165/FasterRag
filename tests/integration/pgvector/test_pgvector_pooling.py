"""Concurrency and statement-budget behaviour of the pgvector adapter, against real PG.

Every case here is built on the same rendezvous, because timing assertions on a laptop
running Docker measure the laptop. An external session takes an ``ACCESS EXCLUSIVE`` lock on
a collection's table, which makes any adapter statement touching that table block for
exactly as long as the test chooses — no sleeps, no races, and a failure that is a real
failure rather than a slow machine.

That rendezvous answers "what input makes this fail?" for both fixes:

* **Pooling (TASK-0239)** — while one operation is blocked on the lock, an operation needing
  no lock at all must still answer. Held on a single serialized connection it cannot, and the
  case fails by timing out, which is what it did before the pool landed.
* **Budgets (TASK-0240)** — a blocked statement is cancelled by the server at whichever
  budget applies, so the two budgets can be told apart by which one a blocked statement dies
  at. A query outliving the query budget, or a snapshot dying at it, both fail here.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator
from typing import ClassVar

import psycopg
import pytest

from fasterrag.adapters.vectordb.base import CollectionSpec, SearchQuery, VectorDBAdapter
from fasterrag.adapters.vectordb.pgvector import PgvectorAdapter, table_name_for
from fasterrag.config.schema import Settings
from fasterrag.errors import FasterRagError
from tests.contract.vectordb import DIMENSIONS
from tests.integration.pgvector.conftest import DSN_VAR, PGVECTOR_DSN

pytestmark = pytest.mark.integration

QUERY_BUDGET_MS = 1500
LONG_MAINTENANCE_MS = 60000
SHORT_MAINTENANCE_MS = 1500

_LOCK_WAIT_TIMEOUT = 15.0
_LOCK_POLL_SECONDS = 0.05

# A blocked statement is cancelled by PostgreSQL at its budget, not by a stopwatch here, so
# this only has to be loose enough to absorb scheduling and one round trip.
_SLACK_SECONDS = 3.0


def tuned_settings(**pgvector_overrides: int) -> Settings:
    """Return pgvector settings with a short query budget and the given overrides.

    The query budget is shortened so a blocked statement dies in seconds rather than the
    default five, which keeps the suite quick without changing what is being proved.
    """
    return Settings.model_validate(
        {
            "vector_db": {
                "provider": "pgvector",
                "mode": "external",
                "pgvector": {"dsn_env": DSN_VAR, **pgvector_overrides},
            },
            "reliability": {"timeouts": {"vector_db_ms": QUERY_BUDGET_MS}},
        }
    )


@pytest.fixture
def blocker(pgvector: Settings) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    """Yield a plain session used to take table locks the adapter will block on."""
    connection = psycopg.connect(PGVECTOR_DSN, connect_timeout=10)
    yield connection
    connection.rollback()
    connection.close()


@pytest.fixture
def observer(pgvector: Settings) -> Iterator[psycopg.Connection[tuple[object, ...]]]:
    """Yield an autocommit session for reading ``pg_stat_activity``.

    # CRITICAL: this must not be the ``blocker`` connection, and must stay in autocommit.
    # PostgreSQL runs with ``stats_fetch_consistency = cache``, so the statistics a
    # transaction reads are fetched once and then frozen for its whole life. Polling from
    # inside the transaction that holds the lock returns the snapshot taken before anything
    # blocked, forever — the wait is never observed and the rendezvous silently degrades to a
    # sleep. One statement per transaction is what keeps each poll a fresh read.
    """
    connection = psycopg.connect(PGVECTOR_DSN, connect_timeout=10, autocommit=True)
    yield connection
    connection.close()


def lock_table(blocker: psycopg.Connection[tuple[object, ...]], collection: str) -> None:
    """Take an ``ACCESS EXCLUSIVE`` lock on a collection's table until the test ends."""
    blocker.execute(
        psycopg.sql.SQL("LOCK TABLE {table} IN ACCESS EXCLUSIVE MODE").format(
            table=psycopg.sql.Identifier("fasterrag", table_name_for(collection))
        )
    )


async def wait_until_blocked(observer: psycopg.Connection[tuple[object, ...]]) -> None:
    """Block until some backend is genuinely parked on a lock.

    Without this the concurrency cases could pass by never overlapping at all — the adapter
    operation could finish before the second one starts, and the assertion would prove
    nothing. Asking PostgreSQL to confirm a backend is waiting is what makes the overlap a
    fact rather than an assumption, which is why this raises instead of returning.
    """
    deadline = time.monotonic() + _LOCK_WAIT_TIMEOUT
    while time.monotonic() < deadline:
        waiting = observer.execute(
            "SELECT count(*) FROM pg_stat_activity "
            "WHERE wait_event_type = 'Lock' AND state = 'active'"
        ).fetchone()
        if waiting is not None and int(str(waiting[0])) > 0:
            return
        await asyncio.sleep(_LOCK_POLL_SECONDS)
    raise AssertionError(
        "no backend ever waited on the lock, so nothing in this case actually overlapped"
    )


class PooledFixtures:
    """An adapter built from tuned settings, plus a collection to lock."""

    overrides: ClassVar[dict[str, int]] = {}

    @pytest.fixture
    async def adapter(self, pgvector: Settings) -> AsyncIterator[VectorDBAdapter]:
        built = PgvectorAdapter(tuned_settings(**self.overrides))
        yield built
        await built.close()

    @pytest.fixture
    async def collection(
        self, adapter: VectorDBAdapter, collection_name: str
    ) -> AsyncIterator[str]:
        await adapter.create_collection(CollectionSpec(name=collection_name, dimensions=DIMENSIONS))
        yield collection_name
        await adapter.drop_collection(collection_name)

    @staticmethod
    def search_of(collection: str) -> SearchQuery:
        """Return a dense search that must touch the collection's locked table."""
        return SearchQuery(collection=collection, vector=[1.0] * DIMENSIONS, limit=1)


class TestConcurrency(PooledFixtures):
    """TASK-0239: one blocked operation must not stall every other one."""

    overrides: ClassVar[dict[str, int]] = {"maintenance_timeout_ms": LONG_MAINTENANCE_MS}

    async def test_an_operation_needing_no_lock_answers_while_another_is_blocked(
        self,
        adapter: VectorDBAdapter,
        collection: str,
        blocker: psycopg.Connection[tuple[object, ...]],
        observer: psycopg.Connection[tuple[object, ...]],
    ) -> None:
        """A blocked search must not queue a health check behind it.

        Held on one serialized connection this cannot pass: ``health()`` runs ``SELECT 1``,
        needs no lock, and still waits for the search to release the adapter's single
        connection — which the lock guarantees it will not do.
        """
        lock_table(blocker, collection)
        search = asyncio.create_task(adapter.search(self.search_of(collection)))
        await wait_until_blocked(observer)
        assert not search.done()

        health = await asyncio.wait_for(adapter.health(), timeout=_SLACK_SECONDS)

        assert health.healthy
        assert not search.done()
        with pytest.raises(FasterRagError):
            await search

    async def test_many_operations_answer_while_one_is_blocked(
        self,
        adapter: VectorDBAdapter,
        collection: str,
        blocker: psycopg.Connection[tuple[object, ...]],
        observer: psycopg.Connection[tuple[object, ...]],
    ) -> None:
        """Concurrency is more than two: several operations overlap the blocked one."""
        lock_table(blocker, collection)
        search = asyncio.create_task(adapter.search(self.search_of(collection)))
        await wait_until_blocked(observer)

        answered = await asyncio.wait_for(
            asyncio.gather(*(adapter.health() for _ in range(4))), timeout=_SLACK_SECONDS
        )

        assert [status.healthy for status in answered] == [True] * 4
        with pytest.raises(FasterRagError):
            await search


class TestQueryBudget(PooledFixtures):
    """TASK-0240: an ordinary statement is bounded by ``vector_db_ms``."""

    overrides: ClassVar[dict[str, int]] = {"maintenance_timeout_ms": LONG_MAINTENANCE_MS}

    async def test_a_blocked_search_is_cancelled_at_the_query_budget(
        self,
        adapter: VectorDBAdapter,
        collection: str,
        blocker: psycopg.Connection[tuple[object, ...]],
    ) -> None:
        """A search that cannot proceed dies at its budget, as a typed error.

        Without a statement timeout this never returns while the lock is held, which is the
        state the adapter shipped in.
        """
        lock_table(blocker, collection)
        started = time.perf_counter()

        with pytest.raises(FasterRagError) as raised:
            await asyncio.wait_for(
                adapter.search(self.search_of(collection)),
                timeout=QUERY_BUDGET_MS / 1000 + _SLACK_SECONDS,
            )

        elapsed = time.perf_counter() - started
        assert elapsed >= QUERY_BUDGET_MS / 1000 * 0.5
        assert "budget" in raised.value.detail
        assert isinstance(raised.value.__cause__, psycopg.Error)


class TestMaintenanceBudget(PooledFixtures):
    """TASK-0240: schema changes and copies get their own, longer budget."""

    overrides: ClassVar[dict[str, int]] = {"maintenance_timeout_ms": LONG_MAINTENANCE_MS}

    async def test_a_snapshot_outlives_the_query_budget(
        self,
        adapter: VectorDBAdapter,
        collection: str,
        blocker: psycopg.Connection[tuple[object, ...]],
        observer: psycopg.Connection[tuple[object, ...]],
    ) -> None:
        """A blocked snapshot must still be running long after a query would have died.

        This is the trap TASK-0240 names: a bound tight enough for a query aborts the index
        builds and table copies a collection cannot be built without. Give maintenance the
        query budget and this case fails, because the snapshot dies at 1.5s.
        """
        lock_table(blocker, collection)
        snapshot = asyncio.create_task(adapter.snapshot(collection))
        await wait_until_blocked(observer)

        await asyncio.sleep(QUERY_BUDGET_MS / 1000 + 1.0)
        assert not snapshot.done()

        blocker.rollback()
        name = await asyncio.wait_for(snapshot, timeout=_SLACK_SECONDS * 2)

        assert name in await adapter.list_snapshots(collection)


class TestMaintenanceIsStillBounded(PooledFixtures):
    """TASK-0240: the longer budget is a bound, not an absence of one."""

    overrides: ClassVar[dict[str, int]] = {"maintenance_timeout_ms": SHORT_MAINTENANCE_MS}

    async def test_a_blocked_snapshot_is_cancelled_at_the_maintenance_budget(
        self,
        adapter: VectorDBAdapter,
        collection: str,
        blocker: psycopg.Connection[tuple[object, ...]],
    ) -> None:
        """A runaway maintenance statement is cancelled too, at its own budget.

        Configured short, the snapshot dies; configured long, the case above proves it
        survives. Together they show the budget is applied rather than simply disabled.
        """
        lock_table(blocker, collection)

        with pytest.raises(FasterRagError) as raised:
            await asyncio.wait_for(
                adapter.snapshot(collection),
                timeout=SHORT_MAINTENANCE_MS / 1000 + _SLACK_SECONDS,
            )

        assert "budget" in raised.value.detail


class TestBudgetDoesNotLeak(PooledFixtures):
    """TASK-0240: the maintenance budget dies with its transaction."""

    overrides: ClassVar[dict[str, int]] = {
        "maintenance_timeout_ms": LONG_MAINTENANCE_MS,
        "pool_max_size": 1,
    }

    async def test_a_query_after_maintenance_still_dies_at_the_query_budget(
        self,
        adapter: VectorDBAdapter,
        collection: str,
        blocker: psycopg.Connection[tuple[object, ...]],
    ) -> None:
        """The connection that ran maintenance must go back to the query budget.

        A pool of one is the point: every operation here provably reuses the same backend, so
        a maintenance budget applied with ``SET`` instead of ``SET LOCAL`` would still be in
        force for the search, which would then hang for a minute instead of dying at 1.5s.
        """
        await adapter.snapshot(collection)
        lock_table(blocker, collection)
        started = time.perf_counter()

        with pytest.raises(FasterRagError):
            await asyncio.wait_for(
                adapter.search(self.search_of(collection)),
                timeout=QUERY_BUDGET_MS / 1000 + _SLACK_SECONDS,
            )

        assert time.perf_counter() - started < LONG_MAINTENANCE_MS / 1000
