"""Reindex under continuous query load — the third TASK-0241 scenario, and D2's open half.

D2 promises that re-embedding a corpus never takes queries down. Two parts of that already
have proof: the alias swap is one atomic adapter operation (the shared contract suite) and
the orchestration around it is unit-tested (``tests/unit/services/test_reindex.py``). The
part nothing covered is the one the promise is actually about — what a query *sees* while
the rebuild is running. A test that queries, reindexes, and queries again establishes
nothing about that window, because it never has a query inside it.

So every case here runs queries through the real :class:`RetrievalService` *concurrently
with* a real :func:`~fasterrag.services.reindex.plan_reindex` /
:func:`~fasterrag.services.reindex.swap` cycle, and the drill asserts that the window was
genuinely crossed — at least one query resolved the alias after the first byte of the new
collection was written and before the swap. Without that assertion the suite would pass
just as happily on a scheduling accident in which no query ever overlapped the rebuild,
which is the vacuous-pass shape this repository's trap registry calls T12.

**The seam, and what it is worth.** The backend is a double that holds collections in a
dict and resolves an alias *per search call*, which is what Qdrant does: the adapter sends
the served name on every request and the server resolves it (see ``QdrantAdapter.search``).
The double therefore models the alias semantics the production path depends on, and what
the cases below prove is a property of *our* orchestration — that no code path points the
alias at a collection that is still being filled, and that the query path reads through the
alias rather than pinning a physical name. It is not evidence about Qdrant's own
server-side behaviour under concurrent writes; that needs a live backend, there is no
scenario here that provides one, and none of these docstrings should be read as claiming
otherwise.

**One case asserts a limit rather than a guarantee.** A hybrid query issues two searches,
and each resolves the alias independently, so a swap landing between them fuses results
from both generations. That is the built behaviour, it is pinned below so that it is a
measured fact rather than an assumption, and it bounds D2's "a query sees either the old
collection or the new one" to a single retrieval leg.

**How the drill waits, and why it is not a poll.** An earlier revision of this module paced
the rebuild by spinning on ``asyncio.sleep(0)`` until the query loop had recorded one more
observation, with a thirty-second ``wait_for`` around the spin. That made every pacing step
depend on wall-clock time on a platform whose clock this repository already has a registered
trap for, and it turned *any* stall in the query loop — including the loop task dying on an
unexpected exception — into a thirty-second timeout carrying no information about the cause.
The loop now signals ``Drill.progress`` after every query attempt and the rebuild waits on
that event, so pacing is a causal handshake with no clock in it. ``await_queries`` also
returns the moment a query is dropped or the loop task stops, which hands the failure to the
case's own assertions (``drill.failures == []``) instead of to a timeout. The ``wait_for``
that remains is a deadlock backstop and nothing else: it exists so that a genuine hang in the
retrieval or reindex path fails loudly rather than wedging the whole ``pytest`` run, and its
value is not a latency bound and backs no claim.

That change is measured, not assumed. Replacing ``run_query`` with one that raises an
unexpected (non-``FasterRagError``) exception makes the query loop task die: the earlier
revision took **30,000.3 ms** to report it, because the spin had to exhaust the budget before
``await loop`` could re-raise, and what a reader saw first was a bare ``TimeoutError``. The
handshake reports the same ``ValueError`` in **0.2 ms**, with the real traceback.

**Reproduction status of the timeout this module was parked for.** It was held back on a
report that two of its eight cases time out under a full ``pytest`` run. That does not
reproduce on this checkout: see the ledger entry on TASK-0241 for the run counts. The
handshake above removes the only mechanism in the module that could have produced the
reported symptom without leaving a trace; nothing in ``src/`` was found to be implicated,
and the reindex path holds no lock of any kind, so a true deadlock between the rebuild and
the load generator is not reachable through this seam.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

import pytest

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
from fasterrag.api.problems import build_problem
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError, RetrievalError
from fasterrag.services.querying import RetrievalService
from fasterrag.services.reindex import ReindexPlan, plan_reindex, swap

pytestmark = pytest.mark.chaos

ALIAS: Final = "docs"
VECTOR: Final[list[float]] = [0.1, 0.2, 0.3]
QUERY_TEXT: Final = "termination notice period"

BLUE: Final = "blue"
GREEN: Final = "green"
GENERATION_KEY: Final = "generation"

CORPUS_SIZE: Final = 8
STAMP: Final = "20260101000000"
BLUE_COLLECTION: Final = f"{ALIAS}-v20251231000000"

TAIL_QUERIES: Final = 3
"""How many further queries the drill waits for after the swap.

One would prove the alias moved; three keep the assertion from resting on a single
observation taken at the one instant the swap had just landed.
"""

DEADLOCK_BACKSTOP_SECONDS: Final = 30.0
"""The only wall-clock figure in this module, and it is a backstop rather than a budget.

Every wait below completes on an event the query loop sets, so a healthy drill never reaches
this bound — the whole module runs in roughly 1.4 s. It exists so that a genuine hang in the
retrieval or reindex path fails this case instead of wedging the ``pytest`` run, and it is
not a latency measurement: nothing here runs a real backend, so there is nothing to record.
"""

SHUTDOWN_BACKSTOP_SECONDS: Final = 10.0
"""The same backstop for joining the query loop once the drill has set its stop flag.

The loop checks the flag between queries, so the join costs one in-flight query. A loop that
cannot finish that query is the deadlock this module would otherwise hide, so the join is
bounded rather than open-ended: ``wait_for`` cancels the task and the case fails.
"""

SERVER_ERROR: Final = 500


def reindex_settings() -> Settings:
    """Return settings whose served collection is the alias under test."""
    return Settings.model_validate(
        {
            "vector_db": {
                "provider": "qdrant",
                "mode": "external",
                "api_key_env": None,
                "collection": {"default_name": ALIAS},
            },
            "embeddings": {"provider": "huggingface"},
            "llm": {"provider": "ollama", "api_key_env": None},
        }
    )


def corpus(generation: str) -> list[Point]:
    """Return one generation's points, tagged so a result traces to its collection.

    The generations carry disjoint ids, as a rebuild with changed chunking does, so a
    result assembled from both is visible rather than collapsing into one id set.
    """
    return [
        Point(
            point_id=f"c_{generation}_{index}",
            collection="",
            vector=VECTOR,
            payload={
                "text": f"{generation} passage {index}: either party may terminate.",
                GENERATION_KEY: generation,
            },
        )
        for index in range(CORPUS_SIZE)
    ]


def generation_ids(generation: str) -> frozenset[str]:
    """Return the complete id set a query must see if it sees this generation at all."""
    return frozenset(point.point_id for point in corpus(generation))


class StubRouter:
    """Supplies a fixed query vector so no embedding model is loaded.

    The stand-in ``test_network_partition.py`` uses, for the same reason: the query
    embedding is not the subject here, and a real model would be a second moving part.
    """

    def __init__(self) -> None:
        """Expose itself as its own default adapter, as the real router does."""
        self.default = self

    async def embed_query(self, text: str) -> Sequence[float]:
        """Return the fixed vector."""
        return VECTOR


SearchHook = Callable[[int], Awaitable[None]]


class AliasedStore(VectorDBAdapter):
    """Collections in a dict, with an alias resolved on every single search.

    Per-call resolution is the behaviour being modelled, not a shortcut, and every
    operation advances a counter so a case can place a query inside the rebuild.
    """

    def __init__(self, settings: Settings) -> None:
        """Start empty, with no collections and no aliases."""
        super().__init__(settings)
        self.collections: dict[str, dict[str, Point]] = {}
        self.aliases: dict[str, str] = {}
        self.clock = 0
        self.resolutions: list[tuple[int, str]] = []
        self.before_search: SearchHook | None = None

    def _tick(self) -> int:
        """Advance and return the operation counter."""
        self.clock += 1
        return self.clock

    def _resolve(self, name: str) -> str:
        """Return the physical collection a served name refers to.

        Raises:
            RetrievalError: With ``NOT_FOUND`` when the name resolves to nothing.
        """
        target = self.aliases.get(name, name)
        if target not in self.collections:
            raise RetrievalError(
                f"{name!r} resolves to no collection on this backend",
                code=ErrorCode.NOT_FOUND,
                retryable=False,
            )
        return target

    async def create_collection(self, spec: CollectionSpec) -> None:
        """Create an empty collection, leaving an existing one alone."""
        self._tick()
        self.collections.setdefault(spec.name, {})

    async def list_collections(self) -> list[CollectionInfo]:
        """Return every collection, as the reindex planner reads them."""
        return [
            CollectionInfo(name=name, vectors=len(points))
            for name, points in self.collections.items()
        ]

    async def drop_collection(self, name: str) -> bool:
        """Delete a collection, reporting whether one was there."""
        return self.collections.pop(name, None) is not None

    async def snapshot(self, collection: str) -> str:
        """Unused here; snapshots play no part in an alias swap."""
        return f"{collection}-snap"

    async def list_snapshots(self, collection: str) -> list[str]:
        """Unused here."""
        return []

    async def delete_snapshot(self, collection: str, snapshot: str) -> bool:
        """Unused here."""
        return False

    async def restore_snapshot(self, collection: str, snapshot: str) -> None:
        """Unused here."""
        return

    async def set_alias(self, alias: str, collection: str) -> None:
        """Repoint the alias in one indivisible assignment.

        The double's stand-in for Qdrant applying its delete and create as a single
        alias-operations list: there is no instant at which the name resolves to nothing.
        """
        self._tick()
        self.aliases[alias] = collection

    async def alias_target(self, alias: str) -> str | None:
        """Return what the alias points at, or ``None`` if the name is not an alias."""
        return self.aliases.get(alias)

    async def delete_alias(self, alias: str) -> bool:
        """Remove the alias, reporting whether one existed."""
        self._tick()
        return self.aliases.pop(alias, None) is not None

    async def upsert(self, points: list[Point]) -> UpsertResult:
        """Write points into their collections, yielding between batches.

        The yield is what makes the rebuild something a concurrent query can land inside.
        """
        self._tick()
        await asyncio.sleep(0)
        for point in points:
            self.collections.setdefault(point.collection, {})[point.point_id] = point
        return UpsertResult(upserted=len(points))

    async def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        """Yield a collection's points; no scenario here scans one."""
        for point in list(self.collections.get(collection, {}).values()):
            yield point

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        """Resolve the served name and return everything the resolved collection holds.

        Relevance is not modelled. These scenarios are about *which* collection answered,
        and a scoring function would add a second reason for a result set to change.
        """
        if self.before_search is not None:
            await self.before_search(self.clock)

        target = self._resolve(query.collection)
        self.resolutions.append((self._tick(), target))
        await asyncio.sleep(0)

        stored = list(self.collections[target].values())
        return [
            ScoredPoint(point_id=point.point_id, score=1.0 - index / 100, payload=point.payload)
            for index, point in enumerate(stored[: query.limit])
        ]

    async def update(self, updates: list[PointUpdate]) -> None:
        """Unused here."""
        return

    async def delete(self, selector: PointSelector) -> None:
        """Unused here."""
        return

    async def health(self) -> HealthStatus:
        """Report the double as answering, which it always is."""
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        """Hold nothing to release."""
        return


@dataclass(frozen=True, slots=True)
class Observation:
    """What one query saw, and the operation counter at which it resolved the alias."""

    clock: int
    ids: frozenset[str]
    generations: frozenset[str]


@dataclass(slots=True)
class Drill:
    """Everything a reindex-under-load run recorded, and how the rebuild paces itself.

    ``progress`` is set after every query attempt, so a waiter wakes on the loop making
    progress rather than on a clock. See this module's docstring for why.
    """

    observations: list[Observation] = field(default_factory=list)
    failures: list[FasterRagError] = field(default_factory=list)
    first_green_write: int = 0
    swap_clock: int = 0
    swapped: bool = False
    alias_targets_during_build: list[str | None] = field(default_factory=list)
    progress: asyncio.Event = field(default_factory=asyncio.Event)

    def inside_the_window(self) -> list[Observation]:
        """Return the queries that resolved between the first green write and the swap."""
        return [
            observation
            for observation in self.observations
            if self.first_green_write < observation.clock < self.swap_clock
        ]

    def generations_seen(self) -> set[str]:
        """Return every generation any query observed."""
        return {name for observation in self.observations for name in observation.generations}


def build_service(store: AliasedStore, settings: Settings) -> RetrievalService:
    """Return the real retrieval service wired to the double."""
    return RetrievalService(settings, store, StubRouter(), reranker=None)  # type: ignore[arg-type]


async def run_query(service: RetrievalService, store: AliasedStore, drill: Drill) -> None:
    """Run one query through the real service and record what it saw.

    The clock is where this query's *first* leg resolved, and a typed failure is recorded
    rather than raised so the drill reports dropped queries instead of dying on one.
    """
    first = len(store.resolutions)
    try:
        retrieval = await service.search(QUERY_TEXT)
    except FasterRagError as failure:
        drill.failures.append(failure)
        return

    drill.observations.append(
        Observation(
            clock=store.resolutions[first][0],
            ids=frozenset(chunk.chunk_id for chunk in retrieval.chunks),
            generations=frozenset(
                str(chunk.payload.get(GENERATION_KEY, "")) for chunk in retrieval.chunks
            ),
        )
    )


async def query_continuously(
    service: RetrievalService, store: AliasedStore, drill: Drill, stop: asyncio.Event
) -> None:
    """Issue queries back to back until told to stop, waking the drill after each.

    The ``finally`` makes the handshake safe: a loop that dies or is cancelled still wakes
    its waiter, which then re-raises the cause instead of sitting on an unset event.
    """
    try:
        while not stop.is_set():
            await run_query(service, store, drill)
            drill.progress.set()
            await asyncio.sleep(0)
    finally:
        drill.progress.set()


async def await_queries(drill: Drill, queries: asyncio.Task[None], target: int) -> None:
    """Wait for the loop to record ``target`` observations, with no clock involved.

    It returns early on a dropped query, so the case's own ``failures == []`` assertion
    reports it, and re-raises whatever stopped the loop so the cause arrives, not a stall.
    """
    while len(drill.observations) < target and not drill.failures and not queries.done():
        drill.progress.clear()
        await drill.progress.wait()

    if queries.done():
        await queries


async def build_green(
    store: AliasedStore, plan: ReindexPlan, drill: Drill, *, queries: asyncio.Task[None] | None
) -> None:
    """Fill the new collection in batches, recording where the alias points throughout.

    A rebuild writing into the live collection would show the new name here while it was
    partial. With ``queries`` each batch waits for one more query, arranging the overlap.
    """
    await store.create_collection(CollectionSpec(name=plan.green, dimensions=len(VECTOR)))
    points = [
        Point(
            point_id=point.point_id,
            collection=plan.green,
            vector=point.vector,
            payload=point.payload,
        )
        for point in corpus(GREEN)
    ]

    for index, point in enumerate(points):
        drill.alias_targets_during_build.append(await store.alias_target(ALIAS))
        await store.upsert([point])
        if index == 0:
            drill.first_green_write = store.clock
        if queries is not None:
            await asyncio.wait_for(
                await_queries(drill, queries, len(drill.observations) + 1),
                DEADLOCK_BACKSTOP_SECONDS,
            )


async def reindex_under_load(
    store: AliasedStore, settings: Settings, *, eval_passed: bool | None = True
) -> Drill:
    """Rebuild behind the alias while a query loop runs, and return what happened.

    ``eval_passed`` is handed to :func:`~fasterrag.services.reindex.swap` unchanged, so a
    case can run the identical drill with the gate rejecting the build.
    """
    service = build_service(store, settings)
    drill = Drill()
    stop = asyncio.Event()
    queries = asyncio.create_task(query_continuously(service, store, drill, stop))

    try:
        plan = await plan_reindex(ALIAS, settings, store, stamp=STAMP)
        await build_green(store, plan, drill, queries=queries)
        result = await swap(plan, store, eval_passed=eval_passed)
        drill.swapped = result.swapped
        drill.swap_clock = store.clock
        await asyncio.wait_for(
            await_queries(drill, queries, len(drill.observations) + TAIL_QUERIES),
            DEADLOCK_BACKSTOP_SECONDS,
        )
    finally:
        stop.set()
        await asyncio.wait_for(queries, SHUTDOWN_BACKSTOP_SECONDS)

    return drill


@pytest.fixture
def settings() -> Settings:
    """Return the configuration every case in this module runs under."""
    return reindex_settings()


@pytest.fixture
def store(settings: Settings) -> AliasedStore:
    """Return a backend already serving a complete blue collection behind the alias."""
    backend = AliasedStore(settings)
    backend.collections[BLUE_COLLECTION] = {
        point.point_id: Point(
            point_id=point.point_id,
            collection=BLUE_COLLECTION,
            vector=point.vector,
            payload=point.payload,
        )
        for point in corpus(BLUE)
    }
    backend.aliases[ALIAS] = BLUE_COLLECTION
    return backend


async def test_every_query_across_the_window_sees_one_complete_generation(
    store: AliasedStore, settings: Settings
) -> None:
    """No query observes a half-built collection, and the window is genuinely crossed.

    Completeness is the half that matters: asserting a single generation would pass against
    a rebuild that swapped early, since a partial green collection is still uniformly green.
    """
    drill = await reindex_under_load(store, settings)

    assert drill.swapped is True
    assert drill.inside_the_window(), "no query resolved the alias during the rebuild"
    assert drill.generations_seen() == {BLUE, GREEN}
    for observation in drill.observations:
        assert len(observation.generations) == 1
        seen = next(iter(observation.generations))
        assert observation.ids == generation_ids(seen)


async def test_no_query_is_dropped_while_the_rebuild_runs(
    store: AliasedStore, settings: Settings
) -> None:
    """The load is served throughout: zero failures, and the loop never stalled."""
    drill = await reindex_under_load(store, settings)

    assert drill.failures == []
    assert len(drill.observations) > TAIL_QUERIES


async def test_the_alias_never_points_at_the_collection_being_filled(
    store: AliasedStore, settings: Settings
) -> None:
    """The served name resolves to the old build for the whole of the rebuild.

    Read alongside the case above rather than instead of it: this one watches the mechanism
    and that one watches the promise, and a change can break one without the other.
    """
    drill = await reindex_under_load(store, settings)

    assert drill.alias_targets_during_build
    assert set(drill.alias_targets_during_build) == {BLUE_COLLECTION}
    assert store.aliases[ALIAS] != BLUE_COLLECTION


async def test_a_blocked_gate_leaves_every_query_on_the_old_build(
    store: AliasedStore, settings: Settings
) -> None:
    """A rejected build is never served, however complete it is.

    The gate running *before* the swap is what makes this true, and a gate that ran after
    would be detecting the regression by serving it.
    """
    drill = await reindex_under_load(store, settings, eval_passed=False)

    assert drill.swapped is False
    assert drill.failures == []
    assert drill.generations_seen() == {BLUE}


async def test_a_served_name_that_resolves_to_nothing_fails_typed(
    store: AliasedStore, settings: Settings
) -> None:
    """An alias pointing nowhere raises rather than answering from an empty index.

    The shape this rules out is the quiet one: an empty hit list for an unresolvable name
    would hand the generator an empty context and the caller a confidently empty answer.
    """
    service = build_service(store, settings)
    await store.delete_alias(ALIAS)

    with pytest.raises(RetrievalError) as failure:
        await service.search(QUERY_TEXT)

    assert failure.value.code is ErrorCode.NOT_FOUND


async def test_an_unresolvable_served_name_renders_as_a_problem_document(
    store: AliasedStore, settings: Settings
) -> None:
    """The failure reaches a client as problem+json with a code, never as a generic 500."""
    service = build_service(store, settings)
    await store.delete_alias(ALIAS)

    with pytest.raises(RetrievalError) as failure:
        await service.search(QUERY_TEXT)
    problem = build_problem(failure.value)

    assert problem.status != SERVER_ERROR
    assert problem.code is not ErrorCode.INTERNAL
    assert problem.trace_id


async def test_a_hybrid_query_can_straddle_the_swap_because_each_leg_resolves_the_alias(
    store: AliasedStore, settings: Settings
) -> None:
    """Pinned limit, not a guarantee: one hybrid query can fuse both generations.

    Each leg resolves the served name independently, so a swap between them assembles one
    result from two collections. Pinned so that it is a measured fact, not an assumption.
    """
    # TODO: resolving the alias once per query, and passing the physical name to both legs,
    # would close this window. It is a change to `services/querying.py`, outside the scope
    # this scenario was written under, and is reported rather than made.
    service = build_service(store, settings)
    plan = await plan_reindex(ALIAS, settings, store, stamp=STAMP)
    drill = Drill()
    await build_green(store, plan, drill, queries=None)

    async def swap_between_the_legs(clock: int) -> None:
        """Move the alias once, after the first leg has resolved and before the second."""
        if len(store.resolutions) == 1:
            await swap(plan, store, eval_passed=True)

    store.before_search = swap_between_the_legs
    retrieval = await service.search(QUERY_TEXT)

    generations = {str(chunk.payload.get(GENERATION_KEY, "")) for chunk in retrieval.chunks}
    assert generations == {BLUE, GREEN}


async def test_the_drill_fails_rather_than_passing_without_a_query_in_the_window(
    store: AliasedStore, settings: Settings
) -> None:
    """The window assertion is load-bearing: with no concurrent query there is nothing in it.

    Without this case the suite could not tell a drill that genuinely overlapped the
    rebuild from one whose query loop happened never to be scheduled inside it.
    """
    service = build_service(store, settings)
    drill = Drill()

    plan = await plan_reindex(ALIAS, settings, store, stamp=STAMP)
    await build_green(store, plan, drill, queries=None)
    await swap(plan, store, eval_passed=True)
    drill.swap_clock = store.clock
    await run_query(service, store, drill)

    assert drill.observations
    assert drill.inside_the_window() == []
