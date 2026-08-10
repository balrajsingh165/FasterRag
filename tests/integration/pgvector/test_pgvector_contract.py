"""The shared adapter contract suite, run against a real PostgreSQL with pgvector.

The same ``VectorDBContract`` the Qdrant suite runs, subclassed rather than copied: two
copies of a contract drift, and the point of this adapter is to prove the contract holds
against a paradigm it was not written for.

The cases below the contract class cover what the shared suite does not, because Qdrant
and PostgreSQL genuinely differ there — snapshots are a table copy rather than a backend
file, and ``iterate_points`` is keyset pagination rather than a scroll cursor. They live
here, not in the shared suite, until they can be verified against Qdrant too.
"""

from collections.abc import AsyncIterator

import pytest

from fasterrag.adapters.vectordb.base import (
    CollectionSpec,
    Point,
    PointSelector,
    SearchQuery,
    VectorDBAdapter,
)
from fasterrag.adapters.vectordb.pgvector import PgvectorAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.bm25 import encode_document, encode_query
from fasterrag.errors import ErrorCode, FasterRagError
from tests.contract.vectordb import DIMENSIONS, VECTORS, VectorDBContract, point
from tests.integration.pgvector.conftest import WRONG_DSN_VAR

pytestmark = pytest.mark.integration


class PgvectorFixtures:
    """Adapter and collection wiring shared by the contract run and the extra cases."""

    @pytest.fixture
    async def adapter(self, pgvector: Settings) -> AsyncIterator[VectorDBAdapter]:
        built = PgvectorAdapter(pgvector)
        yield built
        await built.close()

    @pytest.fixture
    async def collection(
        self, adapter: VectorDBAdapter, collection_name: str
    ) -> AsyncIterator[str]:
        await adapter.create_collection(CollectionSpec(name=collection_name, dimensions=DIMENSIONS))
        yield collection_name
        await adapter.drop_collection(collection_name)

    @pytest.fixture
    async def hybrid_collection(
        self, adapter: VectorDBAdapter, collection_name: str
    ) -> AsyncIterator[str]:
        name = f"{collection_name}-hybrid"
        await adapter.create_collection(
            CollectionSpec(name=name, dimensions=DIMENSIONS, sparse=True)
        )
        yield name
        await adapter.drop_collection(name)


class TestPgvectorContract(PgvectorFixtures, VectorDBContract):
    """Every behaviour the shared contract requires, over SQL."""

    @pytest.fixture
    async def misconfigured_adapter(self, pgvector: Settings) -> AsyncIterator[VectorDBAdapter]:
        wrong = Settings.model_validate(
            {
                "vector_db": {
                    "provider": "pgvector",
                    "mode": "external",
                    "pgvector": {"dsn_env": WRONG_DSN_VAR},
                }
            }
        )
        built = PgvectorAdapter(wrong)
        yield built
        await built.close()


class TestPgvectorSnapshots(PgvectorFixtures):
    """Snapshots as a same-database table copy, with the limits that implies."""

    async def seed(self, adapter: VectorDBAdapter, collection: str) -> None:
        await adapter.upsert(
            [
                point("c_a", collection, VECTORS["a"], department="legal"),
                point("c_b", collection, VECTORS["b"], department="legal"),
            ]
        )

    async def test_a_snapshot_is_listed_after_it_is_taken(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await self.seed(adapter, collection)

        name = await adapter.snapshot(collection)

        assert await adapter.list_snapshots(collection) == [name]

    async def test_restoring_undoes_writes_made_after_the_snapshot(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await self.seed(adapter, collection)
        name = await adapter.snapshot(collection)
        await adapter.delete(PointSelector(collection=collection, point_ids=["c_a"]))

        await adapter.restore_snapshot(collection, name)

        hits = await adapter.search(
            SearchQuery(collection=collection, vector=VECTORS["a"], limit=10)
        )
        assert {hit.point_id for hit in hits} == {"c_a", "c_b"}

    async def test_restoring_removes_writes_made_after_the_snapshot(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await self.seed(adapter, collection)
        name = await adapter.snapshot(collection)
        await adapter.upsert([point("c_c", collection, VECTORS["c"], department="finance")])

        await adapter.restore_snapshot(collection, name)

        listed = {info.name: info for info in await adapter.list_collections()}
        assert listed[collection].vectors == 2

    async def test_deleting_a_snapshot_reports_whether_one_was_there(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await self.seed(adapter, collection)
        name = await adapter.snapshot(collection)

        assert await adapter.delete_snapshot(collection, name) is True
        assert await adapter.delete_snapshot(collection, name) is False
        assert await adapter.list_snapshots(collection) == []

    async def test_restoring_an_unknown_snapshot_is_a_typed_error(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.restore_snapshot(collection, "snapshot-that-never-existed")

        assert caught.value.code is ErrorCode.NOT_FOUND

    async def test_dropping_a_collection_takes_its_snapshots_with_it(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await self.seed(adapter, collection)
        await adapter.snapshot(collection)

        assert await adapter.drop_collection(collection) is True

        await adapter.create_collection(CollectionSpec(name=collection, dimensions=DIMENSIONS))
        assert await adapter.list_snapshots(collection) == []

    async def test_a_hybrid_collection_snapshots_its_postings_too(
        self, adapter: VectorDBAdapter, hybrid_collection: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    point_id="c_notice",
                    collection=hybrid_collection,
                    vector=VECTORS["a"],
                    sparse=encode_document("Either party may terminate with written notice."),
                )
            ]
        )
        name = await adapter.snapshot(hybrid_collection)
        await adapter.delete(PointSelector(collection=hybrid_collection, point_ids=["c_notice"]))

        await adapter.restore_snapshot(hybrid_collection, name)

        hits = await adapter.search(
            SearchQuery(collection=hybrid_collection, sparse=encode_query("terminate"), limit=5)
        )
        assert [hit.point_id for hit in hits] == ["c_notice"]


class TestPgvectorIteration(PgvectorFixtures):
    """``iterate_points`` walks a live collection — the read side of portability (D11)."""

    async def test_every_point_is_yielded_once(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await adapter.upsert(
            [point(f"c_{index}", collection, VECTORS["a"], index=index) for index in range(5)]
        )

        walked = [found async for found in adapter.iterate_points(collection)]

        assert sorted(found.point_id for found in walked) == [f"c_{index}" for index in range(5)]
        assert {found.payload["index"] for found in walked} == set(range(5))

    async def test_the_walk_spans_pages(self, adapter: VectorDBAdapter, collection: str) -> None:
        await adapter.upsert([point(f"c_{index}", collection, VECTORS["a"]) for index in range(7)])

        walked = [found async for found in adapter.iterate_points(collection, batch_size=2)]

        assert len(walked) == 7

    async def test_vectors_come_back_when_asked_for(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await adapter.upsert([point("c_a", collection, VECTORS["a"])])

        walked = [found async for found in adapter.iterate_points(collection, with_vectors=True)]

        assert len(walked[0].vector) == DIMENSIONS

    async def test_an_empty_collection_yields_nothing(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        walked = [found async for found in adapter.iterate_points(collection)]

        assert walked == []
