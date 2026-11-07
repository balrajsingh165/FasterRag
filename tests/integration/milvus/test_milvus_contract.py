"""The shared adapter contract suite, run against a real Milvus.

The same ``VectorDBContract`` the Qdrant, pgvector, and Chroma suites run, subclassed
rather than copied. Milvus is the fourth paradigm the contract has to survive — a
declared schema, a load step before any read, and a default consistency level under
which a point just written is not necessarily visible.

The classes below the contract cover what the shared suite does not, for the same reason
the pgvector and Chroma packages have their own: snapshots are a collection copy rather
than a backend file, the keyword leg's IDF is assembled from server-side counts rather
than taken from a scoring primitive, and the derived collection names are the only thing
mapping a fasterRag collection onto a legal Milvus identifier.

**None of this has run.** There is no Milvus on the machine the adapter was built on, so
every case here skips (B7.2); see this package's ``conftest`` for how to point it at one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from fasterrag.adapters.vectordb.base import (
    CollectionSpec,
    Point,
    PointSelector,
    SearchQuery,
    SparseVector,
    VectorDBAdapter,
)
from fasterrag.adapters.vectordb.milvus import (
    MAXIMUM_TERMS,
    MilvusAdapter,
    collection_name_for,
)
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.bm25 import encode_document, encode_query
from fasterrag.errors import EmbedError, ErrorCode, FasterRagError
from tests.contract.vectordb import DIMENSIONS, VECTORS, VectorDBContract
from tests.integration.milvus.conftest import WRONG_TOKEN_VAR, milvus_settings

pytestmark = pytest.mark.integration


class MilvusFixtures:
    """Adapter and collection wiring shared by the contract run and the extra cases."""

    @pytest.fixture
    async def adapter(self, milvus: Settings) -> AsyncIterator[VectorDBAdapter]:
        built = MilvusAdapter(milvus)
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


class TestMilvusContract(MilvusFixtures, VectorDBContract):
    """Every behaviour the shared contract requires, against a live Milvus."""

    @pytest.fixture
    async def misconfigured_adapter(
        self, milvus: Settings, milvus_uri: str
    ) -> AsyncIterator[VectorDBAdapter | None]:
        """A token the server will reject, so the auth case is exercised, not skipped.

        It depends on ``milvus`` only to inherit that fixture's skip: without a server
        every connection fails, and the case would pass on the connection error instead.
        """
        built = MilvusAdapter(milvus_settings(milvus_uri, token_env=WRONG_TOKEN_VAR))
        yield built
        await built.close()


class TestMilvusNames(MilvusFixtures):
    """A legal fasterRag collection name is not necessarily a legal Milvus identifier."""

    async def test_a_hyphenated_name_round_trips(
        self, adapter: VectorDBAdapter, collection_name: str
    ) -> None:
        """Milvus identifiers forbid ``-``; the listing must still show the real name."""
        assert "-" in collection_name
        await adapter.create_collection(CollectionSpec(name=collection_name, dimensions=DIMENSIONS))
        try:
            listed = {info.name for info in await adapter.list_collections()}
            assert collection_name in listed
            assert "-" not in collection_name_for(collection_name)
        finally:
            await adapter.drop_collection(collection_name)

    async def test_a_name_starting_with_a_digit_is_usable(self, adapter: VectorDBAdapter) -> None:
        """Milvus identifiers must start with a letter or underscore; ours need not."""
        name = "2026-archive"
        await adapter.create_collection(CollectionSpec(name=name, dimensions=DIMENSIONS))
        try:
            assert name in {info.name for info in await adapter.list_collections()}
        finally:
            await adapter.drop_collection(name)


class TestMilvusSnapshots(MilvusFixtures):
    """Snapshots are a collection copy, so their contents are this adapter's problem."""

    async def test_a_snapshot_holds_what_the_collection_held(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await adapter.upsert(
            [
                Point(point_id="c_a", collection=collection, vector=VECTORS["a"], payload={"n": 1}),
                Point(point_id="c_b", collection=collection, vector=VECTORS["b"], payload={"n": 2}),
            ]
        )

        taken = await adapter.snapshot(collection)
        await adapter.delete(PointSelector(collection=collection, point_ids=["c_a", "c_b"]))
        await adapter.restore_snapshot(collection, taken)

        hits = await adapter.search(
            SearchQuery(collection=collection, vector=VECTORS["a"], limit=10)
        )
        assert {hit.point_id for hit in hits} == {"c_a", "c_b"}
        assert hits[0].payload["n"] == 1
        assert await adapter.delete_snapshot(collection, taken) is True

    async def test_a_snapshot_is_listed_and_then_is_not(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        taken = await adapter.snapshot(collection)

        assert taken in await adapter.list_snapshots(collection)
        assert await adapter.delete_snapshot(collection, taken) is True
        assert taken not in await adapter.list_snapshots(collection)
        assert await adapter.delete_snapshot(collection, taken) is False

    async def test_a_snapshot_is_not_itself_a_collection(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        """A snapshot is a Milvus collection, and must not leak into the listing."""
        taken = await adapter.snapshot(collection)
        try:
            assert taken not in {info.name for info in await adapter.list_collections()}
        finally:
            await adapter.delete_snapshot(collection, taken)


class TestMilvusKeywordLeg(MilvusFixtures):
    """IDF is assembled from server-side counts, which is the claim worth testing live."""

    async def test_a_rarer_term_outranks_a_common_one(
        self, adapter: VectorDBAdapter, hybrid_collection: str
    ) -> None:
        """Without IDF these two would tie; the ranking is the whole proof it applies."""
        await adapter.upsert(
            [
                Point(
                    point_id=f"c_common_{index}",
                    collection=hybrid_collection,
                    vector=VECTORS["a"],
                    sparse=encode_document("notice notice notice"),
                )
                for index in range(5)
            ]
            + [
                Point(
                    point_id="c_rare",
                    collection=hybrid_collection,
                    vector=VECTORS["b"],
                    sparse=encode_document("notice indemnification"),
                )
            ]
        )

        hits = await adapter.search(
            SearchQuery(
                collection=hybrid_collection,
                sparse=encode_query("notice indemnification"),
                limit=10,
            )
        )

        assert hits[0].point_id == "c_rare"

    async def test_a_term_absent_from_the_corpus_scores_nothing(
        self, adapter: VectorDBAdapter, hybrid_collection: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    point_id="c_notice",
                    collection=hybrid_collection,
                    vector=VECTORS["a"],
                    sparse=encode_document("termination notice period"),
                )
            ]
        )

        hits = await adapter.search(
            SearchQuery(collection=hybrid_collection, sparse=encode_query("velociraptor"), limit=5)
        )

        assert hits == []

    async def test_too_many_terms_is_refused_rather_than_truncated(
        self, adapter: VectorDBAdapter, hybrid_collection: str
    ) -> None:
        """A Milvus array has a capacity; a silently truncated posting list scores wrong."""
        oversized = SparseVector(
            indices=list(range(MAXIMUM_TERMS + 1)),
            values=[1.0] * (MAXIMUM_TERMS + 1),
        )

        with pytest.raises(EmbedError) as caught:
            await adapter.upsert(
                [
                    Point(
                        point_id="c_huge",
                        collection=hybrid_collection,
                        vector=VECTORS["a"],
                        sparse=oversized,
                    )
                ]
            )

        assert caught.value.retryable is False
        assert str(MAXIMUM_TERMS) in caught.value.detail

    async def test_a_keyword_search_on_a_dense_collection_is_refused(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.search(
                SearchQuery(collection=collection, sparse=encode_query("notice"), limit=5)
            )

        assert caught.value.code is ErrorCode.VALIDATION_FAILED
