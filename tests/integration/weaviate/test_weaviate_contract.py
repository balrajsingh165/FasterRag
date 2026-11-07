"""The shared adapter contract suite, run against a real Weaviate.

The same ``VectorDBContract`` the Qdrant, pgvector, Chroma, and Milvus suites run,
subclassed rather than copied. Weaviate is the fifth paradigm the contract has to survive,
and the first backend that already ships a *competing* answer to the question the contract
asks: its own BM25, its own hybrid fusion, and its own tokenizer, none of which this
adapter uses.

The classes below the contract cover what the shared suite does not, for the same reason
the pgvector, Chroma, and Milvus packages have their own: the collection-name and object-id
derivations are the only thing mapping fasterRag names onto legal Weaviate ones, snapshots
are a collection copy rather than a server backup, payloads are split between flattened
properties and a JSON remainder, and the keyword leg is scored here rather than by the
server.

**None of this has run.** There is no Weaviate on the machine the adapter was built on and
no Docker daemon to start one, so every case here skips (B7.2); see this package's
``conftest`` for how to point it at one. Until it does, the unit suite's fake proves that
the adapter issues the calls it means to and nothing about what Weaviate does with them —
and the cases most worth running first are the ones below, because each pins a belief about
the server rather than about the adapter.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator

import pytest

from fasterrag.adapters.vectordb.base import (
    CollectionSpec,
    Point,
    PointSelector,
    SearchQuery,
    VectorDBAdapter,
)
from fasterrag.adapters.vectordb.weaviate import (
    ALIAS_COLLECTION,
    PK_PROPERTY,
    WeaviateAdapter,
    collection_name_for,
    point_uuid_for,
)
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.bm25 import encode_document, encode_query
from fasterrag.errors import ErrorCode, FasterRagError
from tests.contract.vectordb import DIMENSIONS, VECTORS, VectorDBContract
from tests.integration.weaviate.conftest import WRONG_KEY_VAR, weaviate_settings

pytestmark = pytest.mark.integration

WEAVIATE_NAME = re.compile(r"^[A-Z][_0-9A-Za-z]*$")


class WeaviateFixtures:
    """Adapter and collection wiring shared by the contract run and the extra cases."""

    @pytest.fixture
    async def adapter(self, weaviate: Settings) -> AsyncIterator[VectorDBAdapter]:
        built = WeaviateAdapter(weaviate)
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


class TestWeaviateContract(WeaviateFixtures, VectorDBContract):
    """Every behaviour the shared contract requires, against a live Weaviate."""

    @pytest.fixture
    async def misconfigured_adapter(
        self, weaviate: Settings, weaviate_endpoints: tuple[str, str]
    ) -> AsyncIterator[VectorDBAdapter | None]:
        """A key the server will reject, so the auth case is exercised, not skipped.

        It depends on ``weaviate`` only to inherit that fixture's skip: without a server
        every connection fails, and the case would pass on the connection error instead.
        """
        http, grpc = weaviate_endpoints
        built = WeaviateAdapter(weaviate_settings(http, grpc, key_env=WRONG_KEY_VAR))
        yield built
        await built.close()


class TestWeaviateNames(WeaviateFixtures):
    """A legal fasterRag collection name is not necessarily a legal Weaviate one."""

    async def test_a_hyphenated_name_round_trips(
        self, adapter: VectorDBAdapter, collection_name: str
    ) -> None:
        """Weaviate collection names forbid ``-``; the listing must still show the real one."""
        assert "-" in collection_name
        await adapter.create_collection(CollectionSpec(name=collection_name, dimensions=DIMENSIONS))
        try:
            listed = {info.name for info in await adapter.list_collections()}
            assert collection_name in listed
            assert WEAVIATE_NAME.match(collection_name_for(collection_name))
        finally:
            await adapter.drop_collection(collection_name)

    async def test_a_name_starting_with_a_digit_is_usable(self, adapter: VectorDBAdapter) -> None:
        """Weaviate names must start with a capital letter; fasterRag names need not."""
        name = "2026-archive"
        await adapter.create_collection(CollectionSpec(name=name, dimensions=DIMENSIONS))
        try:
            assert name in {info.name for info in await adapter.list_collections()}
        finally:
            await adapter.drop_collection(name)

    async def test_a_chunk_id_survives_the_uuid_derivation(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        """A Weaviate object id is a UUID and a chunk id is not, so one is derived."""
        await adapter.upsert(
            [Point(point_id="c_abc-123", collection=collection, vector=VECTORS["a"])]
        )

        walked = [found async for found in adapter.iterate_points(collection)]

        assert [found.point_id for found in walked] == ["c_abc-123"]
        assert point_uuid_for("c_abc-123") != "c_abc-123"

    async def test_the_alias_collection_never_appears_in_the_listing(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await adapter.set_alias(f"{collection}-alias", collection)
        try:
            listed = {info.name for info in await adapter.list_collections()}
            assert ALIAS_COLLECTION not in listed
        finally:
            await adapter.delete_alias(f"{collection}-alias")


class TestWeaviatePayloads(WeaviateFixtures):
    """A payload is split across a flattened half and a JSON remainder; both must survive.

    This is the class most worth running first. Auto-schema inferring a usable type for a
    flattened property is a *server* behaviour, and the unit suite's fake cannot see it
    fail.
    """

    async def test_a_nested_payload_round_trips(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        payload = {
            "department": "legal",
            "year": 2024,
            "reviewed": True,
            "span": {"start": 0, "end": 9},
            "tags": ["a", "b"],
            "section": None,
            "source-uri": "file:///a",
        }
        await adapter.upsert(
            [
                Point(
                    point_id="c_a",
                    collection=collection,
                    vector=VECTORS["a"],
                    payload=payload,
                )
            ]
        )

        walked = [found async for found in adapter.iterate_points(collection)]

        assert dict(walked[0].payload) == payload

    async def test_a_filter_on_a_key_the_remainder_holds_is_refused_not_silent(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        """Such a key is not a Weaviate property, so a pushed-down filter would match none."""
        await adapter.upsert(
            [
                Point(
                    point_id="c_a",
                    collection=collection,
                    vector=VECTORS["a"],
                    payload={"source-uri": "file:///a"},
                )
            ]
        )

        with pytest.raises(FasterRagError) as caught:
            await adapter.search(
                SearchQuery(
                    collection=collection,
                    vector=VECTORS["a"],
                    filters={"source-uri": "file:///a"},
                )
            )

        assert caught.value.code is ErrorCode.VALIDATION_FAILED

    async def test_a_payload_key_colliding_with_a_reserved_property_is_not_a_collision(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    point_id="c_a",
                    collection=collection,
                    vector=VECTORS["a"],
                    payload={PK_PROPERTY: "spoofed"},
                )
            ]
        )

        walked = [found async for found in adapter.iterate_points(collection)]

        assert walked[0].point_id == "c_a"
        assert dict(walked[0].payload) == {PK_PROPERTY: "spoofed"}


class TestWeaviateWrites(WeaviateFixtures):
    """Whether the batch endpoint overwrites an existing id is the server's business.

    The adapter falls back to ``replace`` on any per-object batch failure so that upserts
    are idempotent either way. That fallback has never run against a server, and this is
    where it would be proved.
    """

    async def test_rewriting_a_point_leaves_one_object_with_the_new_payload(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    point_id="c_a",
                    collection=collection,
                    vector=VECTORS["a"],
                    payload={"version": 1},
                )
            ]
        )
        await adapter.upsert(
            [
                Point(
                    point_id="c_a",
                    collection=collection,
                    vector=VECTORS["b"],
                    payload={"version": 2},
                )
            ]
        )

        walked = [found async for found in adapter.iterate_points(collection, with_vectors=True)]

        assert len(walked) == 1
        assert walked[0].payload["version"] == 2
        assert list(walked[0].vector) == VECTORS["b"]

    async def test_a_large_batch_crosses_the_page_boundary(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        """The write batch is 200, so 250 points prove the paging rather than assume it."""
        await adapter.upsert(
            [
                Point(point_id=f"c_{index}", collection=collection, vector=VECTORS["a"])
                for index in range(250)
            ]
        )

        walked = [found async for found in adapter.iterate_points(collection)]

        assert len(walked) == 250


class TestWeaviateSnapshots(WeaviateFixtures):
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
        """A snapshot is a Weaviate collection, and must not leak into the listing."""
        taken = await adapter.snapshot(collection)
        try:
            assert taken not in {info.name for info in await adapter.list_collections()}
        finally:
            await adapter.delete_snapshot(collection, taken)

    async def test_a_snapshot_carries_the_keyword_leg(
        self, adapter: VectorDBAdapter, hybrid_collection: str
    ) -> None:
        """A copy that drops the postings restores a collection that answers differently."""
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

        taken = await adapter.snapshot(hybrid_collection)
        await adapter.delete(PointSelector(collection=hybrid_collection, point_ids=["c_notice"]))
        await adapter.restore_snapshot(hybrid_collection, taken)

        hits = await adapter.search(
            SearchQuery(collection=hybrid_collection, sparse=encode_query("terminate"), limit=5)
        )
        assert [hit.point_id for hit in hits] == ["c_notice"]
        await adapter.delete_snapshot(hybrid_collection, taken)


class TestWeaviateKeywordLeg(WeaviateFixtures):
    """IDF is computed here from a server-filtered candidate set, which is the claim to test.

    Whether ``ContainsAny`` over an ``INT_ARRAY`` selects the documents this adapter thinks
    it selects is a server behaviour. If it does not, every case here goes red and the fake
    stays green, which is exactly why these exist.
    """

    async def test_a_rarer_term_outranks_a_common_one(
        self, adapter: VectorDBAdapter, hybrid_collection: str
    ) -> None:
        """Without IDF these would tie; the ranking is the whole proof it is applied."""
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

    async def test_the_candidate_walk_crosses_the_page_boundary(
        self, adapter: VectorDBAdapter, hybrid_collection: str
    ) -> None:
        """Candidate paging is unbounded by design; 600 matches prove it pages."""
        await adapter.upsert(
            [
                Point(
                    point_id=f"c_{index}",
                    collection=hybrid_collection,
                    vector=VECTORS["a"],
                    sparse=encode_document("termination notice period"),
                )
                for index in range(600)
            ]
        )

        hits = await adapter.search(
            SearchQuery(collection=hybrid_collection, sparse=encode_query("termination"), limit=600)
        )

        assert len(hits) == 600

    async def test_a_keyword_search_on_a_dense_collection_is_refused(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.search(
                SearchQuery(collection=collection, sparse=encode_query("notice"), limit=5)
            )

        assert caught.value.code is ErrorCode.VALIDATION_FAILED
