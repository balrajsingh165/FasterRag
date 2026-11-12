"""The shared vector database adapter contract suite.

One suite that every ``VectorDBAdapter`` implementation must pass, built-in or
registered by a third party through the ``fasterrag.vectordb`` entry point. This is what
turns "any vector database" into a tested promise instead of a hope
(``docs/testing-strategy.md`` §1.5).

To run it against an adapter, subclass :class:`VectorDBContract` and supply two
fixtures:

* ``adapter`` — a connected adapter under test.
* ``collection`` — the name of a freshly created, empty collection, cleaned up
  afterwards. Creating and dropping collections is vendor-specific setup, so it lives
  in the subclass and keeps this suite free of any vendor import.
* ``hybrid_collection`` — the same, but created with a sparse index so the keyword leg
  can be exercised.

Optionally override ``misconfigured_adapter`` to return an adapter holding invalid
credentials; the authentication case is skipped explicitly when it is not supplied,
never silently passed.
"""

from __future__ import annotations

import pytest

from fasterrag.adapters.vectordb.base import (
    CollectionSpec,
    Point,
    PointSelector,
    PointUpdate,
    SearchQuery,
    VectorDBAdapter,
)
from fasterrag.core.retrieval.bm25 import encode_document, encode_query
from fasterrag.errors import EmbedError, ErrorCode, FasterRagError

DIMENSIONS = 4

VECTORS: dict[str, list[float]] = {
    "a": [1.0, 0.0, 0.0, 0.0],
    "b": [0.0, 1.0, 0.0, 0.0],
    "c": [0.0, 0.0, 1.0, 0.0],
}


def point(point_id: str, collection: str, vector: list[float], **payload: object) -> Point:
    """Build a point for the contract fixtures."""
    return Point(point_id=point_id, collection=collection, vector=vector, payload=payload)


class VectorDBContract:
    """Behavior every vector database adapter must exhibit."""

    @pytest.fixture
    def misconfigured_adapter(self) -> VectorDBAdapter | None:
        """Return an adapter with invalid credentials, or None to skip that case."""
        return None

    async def seed(self, adapter: VectorDBAdapter, collection: str) -> None:
        """Write the standard three points used by the read-side cases."""
        await adapter.upsert(
            [
                point("c_a", collection, VECTORS["a"], department="legal", year=2024),
                point("c_b", collection, VECTORS["b"], department="legal", year=2020),
                point("c_c", collection, VECTORS["c"], department="finance", year=2024),
            ]
        )

    async def test_health_reports_a_reachable_backend(self, adapter: VectorDBAdapter) -> None:
        status = await adapter.health()

        assert status.healthy is True

    async def test_creating_an_existing_collection_is_idempotent(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        spec = CollectionSpec(name=collection, dimensions=DIMENSIONS)

        await adapter.create_collection(spec)
        await adapter.create_collection(spec)

    async def test_a_created_collection_appears_in_the_listing(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await adapter.create_collection(CollectionSpec(name=collection, dimensions=DIMENSIONS))

        listed = {info.name: info for info in await adapter.list_collections()}

        assert collection in listed
        assert listed[collection].dimensions == DIMENSIONS

    async def test_the_listing_counts_what_was_written(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await adapter.create_collection(CollectionSpec(name=collection, dimensions=DIMENSIONS))
        await self.seed(adapter, collection)

        listed = {info.name: info for info in await adapter.list_collections()}

        assert listed[collection].vectors == 3

    async def test_dropping_a_collection_removes_it(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await adapter.create_collection(CollectionSpec(name=collection, dimensions=DIMENSIONS))

        assert await adapter.drop_collection(collection) is True
        assert collection not in {info.name for info in await adapter.list_collections()}

    async def test_dropping_an_absent_collection_is_not_an_error(
        self, adapter: VectorDBAdapter
    ) -> None:
        assert await adapter.drop_collection("collection-that-never-existed") is False

    async def test_an_alias_resolves_to_its_collection(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await adapter.create_collection(CollectionSpec(name=collection, dimensions=DIMENSIONS))
        alias = f"{collection}-alias"

        await adapter.set_alias(alias, collection)

        assert await adapter.alias_target(alias) == collection
        await adapter.delete_alias(alias)

    async def test_an_unset_alias_resolves_to_nothing(self, adapter: VectorDBAdapter) -> None:
        assert await adapter.alias_target("alias-that-never-existed") is None

    async def test_repointing_an_alias_replaces_its_target(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        second = f"{collection}-green"
        await adapter.create_collection(CollectionSpec(name=collection, dimensions=DIMENSIONS))
        await adapter.create_collection(CollectionSpec(name=second, dimensions=DIMENSIONS))
        alias = f"{collection}-alias"

        await adapter.set_alias(alias, collection)
        await adapter.set_alias(alias, second)

        assert await adapter.alias_target(alias) == second
        await adapter.delete_alias(alias)
        await adapter.drop_collection(second)

    async def test_deleting_an_alias_leaves_the_collection(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await adapter.create_collection(CollectionSpec(name=collection, dimensions=DIMENSIONS))
        alias = f"{collection}-alias"
        await adapter.set_alias(alias, collection)

        assert await adapter.delete_alias(alias) is True
        assert await adapter.alias_target(alias) is None
        assert collection in {info.name for info in await adapter.list_collections()}

    async def test_deleting_an_absent_alias_is_not_an_error(self, adapter: VectorDBAdapter) -> None:
        assert await adapter.delete_alias("alias-that-never-existed") is False

    async def test_an_alias_is_searchable_as_if_it_were_the_collection(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await adapter.create_collection(CollectionSpec(name=collection, dimensions=DIMENSIONS))
        await self.seed(adapter, collection)
        alias = f"{collection}-alias"
        await adapter.set_alias(alias, collection)

        hits = await adapter.search(SearchQuery(collection=alias, vector=VECTORS["a"], limit=3))

        assert hits
        await adapter.delete_alias(alias)

    async def test_conflicting_dimensions_are_refused(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.create_collection(
                CollectionSpec(name=collection, dimensions=DIMENSIONS + 1)
            )

        assert caught.value.code is ErrorCode.CONFLICT

    async def test_a_keyword_leg_the_collection_lacks_is_refused(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        """Accepting it deferred the failure to the first upsert, far from its cause."""
        with pytest.raises(FasterRagError) as caught:
            await adapter.create_collection(
                CollectionSpec(name=collection, dimensions=DIMENSIONS, sparse=True)
            )

        assert caught.value.code is ErrorCode.CONFLICT

    async def test_upserted_points_are_searchable_by_their_own_ids(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await self.seed(adapter, collection)

        hits = await adapter.search(
            SearchQuery(collection=collection, vector=VECTORS["a"], limit=3)
        )

        assert hits[0].point_id == "c_a"
        assert hits[0].payload["department"] == "legal"

    async def test_a_closer_point_scores_higher_than_a_further_one(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        """Under the default cosine distance, ``score`` must rank higher-is-better."""
        await self.seed(adapter, collection)

        hits = await adapter.search(
            SearchQuery(collection=collection, vector=VECTORS["a"], limit=3)
        )

        assert hits[0].point_id == "c_a"
        assert hits[0].score > hits[-1].score
        assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)

    async def test_upsert_is_idempotent(self, adapter: VectorDBAdapter, collection: str) -> None:
        await self.seed(adapter, collection)
        await self.seed(adapter, collection)

        hits = await adapter.search(
            SearchQuery(collection=collection, vector=VECTORS["a"], limit=10)
        )

        assert len(hits) == 3
        assert len({hit.point_id for hit in hits}) == 3

    async def test_upsert_reports_how_many_points_it_wrote(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        result = await adapter.upsert(
            [point(f"c_{index}", collection, VECTORS["a"]) for index in range(5)]
        )

        assert result.upserted == 5

    async def test_search_respects_the_limit(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await self.seed(adapter, collection)

        hits = await adapter.search(
            SearchQuery(collection=collection, vector=VECTORS["a"], limit=2)
        )

        assert len(hits) == 2

    async def test_equality_filters_are_pushed_down(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await self.seed(adapter, collection)

        hits = await adapter.search(
            SearchQuery(
                collection=collection,
                vector=VECTORS["a"],
                limit=10,
                filters={"department": "finance"},
            )
        )

        assert [hit.point_id for hit in hits] == ["c_c"]

    async def test_range_filters_are_pushed_down(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await self.seed(adapter, collection)

        hits = await adapter.search(
            SearchQuery(
                collection=collection,
                vector=VECTORS["a"],
                limit=10,
                filters={"year": {"$gte": 2024}},
            )
        )

        assert {hit.point_id for hit in hits} == {"c_a", "c_c"}

    async def test_set_membership_filters_are_pushed_down(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await self.seed(adapter, collection)

        hits = await adapter.search(
            SearchQuery(
                collection=collection,
                vector=VECTORS["a"],
                limit=10,
                filters={"department": {"$in": ["finance", "hr"]}},
            )
        )

        assert {hit.point_id for hit in hits} == {"c_c"}

    async def test_vectors_are_returned_on_request(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await self.seed(adapter, collection)

        hits = await adapter.search(
            SearchQuery(collection=collection, vector=VECTORS["a"], limit=1, with_vectors=True)
        )

        assert hits[0].vector is not None
        assert len(hits[0].vector) == DIMENSIONS

    async def test_updates_merge_metadata_without_touching_vectors(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await self.seed(adapter, collection)

        await adapter.update(
            [PointUpdate(point_id="c_a", collection=collection, payload={"reviewed": True})]
        )
        hits = await adapter.search(
            SearchQuery(collection=collection, vector=VECTORS["a"], limit=1)
        )

        assert hits[0].point_id == "c_a"
        assert hits[0].payload["reviewed"] is True
        assert hits[0].payload["department"] == "legal"

    async def test_delete_by_ids_removes_only_those_points(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await self.seed(adapter, collection)

        await adapter.delete(PointSelector(collection=collection, point_ids=["c_a"]))
        hits = await adapter.search(
            SearchQuery(collection=collection, vector=VECTORS["a"], limit=10)
        )

        assert {hit.point_id for hit in hits} == {"c_b", "c_c"}

    async def test_delete_by_filter_removes_matching_points(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await self.seed(adapter, collection)

        await adapter.delete(PointSelector(collection=collection, filters={"department": "legal"}))
        hits = await adapter.search(
            SearchQuery(collection=collection, vector=VECTORS["a"], limit=10)
        )

        assert {hit.point_id for hit in hits} == {"c_c"}

    async def seed_documents(self, adapter: VectorDBAdapter, collection: str) -> None:
        """Write two documents' chunks carrying the payload keys ``workers.indexer`` writes."""
        await adapter.upsert(
            [
                point("d1_0", collection, VECTORS["a"], document_id="d_one", chunk_index=0),
                point("d1_1", collection, VECTORS["b"], document_id="d_one", chunk_index=1),
                point("d1_2", collection, VECTORS["c"], document_id="d_one", chunk_index=2),
                point("d2_0", collection, VECTORS["a"], document_id="d_two", chunk_index=0),
                point("d2_1", collection, VECTORS["b"], document_id="d_two", chunk_index=1),
            ]
        )

    async def remaining(self, adapter: VectorDBAdapter, collection: str) -> set[str]:
        """Return the ids of every point a collection still holds."""
        hits = await adapter.search(
            SearchQuery(collection=collection, vector=VECTORS["a"], limit=10)
        )
        return {hit.point_id for hit in hits}

    async def test_delete_by_document_id_removes_every_chunk_of_that_document_and_no_other(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        """What deleting a document rests on (TASK-0443): one filter, the whole document."""
        await self.seed_documents(adapter, collection)

        await adapter.delete(PointSelector(collection=collection, filters={"document_id": "d_one"}))

        assert await self.remaining(adapter, collection) == {"d2_0", "d2_1"}

    async def test_delete_by_document_id_and_chunk_index_removes_only_that_documents_tail(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        """What trimming a shrunken document rests on: equality and a range, conjoined."""
        await self.seed_documents(adapter, collection)

        await adapter.delete(
            PointSelector(
                collection=collection,
                filters={"document_id": "d_one", "chunk_index": {"$gte": 1}},
            )
        )

        assert await self.remaining(adapter, collection) == {"d1_0", "d2_0", "d2_1"}

    async def test_delete_by_document_id_within_a_tenant_spares_another_tenants_chunks(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        """Two equalities conjoined, and a point without the second key never matches."""
        await adapter.upsert(
            [
                point("t_a", collection, VECTORS["a"], document_id="d_one", tenant="acme"),
                point("t_b", collection, VECTORS["b"], document_id="d_one", tenant="globex"),
                point("t_c", collection, VECTORS["c"], document_id="d_one"),
            ]
        )

        await adapter.delete(
            PointSelector(collection=collection, filters={"document_id": "d_one", "tenant": "acme"})
        )

        assert await self.remaining(adapter, collection) == {"t_b", "t_c"}

    async def test_deleting_a_document_nothing_holds_is_not_an_error(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        """A delete has to be repeatable, so a filter that matches nothing must not raise."""
        await self.seed_documents(adapter, collection)

        await adapter.delete(
            PointSelector(collection=collection, filters={"document_id": "d_absent"})
        )

        assert len(await self.remaining(adapter, collection)) == 5

    async def test_deleting_from_an_unknown_collection_is_not_found(
        self, adapter: VectorDBAdapter
    ) -> None:
        """The code a caller branches on to answer 404 rather than a retryable 503."""
        with pytest.raises(FasterRagError) as caught:
            await adapter.delete(
                PointSelector(
                    collection="collection-that-does-not-exist", filters={"document_id": "d_one"}
                )
            )

        assert caught.value.code is ErrorCode.NOT_FOUND

    async def test_dimension_mismatch_is_rejected(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        with pytest.raises(EmbedError) as caught:
            await adapter.upsert([point("c_bad", collection, [1.0, 0.0])])

        assert caught.value.retryable is False
        assert str(DIMENSIONS) in caught.value.detail

    async def test_an_unknown_collection_raises_a_typed_error(
        self, adapter: VectorDBAdapter
    ) -> None:
        with pytest.raises(FasterRagError):
            await adapter.search(
                SearchQuery(collection="collection-that-does-not-exist", vector=VECTORS["a"])
            )

    async def test_searching_an_unknown_collection_is_not_found(
        self, adapter: VectorDBAdapter
    ) -> None:
        """Any typed error passed; a query edge with one defined code needs this one (TASK-0468)."""
        with pytest.raises(FasterRagError) as caught:
            await adapter.search(
                SearchQuery(collection="collection-that-does-not-exist", vector=VECTORS["a"])
            )

        assert caught.value.code is ErrorCode.NOT_FOUND
        assert caught.value.retryable is False

    async def test_a_keyword_search_of_an_unknown_collection_is_not_found(
        self, adapter: VectorDBAdapter
    ) -> None:
        """The keyword leg reaches the backend by its own path and must say the same."""
        with pytest.raises(FasterRagError) as caught:
            await adapter.search(
                SearchQuery(
                    collection="collection-that-does-not-exist",
                    sparse=encode_query("termination clause"),
                )
            )

        assert caught.value.code is ErrorCode.NOT_FOUND

    async def test_a_keyword_search_finds_the_document_containing_the_term(
        self, adapter: VectorDBAdapter, hybrid_collection: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    point_id="c_notice",
                    collection=hybrid_collection,
                    vector=VECTORS["a"],
                    payload={"kind": "notice"},
                    sparse=encode_document("Either party may terminate with written notice."),
                ),
                Point(
                    point_id="c_payment",
                    collection=hybrid_collection,
                    vector=VECTORS["b"],
                    payload={"kind": "payment"},
                    sparse=encode_document("Invoices are payable within thirty days."),
                ),
            ]
        )

        hits = await adapter.search(
            SearchQuery(collection=hybrid_collection, sparse=encode_query("terminate"), limit=5)
        )

        assert [hit.point_id for hit in hits] == ["c_notice"]

    async def test_a_keyword_search_matches_across_inflections(
        self, adapter: VectorDBAdapter, hybrid_collection: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    point_id="c_notice",
                    collection=hybrid_collection,
                    vector=VECTORS["a"],
                    sparse=encode_document("The agreement covers terminations and notices."),
                )
            ]
        )

        hits = await adapter.search(
            SearchQuery(collection=hybrid_collection, sparse=encode_query("terminate"), limit=5)
        )

        assert [hit.point_id for hit in hits] == ["c_notice"]

    async def test_both_legs_are_searchable_in_one_collection(
        self, adapter: VectorDBAdapter, hybrid_collection: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    point_id="c_a",
                    collection=hybrid_collection,
                    vector=VECTORS["a"],
                    sparse=encode_document("termination notice period"),
                )
            ]
        )

        dense = await adapter.search(
            SearchQuery(collection=hybrid_collection, vector=VECTORS["a"], limit=5)
        )
        sparse = await adapter.search(
            SearchQuery(collection=hybrid_collection, sparse=encode_query("notice"), limit=5)
        )

        assert [hit.point_id for hit in dense] == ["c_a"]
        assert [hit.point_id for hit in sparse] == ["c_a"]

    async def test_filters_are_pushed_down_on_the_keyword_leg_too(
        self, adapter: VectorDBAdapter, hybrid_collection: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    point_id="c_legal",
                    collection=hybrid_collection,
                    vector=VECTORS["a"],
                    payload={"department": "legal"},
                    sparse=encode_document("termination notice"),
                ),
                Point(
                    point_id="c_finance",
                    collection=hybrid_collection,
                    vector=VECTORS["b"],
                    payload={"department": "finance"},
                    sparse=encode_document("termination notice"),
                ),
            ]
        )

        hits = await adapter.search(
            SearchQuery(
                collection=hybrid_collection,
                sparse=encode_query("termination"),
                filters={"department": "finance"},
                limit=5,
            )
        )

        assert [hit.point_id for hit in hits] == ["c_finance"]

    async def test_a_walk_carries_the_stored_sparse_vector(
        self, adapter: VectorDBAdapter, hybrid_collection: str
    ) -> None:
        """What the keyword leg needs is the terms and their weights; IDF is the corpus'."""
        written = encode_document("Either party may terminate with written notice.")
        await adapter.upsert(
            [
                Point(
                    point_id="c_notice",
                    collection=hybrid_collection,
                    vector=VECTORS["a"],
                    sparse=written,
                )
            ]
        )

        walked = [
            found async for found in adapter.iterate_points(hybrid_collection, with_vectors=True)
        ]

        assert len(walked) == 1
        carried = walked[0].sparse
        assert carried is not None
        assert dict(zip(carried.indices, carried.values, strict=True)) == pytest.approx(
            dict(zip(written.indices, written.values, strict=True))
        )
        assert list(carried.indices) == sorted(carried.indices)

    async def test_a_walked_sparse_vector_scores_the_same_in_a_copy(
        self, adapter: VectorDBAdapter, hybrid_collection: str
    ) -> None:
        """Fidelity is the ranking, not the bytes: a leg that survives but rescores is lost."""
        await adapter.upsert(
            [
                Point(
                    point_id="c_notice",
                    collection=hybrid_collection,
                    vector=VECTORS["a"],
                    payload={"department": "legal"},
                    sparse=encode_document("Either party may terminate with written notice."),
                ),
                Point(
                    point_id="c_payment",
                    collection=hybrid_collection,
                    vector=VECTORS["b"],
                    payload={"department": "finance"},
                    sparse=encode_document("Invoices are payable within thirty days of notice."),
                ),
            ]
        )
        query = encode_query("terminate notice")
        before = await adapter.search(
            SearchQuery(collection=hybrid_collection, sparse=query, limit=5)
        )

        copy = f"{hybrid_collection}-copy"
        await adapter.create_collection(
            CollectionSpec(name=copy, dimensions=DIMENSIONS, sparse=True)
        )
        try:
            await adapter.upsert(
                [
                    Point(
                        point_id=found.point_id,
                        collection=copy,
                        vector=found.vector,
                        payload=found.payload,
                        sparse=found.sparse,
                    )
                    async for found in adapter.iterate_points(hybrid_collection, with_vectors=True)
                ]
            )
            after = await adapter.search(SearchQuery(collection=copy, sparse=query, limit=5))
        finally:
            await adapter.drop_collection(copy)

        assert [hit.point_id for hit in before] == ["c_notice", "c_payment"]
        assert [hit.point_id for hit in after] == [hit.point_id for hit in before]
        assert [hit.score for hit in after] == pytest.approx([hit.score for hit in before])

    async def test_a_walk_without_vectors_carries_no_sparse_vector(
        self, adapter: VectorDBAdapter, hybrid_collection: str
    ) -> None:
        """``with_vectors`` is the one gate; a payload-only walk must not pay for postings."""
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

        walked = [found async for found in adapter.iterate_points(hybrid_collection)]

        assert [found.sparse for found in walked] == [None]

    async def test_a_dense_only_collection_walks_without_a_sparse_vector(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        """A collection with no keyword leg has none to report, asked for or not."""
        await self.seed(adapter, collection)

        walked = [found async for found in adapter.iterate_points(collection, with_vectors=True)]

        assert len(walked) == 3
        assert [found.sparse for found in walked] == [None, None, None]

    async def test_a_query_must_carry_exactly_one_leg(self) -> None:
        with pytest.raises(FasterRagError):
            SearchQuery(collection="any")

        with pytest.raises(FasterRagError):
            SearchQuery(collection="any", vector=VECTORS["a"], sparse=encode_query("x"))

    async def test_invalid_credentials_are_not_retryable(
        self, misconfigured_adapter: VectorDBAdapter | None
    ) -> None:
        if misconfigured_adapter is None:
            pytest.skip("this adapter did not supply a misconfigured_adapter fixture")

        with pytest.raises(FasterRagError) as caught:
            await misconfigured_adapter.search(SearchQuery(collection="any", vector=VECTORS["a"]))

        assert caught.value.retryable is False
