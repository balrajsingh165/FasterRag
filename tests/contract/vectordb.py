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
        """Under the default cosine distance, ``score`` must rank higher-is-better.

        Nothing else in this contract looked at ``score``, only at order — and the two come
        apart. A SQL adapter naturally orders by the distance operator itself, so returning a
        raw cosine *distance* as the score leaves every result in the right position with the
        sign of its score inverted. Breaking pgvector's score expression that way passed all
        43 of its cases, which is what prompted this.

        It matters because the score leaves the adapter: it reaches API responses, traces,
        and the dashboard, and a caller comparing two backends' numbers would read one of
        them exactly backwards.
        """
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
