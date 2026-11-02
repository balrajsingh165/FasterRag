"""The shared adapter contract suite, run against a real embedded Chroma store.

The same ``VectorDBContract`` the Qdrant and pgvector suites run, subclassed rather than
copied. Chroma is the third paradigm the contract has to survive — no server process at
all — and the parts it has no native answer for (aliases, snapshots, IDF) are exactly the
parts these cases exist to hold honest.

The classes below the contract cover what the shared suite does not, for the same reason
the pgvector package has its own: snapshots are a collection copy rather than a backend
file, ``iterate_points`` is id paging rather than a scroll cursor, and the alias
primitive is a record this adapter writes itself, so its durability has to be proved
against a genuinely separate process rather than assumed.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from fasterrag.adapters.vectordb.base import (
    CollectionSpec,
    Point,
    PointSelector,
    SearchQuery,
    VectorDBAdapter,
)
from fasterrag.adapters.vectordb.chroma import ChromaAdapter
from fasterrag.config.schema import Settings
from fasterrag.core.retrieval.bm25 import encode_document, encode_query
from fasterrag.errors import ErrorCode, FasterRagError
from tests.contract.vectordb import DIMENSIONS, VECTORS, VectorDBContract, point
from tests.integration.chroma.conftest import chroma_settings

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[3]

RESOLVE_ALIAS = textwrap.dedent(
    """
    import asyncio
    import sys

    from fasterrag.adapters.vectordb.chroma import ChromaAdapter
    from fasterrag.config.schema import Settings

    settings = Settings.model_validate(
        {"vector_db": {"provider": "chroma", "chroma": {"path": sys.argv[1]}}}
    )


    async def main():
        adapter = ChromaAdapter(settings)
        print(await adapter.alias_target(sys.argv[2]))
        await adapter.close()


    asyncio.run(main())
    """
)


class ChromaFixtures:
    """Adapter and collection wiring shared by the contract run and the extra cases."""

    @pytest.fixture
    async def adapter(self, chroma: Settings) -> AsyncIterator[VectorDBAdapter]:
        built = ChromaAdapter(chroma)
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


class TestChromaContract(ChromaFixtures, VectorDBContract):
    """Every behaviour the shared contract requires, in-process and file-backed."""

    @pytest.fixture
    async def misconfigured_adapter(
        self, chroma_not_a_directory: Path
    ) -> AsyncIterator[VectorDBAdapter]:
        """Chroma has no credential, so the unusable-configuration case is a bad path."""
        built = ChromaAdapter(chroma_settings(chroma_not_a_directory))
        yield built
        await built.close()


class TestChromaAliases(ChromaFixtures):
    """Aliases are records this adapter writes, so their durability is its problem."""

    async def test_an_alias_survives_reopening_the_directory(
        self, chroma: Settings, collection_name: str
    ) -> None:
        first = ChromaAdapter(chroma)
        await first.create_collection(CollectionSpec(name=collection_name, dimensions=DIMENSIONS))
        await first.set_alias(f"{collection_name}-alias", collection_name)
        await first.close()

        second = ChromaAdapter(chroma)
        try:
            assert await second.alias_target(f"{collection_name}-alias") == collection_name
        finally:
            await second.delete_alias(f"{collection_name}-alias")
            await second.drop_collection(collection_name)
            await second.close()

    async def test_an_alias_survives_a_separate_process(
        self, chroma: Settings, collection_name: str
    ) -> None:
        """The claim an in-process store makes hardest: proved by a second interpreter."""
        adapter = ChromaAdapter(chroma)
        await adapter.create_collection(CollectionSpec(name=collection_name, dimensions=DIMENSIONS))
        await adapter.set_alias(f"{collection_name}-alias", collection_name)
        await adapter.close()

        environment = dict(os.environ, PYTHONPATH=str(REPO_ROOT / "src"))
        finished = subprocess.run(
            [
                sys.executable,
                "-c",
                RESOLVE_ALIAS,
                chroma.vector_db.chroma.path,
                f"{collection_name}-alias",
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            env=environment,
        )

        assert finished.returncode == 0, finished.stderr
        assert finished.stdout.strip().splitlines()[-1] == collection_name

        cleanup = ChromaAdapter(chroma)
        await cleanup.delete_alias(f"{collection_name}-alias")
        await cleanup.drop_collection(collection_name)
        await cleanup.close()

    async def test_dropping_a_collection_removes_the_alias_pointing_at_it(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        alias = f"{collection}-alias"
        await adapter.set_alias(alias, collection)

        assert await adapter.drop_collection(collection) is True
        assert await adapter.alias_target(alias) is None

    async def test_an_alias_may_not_shadow_a_collection(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        with pytest.raises(FasterRagError) as caught:
            await adapter.set_alias(collection, collection)

        assert caught.value.code is ErrorCode.CONFLICT


class TestChromaSnapshots(ChromaFixtures):
    """Snapshots as a copy into a second collection, with the limits that implies."""

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


class TestChromaIteration(ChromaFixtures):
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


class TestChromaPayloads(ChromaFixtures):
    """Chroma metadata is scalar-only, so the payload flattening has to be lossless."""

    async def test_a_nested_payload_round_trips(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        stored = {
            "span": {"start": 0, "end": 12},
            "tags": ["legal", "eu"],
            "page": 3,
            "section": None,
            "text": "a clause",
        }
        await adapter.upsert(
            [Point(point_id="c_a", collection=collection, vector=VECTORS["a"], payload=stored)]
        )

        hits = await adapter.search(
            SearchQuery(collection=collection, vector=VECTORS["a"], limit=1)
        )

        assert hits[0].payload == stored

    async def test_a_scalar_inside_a_nested_payload_still_filters(
        self, adapter: VectorDBAdapter, collection: str
    ) -> None:
        await adapter.upsert(
            [
                Point(
                    point_id="c_a",
                    collection=collection,
                    vector=VECTORS["a"],
                    payload={"span": {"start": 0}, "department": "legal"},
                ),
                Point(
                    point_id="c_b",
                    collection=collection,
                    vector=VECTORS["b"],
                    payload={"span": {"start": 9}, "department": "finance"},
                ),
            ]
        )

        hits = await adapter.search(
            SearchQuery(
                collection=collection,
                vector=VECTORS["a"],
                limit=10,
                filters={"department": "finance"},
            )
        )

        assert [hit.point_id for hit in hits] == ["c_b"]
