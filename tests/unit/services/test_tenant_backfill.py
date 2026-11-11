"""The tenant payload backfill: what it counts, what it writes, and what it refuses.

The adapter here is stateful on purpose. A fake that only records calls cannot show the one
property the command is built around — that a point written once is not selected again — so
this one applies the merge the ``update`` contract promises and is read back by the next run.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

import pytest

from fasterrag.adapters.vectordb.base import CollectionInfo, Point, PointUpdate, VectorDBAdapter
from fasterrag.errors import ErrorCode, FasterRagError
from fasterrag.services.backfill import (
    BackfillProgress,
    audit_collection,
    audit_collections,
    backfill_tenant,
    inventory_untagged,
)


class MemoryAdapter:
    """Holds payloads, merges metadata updates into them, and can fail a chosen batch."""

    def __init__(self, collections: dict[str, dict[str, dict[str, Any]]]) -> None:
        self.payloads = collections
        self.attempted: list[list[str]] = []
        self.written: list[str] = []
        self.fail_on_batch: int | None = None
        self.apply_before_failing = False

    async def list_collections(self) -> list[CollectionInfo]:
        return [
            CollectionInfo(name=name, vectors=len(points)) for name, points in self.payloads.items()
        ]

    async def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        for point_id, payload in list(self.payloads.get(collection, {}).items()):
            yield Point(point_id=point_id, collection=collection, vector=[], payload=dict(payload))

    async def update(self, updates: list[PointUpdate]) -> None:
        self.attempted.append([update.point_id for update in updates])
        failing = self.fail_on_batch is not None and len(self.attempted) == self.fail_on_batch

        if not failing or self.apply_before_failing:
            for update in updates:
                self.payloads[update.collection][update.point_id].update(dict(update.payload))
                self.written.append(update.point_id)

        if failing:
            raise FasterRagError(
                "the backend dropped the connection",
                code=ErrorCode.RETRIEVAL_FAILED,
                retryable=True,
            )


def corpus(**tags: str | None) -> MemoryAdapter:
    """Build a one-collection adapter: each keyword is a point id, its value the tag."""
    points: dict[str, dict[str, Any]] = {}
    for point_id, tag in tags.items():
        payload: dict[str, Any] = {"text": f"body of {point_id}", "document_id": "d_1"}
        if tag is not None:
            payload["tenant"] = tag
        points[point_id] = payload
    return MemoryAdapter({"docs": points})


def adapter_of(memory: MemoryAdapter) -> VectorDBAdapter:
    """Present the stateful double as the contract the service takes."""
    return cast("VectorDBAdapter", memory)


async def test_the_audit_counts_untagged_points_and_names_the_tenants_present() -> None:
    memory = corpus(a=None, b="acme", c="acme", d="globex")

    audit = await audit_collection(adapter_of(memory), "docs")

    assert audit.scanned == 4
    assert audit.untagged == 1
    assert audit.tagged == {"acme": 2, "globex": 1}
    assert audit.tenants == ["acme", "globex"]
    assert audit.backfillable is True


async def test_an_empty_tag_counts_as_untagged() -> None:
    audit = await audit_collection(adapter_of(corpus(a="")), "docs")

    assert audit.untagged == 1
    assert audit.tagged == {}


async def test_the_audit_covers_every_collection_so_the_blackout_is_discoverable() -> None:
    memory = MemoryAdapter(
        {
            "docs": {"a": {"tenant": "acme"}},
            "legacy": {"b": {}, "c": {}},
        }
    )

    audits = {audit.collection: audit for audit in await audit_collections(adapter_of(memory))}

    assert audits["docs"].untagged == 0
    assert audits["legacy"].untagged == 2


async def test_a_dry_run_writes_nothing_and_still_reports_the_numbers() -> None:
    memory = corpus(a=None, b=None, c="acme")

    report = await backfill_tenant(adapter_of(memory), "docs", "acme")

    assert memory.attempted == []
    assert report.applied is False
    assert (report.scanned, report.untagged, report.already, report.foreign) == (3, 2, 1, 0)
    assert report.updated == 0
    assert report.pending == 2


async def test_applying_tags_every_untagged_point_and_leaves_the_rest_alone() -> None:
    memory = corpus(a=None, b=None, c="acme")

    report = await backfill_tenant(adapter_of(memory), "docs", "acme", apply=True)

    assert report.updated == 2
    assert report.pending == 0
    assert [payload["tenant"] for payload in memory.payloads["docs"].values()] == ["acme"] * 3
    assert memory.written == ["a", "b"]


async def test_the_update_carries_only_the_tenant_key() -> None:
    memory = corpus(a=None)
    seen: list[PointUpdate] = []
    original = memory.update

    async def recording(updates: list[PointUpdate]) -> None:
        seen.extend(updates)
        await original(updates)

    memory.update = recording  # type: ignore[method-assign]
    # CRITICAL: the backfill must send the tag alone. `update` merges, so restating other
    # fields would make a backfill able to overwrite a payload it only meant to extend.
    await backfill_tenant(adapter_of(memory), "docs", "acme", apply=True)

    assert [dict(update.payload) for update in seen] == [{"tenant": "acme"}]
    assert memory.payloads["docs"]["a"]["text"] == "body of a"


async def test_running_it_again_after_a_complete_run_writes_nothing() -> None:
    memory = corpus(a=None, b=None)
    await backfill_tenant(adapter_of(memory), "docs", "acme", apply=True)

    again = await backfill_tenant(adapter_of(memory), "docs", "acme", apply=True)

    assert again.updated == 0
    assert again.untagged == 0
    assert len(memory.attempted) == 1


async def test_a_run_killed_halfway_resumes_without_writing_a_point_twice() -> None:
    memory = corpus(a=None, b=None, c=None, d=None)
    memory.fail_on_batch = 2
    seen: list[BackfillProgress] = []

    with pytest.raises(FasterRagError):
        await backfill_tenant(
            adapter_of(memory),
            "docs",
            "acme",
            apply=True,
            batch_size=2,
            progress=seen.append,
        )

    assert seen == [BackfillProgress(updated=2, batches=1)]
    assert memory.written == ["a", "b"]

    resumed = await backfill_tenant(adapter_of(memory), "docs", "acme", apply=True, batch_size=2)

    assert resumed.untagged == 2
    assert resumed.updated == 2
    assert memory.written == ["a", "b", "c", "d"]
    assert sorted(memory.written) == sorted(set(memory.written))


async def test_a_batch_that_landed_before_the_failure_is_not_written_again() -> None:
    memory = corpus(a=None, b=None)
    memory.fail_on_batch = 1
    memory.apply_before_failing = True

    with pytest.raises(FasterRagError):
        await backfill_tenant(adapter_of(memory), "docs", "acme", apply=True, batch_size=2)

    resumed = await backfill_tenant(adapter_of(memory), "docs", "acme", apply=True)

    assert resumed.untagged == 0
    assert resumed.updated == 0
    assert memory.written == ["a", "b"]


async def test_another_tenants_points_are_refused_before_anything_is_written() -> None:
    memory = corpus(a=None, b="globex")

    with pytest.raises(FasterRagError) as caught:
        await backfill_tenant(adapter_of(memory), "docs", "acme", apply=True)

    assert caught.value.code is ErrorCode.CONFLICT
    assert "globex" in caught.value.detail
    assert memory.attempted == []
    assert "tenant" not in memory.payloads["docs"]["a"]


async def test_the_refusal_also_stops_a_dry_run_rather_than_reporting_a_plan() -> None:
    memory = corpus(a=None, b="globex")

    with pytest.raises(FasterRagError) as caught:
        await backfill_tenant(adapter_of(memory), "docs", "acme")

    assert caught.value.code is ErrorCode.CONFLICT


async def test_overwrite_relabels_every_point_and_stays_a_fixpoint() -> None:
    memory = corpus(a=None, b="globex", c="acme")

    report = await backfill_tenant(adapter_of(memory), "docs", "acme", apply=True, overwrite=True)

    assert (report.untagged, report.already, report.foreign) == (1, 1, 1)
    assert report.updated == 2
    assert report.pending == 0
    assert {payload["tenant"] for payload in memory.payloads["docs"].values()} == {"acme"}

    again = await backfill_tenant(adapter_of(memory), "docs", "acme", apply=True, overwrite=True)

    assert again.updated == 0


async def test_an_empty_collection_is_reported_rather_than_silently_succeeding() -> None:
    with pytest.raises(FasterRagError) as caught:
        await backfill_tenant(adapter_of(MemoryAdapter({"docs": {}})), "docs", "acme")

    assert caught.value.code is ErrorCode.NOT_FOUND


@pytest.mark.parametrize("tenant", ["acme__x", "acme_", "", "ac me"])
async def test_an_id_that_could_forge_a_collection_prefix_is_refused(tenant: str) -> None:
    memory = corpus(a=None)

    with pytest.raises(FasterRagError) as caught:
        await backfill_tenant(adapter_of(memory), "docs", tenant, apply=True)

    assert caught.value.code is ErrorCode.TENANT_FORBIDDEN
    assert memory.attempted == []


async def test_the_batch_size_is_what_reaches_the_adapter() -> None:
    memory = corpus(a=None, b=None, c=None, d=None, e=None)

    report = await backfill_tenant(adapter_of(memory), "docs", "acme", apply=True, batch_size=2)

    assert memory.attempted == [["a", "b"], ["c", "d"], ["e"]]
    assert report.batches == 3


async def test_assume_untagged_tags_the_residue_and_leaves_the_other_tenant_alone() -> None:
    memory = corpus(a=None, b=None, c="globex", d="acme")

    report = await backfill_tenant(
        adapter_of(memory), "docs", "acme", apply=True, assume_untagged=True
    )

    assert report.assumed is True
    assert (report.untagged, report.already, report.foreign) == (2, 1, 1)
    assert report.updated == 2
    assert report.pending == 0
    assert memory.written == ["a", "b"]
    assert memory.payloads["docs"]["c"]["tenant"] == "globex"


async def test_a_run_that_assumed_ownership_says_so_in_the_report_it_leaves_behind() -> None:
    memory = corpus(a=None, b="globex")

    report = await backfill_tenant(adapter_of(memory), "docs", "acme", assume_untagged=True)

    # CRITICAL: an assertion the operator made must never read back as one the system
    # derived. The flag is the only thing that let this collection through at all.
    assert report.as_dict()["assumed"] is True
    assert report.as_dict()["overwrite"] is False
    assert memory.attempted == []


async def test_assume_untagged_dry_runs_like_every_other_backfill() -> None:
    memory = corpus(a=None, b=None, c="globex")

    report = await backfill_tenant(adapter_of(memory), "docs", "acme", assume_untagged=True)

    assert memory.attempted == []
    assert report.applied is False
    assert report.pending == 2
    assert "tenant" not in memory.payloads["docs"]["a"]


async def test_an_assumed_run_is_a_fixpoint_and_never_claims_the_other_tenant() -> None:
    memory = corpus(a=None, b="globex")
    await backfill_tenant(adapter_of(memory), "docs", "acme", apply=True, assume_untagged=True)

    again = await backfill_tenant(
        adapter_of(memory), "docs", "acme", apply=True, assume_untagged=True
    )

    assert again.updated == 0
    assert again.foreign == 1
    assert memory.payloads["docs"]["b"]["tenant"] == "globex"


async def test_the_two_ownership_assertions_cannot_be_made_together() -> None:
    memory = corpus(a=None, b="globex")

    with pytest.raises(FasterRagError) as caught:
        await backfill_tenant(
            adapter_of(memory), "docs", "acme", apply=True, overwrite=True, assume_untagged=True
        )

    assert caught.value.code is ErrorCode.VALIDATION_FAILED
    assert memory.attempted == []


async def test_the_conflict_names_both_ways_out_rather_than_only_the_destructive_one() -> None:
    memory = corpus(a=None, b="globex")

    with pytest.raises(FasterRagError) as caught:
        await backfill_tenant(adapter_of(memory), "docs", "acme")

    assert "assume_untagged" in caught.value.detail
    assert "overwrite" in caught.value.detail
    assert "index tenants --untagged" in caught.value.detail


def mixed_corpus() -> MemoryAdapter:
    """A collection whose untagged points come from two documents and one stray chunk."""
    return MemoryAdapter(
        {
            "docs": {
                "a": {"document_id": "d_1", "source_uri": "s3://bucket/one.pdf"},
                "b": {"document_id": "d_1", "source_uri": "s3://bucket/one.pdf"},
                "c": {"document_id": "d_2", "source_uri": "s3://bucket/two.pdf"},
                "d": {"document_id": "d_3", "source_uri": "x", "tenant": "globex"},
                "e": {"text": "no document at all"},
            }
        }
    )


async def test_the_untagged_inventory_groups_by_document_and_skips_tagged_points() -> None:
    rows = await inventory_untagged(adapter_of(mixed_corpus()), "docs")

    assert [(row.document_id, row.chunks) for row in rows] == [("d_1", 2), ("", 1), ("d_2", 1)]
    assert rows[0].source_uri == "s3://bucket/one.pdf"
    assert all(row.collection == "docs" for row in rows)
    assert "d_3" not in {row.document_id for row in rows}


async def test_chunks_with_no_document_id_are_one_unattributable_row_rather_than_many() -> None:
    memory = MemoryAdapter({"docs": {"a": {}, "b": {}, "c": {}}})

    rows = await inventory_untagged(adapter_of(memory), "docs")

    assert [row.as_dict() for row in rows] == [
        {"collection": "docs", "document_id": "", "source_uri": "", "chunks": 3}
    ]


async def test_a_fully_tagged_collection_has_nothing_to_inspect() -> None:
    assert await inventory_untagged(adapter_of(corpus(a="acme", b="globex")), "docs") == []
