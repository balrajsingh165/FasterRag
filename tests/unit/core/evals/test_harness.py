import json
from pathlib import Path
from typing import Any

import pytest

from fasterrag.core.evals import evaluate, load_golden_set, write_golden_set
from fasterrag.core.evals.golden import GoldenRecord
from fasterrag.core.retrieval.models import ScoredChunk
from fasterrag.errors import ErrorCode, FasterRagError


class ScriptedRetriever:
    """Returns a fixed chunk-id list per query."""

    def __init__(self, results: dict[str, list[str]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int | None]] = []

    async def retrieve(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        filters: Any = None,
    ) -> list[ScoredChunk]:
        self.calls.append((text, top_k))
        return [
            ScoredChunk(
                chunk_id=chunk_id,
                text="body",
                payload={"document_id": chunk_id.replace("c_", "d_")},
                final_rank=position,
            )
            for position, chunk_id in enumerate(self.results.get(text, []), start=1)
        ]


def record(
    identifier: str = "q_1",
    query: str = "what is the notice period",
    chunks: tuple[str, ...] = ("c_a",),
    **extra: Any,
) -> GoldenRecord:
    return GoldenRecord(
        id=identifier,
        query=query,
        source="human",
        created_at="2026-07-30",
        relevant_chunk_ids=chunks,
        **extra,
    )


async def test_a_perfect_retriever_scores_one() -> None:
    golden = [record(chunks=("c_a",))]
    retriever = ScriptedRetriever({"what is the notice period": ["c_a", "c_b"]})

    report = await evaluate(golden, retriever, k=10)

    assert report.recall_at_k == pytest.approx(1.0)
    assert report.mrr == pytest.approx(1.0)
    assert report.ndcg_at_k == pytest.approx(1.0)
    assert report.scored == 1


async def test_a_retriever_that_finds_nothing_scores_zero() -> None:
    golden = [record(chunks=("c_a",))]
    retriever = ScriptedRetriever({"what is the notice period": ["c_x", "c_y"]})

    report = await evaluate(golden, retriever)

    assert report.recall_at_k == 0.0
    assert report.mrr == 0.0
    assert [miss.id for miss in report.misses] == ["q_1"]


async def test_position_of_the_first_hit_drives_mrr() -> None:
    golden = [record(chunks=("c_b",))]
    retriever = ScriptedRetriever({"what is the notice period": ["c_a", "c_b"]})

    report = await evaluate(golden, retriever)

    assert report.mrr == pytest.approx(0.5)


async def test_document_level_ground_truth_survives_rechunking() -> None:
    golden = [
        GoldenRecord(
            id="q_1",
            query="q",
            source="human",
            created_at="2026-07-30",
            relevant_document_ids=("d_a",),
        )
    ]
    retriever = ScriptedRetriever({"q": ["c_a"]})

    report = await evaluate(golden, retriever)

    assert report.recall_at_k == pytest.approx(1.0)


async def test_adversarial_records_are_counted_but_not_averaged() -> None:
    golden = [
        record(identifier="q_1", query="answerable", chunks=("c_a",)),
        GoldenRecord(id="q_2", query="unanswerable", source="human", created_at="2026-07-30"),
    ]
    retriever = ScriptedRetriever({"answerable": ["c_a"], "unanswerable": ["c_z"]})

    report = await evaluate(golden, retriever)

    assert report.adversarial == 1
    assert report.scored == 1
    assert report.recall_at_k == pytest.approx(1.0)
    assert report.misses == []


async def test_the_window_is_passed_to_the_retriever() -> None:
    retriever = ScriptedRetriever({"what is the notice period": ["c_a"]})

    await evaluate([record()], retriever, k=5)

    assert retriever.calls == [("what is the notice period", 5)]


async def test_per_query_detail_is_kept_for_diagnosis() -> None:
    golden = [record(chunks=("c_b",))]
    retriever = ScriptedRetriever({"what is the notice period": ["c_a", "c_b"]})

    report = await evaluate(golden, retriever)
    score = report.per_query[0]

    assert score.retrieved == ("c_a", "c_b")
    assert score.hits == (False, True)
    assert score.total_relevant == 1


async def test_an_empty_golden_set_reports_zeroes() -> None:
    report = await evaluate([], ScriptedRetriever({}))

    assert report.scored == 0
    assert report.recall_at_k == 0.0
    assert report.as_dict()["misses"] == []


async def test_the_report_serializes_for_ci() -> None:
    golden = [record(chunks=("c_a",))]
    payload = (
        await evaluate(golden, ScriptedRetriever({"what is the notice period": ["c_a"]}))
    ).as_dict()

    assert payload["recall_at_k"] == 1.0
    assert payload["k"] == 10
    assert payload["scored"] == 1


def test_a_golden_set_round_trips_through_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    original = [record(identifier="q_1"), record(identifier="q_2", chunks=("c_b", "c_c"))]

    write_golden_set(path, original)

    assert load_golden_set(path) == original


def test_the_documented_schema_loads(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(
        json.dumps(
            {
                "id": "q_0001",
                "query": "What does the vendor agreement say about termination?",
                "relevant_chunk_ids": ["c_9f2", "c_a01"],
                "relevant_document_ids": ["d_112"],
                "answer_reference": "Either party may terminate with 30 days written notice.",
                "metadata": {"department": "legal"},
                "source": "human",
                "created_at": "2026-07-29",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    loaded = load_golden_set(path)[0]

    assert loaded.id == "q_0001"
    assert loaded.relevant_chunk_ids == ("c_9f2", "c_a01")
    assert loaded.metadata == {"department": "legal"}
    assert loaded.adversarial is False


def test_an_adversarial_record_is_recognized() -> None:
    bare = GoldenRecord(id="q_1", query="q", source="human", created_at="2026-07-30")

    assert bare.adversarial is True


def test_a_generated_record_is_marked_as_such() -> None:
    generated = record(identifier="q_1")
    assert generated.generated is False

    from dataclasses import replace

    assert replace(generated, source="autopilot").generated is True


def test_a_missing_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FasterRagError, match="not found") as caught:
        load_golden_set(tmp_path / "absent.jsonl")

    assert caught.value.code is ErrorCode.NOT_FOUND


def test_a_malformed_line_names_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text('{"id": "q_1"}\n{not json\n', encoding="utf-8")

    with pytest.raises(FasterRagError, match="line 1"):
        load_golden_set(path)


def test_a_record_missing_required_fields_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    path.write_text(json.dumps({"id": "q_1", "query": "q"}) + "\n", encoding="utf-8")

    with pytest.raises(FasterRagError, match="missing required fields"):
        load_golden_set(path)


def test_duplicate_ids_are_refused_so_scores_cannot_double_count(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    line = json.dumps({"id": "q_1", "query": "q", "source": "human", "created_at": "2026-07-30"})
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")

    with pytest.raises(FasterRagError, match="repeats the query id"):
        load_golden_set(path)


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "golden.jsonl"
    line = json.dumps({"id": "q_1", "query": "q", "source": "human", "created_at": "2026-07-30"})
    path.write_text(f"\n{line}\n\n", encoding="utf-8")

    assert len(load_golden_set(path)) == 1
