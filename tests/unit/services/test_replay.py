from pathlib import Path
from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.core.tracing import Span, SpanRecorder, Trace, config_snapshot
from fasterrag.errors import FasterRagError
from fasterrag.services.replay import diff_config, diff_retrieval, replay_trace
from fasterrag.services.traces import TraceStore, create_trace_store


def chunk(chunk_id: str) -> dict[str, Any]:
    return {"chunk_id": chunk_id, "text": "body", "rrf_score": 0.5}


def trace(**overrides: Any) -> Trace:
    defaults: dict[str, Any] = {
        "trace_id": "t_1",
        "query": "what is the notice period?",
        "collection": "docs",
        "config_snapshot": config_snapshot(Settings()),
        "retrieved": [chunk("c_a"), chunk("c_b")],
        "prompt": "<context>...</context>",
        "response": "thirty days",
        "result": {"answer": "thirty days", "citations": [{"chunk_id": "c_a"}]},
        "created_at": "2026-08-01T00:00:00+00:00",
    }
    return Trace(**{**defaults, **overrides})


def test_identical_retrieval_reports_no_difference() -> None:
    before = [chunk("c_a"), chunk("c_b")]

    assert diff_retrieval(before, list(before)).identical is True


def test_a_new_chunk_is_reported_as_added() -> None:
    result = diff_retrieval([chunk("c_a")], [chunk("c_a"), chunk("c_b")])

    assert result.added == ["c_b"]
    assert result.removed == []
    assert result.identical is False


def test_a_lost_chunk_is_reported_as_removed() -> None:
    result = diff_retrieval([chunk("c_a"), chunk("c_b")], [chunk("c_a")])

    assert result.removed == ["c_b"]
    assert result.added == []


def test_a_moved_chunk_reports_both_ranks() -> None:
    result = diff_retrieval([chunk("c_a"), chunk("c_b")], [chunk("c_b"), chunk("c_a")])

    moves = {move["chunk_id"]: (move["was"], move["now"]) for move in result.reordered}
    assert moves["c_a"] == (1, 2)
    assert moves["c_b"] == (2, 1)


def test_ranks_are_reported_one_based() -> None:
    result = diff_retrieval([chunk("c_a"), chunk("c_b")], [chunk("c_b"), chunk("c_a")])

    assert all(move["was"] >= 1 and move["now"] >= 1 for move in result.reordered)


def test_a_score_change_without_a_move_is_not_a_finding() -> None:
    before = [{"chunk_id": "c_a", "rrf_score": 0.5}]
    after = [{"chunk_id": "c_a", "rrf_score": 0.9}]

    assert diff_retrieval(before, after).identical is True


def test_an_unchanged_config_diffs_to_nothing() -> None:
    snapshot = config_snapshot(Settings())

    assert diff_config(snapshot, snapshot) == []


def test_a_changed_key_names_itself_with_both_values() -> None:
    before = config_snapshot(Settings())
    after = config_snapshot(Settings.model_validate({"retrieval": {"rrf_k": 10}}))

    changes = diff_config(before, after)

    assert {"key": "retrieval.rrf_k", "was": 60.0, "now": 10.0} in changes


def test_the_diff_names_the_exact_nested_key() -> None:
    before = config_snapshot(Settings())
    after = config_snapshot(Settings.model_validate({"llm": {"temperature": 0.9}}))

    assert [change["key"] for change in diff_config(before, after)] == ["llm.temperature"]


def test_an_unrelated_key_is_absent_from_the_snapshot() -> None:
    snapshot = config_snapshot(Settings())

    assert "app" not in snapshot
    assert "security" not in snapshot


def test_the_snapshot_never_holds_a_secret() -> None:
    rendered = repr(config_snapshot(Settings()))

    assert "api_key" not in rendered


def test_a_span_reports_its_duration() -> None:
    span = Span(name="retrieval", start_ms=10.0, end_ms=42.5)

    assert span.duration_ms == pytest.approx(32.5)


def test_a_span_round_trips_through_its_serialized_form() -> None:
    span = Span(name="generation", start_ms=1.0, end_ms=2.0, attributes={"model": "gpt"})

    restored = Span.from_dict(span.as_dict())

    assert restored.name == "generation"
    assert restored.attributes == {"model": "gpt"}


def test_spans_share_one_clock_origin() -> None:
    recorder = SpanRecorder()

    first = recorder.record("retrieval", 0.0)
    second = recorder.record("generation", first.end_ms)

    assert second.start_ms >= first.end_ms
    assert [span.name for span in recorder.spans] == ["retrieval", "generation"]


def test_a_trace_round_trips_through_its_serialized_form() -> None:
    original = trace(spans=[Span(name="retrieval", start_ms=0.0, end_ms=5.0)])

    restored = Trace.from_dict(original.as_dict())

    assert restored.trace_id == original.trace_id
    assert restored.query == original.query
    assert [span.name for span in restored.spans] == ["retrieval"]


def test_a_stored_trace_reads_back(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    store.store(trace())

    loaded = store.load("t_1")

    assert loaded is not None
    assert loaded.query == "what is the notice period?"


def test_a_disabled_store_writes_nothing(tmp_path: Path) -> None:
    store = TraceStore(tmp_path, enabled=False)
    store.store(trace())

    assert store.load("t_1") is None
    assert list(tmp_path.glob("*.json")) == []


def test_an_absent_trace_reads_as_none(tmp_path: Path) -> None:
    assert TraceStore(tmp_path).load("t_never") is None


def test_a_corrupt_trace_reads_as_none_rather_than_raising(tmp_path: Path) -> None:
    store = TraceStore(tmp_path)
    store.store(trace())
    (tmp_path / "t_1.json").write_text("{not json", encoding="utf-8")

    assert store.load("t_1") is None


def test_an_unwritable_root_never_raises(tmp_path: Path) -> None:
    blocker = tmp_path / "traces"
    blocker.write_text("not a directory", encoding="utf-8")

    TraceStore(blocker).store(trace())


def test_recent_lists_newest_first(tmp_path: Path) -> None:
    import os
    import time

    store = TraceStore(tmp_path)
    store.store(trace(trace_id="t_old"))
    store.store(trace(trace_id="t_new"))
    old = tmp_path / "t_old.json"
    os.utime(old, (time.time() - 60, time.time() - 60))

    assert store.recent()[0] == "t_new"


def test_pruning_removes_traces_past_the_window(tmp_path: Path) -> None:
    import os
    import time

    store = TraceStore(tmp_path, retention_days=1)
    store.store(trace(trace_id="t_old"))
    store.store(trace(trace_id="t_new"))
    stale = time.time() - 2 * 86400
    os.utime(tmp_path / "t_old.json", (stale, stale))

    removed = store.prune()

    assert removed == 1
    assert store.load("t_old") is None
    assert store.load("t_new") is not None


def test_pruning_an_absent_directory_is_harmless(tmp_path: Path) -> None:
    assert TraceStore(tmp_path / "never").prune() == 0


def test_the_store_is_built_from_configuration() -> None:
    settings = Settings.model_validate({"traces": {"store": False, "retention_days": 7}})

    store = create_trace_store(settings)

    assert store.enabled is False
    assert store.retention_days == 7


class StubGeneration:
    """Returns a scripted answer and candidate set."""

    def __init__(self, answer: Any, candidates: list[Any], traces: Any = None) -> None:
        self.answer_value = answer
        self.candidates = candidates
        self.traces = traces

    async def answer_with_candidates(self, question: str, **kwargs: Any) -> tuple[Any, list[Any]]:
        return self.answer_value, self.candidates

    async def close(self) -> None:
        return None


async def test_a_replay_refuses_to_record_its_own_trace(tmp_path: Path) -> None:
    from fasterrag.services.generation import Answer

    service = StubGeneration(Answer(answer="a"), [], traces=TraceStore(tmp_path))

    with pytest.raises(FasterRagError, match="tracing disabled"):
        await replay_trace(trace(), Settings(), service)  # type: ignore[arg-type]


async def test_an_identical_replay_is_reported_as_deterministic() -> None:
    from fasterrag.core.retrieval.models import ScoredChunk
    from fasterrag.services.generation import Answer

    candidates = [ScoredChunk(chunk_id="c_a", text="body"), ScoredChunk(chunk_id="c_b", text="b")]
    service = StubGeneration(Answer(answer="thirty days"), candidates)

    result = await replay_trace(trace(), Settings(), service)  # type: ignore[arg-type]

    assert result.retrieval.identical is True
    assert result.config_changes == []
    assert result.deterministic is True
    assert result.answer_changed is False


async def test_a_changed_config_is_not_a_determinism_failure() -> None:
    from fasterrag.core.retrieval.models import ScoredChunk
    from fasterrag.services.generation import Answer

    candidates = [ScoredChunk(chunk_id="c_b", text="b"), ScoredChunk(chunk_id="c_a", text="a")]
    service = StubGeneration(Answer(answer="different"), candidates)
    candidate = Settings.model_validate({"retrieval": {"rrf_k": 10}})

    result = await replay_trace(trace(), candidate, service)  # type: ignore[arg-type]

    assert result.deterministic is False
    assert result.config_changes
    assert result.retrieval.reordered


async def test_a_replay_reports_both_answers_side_by_side() -> None:
    from fasterrag.services.generation import Answer

    service = StubGeneration(Answer(answer="ninety days"), [])

    result = await replay_trace(trace(), Settings(), service)  # type: ignore[arg-type]

    assert result.original_answer == "thirty days"
    assert result.replayed_answer == "ninety days"
    assert result.answer_changed is True


async def test_the_replay_body_carries_the_documented_members() -> None:
    from fasterrag.services.generation import Answer

    service = StubGeneration(Answer(answer="a"), [])

    payload = (await replay_trace(trace(), Settings(), service)).as_dict()  # type: ignore[arg-type]

    assert set(payload) == {
        "trace_id",
        "query",
        "config_changes",
        "retrieval",
        "answer_changed",
        "original",
        "replayed",
    }
