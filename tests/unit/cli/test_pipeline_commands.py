import argparse
import asyncio
import contextlib
import json
import threading
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

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
from fasterrag.cli.commands import pipeline
from fasterrag.cli.output import Console, ExitCode
from fasterrag.cli.parser import build_parser
from fasterrag.config.schema import Settings
from fasterrag.errors import FasterRagError, RetrievalError
from fasterrag.services.journal import JobRecord, Journal
from fasterrag.services.reindex import ReindexPlan

CONFIG = """
vector_db:
  provider: qdrant
  mode: external
  api_key_env: null
embeddings:
  provider: huggingface
llm:
  provider: ollama
  api_key_env: null
"""


class FakeAdapter(VectorDBAdapter):
    """Records what the index commands asked for."""

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self.collections: list[CollectionInfo] = []
        self.created: list[CollectionSpec] = []
        self.dropped: list[str] = []
        self.error: Exception | None = None
        self.aliases: dict[str, str] = {}
        self.snapshots: dict[str, list[str]] = {}
        self.restored: list[tuple[str, str]] = []

    async def create_collection(self, spec: CollectionSpec) -> None:
        if self.error is not None:
            raise self.error
        self.created.append(spec)

    async def list_collections(self) -> list[CollectionInfo]:
        if self.error is not None:
            raise self.error
        return list(self.collections)

    async def drop_collection(self, name: str) -> bool:
        self.dropped.append(name)
        return any(info.name == name for info in self.collections)

    async def snapshot(self, collection: str) -> str:
        self.snapshots.setdefault(collection, []).append(f"{collection}-snap")
        return f"{collection}-snap"

    async def list_snapshots(self, collection: str) -> list[str]:
        return list(self.snapshots.get(collection, []))

    async def delete_snapshot(self, collection: str, snapshot: str) -> bool:
        return True

    async def restore_snapshot(self, collection: str, snapshot: str) -> None:
        self.restored.append((collection, snapshot))

    async def set_alias(self, alias: str, collection: str) -> None:
        self.aliases[alias] = collection

    async def alias_target(self, alias: str) -> str | None:
        return self.aliases.get(alias)

    async def delete_alias(self, alias: str) -> bool:
        return self.aliases.pop(alias, None) is not None

    async def upsert(self, points: list[Point]) -> UpsertResult:
        return UpsertResult(upserted=len(points))

    async def iterate_points(
        self, collection: str, *, with_vectors: bool = False, batch_size: int = 256
    ) -> AsyncIterator[Point]:
        """Yield nothing; these doubles hold no scannable state."""
        empty: list[Point] = []
        for point in empty:
            yield point

    async def search(self, query: SearchQuery) -> list[ScoredPoint]:
        return []

    async def update(self, updates: list[PointUpdate]) -> None:
        return None

    async def delete(self, selector: PointSelector) -> None:
        return None

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        return None


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key")
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    return str(path)


@pytest.fixture
def adapter(monkeypatch: pytest.MonkeyPatch) -> FakeAdapter:
    built = FakeAdapter(Settings())
    monkeypatch.setattr(pipeline, "create_vector_db_adapter", lambda settings: built)
    return built


def namespace(**values: Any) -> argparse.Namespace:
    defaults = {
        "config": "config.yaml",
        "collection": None,
        "as_json": False,
        "quiet": False,
        "verbose": False,
    }
    return argparse.Namespace(**{**defaults, **values})


def test_key_value_pairs_are_parsed() -> None:
    parsed = pipeline._pairs(["a=1", "b=two"], Console())

    assert parsed == {"a": "1", "b": "two"}


def test_a_value_containing_an_equals_sign_survives() -> None:
    assert pipeline._pairs(["url=http://x?a=b"], Console()) == {"url": "http://x?a=b"}


def test_an_empty_value_is_allowed() -> None:
    assert pipeline._pairs(["a="], Console()) == {"a": ""}


@pytest.mark.parametrize("value", ["novalue", "=orphan"])
def test_a_malformed_pair_is_rejected(value: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert pipeline._pairs([value], Console()) is None
    assert "expected KEY=VALUE" in capsys.readouterr().err


async def test_index_list_reports_an_empty_backend(
    config: str, adapter: FakeAdapter, capsys: pytest.CaptureFixture[str]
) -> None:
    code = await pipeline.run_index(namespace(config=config, action="list"), Console())

    assert code == ExitCode.SUCCESS
    assert "no collections" in capsys.readouterr().out


async def test_index_list_prints_each_collection(
    config: str, adapter: FakeAdapter, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter.collections = [CollectionInfo(name="docs", vectors=42, dimensions=384, sparse=True)]

    await pipeline.run_index(namespace(config=config, action="list"), Console())
    out = capsys.readouterr().out

    assert "docs" in out
    assert "42" in out


async def test_index_list_reports_json(
    config: str, adapter: FakeAdapter, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter.collections = [CollectionInfo(name="docs", vectors=7, dimensions=384)]

    await pipeline.run_index(
        namespace(config=config, action="list", as_json=True), Console(as_json=True)
    )
    payload = json.loads(capsys.readouterr().out)

    listed = payload["collections"][0]

    assert {
        "name": "docs",
        "vectors": 7,
        "dimensions": 384,
        "distance": None,
        "sparse": False,
    }.items() <= listed.items()


async def test_index_list_reports_drift_status_per_collection(
    config: str, adapter: FakeAdapter, capsys: pytest.CaptureFixture[str]
) -> None:
    adapter.collections = [CollectionInfo(name="docs", vectors=7)]

    await pipeline.run_index(
        namespace(config=config, action="list", as_json=True), Console(as_json=True)
    )
    drift = json.loads(capsys.readouterr().out)["collections"][0]["drift"]

    assert drift["missing_lock"] is True
    assert drift["detected"] is False


async def test_an_unreachable_backend_exits_with_the_unreachable_code(
    config: str, adapter: FakeAdapter
) -> None:
    adapter.error = RetrievalError("qdrant is unreachable", retryable=True)

    code = await pipeline.run_index(namespace(config=config, action="list"), Console())

    assert code == ExitCode.UNREACHABLE


async def test_a_non_retryable_backend_error_exits_with_the_failure_code(
    config: str, adapter: FakeAdapter
) -> None:
    adapter.error = FasterRagError("collection configuration is unreadable", retryable=False)

    code = await pipeline.run_index(namespace(config=config, action="list"), Console())

    assert code == ExitCode.FAILURE


async def test_deleting_without_force_is_refused(
    config: str, adapter: FakeAdapter, capsys: pytest.CaptureFixture[str]
) -> None:
    code = await pipeline.run_index(
        namespace(config=config, action="delete", name="docs", force=False), Console()
    )

    assert code == ExitCode.USAGE
    assert adapter.dropped == []
    assert "--force" in capsys.readouterr().err


async def test_deleting_with_force_drops_the_collection(config: str, adapter: FakeAdapter) -> None:
    adapter.collections = [CollectionInfo(name="docs", vectors=1)]

    code = await pipeline.run_index(
        namespace(config=config, action="delete", name="docs", force=True), Console()
    )

    assert code == ExitCode.SUCCESS
    assert adapter.dropped == ["docs"]


async def test_deleting_an_absent_collection_says_so(
    config: str, adapter: FakeAdapter, capsys: pytest.CaptureFixture[str]
) -> None:
    code = await pipeline.run_index(
        namespace(config=config, action="delete", name="gone", force=True), Console()
    )

    assert code == ExitCode.SUCCESS
    assert "no such collection" in capsys.readouterr().out


async def test_an_unknown_distance_is_a_usage_error(
    config: str, adapter: FakeAdapter, capsys: pytest.CaptureFixture[str]
) -> None:
    code = await pipeline.run_index(
        namespace(
            config=config,
            action="create",
            name="docs",
            distance="manhattan",
            shards=None,
            replicas=None,
        ),
        Console(),
    )

    assert code == ExitCode.USAGE
    assert adapter.created == []
    assert "--distance must be one of" in capsys.readouterr().err


async def test_an_invalid_config_never_reaches_the_backend(
    tmp_path: Path, adapter: FakeAdapter
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("retrieval:\n  top_k: 9999\n", encoding="utf-8")

    code = await pipeline.run_index(namespace(config=str(path), action="list"), Console())

    assert code == ExitCode.USAGE
    assert adapter.collections == []


async def test_a_dry_run_ingest_never_touches_the_pipeline(
    config: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "note.txt"
    source.write_text("Either party may terminate on thirty days notice.")

    code = await pipeline.run_ingest(
        namespace(
            config=config,
            sources=[str(source)],
            metadata=[],
            dry_run=True,
            as_json=True,
        ),
        Console(as_json=True),
    )
    payload = json.loads(capsys.readouterr().out)

    assert code == ExitCode.SUCCESS
    assert payload["dry_run"] is True
    assert payload["documents"] == 1
    assert payload["chunks"] >= 1


async def test_a_dry_run_is_refused_when_the_estimator_is_switched_off(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--dry-run` is the estimator, so it answers to `cost.estimator` (TASK-0200).

    Left ungated it would be the way around the switch: the same token counts and the same
    projected cost, reached by a different command.
    """
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key")
    path = tmp_path / "estimator-off.yaml"
    path.write_text(f"{CONFIG}cost:\n  estimator: false\n", encoding="utf-8")
    source = tmp_path / "note.txt"
    source.write_text("Either party may terminate on thirty days notice.")

    code = await pipeline.run_ingest(
        namespace(config=str(path), sources=[str(source)], metadata=[], dry_run=True),
        Console(),
    )
    captured = capsys.readouterr()

    assert code == ExitCode.USAGE
    assert "cost.estimator" in captured.err
    assert "would index" not in captured.out


async def test_malformed_ingest_metadata_is_refused_before_any_work(
    config: str, tmp_path: Path
) -> None:
    code = await pipeline.run_ingest(
        namespace(
            config=config,
            sources=[str(tmp_path)],
            metadata=["not-a-pair"],
            dry_run=True,
        ),
        Console(),
    )

    assert code == ExitCode.USAGE


class StoppedBuild:
    """An ingestion service whose build fails, so a reembed stops after handing over sources."""

    def __init__(self) -> None:
        self.sources: list[str] = []
        self.journal: FakeJournal | None = None

    async def ingest(self, sources: Any, *, collection: str) -> JobRecord:
        self.sources = list(sources)
        return JobRecord(job_id="j", collection=collection, status="failed")

    async def accept(self, sources: Any, *, collection: str | None = None) -> JobRecord:
        self.sources = list(sources)
        return JobRecord(job_id="j", collection=collection or "docs", status="queued")

    async def run(self, record: JobRecord, *, metadata: Any = None) -> JobRecord:
        if self.journal is not None:
            assert await asyncio.to_thread(self.journal.drained.wait, 5)
        return JobRecord(job_id=record.job_id, collection=record.collection, status="failed")

    async def close(self) -> None:
        return None


@pytest.fixture
def stopped_build(monkeypatch: pytest.MonkeyPatch) -> StoppedBuild:
    service = StoppedBuild()

    async def fake_plan(name: str, settings: Settings, adapter: Any) -> ReindexPlan:
        return ReindexPlan(
            alias=name,
            blue=None,
            green=f"{name}-green",
            strategy="blue_green",
            eval_gate=False,
            rollback_retention_hours=24,
        )

    monkeypatch.setattr(pipeline, "plan_reindex", fake_plan)
    monkeypatch.setattr(pipeline, "IngestionService", lambda *args, **kwargs: service)
    monkeypatch.setattr(pipeline, "create_journal", lambda settings: object())
    monkeypatch.setattr(pipeline, "create_lock_store", lambda settings: object())
    return service


def reembed(config: str, source: str, *flags: str) -> argparse.Namespace:
    """Build a reembed namespace through the real parser."""
    return build_parser().parse_args(
        ["index", "--config", config, "reembed", "docs", source, "--no-eval-gate", *flags]
    )


async def test_a_reembed_expands_a_directory_source_into_its_files(
    config: str, tmp_path: Path, adapter: FakeAdapter, stopped_build: StoppedBuild
) -> None:
    """``ingest ./corpus`` and ``index reembed docs ./corpus`` must build the same thing."""
    (tmp_path / "a.txt").write_text("thirty days written notice", encoding="utf-8")
    (tmp_path / "b.txt").write_text("payable within forty-five days", encoding="utf-8")

    await pipeline.run_index(reembed(config, str(tmp_path)), Console())

    assert sorted(Path(name).name for name in stopped_build.sources) == [
        "a.txt",
        "b.txt",
        "config.yaml",
    ]


async def test_a_reembed_without_recursive_leaves_a_subdirectory_alone(
    config: str, tmp_path: Path, adapter: FakeAdapter, stopped_build: StoppedBuild
) -> None:
    annex = tmp_path / "annex"
    annex.mkdir()
    (annex / "c.txt").write_text("the agreed schedule of rates", encoding="utf-8")

    await pipeline.run_index(reembed(config, str(tmp_path)), Console())

    assert "c.txt" not in [Path(name).name for name in stopped_build.sources]


async def test_the_recursive_flag_reaches_a_reembed(
    config: str, tmp_path: Path, adapter: FakeAdapter, stopped_build: StoppedBuild
) -> None:
    """The handler read ``args.recursive`` while the parser declared it nowhere."""
    annex = tmp_path / "annex"
    annex.mkdir()
    (annex / "c.txt").write_text("the agreed schedule of rates", encoding="utf-8")

    await pipeline.run_index(reembed(config, str(tmp_path), "--recursive"), Console())

    assert "c.txt" in [Path(name).name for name in stopped_build.sources]


class FakeJournal(Journal):
    """A journal whose per-document tally advances one poll at a time."""

    def __init__(self, tallies: Sequence[Mapping[str, int]]) -> None:
        """Serve ``tallies`` in order, repeating the last one once they run out."""
        super().__init__(Path("journal-root-never-read"))
        self.tallies = [dict(tally) for tally in tallies]
        self.polls = 0
        self.drained = threading.Event()

    def counts(self, job: str) -> dict[str, int]:
        """Return the next tally, as ``Journal.counts`` does."""
        tally = self.tallies[min(self.polls, len(self.tallies) - 1)]
        self.polls += 1
        if self.polls > len(self.tallies):
            self.drained.set()
        return dict(tally)


class WatchedIngest:
    """An ingestion service that settles once the follower has seen every tally."""

    def __init__(self, journal: FakeJournal) -> None:
        """Hold the journal whose exhaustion ends the run."""
        self.journal = journal
        self.accepted: list[str] = []
        self.metadata: dict[str, Any] | None = None
        self.ingested = False

    async def accept(self, sources: Any, *, collection: str | None = None) -> JobRecord:
        self.accepted = list(sources)
        return JobRecord(job_id="job-1", collection=collection or "docs", status="queued")

    async def run(self, record: JobRecord, *, metadata: dict[str, Any] | None = None) -> JobRecord:
        self.metadata = metadata
        assert await asyncio.to_thread(self.journal.drained.wait, 5)
        return JobRecord(
            job_id=record.job_id,
            collection=record.collection,
            status="completed",
            counts={"indexed": 2, "dead_lettered": 0},
        )

    async def ingest(self, sources: Any, **kwargs: Any) -> JobRecord:
        self.ingested = True
        return JobRecord(
            job_id="job-1", collection="docs", status="completed", counts={"indexed": 2}
        )

    async def close(self) -> None:
        return None


@pytest.fixture
def instant_polling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Poll without waiting, so a watch test takes microseconds rather than seconds."""
    monkeypatch.setattr(pipeline, "_WATCH_POLL_SECONDS", 0.0)


def watched(journal: FakeJournal, monkeypatch: pytest.MonkeyPatch) -> WatchedIngest:
    """Install the ingestion service and journal an ingest command will build."""
    service = WatchedIngest(journal)
    monkeypatch.setattr(pipeline, "IngestionService", lambda *args, **kwargs: service)
    monkeypatch.setattr(pipeline, "create_journal", lambda settings: journal)
    monkeypatch.setattr(pipeline, "create_lock_store", lambda settings: object())
    return service


def ingest_args(config: str, source: str, *flags: str) -> argparse.Namespace:
    """Build an ingest namespace through the real parser."""
    return build_parser().parse_args(["ingest", "--config", config, source, *flags])


async def test_the_follower_prints_a_line_per_change_not_per_poll(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A tally that has not moved is not news; printing it every second would be noise."""
    journal = FakeJournal(
        [{"total": 1, "indexed": 1}, {"total": 1, "indexed": 1}, {"total": 2, "indexed": 2}]
    )

    follower = asyncio.create_task(pipeline._follow_job(journal, "j", Console(), interval=0.0))
    assert await asyncio.to_thread(journal.drained.wait, 5)
    follower.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await follower

    assert [line for line in capsys.readouterr().out.splitlines() if line] == [
        "progress        documents_indexed=1 documents_total=1",
        "progress        documents_indexed=2 documents_total=2",
    ]


async def test_watching_an_ingest_prints_the_job_id_then_progress_then_the_summary(
    config: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instant_polling: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The point of the flag: the job is identifiable and visible while it is still running."""
    source = tmp_path / "note.txt"
    source.write_text("Either party may terminate on thirty days notice.", encoding="utf-8")
    journal = FakeJournal([{"total": 1, "indexed": 1}, {"total": 2, "indexed": 2}])
    service = watched(journal, monkeypatch)

    code = await pipeline.run_ingest(ingest_args(config, str(source), "--watch"), Console())
    printed = capsys.readouterr().out

    assert code == ExitCode.SUCCESS
    assert service.ingested is False
    assert printed.index("job             job-1") < printed.index("progress        ")
    assert printed.index("progress        documents_indexed=2 documents_total=2") < printed.index(
        "status "
    )
    assert printed.count("job             job-1") == 1


async def test_watching_still_carries_the_metadata_the_unwatched_path_carries(
    config: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instant_polling: None,
) -> None:
    """The watched path accepts and runs by hand, so it is the one that could drop metadata."""
    source = tmp_path / "note.txt"
    source.write_text("thirty days written notice", encoding="utf-8")
    service = watched(FakeJournal([{"total": 1, "indexed": 1}]), monkeypatch)

    await pipeline.run_ingest(
        ingest_args(config, str(source), "--watch", "--metadata", "team=legal"), Console()
    )

    assert service.metadata == {"team": "legal"}


async def test_an_unwatched_ingest_prints_no_progress_at_all(
    config: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instant_polling: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "note.txt"
    source.write_text("thirty days written notice", encoding="utf-8")
    service = watched(FakeJournal([{"total": 1, "indexed": 1}]), monkeypatch)

    await pipeline.run_ingest(ingest_args(config, str(source)), Console())
    printed = capsys.readouterr().out

    assert service.ingested is True
    assert "progress" not in printed
    assert "job             job-1" in printed


async def test_watching_under_json_says_so_and_leaves_one_document_on_stdout(
    config: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instant_polling: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Progress lines are suppressed under --json, so the flag has to admit it shows nothing."""
    source = tmp_path / "note.txt"
    source.write_text("thirty days written notice", encoding="utf-8")
    service = watched(FakeJournal([{"total": 1, "indexed": 1}]), monkeypatch)

    code = await pipeline.run_ingest(
        ingest_args(config, str(source), "--watch", "--json"), Console(as_json=True)
    )
    captured = capsys.readouterr()

    assert code == ExitCode.SUCCESS
    assert service.ingested is True
    assert "--watch prints nothing under --json or --quiet" in captured.err
    assert json.loads(captured.out)["job_id"] == "job-1"


async def test_watching_a_dry_run_says_there_is_no_job_to_follow(
    config: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A dry run creates no job, so following nothing in silence would be the old bug again."""
    source = tmp_path / "note.txt"
    source.write_text("Either party may terminate on thirty days notice.", encoding="utf-8")

    code = await pipeline.run_ingest(
        ingest_args(config, str(source), "--watch", "--dry-run"), Console()
    )
    captured = capsys.readouterr()

    assert code == ExitCode.SUCCESS
    assert "--watch has nothing to follow under --dry-run" in captured.err
    assert "would index" in captured.out


async def test_watching_a_reembed_follows_the_green_build(
    config: str,
    tmp_path: Path,
    adapter: FakeAdapter,
    stopped_build: StoppedBuild,
    monkeypatch: pytest.MonkeyPatch,
    instant_polling: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A re-embed builds by ingesting, so it follows its build the same way."""
    journal = FakeJournal([{"total": 1, "indexed": 1}, {"total": 2, "indexed": 2}])
    stopped_build.journal = journal
    monkeypatch.setattr(pipeline, "create_journal", lambda settings: journal)

    await pipeline.run_index(reembed(config, str(tmp_path), "--watch"), Console())
    printed = capsys.readouterr().out

    assert "job             j" in printed
    assert "progress        documents_indexed=2 documents_total=2" in printed


async def test_a_reembed_without_watch_prints_no_progress(
    config: str,
    tmp_path: Path,
    adapter: FakeAdapter,
    stopped_build: StoppedBuild,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await pipeline.run_index(reembed(config, str(tmp_path)), Console())

    assert "progress" not in capsys.readouterr().out


class IngestWaitingOnTheFollower:
    """An ingestion service that settles only once the follower has had its turn."""

    def __init__(self, started: asyncio.Event) -> None:
        """Hold the event the stand-in follower sets before it fails."""
        self.started = started

    async def accept(self, sources: Any, *, collection: str | None = None) -> JobRecord:
        return JobRecord(job_id="job-1", collection=collection or "docs", status="queued")

    async def run(self, record: JobRecord, *, metadata: dict[str, Any] | None = None) -> JobRecord:
        await self.started.wait()
        return JobRecord(
            job_id=record.job_id,
            collection=record.collection,
            status="completed",
            counts={"indexed": 2, "dead_lettered": 0},
        )

    async def close(self) -> None:
        return None


async def test_a_failing_follower_cannot_destroy_a_finished_ingest(
    config: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reporting progress is subordinate to the run: its failure must not eat the result."""
    source = tmp_path / "note.txt"
    source.write_text("Either party may terminate on thirty days notice.", encoding="utf-8")
    started = asyncio.Event()

    async def explode(
        journal: Any, job_id: str, console: Console, *, interval: float = 0.0
    ) -> None:
        started.set()
        raise RuntimeError("the journal directory vanished")

    service = IngestWaitingOnTheFollower(started)
    monkeypatch.setattr(pipeline, "_follow_job", explode)
    monkeypatch.setattr(pipeline, "IngestionService", lambda *args, **kwargs: service)
    monkeypatch.setattr(pipeline, "create_journal", lambda settings: object())
    monkeypatch.setattr(pipeline, "create_lock_store", lambda settings: object())

    with caplog.at_level("WARNING", logger="fasterrag.cli.commands.pipeline"):
        code = await pipeline.run_ingest(ingest_args(config, str(source), "--watch"), Console())
    captured = capsys.readouterr()

    assert code == ExitCode.SUCCESS
    assert "status          completed" in captured.out
    assert "indexed         2 chunks" in captured.out
    assert "[INTERNAL] --watch stopped reporting progress for job job-1" in captured.err
    assert "the journal directory vanished" in captured.err
    assert "Traceback" not in captured.err
    assert [getattr(record, "job_id", None) for record in caplog.records] == ["job-1"]


async def test_the_two_indexed_numbers_of_a_watched_run_cannot_be_confused(
    config: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    instant_polling: None,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Progress counts documents and the summary counts chunks; one run prints both."""
    source = tmp_path / "note.txt"
    source.write_text("Either party may terminate on thirty days notice.", encoding="utf-8")
    watched(FakeJournal([{"total": 2, "indexed": 2}]), monkeypatch)

    await pipeline.run_ingest(ingest_args(config, str(source), "--watch"), Console())
    printed = capsys.readouterr().out

    assert "progress        documents_indexed=2 documents_total=2" in printed
    assert "indexed         2 chunks" in printed
    assert "progress        indexed=" not in printed
