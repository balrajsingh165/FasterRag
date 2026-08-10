import argparse
import json
from collections.abc import AsyncIterator
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
from fasterrag.services.journal import JobRecord
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
    """An ingestion service whose build fails, so a reembed stops after handing over sources.

    The swap, the eval gate, and the alias flip have their own tests. What is asserted here
    is the one thing only the wrapper does: what it decided the sources were.
    """

    def __init__(self) -> None:
        self.sources: list[str] = []

    async def ingest(self, sources: Any, *, collection: str) -> JobRecord:
        self.sources = list(sources)
        return JobRecord(job_id="j", collection=collection, status="failed")

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
    """Build a reembed namespace through the real parser.

    Going through ``build_parser`` is the point: the handler already read ``args.recursive``
    while the parser declared it nowhere, so a hand-built namespace would assert the flag
    works while the command line still rejected it.
    """
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


async def test_a_reembed_says_watch_is_not_implemented_rather_than_ignoring_it(
    config: str,
    tmp_path: Path,
    adapter: FakeAdapter,
    stopped_build: StoppedBuild,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Accepting a flag and doing nothing leaves the operator waiting on a finished job."""
    await pipeline.run_index(reembed(config, str(tmp_path), "--watch"), Console())

    assert "--watch is not implemented yet" in capsys.readouterr().err


async def test_a_reembed_without_watch_prints_no_such_notice(
    config: str,
    tmp_path: Path,
    adapter: FakeAdapter,
    stopped_build: StoppedBuild,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await pipeline.run_index(reembed(config, str(tmp_path)), Console())

    assert "--watch" not in capsys.readouterr().err
