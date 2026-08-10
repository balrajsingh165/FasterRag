"""The ``autopilot`` wrappers (D6): the never-writes-config guard, both refusals, and --size.

``autopilot.golden_set_size`` was declared and read by nothing (TASK-0201). The flag carried
its own default of 100 and the setting's default is also 100, so the configured value was
ignored rather than visibly wrong — a ``golden_set_size: 40`` produced a hundred records and
nothing said otherwise. The size tests parse through the real parser, because the defect
lived in the default the parser supplied.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from fasterrag.cli.commands import autopilot as autopilot_command
from fasterrag.cli.main import main
from fasterrag.cli.output import Console, ExitCode
from fasterrag.cli.parser import build_parser
from fasterrag.errors import FasterRagError
from fasterrag.services.autopilot import Candidate, Suggestion, TrialResult
from tests.unit.cli.conftest import Closeable, corpus


class FakeRouter(Closeable):
    """A tiering router the wrapper only builds, hands on, and closes."""

    default = object()


def trial(label: dict[str, Any], ndcg: float) -> TrialResult:
    return TrialResult(candidate=Candidate(label), recall_at_k=ndcg, mrr=ndcg, ndcg_at_k=ndcg, k=5)


def suggestion(*, best_ndcg: float = 0.9, baseline_ndcg: float = 0.5) -> Suggestion:
    baseline = trial({}, baseline_ndcg)
    best = trial({"retrieval.top_k": 8}, best_ndcg)
    return Suggestion(
        baseline=baseline,
        best=best,
        trials=[baseline, best],
        evaluated=2,
        skipped=1,
        seconds=1.5,
        created_at="2026-08-09T00:00:00Z",
    )


class Harness:
    """The doubles an autopilot run needs, plus the knobs a test turns."""

    def __init__(self) -> None:
        self.adapter = Closeable()
        self.router = FakeRouter()
        self.suggestion = suggestion()
        self.error: Exception | None = None
        self.writes_config_to: Path | None = None
        self.collections: list[str] = []


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    built = Harness()

    async def fake_tune(dataset: object, settings: object, *args: Any, **kwargs: Any) -> Suggestion:
        built.collections.append(str(kwargs["collection"]))
        if built.writes_config_to is not None:
            built.writes_config_to.write_text("app:\n  port: 9999\n", encoding="utf-8")
        if built.error is not None:
            raise built.error
        return built.suggestion

    monkeypatch.setattr(autopilot_command, "create_vector_db_adapter", lambda s: built.adapter)
    monkeypatch.setattr(autopilot_command, "create_embedding_router", lambda s: built.router)
    monkeypatch.setattr(autopilot_command, "load_dataset", lambda root: object())
    monkeypatch.setattr(autopilot_command, "tune", fake_tune)
    monkeypatch.setattr(autopilot_command, "render_suggestion", lambda item: "# suggestion\n")
    return built


def run_autopilot(config: str, tmp_path: Path, *extra: str) -> int:
    """Run ``autopilot run`` with --out always inside the test's directory."""
    return main(
        [
            "autopilot",
            "--config",
            config,
            "run",
            "--dataset",
            str(tmp_path),
            "--out",
            str(tmp_path / "suggestion.yaml"),
            *extra,
        ]
    )


def test_a_run_without_a_dataset_is_refused_before_any_backend_is_built(
    config: str, harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["autopilot", "--config", config, "run"])

    assert code == ExitCode.USAGE
    assert "--dataset" in capsys.readouterr().err
    assert harness.adapter.closed == 0


def test_a_run_writes_its_suggestion_beside_the_config_and_never_into_it(
    config: str, tmp_path: Path, harness: Harness
) -> None:
    before = Path(config).read_bytes()

    code = run_autopilot(config, tmp_path)

    assert code == ExitCode.SUCCESS
    assert (tmp_path / "suggestion.yaml").read_text(encoding="utf-8") == "# suggestion\n"
    assert Path(config).read_bytes() == before


def test_a_run_that_touches_config_yaml_fails_loudly(
    config: str, tmp_path: Path, harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole of D6 rests on this; an unchecked promise is a hope."""
    harness.writes_config_to = Path(config)

    code = run_autopilot(config, tmp_path)

    assert code == ExitCode.FAILURE
    assert "must never write it" in capsys.readouterr().err


def test_every_trial_is_printed_not_just_the_winner(
    config: str, tmp_path: Path, harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    out = (run_autopilot(config, tmp_path), capsys.readouterr().out)[1]

    assert "baseline (current configuration)" in out
    assert "retrieval.top_k=8" in out


def test_the_winning_trial_is_starred(
    config: str, tmp_path: Path, harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    run_autopilot(config, tmp_path)

    starred = [line for line in capsys.readouterr().out.splitlines() if line.startswith(" *")]
    assert len(starred) == 1
    assert "retrieval.top_k=8" in starred[0]


def test_a_search_that_beat_nothing_says_so(
    config: str, tmp_path: Path, harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    harness.suggestion = suggestion(best_ndcg=0.5, baseline_ndcg=0.5)

    run_autopilot(config, tmp_path)

    assert "no candidate beat the current configuration" in capsys.readouterr().out


def test_the_skipped_count_is_reported(
    config: str, tmp_path: Path, harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    run_autopilot(config, tmp_path)

    assert "skipped         1 at the budget" in capsys.readouterr().out


def test_the_budget_flag_is_converted_from_minutes_to_seconds(
    config: str, tmp_path: Path, harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[float] = []

    async def record(dataset: object, settings: object, *args: Any, **kwargs: Any) -> Suggestion:
        seen.append(float(kwargs["budget_seconds"]))
        return harness.suggestion

    monkeypatch.setattr(autopilot_command, "tune", record)

    run_autopilot(config, tmp_path, "--budget-minutes", "2")

    assert seen == [120.0]


def test_the_collection_flag_beats_the_configured_default(
    config: str, tmp_path: Path, harness: Harness
) -> None:
    run_autopilot(config, tmp_path, "--collection", "legal")

    assert harness.collections == ["legal"]


def test_the_configured_collection_is_used_when_none_is_named(
    config: str, tmp_path: Path, harness: Harness
) -> None:
    run_autopilot(config, tmp_path)

    assert harness.collections == ["default"]


def test_a_run_as_json_is_one_document_with_no_prose(
    config: str, tmp_path: Path, harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    run_autopilot(config, tmp_path, "--json")

    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["applied"] is False
    assert payload["suggestion_file"].endswith("suggestion.yaml")
    assert "trials (ndcg" not in out


@pytest.mark.parametrize(
    ("retryable", "expected"),
    [(True, ExitCode.UNREACHABLE), (False, ExitCode.FAILURE)],
)
def test_a_failed_run_maps_the_error_to_its_documented_code(
    config: str, tmp_path: Path, harness: Harness, retryable: bool, expected: ExitCode
) -> None:
    harness.error = FasterRagError("the backend is not answering", retryable=retryable)

    assert run_autopilot(config, tmp_path) == expected


def test_a_failed_run_still_closes_the_router_and_the_adapter(
    config: str, tmp_path: Path, harness: Harness
) -> None:
    harness.error = FasterRagError("boom", retryable=False)

    run_autopilot(config, tmp_path)

    assert (harness.router.closed, harness.adapter.closed) == (1, 1)


def test_a_run_refuses_an_invalid_config(bad_config: str, tmp_path: Path) -> None:
    assert run_autopilot(bad_config, tmp_path) == ExitCode.USAGE


class Generated:
    """Records how the golden-set generator was called."""

    def __init__(self) -> None:
        self.sources: list[str] = []
        self.error: Exception | None = None


@pytest.fixture
def generator(monkeypatch: pytest.MonkeyPatch) -> Generated:
    recorded = Generated()

    async def fake(
        sources: list[str], settings: object, **kwargs: Any
    ) -> tuple[list[object], dict[str, int]]:
        recorded.sources = list(sources)
        if recorded.error is not None:
            raise recorded.error
        Path(kwargs["destination"]).write_text("{}\n", encoding="utf-8")
        return [object(), object()], {"answerable": 2}

    monkeypatch.setattr(autopilot_command, "generate_from_sources", fake)
    return recorded


def test_a_directory_of_sources_is_expanded_into_its_files(
    config: str, tmp_path: Path, generator: Generated
) -> None:
    """Passing the corpus folder is how anybody would name a corpus; it reached the service raw."""
    source = corpus(tmp_path / "corpus")

    code = main(
        [
            "autopilot",
            "--config",
            config,
            "generate-golden-set",
            str(source),
            "--out",
            str(tmp_path / "golden.jsonl"),
        ]
    )

    assert code == ExitCode.SUCCESS
    assert sorted(Path(name).name for name in generator.sources) == ["a.txt", "b.txt"]


def test_an_existing_golden_set_is_never_overwritten(
    config: str, tmp_path: Path, generator: Generated, capsys: pytest.CaptureFixture[str]
) -> None:
    """It is hand-curated after generation, so overwriting it destroys the curation."""
    destination = tmp_path / "golden.jsonl"
    destination.write_text("curated", encoding="utf-8")

    code = main(
        [
            "autopilot",
            "--config",
            config,
            "generate-golden-set",
            str(tmp_path),
            "--out",
            str(destination),
        ]
    )

    assert code == ExitCode.USAGE
    assert destination.read_text(encoding="utf-8") == "curated"
    assert "already exists" in capsys.readouterr().err


def test_a_generated_set_reports_its_tally(
    config: str, tmp_path: Path, generator: Generated, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "autopilot",
            "--config",
            config,
            "generate-golden-set",
            str(corpus(tmp_path / "corpus")),
            "--out",
            str(tmp_path / "golden.jsonl"),
        ]
    )

    out = capsys.readouterr().out
    assert "records         2" in out
    assert "answerable" in out
    assert "they are generated" in out


def test_a_generated_set_as_json_is_one_document(
    config: str, tmp_path: Path, generator: Generated, capsys: pytest.CaptureFixture[str]
) -> None:
    main(
        [
            "autopilot",
            "--config",
            config,
            "generate-golden-set",
            str(corpus(tmp_path / "corpus")),
            "--out",
            str(tmp_path / "golden.jsonl"),
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["records"] == 2
    assert payload["answerable"] == 2


def test_a_generator_failure_exits_one(
    config: str, tmp_path: Path, generator: Generated, capsys: pytest.CaptureFixture[str]
) -> None:
    generator.error = FasterRagError("no readable chunks were produced", retryable=False)

    code = main(
        [
            "autopilot",
            "--config",
            config,
            "generate-golden-set",
            str(corpus(tmp_path / "corpus")),
            "--out",
            str(tmp_path / "golden.jsonl"),
        ]
    )

    assert code == ExitCode.FAILURE
    assert "no readable chunks" in capsys.readouterr().err


def test_generation_refuses_an_invalid_config(bad_config: str, tmp_path: Path) -> None:
    code = main(
        [
            "autopilot",
            "--config",
            bad_config,
            "generate-golden-set",
            str(tmp_path),
            "--out",
            str(tmp_path / "golden.jsonl"),
        ]
    )

    assert code == ExitCode.USAGE


SIZED_CONFIG = """
vector_db:
  provider: qdrant
  mode: external
  api_key_env: null
embeddings:
  provider: huggingface
llm:
  provider: ollama
  api_key_env: null
autopilot:
  enabled: false
  golden_set_size: 40
"""


@pytest.fixture
def sized_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key")
    path = tmp_path / "config.yaml"
    path.write_text(SIZED_CONFIG, encoding="utf-8")
    return str(path)


@pytest.fixture
def requested(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the size each run asks the generator for, without calling an LLM."""
    seen: list[int] = []

    async def fake(*args: Any, size: int, **kwargs: Any) -> tuple[list[Any], dict[str, int]]:
        seen.append(size)
        return [], {}

    monkeypatch.setattr(autopilot_command, "generate_from_sources", fake)
    return seen


def parsed(sized_config: str, tmp_path: Path, *extra: str) -> argparse.Namespace:
    source = tmp_path / "corpus.md"
    source.write_text("# Handbook\n\nEmployees accrue leave monthly.\n", encoding="utf-8")
    return build_parser().parse_args(
        [
            "autopilot",
            "--config",
            sized_config,
            "generate-golden-set",
            str(source),
            "--out",
            str(tmp_path / "golden.jsonl"),
            *extra,
        ]
    )


def test_the_flag_carries_no_default_of_its_own() -> None:
    """A second hardcoded 100 is what let the configured value be ignored silently."""
    args = build_parser().parse_args(["autopilot", "generate-golden-set", "corpus.md"])

    assert args.size is None


async def test_the_configured_size_reaches_the_generator(
    sized_config: str, tmp_path: Path, requested: list[int]
) -> None:
    code = await autopilot_command.run_generate_golden_set(
        parsed(sized_config, tmp_path), Console()
    )

    assert code == ExitCode.SUCCESS
    assert requested == [40]


async def test_the_flag_still_overrides_the_configured_size(
    sized_config: str, tmp_path: Path, requested: list[int]
) -> None:
    """Per-run override is the point of the flag; the setting is only its default."""
    await autopilot_command.run_generate_golden_set(
        parsed(sized_config, tmp_path, "--size", "7"), Console()
    )

    assert requested == [7]


async def test_the_size_actually_used_is_reported(
    sized_config: str, tmp_path: Path, requested: list[int], capsys: pytest.CaptureFixture[str]
) -> None:
    """A resolved value nothing prints is the same invisibility the defect had."""
    args = parsed(sized_config, tmp_path)
    args.as_json = True

    await autopilot_command.run_generate_golden_set(args, Console(as_json=True))

    assert json.loads(capsys.readouterr().out)["size"] == 40
