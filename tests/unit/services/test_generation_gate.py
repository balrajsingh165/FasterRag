"""Generation quality is gated, offline, through ``benchmark --suite eval --faithfulness``.

Nothing evaluated an answer: ``score_faithfulness`` had no caller, and the only CI gate
scored retrieval, so a prompt change was gated by nothing (TASK-0481). These tests run the
real command over the handbook dataset with the providers CI's nightly uses where it can:
the scripted LLM from ``examples/scripted-llm`` answers *and* grades, reached through an
entry point like any installed provider, and the store and embedder are the in-memory acme
plugin. Only the prompt builder is replaced, and only by the degraded double.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, Final

import pytest

from fasterrag.cli.main import main
from fasterrag.cli.output import ExitCode
from fasterrag.config.schema import Settings
from fasterrag.core.context import AssembledContext
from fasterrag.core.generation import build_prompt
from fasterrag.facade import FasterRag
from fasterrag.services import generation as generation_module
from fasterrag.services.evaluation import (
    GENERATION_BASELINE_NAME,
    check_generation,
    load_generation_baseline,
)
from tests.support import doc_examples
from tests.support.plugins import install_distribution

ROOT: Final = Path(__file__).resolve().parents[3]
HANDBOOK: Final = ROOT / "tests" / "eval" / "datasets" / "handbook"
SCRIPTED: Final = ROOT / "examples" / "scripted-llm" / "src"

CONFIG: Final = """\
vector_db:
  provider: shared
  api_key_env: null
embeddings:
  provider: acme
  model: acme-embed
llm:
  provider: scripted
  model: scripted
  api_key_env: null
chunking:
  token_counter: estimate
retrieval:
  rerank: false
  top_k: 5
eval:
  regression_gate: false
"""

FABRICATED: Final = (
    "[^example-1]\nStaff on the orbital station accrue forty days of leave per orbit.\n\n"
)


def degraded_prompt(question: str, context: AssembledContext, texts: Sequence[str]) -> str:
    """A P1 template that leads with a worked example ahead of the evidence."""
    return FABRICATED + build_prompt(question, context, texts)


@pytest.fixture
def deployment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A deployment with the handbook corpus indexed and the scripted LLM installed."""
    install_distribution(
        tmp_path,
        monkeypatch,
        {
            "fasterrag.vectordb": {"shared": "tests.support.doc_examples:SharedVectorDB"},
            "fasterrag.embeddings": {"acme": "tests.support.acme_plugin:AcmeEmbedder"},
            "fasterrag.llm": {"scripted": "fasterrag_scripted_llm:ScriptedLLM"},
        },
        name="scripted-eval",
    )
    monkeypatch.syspath_prepend(str(SCRIPTED))
    root = tmp_path / "deployment"
    shutil.copytree(HANDBOOK, root / "handbook")
    (root / "handbook" / GENERATION_BASELINE_NAME).unlink(missing_ok=True)
    (root / "config.yaml").write_text(CONFIG, encoding="utf-8")
    monkeypatch.chdir(root)
    monkeypatch.setenv("FASTERRAG_API_KEY", "unit-test-control-plane-key")
    doc_examples.reset()
    asyncio.run(_index(root))
    yield root
    doc_examples.reset()


async def _index(root: Path) -> None:
    """Index the handbook corpus into the collection the eval suite scores."""
    async with FasterRag.from_config(root / "config.yaml") as rag:
        job = await rag.ingest([str(root / "handbook" / "corpus")], collection="handbook")
    assert job.status == "completed"


def run_eval(root: Path, capsys: pytest.CaptureFixture[str], *extra: str) -> tuple[int, Any]:
    """Run ``benchmark --suite eval --faithfulness`` and return its exit code and document."""
    capsys.readouterr()
    code = main(
        [
            "benchmark",
            "--config",
            str(root / "config.yaml"),
            "--json",
            "--suite",
            "eval",
            "--dataset",
            str(root / "handbook"),
            "--collection",
            "handbook",
            "--faithfulness",
            *extra,
        ]
    )
    return code, json.loads(capsys.readouterr().out)


def test_faithfulness_runs_only_when_asked(
    deployment: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """It costs two model calls per record, so a plain eval run must not start generating."""
    capsys.readouterr()
    code = main(
        [
            "benchmark",
            "--config",
            str(deployment / "config.yaml"),
            "--json",
            "--suite",
            "eval",
            "--dataset",
            str(deployment / "handbook"),
            "--collection",
            "handbook",
        ]
    )
    document = json.loads(capsys.readouterr().out)

    assert code == ExitCode.SUCCESS
    assert "generation" not in document
    assert "precision_at_k" in document["eval"]


def test_without_a_generation_baseline_the_gate_compares_nothing_and_fails(
    deployment: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A gate with nothing to compare against is not a pass."""
    code, document = run_eval(deployment, capsys)

    assert code == ExitCode.REGRESSION
    assert document["generation_gate"]["verdict"] == "not-comparable"


def test_the_scripted_pipeline_is_faithful_and_its_baseline_holds(
    deployment: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The honest prompt grounds every answer, so the recorded baseline gates itself green."""
    code, recorded = run_eval(deployment, capsys, "--record-generation-baseline")
    baseline = load_generation_baseline(deployment / "handbook")

    assert code == ExitCode.SUCCESS
    assert baseline is not None
    assert recorded["generation"]["answerable"] == 12
    assert recorded["generation"]["graded"] == 12
    assert baseline.mean_faithfulness == pytest.approx(1.0)
    assert baseline.citation_rate == pytest.approx(1.0)

    code, document = run_eval(deployment, capsys)

    assert code == ExitCode.SUCCESS
    assert document["generation_gate"]["verdict"] == "within-tolerance"


def test_a_degraded_prompt_fails_the_generation_gate(
    deployment: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect: a prompt that let answers drift from their evidence was gated by nothing."""
    run_eval(deployment, capsys, "--record-generation-baseline")
    monkeypatch.setattr(generation_module, "build_prompt", degraded_prompt)

    code, document = run_eval(deployment, capsys)

    gate = document["generation_gate"]
    assert code == ExitCode.REGRESSION
    assert gate["verdict"] == "regressed"
    assert any(failure.startswith("mean_faithfulness fell") for failure in gate["failures"])
    assert any(failure.startswith("citation_rate fell") for failure in gate["failures"])
    assert document["generation"]["mean_faithfulness"] == 0.0


def test_a_baseline_from_another_model_is_not_comparable(
    deployment: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A score from another model measures the model, so it is refused rather than compared."""
    run_eval(deployment, capsys, "--record-generation-baseline")
    code, document = run_eval(deployment, capsys, "--set", "llm.model=another-model")

    assert code == ExitCode.REGRESSION
    assert document["generation_gate"]["verdict"] == "not-comparable"
    assert any("llm.model" in obstacle for obstacle in document["generation_gate"]["obstacles"])


def test_the_gate_reads_a_missing_baseline_as_an_obstacle() -> None:
    gate = check_generation({"mean_faithfulness": 1.0}, None, Settings.model_validate({}))

    assert not gate.passed
    assert gate.verdict.value == "not-comparable"
