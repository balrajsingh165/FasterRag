"""Where `autopilot generate-golden-set` gets its size from.

`autopilot.golden_set_size` was declared and read by nothing (TASK-0201). The flag carried
its own default of 100 and the setting's default is also 100, so the configured value was
ignored rather than visibly wrong — a `golden_set_size: 40` produced a hundred records and
nothing said otherwise. Parsing goes through the real parser here, because the defect lived
in the default the parser supplied.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from fasterrag.cli.commands import autopilot
from fasterrag.cli.output import Console, ExitCode
from fasterrag.cli.parser import build_parser

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
autopilot:
  enabled: false
  golden_set_size: 40
"""


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("FASTERRAG_API_KEY", "test-key")
    path = tmp_path / "config.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    return str(path)


@pytest.fixture
def requested(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the size each run asks the generator for, without calling an LLM."""
    seen: list[int] = []

    async def fake(*args: Any, size: int, **kwargs: Any) -> tuple[list[Any], dict[str, int]]:
        seen.append(size)
        return [], {}

    monkeypatch.setattr(autopilot, "generate_from_sources", fake)
    return seen


def parsed(config: str, tmp_path: Path, *extra: str) -> argparse.Namespace:
    source = tmp_path / "corpus.md"
    source.write_text("# Handbook\n\nEmployees accrue leave monthly.\n", encoding="utf-8")
    return build_parser().parse_args(
        [
            "autopilot",
            "--config",
            config,
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
    config: str, tmp_path: Path, requested: list[int]
) -> None:
    code = await autopilot.run_generate_golden_set(parsed(config, tmp_path), Console())

    assert code == ExitCode.SUCCESS
    assert requested == [40]


async def test_the_flag_still_overrides_the_configured_size(
    config: str, tmp_path: Path, requested: list[int]
) -> None:
    """Per-run override is the point of the flag; the setting is only its default."""
    await autopilot.run_generate_golden_set(parsed(config, tmp_path, "--size", "7"), Console())

    assert requested == [7]


async def test_the_size_actually_used_is_reported(
    config: str, tmp_path: Path, requested: list[int], capsys: pytest.CaptureFixture[str]
) -> None:
    """A resolved value nothing prints is the same invisibility the defect had."""
    args = parsed(config, tmp_path)
    args.as_json = True

    await autopilot.run_generate_golden_set(args, Console(as_json=True))

    assert json.loads(capsys.readouterr().out)["size"] == 40
