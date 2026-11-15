"""Every hosted model the configuration or the documents name has a published price (TASK-0554).

TASK-0543 found the Anthropic table three months stale by reading it by hand: the current
models had no entry, so `fasterrag estimate` and the cost governor counted their tokens as
unpriced. Nothing failed. This module fails instead, whenever `config.yaml`, a packaged
template, `docs/` or `userdocs/` names an OpenAI, Anthropic or Cohere model id that the price
tables in `services/estimation.py` do not price for that provider.

The changelog and the task ledger are history: a model retired years ago is named there
because it was once used, not recommended, so neither is scanned.
"""

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Final

import pytest

from fasterrag.services.estimation import EMBEDDING_PRICES, GENERATION_PRICES

ROOT: Final = Path(__file__).resolve().parents[3]

HISTORY: Final = frozenset({"CHANGELOG.md", "todo.md"})

MODEL_IDS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    ("openai", re.compile(r"(?<![\w.-])(gpt-[a-z0-9.-]*[a-z0-9])(?![\w-])")),
    ("openai", re.compile(r"(?<![\w.-])(text-embedding-[a-z0-9.-]*[a-z0-9])(?![\w-])")),
    ("openai", re.compile(r"(?<![\w./-])(o[134](?:-mini|-pro)?)(?![\w./-])")),
    ("anthropic", re.compile(r"(?<![\w.-])(claude-(?:opus|sonnet|haiku|fable|mythos)[a-z0-9-]*)")),
    ("cohere", re.compile(r"(?<![\w.-])(command-(?:r|a|light)[a-z0-9.-]*[a-z0-9])(?![\w-])")),
    ("cohere", re.compile(r"(?<![\w.-])(embed-(?:english|multilingual|v)[a-z0-9.-]*[a-z0-9])")),
)


def named_files() -> list[Path]:
    """Return every file whose model ids a user may copy into a configuration."""
    return [
        ROOT / "config.yaml",
        *sorted((ROOT / "src" / "fasterrag" / "data").glob("*.yaml")),
        *sorted(path for path in (ROOT / "docs").rglob("*.md") if path.name not in HISTORY),
        *sorted((ROOT / "userdocs").rglob("*.md")),
    ]


def unpriced(paths: Iterable[Path]) -> list[str]:
    """Return ``file:line: provider model`` for each named id with no price for its provider."""
    prices = {**EMBEDDING_PRICES, **GENERATION_PRICES}
    found = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for provider, pattern in MODEL_IDS:
                for model in pattern.findall(line):
                    entry = prices.get(model)
                    if entry is None or entry.provider != provider:
                        found.append(f"{path.name}:{number}: {provider} {model}")
    return found


def test_every_model_the_configuration_and_documents_name_has_a_price() -> None:
    """A named model with no entry is counted as unpriced spend, and nothing used to say so."""
    assert unpriced(named_files()) == []


def test_the_scan_covers_the_files_users_copy_from() -> None:
    files = {path.relative_to(ROOT).as_posix() for path in named_files()}

    assert "config.yaml" in files
    assert "src/fasterrag/data/config.local.yaml" in files
    assert "userdocs/quickstart.md" in files
    assert "docs/config-reference.md" in files
    assert "docs/CHANGELOG.md" not in files


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("llm:\n  model: gpt-9-imaginary", ["openai gpt-9-imaginary"]),
        ("model: `claude-opus-9-9`", ["anthropic claude-opus-9-9"]),
        ("`command-r7b-12-2024` is not listed", ["cohere command-r7b-12-2024"]),
        ("embeddings: embed-v4.0", ["cohere embed-v4.0"]),
        ("a model: `gpt-4o-mini` and `claude-sonnet-5-5`", []),
        ("the command-line and o3 and the embed-in-products funnel", []),
    ],
)
def test_an_unpriced_id_is_reported_and_a_priced_one_is_not(
    tmp_path: Path, line: str, expected: list[str]
) -> None:
    """The scan must catch a new id and must not flag prose that only looks like one."""
    page = tmp_path / "page.md"
    page.write_text(line, encoding="utf-8")

    assert [finding.split(": ", 1)[1] for finding in unpriced([page])] == expected
