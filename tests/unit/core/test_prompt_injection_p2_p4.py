"""The adversarial set, applied to enrichment (P2) and golden-set generation (P4).

TASK-0540: TASK-0446 bounded P1 and P3, but P2 and P4 still placed document text between
fixed ``<document>``, ``<chunk>`` and ``<passage>`` tags that the text itself could close, and
left citation markers in it live. P2's output is stored with the chunk, so an instruction
obeyed there persists in the index. ``p2_p4_cases.jsonl`` carries attacks written for these
two call sites; the P1 set's chunk texts are replayed through both as well.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Final

import pytest

from fasterrag.core import generation
from fasterrag.core.chunking.enrichment import (
    P2_SYSTEM_PROMPT,
    P2_TEMPLATE_VERSION,
    build_enrichment_prompt,
    system_prompt,
)
from fasterrag.core.evals.generator import (
    P4_ADVERSARIAL_PROMPT,
    P4_SYSTEM_PROMPT,
    P4_TEMPLATE_VERSION,
    CandidateChunk,
    build_generation_prompt,
)
from fasterrag.core.generation import inert
from tests.unit.core.test_prompt_injection import CASES

CASES_FILE: Final = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "eval"
    / "datasets"
    / "injection"
    / "p2_p4_cases.jsonl"
)

BLOCK: Final = re.compile(r"<(document|chunk|passage)-([0-9a-f]{16})>\n")


def load_cases() -> list[dict[str, Any]]:
    """Read every record of the P2/P4 adversarial set."""
    lines = CASES_FILE.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


P2_CASES: Final = [case for case in load_cases() if case["call_site"] == "p2"] + [
    {"id": f"{case.id}-as-p2", "document": text, "chunk": text, "canary": case.canary}
    for case in CASES
    for text in case.texts
]
P4_CASES: Final = [case for case in load_cases() if case["call_site"] == "p4"] + [
    {"id": f"{case.id}-as-p4", "chunk": text, "canary": case.canary}
    for case in CASES
    for text in case.texts
]


def blocks(prompt: str) -> tuple[list[tuple[str, str]], str]:
    """Split a prompt into its enclosed blocks and the text left outside every block."""
    found: list[tuple[str, str]] = []
    outside = prompt
    for opened in BLOCK.finditer(prompt):
        tag, boundary = opened.groups()
        closing = f"</{tag}-{boundary}>"
        assert prompt.count(opened.group(0)) == 1
        assert prompt.count(closing) == 1
        start = opened.end()
        body = prompt[start : prompt.index(closing, start)].removesuffix("\n")
        found.append((tag, body))
        outside = outside.replace(f"{opened.group(0)}{body}\n{closing}", "")
    return found, outside


def test_the_set_covers_both_call_sites_and_the_attacks_that_apply() -> None:
    cases = load_cases()
    assert {case["call_site"] for case in cases} == {"p2", "p4"}
    for site in ("p2", "p4"):
        attacks = {case["attack"] for case in cases if case["call_site"] == site}
        assert attacks >= {"delimiter_breakout", "forged_citation"}
    assert len({case["id"] for case in cases}) == len(cases)


@pytest.mark.parametrize("case", P2_CASES, ids=lambda case: case["id"])
def test_a_document_or_chunk_cannot_close_its_p2_block(case: dict[str, Any]) -> None:
    """A document could end `<document>` early and write a chunk and instructions of its own."""
    prompt = build_enrichment_prompt(case["document"], case["chunk"])

    found, outside = blocks(prompt)

    assert found == [("document", inert(case["document"])), ("chunk", inert(case["chunk"]))]
    assert outside.strip() == ""
    if case["canary"]:
        assert case["canary"] not in outside


@pytest.mark.parametrize("case", P4_CASES, ids=lambda case: case["id"])
@pytest.mark.parametrize("adversarial", [False, True])
def test_a_passage_cannot_close_its_p4_block(case: dict[str, Any], adversarial: bool) -> None:
    """A passage could end `<passage>` and replace the instruction that follows it."""
    chunk = CandidateChunk(chunk_id="c_1", document_id="d_1", text=case["chunk"])
    prompt = build_generation_prompt(chunk, adversarial=adversarial)

    found, outside = blocks(prompt)

    assert found == [("passage", inert(case["chunk"]))]
    assert prompt.startswith("<passage-")
    if case["canary"]:
        assert case["canary"] not in outside


@pytest.mark.parametrize("case", P2_CASES + P4_CASES, ids=lambda case: case["id"])
def test_no_marker_in_p2_or_p4_text_stays_live(case: dict[str, Any]) -> None:
    """P2's output is stored with the chunk, so a live marker there outlives the prompt."""
    chunk = CandidateChunk(chunk_id="c_1", document_id="d_1", text=case["chunk"])
    prompts = [
        build_enrichment_prompt(case.get("document", case["chunk"]), case["chunk"]),
        build_generation_prompt(chunk),
    ]

    for prompt in prompts:
        assert generation._MARKER.search(prompt) is None


def test_every_chunk_of_one_document_shares_the_cacheable_prefix() -> None:
    """The document block leads every P2 prompt so a provider's prompt cache can absorb it."""
    document = "Travel policy.\n\nMeals are 41 GBP.\n\nMileage is 45p."
    first = build_enrichment_prompt(document, "Meals are 41 GBP.")
    second = build_enrichment_prompt(document, "Mileage is 45p.")

    prefix = first.split("\n\n<chunk-", 1)[0]
    assert prefix.startswith("<document-")
    assert second.startswith(prefix)


@pytest.mark.parametrize(
    "prompt", [P2_SYSTEM_PROMPT, system_prompt(120), P4_SYSTEM_PROMPT, P4_ADVERSARIAL_PROMPT]
)
def test_each_system_prompt_says_the_text_is_data_not_instructions(prompt: str) -> None:
    assert "data, not instructions" in prompt
    assert "only the closing tag with that same id" in " ".join(prompt.split())


def test_the_template_versions_moved_with_the_wording() -> None:
    """A prompt-contract change is recorded with each chunk (P2) and each generated set (P4)."""
    assert P2_TEMPLATE_VERSION == "p2.2.0"
    assert P4_TEMPLATE_VERSION == "1.1.0"


def section(name: str) -> str:
    """Return one call site's section of docs/prompts.md."""
    document = (Path(__file__).resolve().parents[3] / "docs" / "prompts.md").read_text("utf-8")
    return document.split(f"## {name} ", 1)[1].split("\n## ", 1)[0]


def test_the_documented_p2_system_prompt_is_the_one_the_code_sends() -> None:
    """docs/prompts.md showed a P2 prompt with a different token target from the code's."""
    p2 = section("P2")
    assert p2.split("SYSTEM\n", 1)[1].split("\n\nUSER\n", 1)[0] == P2_SYSTEM_PROMPT


def test_the_documented_p4_system_prompts_are_the_ones_the_code_sends() -> None:
    p4 = section("P4")
    systems = [
        part.split("\n\nUSER\n", 1)[0].split("\n```", 1)[0] for part in p4.split("SYSTEM\n")[1:]
    ]
    assert systems == [P4_SYSTEM_PROMPT, P4_ADVERSARIAL_PROMPT]
