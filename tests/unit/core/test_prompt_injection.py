"""The committed adversarial set, checked without a model (TASK-0446).

``tests/eval/datasets/injection/cases.jsonl`` holds documents written to attack the prompt:
an instruction override, a break-out of the context delimiter, a forged citation, and three
attempts on the faithfulness grader. This tier asserts what can be asserted with no model
call — that every attack arrives inside a block it cannot close, that no marker in a passage
can be read as a citation, and that the grader's reply is parsed to the verdict it states.
``tests/eval/test_prompt_injection_live.py`` runs the same set through a real model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest

from fasterrag.core import generation
from fasterrag.core.context import AssembledContext, Citation
from fasterrag.core.faithfulness import build_grading_prompt, parse_verdict
from fasterrag.core.generation import build_prompt, inert

CASES_FILE: Final = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "eval"
    / "datasets"
    / "injection"
    / "cases.jsonl"
)

ATTACKS: Final = frozenset(
    {"instruction_override", "delimiter_breakout", "forged_citation", "grader_manipulation"}
)

OPENING: Final = re.compile(r"<context-([0-9a-f]{16})>\n")


@dataclass(frozen=True, slots=True)
class Case:
    """One adversarial record."""

    id: str
    attack: str
    question: str
    chunks: list[dict[str, str]]
    canary: str = ""
    forbidden_citation: str | None = None
    forged_claim: str | None = None
    answer: str | None = None
    grader_reply: str | None = None
    expected_score: float | None = None
    max_live_score: float | None = None

    @property
    def context(self) -> AssembledContext:
        """Return the context the case's chunks assemble into, in file order."""
        citations = [Citation(chunk_id=c["chunk_id"], source=c.get("source")) for c in self.chunks]
        return AssembledContext(text="", citations=citations, used=len(citations))

    @property
    def texts(self) -> list[str]:
        """Return the chunk texts, in the order of the context's citations."""
        return [chunk["text"] for chunk in self.chunks]


def load_cases() -> list[Case]:
    """Read every record of the committed adversarial set."""
    lines = CASES_FILE.read_text(encoding="utf-8").splitlines()
    return [Case(**json.loads(line)) for line in lines if line.strip()]


CASES: Final = load_cases()
GRADED: Final = [case for case in CASES if case.attack == "grader_manipulation"]


def test_the_set_covers_every_attack_it_claims_to() -> None:
    assert {case.attack for case in CASES} == ATTACKS
    assert len({case.id for case in CASES}) == len(CASES)


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_every_attack_stays_inside_a_block_it_cannot_close(case: Case) -> None:
    """A passage could end ``<context>`` early and write a question of its own after it."""
    prompt = build_prompt(case.question, case.context, case.texts)

    opened = OPENING.match(prompt)
    assert opened is not None
    closing = f"</context-{opened.group(1)}>"
    inside, after = prompt[opened.end() :].split(closing)

    assert prompt.count(opened.group(0)) == 1
    assert after == f"\n\nQuestion: {case.question}"
    for text in case.texts:
        assert inert(text) in inside
    if case.canary:
        assert case.canary not in after


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_no_passage_can_supply_a_marker_of_its_own(case: Case) -> None:
    """A marker in chunk text named a real chunk, so the resolver accepted the forgery."""
    prompt = build_prompt(case.question, case.context, case.texts)

    markers = [match.group(1) for match in generation._MARKER.finditer(prompt)]

    assert markers == [chunk["chunk_id"] for chunk in case.chunks]


@pytest.mark.parametrize("case", GRADED, ids=lambda case: case.id)
def test_the_answer_under_judgement_cannot_address_the_grader(case: Case) -> None:
    assert case.answer is not None
    prompt = build_grading_prompt(case.question, case.answer, case.context, case.texts)

    opened = OPENING.match(prompt)
    assert opened is not None
    closing = f"</answer-{opened.group(1)}>"

    assert prompt.count(closing) == 1
    assert prompt.endswith(f"{case.answer}\n{closing}")


@pytest.mark.parametrize("case", GRADED, ids=lambda case: case.id)
def test_a_grader_reply_quoting_injected_json_is_parsed_to_its_own_verdict(case: Case) -> None:
    """A greedy brace-to-brace match spanned a quoted object and the verdict: it failed open."""
    assert case.grader_reply is not None
    verdict = parse_verdict(case.grader_reply)

    assert verdict.score == pytest.approx(case.expected_score)
