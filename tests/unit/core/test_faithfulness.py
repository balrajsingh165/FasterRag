import pytest

from fasterrag.core.context import AssembledContext, Citation
from fasterrag.core.faithfulness import (
    P3_SYSTEM_PROMPT,
    FaithfulnessVerdict,
    build_grading_prompt,
    parse_verdict,
)


def context() -> AssembledContext:
    return AssembledContext(
        text="Either party may terminate on thirty days notice.",
        citations=[Citation(chunk_id="c_a", source="s3://contracts/vendor.pdf", page=12)],
        used=1,
    )


def test_the_grading_prompt_carries_context_question_and_answer() -> None:
    prompt = build_grading_prompt(
        "what is the notice period?",
        "Thirty days.",
        context(),
        ["Either party may terminate on thirty days notice."],
    )

    assert "<context>" in prompt
    assert "thirty days notice" in prompt
    assert "Question: what is the notice period?" in prompt
    assert "<answer>\nThirty days.\n</answer>" in prompt


def test_the_context_keeps_the_markers_the_generator_used() -> None:
    prompt = build_grading_prompt("q", "a", context(), ["passage"])

    assert "[^c_a]" in prompt
    assert "s3://contracts/vendor.pdf" in prompt


def test_the_answer_comes_after_the_context_so_the_prefix_stays_cacheable() -> None:
    prompt = build_grading_prompt("q", "the answer text", context(), ["passage"])

    assert prompt.index("<context>") < prompt.index("<answer>")


def test_the_grader_is_never_told_to_author_or_improve_the_answer() -> None:
    assert "you only judge support" in P3_SYSTEM_PROMPT
    assert "not the author" in P3_SYSTEM_PROMPT


def test_a_correct_refusal_scores_full_marks_by_instruction() -> None:
    assert "insufficient scores 1.0" in P3_SYSTEM_PROMPT


def test_a_plain_json_verdict_is_parsed() -> None:
    verdict = parse_verdict('{"score": 0.93, "unsupported_claims": ["x"], "reasoning": "why"}')

    assert verdict.score == pytest.approx(0.93)
    assert verdict.unsupported_claims == ["x"]
    assert verdict.reasoning == "why"
    assert verdict.graded is True


def test_a_fenced_verdict_is_parsed() -> None:
    verdict = parse_verdict('```json\n{"score": 0.5}\n```')

    assert verdict.score == pytest.approx(0.5)


def test_a_verdict_wrapped_in_prose_is_parsed() -> None:
    verdict = parse_verdict('Here is my judgement:\n{"score": 0.25}\nHope that helps.')

    assert verdict.score == pytest.approx(0.25)


@pytest.mark.parametrize(("value", "expected"), [(1.5, 1.0), (-0.2, 0.0), (1, 1.0), (0, 0.0)])
def test_a_score_outside_the_range_is_clamped(value: float, expected: float) -> None:
    verdict = parse_verdict(f'{{"score": {value}}}')

    assert verdict.score == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "I could not grade this.",
        '{"score": "high"}',
        '{"score": true}',
        '{"score": null}',
        "{not json at all}",
        '["score", 0.9]',
        "",
    ],
)
def test_an_unusable_grader_response_is_ungraded_rather_than_zero(text: str) -> None:
    verdict = parse_verdict(text)

    assert verdict.score is None
    assert verdict.graded is False


def test_an_ungraded_verdict_never_withholds() -> None:
    assert FaithfulnessVerdict(score=None).withholds(0.7) is False


def test_a_score_below_the_threshold_withholds() -> None:
    assert FaithfulnessVerdict(score=0.38).withholds(0.7) is True


def test_a_score_at_the_threshold_does_not_withhold() -> None:
    assert FaithfulnessVerdict(score=0.7).withholds(0.7) is False


def test_malformed_claim_fields_degrade_to_empty_rather_than_failing() -> None:
    verdict = parse_verdict('{"score": 0.9, "unsupported_claims": "one claim", "reasoning": 7}')

    assert verdict.score == pytest.approx(0.9)
    assert verdict.unsupported_claims == []
    assert verdict.reasoning == ""


def test_non_string_claims_are_coerced() -> None:
    verdict = parse_verdict('{"score": 0.9, "unsupported_claims": [1, null]}')

    assert verdict.unsupported_claims == ["1", "None"]
