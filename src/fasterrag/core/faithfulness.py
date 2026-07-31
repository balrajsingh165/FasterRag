"""P3 faithfulness scoring: how much of an answer its context actually supports.

Implements the P3 contract of ``docs/prompts.md``, the score that gates grounded-or-refuse
(D5). Three properties of that contract drive this module's shape:

**It is a separate call.** The grader never sees P1's generation instructions, so it cannot
justify an answer by the instructions that produced it. Nothing here imports the P1 prompt,
and the caller passes a distinct adapter when it wants a cheaper grading model.

**A grader failure yields ``None``, never a low score.** ``None`` means "not graded" and can
never withhold an answer. Failing closed on a grader outage would convert a monitoring
problem into an availability one — an unavailable grader is not evidence of hallucination.

**The output is strict JSON.** Providers wrap JSON in prose and code fences often enough that
tolerating both is required rather than lenient; anything that still will not parse is
treated as a grader failure, which is to say ``None``.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from fasterrag.adapters.llm.base import LLMAdapter
from fasterrag.core.context import AssembledContext
from fasterrag.core.generation import build_context_block
from fasterrag.errors import FasterRagError
from fasterrag.observability.logging import get_logger

__all__ = [
    "P3_SYSTEM_PROMPT",
    "P3_TEMPLATE_VERSION",
    "UNGRADED",
    "FaithfulnessVerdict",
    "build_grading_prompt",
    "grade",
    "parse_verdict",
]

P3_TEMPLATE_VERSION: Final = "1.0.0"

P3_SYSTEM_PROMPT: Final = """\
You grade whether an answer is supported by its context. You are not the author of
the answer and you do not improve it — you only judge support.

Break the answer into factual claims. For each, decide whether the context directly
supports it. Score = supported claims / total claims. Opinions, hedges, and explicit
statements of uncertainty are not claims and are excluded from the count.
An answer that correctly says the context is insufficient scores 1.0.

Respond with JSON only: {"score": <0-1>, "unsupported_claims": [...], "reasoning": "..."}"""

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FaithfulnessVerdict:
    """The grader's judgement of one answer."""

    score: float | None
    unsupported_claims: list[str] = field(default_factory=list)
    reasoning: str = ""

    @property
    def graded(self) -> bool:
        """Return whether a score was actually produced."""
        return self.score is not None

    def withholds(self, threshold: float) -> bool:
        """Return whether this verdict should withhold the answer.

        An ungraded verdict never withholds: ``None`` means the grader did not run, which
        is not evidence against the answer.
        """
        return self.score is not None and self.score < threshold


UNGRADED: Final = FaithfulnessVerdict(score=None)


def build_grading_prompt(
    question: str, answer: str, context: AssembledContext, texts: Sequence[str]
) -> str:
    """Build the P3 user turn: context, question, and the answer under judgement.

    Args:
        question: The original question.
        answer: The generated answer being graded.
        context: The context that was supplied to the generator.
        texts: The packed chunk texts, in the same order as the context's citations.

    Returns:
        The grading prompt. The context is rendered with the same markers P1 used, so the
        grader can name the chunk that supports a claim; the answer goes last so the
        stable context prefix stays cacheable across the P1 and P3 calls of one query.
    """
    block = build_context_block(context, texts)
    return (
        f"<context>\n{block}\n</context>\n\nQuestion: {question}\n\n<answer>\n{answer}\n</answer>"
    )


def _extract_object(text: str) -> str | None:
    """Return the JSON object in ``text``, unwrapping a code fence if present."""
    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group(1)

    bare = _OBJECT.search(text)
    return bare.group(0) if bare else None


def _coerce_score(value: Any) -> float | None:
    """Return ``value`` as a score clamped to the 0-1 range, or ``None`` if not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return max(0.0, min(1.0, float(value)))


def parse_verdict(text: str) -> FaithfulnessVerdict:
    """Parse a grader response into a verdict.

    Args:
        text: Whatever the grading model returned.

    Returns:
        The parsed verdict, or an ungraded one when the response is not usable. An
        unparseable grader response is a grader failure, and a grader failure never
        withholds an answer.
    """
    payload = _extract_object(text)
    if payload is None:
        _logger.warning("faithfulness grader returned no JSON object", extra={"length": len(text)})
        return UNGRADED

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        _logger.warning("faithfulness grader returned malformed JSON", extra={"error": str(exc)})
        return UNGRADED

    if not isinstance(parsed, dict):
        _logger.warning(
            "faithfulness grader returned a non-object", extra={"kind": type(parsed).__name__}
        )
        return UNGRADED

    score = _coerce_score(parsed.get("score"))
    if score is None:
        _logger.warning("faithfulness grader returned no usable score")
        return UNGRADED

    claims = parsed.get("unsupported_claims")
    reasoning = parsed.get("reasoning")
    return FaithfulnessVerdict(
        score=score,
        unsupported_claims=[str(claim) for claim in claims] if isinstance(claims, list) else [],
        reasoning=reasoning if isinstance(reasoning, str) else "",
    )


async def grade(
    grader: LLMAdapter,
    question: str,
    answer: str,
    context: AssembledContext,
    texts: Sequence[str],
) -> FaithfulnessVerdict:
    """Score how well ``context`` supports ``answer``.

    Args:
        grader: The grading provider; typically a cheaper model than the generator.
        question: The original question.
        answer: The generated answer.
        context: The context the answer was generated from.
        texts: The packed chunk texts, in the same order as the context's citations.

    Returns:
        The verdict, or an ungraded one if the grader failed. Grading is best-effort by
        contract, so a provider failure is logged and turned into ``None`` rather than
        propagated — the answer it was grading is still valid.
    """
    prompt = build_grading_prompt(question, answer, context, texts)
    try:
        completion = await grader.complete(prompt, system=P3_SYSTEM_PROMPT)
    except FasterRagError as exc:
        _logger.warning(
            "faithfulness grading failed; the answer is returned ungated",
            extra={"code": exc.code.value, "trace_id": exc.trace_id},
        )
        return UNGRADED

    return parse_verdict(completion.text)
