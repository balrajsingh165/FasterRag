"""P4 golden-set generation from a real corpus.

Implements the P4 contract of ``docs/prompts.md``: sample chunks from the corpus, ask a model
for questions a real user would actually ask about them, and emit records in the golden-set
schema the eval harness, the regression gate (D7), and Autopilot (D6) all share.

**Sampling is stratified across documents.** Drawing uniformly from a flat chunk list lets
one long document supply most of the set, and a golden set dominated by a single source
measures retrieval into that source rather than retrieval.

**A fraction of the set is deliberately unanswerable.** Without adversarial records, every
tuning pass is rewarded for answering more confidently, and a system that has learned to
guess scores identically to one that has learned to retrieve. These are also what exercise
grounded-or-refuse (D5).

**Generated records are never `source: "human"`.** They carry ``autopilot``, and promotion
requires a person. An ungoverned generated set lets the system grade its own homework: the
same model family writes the questions, answers them, and scores the answers.
"""

from __future__ import annotations

import json
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Final

from fasterrag.adapters.llm.base import LLMAdapter
from fasterrag.core.evals.golden import GoldenRecord
from fasterrag.errors import FasterRagError
from fasterrag.observability.logging import get_logger

__all__ = [
    "ADVERSARIAL_FRACTION",
    "GENERATED_SOURCE",
    "P4_SYSTEM_PROMPT",
    "P4_TEMPLATE_VERSION",
    "CandidateChunk",
    "build_generation_prompt",
    "generate_golden_set",
    "parse_generated",
    "stratified_sample",
]

P4_TEMPLATE_VERSION: Final = "1.0.0"

GENERATED_SOURCE: Final = "autopilot"

# CRITICAL: without adversarial records the gate rewards a system for answering more, not
# for retrieving better. One in five is enough to make a confident guesser score worse than
# an honest refuser without swamping the retrieval metrics the set mainly exists to measure.
ADVERSARIAL_FRACTION: Final = 0.2

P4_SYSTEM_PROMPT: Final = """\
You write evaluation questions for a retrieval system, from a passage of a real corpus.

Write questions a real user of this corpus would ask, whose answer is contained in
the passage. Use the corpus's own terminology. Make the question answerable without
seeing the passage — no "according to the text" or "in this section".
Avoid questions answerable from general knowledge alone.

Respond with JSON only:
{"query": "...", "answer_reference": "...", "unanswerable": false}"""

P4_ADVERSARIAL_PROMPT: Final = """\
You write *unanswerable* evaluation questions for a retrieval system.

Given a passage, write a question that sounds like it belongs to the same corpus and
uses its terminology, but whose answer is **not** present anywhere in the passage.
It must be specific enough that a system cannot answer it by guessing, and plausible
enough that a system might try.

Respond with JSON only:
{"query": "...", "answer_reference": null, "unanswerable": true}"""

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CandidateChunk:
    """One chunk a question may be generated from."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, Any] | None = None


def stratified_sample(
    chunks: Sequence[CandidateChunk], size: int, *, seed: int = 0
) -> list[CandidateChunk]:
    """Return ``size`` chunks spread across documents rather than drawn uniformly.

    Documents are visited round-robin, taking one chunk from each in turn, so a corpus of
    one 900-chunk manual and ten 5-chunk memos yields a set that covers the memos instead of
    one that is 95% manual.

    Args:
        chunks: Every candidate chunk.
        size: How many to draw. Returns everything when the corpus is smaller.
        seed: Makes the draw reproducible — regenerating a set with the same seed and corpus
            gives the same questions, so a metric change means the system changed.

    Returns:
        The sampled chunks.
    """
    if size <= 0 or not chunks:
        return []

    by_document: dict[str, list[CandidateChunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.document_id, []).append(chunk)

    rng = random.Random(seed)
    for group in by_document.values():
        rng.shuffle(group)

    documents = sorted(by_document)
    rng.shuffle(documents)

    sampled: list[CandidateChunk] = []
    depth = 0
    while len(sampled) < size:
        added = False
        for document in documents:
            group = by_document[document]
            if depth < len(group):
                sampled.append(group[depth])
                added = True
                if len(sampled) == size:
                    return sampled
        if not added:
            break
        depth += 1

    return sampled


def build_generation_prompt(chunk: CandidateChunk, *, adversarial: bool = False) -> str:
    """Build the P4 user turn for one chunk."""
    intent = (
        "Write one unanswerable question in the style of this corpus."
        if adversarial
        else "Write one question a real user would ask, answered by this passage."
    )
    return f"<passage>\n{chunk.text}\n</passage>\n\n{intent}"


def _extract_object(text: str) -> str | None:
    """Return the JSON object in a model response, unwrapping a code fence if present."""
    fenced = _FENCE.search(text)
    if fenced:
        return fenced.group(1)
    bare = _OBJECT.search(text)
    return bare.group(0) if bare else None


def parse_generated(text: str) -> dict[str, Any] | None:
    """Parse a P4 response, or return ``None`` when it is unusable.

    An unparseable response drops the record rather than failing the run. Generating a
    hundred questions and losing three is a smaller harm than losing the whole set, and the
    count of what was dropped is reported so the loss is never silent.
    """
    payload = _extract_object(text)
    if payload is None:
        return None

    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    query = parsed.get("query")
    if not isinstance(query, str) or not query.strip():
        return None

    reference = parsed.get("answer_reference")
    return {
        "query": query.strip(),
        "answer_reference": reference if isinstance(reference, str) else None,
        "unanswerable": bool(parsed.get("unanswerable", False)),
    }


async def generate_golden_set(
    chunks: Sequence[CandidateChunk],
    generator: LLMAdapter,
    *,
    size: int = 100,
    adversarial_fraction: float = ADVERSARIAL_FRACTION,
    seed: int = 0,
) -> tuple[list[GoldenRecord], dict[str, int]]:
    """Generate a golden set from a corpus (P4).

    Args:
        chunks: Every candidate chunk in the corpus.
        generator: The model writing the questions.
        size: How many records to aim for.
        adversarial_fraction: Share of the set generated as deliberately unanswerable.
        seed: Makes both the sampling and the adversarial selection reproducible.

    Returns:
        The records, and a tally of what happened — generated, adversarial, and dropped.
        The tally is returned rather than logged alone because a caller committing a golden
        set needs to know it asked for a hundred records and got eighty-three.
    """
    sampled = stratified_sample(chunks, size, seed=seed)
    if not sampled:
        return [], {"generated": 0, "adversarial": 0, "dropped": 0, "requested": size}

    rng = random.Random(seed)
    adversarial_count = int(len(sampled) * adversarial_fraction)
    adversarial_indices = set(rng.sample(range(len(sampled)), adversarial_count))

    created = datetime.now(tz=UTC).date().isoformat()
    records: list[GoldenRecord] = []
    dropped = 0

    for index, chunk in enumerate(sampled):
        adversarial = index in adversarial_indices
        prompt = build_generation_prompt(chunk, adversarial=adversarial)
        system = P4_ADVERSARIAL_PROMPT if adversarial else P4_SYSTEM_PROMPT

        try:
            completion = await generator.complete(prompt, system=system)
        except FasterRagError as exc:
            dropped += 1
            _logger.warning(
                "a golden-set record could not be generated",
                extra={"chunk_id": chunk.chunk_id, "code": exc.code.value},
            )
            continue

        parsed = parse_generated(completion.text)
        if parsed is None:
            dropped += 1
            continue

        # CRITICAL: an adversarial record carries no ground truth by construction. Giving it
        # the chunk it was written from would make a system that retrieves that chunk score
        # as correct, when the correct behavior is to refuse.
        unanswerable = adversarial or parsed["unanswerable"]
        records.append(
            GoldenRecord(
                id=f"q_{len(records) + 1:04d}",
                query=parsed["query"],
                source=GENERATED_SOURCE,
                created_at=created,
                relevant_chunk_ids=() if unanswerable else (chunk.chunk_id,),
                relevant_document_ids=() if unanswerable else (chunk.document_id,),
                answer_reference=None if unanswerable else parsed["answer_reference"],
                metadata=dict(chunk.metadata or {}),
            )
        )

    tally = {
        "requested": size,
        "generated": len(records),
        "adversarial": sum(1 for record in records if record.adversarial),
        "dropped": dropped,
    }
    _logger.info("golden set generated", extra=tally)
    return records, tally
