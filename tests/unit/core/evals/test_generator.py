from collections.abc import AsyncIterator

import pytest

from fasterrag.adapters.llm.base import Completion, LLMAdapter
from fasterrag.adapters.vectordb.base import HealthStatus
from fasterrag.config.schema import Settings
from fasterrag.core.evals.generator import (
    ADVERSARIAL_FRACTION,
    GENERATED_SOURCE,
    CandidateChunk,
    build_generation_prompt,
    generate_golden_set,
    parse_generated,
    stratified_sample,
)
from fasterrag.errors import GenerationError


class ScriptedGenerator(LLMAdapter):
    """Returns a scripted response per call, cycling if it runs out."""

    provider = "scripted"

    def __init__(self, settings: Settings, responses: list[str], error: Exception | None = None):
        super().__init__(settings)
        self.responses = responses
        self.error = error
        self.systems: list[str | None] = []
        self.calls = 0

    async def complete(self, prompt: str, *, system: str | None = None) -> Completion:
        self.systems.append(system)
        self.calls += 1
        if self.error is not None:
            raise self.error
        text = self.responses[(self.calls - 1) % len(self.responses)]
        return Completion(text=text, model="scripted", prompt_tokens=1, completion_tokens=1)

    async def stream(self, prompt: str, *, system: str | None = None) -> AsyncIterator[str]:
        """Never used: golden-set generation is a completion call, not a stream."""
        if self.provider:
            raise NotImplementedError
        yield ""

    async def health(self) -> HealthStatus:
        return HealthStatus(healthy=True)

    async def close(self) -> None:
        return None


def chunks(*specs: tuple[str, int]) -> list[CandidateChunk]:
    built: list[CandidateChunk] = []
    for document, count in specs:
        for index in range(count):
            built.append(
                CandidateChunk(
                    chunk_id=f"c_{document}_{index}",
                    document_id=document,
                    text=f"passage {index} of {document}",
                    metadata={"source": document},
                )
            )
    return built


ANSWERABLE = (
    '{"query": "What is the notice period?", '
    '"answer_reference": "Thirty days.", "unanswerable": false}'
)
UNANSWERABLE = (
    '{"query": "What is the CEO salary?", "answer_reference": null, "unanswerable": true}'
)


def test_an_empty_corpus_samples_nothing() -> None:
    assert stratified_sample([], 10) == []


def test_a_non_positive_size_samples_nothing() -> None:
    assert stratified_sample(chunks(("d1", 5)), 0) == []


def test_a_corpus_smaller_than_the_request_returns_everything() -> None:
    assert len(stratified_sample(chunks(("d1", 3)), 10)) == 3


def test_sampling_returns_exactly_what_was_asked_for() -> None:
    assert len(stratified_sample(chunks(("d1", 50), ("d2", 50)), 20)) == 20


def test_one_long_document_never_dominates_the_sample() -> None:
    corpus = chunks(("manual", 900), ("memo_a", 5), ("memo_b", 5))

    sampled = stratified_sample(corpus, 9)
    documents = {chunk.document_id for chunk in sampled}

    assert documents == {"manual", "memo_a", "memo_b"}
    assert sum(1 for chunk in sampled if chunk.document_id == "manual") == 3


def test_sampling_is_reproducible_for_a_seed() -> None:
    corpus = chunks(("d1", 20), ("d2", 20))

    first = [chunk.chunk_id for chunk in stratified_sample(corpus, 10, seed=7)]
    second = [chunk.chunk_id for chunk in stratified_sample(corpus, 10, seed=7)]

    assert first == second


def test_a_different_seed_draws_differently() -> None:
    corpus = chunks(("d1", 40), ("d2", 40))

    first = [chunk.chunk_id for chunk in stratified_sample(corpus, 10, seed=1)]
    second = [chunk.chunk_id for chunk in stratified_sample(corpus, 10, seed=2)]

    assert first != second


def test_no_chunk_is_sampled_twice() -> None:
    sampled = stratified_sample(chunks(("d1", 10), ("d2", 10)), 20)

    assert len({chunk.chunk_id for chunk in sampled}) == 20


def test_the_prompt_carries_the_passage() -> None:
    prompt = build_generation_prompt(chunks(("d1", 1))[0])

    assert "<passage>" in prompt
    assert "passage 0 of d1" in prompt


def test_the_adversarial_prompt_asks_for_an_unanswerable_question() -> None:
    prompt = build_generation_prompt(chunks(("d1", 1))[0], adversarial=True)

    assert "unanswerable" in prompt


def test_a_plain_response_is_parsed() -> None:
    parsed = parse_generated(ANSWERABLE)

    assert parsed is not None
    assert parsed["query"] == "What is the notice period?"
    assert parsed["answer_reference"] == "Thirty days."
    assert parsed["unanswerable"] is False


def test_a_fenced_response_is_parsed() -> None:
    assert parse_generated(f"```json\n{ANSWERABLE}\n```") is not None


def test_a_response_wrapped_in_prose_is_parsed() -> None:
    assert parse_generated(f"Here you go:\n{ANSWERABLE}\nHope that helps.") is not None


@pytest.mark.parametrize(
    "text",
    ["not json at all", '{"answer_reference": "x"}', '{"query": ""}', '{"query": 7}', "", "[1,2]"],
)
def test_an_unusable_response_is_dropped(text: str) -> None:
    assert parse_generated(text) is None


async def test_a_generated_set_carries_the_autopilot_provenance() -> None:
    settings = Settings()
    records, _ = await generate_golden_set(
        chunks(("d1", 5)), ScriptedGenerator(settings, [ANSWERABLE]), size=5
    )

    assert records
    assert all(record.source == GENERATED_SOURCE for record in records)


async def test_generated_ids_are_unique_and_sequential() -> None:
    settings = Settings()
    records, _ = await generate_golden_set(
        chunks(("d1", 6)), ScriptedGenerator(settings, [ANSWERABLE]), size=6
    )

    assert [record.id for record in records] == [f"q_{n:04d}" for n in range(1, len(records) + 1)]


async def test_an_answerable_record_points_at_the_chunk_it_came_from() -> None:
    settings = Settings()
    records, _ = await generate_golden_set(
        chunks(("d1", 4)),
        ScriptedGenerator(settings, [ANSWERABLE]),
        size=4,
        adversarial_fraction=0.0,
    )

    for record in records:
        assert len(record.relevant_chunk_ids) == 1
        assert record.relevant_document_ids == ("d1",)
        assert record.answer_reference


async def test_an_adversarial_record_carries_no_ground_truth() -> None:
    settings = Settings()
    records, _ = await generate_golden_set(
        chunks(("d1", 10)),
        ScriptedGenerator(settings, [UNANSWERABLE]),
        size=10,
        adversarial_fraction=1.0,
    )

    assert records
    for record in records:
        assert record.relevant_chunk_ids == ()
        assert record.relevant_document_ids == ()
        assert record.answer_reference is None
        assert record.adversarial is True


async def test_the_adversarial_fraction_is_honoured() -> None:
    settings = Settings()
    records, tally = await generate_golden_set(
        chunks(("d1", 10)),
        ScriptedGenerator(settings, [ANSWERABLE]),
        size=10,
        adversarial_fraction=0.2,
    )

    assert tally["adversarial"] == 2
    assert sum(1 for record in records if record.adversarial) == 2


async def test_an_adversarial_slot_ignores_a_model_that_answers_anyway() -> None:
    """A model told to write an unanswerable question sometimes writes an answerable one.

    The slot's intent decides the record, not the model's self-report: otherwise a
    disobedient model quietly removes the adversarial coverage the set depends on.
    """
    settings = Settings()
    records, _ = await generate_golden_set(
        chunks(("d1", 5)),
        ScriptedGenerator(settings, [ANSWERABLE]),
        size=5,
        adversarial_fraction=1.0,
    )

    assert all(record.adversarial for record in records)


async def test_the_adversarial_prompt_is_used_for_adversarial_slots() -> None:
    settings = Settings()
    generator = ScriptedGenerator(settings, [UNANSWERABLE])

    await generate_golden_set(chunks(("d1", 5)), generator, size=5, adversarial_fraction=1.0)

    assert all(system and "unanswerable" in system for system in generator.systems)


async def test_a_dropped_record_is_counted_not_hidden() -> None:
    settings = Settings()
    records, tally = await generate_golden_set(
        chunks(("d1", 4)),
        ScriptedGenerator(settings, ["not json"]),
        size=4,
        adversarial_fraction=0.0,
    )

    assert records == []
    assert tally["dropped"] == 4
    assert tally["requested"] == 4
    assert tally["generated"] == 0


async def test_a_provider_failure_drops_one_record_rather_than_the_run() -> None:
    settings = Settings()
    records, tally = await generate_golden_set(
        chunks(("d1", 3)),
        ScriptedGenerator(settings, [ANSWERABLE], error=GenerationError("provider is down")),
        size=3,
    )

    assert records == []
    assert tally["dropped"] == 3


async def test_an_empty_corpus_generates_nothing_without_calling_the_model() -> None:
    settings = Settings()
    generator = ScriptedGenerator(settings, [ANSWERABLE])

    records, tally = await generate_golden_set([], generator, size=10)

    assert records == []
    assert generator.calls == 0
    assert tally["requested"] == 10


def test_the_default_adversarial_fraction_is_documented() -> None:
    assert 0.0 < ADVERSARIAL_FRACTION < 1.0
