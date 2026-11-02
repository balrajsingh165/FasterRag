"""The eval comparability fingerprint, and the two directions it has to get right.

One hash used to answer two questions. ``index.lock`` asks "has the index drifted from the
config that built it?", where *any* difference counts because the index really was built
under the old config. A baseline asks "is this result comparable to the recorded one?",
where only settings that change what retrieval returns count — and a knob added later whose
default preserves behaviour must not retire a baseline.

Both errors are tested here because they are not symmetric. Too broad retires good
baselines: visible the moment it happens. Too narrow silently compares runs that are not
comparable: invisible, and the reason the exhaustiveness case below exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from fasterrag.config.schema import Settings
from fasterrag.services.regression import (
    BEHAVIOUR_NEUTRAL_FIELDS,
    COMPARABILITY_FIELDS,
    COMPARABILITY_SECTIONS,
    COMPARABILITY_VERSION,
    Baseline,
    comparability_fingerprint,
    load_baseline,
)

DATASETS = Path(__file__).resolve().parents[3] / "tests" / "eval" / "datasets"

# CRITICAL: one entry per COMPARABILITY_FIELD, each moving that field and nothing else. The
# coverage case below fails when a field is added without one, so no listed field can go
# untested — which is how a field could end up listed and inert.
MUTATIONS: dict[str, dict[str, Any]] = {
    "chunking.strategy": {"chunking": {"strategy": "fixed"}},
    "chunking.chunk_size": {"chunking": {"chunk_size": 512}},
    "chunking.overlap": {"chunking": {"overlap": 32}},
    "chunking.token_counter": {"chunking": {"token_counter": "estimate"}},
    "chunking.chars_per_token": {"chunking": {"chars_per_token": 3}},
    "chunking.semantic_percentile": {"chunking": {"semantic_percentile": 0.90}},
    "chunking.contextual_enrichment": {"chunking": {"contextual_enrichment": True}},
    "chunking.context_tokens": {"chunking": {"context_tokens": 100}},
    "embeddings.provider": {"embeddings": {"provider": "ollama"}},
    "embeddings.model": {"embeddings": {"model": "other/model"}},
    "embeddings.dimensions": {"embeddings": {"dimensions": 384}},
    "embeddings.tiering": {
        "embeddings": {
            "tiering": {
                "enabled": True,
                "rules": [
                    {
                        "match": {"priority_class": "bulk"},
                        "provider": "huggingface",
                        "model": "BAAI/bge-small-en-v1.5",
                    }
                ],
            }
        }
    },
    "retrieval.top_k": {"retrieval": {"top_k": 7}},
    "retrieval.hybrid": {"retrieval": {"hybrid": False}},
    "retrieval.bm25_weight": {"retrieval": {"bm25_weight": 0.5}},
    "retrieval.dense_weight": {"retrieval": {"dense_weight": 0.5}},
    "retrieval.rrf_k": {"retrieval": {"rrf_k": 30}},
    "retrieval.bm25_k1": {"retrieval": {"bm25_k1": 1.5}},
    "retrieval.bm25_b": {"retrieval": {"bm25_b": 0.5}},
    "retrieval.rerank": {"retrieval": {"rerank": False}},
    "retrieval.reranker_model": {"retrieval": {"reranker_model": "other/reranker"}},
    "retrieval.rerank_top_n": {"retrieval": {"rerank_top_n": 200}},
}

NEUTRAL_MUTATIONS: dict[str, dict[str, Any]] = {
    "embeddings.api_key_env": {"embeddings": {"api_key_env": "SOME_OTHER_KEY"}},
    "embeddings.batch_size": {"embeddings": {"batch_size": 32}},
    "embeddings.cache": {"embeddings": {"cache": {"enabled": False, "backend": "memory"}}},
}


def canonical() -> Settings:
    """Return the shipped defaults, with the gate on."""
    return Settings.model_validate({"eval": {"regression_gate": True}})


def recorded(under: Settings, *, omitting: str | None = None) -> Baseline:
    """Return a baseline recorded under ``under``, optionally predating one field."""
    fingerprint = comparability_fingerprint(under)
    if omitting is not None:
        fingerprint.pop(omitting)
    return Baseline(
        recorded_at="2026-11-02T00:00:00+00:00",
        k=5,
        recall_at_k=1.0,
        mrr=1.0,
        ndcg_at_k=1.0,
        scored=12,
        embedding_model=under.embeddings.model,
        config_hash="not-used-for-comparability",
        fingerprint_version=COMPARABILITY_VERSION,
        fingerprint=fingerprint,
    )


def test_every_setting_of_a_covered_section_is_classified() -> None:
    """A knob added to the schema cannot be silently left out of the comparison."""
    live = {
        f"{section}.{name}"
        for section in COMPARABILITY_SECTIONS
        for name in type(getattr(canonical(), section)).model_fields
    }

    assert live == set(COMPARABILITY_FIELDS) | BEHAVIOUR_NEUTRAL_FIELDS


def test_no_setting_is_both_compared_and_declared_neutral() -> None:
    assert not set(COMPARABILITY_FIELDS) & BEHAVIOUR_NEUTRAL_FIELDS


def test_every_compared_field_has_a_mutation_proving_it_is_compared() -> None:
    assert set(MUTATIONS) == set(COMPARABILITY_FIELDS)
    assert set(NEUTRAL_MUTATIONS) == BEHAVIOUR_NEUTRAL_FIELDS


@pytest.mark.parametrize("dotted", sorted(MUTATIONS))
def test_a_behaviour_changing_setting_retires_the_baseline(dotted: str) -> None:
    """Direction one: move something that changes what retrieval returns, lose the baseline."""
    baseline = recorded(canonical())
    changed = Settings.model_validate(MUTATIONS[dotted])

    reasons = baseline.incomparabilities(changed)

    assert baseline.comparable_to(changed) is False
    assert any(reason.startswith(f"{dotted} was ") for reason in reasons), reasons


@pytest.mark.parametrize("dotted", sorted(NEUTRAL_MUTATIONS))
def test_a_behaviour_neutral_setting_keeps_the_baseline(dotted: str) -> None:
    """Changing where a key lives or how large a batch is cannot change what comes back."""
    baseline = recorded(canonical())
    changed = Settings.model_validate(NEUTRAL_MUTATIONS[dotted])

    assert baseline.comparable_to(changed) is True


@pytest.mark.parametrize("dotted", sorted(MUTATIONS))
def test_a_setting_added_after_the_recording_keeps_the_baseline(dotted: str) -> None:
    """Direction two: a knob the baseline predates, left at the default it was given.

    This is the whole defect TASK-0244 names. A baseline recorded before a field existed
    mentions no value for it, and the default is behaviour-preserving by the rule
    ``COMPARABILITY_VERSION`` enforces — so the recording survives the addition.
    """
    baseline = recorded(canonical(), omitting=dotted)

    assert baseline.comparable_to(canonical()) is True


@pytest.mark.parametrize("dotted", sorted(MUTATIONS))
def test_a_setting_the_baseline_predates_is_still_compared_once_it_moves(dotted: str) -> None:
    """Absent means "left at the default", not "stop checking this field"."""
    baseline = recorded(canonical(), omitting=dotted)
    changed = Settings.model_validate(MUTATIONS[dotted])

    assert baseline.comparable_to(changed) is False


def test_a_baseline_from_an_older_fingerprint_version_is_refused() -> None:
    """Version 0 is every baseline recorded before the fingerprint existed."""
    legacy = Baseline(
        recorded_at="2026-08-02T07:39:06.928460+00:00",
        k=5,
        recall_at_k=1.0,
        mrr=1.0,
        ndcg_at_k=1.0,
        scored=12,
        embedding_model="BAAI/bge-small-en-v1.5",
        config_hash="17c289717c02f81d32d71a9f229996dcaebc04aca82c574cdd883a13a61c0072",
    )

    assert legacy.comparable_to(canonical()) is False
    assert any("version 0" in reason for reason in legacy.incomparabilities(canonical()))


def test_the_fingerprint_names_the_field_rather_than_a_digest() -> None:
    """The hash could only say "different"; a reviewer needs to know what moved."""
    baseline = recorded(canonical())
    changed = Settings.model_validate({"retrieval": {"rerank": False}})

    assert baseline.incomparabilities(changed) == ["retrieval.rerank was true and is false"]


@pytest.mark.parametrize("dataset", ["handbook", "policies"])
def test_the_committed_baselines_carry_a_fingerprint(dataset: str) -> None:
    baseline = load_baseline(DATASETS / dataset / "baseline.json")

    assert baseline is not None
    assert baseline.fingerprint_version == COMPARABILITY_VERSION
    assert set(baseline.fingerprint) == set(COMPARABILITY_FIELDS)


def test_the_handbook_baseline_states_exactly_why_it_is_not_yet_comparable() -> None:
    """The honest state of TASK-0244: named reasons, not an opaque hash mismatch.

    All three are real behaviour differences from the shipped defaults. The five knobs
    added since the recording are *not* among them, which is the point of the change.
    """
    baseline = load_baseline(DATASETS / "handbook" / "baseline.json")

    assert baseline is not None
    assert baseline.incomparabilities(canonical()) == [
        'chunking.token_counter was "estimate" and is "auto"',
        "retrieval.top_k was 5 and is 10",
        "retrieval.rerank was false and is true",
    ]


def test_the_handbook_baseline_is_comparable_to_the_setup_it_records() -> None:
    """The configuration the CI gate pins, and the one the recorded numbers belong to."""
    baseline = load_baseline(DATASETS / "handbook" / "baseline.json")
    pinned = Settings.model_validate(
        {
            "chunking": {"token_counter": "estimate"},
            "retrieval": {"top_k": 5, "rerank": False},
            "eval": {"regression_gate": True},
        }
    )

    assert baseline is not None
    assert baseline.comparable_to(pinned) is True


def test_a_recorded_fingerprint_survives_a_json_round_trip() -> None:
    """The baseline is a committed file, so every recorded value has to be plain JSON."""
    fingerprint = comparability_fingerprint(canonical())

    assert json.loads(json.dumps(fingerprint)) == fingerprint
    assert all(isinstance(value, str) for value in fingerprint.values())
