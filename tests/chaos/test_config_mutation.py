"""The config-mutation chaos scenario D1's proof metric rests on (TASK-0241).

D1 claims "zero false negatives across config-mutation scenarios". Until this module
existed there were no such scenarios, so the claim rested on nothing: the unit suite
(``tests/unit/services/test_lockfile.py``) samples four fields by hand, which proves those
four and says nothing about the rest.

The fault injected here is a *configuration mutation* — the fault an operator causes by
editing ``config.yaml`` under a built index — and it is injected at the honest seam: the
lockfile is written through :class:`~fasterrag.services.lockfile.LockStore` onto a real
directory, the mutated configuration is validated by the real schema, and the lock is read
back from disk before being compared. A mutation that the schema would reject is not a
mutation an operator can make, so it is never asserted on.

**What this establishes, and what it does not.** The mutation set is derived by asking
``retrieval_config_hash`` itself which keys it reads — every key of every section is moved
through the schema and the digest recomputed — rather than being mirrored by hand, so a
field added to a hashed section is mutated the day it lands, and a key the schema never
lets us move fails the suite instead of quietly dropping out of it. Within that subset, for
single-field edits, zero false negatives is established rather than sampled. It is *not*
established for: edits that move two fields at once (not enumerated,
though any one of them already trips the hash), drift arriving by a route other than
configuration (one scenario each below for corpus, embedding model, and identity scheme —
sampled, not exhaustive), or a field the hash deliberately omits, which is a *correct*
negative and is asserted as one.

The false-positive half matters as much as the false-negative half. A detector that reports
drift on every edit has no false negatives and is useless, because the first time
``app.port`` reports a stale index nobody believes the next report either. So each scenario
below has a counterpart mutating something the index does not depend on.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final, Literal, get_args, get_origin

import pytest
from pydantic import BaseModel, ValidationError

from fasterrag.config.schema import Settings
from fasterrag.core.identity import IDENTITY_VERSION, retrieval_config_hash
from fasterrag.services.lockfile import DriftReport, IndexLock, LockStore, build_lock, detect_drift

pytestmark = pytest.mark.chaos

COLLECTION: Final = "docs"
EMBEDDING_MODEL: Final = "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_VERSION: Final = "bge-small-en-v1.5@a1b2c3"
DIMENSIONS: Final = 384
CORPUS: Final[dict[str, str]] = {"d_1": "hash-a", "d_2": "hash-b"}

OVERRIDES: Final[dict[str, tuple[Any, ...]]] = {
    "embeddings.tiering": (
        {
            "enabled": True,
            "rules": [
                {"match": {"kind": "faq"}, "provider": "ollama", "model": "nomic-embed-text"}
            ],
        },
        {
            "enabled": False,
            "rules": [
                {"match": {"kind": "faq"}, "provider": "ollama", "model": "nomic-embed-text"}
            ],
        },
    ),
    "embeddings.dimensions": (DIMENSIONS,),
}
"""Mutation units that cannot be expressed as a single leaf edit, keyed by dotted path.

A path listed here is mutated as a whole and is never descended into.

``embeddings.tiering`` cannot be enabled on its own: cross-field rule 8 requires a rule
list alongside the toggle, so the two move together, and the second candidate moves the
rules alone, which is legal while the feature stays disabled. ``embeddings.dimensions``
defaults to ``None``, and ``None`` carries no type from which to derive a smaller edit.
"""

IRRELEVANT: Final[tuple[tuple[str, Any], ...]] = (
    ("app.port", 9999),
    ("app.log_level", "debug"),
    ("app.workers", 8),
    ("embeddings.batch_size", 128),
    ("embeddings.cache.enabled", False),
    ("embeddings.cache.backend", "memory"),
    ("embeddings.cache.max_entries", 50),
    ("workers.embedding_pool_size", 4),
    ("workers.queue_depth", 500),
    ("cache.semantic", True),
    ("cache.ttl", 60),
    ("generation.faithfulness_threshold", 0.9),
    ("generation.citations", False),
    ("index.reindex.eval_gate", False),
    ("index.reindex.rollback_retention_hours", 24),
    ("reliability.timeouts.vector_db_ms", 2500),
    ("reliability.retries.max_attempts", 5),
)
"""Settings an index does not depend on, each paired with a legal value to move it to.

Every entry is an edit an operator may make at any time and that must leave a built index
alone. ``embeddings.batch_size`` and ``embeddings.cache.*`` are the sharpest of them,
because they sit inside a section the hash otherwise reads.
"""


def leaves(payload: dict[str, Any], prefix: tuple[str, ...] = ()) -> Iterator[tuple[str, Any]]:
    """Yield every dotted path and value below ``payload``, honouring the override units."""
    for key, value in payload.items():
        path = (*prefix, key)
        dotted = ".".join(path)
        if isinstance(value, dict) and dotted not in OVERRIDES:
            yield from leaves(value, path)
        else:
            yield dotted, value


def key_verdicts(settings: Settings) -> dict[str, dict[str, bool | None]]:
    """Classify every top-level configuration key by whether the live hash reads it."""
    payload = settings.model_dump(mode="json")
    baseline = retrieval_config_hash(settings)
    return {
        section: {
            key: verdict_for(settings, baseline, leaves({key: value}, (section,)))
            for key, value in subtree.items()
        }
        for section, subtree in payload.items()
        if isinstance(subtree, dict)
    }


def verdict_for(settings: Settings, baseline: str, units: Iterator[tuple[str, Any]]) -> bool | None:
    """Return whether any accepted mutation of these units moves the hash, or ``None``."""
    probed = False
    for dotted, value in units:
        for candidate in candidates(dotted, value):
            mutated = mutate(settings, dotted, candidate)
            if mutated is None:
                continue
            probed = True
            if retrieval_config_hash(mutated) != baseline:
                return True
    return False if probed else None


def hashed_subset(settings: Settings) -> dict[str, tuple[str, ...]]:
    """Return the sections and keys the live hash reads, as derived from the hash."""
    subset = {
        section: tuple(key for key, verdict in keys.items() if verdict is True)
        for section, keys in key_verdicts(settings).items()
    }
    return {section: keys for section, keys in subset.items() if keys}


def hashed_units(settings: Settings) -> list[tuple[str, Any]]:
    """Return every mutation unit of the drift-relevant configuration subset."""
    payload = settings.model_dump(mode="json")
    units: list[tuple[str, Any]] = []
    for section, keys in hashed_subset(settings).items():
        units.extend(leaves({key: payload[section][key] for key in keys}, (section,)))
    return units


def annotation_at(model: type[BaseModel], path: tuple[str, ...]) -> Any:
    """Return the declared annotation of a dotted path, or ``None`` when it is not reachable."""
    field = model.model_fields.get(path[0])
    if field is None:
        return None
    if len(path) == 1:
        return field.annotation
    nested = field.annotation
    if isinstance(nested, type) and issubclass(nested, BaseModel):
        return annotation_at(nested, path[1:])
    return None


def unset_candidate(annotation: Any) -> list[Any]:
    """Return one value of the declared type for a unit whose current value is ``None``."""
    samples: Final[dict[Any, Any]] = {str: "mutated", int: 1, float: 0.5, bool: True}
    for member in get_args(annotation):
        if member in samples:
            return [samples[member]]
    return []


def candidates(dotted: str, value: Any) -> list[Any]:
    """Return every value this unit could be mutated to, before validation filters them.

    A ``Literal`` field offers its other members; otherwise the smallest edit of the value's
    own type is used, in both directions where a bound might reject one of them.
    """
    if dotted in OVERRIDES:
        return list(OVERRIDES[dotted])

    annotation = annotation_at(Settings, tuple(dotted.split(".")))
    if get_origin(annotation) is Literal:
        return [member for member in get_args(annotation) if member != value]
    if value is None:
        return unset_candidate(annotation)
    if isinstance(value, bool):
        return [not value]
    if isinstance(value, int):
        return [value + 1, value - 1]
    if isinstance(value, float):
        return [value + 0.25, value - 0.25, value / 2]
    if isinstance(value, str):
        return [f"{value}-mutated"]
    return []


def with_value(payload: dict[str, Any], dotted: str, value: Any) -> dict[str, Any]:
    """Return a copy of ``payload`` with one dotted path replaced."""
    mutated = copy.deepcopy(payload)
    cursor = mutated
    *parents, leaf = dotted.split(".")
    for part in parents:
        cursor = cursor[part]
    cursor[leaf] = value
    return mutated


def mutate(settings: Settings, dotted: str, value: Any) -> Settings | None:
    """Return the settings with one unit changed, or ``None`` when the schema refuses it.

    A rejected mutation is not a missed detection: the configuration never loads, so no
    index can be built against it and there is nothing to drift from.
    """
    try:
        return Settings.model_validate(with_value(settings.model_dump(mode="json"), dotted, value))
    except ValidationError:
        return None


def lock_of(settings: Settings, root: Path, **overrides: Any) -> IndexLock:
    """Write the lockfile for ``settings`` and read it back from disk."""
    built = build_lock(
        COLLECTION,
        settings,
        embedding_model=EMBEDDING_MODEL,
        embedding_model_version=EMBEDDING_MODEL_VERSION,
        dimensions=DIMENSIONS,
        document_hashes=dict(CORPUS),
    )
    if overrides:
        built = IndexLock(**{**built.as_dict(), **overrides})

    store = LockStore(root)
    store.write(built)
    restored = store.read(COLLECTION)
    assert restored is not None
    return restored


def drift(lock: IndexLock, settings: Settings, **kwargs: Any) -> DriftReport:
    """Run the live comparison the way ``index lock verify`` runs it."""
    return detect_drift(
        lock,
        settings,
        collection=COLLECTION,
        embedding_model=kwargs.pop("embedding_model", EMBEDDING_MODEL),
        embedding_model_version=kwargs.pop("embedding_model_version", EMBEDDING_MODEL_VERSION),
        document_hashes=kwargs.pop("document_hashes", dict(CORPUS)),
    )


def test_the_mutation_set_covers_every_leaf_of_the_hashed_subset(
    chaos_settings: Settings,
) -> None:
    """Every unit has at least one mutation the schema accepts, so nothing is skipped silently."""
    unmutable = [
        dotted
        for dotted, value in hashed_units(chaos_settings)
        if not [
            candidate
            for candidate in candidates(dotted, value)
            if mutate(chaos_settings, dotted, candidate) is not None
        ]
    ]

    assert unmutable == []


def test_the_hashed_subset_is_not_empty(chaos_settings: Settings) -> None:
    """A guard on the enumeration itself: an empty unit list would pass every case below."""
    units = hashed_units(chaos_settings)

    assert len(units) >= 20
    assert "chunking.chunk_size" in dict(units)
    assert "embeddings.model" in dict(units)
    assert "retrieval.top_k" in dict(units)


def test_no_key_of_a_hashed_section_escapes_classification(chaos_settings: Settings) -> None:
    """A key the derivation cannot move is a hole, and it fails here rather than passing quietly."""
    verdicts = key_verdicts(chaos_settings)

    unclassified = [
        f"{section}.{key}"
        for section, keys in verdicts.items()
        if any(verdict is True for verdict in keys.values())
        for key, verdict in keys.items()
        if verdict is None
    ]

    assert unclassified == []


def test_a_field_added_to_the_hash_is_covered_without_editing_this_module(
    chaos_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The claim this module opens with, asserted instead of argued."""
    live = retrieval_config_hash

    def widened(settings: Settings) -> str:
        """Return the live digest extended with a key the real hash ignores."""
        return f"{live(settings)}:{settings.embeddings.batch_size}"

    assert "embeddings.batch_size" not in dict(hashed_units(chaos_settings))

    monkeypatch.setitem(globals(), "retrieval_config_hash", widened)

    assert "embeddings.batch_size" in dict(hashed_units(chaos_settings))


def test_every_drift_relevant_mutation_is_reported(
    chaos_settings: Settings, tmp_path: Path
) -> None:
    """Mutating any single drift-relevant setting under a built index reports drift."""
    lock = lock_of(chaos_settings, tmp_path)
    missed: list[str] = []
    unnamed: list[str] = []

    for dotted, value in hashed_units(chaos_settings):
        for candidate in candidates(dotted, value):
            mutated = mutate(chaos_settings, dotted, candidate)
            if mutated is None:
                continue
            report = drift(lock, mutated)
            if not report.detected:
                missed.append(f"{dotted}={candidate!r}")
            elif not report.fields:
                unnamed.append(f"{dotted}={candidate!r}")

    assert missed == []
    assert unnamed == []


def test_a_drift_irrelevant_mutation_is_never_reported(
    chaos_settings: Settings, tmp_path: Path
) -> None:
    """The false-positive half: editing what the index does not depend on is not drift."""
    lock = lock_of(chaos_settings, tmp_path)
    false_positives: list[str] = []

    for dotted, value in IRRELEVANT:
        mutated = mutate(chaos_settings, dotted, value)
        assert mutated is not None, f"{dotted} is not a legal edit, so it cannot control anything"
        if drift(lock, mutated).detected:
            false_positives.append(dotted)

    assert false_positives == []


def test_no_drift_irrelevant_control_sits_inside_the_hashed_subset(
    chaos_settings: Settings,
) -> None:
    """The controls must be genuinely outside the hash, or the case above proves nothing."""
    hashed = set(dict(hashed_units(chaos_settings)))

    assert [dotted for dotted, _ in IRRELEVANT if dotted in hashed] == []


def test_rewriting_the_same_configuration_is_not_drift(
    chaos_settings: Settings, tmp_path: Path
) -> None:
    """Reformatting ``config.yaml`` without changing a value leaves the index alone."""
    rewritten = Settings.model_validate(chaos_settings.model_dump(mode="json"))

    assert drift(lock_of(chaos_settings, tmp_path), rewritten).detected is False


def test_the_mutated_field_is_named_rather_than_only_the_hash(
    chaos_settings: Settings, tmp_path: Path
) -> None:
    """The four chunker fields the lock stores by name are named individually."""
    lock = lock_of(chaos_settings, tmp_path)
    expected = {
        "chunk_size": ("chunking.chunk_size", 512),
        "overlap": ("chunking.overlap", 128),
        "chunker_strategy": ("chunking.strategy", "fixed"),
        "contextual_enrichment": ("chunking.contextual_enrichment", True),
    }

    for field, (dotted, value) in expected.items():
        mutated = mutate(chaos_settings, dotted, value)
        assert mutated is not None
        report = drift(lock, mutated)

        assert field in report.fields, f"{dotted} moved but {field} was not named"
        assert "config_hash" in report.fields
        detail = next(item for item in report.details if item["field"] == field)
        assert detail["live"] == value


def test_a_mutated_corpus_is_reported_per_document(
    chaos_settings: Settings, tmp_path: Path
) -> None:
    """Corpus drift — the mutation an operator makes by editing the files, not the config."""
    report = drift(
        lock_of(chaos_settings, tmp_path),
        chaos_settings,
        document_hashes={"d_1": "hash-a", "d_2": "EDITED", "d_3": "hash-c"},
    )

    assert report.documents_changed == ["d_2"]
    assert report.documents_added == ["d_3"]
    assert report.documents_removed == []
    assert report.detected is True


def test_an_unchanged_corpus_is_not_reported(chaos_settings: Settings, tmp_path: Path) -> None:
    """Re-running verify over the same corpus must stay silent."""
    assert drift(lock_of(chaos_settings, tmp_path), chaos_settings).detected is False


def test_new_weights_behind_a_stable_model_name_are_reported(
    chaos_settings: Settings, tmp_path: Path
) -> None:
    """The mutation nobody makes on purpose: a provider changing the weights under a name."""
    report = drift(
        lock_of(chaos_settings, tmp_path),
        chaos_settings,
        embedding_model_version=f"{EMBEDDING_MODEL_VERSION}-rebuilt",
    )

    assert report.fields == ["embedding_model_version"]


def test_an_index_built_under_an_older_identity_scheme_is_reported(
    chaos_settings: Settings, tmp_path: Path
) -> None:
    """A lock written before the current id scheme is drift, not a silent zero-recall index."""
    lock = lock_of(chaos_settings, tmp_path, identity_version=IDENTITY_VERSION - 1)

    report = drift(lock, chaos_settings)

    assert "identity_version" in report.fields


def test_an_absent_lockfile_is_not_reported_as_drift(
    chaos_settings: Settings, tmp_path: Path
) -> None:
    """An un-locked index is distinguishable from a drifted one.

    Reporting a missing lock as drift would make every pre-lockfile index look corrupted,
    which is the false positive that would teach an operator to ignore the report.
    """
    report = detect_drift(
        LockStore(tmp_path).read(COLLECTION), chaos_settings, collection=COLLECTION
    )

    assert report.missing_lock is True
    assert report.detected is False
