from fasterrag.config.schema import Settings
from fasterrag.core.identity import IDENTITY_VERSION
from fasterrag.services.lockfile import IndexLock, build_lock, detect_drift


def settings() -> Settings:
    return Settings.model_validate({})


def lock(**overrides: object) -> IndexLock:
    built = build_lock(
        "docs",
        settings(),
        embedding_model="m",
        embedding_model_version="1",
        dimensions=4,
    )
    for name, value in overrides.items():
        object.__setattr__(built, name, value)
    return built


def test_a_new_lock_records_the_current_scheme() -> None:
    assert lock().identity_version == IDENTITY_VERSION


def test_the_scheme_is_persisted() -> None:
    """It has to survive the round trip, or nothing can compare against it later."""
    assert lock().as_dict()["identity_version"] == IDENTITY_VERSION


def test_a_matching_scheme_is_not_drift() -> None:
    report = detect_drift(lock(), settings(), collection="docs")

    assert "identity_version" not in report.fields


def test_an_older_scheme_is_reported_as_drift() -> None:
    """Without this the symptom is recall 0.0, which reads as a broken retriever."""
    report = detect_drift(lock(identity_version=1), settings(), collection="docs")

    assert "identity_version" in report.fields


def test_a_lock_written_before_the_field_existed_defaults_to_the_original_scheme() -> None:
    """Defaulting to "current" would declare a stale index up to date, hiding the mismatch."""
    payload = lock().as_dict()
    del payload["identity_version"]

    assert IndexLock.from_dict(payload).identity_version == 1


def test_such_a_lock_is_detected_as_drifted() -> None:
    payload = lock().as_dict()
    del payload["identity_version"]

    report = detect_drift(IndexLock.from_dict(payload), settings(), collection="docs")

    assert "identity_version" in report.fields


def test_a_missing_lock_is_still_reported_as_missing_not_drifted() -> None:
    """Nothing changed; there is simply nothing to compare against."""
    report = detect_drift(None, settings(), collection="docs")

    assert report.missing_lock
    assert not report.fields
