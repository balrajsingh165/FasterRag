import json
import logging

import pytest

from fasterrag.observability.logging import (
    REDACTED,
    JsonFormatter,
    SecretMaskingFilter,
    bind_trace_id,
    configure_logging,
    current_trace_id,
    new_trace_id,
    use_trace_id,
)


def make_record(message: str = "hello", **extra: object) -> logging.LogRecord:
    record = logging.LogRecord(
        name="fasterrag.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_trace_id_is_otel_shaped() -> None:
    trace_id = new_trace_id()
    assert len(trace_id) == 32
    assert trace_id == trace_id.lower()
    int(trace_id, 16)


def test_trace_ids_are_unique() -> None:
    assert new_trace_id() != new_trace_id()


def test_use_trace_id_restores_the_previous_binding() -> None:
    outer = bind_trace_id("a" * 32)
    with use_trace_id("b" * 32) as inner:
        assert current_trace_id() == inner
    assert current_trace_id() == outer


def test_current_trace_id_mints_when_unbound() -> None:
    with use_trace_id() as minted:
        assert current_trace_id() == minted
        assert len(minted) == 32


def test_formatter_emits_one_json_line_with_the_correlation_id() -> None:
    with use_trace_id("c" * 32):
        line = JsonFormatter().format(make_record("ingest accepted"))

    assert "\n" not in line
    payload = json.loads(line)
    assert payload["message"] == "ingest accepted"
    assert payload["level"] == "info"
    assert payload["logger"] == "fasterrag.test"
    assert payload["trace_id"] == "c" * 32
    assert payload["ts"].endswith("+00:00")


def test_formatter_includes_extra_fields() -> None:
    line = JsonFormatter().format(make_record("job queued", job_id="job_01J8Z3W7", count=3))
    payload = json.loads(line)
    assert payload["job_id"] == "job_01J8Z3W7"
    assert payload["count"] == 3


def test_formatter_prefers_the_record_trace_id_over_the_context() -> None:
    with use_trace_id("d" * 32):
        line = JsonFormatter().format(make_record("replayed", trace_id="e" * 32))
    assert json.loads(line)["trace_id"] == "e" * 32


def test_formatter_renders_exceptions() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        record = make_record("failed")
        import sys

        record.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(record))
    assert "ValueError: boom" in payload["exception"]


def test_configure_logging_is_idempotent() -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        configure_logging("debug")
        configure_logging("warning")
        installed = [h for h in root.handlers if h.get_name() == "fasterrag-json"]
        assert len(installed) == 1
        assert root.level == logging.WARNING
    finally:
        for handler in list(root.handlers):
            if handler.get_name() == "fasterrag-json":
                root.removeHandler(handler)
        root.handlers = original_handlers
        root.setLevel(original_level)


def test_configure_logging_rejects_unsupported_levels() -> None:
    with pytest.raises(ValueError, match="unsupported log level"):
        configure_logging("trace")


KEY = "frk-task0491-logged-by-mistake-41c7e2"
SECOND = "frk-task0491-second-key-9b03d5a1"


def masked(record: logging.LogRecord, *secret_env: str) -> str:
    """Return the JSON line ``record`` becomes once the masking filter has seen it."""
    assert SecretMaskingFilter(secret_env).filter(record) is True
    return JsonFormatter().format(record)


def test_a_key_logged_by_mistake_is_masked_in_the_message(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing stood between a key that reached a log call and the log line it wrote."""
    monkeypatch.setenv("FASTERRAG_API_KEY", f"{KEY}:*;{SECOND}:ingest:acme")
    record = make_record("presented %s then %s")
    record.args = (KEY, SECOND)

    line = masked(record)

    assert KEY not in line
    assert SECOND not in line
    assert json.loads(line)["message"] == f"presented {REDACTED} then {REDACTED}"


def test_a_key_in_extra_fields_and_exception_text_is_masked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    try:
        raise ValueError(f"provider rejected {KEY}")
    except ValueError:
        import sys

        record = make_record("call failed", detail=f"key={KEY}", nested={"keys": [KEY]})
        record.exc_info = sys.exc_info()

    line = masked(record)

    assert KEY not in line
    payload = json.loads(line)
    assert payload["detail"] == f"key={REDACTED}"
    assert payload["nested"] == {"keys": [REDACTED]}
    assert f"provider rejected {REDACTED}" in payload["exception"]


def test_a_variable_the_configuration_names_is_masked_whatever_it_is_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MY_CREDENTIAL", KEY)

    assert KEY in masked(make_record(f"saw {KEY}"))
    assert KEY not in masked(make_record(f"saw {KEY}"), "MY_CREDENTIAL")


def test_a_url_valued_credential_is_masked_whole_and_not_split_on_its_colons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dsn = "postgresql://rag:s3cret-pass-1@db:5432/rag"
    monkeypatch.setenv("PGVECTOR_DSN", dsn)

    line = masked(make_record(f"connecting with {dsn} over postgresql"))

    assert dsn not in line
    assert "over postgresql" in line


def test_a_short_value_is_not_masked_so_ordinary_text_survives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TINY_TOKEN", "abc")

    assert json.loads(masked(make_record("abc is fine")))["message"] == "abc is fine"


def test_a_record_whose_arguments_do_not_fit_is_masked_and_still_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FASTERRAG_API_KEY", KEY)
    record = make_record(f"broken call with {KEY} %s %s")
    record.args = (KEY,)

    assert SecretMaskingFilter().filter(record) is True

    assert KEY not in str(record.msg)
    assert record.args == (REDACTED,)


def test_the_installed_handler_masks_what_it_writes(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("NAMED_BY_CONFIG", KEY)
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        configure_logging("info", secret_env=["NAMED_BY_CONFIG"])
        logging.getLogger("fasterrag.test").warning("leaking %s", KEY)
    finally:
        root.handlers = original_handlers
        root.setLevel(original_level)

    written = capsys.readouterr().err
    assert f"leaking {REDACTED}" in written
    assert KEY not in written
