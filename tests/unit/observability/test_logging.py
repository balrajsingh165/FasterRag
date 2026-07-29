import json
import logging

import pytest

from fasterrag.observability.logging import (
    JsonFormatter,
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
