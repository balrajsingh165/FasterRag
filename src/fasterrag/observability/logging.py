"""Structured JSON logging and request correlation ids.

Every log line is one JSON object carrying the ``trace_id`` bound to the current
context, so a log line, an OTel span, a problem response, and a persisted trace all
identify the same request (``docs/reliability.md`` §7). Trace ids are 32 lowercase hex
characters — the OpenTelemetry trace-id shape — so correlation survives export to any
OTel backend.

Secret values are never logged: configuration references credentials by environment
variable *name*, and error details carry the name only.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from secrets import token_hex
from typing import Any, Final

__all__ = [
    "JsonFormatter",
    "bind_trace_id",
    "configure_logging",
    "current_trace_id",
    "get_logger",
    "new_trace_id",
    "use_trace_id",
]

_TRACE_ID_BYTES: Final = 16
_HANDLER_NAME: Final = "fasterrag-json"

_LEVELS: Final[dict[str, int]] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

_RESERVED: Final[frozenset[str]] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

_trace_id: ContextVar[str | None] = ContextVar("fasterrag_trace_id", default=None)


def new_trace_id() -> str:
    """Return a fresh 32-character hex trace id in the OpenTelemetry trace-id shape."""
    return token_hex(_TRACE_ID_BYTES)


def bind_trace_id(trace_id: str | None = None) -> str:
    """Bind ``trace_id`` (or a fresh one) to the current context and return it."""
    value = trace_id if trace_id is not None else new_trace_id()
    _trace_id.set(value)
    return value


def current_trace_id() -> str:
    """Return the context's trace id, minting and binding one if none is bound yet."""
    value = _trace_id.get()
    if value is None:
        value = bind_trace_id()
    return value


@contextmanager
def use_trace_id(trace_id: str | None = None) -> Iterator[str]:
    """Bind a trace id for the duration of the block, restoring the previous id after."""
    value = trace_id if trace_id is not None else new_trace_id()
    token = _trace_id.set(value)
    try:
        yield value
    finally:
        _trace_id.reset(token)


class JsonFormatter(logging.Formatter):
    """Render log records as single-line JSON objects with the active correlation id."""

    def format(self, record: logging.LogRecord) -> str:
        """Return ``record`` as a compact JSON line."""
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }

        trace_id = getattr(record, "trace_id", None) or _trace_id.get()
        if trace_id is not None:
            payload["trace_id"] = trace_id

        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_") and key != "trace_id":
                payload[key] = value

        if record.exc_info is not None:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(level: str = "info") -> None:
    """Install the JSON handler on the root logger at ``level``.

    Idempotent: repeated calls replace the previously installed handler rather than
    stacking duplicates, so an embedding application can reconfigure freely.

    Args:
        level: One of the ``app.log_level`` values (``debug``, ``info``, ``warning``,
            ``error``).

    Raises:
        ValueError: If ``level`` is not one of the four supported values. Configuration
            validation rejects other values long before this point.
    """
    level_no = _LEVELS.get(level.lower())
    if level_no is None:
        supported = ", ".join(sorted(_LEVELS))
        raise ValueError(f"unsupported log level {level!r}; expected one of: {supported}")

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.set_name(_HANDLER_NAME)

    root = logging.getLogger()
    for existing in list(root.handlers):
        if existing.get_name() == _HANDLER_NAME:
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level_no)


def get_logger(name: str) -> logging.Logger:
    """Return the named logger; correlation ids are attached by the formatter."""
    return logging.getLogger(name)
