"""Structured JSON logging and request correlation ids.

Every log line is one JSON object carrying the ``trace_id`` bound to the current
context, so a log line, an OTel span, a problem response, and a persisted trace all
identify the same request (``docs/reliability.md`` §7). Trace ids are 32 lowercase hex
characters — the OpenTelemetry trace-id shape — so correlation survives export to any
OTel backend.

Secret values are never logged: configuration references credentials by environment
variable *name*, and error details carry the name only. As defence in depth the handler
``configure_logging`` installs carries a :class:`SecretMaskingFilter` (TASK-0491), which
replaces any loaded credential value — every variable whose name ends in ``_KEY``,
``_TOKEN``, ``_SECRET``, ``_PASSWORD`` or ``_DSN``, plus every variable the configuration
names in a ``*_env`` field — with ``[redacted]`` in the message, the extra fields and the
exception text. It reads values from the environment at the moment a record is written, so
it holds no copy of any secret, and it logs nothing about what it masked.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from secrets import token_hex
from typing import Any, Final

__all__ = [
    "REDACTED",
    "JsonFormatter",
    "SecretMaskingFilter",
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

REDACTED: Final = "[redacted]"

_CREDENTIAL_SUFFIXES: Final = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD", "_DSN")
_MINIMUM_SECRET_LENGTH: Final = 8


def _fragments(value: str) -> set[str]:
    """Return the parts of a credential value that must never be written to a log.

    A key list ``secret[:scopes[:tenant]];…`` yields each secret as well as the whole value.
    """
    whole = value.strip()
    parts = {whole}
    if "://" not in whole:
        for entry in whole.split(";"):
            parts.add(entry.strip())
            parts.add(entry.partition(":")[0].strip())
    return {part for part in parts if len(part) >= _MINIMUM_SECRET_LENGTH}


def _mask(text: str, secrets: Sequence[str]) -> str:
    """Return ``text`` with every secret replaced by the redaction marker."""
    for secret in secrets:
        if secret in text:
            text = text.replace(secret, REDACTED)
    return text


def _mask_value(value: Any, secrets: Sequence[str]) -> Any:
    """Return ``value`` with secrets masked, keeping containers' shape and other values."""
    if isinstance(value, str):
        return _mask(value, secrets)
    if isinstance(value, dict):
        return {key: _mask_value(item, secrets) for key, item in value.items()}
    if isinstance(value, list):
        return [_mask_value(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(_mask_value(item, secrets) for item in value)
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    rendered = str(value)
    masked = _mask(rendered, secrets)
    return masked if masked != rendered else value


class SecretMaskingFilter(logging.Filter):
    """Mask every loaded credential value in a record before a handler writes it.

    Defence in depth: nothing should log a key, and this makes a mistake that does unreadable.
    """

    def __init__(self, secret_env: Iterable[str] = ()) -> None:
        """Mask credential-named variables, plus every variable ``secret_env`` names."""
        super().__init__()
        self.secret_env = frozenset(secret_env)

    def secrets(self) -> list[str]:
        """Return the values to mask, longest first so a key's prefix never splits it."""
        names = {name for name in os.environ if name.upper().endswith(_CREDENTIAL_SUFFIXES)}
        found: set[str] = set()
        for name in names | self.secret_env:
            value = os.environ.get(name)
            if value:
                found |= _fragments(value)
        return sorted(found, key=len, reverse=True)

    def filter(self, record: logging.LogRecord) -> bool:
        """Rewrite the record's message, extra fields and exception text; never drop it.

        A record whose arguments do not fit its message is masked piece by piece and left
        malformed, so the formatter still reports the broken call through ``handleError``.
        """
        secrets = self.secrets()
        if not secrets:
            return True

        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            record.msg = _mask_value(record.msg, secrets)
            record.args = _mask_value(record.args, secrets)
        else:
            masked = _mask(message, secrets)
            if masked != message:
                record.msg, record.args = masked, None

        for key, value in list(vars(record).items()):
            if key not in _RESERVED and not key.startswith("_"):
                setattr(record, key, _mask_value(value, secrets))

        if record.exc_info is not None and record.exc_text is None:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = _mask(record.exc_text, secrets)
        return True


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
            payload["exception"] = record.exc_text or self.formatException(record.exc_info)

        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_logging(level: str = "info", *, secret_env: Iterable[str] = ()) -> None:
    """Install the JSON handler on the root logger at ``level``, masking loaded credentials.

    ``secret_env`` names variables to mask beyond the credential-named ones.
    """
    level_no = _LEVELS.get(level.lower())
    if level_no is None:
        supported = ", ".join(sorted(_LEVELS))
        raise ValueError(f"unsupported log level {level!r}; expected one of: {supported}")

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(SecretMaskingFilter(secret_env))
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
