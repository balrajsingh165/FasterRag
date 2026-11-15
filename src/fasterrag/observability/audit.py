"""The audit trail for administrative actions (TASK-0493).

Dropping or creating a collection, deleting a document, backing up or restoring, provisioning
or stopping a tool, repairing with ``doctor --fix``, exporting or importing an archive and
retrying a job's dead letters each write exactly one audit record: who acted, what they did,
to what, how it ended, and the trace id that ties the record to the request's log lines and
problem body.

**Who is named by the key's name and digest prefix, never the key.** ``actor_key`` is the
name the key list gives the entry (``key-1``, ``file-key-3``) and ``actor_key_id`` the first
twelve hex digits of its SHA-256 digest: enough to tell two keys apart, useless for
presenting one. With ``security.auth`` off there is no key and both are ``null``. A CLI
command names the operating-system user instead (``actor_user``), since it runs as the
operator.

**Records are JSON lines under the ``fasterrag.audit`` logger, at INFO.** They go through the
normal, credential-masking log handler whenever the log level admits INFO — the API's
default — and so a CLI command shows its record only with ``--verbose``: a JSON line ahead
of every error message would bury what the operator ran the command to read.
``security.audit_log_file`` writes every record to a file of its own whatever the log level,
which is the trail to keep.
"""

from __future__ import annotations

import getpass
import logging
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Literal

from fasterrag.errors import ConfigError, ErrorCode, FasterRagError
from fasterrag.observability.logging import (
    JsonFormatter,
    SecretMaskingFilter,
    current_trace_id,
)

__all__ = [
    "AUDIT_LOGGER",
    "KEY_ID_LENGTH",
    "Actor",
    "AuditEntry",
    "audited",
    "cli_actor",
    "configure_audit_file",
    "record_audit",
]

AUDIT_LOGGER: Final = "fasterrag.audit"
KEY_ID_LENGTH: Final = 12

_FILE_HANDLER_NAME: Final = "fasterrag-audit-file"

Outcome = Literal["succeeded", "failed", "cancelled"]

_logger = logging.getLogger(AUDIT_LOGGER)

# CRITICAL: the file handler is held here, not attached to the logger. Attached, it would see
# only what the logger's effective level admits, and the trail kept for auditors would vanish
# whenever an operator turned the log level down to `warning`, the CLI's default.
_file_handler: logging.Handler | None = None


@dataclass(frozen=True, slots=True)
class Actor:
    """Who acted: a key's name and digest prefix, or a local user, and the tenant."""

    via: Literal["api", "cli"]
    key: str | None = None
    key_id: str | None = None
    tenant: str | None = None
    user: str | None = None


@dataclass(slots=True)
class AuditEntry:
    """What the audited block reports back: extra detail, and a failure it did not raise."""

    detail: dict[str, Any] = field(default_factory=dict)
    failure: str | None = None

    def fail(self, code: str) -> None:
        """Record the action as failed although it returned, as a mismatched restore does."""
        self.failure = code


def cli_actor(tenant: str | None = None) -> Actor:
    """Return the actor for a CLI command: the operating-system user it runs as."""
    try:
        user: str | None = getpass.getuser()
    except (KeyError, OSError):
        user = None
    return Actor(via="cli", tenant=tenant, user=user)


def record_audit(
    action: str,
    target: str,
    outcome: Outcome,
    actor: Actor,
    *,
    code: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Write one audit record to the normal log, if it admits INFO, and to the audit file."""
    record = _logger.makeRecord(
        AUDIT_LOGGER,
        logging.INFO,
        __file__,
        0,
        "audit %s %s: %s",
        (action, target, outcome),
        None,
        extra={
            "audit": True,
            "action": action,
            "target": target,
            "outcome": outcome,
            "error_code": code,
            "via": actor.via,
            "actor_key": actor.key,
            "actor_key_id": actor.key_id,
            "actor_user": actor.user,
            "tenant": actor.tenant,
            "trace_id": current_trace_id(),
            "detail": dict(detail or {}),
        },
    )
    if _logger.isEnabledFor(logging.INFO):
        _logger.handle(record)
    if _file_handler is not None:
        _file_handler.handle(record)


@contextmanager
def audited(action: str, target: str, actor: Actor) -> Iterator[AuditEntry]:
    """Audit the enclosed action exactly once, re-raising whatever it raised.

    ``succeeded`` on return, ``failed`` with the error code on an error, ``cancelled`` on cancel.
    """
    entry = AuditEntry()
    try:
        yield entry
    except FasterRagError as exc:
        record_audit(action, target, "failed", actor, code=exc.code.value, detail=entry.detail)
        raise
    except Exception:
        record_audit(
            action, target, "failed", actor, code=ErrorCode.INTERNAL.value, detail=entry.detail
        )
        raise
    except BaseException:
        record_audit(action, target, "cancelled", actor, detail=entry.detail)
        raise
    if entry.failure is not None:
        record_audit(action, target, "failed", actor, code=entry.failure, detail=entry.detail)
    else:
        record_audit(action, target, "succeeded", actor, detail=entry.detail)


def configure_audit_file(path: str | None, *, secret_env: Iterable[str] = ()) -> None:
    """Append audit records to ``path`` as well as the normal log; ``None`` removes the file.

    The file is opened at once, so a path that cannot be written stops startup naming the key.
    """
    global _file_handler
    if _file_handler is not None:
        _file_handler.close()
        _file_handler = None
    if path is None:
        return

    target = Path(path).expanduser()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(target, encoding="utf-8")
    except OSError as exc:
        raise ConfigError(
            f"security.audit_log_file {str(target)!r} cannot be opened for appending "
            f"({type(exc).__name__}: {exc.strerror or exc}); fix the path or unset the key"
        ) from exc
    handler.setFormatter(JsonFormatter())
    handler.addFilter(SecretMaskingFilter(secret_env))
    handler.set_name(_FILE_HANDLER_NAME)
    _file_handler = handler
