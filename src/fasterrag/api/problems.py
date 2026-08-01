"""RFC 9457 ``application/problem+json`` responses.

Every error leaving the API is a problem document carrying a stable machine-readable
``code`` and the ``trace_id`` that correlates it with logs and spans
(``docs/api-reference.md``). A generic 500 without a problem body is never returned:
the unhandled-exception handler is the last line of that guarantee.

Offending input values are never echoed back — reported fields carry the location and
reason only, so a credential sent in the wrong field cannot be reflected into logs.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import Response

from fasterrag.errors import ErrorCode, FasterRagError, ProvisioningError, problem_spec
from fasterrag.observability import metrics
from fasterrag.observability.logging import get_logger

__all__ = [
    "PROBLEM_MEDIA_TYPE",
    "ProblemDocument",
    "build_problem",
    "install_exception_handlers",
    "problem_response",
]

PROBLEM_MEDIA_TYPE = "application/problem+json"

_SERVER_ERROR_THRESHOLD = 500

_STATUS_CODES: dict[int, ErrorCode] = {
    400: ErrorCode.VALIDATION_FAILED,
    401: ErrorCode.AUTH_MISSING,
    402: ErrorCode.BUDGET_EXCEEDED,
    403: ErrorCode.AUTH_SCOPE,
    404: ErrorCode.NOT_FOUND,
    409: ErrorCode.CONFLICT,
    413: ErrorCode.PAYLOAD_TOO_LARGE,
    422: ErrorCode.VALIDATION_FAILED,
    429: ErrorCode.RATE_LIMITED,
    503: ErrorCode.NOT_READY,
}

_logger = get_logger(__name__)


class ProblemDocument(BaseModel):
    """An RFC 9457 problem document plus fasterRag's stable extension members."""

    model_config = ConfigDict(extra="allow")

    type: str
    title: str
    status: int
    detail: str
    instance: str | None = None
    code: ErrorCode
    trace_id: str
    retryable: bool


def build_problem(
    error: FasterRagError,
    *,
    instance: str | None = None,
    status: int | None = None,
    extensions: dict[str, Any] | None = None,
) -> ProblemDocument:
    """Render ``error`` as a problem document.

    Args:
        error: The typed error being reported.
        instance: Request path the failure occurred on.
        status: Overrides the HTTP status registered for the code. Used when the
            transport status is known independently, as for handled HTTP exceptions.
        extensions: Additional RFC 9457 extension members, such as ``errors`` for
            field-level validation failures.

    Returns:
        The problem document ready to be serialized.
    """
    spec = problem_spec(error.code)
    detail = error.detail
    if isinstance(error, ProvisioningError) and error.fix:
        detail = f"{detail} Fix: {error.fix}"

    return ProblemDocument(
        type=spec.type_uri,
        title=spec.title,
        status=status if status is not None else spec.status,
        detail=detail,
        instance=instance,
        code=error.code,
        trace_id=error.trace_id,
        retryable=error.retryable,
        **(extensions or {}),
    )


def problem_response(
    error: FasterRagError,
    *,
    instance: str | None = None,
    status: int | None = None,
    headers: dict[str, str] | None = None,
    extensions: dict[str, Any] | None = None,
) -> JSONResponse:
    """Return ``error`` as an ``application/problem+json`` response."""
    document = build_problem(error, instance=instance, status=status, extensions=extensions)
    return JSONResponse(
        status_code=document.status,
        content=document.model_dump(mode="json", exclude_none=True),
        media_type=PROBLEM_MEDIA_TYPE,
        headers=headers,
    )


def route_template(request: Request) -> str:
    """Return the route pattern a request matched, never its concrete path.

    # CRITICAL: metric labels must be the template. RFC 9457's ``instance`` identifies the
    # one occurrence, so the problem body rightly carries the concrete path — but putting
    # that same value in a label would mint a new time series per job id or collection name,
    # which is the textbook way to bring down a Prometheus server.
    """
    route = request.scope.get("route")
    return str(getattr(route, "path", None) or request.url.path)


def _log_problem(
    document: ProblemDocument, exc: BaseException | None = None, endpoint: str | None = None
) -> None:
    """Log a problem at a severity matching its status, always with the trace id.

    Also the single place errors are counted. Every problem document is built here, whatever
    raised it, so counting at this funnel cannot miss a path the way per-handler counting
    would the first time a new handler is added.
    """
    metrics.ERRORS.increment(
        endpoint=endpoint or "unknown",
        code=document.code.value,
        tenant="none",
    )

    level = "error" if document.status >= _SERVER_ERROR_THRESHOLD else "warning"
    getattr(_logger, level)(
        "request failed",
        extra={
            "code": document.code.value,
            "status": document.status,
            "trace_id": document.trace_id,
            "path": document.instance,
        },
        exc_info=exc if document.status >= _SERVER_ERROR_THRESHOLD else None,
    )


async def fasterrag_error_handler(request: Request, exc: Exception) -> Response:
    """Render a typed fasterRag error as a problem document."""
    if not isinstance(exc, FasterRagError):
        raise exc

    response = problem_response(exc, instance=request.url.path)
    _log_problem(build_problem(exc, instance=request.url.path), exc, route_template(request))
    return response


async def validation_error_handler(request: Request, exc: Exception) -> Response:
    """Render request-schema failures as ``VALIDATION_FAILED`` with the offending fields."""
    if not isinstance(exc, RequestValidationError):
        raise exc

    fields = [
        {
            "field": ".".join(str(part) for part in error["loc"]),
            "message": error["msg"],
        }
        for error in exc.errors()
    ]
    error = FasterRagError(
        "the request body failed schema validation",
        code=ErrorCode.VALIDATION_FAILED,
    )
    document = build_problem(error, instance=request.url.path, extensions={"errors": fields})
    _log_problem(document, endpoint=route_template(request))
    return JSONResponse(
        status_code=document.status,
        content=document.model_dump(mode="json", exclude_none=True),
        media_type=PROBLEM_MEDIA_TYPE,
    )


async def http_exception_handler(request: Request, exc: Exception) -> Response:
    """Render Starlette HTTP exceptions as problem documents with a mapped code."""
    if not isinstance(exc, StarletteHTTPException):
        raise exc

    code = _STATUS_CODES.get(exc.status_code, ErrorCode.INTERNAL)
    detail = str(exc.detail) if exc.detail else problem_spec(code).title
    error = FasterRagError(detail, code=code)
    document = build_problem(error, instance=request.url.path, status=exc.status_code)
    _log_problem(document, endpoint=route_template(request))
    return JSONResponse(
        status_code=document.status,
        content=document.model_dump(mode="json", exclude_none=True),
        media_type=PROBLEM_MEDIA_TYPE,
        headers=getattr(exc, "headers", None),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> Response:
    """Render any unclassified failure as ``INTERNAL``, never as a bare 500.

    The exception is logged with its correlation id and full traceback, so nothing is
    silently swallowed; the client receives a problem body carrying the same trace id.
    """
    error = FasterRagError("an unexpected error occurred", code=ErrorCode.INTERNAL)
    document = build_problem(error, instance=request.url.path)
    _log_problem(document, exc, route_template(request))
    return JSONResponse(
        status_code=document.status,
        content=document.model_dump(mode="json", exclude_none=True),
        media_type=PROBLEM_MEDIA_TYPE,
    )


def install_exception_handlers(app: FastAPI) -> None:
    """Register every handler that turns an error into a problem document."""
    app.add_exception_handler(FasterRagError, fasterrag_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
