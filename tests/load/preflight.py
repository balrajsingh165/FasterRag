"""The probe that must pass before any traffic is generated.

A load generator pointed at nothing produces a clean report: zero requests, zero failures,
a p95 of zero. The numbers are internally consistent and completely false, and the person
reading them has no way to tell. That is the single most important failure this harness has
to make impossible, so the target is proven to exist, to be ready, to be fasterRag, and to
answer the exact requests the scenario will send — before the first user is spawned.

Each check earns its place by ruling out a specific way a run can be vacuous:

* ``/healthz`` — the process is up. Connection refused here is the obvious case.
* ``/readyz`` — its dependencies are up. A ready-looking API whose vector database is down
  answers every query with a typed error in single-digit milliseconds, which is the fastest
  a RAG system can possibly look and means nothing at all.
* ``/metrics`` — the body parses, and it carries ``fasterrag_requests_total``. Without this
  the harness cannot tell fasterRag from any other service that happens to answer on that
  port, and four of the seven SLIs are read from this endpoint.
* one real request per endpoint the scenario will drive, with a real query from the dataset.
  A 404 from a path that moved, a 401 from a missing key, a 422 from a schema change — each
  turns an entire run into failures, and finding out after an hour of traffic is a waste of
  the reference hardware's time.
* the query response carries a usable ``timings_ms``. The headline SLI is derived from it;
  if it is absent the harness says so here rather than reporting a p95 of nothing.

The preflight runs twice by design: once in ``run_load.py`` before Locust is launched, and
again inside the Locust process on ``test_start``. The second one exists because
``locust -f tests/load/locustfile.py`` is a legal thing to type, and a guard that a direct
invocation walks past is not a guard.

Only the standard library is used, so this module imports in any interpreter — including
the ordinary test run, which exercises it against a closed port.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Final
from urllib.parse import urljoin

from tests.load.harness import (
    ExitCode,
    LoadConfig,
    LoadHarnessError,
    Series,
    load_corpus,
    load_queries,
    parse_exposition,
)

__all__ = ["PreflightResult", "fetch", "preflight"]

PROBE_TIMEOUT_S: Final = 15.0

# CRITICAL: the one series that identifies the target as fasterRag. Any HTTP server can
# answer /healthz with a 200; only this one exports the metrics catalogue the SLIs are read
# from, and pointing a load run at the wrong service is otherwise undetectable.
IDENTIFYING_SERIES: Final = "fasterrag_requests_total"


@dataclass(frozen=True, slots=True)
class Response:
    """One probe response: its status, its body, and the URL that produced it."""

    url: str
    status: int
    body: bytes

    def json(self) -> Any:
        """Return the parsed body, raising a preflight failure when it is not JSON."""
        try:
            return json.loads(self.body or b"null")
        except json.JSONDecodeError as exc:
            raise LoadHarnessError(
                f"{self.url} returned {self.status} with a body that is not JSON: {exc}",
                ExitCode.PREFLIGHT_FAILED,
            ) from exc


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """What the probe learned: the opening metrics scrape and the dataset it validated."""

    metrics: tuple[Series, ...]
    queries: tuple[str, ...]
    corpus_documents: int


def _headers(config: LoadConfig, api_key: str | None) -> dict[str, str]:
    """Return the headers every probe and every load request carries."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if config.tenant:
        headers["X-Tenant-ID"] = config.tenant
    return headers


def fetch(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    payload: object | None = None,
    timeout: float = PROBE_TIMEOUT_S,
) -> Response:
    """Make one probe request, turning every transport failure into a typed refusal."""
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=body, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            return Response(url=url, status=int(handle.status), body=handle.read())
    except urllib.error.HTTPError as exc:
        return Response(url=url, status=int(exc.code), body=exc.read())
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise LoadHarnessError(
            f"{method} {url} could not be reached: {exc}. A load run against an unreachable "
            "target would report zero requests and zero failures, which reads as a fast "
            "system; it is refused instead",
            ExitCode.PREFLIGHT_FAILED,
        ) from exc


def _require_ok(response: Response, what: str) -> Response:
    """Refuse the run unless a probe returned 200."""
    if response.status != 200:
        raise LoadHarnessError(
            f"{what}: {response.url} returned {response.status}, not 200 "
            f"({response.body[:300].decode('utf-8', 'replace')})",
            ExitCode.PREFLIGHT_FAILED,
        )
    return response


def _probe_metrics(base: str, headers: dict[str, str]) -> tuple[Series, ...]:
    """Prove the target exposes a parseable fasterRag metrics catalogue."""
    response = _require_ok(fetch(urljoin(base, "/metrics"), headers=headers), "metrics scrape")
    series = parse_exposition(response.body.decode("utf-8", "replace"))
    if not any(item.name == IDENTIFYING_SERIES for item in series):
        raise LoadHarnessError(
            f"{response.url} parsed but carries no {IDENTIFYING_SERIES} series, so the "
            "target is not a fasterRag instance and four of the seven SLIs cannot be read "
            "from it",
            ExitCode.PREFLIGHT_FAILED,
        )
    return series


def _probe_query(base: str, headers: dict[str, str], config: LoadConfig, query: str) -> None:
    """Prove a real query answers and carries the timings the headline SLI is derived from."""
    payload: dict[str, Any] = {"query": query, "stream": False}
    if config.collection:
        payload["collection"] = config.collection

    _require_ok(
        fetch(urljoin(base, "/v1/retrieve"), method="POST", headers=headers, payload=payload),
        "retrieve probe",
    )
    response = _require_ok(
        fetch(urljoin(base, "/v1/query"), method="POST", headers=headers, payload=payload),
        "query probe",
    )

    body = response.json()
    if not isinstance(body, dict):
        raise LoadHarnessError(
            f"{response.url} returned {type(body).__name__}, not an answer object",
            ExitCode.PREFLIGHT_FAILED,
        )
    timings = body.get("timings_ms")
    if not isinstance(timings, dict) or not {key for key in timings if key != "generate"}:
        raise LoadHarnessError(
            f"{response.url} returned no pre-generate timings_ms keys (got {timings!r}). "
            "The p95 retrieval SLI is the sum of those keys, so this run could only report "
            "a latency it never measured",
            ExitCode.PREFLIGHT_FAILED,
        )


def _probe_ingest(base: str, headers: dict[str, str], config: LoadConfig) -> None:
    """Prove the ingest path accepts a job, which the ingest scenarios will do repeatedly."""
    payload: dict[str, Any] = {
        "sources": [{"type": "inline", "value": "fasterRag load-harness preflight document."}]
    }
    if config.collection:
        payload["collection"] = config.collection

    response = fetch(urljoin(base, "/v1/ingest"), method="POST", headers=headers, payload=payload)
    if response.status != 202:
        raise LoadHarnessError(
            f"ingest probe: {response.url} returned {response.status}, not 202 "
            f"({response.body[:300].decode('utf-8', 'replace')})",
            ExitCode.PREFLIGHT_FAILED,
        )
    body = response.json()
    if not isinstance(body, dict) or not body.get("job_id"):
        raise LoadHarnessError(
            f"ingest probe: {response.url} accepted the job but returned no job_id",
            ExitCode.PREFLIGHT_FAILED,
        )


def preflight(config: LoadConfig, api_key: str | None = None) -> PreflightResult:
    """Prove the target is a ready fasterRag that answers this scenario's requests."""
    queries = load_queries(config.dataset)
    documents = load_corpus(config.dataset) if config.scenario != "query" else ()

    base = config.target if config.target.endswith("/") else f"{config.target}/"
    headers = _headers(config, api_key)

    _require_ok(fetch(urljoin(base, "/healthz"), headers=headers), "liveness probe")
    _require_ok(fetch(urljoin(base, "/readyz"), headers=headers), "readiness probe")
    metrics = _probe_metrics(base, headers)

    if config.scenario != "ingest":
        _probe_query(base, headers, config, queries[0])
    if config.scenario != "query":
        _probe_ingest(base, headers, config)

    return PreflightResult(metrics=metrics, queries=queries, corpus_documents=len(documents))
