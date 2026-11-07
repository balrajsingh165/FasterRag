"""The Locust adapter: the only module in this package that touches the load generator.

Everything that decides anything lives in ``harness.py`` and ``preflight.py``, which import
no third-party package at all. This file generates traffic and translates what it sees into
Locust's statistics, and it is kept deliberately small so that swapping the generator would
not mean rewriting the guards.

**Why Locust and not k6.** k6 is a single static Go binary, which is genuinely easier to
install and a faster generator per core. It was rejected on four counts specific to this
repository. Its scripts are JavaScript, and this tree is Python everywhere — one language
is a maintenance property, not a preference. A JavaScript runtime cannot import the pieces
this harness reuses: the committed golden-set loader, the machine fingerprint from
``services/benchmark.py``, the Prometheus parsing the SLI derivation needs; k6 would mean a
second implementation of each, and a second implementation of a measurement rule is how two
numbers that disagree get published. CI and the reference rig would need a binary installed
out of band, whereas ``pip install -r tests/load/requirements.txt`` is the same command this
project already uses for everything else. And a contributor without the tool is the case
that decided it: with k6 they get ``k6: command not found`` from a shell; here they get a
typed refusal naming the exact install command, because ``run_load.py`` checks for the
import before it runs anything.

The one real cost is stated rather than hidden: Locust's generator is Python and a single
process is GIL-bound, so at high request rates the generator saturates before the system
under test does. That is a measurement error in the dangerous direction — it makes fasterRag
look slow — so the operator procedure in ``docs/performance.md`` §5 requires watching the
generator's own CPU and using ``--processes``, and the harness records the generator's
hardware in every report alongside the system's.

**Locust is typed but optional.** It is deliberately absent from the ``dev`` extra, so mypy
in CI cannot resolve it. The base classes are therefore bound through explicitly ``Any``
names, which means the suppression below reads the same whether or not Locust is installed:
an ignore that is needed in one environment and unused in another fails strict mypy in
whichever one the author did not try.

Running this file directly — ``locust -f tests/load/locustfile.py`` — is supported and still
cannot skip the preflight or the vacuity audit: both are wired to Locust's own lifecycle
events here, not only to ``run_load.py``.
"""

from __future__ import annotations

import os
import random
import time
from typing import Any, Final

# CRITICAL: locust is an optional, load-only dependency that `.[dev]` does not install, so
# mypy in CI cannot find it. The names are rebound as `Any` below precisely so this single
# suppression behaves identically with and without the package present; widening it, or
# importing locust names directly, reintroduces the unused-ignore failure it avoids.
import locust  # type: ignore[import-not-found]

from tests.load.harness import (
    API_KEY_VARIABLE,
    INGEST_ENDPOINT,
    JOB_ENDPOINT,
    QUERY_ENDPOINT,
    RETRIEVAL_SLI_ENTRY,
    RETRIEVE_ENDPOINT,
    STATUS_2XX_ENTRY,
    STATUS_4XX_ENTRY,
    STATUS_5XX_ENTRY,
    STATUS_429_ENTRY,
    STREAM_ENDPOINT,
    TRANSPORT_ERROR_ENTRY,
    TTFT_SLI_ENTRY,
    ExitCode,
    LoadConfig,
    LoadHarnessError,
    config_from_env,
    load_corpus,
    load_queries,
)
from tests.load.preflight import preflight

_HttpUser: Any = locust.HttpUser
_between: Any = locust.between
_events: Any = locust.events

CONFIG: Final[LoadConfig] = config_from_env(os.environ)
QUERIES: Final = load_queries(CONFIG.dataset)
CORPUS: Final = load_corpus(CONFIG.dataset) if CONFIG.scenario != "query" else ()
API_KEY: Final = os.environ.get(API_KEY_VARIABLE, "").strip() or None

MIXED_QUERY_WEIGHT: Final = int(os.environ.get("FASTERRAG_LOAD_QUERY_WEIGHT", "4"))
JOB_POLL_TIMEOUT_S: Final = 120.0
JOB_POLL_INTERVAL_S: Final = 1.0


def _headers() -> dict[str, str]:
    """Return the headers every generated request carries."""
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"
    if CONFIG.tenant:
        headers["X-Tenant-ID"] = CONFIG.tenant
    return headers


def _fire(user: Any, name: str, elapsed_ms: float) -> None:
    """Record one derived observation into Locust's own statistics."""
    user.environment.events.request.fire(
        request_type="SLI",
        name=name,
        response_time=elapsed_ms,
        response_length=0,
        exception=None,
        context={},
    )


def _record_status(user: Any, status: int, elapsed_ms: float) -> None:
    """Classify one response, because availability and error rate are status ratios."""
    if 200 <= status < 300:
        _fire(user, STATUS_2XX_ENTRY, elapsed_ms)
    elif status == 429:
        _fire(user, STATUS_429_ENTRY, elapsed_ms)
    elif 400 <= status < 500:
        _fire(user, STATUS_4XX_ENTRY, elapsed_ms)
    elif status >= 500:
        _fire(user, STATUS_5XX_ENTRY, elapsed_ms)
    else:
        _fire(user, TRANSPORT_ERROR_ENTRY, elapsed_ms)


def _payload(query: str) -> dict[str, Any]:
    """Return a query body for the configured collection."""
    body: dict[str, Any] = {"query": query}
    if CONFIG.collection:
        body["collection"] = CONFIG.collection
    return body


def _retrieval_ms(body: Any) -> float | None:
    """Return the server's pre-generate stage total, or ``None`` when it reported none."""
    if not isinstance(body, dict):
        return None
    timings = body.get("timings_ms")
    if not isinstance(timings, dict):
        return None
    stages = [float(value) for key, value in timings.items() if key != "generate"]
    return sum(stages) if stages else None


def retrieve_task(user: Any) -> None:
    """Drive the retrieval-only path, which is the SLI's subject without generation."""
    payload = _payload(random.choice(QUERIES))
    with user.client.post(
        "/v1/retrieve",
        json=payload,
        headers=_headers(),
        name=RETRIEVE_ENDPOINT,
        catch_response=True,
    ) as response:
        _record_status(user, response.status_code, response.elapsed.total_seconds() * 1000)
        if response.status_code == 429:
            response.success()
        elif response.status_code != 200:
            response.failure(f"{response.status_code}: {response.text[:200]}")


def query_task(user: Any) -> None:
    """Drive a non-streamed answer and record the server's own stage timings."""
    payload = _payload(random.choice(QUERIES))
    with user.client.post(
        "/v1/query",
        json={**payload, "stream": False},
        headers=_headers(),
        name=QUERY_ENDPOINT,
        catch_response=True,
    ) as response:
        _record_status(user, response.status_code, response.elapsed.total_seconds() * 1000)
        if response.status_code == 429:
            response.success()
            return
        if response.status_code != 200:
            response.failure(f"{response.status_code}: {response.text[:200]}")
            return
        retrieval_ms = _retrieval_ms(response.json())
        if retrieval_ms is None:
            response.failure("the answer carried no pre-generate timings_ms")
            return
        _fire(user, RETRIEVAL_SLI_ENTRY, retrieval_ms)


def stream_task(user: Any) -> None:
    """Drive a streamed answer and time the first token event, which is the TTFT SLI."""
    payload = _payload(random.choice(QUERIES))
    started = time.perf_counter()
    with user.client.post(
        "/v1/query",
        json={**payload, "stream": True},
        headers={**_headers(), "Accept": "text/event-stream"},
        name=STREAM_ENDPOINT,
        stream=True,
        catch_response=True,
    ) as response:
        if response.status_code != 200:
            _record_status(user, response.status_code, (time.perf_counter() - started) * 1000)
            response.failure(f"{response.status_code}: {response.text[:200]}")
            return

        ttft_ms: float | None = None
        saw_done = False
        for raw in response.iter_lines():
            line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
            if line.startswith("event: token") and ttft_ms is None:
                ttft_ms = (time.perf_counter() - started) * 1000
            elif line.startswith("event: error"):
                response.failure("the stream ended on an error event")
                return
            elif line.startswith("event: done"):
                saw_done = True

        _record_status(user, response.status_code, (time.perf_counter() - started) * 1000)
        if ttft_ms is None:
            response.failure("the stream produced no token event, so there is no TTFT")
            return
        if not saw_done:
            response.failure("the stream ended without a done event")
            return
        _fire(user, TTFT_SLI_ENTRY, ttft_ms)


def ingest_task(user: Any) -> None:
    """Submit one document and follow its job, which is what concurrent ingest costs."""
    document = random.choice(CORPUS)
    body: dict[str, Any] = {
        "sources": [{"type": "inline", "value": document.read_text(encoding="utf-8")}],
        "metadata": {"source_name": document.name},
    }
    if CONFIG.collection:
        body["collection"] = CONFIG.collection

    with user.client.post(
        "/v1/ingest",
        json=body,
        headers=_headers(),
        name=INGEST_ENDPOINT,
        catch_response=True,
    ) as response:
        _record_status(user, response.status_code, response.elapsed.total_seconds() * 1000)
        if response.status_code == 429:
            response.success()
            return
        if response.status_code != 202:
            response.failure(f"{response.status_code}: {response.text[:200]}")
            return
        job_id = response.json().get("job_id")
        if not job_id:
            response.failure("the accepted job carried no job_id")
            return

    _await_job(user, str(job_id))


def _await_job(user: Any, job_id: str) -> None:
    """Poll one ingestion job to a terminal state, so the run measures completed work."""
    deadline = time.perf_counter() + JOB_POLL_TIMEOUT_S
    while time.perf_counter() < deadline:
        with user.client.get(
            f"/v1/ingest/{job_id}",
            headers=_headers(),
            name=JOB_ENDPOINT,
            catch_response=True,
        ) as response:
            _record_status(user, response.status_code, response.elapsed.total_seconds() * 1000)
            if response.status_code != 200:
                response.failure(f"{response.status_code}: {response.text[:200]}")
                return
            status = str(response.json().get("status", ""))
        if status in {"succeeded", "failed", "partial"}:
            return
        time.sleep(JOB_POLL_INTERVAL_S)


def _read_tasks() -> dict[Any, int]:
    """Return the reader task mix, dropping any task the operator weighted to zero."""
    mix = {retrieve_task: 3, query_task: 3, stream_task: CONFIG.stream_weight}
    return {name: weight for name, weight in mix.items() if weight > 0}


class QueryUser(_HttpUser):  # type: ignore[misc]
    """A reader: retrieval, non-streamed answers, and streamed answers."""

    host = CONFIG.target
    weight = MIXED_QUERY_WEIGHT
    wait_time = _between(0.0, 0.1)
    tasks = _read_tasks()


class IngestUser(_HttpUser):  # type: ignore[misc]
    """A writer: submit a document and follow its job to a terminal state."""

    host = CONFIG.target
    weight = 1
    wait_time = _between(0.0, 0.1)
    tasks = (ingest_task,)


# CRITICAL: the preflight's verdict is held here as well as on the environment, so the
# quitting handler cannot overwrite a preflight refusal with the merely-empty code the same
# run also earns. A run aborted before its first request is both at once, and "the target
# was unreachable" is the one an operator can act on.
_FATAL: Final[dict[str, int]] = {}


def _on_test_start(environment: Any, **_: Any) -> None:
    """Re-run the preflight inside Locust, so a direct invocation cannot skip it."""
    try:
        preflight(CONFIG, API_KEY)
    except LoadHarnessError as exc:
        _FATAL["exit_code"] = int(exc.exit_code)
        environment.process_exit_code = int(exc.exit_code)
        environment.runner.quit()
        raise


def _on_quitting(environment: Any, **_: Any) -> None:
    """Refuse to exit zero on a run that recorded nothing, however it was launched."""
    if "exit_code" in _FATAL:
        environment.process_exit_code = _FATAL["exit_code"]
        return
    stats = environment.stats
    if stats.total.num_requests <= 0:
        environment.process_exit_code = int(ExitCode.VACUOUS)
    elif stats.total.num_failures / stats.total.num_requests > CONFIG.max_failure_ratio:
        environment.process_exit_code = int(ExitCode.TOO_MANY_FAILURES)


_events.test_start.add_listener(_on_test_start)
_events.quitting.add_listener(_on_quitting)
