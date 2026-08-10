"""Chaos variants injected at the operating system rather than at a code seam (D12).

``tests/chaos/test_chaos.py`` injects the stop-Qdrant and disk-full scenarios at the adapter
and filesystem seams. That proves how fasterRag responds to those conditions but not how the
operating system reports them, and the two are not the same thing: a double raises the
exception its author expected, whereas a stopped container takes its published port with it
— refusing the connection, or, under Docker Desktop's proxy, accepting it and hanging until
our own timeout fires — and a full filesystem returns ``ENOSPC`` from a syscall a client
library has to translate first.
These variants close that gap for both scenarios (TASK-0135). They complement the scripted
suite rather than replacing it — the seam-level cases still run without Docker.

* **Container stop.** A throwaway Qdrant with its own container name, ports, and named
  volume is really stopped with ``docker stop`` and really started again. Nothing here goes
  near the container the provisioner manages or the volume holding its data.

* **Disk quota.** Windows cannot constrain a filesystem without administrator rights: a VHD
  has to be mounted and an NTFS quota has to be set, and both need elevation a test must
  never require. The journal write therefore runs inside a Linux container whose journal
  directory is a tmpfs mounted with a hard size limit, so the ``ENOSPC`` the real journal
  sees comes from a real filesystem rather than from a directory made unwritable. The same
  path runs unchanged on a Linux or macOS host.

Both provisioners skip when the Docker daemon is unavailable, the way
``tests/integration/conftest.py`` does, so the suite stays runnable without Docker.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import pytest

import fasterrag
from fasterrag.adapters.vectordb.base import CollectionSpec, Point, SearchQuery
from fasterrag.adapters.vectordb.qdrant import QdrantAdapter
from fasterrag.api.problems import build_problem
from fasterrag.config.schema import Settings
from fasterrag.errors import ErrorCode, FasterRagError, ProvisioningError
from fasterrag.services.provisioning import docker_available, run_docker

pytestmark = [pytest.mark.chaos, pytest.mark.integration]

# CRITICAL: these must never collide with the provisioner's own container, volume, or ports
# (fasterrag-qdrant, fasterrag_qdrant_storage, 6333/6334). A developer machine commonly has
# that container running with real state in it, and a chaos case that stopped or removed it
# would be a fault injected into the developer rather than into the system under test.
QDRANT_CONTAINER: Final = "fasterrag-chaos-qdrant"
QDRANT_VOLUME: Final = "fasterrag_chaos_qdrant_storage"
QDRANT_IMAGE: Final = "qdrant/qdrant:v1.18.1"
REST_PORT: Final = 6733
GRPC_PORT: Final = 6734
COLLECTION: Final = "chaos-real"
POINT_ID: Final = "c_chaos_1"
VECTOR: Final = [0.1, 0.2, 0.3]

DISK_CONTAINER: Final = "fasterrag-chaos-diskquota"
DISK_IMAGE: Final = "python:3.12-slim"
DISK_MOUNT: Final = "/journal"
DISK_QUOTA: Final = "512k"
PROBE_DEPENDENCY: Final = "pydantic-settings[yaml]"

READY_TIMEOUT_SECONDS: Final = 90.0
READY_POLL_SECONDS: Final = 0.5
DOCKER_TIMEOUT_SECONDS: Final = 120.0
PULL_TIMEOUT_SECONDS: Final = 600.0
HANG_BUDGET_SECONDS: Final = 30.0
FAIL_FAST_SLACK_SECONDS: Final = 5.0
SERVICE_UNAVAILABLE: Final = 503


def chaos_settings() -> Settings:
    """Return settings pointed at the throwaway container, never at the managed one."""
    return Settings.model_validate(
        {
            "vector_db": {
                "mode": "external",
                "host": "localhost",
                "port": REST_PORT,
                "grpc_port": GRPC_PORT,
                "api_key_env": None,
            },
            "embeddings": {"provider": "huggingface"},
            "llm": {"provider": "ollama", "api_key_env": None},
        }
    )


def require_docker() -> None:
    """Skip rather than fail when Docker cannot be used.

    A daemon that is merely stopped makes ``docker info`` exit non-zero, but a machine with
    no Docker CLI at all makes the provisioner raise instead, and both mean the same thing
    here. The skip carries the reason so an unusable daemon in CI is still legible.
    """
    try:
        available = asyncio.run(docker_available())
    except ProvisioningError as exc:
        pytest.skip(f"docker is not usable: {exc.detail}")

    if not available:
        pytest.skip("docker daemon is not available")


def query(collection: str = COLLECTION) -> SearchQuery:
    """Return the one search every case in this module runs."""
    return SearchQuery(collection=collection, vector=VECTOR, limit=1)


async def wait_for_health(settings: Settings) -> bool:
    """Poll until the backend answers an API call, or the readiness budget runs out."""
    adapter = QdrantAdapter(settings)
    deadline = time.monotonic() + READY_TIMEOUT_SECONDS
    try:
        while time.monotonic() < deadline:
            if (await adapter.health()).healthy:
                return True
            await asyncio.sleep(READY_POLL_SECONDS)
        return False
    finally:
        await adapter.close()


async def docker_or_fail(args: Sequence[str], *, timeout: float = DOCKER_TIMEOUT_SECONDS) -> str:
    """Run a docker command, failing the test with its stderr if it does not succeed."""
    result = await run_docker(args, timeout=timeout)
    if not result.ok:
        pytest.fail(f"docker {' '.join(args)} failed: {result.stderr or result.stdout}")
    return result.stdout


async def start_throwaway_qdrant(settings: Settings) -> None:
    """Create the throwaway container, its volume, and the collection cases search."""
    await run_docker(["rm", "--force", QDRANT_CONTAINER], timeout=DOCKER_TIMEOUT_SECONDS)
    await docker_or_fail(["volume", "create", QDRANT_VOLUME])
    await docker_or_fail(
        [
            "run",
            "--detach",
            "--name",
            QDRANT_CONTAINER,
            "--publish",
            f"{REST_PORT}:6333",
            "--publish",
            f"{GRPC_PORT}:6334",
            "--env",
            "QDRANT__SERVICE__GRPC_PORT=6334",
            "--volume",
            f"{QDRANT_VOLUME}:/qdrant/storage",
            QDRANT_IMAGE,
        ],
        timeout=PULL_TIMEOUT_SECONDS,
    )

    if not await wait_for_health(settings):
        pytest.fail(f"the throwaway qdrant never answered within {READY_TIMEOUT_SECONDS:.0f}s")

    adapter = QdrantAdapter(settings)
    try:
        await adapter.create_collection(
            CollectionSpec(name=COLLECTION, dimensions=len(VECTOR)),
        )
        await adapter.upsert(
            [
                Point(
                    point_id=POINT_ID,
                    collection=COLLECTION,
                    vector=VECTOR,
                    payload={"source_uri": "corpus/doc.pdf"},
                )
            ]
        )
    finally:
        await adapter.close()


async def remove_throwaway_qdrant() -> None:
    """Remove the throwaway container and its volume, leaving the machine as it was."""
    await run_docker(["rm", "--force", QDRANT_CONTAINER], timeout=DOCKER_TIMEOUT_SECONDS)
    await run_docker(["volume", "rm", "--force", QDRANT_VOLUME], timeout=DOCKER_TIMEOUT_SECONDS)


@pytest.fixture(scope="session")
def real_qdrant() -> Iterator[Settings]:
    """Provision a throwaway Qdrant holding one indexed point, and remove it afterwards."""
    require_docker()

    settings = chaos_settings()
    asyncio.run(start_throwaway_qdrant(settings))
    yield settings
    asyncio.run(remove_throwaway_qdrant())


@pytest.fixture
async def adapter(real_qdrant: Settings) -> AsyncIterator[QdrantAdapter]:
    """Return an adapter pointed at the throwaway container."""
    built = QdrantAdapter(real_qdrant)
    yield built
    await built.close()


@pytest.fixture
def stopped(real_qdrant: Settings) -> Iterator[Settings]:
    """Really stop the throwaway container for one case, then bring it back.

    The restart lives in teardown so a failing assertion cannot leave the container down for
    whatever runs next, and ``docker start`` is idempotent, so a case that restarts the
    container itself to assert recovery is free to do so.
    """
    asyncio.run(docker_or_fail(["stop", QDRANT_CONTAINER]))
    yield real_qdrant
    asyncio.run(docker_or_fail(["start", QDRANT_CONTAINER]))
    if not asyncio.run(wait_for_health(real_qdrant)):
        pytest.fail("the throwaway qdrant did not come back after the chaos case")


async def test_a_real_query_against_a_running_container_succeeds(adapter: QdrantAdapter) -> None:
    """The baseline: without this, a later failure proves nothing about the fault."""
    hits = await adapter.search(query())

    assert [hit.point_id for hit in hits] == [POINT_ID]


async def test_a_really_stopped_container_fails_typed_retryable_and_fast(
    adapter: QdrantAdapter, stopped: Settings
) -> None:
    """A container stopped by Docker fails typed and retryable inside the timeout budget.

    This is the variant that exercises the operating system's reporting: the published port
    disappears with the container, so the refusal comes from the kernel rather than from a
    double that decided to raise. The error is still classified retryable, which is what lets
    a breaker act on it instead of treating a restart as a permanent fault.

    What is asserted is the *bound*, not a latency figure, because the latency is bimodal and
    a pinned number would be a lie on half the runs. Four consecutive runs on the Windows +
    WSL2 host measured 44.2 ms, 5040.9 ms, 36.1 ms, 5043.9 ms: sometimes the kernel refuses
    the connection immediately, and sometimes Docker Desktop's port proxy accepts and hangs,
    so the failure arrives from ``reliability.timeouts.vector_db_ms`` (5 s) instead. Both are
    the guarantee — a bounded failure, never an unbounded hang — and only the second shows
    that the bound is ours rather than the operating system's.

    The code is ``RETRIEVAL_FAILED``, the same one the scripted suite's double raises, which
    is what TASK-0226 corrected: a real outage now names the vector database rather than
    sending an operator to inspect a healthy embedding provider.
    """
    started = time.perf_counter()

    with pytest.raises(FasterRagError) as failure:
        await asyncio.wait_for(adapter.search(query()), timeout=HANG_BUDGET_SECONDS)

    elapsed = time.perf_counter() - started
    budget = stopped.reliability.timeouts.vector_db_ms / 1000 + FAIL_FAST_SLACK_SECONDS

    assert failure.value.retryable is True
    assert failure.value.status == SERVICE_UNAVAILABLE
    assert failure.value.code is ErrorCode.RETRIEVAL_FAILED
    assert elapsed < budget


async def test_a_really_stopped_container_reports_itself_unhealthy(
    adapter: QdrantAdapter, stopped: Settings
) -> None:
    """The readiness probe describes the outage instead of raising through it."""
    status = await adapter.health()

    assert status.healthy is False
    assert status.detail


async def test_a_really_stopped_container_never_yields_a_generic_500(
    adapter: QdrantAdapter, stopped: Settings
) -> None:
    """The failure renders as a problem document a client can branch on."""
    with pytest.raises(FasterRagError) as failure:
        await asyncio.wait_for(adapter.search(query()), timeout=HANG_BUDGET_SECONDS)

    problem = build_problem(failure.value)

    assert problem.status == SERVICE_UNAVAILABLE
    assert problem.retryable is True
    assert problem.code is not ErrorCode.INTERNAL
    assert problem.trace_id


async def test_a_restarted_container_serves_its_indexed_data_again(
    adapter: QdrantAdapter, stopped: Settings
) -> None:
    """Recovery is automatic, and the named volume kept the data across the stop."""
    await docker_or_fail(["start", QDRANT_CONTAINER])

    assert await wait_for_health(stopped)

    hits = await adapter.search(query())

    assert [hit.point_id for hit in hits] == [POINT_ID]


async def run_disk_quota_probe() -> Mapping[str, Any]:
    """Run the journal write path inside a container whose journal tmpfs is size-limited."""
    await run_docker(["rm", "--force", DISK_CONTAINER], timeout=DOCKER_TIMEOUT_SECONDS)
    await docker_or_fail(
        [
            "run",
            "--detach",
            "--name",
            DISK_CONTAINER,
            "--tmpfs",
            f"{DISK_MOUNT}:size={DISK_QUOTA}",
            DISK_IMAGE,
            "sleep",
            "600",
        ],
        timeout=PULL_TIMEOUT_SECONDS,
    )
    await docker_or_fail(["exec", DISK_CONTAINER, "mkdir", "-p", "/src"])

    package = Path(str(fasterrag.__file__)).parent
    await docker_or_fail(["cp", str(package), f"{DISK_CONTAINER}:/src/"])
    probe = Path(__file__).with_name("disk_quota_probe.py")
    await docker_or_fail(["cp", str(probe), f"{DISK_CONTAINER}:/probe.py"])

    installed = await run_docker(
        ["exec", DISK_CONTAINER, "pip", "install", "--quiet", PROBE_DEPENDENCY],
        timeout=PULL_TIMEOUT_SECONDS,
    )
    if not installed.ok:
        pytest.skip(f"the probe container could not install its imports: {installed.stderr[:200]}")

    reported = await docker_or_fail(
        ["exec", "--env", "PYTHONPATH=/src", DISK_CONTAINER, "python", "/probe.py"],
        timeout=PULL_TIMEOUT_SECONDS,
    )
    parsed: Mapping[str, Any] = json.loads(reported.splitlines()[-1])
    return parsed


@pytest.fixture(scope="session")
def disk_quota() -> Iterator[Mapping[str, Any]]:
    """Return the probe's report from a journal filesystem that really ran out of space."""
    require_docker()

    try:
        yield asyncio.run(run_disk_quota_probe())
    finally:
        asyncio.run(run_docker(["rm", "--force", DISK_CONTAINER], timeout=DOCKER_TIMEOUT_SECONDS))


def test_the_probe_really_ran_out_of_space(disk_quota: Mapping[str, Any]) -> None:
    """The fault is real: the filesystem reports zero free bytes, not a mocked failure."""
    assert disk_quota["filled_bytes"] > 0
    assert disk_quota["free_bytes"] == 0
    assert disk_quota["baseline"]["loaded"] is True


def test_a_genuinely_full_filesystem_halts_the_journal_loudly(
    disk_quota: Mapping[str, Any],
) -> None:
    """Starting a new job on a full filesystem raises; it never silently loses the record.

    The assertion is on the operating system's own words rather than on ``errno``, so that
    translating the failure into a typed error (TASK-0234) keeps this case passing — the
    guarantee is that the write is loud, not that it stays untyped. What must go red when
    that lands is ``test_the_journal_does_not_yet_type_a_full_disk`` alone.
    """
    failure = disk_quota["create_while_full"]

    assert failure is not None
    assert "No space left on device" in failure["detail"]


def test_a_genuinely_full_filesystem_does_not_corrupt_the_checkpoint(
    disk_quota: Mapping[str, Any],
) -> None:
    """The clean halt of failure-modes.md row 33: checkpointing survives a full disk.

    Measured, a *new* job cannot be written while a *running* one can still checkpoint, and
    the asymmetry is not luck. ``_write_atomically`` renames the current record over the
    stale ``.previous`` copy before writing, which frees a file the same size as the one it
    is about to write; a new job frees nothing and so has nowhere to go. That is what makes
    "halt cleanly at a checkpoint" reachable on the very disk that caused the halt.
    """
    assert disk_quota["checkpoint_while_full"] is None

    surviving = disk_quota["reload_while_full"]

    assert surviving["loaded"] is True
    assert surviving["checkpoint"] is not None


def test_the_journal_writes_again_once_space_is_freed(disk_quota: Mapping[str, Any]) -> None:
    """Recovery is nothing more than freeing space; the job is still there to resume."""
    assert disk_quota["create_after_free"]["ok"] is True
    assert disk_quota["reload_after_free"]["loaded"] is True


def test_the_journal_does_not_yet_type_a_full_disk(disk_quota: Mapping[str, Any]) -> None:
    """The gap this variant exists to expose, asserted so it cannot be forgotten.

    failure-modes.md row 33 promised the OS write error becomes a typed ``IngestionError``.
    It does not: the raw ``OSError`` escapes ``Journal._write_atomically`` untranslated, and
    only a real ``ENOSPC`` shows it, because the scripted suite's unwritable-directory case
    accepts ``(IngestionError, OSError)`` and so passes either way. Filed as TASK-0234. When
    that lands this case fails, which is the point — the fix must move the row and this
    assertion together.
    """
    failure = disk_quota["create_while_full"]

    assert failure["typed"] is False
    assert failure["type"] == "OSError"
    assert failure["errno"] == 28
