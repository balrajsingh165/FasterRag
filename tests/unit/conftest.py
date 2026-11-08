"""Unit-tier guard: a unit test may not open a network client.

TASK-0353. ``qdrant_client/async_qdrant_remote.py:231: UserWarning: Failed to obtain server
version`` appeared in 1 of 6 full unit runs on a frozen tree, printed to stderr *after* the
final progress line. Nothing was wrong with pytest's warning capture: ``AsyncQdrantRemote``
starts a **daemon thread** in its constructor that issues a real HTTP GET for the server
version, and a warning raised on that thread belongs to no test. It lands whenever the
thread happens to finish, which is why it appeared once in six runs and changed no count.

So the warning was never the defect — constructing the client was. A unit test that reaches
the network is a unit test whose result depends on the machine it ran on, and suppressing
the warning would have kept exactly that. This guard refuses the construction instead, at
the two classes that speak to a server, naming the test that did it. Local-mode clients
(``:memory:`` and on-disk) go through ``AsyncQdrantLocal`` and are untouched, as are the
integration tests, which live outside this directory and are supposed to connect.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, Final

import pytest
from qdrant_client.async_qdrant_remote import AsyncQdrantRemote
from qdrant_client.qdrant_remote import QdrantRemote

_REMOTE_CLIENTS: Final[tuple[Any, ...]] = (AsyncQdrantRemote, QdrantRemote)


def _refusal(name: str, nodeid: str) -> Callable[..., None]:
    """Build the constructor that refuses, naming the test and the way out."""

    def refuse(self: Any, *args: Any, **kwargs: Any) -> None:
        raise AssertionError(
            f"{nodeid} constructed a real {name}, which opens a socket and starts a "
            "background version check against a server. Substitute a fake client, or move "
            "the test to tests/integration/ where connecting is the point (TASK-0353)."
        )

    return refuse


# CRITICAL: patched by hand rather than through the `monkeypatch` fixture. An autouse
# conftest fixture that depends on `monkeypatch` makes pytest build `monkeypatch` before
# every module-level autouse fixture in the tier, which inverts their teardown order — and
# a module fixture that undoes its own patching after the test (tests/unit/core/chunking/
# test_tokenizer.py clears an lru_cache that another fixture had replaced) then runs while
# the monkeypatched object is still installed. The first version of this guard did exactly
# that and broke eight tests in a file it has nothing to do with.
@pytest.fixture(autouse=True)
def _no_network_vector_clients(request: pytest.FixtureRequest) -> Iterator[None]:
    """Fail the test that builds a real Qdrant client instead of letting it dial out."""
    originals = [(remote, remote.__init__) for remote in _REMOTE_CLIENTS]
    for remote in _REMOTE_CLIENTS:
        remote.__init__ = _refusal(remote.__name__, request.node.nodeid)
    try:
        yield
    finally:
        for remote, original in originals:
            remote.__init__ = original
