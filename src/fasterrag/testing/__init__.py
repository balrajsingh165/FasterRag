"""Test suites a plugin author runs against their own plugin: ``fasterrag.testing``.

The adapter contract suite is what every built-in vector database passes, and a plugin is
conformant only when it passes it too (``docs/plugins.md``). Until TASK-0551 it lived only
in ``tests/contract/`` of the repository, which the wheel does not contain, so a third-party
author had to run it from a checkout. It is shipped here instead, and ``tests/contract/``
re-exports it, so a built-in adapter and a plugin run the identical cases.

pytest is not a dependency of fasterrag; it comes with the ``testing`` extra
(``pip install "fasterrag[testing]"``, which also brings pytest-asyncio for the suite's
coroutine cases). Nothing in the rest of the package imports this one, so the core install
never needs pytest; importing this package without it fails with that install line.

Assertion rewriting is registered for the suite modules before they are imported, so a
failing case shows the compared values as it would in a test file of your own.
"""

from __future__ import annotations

import importlib.util

if importlib.util.find_spec("pytest") is None:
    raise ImportError(
        "fasterrag.testing runs under pytest, which is not installed; install it with "
        'pip install "fasterrag[testing]"'
    )

import pytest

pytest.register_assert_rewrite("fasterrag.testing.vectordb")

__all__: list[str] = []
