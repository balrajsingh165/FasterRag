"""The shared vector database contract suite, re-exported from ``fasterrag.testing.vectordb``.

The suite ships in the wheel since TASK-0551, so a third-party plugin author can run it
from an installed fasterrag. This module keeps the in-repo import path, which every
built-in adapter's contract test uses, pointing at the same objects.
"""

from __future__ import annotations

from fasterrag.testing.vectordb import DIMENSIONS, VECTORS, VectorDBContract, point

__all__ = ["DIMENSIONS", "VECTORS", "VectorDBContract", "point"]
