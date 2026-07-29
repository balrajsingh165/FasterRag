"""fasterRag — a backend-only Retrieval-Augmented Generation framework.

The importable package is one of three control surfaces (REST API, CLI, library) that
all call the same service layer, so behavior, configuration, and errors are identical
everywhere. The public compatibility contract is specified in ``docs/python-api.md``:
names exported here and from documented submodules are stable under SemVer; anything
under ``fasterrag._internal`` or absent from ``__all__`` is private.
"""

__all__ = ["__version__"]

__version__ = "0.1.0.dev0"
