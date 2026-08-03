"""fasterRag — a backend-only Retrieval-Augmented Generation framework.

The importable package is one of three control surfaces (REST API, CLI, library) that
all call the same service layer, so behavior, configuration, and errors are identical
everywhere. The public compatibility contract is specified in ``docs/python-api.md``:
names exported here and from documented submodules are stable under SemVer; anything
under ``fasterrag._internal`` or absent from ``__all__`` is private.
"""

__version__ = "0.1.0.dev0"

# CRITICAL: imported lazily through __getattr__, not at module import. Importing the facade
# eagerly would pull the adapter factories, the chunking stack, and pydantic-settings into
# every `import fasterrag` — including `fasterrag.__version__` in a packaging script, and
# the CLI's own startup. The facade costs roughly a second to import; a version read should
# not.
__all__ = ["FasterRag", "__version__"]


def __getattr__(name: str) -> object:
    """Resolve the documented lazy exports on first access."""
    if name == "FasterRag":
        from fasterrag.facade import FasterRag

        return FasterRag
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
