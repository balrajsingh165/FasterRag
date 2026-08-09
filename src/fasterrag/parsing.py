"""Public parsing surface: ``from fasterrag.parsing import parse_document``.

The documented standalone component (``docs/python-api.md``). Applications can adopt the
parser on its own — before, or instead of, the full engine — while the implementation
stays in ``fasterrag.core.parsing`` where the pipeline layering puts it.
"""

from fasterrag.core.parsing import (
    SUPPORTED_SUFFIXES,
    Block,
    ParsedDocument,
    ParseFlag,
    ParsingOptions,
    create_parsing_options,
    parse_bytes,
    parse_document,
)

__all__ = [
    "SUPPORTED_SUFFIXES",
    "Block",
    "ParseFlag",
    "ParsedDocument",
    "ParsingOptions",
    "create_parsing_options",
    "parse_bytes",
    "parse_document",
]
