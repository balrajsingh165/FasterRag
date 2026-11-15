"""Public parsing surface: ``from fasterrag.parsing import parse_document``.

The documented standalone component (``docs/python-api.md``). Applications can adopt the
parser on its own — before, or instead of, the full engine — while the implementation
stays in ``fasterrag.core.parsing`` where the pipeline layering puts it. A parser plugin
implements :class:`ParserPlugin` with a :class:`DocumentBuilder`, declares
``adapter_api_version`` against :data:`PARSER_API_VERSION` and registers under
:data:`PARSERS_GROUP` (``docs/plugins.md``).
"""

from fasterrag.core.parsing import (
    PARSER_API_VERSION,
    PARSERS_GROUP,
    SUPPORTED_SUFFIXES,
    Block,
    DocumentBuilder,
    ParsedDocument,
    ParseFlag,
    ParserPlugin,
    ParsingOptions,
    create_parsing_options,
    parse_bytes,
    parse_document,
)

__all__ = [
    "PARSERS_GROUP",
    "PARSER_API_VERSION",
    "SUPPORTED_SUFFIXES",
    "Block",
    "DocumentBuilder",
    "ParseFlag",
    "ParsedDocument",
    "ParserPlugin",
    "ParsingOptions",
    "create_parsing_options",
    "parse_bytes",
    "parse_document",
]
