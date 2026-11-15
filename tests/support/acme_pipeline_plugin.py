"""A third-party chunker and parser, written the way a plugin author would write them.

Nothing in fasterRag names these classes. They are reached only through the
``fasterrag.chunkers`` and ``fasterrag.parsers`` entry points a fake distribution registers
(``tests/support/plugins.py``), so when one of them shapes an indexed chunk, configuration
selected it by name — in the parent process for validation and in a spawned parse worker
for the work itself.

The format the parser reads is deliberately one no built-in parser knows: ``key: value``
records, one per line, which it rewrites as sentences. A chunk whose text is such a sentence
therefore proves the plugin parser ran, and a chunk holding exactly one record proves the
plugin chunker drew the boundaries.
"""

from __future__ import annotations

from typing import ClassVar

from fasterrag.config.schema import Settings
from fasterrag.core.chunking.models import TextChunk, TokenCounter
from fasterrag.core.parsing.models import DocumentBuilder, ParsedDocument
from fasterrag.core.parsing.options import ParsingOptions

__all__ = [
    "AcmeBlockChunker",
    "AcmeRecordParser",
    "FutureChunker",
    "NotAChunker",
    "UnversionedParser",
]


class AcmeBlockChunker:
    """One chunk per parsed block, whatever the configured size."""

    adapter_api_version: ClassVar[str] = "1.0"
    strategy: ClassVar[str] = "acme-blocks"

    def __init__(self, settings: Settings, *, counter: TokenCounter) -> None:
        """Keep the counter the pipeline hands every chunker."""
        self.settings = settings
        self.counter = counter

    def split(self, document: ParsedDocument) -> list[TextChunk]:
        """Return each block of ``document`` as its own chunk."""
        return [
            TextChunk(
                text=block.text,
                start=block.start,
                end=block.end,
                chunk_index=index,
                token_count=self.counter.count(block.text),
                strategy=self.strategy,
                page=block.page,
                section=block.section,
            )
            for index, block in enumerate(document.blocks)
        ]


class FutureChunker(AcmeBlockChunker):
    """A chunker built against a chunker API this installation does not provide."""

    adapter_api_version: ClassVar[str] = "2.0"


class NotAChunker:
    """An entry point that names a class with no ``split``."""

    adapter_api_version: ClassVar[str] = "1.0"


class AcmeRecordParser:
    """Reads ``key: value`` lines and writes each one as a sentence."""

    adapter_api_version: ClassVar[str] = "1.0"

    def parse(self, data: bytes, *, filename: str, options: ParsingOptions) -> ParsedDocument:
        """Return one paragraph per record in ``data``."""
        builder = DocumentBuilder(mime_type="application/x-acme-records", parser="acme-records")
        for line in data.decode("utf-8").splitlines():
            key, separator, value = line.partition(":")
            if separator:
                builder.add("paragraph", f"The {key.strip()} is {value.strip()}.")
        builder.meta(source_format="acme-records", filename=filename)
        return builder.build()


class UnversionedParser:
    """A parser that never says which parser API it was written against."""

    def parse(self, data: bytes, *, filename: str, options: ParsingOptions) -> ParsedDocument:
        """Return nothing useful; it is refused before it is called."""
        return DocumentBuilder(mime_type="text/plain", parser="unversioned").build()
