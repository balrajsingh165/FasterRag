"""Tunable parser thresholds, and how configuration reaches a parser.

Parsers are pure functions over bytes, so configuration cannot arrive the way it does in a
service: there is no object to construct and nothing to hold state. They take a small
frozen value object instead, built at the boundary by :func:`create_parsing_options`.

This mirrors ``create_chunker``, the house pattern for the same problem: the factory reads
``Settings`` once and hands the component plain values, and the component itself never
sees the schema. Two things follow that matter here.

* ``fasterrag.parsing`` is a documented standalone component (``docs/python-api.md``).
  ``parse_document("spec.pdf")`` has to keep working for a caller who has no
  ``config.yaml`` at all, which a parser reading ``Settings`` directly could not offer.
* Parsing runs inside ``ProcessPoolExecutor`` workers. The options are derived from the
  settings that already cross that boundary, inside the worker, so threading them adds
  nothing to what is pickled per document.

The defaults below are the shipped ``parsing`` defaults, restated here so a standalone
caller gets the same behaviour as the pipeline. A test asserts the two cannot drift.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from fasterrag.config.schema import Settings

__all__ = ["DEFAULT_PARSING_OPTIONS", "ParsingOptions", "create_parsing_options"]


@dataclass(frozen=True, slots=True)
class ParsingOptions:
    """The parser thresholds the ``parsing`` configuration section sets."""

    minimum_chars_per_page: int = 40
    ocr_resolution: int = 200
    heading_size_ratio: float = 1.15
    max_heading_chars: int = 120
    rows_per_block: int = 20


DEFAULT_PARSING_OPTIONS: Final = ParsingOptions()


def create_parsing_options(settings: Settings) -> ParsingOptions:
    """Build the parser thresholds named by the ``parsing`` section."""
    parsing = settings.parsing
    return ParsingOptions(
        minimum_chars_per_page=parsing.minimum_chars_per_page,
        ocr_resolution=parsing.ocr_resolution,
        heading_size_ratio=parsing.heading_size_ratio,
        max_heading_chars=parsing.max_heading_chars,
        rows_per_block=parsing.rows_per_block,
    )
