"""Swap a provider by configuration alone: the same code against two vector databases.

The first run uses whatever config.yaml names (embedded Chroma on the local profile); the second
overrides one key, ``vector_db.provider``, to the ``memory`` store from the sample plugin in
examples/plugin-vectordb. No line of application code changes between the two. The same
override works from the shell as ``FASTERRAG_SET=vector_db.provider=memory`` or ``--set``.

The second run also turns ``ingestion.dedup`` off, for a reason that belongs to the sample store
and not to swapping. The dedup journal is kept per vector database, so a backend it has not seen
starts with nothing known and receives every file. But the memory store forgets everything when
its process exits while the journal on disk does not, so from this script's second run on, the
journal would remember files the store no longer holds. A backend that keeps its data needs no
such override.

    pip install ./examples/plugin-vectordb
    fasterrag config init --profile local
    python examples/swap_provider.py
"""

from __future__ import annotations

import asyncio

from _common import DATA, config_path, require

from fasterrag import FasterRag
from fasterrag.adapters.vectordb import available_providers
from fasterrag.config import load_settings

QUESTION = "How often does a refrigerated starter need feeding?"


async def top_chunk(rag: FasterRag) -> str:
    """Ingest the sample folder into ``rag`` and return its best chunk for the question."""
    async with rag:
        await rag.ingest([str(DATA)])
        chunks = await rag.retrieve(QUESTION, top_k=1)
        name = type(rag.vector_db).__name__
    require(chunks, f"{name} returned no chunk")
    print(f"{name:>16}: {' '.join(chunks[0].text.split())[:70]}")
    return chunks[0].text


async def main() -> None:
    """Retrieve once per vector database and show both found the same passage."""
    if "memory" not in available_providers():
        raise SystemExit("install the sample plugin first: pip install ./examples/plugin-vectordb")

    path = config_path()
    first = await top_chunk(FasterRag.from_settings(load_settings(path)))
    swapped = load_settings(path, overrides=["vector_db.provider=memory", "ingestion.dedup=false"])
    second = await top_chunk(FasterRag.from_settings(swapped))

    require("week" in first and first == second, "the two stores disagreed on the best passage")


if __name__ == "__main__":
    asyncio.run(main())
