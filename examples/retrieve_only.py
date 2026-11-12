"""Retrieve without generating: ingest the sample documents, then print the ranked chunks.

Needs no LLM at all, so it runs as-is on the local profile (embedded Chroma, local embeddings).

    fasterrag config init --profile local
    python examples/retrieve_only.py
"""

from __future__ import annotations

import asyncio

from _common import DATA, config_path, require

from fasterrag import FasterRag


async def main() -> None:
    """Ingest the sample folder and show what retrieval ranks first for one question."""
    async with FasterRag.from_config(config_path()) as rag:
        job = await rag.ingest([str(DATA)])
        print(f"ingest job {job.job_id}: {job.status}")

        chunks = await rag.retrieve("Why does sourdough bread rise?", top_k=3)

    for chunk in chunks:
        text = " ".join(chunk.text.split())[:80]
        print(f"#{chunk.final_rank}  rrf={chunk.rrf_score:.4f}  {text}")
    require(chunks and "yeast" in chunks[0].text, "the sourdough passage was not ranked first")


if __name__ == "__main__":
    asyncio.run(main())
