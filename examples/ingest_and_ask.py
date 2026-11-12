"""Ingest a folder and ask a question: the whole pipeline, answer and citations included.

Generation needs an LLM. The local profile names a local Ollama (``ollama pull llama3.1``);
CI selects the offline scripted double instead, by configuration (see README.md).

    fasterrag config init --profile local
    python examples/ingest_and_ask.py
"""

from __future__ import annotations

import asyncio

from _common import DATA, config_path, require

from fasterrag import FasterRag


async def main() -> None:
    """Ingest the sample folder, ask one question, print the answer and what it cites."""
    async with FasterRag.from_config(config_path()) as rag:
        job = await rag.ingest([str(DATA)])
        print(f"ingest job {job.job_id}: {job.status}")

        answer = await rag.query("What causes the tides?")

    print(f"answer: {answer.answer}")
    for citation in answer.citations:
        print(f"  cited {citation.chunk_id} from {citation.source}")
    print(f"trace id: {answer.trace_id}")
    require(answer.answer, "no answer was produced")
    require(answer.citations, "the answer cites nothing")


if __name__ == "__main__":
    asyncio.run(main())
