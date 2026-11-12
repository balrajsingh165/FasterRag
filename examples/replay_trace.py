"""Replay a stored trace under a changed configuration and print what changed (D8).

Every answered query is stored as a trace (``traces.store``). Replay re-executes it with a
candidate configuration and diffs retrieval and answer against the original. Same LLM
requirement as ingest_and_ask.py.

    fasterrag config init --profile local
    python examples/replay_trace.py
"""

from __future__ import annotations

import asyncio

from _common import DATA, config_path, require

from fasterrag import FasterRag
from fasterrag.config import load_settings


async def main() -> None:
    """Answer a question, then replay its trace with ``retrieval.top_k`` cut to one."""
    path = config_path()
    async with FasterRag.from_config(path) as rag:
        await rag.ingest([str(DATA)])
        answer = await rag.query("What are spring tides?")
        print(f"original answer: {answer.answer}")

        candidate = load_settings(path, overrides=["retrieval.top_k=1"])
        result = await rag.replay(answer.trace_id, candidate)

    print(f"replayed trace {result.trace_id}")
    print(f"config changes: {result.config_changes}")
    print(f"retrieval diff: {result.retrieval.as_dict()}")
    print(f"replayed answer: {result.replayed_answer}")
    require(result.trace_id == answer.trace_id, "replay ran a different trace")
    require(result.config_changes, "the candidate configuration was not recorded as a change")


if __name__ == "__main__":
    asyncio.run(main())
