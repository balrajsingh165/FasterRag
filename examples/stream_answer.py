"""Stream an answer: print tokens as they arrive, then the event types the stream carried.

Same LLM requirement as ingest_and_ask.py.

    fasterrag config init --profile local
    python examples/stream_answer.py
"""

from __future__ import annotations

import asyncio

from _common import DATA, config_path, require

from fasterrag import FasterRag


async def main() -> None:
    """Ingest the sample folder, then stream the answer to one question."""
    kinds: list[str] = []
    text = ""
    async with FasterRag.from_config(config_path()) as rag:
        await rag.ingest([str(DATA)])
        async for event in rag.query_stream("Who designed the Fresnel lens?"):
            kinds.append(event.type)
            if event.type == "token":
                delta = str(event.data.get("text", ""))
                text += delta
                print(delta, end="", flush=True)
    print()
    print(f"event types, in order of first appearance: {list(dict.fromkeys(kinds))}")
    require(text.strip(), "the stream carried no tokens")


if __name__ == "__main__":
    asyncio.run(main())
