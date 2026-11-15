"""Run the local profile end to end against an installed fasterrag: ingest, then retrieve.

`config init --profile local`, ingest a Markdown note, and ask a question exactly one of its
chunks answers, with no network service and no API key (TASK-0444). The note is several
chunks long, so the check shows the right chunk ranks first among several rather than that
the only chunk comes back (TASK-0542).

Two callers run this same file, so neither can prove less than the other:
``scripts/check_package.py`` against the wheel it has just built, and
``.github/scripts/release_smoke.sh`` against the wheel about to be published and against the
version PyPI serves. It is run as ``python -I local_profile_probe.py <fasterrag console>`` in
an empty directory, and it imports nothing from this repository.
"""

from __future__ import annotations

import asyncio
import pathlib
import subprocess
import sys
from typing import Final

QUESTION: Final = "What makes sourdough rise?"

ANSWER: Final = "yeast"

_FILLER: Final[dict[str, tuple[str, ...]]] = {
    "Lighthouses": (
        "A Fresnel lens gathers the lamp's light into a beam that carries for many miles.",
        "Keepers once trimmed wicks through the night and wound the clockwork that turned it.",
        "Each tower has its own pattern of flashes so a navigator can tell which one it is.",
        "Stone towers on exposed reefs were built in courses dovetailed against the waves.",
        "Fog signals took over when the beam could not be seen, sounding at fixed intervals.",
    ),
    "Glaciers": (
        "A glacier forms where more snow falls each winter than melts the following summer.",
        "Its weight presses the buried snow into dense blue ice that flows slowly downhill.",
        "Moraines are the ridges of rock and gravel a glacier pushes ahead of it or leaves.",
        "Crevasses open where the ice bends over a step in the bedrock beneath the glacier.",
        "Meltwater carves channels under the ice and carries fine rock flour to the rivers.",
    ),
    "Honeybees": (
        "A worker honeybee tells her sisters where nectar is by dancing on the comb.",
        "The angle of the waggle run gives the direction of the flowers relative to the sun.",
        "Wax is secreted as thin scales from glands on the underside of a worker's abdomen.",
        "Bees fan their wings at the hive entrance to evaporate water out of the nectar.",
        "A colony overwinters as a tight cluster that shivers its flight muscles for warmth.",
    ),
}

_BREAD: Final = (
    "Sourdough rises because wild yeast and lactic acid bacteria ferment the dough, "
    "releasing carbon dioxide that the gluten network traps as bubbles."
)


def _section(title: str, sentences: tuple[str, ...], paragraphs: int) -> str:
    """Return one Markdown section of several paragraphs built from its own sentences."""
    body = "\n\n".join(" ".join(sentences) for _ in range(paragraphs))
    return f"## {title}\n\n{body}\n"


NOTE: Final = "\n".join(
    [
        "# Field notes\n",
        _section("Lighthouses", _FILLER["Lighthouses"], 6),
        _section("Glaciers", _FILLER["Glaciers"], 3),
        f"## Bread\n\n{_BREAD}\n",
        _section("Glaciers, continued", _FILLER["Glaciers"], 3),
        _section("Honeybees", _FILLER["Honeybees"], 6),
    ]
)
"""Several chunks at the local profile's chunk size; only the Bread section says ``ANSWER``."""


async def _retrieve(config: pathlib.Path) -> None:
    """Retrieve every chunk for the question and check the answering one ranks first."""
    from fasterrag.facade import FasterRag

    async with FasterRag.from_config(str(config)) as rag:
        chunks = await rag.retrieve(QUESTION, top_k=20)
    assert len(chunks) >= 3, f"expected several chunks, retrieved {len(chunks)}"
    assert ANSWER in chunks[0].text, chunks[0].text
    assert chunks[0].dense_rank == 1, chunks[0]
    others = [chunk for chunk in chunks if ANSWER not in chunk.text]
    assert len(others) >= 2, "the answer is in too many chunks to show a ranking"
    sys.stdout.write(f"retrieved: first of {len(chunks)} chunks: {chunks[0].text[:90]!r}\n")


def main(argv: list[str] | None = None) -> int:
    """Initialise the local profile in the current directory, ingest the note, retrieve."""
    import fasterrag

    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 1:
        sys.stderr.write("usage: python -I local_profile_probe.py <fasterrag console script>\n")
        return 2
    assert "site-packages" in str(fasterrag.__file__), fasterrag.__file__
    console = arguments[0]
    subprocess.run([console, "config", "init", "--profile", "local"], check=True)
    pathlib.Path("note.md").write_text(NOTE, encoding="utf-8")
    subprocess.run([console, "ingest", "note.md"], check=True)
    asyncio.run(_retrieve(pathlib.Path("config.yaml")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
