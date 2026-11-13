---
title: fasterRag — a backend-only RAG framework for FastAPI
description: >-
  fasterRag is an open-source, backend-only Retrieval-Augmented Generation framework built on
  FastAPI: ingest documents, ask questions, get cited answers. Any vector database, embedding
  model or LLM provider, chosen through configuration. REST API, CLI and Python package.
hide:
  - toc
---

# fasterRag

<div class="fr-stage">
<div class="fr-orb fr-orb--a"></div><div class="fr-orb fr-orb--b"></div><div class="fr-orb fr-orb--c"></div>
<svg class="fr-sky" viewBox="0 0 1000 520" preserveAspectRatio="xMidYMid slice" aria-hidden="true"><line style="--i:0" x1="331" y1="88" x2="287" y2="218"/><line style="--i:1" x1="331" y1="88" x2="455" y2="161"/><line style="--i:2" x1="645" y1="50" x2="546" y2="45"/><line style="--i:3" x1="645" y1="50" x2="586" y2="143"/><line style="--i:4" x1="534" y1="194" x2="574" y2="209"/><line style="--i:5" x1="534" y1="194" x2="577" y2="238"/><line style="--i:6" x1="76" y1="264" x2="97" y2="235"/><line style="--i:7" x1="76" y1="264" x2="42" y2="241"/><line style="--i:8" x1="56" y1="227" x2="42" y2="241"/><line style="--i:0" x1="56" y1="227" x2="24" y2="220"/><line style="--i:1" x1="87" y1="59" x2="77" y2="115"/><line style="--i:2" x1="87" y1="59" x2="158" y2="72"/><line style="--i:3" x1="428" y1="421" x2="421" y2="387"/><line style="--i:4" x1="428" y1="421" x2="395" y2="443"/><line style="--i:5" x1="139" y1="124" x2="144" y2="136"/><line style="--i:6" x1="139" y1="124" x2="189" y2="128"/><line style="--i:7" x1="622" y1="480" x2="641" y2="503"/><line style="--i:8" x1="622" y1="480" x2="547" y2="449"/><line style="--i:0" x1="574" y1="209" x2="577" y2="238"/><line style="--i:1" x1="574" y1="209" x2="633" y2="197"/><line style="--i:2" x1="957" y1="37" x2="961" y2="72"/><line style="--i:3" x1="957" y1="37" x2="939" y2="88"/><line style="--i:4" x1="844" y1="156" x2="860" y2="168"/><line style="--i:5" x1="844" y1="156" x2="809" y2="154"/><line style="--i:6" x1="158" y1="72" x2="181" y2="72"/><line style="--i:7" x1="158" y1="72" x2="189" y2="128"/><line style="--i:8" x1="316" y1="416" x2="364" y2="449"/><line style="--i:0" x1="316" y1="416" x2="395" y2="443"/><line style="--i:1" x1="193" y1="300" x2="166" y2="255"/><line style="--i:2" x1="193" y1="300" x2="254" y2="297"/><line style="--i:3" x1="633" y1="197" x2="673" y2="224"/><line style="--i:4" x1="633" y1="197" x2="577" y2="238"/><line style="--i:5" x1="546" y1="45" x2="586" y2="143"/><line style="--i:6" x1="546" y1="45" x2="455" y2="161"/><line style="--i:7" x1="77" y1="115" x2="144" y2="136"/><line style="--i:8" x1="77" y1="115" x2="189" y2="128"/><line style="--i:0" x1="673" y1="224" x2="687" y2="306"/><line style="--i:1" x1="673" y1="224" x2="720" y2="156"/><line style="--i:2" x1="322" y1="302" x2="254" y2="297"/><line style="--i:3" x1="322" y1="302" x2="390" y2="343"/><line style="--i:4" x1="455" y1="161" x2="586" y2="143"/><line style="--i:5" x1="455" y1="161" x2="577" y2="238"/><line style="--i:6" x1="783" y1="358" x2="754" y2="296"/><line style="--i:7" x1="783" y1="358" x2="807" y2="439"/><line style="--i:8" x1="254" y1="297" x2="244" y2="253"/><line style="--i:0" x1="254" y1="297" x2="287" y2="218"/><line style="--i:1" x1="524" y1="445" x2="547" y2="449"/><line style="--i:2" x1="524" y1="445" x2="475" y2="341"/><line style="--i:3" x1="720" y1="156" x2="809" y2="154"/><line style="--i:4" x1="720" y1="156" x2="586" y2="143"/><line style="--i:5" x1="961" y1="72" x2="939" y2="88"/><line style="--i:6" x1="961" y1="72" x2="860" y2="168"/><line style="--i:7" x1="421" y1="387" x2="390" y2="343"/><line style="--i:8" x1="421" y1="387" x2="395" y2="443"/><line style="--i:0" x1="166" y1="255" x2="97" y2="235"/><line style="--i:1" x1="166" y1="255" x2="244" y2="253"/><line style="--i:2" x1="58" y1="343" x2="78" y2="359"/><line style="--i:3" x1="58" y1="343" x2="77" y2="392"/><line style="--i:4" x1="754" y1="296" x2="687" y2="306"/><line style="--i:5" x1="860" y1="168" x2="809" y2="154"/><line style="--i:6" x1="860" y1="168" x2="939" y2="88"/><line style="--i:7" x1="687" y1="306" x2="577" y2="238"/><line style="--i:8" x1="577" y1="238" x2="586" y2="143"/><line style="--i:0" x1="577" y1="238" x2="475" y2="341"/><line style="--i:1" x1="826" y1="479" x2="807" y2="439"/><line style="--i:2" x1="475" y1="341" x2="390" y2="343"/><line style="--i:3" x1="475" y1="341" x2="395" y2="443"/><line style="--i:4" x1="78" y1="359" x2="77" y2="392"/><line style="--i:5" x1="78" y1="359" x2="42" y2="241"/><line style="--i:6" x1="641" y1="503" x2="547" y2="449"/><line style="--i:7" x1="809" y1="154" x2="939" y2="88"/><line style="--i:8" x1="390" y1="343" x2="395" y2="443"/><line style="--i:0" x1="390" y1="343" x2="364" y2="449"/><line style="--i:1" x1="42" y1="241" x2="24" y2="220"/><line style="--i:2" x1="42" y1="241" x2="97" y2="235"/><line style="--i:3" x1="181" y1="72" x2="189" y2="128"/><line style="--i:4" x1="181" y1="72" x2="144" y2="136"/><line style="--i:5" x1="144" y1="136" x2="189" y2="128"/><line style="--i:6" x1="144" y1="136" x2="97" y2="235"/><line style="--i:7" x1="395" y1="443" x2="364" y2="449"/><line style="--i:8" x1="97" y1="235" x2="24" y2="220"/><line style="--i:0" x1="97" y1="235" x2="189" y2="128"/><line style="--i:1" x1="287" y1="218" x2="244" y2="253"/><line style="--i:2" x1="287" y1="218" x2="189" y2="128"/><line style="--i:3" x1="189" y1="128" x2="244" y2="253"/><circle style="--i:0" cx="331" cy="88" r="1.4"/><circle style="--i:1" cx="645" cy="50" r="1.9"/><circle style="--i:2" cx="534" cy="194" r="2.4"/><circle style="--i:3" cx="76" cy="264" r="2.9"/><circle style="--i:4" cx="56" cy="227" r="1.4"/><circle style="--i:5" cx="87" cy="59" r="1.9"/><circle style="--i:6" cx="428" cy="421" r="2.4"/><circle style="--i:7" cx="139" cy="124" r="2.9"/><circle style="--i:8" cx="622" cy="480" r="1.4"/><circle style="--i:9" cx="574" cy="209" r="1.9"/><circle style="--i:10" cx="957" cy="37" r="2.4"/><circle style="--i:0" cx="844" cy="156" r="2.9"/><circle style="--i:1" cx="158" cy="72" r="1.4"/><circle style="--i:2" cx="316" cy="416" r="1.9"/><circle style="--i:3" cx="193" cy="300" r="2.4"/><circle style="--i:4" cx="633" cy="197" r="2.9"/><circle style="--i:5" cx="546" cy="45" r="1.4"/><circle style="--i:6" cx="77" cy="115" r="1.9"/><circle style="--i:7" cx="673" cy="224" r="2.4"/><circle style="--i:8" cx="322" cy="302" r="2.9"/><circle style="--i:9" cx="455" cy="161" r="1.4"/><circle style="--i:10" cx="783" cy="358" r="1.9"/><circle style="--i:0" cx="254" cy="297" r="2.4"/><circle style="--i:1" cx="524" cy="445" r="2.9"/><circle style="--i:2" cx="720" cy="156" r="1.4"/><circle style="--i:3" cx="961" cy="72" r="1.9"/><circle style="--i:4" cx="421" cy="387" r="2.4"/><circle style="--i:5" cx="166" cy="255" r="2.9"/><circle style="--i:6" cx="58" cy="343" r="1.4"/><circle style="--i:7" cx="754" cy="296" r="1.9"/><circle style="--i:8" cx="860" cy="168" r="2.4"/><circle style="--i:9" cx="687" cy="306" r="2.9"/><circle style="--i:10" cx="577" cy="238" r="1.4"/><circle style="--i:0" cx="826" cy="479" r="1.9"/><circle style="--i:1" cx="475" cy="341" r="2.4"/><circle style="--i:2" cx="78" cy="359" r="2.9"/><circle style="--i:3" cx="641" cy="503" r="1.4"/><circle style="--i:4" cx="809" cy="154" r="1.9"/><circle style="--i:5" cx="390" cy="343" r="2.4"/><circle style="--i:6" cx="42" cy="241" r="2.9"/><circle style="--i:7" cx="181" cy="72" r="1.4"/><circle style="--i:8" cx="77" cy="392" r="1.9"/><circle style="--i:9" cx="144" cy="136" r="2.4"/><circle style="--i:10" cx="395" cy="443" r="2.9"/><circle style="--i:0" cx="97" cy="235" r="1.4"/><circle style="--i:1" cx="547" cy="449" r="1.9"/><circle style="--i:2" cx="807" cy="439" r="2.4"/><circle style="--i:3" cx="287" cy="218" r="2.9"/><circle style="--i:4" cx="364" cy="449" r="1.4"/><circle style="--i:5" cx="939" cy="88" r="1.9"/><circle style="--i:6" cx="189" cy="128" r="2.4"/><circle style="--i:7" cx="244" cy="253" r="2.9"/><circle style="--i:8" cx="586" cy="143" r="1.4"/><circle style="--i:9" cx="24" cy="220" r="1.9"/></svg>
<div class="fr-beam"></div>
<div class="fr-hero">
<div class="fr-mark">
<svg class="fr-rings" viewBox="0 0 220 220" aria-hidden="true"><circle cx="110" cy="110" r="100"/><circle cx="110" cy="110" r="82"/><circle cx="110" cy="110" r="64"/></svg>
<svg class="fr-logo" viewBox="0 0 128 128" role="img" aria-label="The fasterRag logo: three chunk bars streaming into a lightning bolt"><defs><linearGradient id="fr-tile" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4f46e5"/><stop offset="1" stop-color="#38bdf8"/></linearGradient></defs><rect width="128" height="128" rx="28" fill="url(#fr-tile)"/><g fill="#fff"><rect class="fr-bar" style="--i:0" x="18" y="40" width="34" height="9" rx="4.5"/><rect class="fr-bar" style="--i:1" x="12" y="59.5" width="30" height="9" rx="4.5"/><rect class="fr-bar" style="--i:2" x="20" y="79" width="22" height="9" rx="4.5"/><path class="fr-bolt" d="M84 14 52 68h20l-8 46 38-58H80z"/></g></svg>
</div>
<p class="fr-title">faster<span>Rag</span></p>
<p class="fr-lede">A backend-only RAG framework for FastAPI. Point it at your documents, ask questions, get answers with citations — from a REST API, a terminal or Python. Any vector database, embedding model or LLM, chosen in one config file.</p>
<div class="fr-facts">
<span><b>3</b> ways in: REST, CLI, Python</span>
<span><b>5</b> vector databases</span>
<span><b>4</b> embedding providers</span>
<span><b>4</b> LLM SDKs and any OpenAI-compatible URL</span>
<span><b>7</b> document formats</span>
</div>
</div>
</div>

[Quickstart](quickstart.md){ .md-button .md-button--primary }
[Python package](python.md){ .md-button }
[REST API](rest-api.md){ .md-button }
[GitHub](https://github.com/balrajsingh165/fasterrag){ .md-button }

## Install

```bash
pip install "fasterrag[all]>=0.1.0b1"
```

Python 3.12 or newer. The `>=0.1.0b1` picks the current beta; don't use `--pre`, which also
pulls pre-release versions of every dependency. See [Installation extras](quickstart.md#install)
to install only what you need.

## Example

```python
import asyncio
from fasterrag import FasterRag


async def main() -> None:
    async with FasterRag.from_config("config.yaml") as rag:
        await rag.ingest(["./my-docs/"], recursive=True)
        result = await rag.query("What are the payment terms?")
        print(result.answer)
        for c in result.citations:
            print(c.source, c.page)


asyncio.run(main())
```

The same thing from a terminal:

```bash
fasterrag config init
fasterrag ingest ./my-docs/ --recursive
fasterrag query "What are the payment terms?"
```

## What you get

- **Hybrid retrieval.** Vector search and keyword (BM25) search run together and are fused,
  with optional cross-encoder reranking.
- **Answers with citations**, streamed or whole, each with a trace id you can look up later.
- **Any backend.** Qdrant, pgvector, Chroma, Milvus or Weaviate; Hugging Face, OpenAI, Cohere or
  Ollama embeddings; OpenAI, Anthropic, Cohere, Ollama or any OpenAI-compatible LLM. Switching is
  a config change. See [Providers](providers.md).
- **Safe re-runs.** Re-ingesting unchanged files does nothing; a file that fails is set aside
  without failing the rest of the job.
- **Clear errors.** Every API error is a JSON problem document with a stable `code`.

## What's new

**0.1.0 beta (`0.1.0b1`) — 13 November 2026.** The first published release:
`pip install "fasterrag>=0.1.0b1"`. It is a beta, so interfaces may change before 1.0, and it
makes no performance claim: no benchmark figure is citable yet.

What changed for you in this release:

- **Smaller core install.** `pip install "fasterrag>=0.1.0b1"` is the library and CLI only. The
  web server and the Qdrant, PDF, DOCX and HTML support are extras: `server`, `qdrant`, `pdf`,
  `docx`, `html` (all included in `all`).
- **`fasterrag config init --profile local`** writes a config that needs no API key and no
  running service.
- **Reranking is off by default.** Set `retrieval.rerank: true` to turn the cross-encoder back on;
  it downloads a 2 GB model and is slow on a CPU.
- **Safer by default.** An unauthenticated server refuses to bind to a non-loopback address unless
  you set `security.allow_unauthenticated: true`; API keys must be at least 16 characters.
- **Anthropic works with current Claude models.** `llm.temperature` defaults to `null`, so no
  sampling parameter is sent to Claude unless you set one.
- **Removed:** the `fasterrag worker` command and the `app.workers` and
  `reliability.degradation_ladder` settings.
- **Deprecated:** the `idempotency_key` field in `POST /v1/ingest`. Send the `Idempotency-Key`
  header instead; the field is removed in 1.0.

The full history is in the
[changelog](https://github.com/balrajsingh165/fasterrag/blob/main/docs/CHANGELOG.md).
