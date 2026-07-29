# quickstart.md — From Zero to Answered Query

> **This is the specified beta contract, not a working tutorial yet.** No implementation code exists ([todo.md](todo.md) tracks the build slices). Every command below is the documented intended behavior; treat it as the acceptance criteria the build must satisfy, and expect it to work only once the corresponding slice ships.

Three paths, same engine and same `config.yaml`: [CLI](#path-a--cli) · [Python](#path-b--python-package) · [REST](#path-c--rest-service). Pick one; they interoperate.

## Prerequisites

| Requirement | Why | Checked by |
|---|---|---|
| Python 3.12+ | Runtime | `fasterrag doctor` |
| Docker (optional) | Only for `vector_db.mode: docker` and auto-provisioned tools | `fasterrag doctor` |
| ~4 GB free RAM, SSD space for the index | Embedding + vector storage | `fasterrag doctor` |
| Provider API key (optional) | Only if you choose a hosted embedding/LLM provider; the default stack is fully local | startup validation |

## Step 0 — Install

```bash
pip install fasterrag                 # core: local HuggingFace embeddings + Qdrant adapter
pip install "fasterrag[openai]"       # add a hosted provider
pip install "fasterrag[all]"          # everything
```

## Step 1 — Create `config.yaml` and `.env`

Copy the canonical default [`config.yaml`](../config.yaml) into your project (it is the complete schema with every default; full annotation in [config-reference.md](config-reference.md)), then copy [`.env.example`](../.env.example) to `.env`.

**The split is non-negotiable**: `config.yaml` holds all behavior and is safe to commit; `.env` holds only secrets and is git-ignored. Config never contains a secret value — it names the environment variable instead ([ADR-0003](adr/ADR-0003-config-yaml-env-split.md)).

A fully local starting point (no API keys, nothing leaves your machine):

```yaml
vector_db:
  provider: qdrant
  mode: docker
embeddings:
  provider: huggingface          # local sentence-transformers
  model: BAAI/bge-small-en-v1.5
llm:
  provider: ollama               # local generation
  model: llama3.1
  base_url: http://localhost:11434
```

Validate before doing anything else:

```bash
fasterrag config validate         # exits 2 and names the offending key if invalid
```

## Step 2 — Preflight with `doctor`

```bash
fasterrag doctor
```

`doctor` checks Docker, free ports (including **both 6333 and 6334** for Qdrant), disk, RAM/GPU, backend reachability, key validity, and config validity. **Every failed check prints a concrete fix.** It must pass before any provisioning runs — this is what makes one-toggle provisioning survivable on an arbitrary machine (D10). Try `--fix` for safe automatic remedies.

## Step 3 — Bring up the vector database

```bash
fasterrag provision qdrant        # system-managed container, named volume, both ports exposed
```

Already run Qdrant yourself, or have it on another machine? Skip this and point config at it instead:

```yaml
vector_db:
  mode: external
  host: 10.0.0.42        # remote machine, or localhost for a local non-Docker instance
  api_key_env: QDRANT_API_KEY
```

Remote deployments must expose **6333 (REST) and 6334 (gRPC)** — exposing only 6333 breaks clients that attempt gRPC ([references.md](references.md) R5).

## Step 4 — Know the cost before you spend it

```bash
fasterrag estimate ./my-docs/ --all-providers
```

Reports document/token counts, projected embedding cost per provider, and projected wall-clock time under your current worker settings — **before** ingestion (D9). With the local default stack the monetary cost is zero; the time estimate still matters.

## Step 5 — Ingest

```bash
fasterrag ingest ./my-docs/ --recursive --watch
```

Returns a `job_id` immediately (the API never blocks on parsing/embedding) and `--watch` follows per-stage progress. Under the hood: CPU workers parse and chunk, streaming into stateful embedding workers that hold the model in memory, feeding a batch indexer ([flow.md](flow.md)).

Ingestion is **checkpointed and idempotent** (D3): kill it mid-run and re-run — it resumes from the last checkpoint, and content-hash dedup makes replayed documents no-ops. Failed documents land in a dead-letter queue with a machine-readable reason:

```bash
fasterrag ingest --help                                    # all flags
# inspect failures for a job, then retry them:
curl "$API/v1/ingest/$JOB/documents?status=dead_lettered"
curl -X POST "$API/v1/ingest/$JOB/retry-dlq"
```

## Step 6 — Query

```bash
fasterrag query "What does the vendor agreement say about termination?"
```

The default pipeline runs dense ANN **and** BM25 in parallel, fuses with RRF (k=60), reranks the fused candidates with a cross-encoder, assembles a token-budgeted context with citations, and streams the answer ([ADR-0004](adr/ADR-0004-hybrid-search-plus-reranking.md)).

Debugging what came back:

```bash
fasterrag query "..." --show-chunks --show-timings   # retrieved chunks + per-stage latency
fasterrag query "..." --filter department=legal      # metadata filter, pushed into both legs
```

Every response carries `degraded` and `mode`. If a component is down you still get an answer, explicitly labelled (`hybrid_only`, `cache_only`, `extractive`) — never a silent quality drop (D4).

## Step 7 — Look at what happened

```bash
fasterrag status                     # queues, DLQ depth, breaker states, cache hit rates
```

For visual inspection, turn on observability. Each toggle auto-installs and configures its stack and returns a running URL, with **zero code changes**:

```yaml
observability:
  dashboard: true      # read-only inspection UI on :8080
  langfuse: true       # self-hosted Langfuse v3 → http://<host>:3000
```

```bash
fasterrag provision langfuse         # doctor-gated, idempotent
```

The dashboard shows cache stats, token usage, cost per query, p50/p95 latency split by stage, and the complete LLM input/output history. It is **observability-only** — it cannot control the RAG ([ADR-0005](adr/ADR-0005-api-cli-only-control-plane.md)).

## Step 8 — Tune with evidence, not guesswork

```bash
fasterrag autopilot run --budget-minutes 30
```

Autopilot builds a golden Q&A set from *your* corpus, searches chunk size / top_k / hybrid weights / rerank settings against it, and writes a **suggested config diff with measured deltas** (e.g. recall@10 before → after). It never edits your config — you review and apply (D6).

Once you have a golden set, lock quality in:

```yaml
eval:
  regression_gate: true    # blocks any change that degrades recall/nDCG beyond tolerance
```

## Path A — CLI

Steps 0–8 above. Full command and flag reference: [cli-reference.md](cli-reference.md).

## Path B — Python package

```python
import asyncio
from fasterrag import FasterRag

async def main() -> None:
    async with FasterRag.from_config("config.yaml") as rag:
        job = await rag.ingest(["./my-docs/"])
        await job.wait()

        result = await rag.query("What are the payment terms?")
        print(result.answer)
        for c in result.citations:
            print(f"  {c.source} p.{c.page} [{c.span.start}:{c.span.end}] score={c.score}")

asyncio.run(main())
```

Not writing async code? `from fasterrag.sync import FasterRag` gives the same API in a blocking facade. Want retrieval only, and your own LLM step? `await rag.retrieve(text, top_k=10)`. Want just one component? The chunkers, fusion, and evaluator are importable standalone. See [python-api.md](python-api.md).

## Path C — REST service

```bash
fasterrag serve                      # API on :8000
fasterrag worker                     # pipeline pools (separate process for heavy ingestion)
```

```bash
curl -X POST localhost:8000/v1/ingest \
  -H 'Content-Type: application/json' \
  -d '{"sources":[{"type":"path","value":"/data/docs"}]}'

curl -N -X POST localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"What are the payment terms?","stream":true}'
```

Streaming responses are SSE with `meta` → `token`* → `citations` → `usage` → `done`; a missing `done` means an incomplete answer. Errors are RFC 9457 problem documents with a stable `code` and a `trace_id`. Full contract: [api-reference.md](api-reference.md).

## Where to go next

| You want to | Read |
|---|---|
| Understand why the pipeline is shaped this way | [architecture.md](architecture.md) |
| Tune quality (chunking, hybrid weights, reranking) | [config-reference.md](config-reference.md), [architecture.md](architecture.md) §5–6 |
| Swap the vector DB or provider | [integrations.md](integrations.md) |
| Run it in production | [deployment.md](deployment.md), [security.md](security.md), [slo.md](slo.md) |
| Know what makes this different | [differentiators.md](differentiators.md) |
| Understand a term exactly | [glossary.md](glossary.md) |
