# how-to-use.md — Using fasterRag (Post-Release User Guide)

The end-user manual for the released **`fasterrag`** package. It assumes `pip install fasterrag` works — i.e. a version per [release.md](release.md) has been published. *(Until the first release lands, substitute `pip install -e ".[all]"` from a source checkout; everything else reads the same. This banner is deleted by the release that makes it false.)*

Companion docs: [quickstart.md](quickstart.md) is the narrated first walkthrough; [cookbook.md](cookbook.md) has ready-made configs per situation; [troubleshooting.md](troubleshooting.md) is symptom→fix; [config-reference.md](config-reference.md) documents every key. This file is the task-oriented manual you keep open while operating.

## 1. Install

Requires **Python 3.12+** (Linux and Windows are both CI-tested). Pick extras for what you actually run — the core install is deliberately lean:

| Install | You get |
|---|---|
| `pip install fasterrag` | Engine, all document parsers, chunking, Qdrant + pgvector adapters, CLI, REST server |
| `pip install "fasterrag[huggingface]"` | + local embeddings (sentence-transformers; multi-GB runtime — the fully-local starting point) |
| `[rerank]` | + cross-encoder reranking (note: the default model is large; see §8 first-run) |
| `[openai]` / `[anthropic]` / `[cohere]` / `[ollama]` | + that hosted/local provider |
| `[redis]` | + the shared cache backend (multi-replica deployments) |
| `[ocr]` | + scanned-PDF OCR (also needs the `tesseract` binary on the host) |
| `[otel]` | + OTLP export |
| `[all]` | everything above |

Selecting a provider whose extra is missing fails at startup with a `ConfigError` naming the exact install command — never an import traceback.

## 2. First five minutes

```bash
fasterrag config init          # writes the canonical config.yaml + .env template beside it
# edit config.yaml (behavior) and .env (secrets only — config refers to them by NAME)
fasterrag config validate      # fail-fast: names the offending key, exit 2
fasterrag doctor               # preflight; every failure prints its fix
fasterrag provision qdrant     # system-managed container (or point config at your own)
fasterrag estimate ./my-docs --all-providers   # tokens + cost BEFORE committing
fasterrag ingest ./my-docs --recursive
fasterrag query "What does the vendor agreement say about termination?"
```

Two files rule everything: `config.yaml` (all behavior, committable, never a secret) and `.env` (secrets only). Re-running an ingest is safe — content-hash dedup makes it a no-op.

## 3. The three ways to drive it (same engine, same errors)

**CLI** — operations and scripts. Everything above, plus: `serve`, `status`, `index list|reembed|rollback|lock verify`, `replay`, `backup`/`restore`, `export`/`import`, `benchmark`, `autopilot run`, `traces list|show`, `provision langfuse|grafana`, `config show`, and `--set section.key=value` to override any setting per invocation. Machine output via `--json`. Full reference: [cli-reference.md](cli-reference.md).

**REST** — `fasterrag serve` (default `:8000`), everything under `/v1`. Streaming answers are SSE (`meta → token… → citations → usage → done`; treat a missing `done` as incomplete). Every error is an RFC 9457 problem with a stable `code` and a `trace_id`. Full contract: [api-reference.md](api-reference.md).

```bash
curl -X POST :8000/v1/ingest -H 'content-type: application/json' \
  -d '{"sources":[{"type":"url","value":"https://example.com/spec.pdf"}]}'
curl -N -X POST :8000/v1/query -d '{"query":"What are the payment terms?","stream":true}'
```

**Python** — embed it in your app:

```python
import asyncio
from fasterrag import FasterRag           # blocking twin: from fasterrag.sync import FasterRag

async def main() -> None:
    async with FasterRag.from_config("config.yaml") as rag:
        job = await rag.ingest(["./my-docs/"])           # awaits completion, returns settled job
        result = await rag.query("What are the payment terms?")
        print(result.answer, [c.source for c in result.citations])
        chunks = await rag.retrieve("payment terms", top_k=20)   # retrieval only — bring your own LLM

if __name__ == "__main__":                # REQUIRED: parsing uses spawned worker processes
    asyncio.run(main())
```

Piecemeal use works without the engine: `from fasterrag.chunking import RecursiveChunker`, `from fasterrag.retrieval import rrf_fuse`, `from fasterrag.evals import evaluate`. Full surface + plugin contract: [python-api.md](python-api.md).

## 4. The settings people actually tune

| Goal | Keys | Notes |
|---|---|---|
| Answer quality | `retrieval.rerank`, `rerank_top_n`, `chunking.strategy`, `chunk_size` (512–1024 working range), `chunking.contextual_enrichment` | Enrichment costs LLM calls at ingest — `estimate` prices it first. Changing chunking/embedding needs a reindex (§6) |
| Honesty | `generation.grounded_or_refuse`, `faithfulness_threshold` | Below threshold you get structured `insufficient_evidence`, not a guess |
| Latency | `rerank_top_n` (the main dial), `llm.streaming`, `cache.semantic` + `cache.backend` | `disk` persists across CLI runs; `redis` is the multi-replica backend |
| Cost | `embeddings.provider` (local = zero token cost), `embeddings.tiering`, `cache.semantic` | The runtime budget keys refuse startup until the governor ships (TASK-0242) — the estimator is the shipped cost tool |
| Scale | `workers.*`, `vector_db.collection.shard_number`/`replication_factor`, remote `vector_db.host` | Both Qdrant ports (6333 **and** 6334) must be reachable remotely |

Ready-made profiles (air-gapped, max-accuracy, cost-optimized, multi-tenant, retrieval-only, CI-gated…): [cookbook.md](cookbook.md).

## 5. Operating it

**Watch it**: `fasterrag status` (queues, DLQ, breaker states, cache hit rates) · `observability.dashboard: true` → read-only UI on `:8080` (authenticated, tenant-scoped — internal networks only, it displays prompts) · `fasterrag provision langfuse` → full trace UI at `:3000` · `provision grafana` → dashboards over the metrics endpoint · `[otel]` + `observability.otel: true` → your collector.

**Ingestion at scale**: jobs are journaled and checkpointed — a crash resumes, never restarts; failures land in the DLQ with machine-readable reasons (`GET /v1/ingest/{job}/documents?status=dead_lettered`, then `retry-dlq`). Run `fasterrag serve` and heavy ingestion in separate processes; queues are bulkheaded so ingest storms cannot starve queries.

**Protect it**: `security.auth: true` (+ `FASTERRAG_API_KEY` in `.env`) turns on key auth with scopes and per-key rate limiting; `security.multi_tenancy: true` isolates collections, caches, traces, and budgets per tenant via `X-Tenant-ID`. Details and threat notes: [security.md](security.md).

**Change it safely**: model or chunking changes go through `fasterrag index reembed` — blue/green, eval-gated (pass `--dataset` with your golden set), atomic swap, instant `index rollback` within the retention window. `index lock verify` tells you when the live config has drifted from what built the index. `fasterrag replay --trace <id> --config candidate.yaml` shows exactly what a config change does to a past query before you adopt it.

**Tune with evidence**: `fasterrag autopilot generate-golden-set` from your corpus (review it — promote records to `source: "human"`), then `autopilot run` for suggested settings with measured deltas; it never edits your config. Lock quality in with `eval.regression_gate: true`.

**Back it up**: `fasterrag backup` writes timestamped sets (collections + lockfile + journal + traces + config; **never** `.env`) with retention pruning; `restore` verifies manifests and counts before touching anything. Full procedure and drill: [disaster-recovery.md](disaster-recovery.md).

**Leave, if you want**: `fasterrag export --out corpus.fragx` / `import` moves documents, chunks, metadata, and optionally vectors between deployments and backends ([archive-format.md](archive-format.md)) — anti-lock-in is a feature, not a promise.

## 6. Upgrading

SemVer: patch/minor are drop-in; the public Python surface, config keys, error codes, and archive format are compatibility contracts. Things that require a **reindex** (via `reembed`, zero-downtime): changing the embedding model/version, chunking strategy/size/overlap, or enrichment; `index lock verify` will tell you. Read the release's CHANGELOG section before upgrading — anything needing action says so there.

## 7. When something is wrong

Grab three things: the `trace_id`, the problem `code`, and `fasterrag status`. Then [troubleshooting.md](troubleshooting.md) (symptom→cause→fix). Degraded answers announce themselves (`degraded: true` + `mode`) — never silently. Bugs: GitHub issues with the template; security problems: private reporting per [SECURITY.md](../.github/SECURITY.md) — never a public issue.

## 8. Known sharp edges (current release line)

First run with `[rerank]`/`[huggingface]` downloads multi-GB models — the first query is minutes, not seconds, and says so rather than hanging silently. The rate limiter counts per replica (TASK-0216). One replica per deployment is the tested shape for ingestion (TASK-0130). Performance numbers: none are published yet by policy — the [ledger](benchmarks.md) is the only place they will ever appear.
