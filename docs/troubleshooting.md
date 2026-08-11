# troubleshooting.md — Symptom → Cause → Fix

The user-facing inverse of [failure-modes.md](failure-modes.md) (which is engineering-facing: failure mode → detection → mitigation → proving test). Start here when something looks wrong.

**Before anything else, run `fasterrag doctor`.** It checks Docker, ports, disk, RAM/GPU, backend reachability, key validity, and config validity, and prints a concrete fix for every failure. A large share of the table below is caught by it in one command. `fasterrag doctor --fix` then applies the repairs that are safe to apply — the missing named storage volume, and fasterRag's own stopped container — and re-checks; anything it will not touch (a held port, full disk, missing secret) is named with the reason and the command to run yourself.

**Three things to grab before investigating anything:** the `trace_id` (on every response and every problem document), the problem `code`, and `fasterrag status` output. Those three make the difference between diagnosis and guesswork.

---

## Startup & configuration

| Symptom | Likely cause | Fix |
|---|---|---|
| Process exits at startup naming a config key | Fail-fast validation rejected the config — by design, a misconfigured process never serves | Fix the named key; `fasterrag config validate` before restarting. Rules in [config-reference.md](config-reference.md) |
| Exits naming an environment variable | A `*_env` key references a variable missing from the environment/`.env` (the value is never logged, only the name) | Add it to `.env`; copy [`.env.example`](../.env.example) if you don't have one |
| `top_k must be ≤ rerank_top_n` and similar | A cross-field validation rule was violated | See the cross-field rules list at the end of [config-reference.md](config-reference.md) |
| Config change appears to do nothing | Config is read at startup/provision time, not live (deliberate — it keeps the control plane auditable) | Restart the process, or re-run the relevant `fasterrag provision` command |

## Vector database

| Symptom | Likely cause | Fix |
|---|---|---|
| `/readyz` returns 503 listing `vector_db` | Backend unreachable; the process starts anyway and serves degraded rather than refusing to boot | Start the backend; the circuit breaker's half-open probe reconnects automatically. Check `fasterrag status` for breaker state |
| Works locally, fails from another machine | **Only port 6333 exposed.** The client attempts gRPC on 6334 ([references.md](references.md) R5) | Expose **both 6333 and 6334**, or set `vector_db.prefer_grpc: false` explicitly |
| Auth errors against Qdrant | Key mismatch between client and server | Server needs `QDRANT__SERVICE__API_KEY`; client reads the variable named by `vector_db.api_key_env`. They must match |
| Data vanished after a container restart (Windows/WSL) | Bind-mount storage — a known Qdrant data-loss mode on Windows/WSL ([references.md](references.md) R6) | Use a **named Docker volume** (`vector_db.docker.volume`). The config loader rejects bind-mount paths on Windows/WSL for exactly this reason |
| Upsert rejected for dimension mismatch | The collection's vector size ≠ the configured embedding model's output | Either restore the original model or re-embed into a new collection: `fasterrag index reembed` (blue/green, zero downtime) |

## Ingestion

| Symptom | Likely cause | Fix |
|---|---|---|
| `POST /v1/ingest` returns 429 `QUEUE_FULL` | Backpressure — bounded queues are full, which protects memory and the live query path | Retry after `Retry-After`; raise `workers.queue_depth`, or add embedding workers if the pool is the bottleneck |
| Job finishes `partial` | Some documents dead-lettered | `GET /v1/ingest/{job}/documents?status=dead_lettered` — each entry names a `reason_code`. Fix sources, then `POST /v1/ingest/{job}/retry-dlq` |
| Re-ingesting adds nothing | Working as designed: content-hash dedup makes replays no-ops (D3) | To force reprocessing, the source content must actually change (or use `index reembed` for model changes) |
| Ingest restarted from the beginning after a crash | Journal disabled | Set `ingestion.journal.enabled: true`; checkpoints every `checkpoint_every` documents let a crash resume exactly where it stopped |
| Scanned PDFs index with no useful text | OCR yielded nothing; the chunk carries a `low_text_yield` parse flag rather than silently indexing emptiness | Check the flag in document metadata; verify the OCR path is available for that source type. If the pages carry a thin text layer (a stamped header, a page number) the OCR path never ran: raise `parsing.minimum_chars_per_page` above what such a page yields, and `parsing.ocr_resolution` if OCR output is garbled on fine print |
| Ingestion is slow, GPU underutilized | CPU parse/chunk stage is the bottleneck, so embedding workers idle | Raise `workers.cpu_pool_size`; the streaming hand-off is designed so embedding never waits on parsing |
| Ingestion slow, CPU idle | Embedding is the bottleneck (or a provider is throttling) | Raise `workers.embedding_pool_size` / `embeddings.batch_size`; check for retry-storm logs indicating provider 429s |

## Retrieval quality

| Symptom | Likely cause | Fix |
|---|---|---|
| Exact identifiers/rare terms not found | Dense-only retrieval — vectors are weak on exact tokens | Set `retrieval.hybrid: true`; the BM25 leg exists for exactly this |
| Right document retrieved, wrong passage | Chunks too large, or boundaries cutting through the answer | Move `chunking.chunk_size` toward 512–1024; try `strategy: layout` or `semantic`. Anything near ~2,500 tokens is past the documented context cliff ([references.md](references.md) R3) |
| Retrieval finds it, but the answer misses it | Reranking off, or `top_k` truncating before the good chunk | Enable `retrieval.rerank` — expected to help most here, though the gain is unmeasured; raise `rerank_top_n` so the reranker sees more candidates |
| Chunks lack context to be understood alone | Pronouns/shorthand referencing surrounding text | Enable `chunking.contextual_enrichment` and reindex ([prompts.md](prompts.md) P2; evidence in [references.md](references.md) R1) |
| Quality dropped after a change | A retrieval-affecting change shipped ungated | Enable `eval.regression_gate: true`; use `fasterrag replay --trace <id> --config candidate.yaml` to see exactly what changed (D8) |
| No idea which knob to turn | Guesswork is the actual problem | `fasterrag autopilot run` — measured suggestions from your own corpus, never auto-applied (D6) |

## Answers & generation

| Symptom | Likely cause | Fix |
|---|---|---|
| `INSUFFICIENT_EVIDENCE` instead of an answer | Grounded-or-refuse declined below the faithfulness threshold — a feature, not a failure (D5) | Inspect `best_candidates`; if refusals are too aggressive, lower `generation.faithfulness_threshold`, or improve retrieval so real evidence reaches the context |
| Answers look plausible but unsupported | Grounding not enforced | Enable `generation.grounded_or_refuse: true`; check the `faithfulness` field and `unsupported_claims` in the trace |
| Response labelled `degraded: true` | The degradation ladder engaged — a component is down and you're being told, rather than silently served worse results (D4) | Read `mode`: `hybrid_only` = reranker down · `cache_only` = vector DB down · `extractive` = LLM down. Fix the named component; recovery is automatic |
| SSE stream ends with no `done` event | Mid-stream failure; the answer is incomplete | Treat as incomplete (never as final); an `error` event carries the problem document and `trace_id` |
| Citations missing | `generation.citations` disabled | Re-enable it. It cannot be off while `grounded_or_refuse` is on — the config loader enforces this |

## Cache

| Symptom | Likely cause | Fix |
|---|---|---|
| Stale answer after updating documents | Should be impossible — corpus changes trigger event-driven invalidation | If reproducible, this is a bug: file it with the `trace_id` and the ingest `job_id` (FMEA row 24) |
| A wrong cached answer served for a similar-but-different question | `cache.similarity_threshold` too permissive | Raise it (valid 0.90–0.99; default 0.95; useful band ~0.92–0.97) |
| Cache hit rate near zero | Genuinely diverse queries, or TTL too short | Check `fasterrag_cache_events_total`; raise `cache.ttl`. Low hit rate is not itself a defect |

## Provisioning & observability

| Symptom | Likely cause | Fix |
|---|---|---|
| Provisioning refuses to run | Doctor gate failed — by design, nothing is mutated while preconditions are broken (D10) | Run `fasterrag doctor`; every failed check prints its fix |
| `vector_db_volume` fails: the named volume does not exist | Nothing has created it yet, or `vector_db.docker.volume` was renamed — the danger is that this looks fine until the container is replaced, and then the index is gone | `fasterrag doctor --fix` creates it (safe and idempotent), or `docker volume create <name>`. `fasterrag provision qdrant` creates it too. Check the name matches the volume your data is actually on before creating a second, empty one |
| `doctor --fix` says `needs human` on a stopped container | The container of that name does not carry `fasterrag.managed=true`, so fasterRag did not create it and will not start somebody else's service | Start it yourself, or remove it and let `fasterrag provision qdrant` create a managed one |
| `PROVISIONING_FAILED` naming a container | One stack component came up unhealthy; the stack is left inspectable and no URL is returned | Follow the hint, then re-run — provisioning is idempotent and converges |
| Langfuse logins/API keys stopped working after a restart | `SALT` / `ENCRYPTION_KEY` / `NEXTAUTH_SECRET` changed ([references.md](references.md) R9) | Restore the original values from your secret store. The provisioner never regenerates them; if they were deleted manually, restore rather than re-create |
| Langfuse headless bootstrap ignored | `LANGFUSE_INIT_*` values double-quoted in compose, or org vars missing (project/user require an org — [references.md](references.md) R10) | Remove the quotes; ensure `LANGFUSE_INIT_ORG_ID`/`_ORG_NAME` are set |
| Grafana edits vanish | Working as designed: provisioned resources are read-only (`editable: false`, `allowUiUpdates: false`) to enforce GitOps | Edit the provisioning YAML/JSON files; they reload on the poll interval without a restart |
| Dashboard down | Observability-only — RAG serving is entirely unaffected | Restart the dashboard container; no query impact (FMEA row 36) |

## Performance

| Symptom | Likely cause | Fix |
|---|---|---|
| p95 latency high, retrieval fine | Generation dominates (provider-bound) | Check the `timings_ms` split; retrieval and generation are measured separately for exactly this reason |
| Rerank stage dominates latency | `rerank_top_n` too high for your reranker/hardware | Lower it, or use a smaller cross-encoder. fasterRag publishes no expected figure for the stage — read your own `fasterrag_stage_duration_seconds` |
| Slow first query after startup | Cold caches and model load | Expected. Benchmarks report cold and warm separately ([performance.md](performance.md)); pre-warm if TTFT matters |
| Memory grows over a long run | Should not happen — the soak test asserts no memory/fd growth | File a bug with duration, RSS trend, and `fasterrag status`; unbounded growth is a defect, not tuning |
| Your numbers disagree with a doc | Docs contain **goals**, not measurements, until a ledger entry exists | Check [benchmarks.md](benchmarks.md). An unmeasured claim presented as fact is itself a bug — file it |

## Still stuck

1. `fasterrag doctor --json` and `fasterrag status` — capture both.
2. Pull the trace: `GET /v1/traces/{trace_id}` (or `fasterrag replay --trace <id>`) — it holds the retrieved set, all leg scores, the exact prompt, and the exact response.
3. Open an issue using the bug template. Include the `code`, `trace_id`, config sections involved, and versions. **Never paste `.env` contents.** Suspected security issue? Use private reporting instead ([.github/SECURITY.md](../.github/SECURITY.md)).
