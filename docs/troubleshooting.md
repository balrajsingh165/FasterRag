# troubleshooting.md — Symptom → Cause → Fix

The user-facing inverse of [failure-modes.md](failure-modes.md) (which is engineering-facing: failure mode → detection → mitigation → proving test). Start here when something looks wrong.

**Before anything else, run `fasterrag doctor`.** It checks Docker, ports, disk, RAM/GPU, backend reachability, key validity, and config validity, and prints a concrete fix for every failure. A large share of the table below is caught by it in one command. `fasterrag doctor --fix` then applies the repairs that are safe to apply — the missing named storage volume, and fasterRag's own stopped container — and re-checks; anything it will not touch (a held port, full disk, missing secret) is named with the reason and the command to run yourself.

**Three things to grab before investigating anything:** the `trace_id` (on every response and every problem document), the problem `code`, and `fasterrag status` output. Those three make the difference between diagnosis and guesswork.

---

## Startup & configuration

| Symptom | Likely cause | Fix |
|---|---|---|
| Process exits at startup naming a config key | Fail-fast validation rejected the config — by design, a misconfigured process never serves | Fix the named key; `fasterrag config validate` before restarting. Rules in [config-reference.md](config-reference.md) |
| Exits naming an environment variable | A `*_env` key references a variable missing from the environment/`.env` (the value is never logged, only the name) | Add it to `.env`; copy [`.env.example`](https://github.com/balrajsingh165/fasterrag/blob/main/.env.example) if you don't have one |
| `top_k must be ≤ rerank_top_n` and similar | A cross-field validation rule was violated | See the cross-field rules list at the end of [config-reference.md](config-reference.md) |
| Config change appears to do nothing | Config is read at startup/provision time, not live (deliberate — it keeps the control plane auditable) | Restart the process, or re-run the relevant `fasterrag provision` command |
| `security.multi_tenancy requires security.auth` at load | Tenancy was turned on with auth off. The tenant of a request is resolved by the authentication middleware, so this combination used to load and enforce nothing (TASK-0460) | Set `security.auth: true` and give every key a tenant (`secret:scopes:tenant`), or set `security.multi_tenancy: false` |

### `fasterrag serve` exits with `refusing to bind … with security.auth off`

`app.host` (or `--host`), or `observability.dashboard_host` with the dashboard on, names an address other machines can reach while `security.auth` is `false`. Anyone who can reach that port could ingest files the process can read and query them back, so since TASK-0460 (2026-11-12) this is a refusal with exit code 2, not the warning it used to be. Three fixes, in order of preference:

1. Turn auth on: `security.auth: true`, a key of at least 16 characters in `FASTERRAG_API_KEY`, and `ingestion.allowed_source_roots` ([security.md](security.md) §2).
2. Bind loopback: keep the default `127.0.0.1`.
3. Opt in with `security.allow_unauthenticated: true` — only when a reverse proxy that authenticates stands in front, or the port is published on the host's loopback, as in the shipped compose files. `serve` then starts and prints a `WARNING`.

**A bare `docker run fasterrag` hits this**: the image runs `serve --host 0.0.0.0` with the shipped config, which has auth off. Pass `-e FASTERRAG_SET=security.allow_unauthenticated=true` together with `-p 127.0.0.1:8000:8000`, or mount a config with auth on ([deployment.md](deployment.md) §3).

### `403 TENANT_FORBIDDEN`: `this key is bound to no tenant and does not hold the 'cross-tenant' scope`

With `security.multi_tenancy` on, a key whose entry has no tenant field used to act for any tenant it named; since TASK-0460 (2026-11-12) it is refused on every request, and startup warns about it by its positional name (`key-2`). Either bind it — `secret:query:acme` — or, for an operator key that genuinely must act for every tenant, add the scope by name: `secret:query,cross-tenant`, or `secret:*,cross-tenant` for every scope. `*` and `admin` do not imply it.

### A key stopped working with `401 AUTH_INVALID` and nothing changed

Look for `an expired api key was presented` in the log: the key's entry carries an `expires=` that has passed (TASK-0492). The refusal is deliberately identical to an unknown key's. Issue a new key, or move the `expires=` and — if it lives in `security.api_key_file` — save the file; the server re-reads it within a second, or on `SIGHUP` on POSIX. A change to `FASTERRAG_API_KEY` itself needs a restart.

### I edited the key file and the old keys still work

The server logged `the api key file changed but could not be loaded; the keys already loaded stay in force` with the reason: a malformed entry, a key under 16 characters, or a `sha256:` value that is not 64 hex digits. Fix the file and save it again. Keys are only replaced by a file that loads in full.

### Startup refuses an API key: `is the placeholder shipped in .env.example` or `is shorter than 16 characters`

With `security.auth: true`, every key in `FASTERRAG_API_KEY` is checked at startup (TASK-0460). `change-me` — the value `.env.example` ships — is refused by name, and any key under 16 characters is refused as guessable. The message names the entry by its 1-based position and never prints the key. Generate one with `python -c "import secrets; print(secrets.token_urlsafe(32))"`.

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
| A document is dead-lettered with `PARSE_TIMEOUT` | Its parse was still running after `ingestion.parse_timeout_seconds` (default 600). Either the parser hung on it — a pathological PDF, a pattern that backtracks without end — or the document is legitimately slower than the limit, which OCR of a long scan can be. The worker pool was recycled to get the worker back, so the log carries `a document outlived its parse timeout; the cpu worker pool was recycled` with the document id | If the document is legitimate, raise `ingestion.parse_timeout_seconds` above what it needs and `POST /v1/ingest/{job}/retry-dlq`. If the parser hangs on it, the document has to be fixed or left out: retrying unchanged runs out the same limit. Every other document of the job was indexed — ones that were mid-parse beside it were parsed again, which costs time, not correctness |
| A document is reported `skipped`, and the job still `completed` | The file parsed without error and held nothing to index — zero bytes, only whitespace, or markup with no text. Its entry carries `reason_code: EMPTY_DOCUMENT`. It is deliberately not `indexed` (the index holds nothing of it) and not dead-lettered (nothing failed) | `GET /v1/ingest/{job}/documents?status=skipped` lists them; the job's `counts.skipped` says how many. Nothing to retry: fix the source and ingest it again. A scanned PDF with no text layer is a different case — that one fails and is dead-lettered |
| A document is dead-lettered with `ENCODING_UNREADABLE` | It is not text in UTF-8, UTF-16, UTF-32 or Windows-1252: no codec read it, and more than half of what a lenient decode produced was replacement characters or control codes. Usually a binary file with a text suffix, or text in a code page such as Shift-JIS or KOI8-R | Convert the file to UTF-8 (`iconv -f <encoding> -t utf-8`) and ingest it again. A file that *was* read by a fallback codec is indexed and carries the `encoding_fallback` parse flag instead |
| A PDF is dead-lettered with `PDF_ENCRYPTED` | The PDF needs a password to open. A PDF protected only against editing or printing opens without one and is not refused | Remove the password (save an unprotected copy) and ingest it again; retrying the same file fails the same way |
| A DOCX is dead-lettered with `PAYLOAD_TOO_LARGE` although the file is small | Its zip members declare more than 256 MiB unpacked — a decompression bomb, or a document carrying very large embedded media | Nothing to do for a bomb. For a real document, remove or compress the embedded media, or split it; the limit is a constant in `core/parsing/docx.py`, not a setting |
| A JSON document is dead-lettered with `NESTING_TOO_DEEP` | It nests more than 512 levels deep. No real document needs that; generated or hostile ones do | Flatten the structure, or split the deep part into separate documents, and ingest again |
| A document is dead-lettered with `SOURCE_LINK_ESCAPES` | It is a symbolic link inside an allowed root that points to a file outside every directory in `ingestion.allowed_source_roots`. The file was not read | Remove the link, or, if its target belongs in the corpus, add the target's directory to `ingestion.allowed_source_roots`. Every other document of the job was ingested |
| Re-ingesting adds nothing | Working as designed: content-hash dedup makes replays no-ops (D3) | To force reprocessing, the source content must actually change (or use `index reembed` for model changes) |
| `POST /v1/ingest` returns 409 `CONFLICT` with no job id, saying the collection *was indexed without a keyword leg* | `retrieval.hybrid: true` over a collection that was created dense-only. The keyword-leg preflight refuses at acceptance (TASK-0119): everything after a `202` runs in a background task whose failures reach a log rather than the caller, so this is the last point at which the caller can be told | Rebuild, do not restart: `fasterrag index reembed <name> <sources>` builds a hybrid collection beside the live one and swaps the alias (D2). Or set `retrieval.hybrid: false` to keep serving dense-only knowingly. `fasterrag index list --json` shows which collections report `sparse: false`. The check costs one metadata round trip per job and has no off switch; it reads the live server's vector configuration, so it is right even on a backend that accepted a `create_collection` without adding the leg |
| Ingest fails: *point … carries a sparse vector but collection … has no sparse index* | The same mismatch as the row above, reaching you through a path the preflight does not yet cover — `fasterrag ingest` on the CLI, and the worker pools (TASK-0119). `retrieval.hybrid` was turned on after the collection was created, so new chunks carry a keyword leg the collection has nowhere to put | The same fix: reindex into a new collection with `fasterrag index reembed`, or set `retrieval.hybrid: false`. pgvector and Chroma refuse earlier, when the collection is opened; on Qdrant the mismatch surfaces at the first upsert (TASK-0305), which is why the error names a point id rather than the configuration that caused it |
| Ingest restarted from the beginning after a crash | Journal disabled | Set `ingestion.journal.enabled: true`; checkpoints every `checkpoint_every` documents let a crash resume exactly where it stopped |
| Scanned PDFs index with no useful text | OCR yielded nothing; the chunk carries a `low_text_yield` parse flag rather than silently indexing emptiness | Check the flag in document metadata; verify the OCR path is available for that source type. If the pages carry a thin text layer (a stamped header, a page number) the OCR path never ran: raise `parsing.minimum_chars_per_page` above what such a page yields, and `parsing.ocr_resolution` if OCR output is garbled on fine print |
| Ingestion is slow, GPU underutilized | CPU parse/chunk stage is the bottleneck, so embedding workers idle | Raise `workers.cpu_pool_size`; the streaming hand-off is designed so embedding never waits on parsing |
| An indexed document is missing from `fasterrag documents list` / `GET /v1/collections/{name}/documents` | The listing is the ingestion journal's per-collection record, not a scan of the vector database (TASK-0469). It records nothing when `ingestion.dedup` is `false`, and an archive import writes nothing to it; with multi-tenancy on, the route lists the `<tenant>__` collection, which `POST /v1/ingest` does not write to (TASK-0381) | Query the document to confirm it is indexed. Re-ingesting it with `ingestion.dedup: true` (the default) records it; the vector database itself is unaffected either way |
| Ingestion slow, CPU idle | Embedding is the bottleneck (or a provider is throttling) | Raise `workers.embedding_pool_size` / `embeddings.batch_size`; check for retry-storm logs indicating provider 429s |

### An ingest job is refused with `chunking.strategy is 'semantic', which needs a local embedding model`

Semantic chunking finds boundaries with an embedding model that each parse worker loads for itself, and only a local model can be loaded there: `embeddings.provider` must be `huggingface`. Any provider reached over the network (`openai`, `cohere`, `ollama`, a plugin) is refused with `CONFIG_INVALID` before the job is recorded. Set `embeddings.provider: huggingface`, or set `chunking.strategy` to `recursive`, `layout`, `fixed` or `late`. Until 2026-11-12 the pipeline refused the semantic strategy for every provider, and before 2026-11-11 it accepted the job and dead-lettered every document.

### A semantic-chunking job is slow to start, or a worker dies at start-up

With `chunking.strategy: semantic`, every parse worker loads its own copy of the embedding model before the first document is handed to the pool, and again in each fresh worker after a parse timeout recycles the pool. Start-up therefore takes one model load, and memory is `workers.cpu_pool_size` times the model's size. A worker killed while loading fails the job with `CHUNK_FAILED`; lower `workers.cpu_pool_size` or choose a smaller model.

### A job has been `running` since the process restarted

A job whose process exited is settled `failed` with `reason_code: INTERRUPTED`, never left `running` and never resumed on its own. A graceful stop settles it on the way down; after a kill, the next server to start settles it once its record has been quiet for `reliability.stale_job_seconds` (default 120), so for up to that long after a restart the job still reads `running`. To continue it from its checkpoint, run `fasterrag ingest --resume <job_id>` or replay the request's `idempotency_key` body field ([api-reference.md](api-reference.md)). The job did not corrupt anything: a document is remembered as indexed only once every chunk of it is in the vector database, so re-submitting the same sources also works — it indexes whatever was missed and deduplicates the rest. An exception *inside* a live process is different — the job is settled `failed` with the error's `code` as its `reason_code` (for example `VALIDATION_FAILED` for a source that cannot be resolved, `INTERNAL` for an unclassified error; TASK-0526), the log line `ingest job failed` carries the detail, and that job is not resumable: fix the cause and submit it again.

### Documents were dead-lettered with `EMBED_PROVIDER_ERROR` and a retry did nothing

Fixed on 2026-11-11 (TASK-0427). Earlier builds recorded a document as indexed when its chunks were queued, so a document that then failed to embed was reported as a duplicate on retry and skipped. On a journal written by an earlier build, the stale content hashes are still in `.fasterrag/journal/` — ingest those sources once with `--set ingestion.dedup=false` to index them.

## Retrieval quality

| Symptom | Likely cause | Fix |
|---|---|---|
| Exact identifiers/rare terms not found | Dense-only retrieval — vectors are weak on exact tokens | Set `retrieval.hybrid: true`; the BM25 leg exists for exactly this. **Over a corpus that is already indexed this is a reindex, not a restart** — see the next row before you flip it |
| Turned on `retrieval.hybrid` over an existing index, and now every query fails: 422 `VALIDATION_FAILED` saying the collection has no sparse index (pgvector, Chroma), or — on Qdrant, which only learns of the missing vector when it answers — 503 `RETRIEVAL_FAILED` quoting a Qdrant status code and naming nothing useful | The collection was created dense-only and a keyword leg cannot be attached to a populated one: Qdrant will not mix a named sparse vector with an unnamed dense one, and the pgvector and Chroma adapters refuse the same mismatch (TASK-0119). The flag is read at startup; nothing re-shapes the index to match it. Queries are *not* silently degraded to dense-only — they fail, and the problem document carries `retryable: false` even where the status is 503 | Confirm with `fasterrag index list --json`: the collection's `sparse` field is `false`. Then rebuild rather than restart — `fasterrag index reembed <name> <sources>` builds a hybrid collection alongside the live one and swaps the alias (D2), so the dense-only index keeps serving throughout. If you cannot reindex now, set `retrieval.hybrid: false`, which leaves you dense-only but knowingly so. The mismatch is now refused before a write rather than discovered by a query: `POST /v1/ingest` and `POST /v1/ingest/{job}/retry-dlq` preflight it and return 409 `CONFLICT`, and `fasterrag index create` re-checks after creating rather than report a success the backend did not deliver (TASK-0119). The query path is deliberately *not* checked — a per-query probe would charge every question for a mistake made once — so on a deployment that only ever reads, a failed query is still the first symptom |
| Right document retrieved, wrong passage | Chunks too large, or boundaries cutting through the answer | Lower `chunking.chunk_size`; try `strategy: layout` or `semantic`. Chunk size is a per-corpus tuning choice with no published optimum we rely on ([config-reference.md](config-reference.md)) — move it with the eval harness, not by guesswork. If instead the passage is retrieved but buried in an overlong prompt, the lever is the assembled retrieved-context budget rather than chunk size — the ~2,500-token *context cliff* bounds that total ([references.md](references.md) R3, a single preliminary preprint). **There is no configuration key for it today** (TASK-0285): the budget is fixed at 4,000 tokens inside `GenerationService`, so the only lever an operator has from `config.yaml` is `retrieval.top_k`, which bounds how many chunks compete for that budget |
| Retrieval finds it, but the answer misses it | Reranking off, or `top_k` truncating before the good chunk | Enable `retrieval.rerank` — expected to help most here, though the gain is unmeasured; raise `rerank_top_n` so the reranker sees more candidates |
| Chunks lack context to be understood alone | Pronouns/shorthand referencing surrounding text | Enable `chunking.contextual_enrichment` and reindex ([prompts.md](prompts.md) P2; evidence in [references.md](references.md) R1) |
| Quality dropped after a change | A retrieval-affecting change shipped ungated | Enable `eval.regression_gate: true`; use `fasterrag replay --trace <id> --config candidate.yaml` to see exactly what changed (D8) |
| Every query fails with 409 `INDEX_DRIFT`, or the log says *was indexed by embedding model … not the configured one* | `embeddings.model` (or its version or dimensions) changed after the collection was indexed; the index's vectors and the query's come from two embedding spaces, so results would be confidently wrong (TASK-0477) | Rebuild the collection under the new model with `fasterrag index reembed <collection> <sources...>`, or put back the embedding settings it was built with. `retrieval.verify_lock: warn` serves such queries and only logs; `off` skips the check |
| No idea which knob to turn | Guesswork is the actual problem | `fasterrag autopilot run` — measured suggestions from your own corpus, never auto-applied (D6) |

## Answers & generation

| Symptom | Likely cause | Fix |
|---|---|---|
| `INSUFFICIENT_EVIDENCE` instead of an answer | Grounded-or-refuse declined below the faithfulness threshold — a feature, not a failure (D5) | Inspect `best_candidates`; if refusals are too aggressive, lower `generation.faithfulness_threshold`, or improve retrieval so real evidence reaches the context |
| Answers look plausible but unsupported | Grounding not enforced | Enable `generation.grounded_or_refuse: true`; check the `faithfulness` field and `unsupported_claims` in the trace |
| Response labelled `degraded: true` | The degradation ladder engaged — a component is down and you're being told, rather than silently served worse results (D4) | Read `mode`: `hybrid_only` = reranker down · `cache_only` = vector DB down · `extractive` = LLM down. Fix the named component; recovery is automatic |
| A query returns `404 NOT_FOUND`: `no collection named '…'` | The query named a collection — or `vector_db.collection.default_name` names one — that the vector database does not hold. Typical after a `DELETE /v1/collections/{name}`, a typo, or with multi-tenancy on: `/v1/query` addresses the name unprefixed (TASK-0339) | `fasterrag index list` (or `GET /v1/collections`) shows what exists; ingest into the collection to create it. The vector contract suite requires every backend to report this the same way, and it does not count against the vector-database circuit breaker |
| A query returns `422 VALIDATION_FAILED`: `a query must contain text to search for` | The query was empty or whitespace only. The request schema's `min_length` counts whitespace, so the service refuses it instead — on REST, the CLI and the Python facades alike (TASK-0468) | Send the question text. Nothing was embedded or charged for the refused query |
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

## Python package

| Symptom | Likely cause | Fix |
|---|---|---|
| `USAGE_ERROR: this FasterRag instance has not been started` | A facade method was called outside `async with` (async facade) or `with` (blocking facade) | Enter the context manager first: `async with FasterRag.from_config(...) as rag:` or, with `fasterrag.sync`, `with FasterRag.from_config(...) as rag:`. Builds before TASK-0461 reported this as `CONFIG_INVALID` |
| `USAGE_ERROR: fasterrag.sync.FasterRag was called from the thread that runs its own event loop` | Code that the blocking facade is itself running — a callback scheduled on its private loop — called back into the blocking facade, which would wait on itself forever | Use the async facade there (`from fasterrag import FasterRag`, `async with`). Calling the blocking facade from a notebook cell or from under `asyncio.run` is fine: it runs its loop on a thread of its own (TASK-0497) |
| `TENANT_FORBIDDEN` from `rag.query`, `rag.query_stream` or `rag.retrieve` | `security.tenant_scoped_retrieval` is on and the call named no tenant | Pass `tenant="<id>"` to the call. Before TASK-0461 the facade had no way to name one, so every read was refused under that flag |

## Performance

| Symptom | Likely cause | Fix |
|---|---|---|
| p95 latency high, retrieval fine | Generation dominates (provider-bound) | Check the `timings_ms` split; retrieval and generation are measured separately for exactly this reason |
| Rerank stage dominates latency | `rerank_top_n` too high for your reranker/hardware | Lower it, or use a smaller cross-encoder. fasterRag publishes no expected figure for the stage — read your own `fasterrag_stage_duration_seconds` |
| Slow first query after startup | Cold caches and model load | Expected. Benchmarks report cold and warm separately ([performance.md](performance.md)); pre-warm if TTFT matters |
| Memory grows over a long run | Should not happen — `pytest -m soak` asserts no RSS or handle trend over a sampled run, though the 24 h run has never been performed ([testing-strategy.md](testing-strategy.md) §1.8) | File a bug with duration, RSS trend, and `fasterrag status`; unbounded growth is a defect, not tuning |
| Your numbers disagree with a doc | Docs contain **goals**, not measurements, until a ledger entry exists | Check [benchmarks.md](benchmarks.md). An unmeasured claim presented as fact is itself a bug — file it |

## Still stuck

1. `fasterrag doctor --json` and `fasterrag status` — capture both.
2. Pull the trace: `GET /v1/traces/{trace_id}` (or `fasterrag replay --trace <id>`) — it holds the retrieved set, all leg scores, the exact prompt, and the exact response.
3. Open an issue using the bug template. Include the `code`, `trace_id`, config sections involved, and versions. **Never paste `.env` contents.** Suspected security issue? Use private reporting instead ([.github/SECURITY.md](https://github.com/balrajsingh165/fasterrag/blob/main/.github/SECURITY.md)).
