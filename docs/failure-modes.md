# failure-modes.md — FMEA (Failure Modes and Effects Analysis)

Every anticipated failure, its detection signal, automatic mitigation, recovery procedure, and **the test that proves it**. Tests reference [testing-strategy.md](testing-strategy.md) (unit/integration/chaos/property suites); chaos scenarios are scripted and repeatable (D12). Error codes are the stable problem `code`s from [api-reference.md](api-reference.md).

| # | Component | Failure mode | Detection signal | Automatic mitigation | Recovery procedure | Test that proves it |
|---|---|---|---|---|---|---|
| 1 | Document parser | Corrupt/malformed file crashes the parse | `ParseError` in worker log with trace id; DLQ entry reason `PARSE_FAILED` | Document routed to DLQ after `dlq.max_retries`; pipeline continues | Inspect `GET /v1/ingest/{job}/documents?status=dead_lettered`; fix source; `retry-dlq` | Chaos: corrupt-document scenario |
| 2 | Document parser | Scanned PDF yields empty text (OCR miss) | Parse-quality flag `low_text_yield` in chunk metadata; ingest counts show 0 chunks for doc | Document flagged, not silently indexed empty; DLQ if zero extractable text | Re-ingest with OCR path forced | Golden-file test: scanned-doc fixture |
| 3 | Document parser | Table structure mangled (reading order lost) | Golden-file snapshot diff in CI | n/a (build-time guard) | Fix parser; snapshot re-approved via review | Golden-file parser suite |
| 4 | Chunker | Invariant violation (empty chunk, offset out of bounds, overlap ignored) | `ChunkError`; property-test failure in CI | Chunk batch rejected; document DLQ'd with `CHUNK_FAILED` | Fix chunker; re-run DLQ | Hypothesis property suite (all 5 invariants) |
| 5 | Chunker | Oversized chunks past the context cliff | Config validation rejects `chunk_size > 2500`; warning logged > 1024 | Fail-fast at startup (`ConfigError`) | Correct `chunking.chunk_size` | Unit: config cross-field rules |
| 6 | Embedding worker pool | Worker process killed mid-batch | Heartbeat loss; queue item lease expires | Batch re-queued; idempotent upserts prevent duplicate vectors; other workers continue | None needed — self-heals; check `fasterrag status` | Chaos: kill-worker scenario (asserts zero duplicate vectors) |
| 7 | Embedding worker pool | Provider rate-limits (429) | `ProviderError(retryable=True)`; retry metric increments | Backoff + jitter retries; queue absorbs burst (backpressure); breaker opens if sustained | Lower `embeddings.batch_size`/pool size or raise provider tier | Integration: mocked 429 storm |
| 8 | Embedding worker pool | Model reload thrash (worker restarts repeatedly) | Worker restart counter metric; slow throughput gauge | Stateful workers load model once; restart loop triggers `EmbedError` after threshold | Inspect worker logs; fix model path/VRAM | Soak test: throughput stability |
| 9 | Embedding worker pool | Embedding dimension mismatch vs collection | Adapter rejects upsert; `EmbedError` naming dims | Batch halted before write; job marked failed with reason | `index lock verify`; re-embed via D2 or fix `embeddings.dimensions` | Adapter contract suite: dim-mismatch case |
| 10 | Job queue | Queue overflow (ingestion storm) | `fasterrag_queue_depth` at max; API returns 429 `QUEUE_FULL` + `Retry-After` | Backpressure: reject new work, never OOM; bulkhead protects query path | Client retries after `Retry-After`; scale workers | Integration: queue-full returns 429; load test |
| 11 | Job queue | Poisoned item retried forever | Retry counter per item | Bounded retries → dead-letter state on job record | Inspect DLQ; fix cause; `retry-dlq` | Unit: retry-policy bounds |
| 12 | Vector DB (managed Docker) | Container exits/crashes | `health()` fails; `/readyz` 503; `fasterrag_circuit_state=2` | Circuit opens; degradation ladder serves `cache_only` (D4); provisioner restarts container | `fasterrag provision qdrant --status`; auto-restart; data safe on named volume | Chaos: stop-Qdrant scenario, plus the real `docker stop` variant |
| 13 | Vector DB (managed Docker) | Bind-mount data loss on Windows/WSL | Loader rejects bind mounts on Windows/WSL at startup | Fail-fast `ConfigError` (named volume required) | Switch `vector_db.docker.volume` to a named volume | Unit: config validation on win32 path |
| 14 | Vector DB (external mode) | Instance not running at startup | `/readyz` 503 listing `vector_db`; doctor check fails with fix string | Startup proceeds degraded; queries `cache_only`; breaker open | Start Qdrant; breaker half-open probe auto-recovers | Integration: external-mode down/up cycle |
| 15 | Vector DB (remote-IP mode) | Only 6333 exposed, gRPC 6334 blocked (Discussion #2195 case) | Doctor reachability check per port fails naming 6334 | Doctor blocks provisioning; adapter falls back to REST if `prefer_grpc: false` | Expose 6334 or set `prefer_grpc: false` explicitly | Doctor matrix test: 6334-blocked case |
| 16 | Vector DB (remote-IP mode) | Network partition mid-query | Timeout `reliability.timeouts.vector_db_ms`; `RETRIEVAL_FAILED` | Retry (retryable); then breaker opens → `cache_only` mode | Restore network; breaker auto-closes | Chaos: partition injection |
| 17 | Vector DB (any) | Auth failure (bad/missing API key) | `ProviderError(retryable=False)` 401/403 from backend | No retry (non-retryable); clear problem naming env var (never the value) | Fix key in `.env`; restart | Adapter contract suite: auth-failure case |
| 18 | Reranker | Model fails to load / OOM | `RetrievalError` at startup or first query; `RERANK_FAILED` | Degradation ladder: `hybrid_only` mode (fusion results served unreranked, flagged) | Fix model/VRAM; toggle `retrieval.rerank` off if needed | Chaos: reranker-down rung |
| 19 | Reranker | Latency blowout (> stage budget) | `fasterrag_stage_duration_seconds{stage=rerank}` p95 alarm | Per-call timeout → fall back to `hybrid_only` for that query, flagged | Reduce `rerank_top_n`; smaller cross-encoder | Load test: rerank stage under concurrency |
| 20 | LLM provider | Hard outage / 5xx | `GenerationError`/`ProviderError`; breaker metric | Retries → breaker opens → ladder serves `extractive` answers (D4) | Provider recovers; breaker half-open probes | Chaos: LLM-down rung |
| 21 | LLM provider | Slow responses (hang) | `reliability.timeouts.llm_ms` exceeded | Timeout aborts; retry once if retryable; else `extractive` mode | Investigate provider/base_url | Chaos: slow-LLM latency injection |
| 22 | LLM provider | Mid-stream failure during SSE | `event: error` emitted, stream closed without `done` | Client sees explicit incomplete-answer signal; trace records partial | Client retries; check trace by `trace_id` | Integration: injected mid-stream fault |
| 23 | Semantic cache | Backend (Redis) down | `CacheError` logged with trace id; cache metrics flatline | Degrade to cache-off: queries run the full pipeline (correctness preserved) | Restore Redis; entries rebuild organically | Integration: cache-backend kill |
| 24 | Semantic cache | Stale answer after corpus change | Invalidation-event counter vs ingest events mismatch | Event-driven invalidation on every corpus change + TTL ceiling | Manual flush via admin if ever needed | Integration: ingest-then-query staleness test |
| 25 | Semantic cache | Threshold too loose (wrong answer served) | Eval harness cache-correctness probe | Bounded by validation (0.90–0.99); default 0.95 | Raise `cache.similarity_threshold` | Eval: paraphrase/near-miss fixture set |
| 26 | Config loader | Invalid YAML / schema violation | Startup aborts with `ConfigError` naming key; exit non-zero | Fail-fast — process never serves misconfigured | Fix `config.yaml`; `fasterrag config validate` | Unit: every cross-field rule has a failing fixture |
| 27 | Secrets loader | Referenced env var missing | Fatal `ConfigError` naming the variable (value never logged) | Fail-fast at startup | Add var to `.env` | Unit: missing-env fixture |
| 28 | Auto-provisioner | Port already taken (e.g. 3000) | Doctor pre-check fails with fix string; `PROVISIONING_FAILED` if raced | Provisioning refused before any mutation (doctor-gated) | Free the port or change config; re-run | Doctor matrix test: port-conflict case |
| 29 | Auto-provisioner | Langfuse secrets regenerated (would invalidate keys) | Provisioner detects existing `.env` values | Never regenerates existing `SALT`/`ENCRYPTION_KEY`/`NEXTAUTH_SECRET`; converge-only | Restore `.env` from backup if manually deleted | Integration: re-provision idempotency test |
| 30 | Auto-provisioner | Compose up partially fails (e.g. ClickHouse unhealthy) | Health-check loop times out; `PROVISIONING_FAILED` with failing container named | Stack left in inspectable state; no URL returned; nothing else mutated | Fix per hint; re-run (idempotent) | Integration: injected unhealthy-container |
| 31 | Ingestion journal | Crash between checkpoints | Journal replay detects last checkpoint < accepted count | Resume from last checkpoint; dedup makes replayed docs no-ops | None — automatic on restart | Chaos: kill-mid-ingest resume test |
| 32 | Ingestion journal | Journal file corruption (torn write) | Checksum mismatch on journal load | Atomic write-temp-then-rename prevents torn writes; last-good checkpoint used | Restore journal from backup if both damaged | Unit: torn-write simulation |
| 33 | Disk | Disk full during ingest | OS write error → typed `IngestionError`, the chained `OSError` keeping errno 28 so the cause survives translation (**shipped**, TASK-0234 ✅; measured against a real `ENOSPC`); disk gauge alarm | Clean halt at a checkpoint; queues drained; no partial index writes | Free space; job resumes from checkpoint | Chaos: disk-full scenario, plus the real size-limited-filesystem variant |
| 34 | Disk | Trace store exceeds retention | Store size metric | Retention pruning at `traces.retention_days` | Lower retention / archive | Unit: retention pruning |
| 35 | Network | DNS/socket failure to any provider | `ProviderError(retryable=True)`; RED error rate | Timeout + retry + breaker + ladder (whichever rung applies) | Restore connectivity; breakers auto-close | Chaos: partition injection (per provider) |
| 36 | Dashboard | Dashboard process down | Dashboard health endpoint fails (its own liveness) | None needed — **observability-only**: RAG serving is entirely unaffected | Restart dashboard container | Integration: kill dashboard, assert query path unaffected |
| 37 | Dashboard | Attempted state mutation via dashboard | No such route exists; 404/405 by construction | Read-only by design (no control endpoints compiled in) | n/a | Unit: route-table assertion (zero mutating routes) |

Coverage cross-check: parser (1–3), chunker (4–5), embedding pool (6–9), job queue (10–11), vector DB managed-Docker (12–13) / external (14) / remote-IP (15–16) / any (17), reranker (18–19), LLM provider (20–22), semantic cache (23–25), config loader (26), secrets loader (27), auto-provisioner (28–30), ingestion journal (31–32), disk (33–34), network (35), dashboard (36–37).

> **As-built note** (corrected by the 2026-08-09 claims audit, TASK-0238 — the previous version of this note said the circuit breaker was unimplemented, which stopped being true on 2026-08-06).
>
> - **The circuit breaker is built and wired** (`core/breaker.py`, TASK-0148 ✅): three states per [reliability.md](reliability.md) §3, consulted by the embedding worker pool and by the Qdrant and pgvector adapters, and exported as `fasterrag_circuit_state`. So rows 12, 14, 16, 20 and 35 get a real breaker. **Exception: the LLM path.** A breaker named `llm` is constructed at startup and no generation code path consults it (TASK-0245), so row 20's "retries → breaker opens" is currently just retries.
> - **`cache_only` is still not implemented** (TASK-0159). The string does not appear anywhere in `src/`. Every row whose mitigation names that rung (12, 14, 16) describes specified, unbuilt behaviour: a vector-DB outage today raises a retryable `RETRIEVAL_FAILED` instead of serving a degraded answer.
> - **Three rows name a test that does not exist.** Rows 8 and 19 cite a soak test and a load test, and rows 16 and 35 cite "partition injection" — none of these suites has been written (TASK-0243 for load/soak, TASK-0241 for the missing chaos scenarios). Row 10's "load test" clause is the same gap.
>
> The chaos log below records what the shipped system actually does today, and is authoritative over the table wherever the two disagree.

## Observed behavior — chaos suite run

D12 requires that the injected faults and their observed behavior are recorded, not merely
asserted in a test file. The suite is `tests/chaos/test_chaos.py`, run with `pytest -m chaos`.

| Scenario (testing-strategy.md §1.9) | Observed | Date |
|---|---|---|
| Kill an embedding worker mid-batch | Re-writing the same batch produced 6 upsert calls carrying only 3 distinct point ids — the deterministic id makes the second write a replace, so no duplicate vectors reach the index | 2026-08-01 |
| Stop the Qdrant container | `health()` reports unhealthy; every operation raises `RETRIEVAL_FAILED` classified `retryable=True`, which is what lets the breaker act on it rather than treating it as a permanent fault; the adapter serves again as soon as the backend answers | 2026-08-01 |
| Feed a corrupt/malformed document | Recorded in the DLQ with `reason_code: PARSE_FAILED` and its attempt count; a sibling document in the same job still reached `indexed`, so one bad file did not stop the pipeline | 2026-08-01 |
| Slow LLM (latency injection) | Degraded to `extractive` with `degraded: true`, serving the retrieved passages and their citations rather than returning nothing | 2026-08-01 |
| Disk-full during ingest | The journal raises rather than corrupting state; the trace store, which writes *after* a query has been answered, swallows the failure and loses the record instead of failing the request; a checkpointed job resumes from index 499 at 500 | 2026-08-01 |

**Scope of this run.** Every scenario injects its fault at the closest honest seam — an
adapter raising what the real failure raises, or a directory that genuinely cannot be
written. Simulating a *symptom* is legitimate; simulating the *handling* would prove nothing.
Recovery *times* per scenario are not yet recorded; that needs the isolated hardware
TASK-0084 is blocked on.

## Observed behavior — real-fault run

The two environmental scenarios above are also run against the operating system itself, which
is a different question: the seam-level cases prove how fasterRag responds to a stopped
backend or a full disk, while these prove that the condition the OS actually produces is the
one it responds to. A double raises the exception its author expected; a stopped container
takes its published port with it, and a full filesystem returns `ENOSPC` from a syscall a
client library has to translate first. The suite is `tests/chaos/test_real_faults.py`, run
with `pytest -m integration`, and it skips when no Docker daemon is present.

| Fault, as really produced | Observed | Date |
|---|---|---|
| `docker stop` on a live Qdrant holding an indexed point | The adapter raises a typed retryable `RETRIEVAL_FAILED` — the same code the scripted double raises, which is what TASK-0226 corrected — rendered as a 503 problem document with a trace id and never a generic 500, and `health()` reports unhealthy with a detail rather than raising through the readiness probe. The failure is bounded but **not uniformly fast**: four consecutive runs measured 44.2 ms, 5040.9 ms, 36.1 ms, 5043.9 ms, because Docker Desktop's port proxy sometimes accepts the connection and hangs instead of the kernel refusing it, in which case the bound is our own `reliability.timeouts.vector_db_ms` (5 s) rather than the OS's. What is proven is the bound, not a latency figure | 2026-08-09 |
| `docker start` on the same container | The backend answers again with no intervention and the search returns the same point id — the named volume kept the data across the stop | 2026-08-09 |
| Journal writes onto a tmpfs mounted with a hard 512k limit, filled to zero free bytes | 516,096 bytes of filler took the filesystem to `free_bytes: 0`. Starting a **new** job then failed with the OS's own `ENOSPC` (errno 28, `[Errno 28] No space left on device`) rather than silently losing the record — but **checkpointing the already-running job still succeeded**, and that asymmetry is the mechanism, not luck: `_write_atomically` renames the live record over the stale `.previous` copy before writing, freeing a file the same size as the one it is about to write, while a new job frees nothing. Removing that rename made the checkpoint fail too, which is how the mechanism was confirmed. The record stayed readable throughout, so the halt is clean and "resume from the last checkpoint" is reachable on the very disk that caused the halt; deleting the filler is the whole recovery — the journal created a new job immediately and the earlier one was still there to resume | 2026-08-09 |

**The gap this run found that the seam-level run could not.** Row 33's promise that the OS
write error becomes a typed `IngestionError` does not hold: the raw `OSError` escapes
`Journal._write_atomically` untranslated (TASK-0234). The scripted suite cannot see it
because its unwritable-directory case accepts `(IngestionError, OSError)` and so passes
either way. It is asserted **as observed** rather than as promised, so the day the fix lands
that case fails and forces the row and the assertion to move together.

**Platform note.** The disk variant runs its write inside a Linux container because the
Windows host it was executed on cannot constrain a filesystem without administrator rights —
an NTFS quota or a mounted VHD both need elevation, and a test may not demand it. The tmpfs
is a real filesystem returning a real `ENOSPC` to the real `Journal`; what is containerised
is where the write lands, not whether the failure is genuine. The same case runs unchanged on
a Linux or macOS host.
