# testing-strategy.md — Testing Pyramid, Eval Harness, CI Gates

Doctrine: **a claim without a measurement is a bug** ([reliability.md](reliability.md)). Every capability in [differentiators.md](differentiators.md) names its acceptance test here or in the FMEA ([failure-modes.md](failure-modes.md)); every performance claim traces to the [benchmark ledger](benchmarks.md).

## 1. Testing pyramid (all layers required)

> **Build status (2026-08-09 claims audit, TASK-0238).** "Required" is the target, not a report. Layers **1.1–1.6 and 1.9 are built and run in CI**. Layers **1.7 (load), 1.8 (soak), and 1.10 (mutation) do not exist**: no k6 or Locust script, no soak suite, and `mutmut` is not a dependency — the `load`, `soak`, and `mutation` markers are declared in `pyproject.toml` and used by no test. They are tracked as TASK-0243, and FMEA rows that name one of them as their proving test are annotated accordingly.

### 1.1 Unit tests

- Fast, isolated, no network/containers; mirror the `src/` layout under `tests/unit/`.
- **Coverage gate ≥ 85% on `core/`, `adapters/`, `workers/`** — CI fails below the gate.

### 1.2 Property-based tests (Hypothesis) — every chunker

Invariants asserted for `fixed`, `recursive`, `semantic`, `layout`, `late` on arbitrary generated documents:

1. Concatenating chunks (minus configured overlap) reconstructs the source text.
2. No empty chunks.
3. Character offsets are monotonic and in-bounds.
4. Configured overlap is respected.
5. No chunk exceeds `chunk_size` beyond tokenizer-boundary tolerance.

### 1.3 Golden-file parser tests

Messy PDFs, complex tables, and scanned/OCR documents parsed → structured output snapshot-compared against committed golden files (reading order, table cells, headings, metadata). Any diff is a reviewed change, never silent.

### 1.4 Integration tests (testcontainers)

- A real Qdrant instance exercised in **Docker-managed mode** and **remote host:port mode** (both 6333 and 6334 paths, `prefer_grpc` both ways, API-key auth on/off).
- Ingest → query round-trips over small corpora; journal resume; DLQ routing.
- Marked `-m integration`; run in CI on every PR.

### 1.5 Adapter contract suite

**One shared test suite that every `VectorDBAdapter` implementation must pass** (`create_collection`, `upsert`, `search`, `update`, `delete`, `health`, filter push-down, batch semantics, idempotent upserts). Two adapters currently pass it — **Qdrant and pgvector** — which is what turns "any vector DB" from an assertion into evidence against a genuinely different (SQL) paradigm; it is not yet a promise tested across all six backends named in [scope.md](scope.md), because four of them are unbuilt (TASK-0049). Third-party adapters registered via entry points run the same suite ([python-api.md](python-api.md)). Embedding and LLM adapters have equivalent (smaller) contract suites.

### 1.6 Retrieval eval harness

- Metrics: **recall@k, MRR, nDCG** (retrieval) and **faithfulness** (generation grounding).
- Dataset fixtures committed under `tests/eval/datasets/`; golden Q&A sets generatable from a corpus (shared machinery with Autopilot, D6).
- Runs as `pytest -m eval` and via `fasterrag benchmark --suite eval`.
- **Regression gate (D7)**: *specified* as blocking every config or index change whose recall@k drops > `eval.recall_tolerance` or nDCG drops > `eval.ndcg_tolerance`. **Not wired into CI, and its committed baseline has drifted** (TASK-0244): the comparison is implemented in `services/regression.py` and reachable through `fasterrag benchmark --suite eval`, but nothing runs it on a push, and the handbook baseline recorded 2026-08-02 is no longer comparable to the canonical config — so the gate would report *blocked* rather than pass or fail. See §6 for the same statement in its list of what CI actually runs.

**Golden-set schema** — shared by the eval harness, the regression gate (D7), and Autopilot (D6), so the three can never diverge. JSONL, one record per line:

```json
{"id": "q_0001", "query": "What does the vendor agreement say about termination?",
 "relevant_chunk_ids": ["c_9f2", "c_a01"], "relevant_document_ids": ["d_112"],
 "answer_reference": "Either party may terminate with 30 days written notice.",
 "metadata": {"department": "legal"}, "source": "human", "created_at": "2026-07-29"}
```

| Field | Type | Required | Meaning |
|---|---|---|---|
| `id` | str | ✅ | Stable query id, unique within the set. |
| `query` | str | ✅ | The question. |
| `relevant_chunk_ids` | list[str] | ✅ (may be empty for adversarial records) | Ground-truth chunks; basis for recall@k/MRR/nDCG. |
| `relevant_document_ids` | list[str] | — | Document-level ground truth (used when chunk ids churn across re-chunking). |
| `answer_reference` | str\|null | — | Reference answer; basis for faithfulness scoring. `null` + empty chunk ids = adversarial "unanswerable" record for D5 testing. |
| `metadata` | object | — | Filter context the query should run with. |
| `source` | str | ✅ | `human` \| `autopilot` (generated sets are marked; human-reviewed promotion flips the value). |
| `created_at` | str | ✅ | ISO date. |

Golden sets are versioned files; a set edit that changes scores is reviewed like code, and archives export them (`golden-set.jsonl`, [archive-format.md](archive-format.md)).

### 1.7 Load tests (k6 or Locust) — **NOT BUILT** (TASK-0243)

*Specified:* scripted scenarios (query-only, ingest-only, mixed) at stepped concurrency; publish p50/p95 **with the exact hardware spec** into the benchmark ledger. No load number is ever quoted without its hardware line.

*As built:* no k6 or Locust script exists in this repository. The `load` pytest marker is declared and unused.

### 1.8 Soak test — **NOT BUILT** (TASK-0243)

*Specified:* 24 h sustained ingest + query at moderate load; assert **no memory growth and no file-descriptor growth** (RSS and fd counts sampled and trend-tested), no queue leakage, no journal bloat.

*As built:* no soak suite exists. The `soak` pytest marker is declared and unused. Nothing currently verifies the absence of memory or fd growth.

### 1.9 Chaos suite (scripted, repeatable — D12)

| Injected fault | Asserted behavior |
|---|---|
| Kill an embedding worker mid-batch | Job resumes from the queue; **no duplicate vectors** (idempotent upserts). |
| Stop the Qdrant container | *Specified:* circuit opens; degraded `cache_only` responses with `degraded: true`; automatic recovery when it returns. **As built:** the breaker opens and recovery is automatic, but the `cache_only` rung does not exist (TASK-0159), so the suite asserts a retryable `RETRIEVAL_FAILED` instead — see [failure-modes.md](failure-modes.md) §observed behavior. |
| Feed a corrupt/malformed document | Routed to DLQ with reason code; pipeline continues. |
| Slow LLM (latency injection) | `reliability.timeouts.llm_ms` triggers; degradation ladder serves `extractive` mode. |
| Disk-full during ingest | Clean halt with typed error; resumable from the journal checkpoint after space is freed. |

Every scenario is a script in the repo; observed behavior is recorded in [failure-modes.md](failure-modes.md). Reliability anyone can re-run.

The five rows above inject their fault at the closest honest code seam, so they run on any machine. The two environmental rows — stop-Qdrant and disk-full — additionally have **real-fault variants** in `tests/chaos/test_real_faults.py`, marked `integration` because they need a Docker daemon and skip cleanly without one: a throwaway Qdrant really stopped with `docker stop`, and the real journal writing onto a tmpfs with a hard size limit. The seam-level cases prove how fasterRag responds to those conditions; the variants prove that what the operating system actually reports is what it responds to.

### 1.10 Mutation testing (mutmut) — **NOT BUILT** (TASK-0243)

*Specified:* sampled on the two places where a subtle bug silently ruins quality — **chunk-boundary logic** and **retrieval-scoring/fusion logic**. Surviving mutants are triaged as missing-test bugs in [todo.md](todo.md).

*As built:* `mutmut` is not a dependency and has no configuration; the `mutation` marker is declared and unused. Mutation checking is done **by hand** today — several slices record deliberately breaking the code and confirming exactly one test fails (TASK-0193, TASK-0208, TASK-0211, TASK-0229, TASK-0232) — which is the practice this layer would automate, not a substitute for it.

## 2. CI quality gates (all blocking)

1. `ruff check` + `ruff format --check` clean.
2. `mypy --strict` zero errors.
3. Unit tests green; coverage ≥ 85% on `core/`, `adapters/`, `workers/`.
4. Integration tests green (testcontainers).
5. Adapter contract suite green for all in-repo adapters.
6. Eval harness runs (`pytest -m eval`, the `models` job). **The D7 tolerance gate is not wired into CI** — `services/regression.py` and `fasterrag benchmark --suite eval` implement it, but no CI step compares against the committed baseline, so this gate is not currently blocking (TASK-0244).
7. Secret scanning clean; dependency lockfile up to date ([security.md](security.md)).
8. Commit-message rule enforced (single line, no trailers) via pre-commit/CI check.
9. Docs updated in the same change for any behavior change (reviewed, not automated).
10. The always-loaded docs must not contradict the tree — `scripts/check_doc_truth.py` fails when `CLAUDE.md` or `README.md` claims a documentation-only repository while `src/fasterrag` holds an implementation. CI-only, not a commit hook.
11. `fasterrag doctor --json` runs against the CI environment itself, after `fasterrag config init` in an empty directory — dogfooding D10 and the packaged-template path on every push.
12. The fast suite runs on **windows-latest** as well as ubuntu. Two shipped defects were Windows-only (a path-case document id, a UTF-8 BOM the config loader rejected), and neither is reachable from an ubuntu-only matrix.

**On gate 3's scope.** The threshold is applied to `core/`, `adapters/`, and `workers/` specifically, not to the repository. A tree-wide number is diluted by CLI plumbing and API wiring, so it can sit comfortably above the line while the retrieval and adapter code the gate exists to protect regresses underneath it. Measured at **87.11% branch coverage over 3525 statements** (2026-08-02) when the gate was set at 85 to leave working room without letting it drift. **Re-measured at 83% over 4554 statements on a Windows dev machine (2026-08-10, release-readiness review)** — below the threshold CI enforces, with the reconciliation tracked as TASK-0252 in [todo.md](todo.md): either the Linux CI measurement differs by platform, CI on main is red, or the gate's scope drifted; whichever it is, the answer is restoring headroom, not lowering the gate.

**`metadata` on a golden record is a retrieval *filter*, not annotation.** The harness passes it straight to `retrieve(filters=...)`, so a descriptive key that no chunk payload carries filters every candidate away and the run scores a flat `0.0` — indistinguishable from total retrieval failure. Keep it `{}` unless you are deliberately scoping the query.

## 3. Test taxonomy and markers

| Marker | Runs | Needs |
|---|---|---|
| (none) unit | every push, < 2 min | nothing external |
| `integration` | every PR | Docker (testcontainers) |
| `eval` | PRs touching retrieval paths + nightly | small datasets in repo |
| `chaos` | runnable now against a local stack | Docker |
| `load`, `soak` | *planned* — no test carries either marker yet (TASK-0243); results would go to the ledger | reference hardware (TASK-0084) |
| `mutation` | *planned* — no test carries the marker; `mutmut` is not installed (TASK-0243) | — |

## 4. Definition of done for any implemented feature

Code with docstrings only (comment policy) · unit + integration tests green · coverage gate met on touched packages · typing and lint clean · benchmark ledger entry if performance-relevant · affected docs updated in the same change · [todo.md](todo.md) ticked with date · single-line commits.
