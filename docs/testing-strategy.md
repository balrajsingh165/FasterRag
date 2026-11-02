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
- Runs as `pytest -m eval`, via `fasterrag benchmark --suite eval`, and in CI as the `eval-gate` job.
- **Regression gate (D7)**: blocks every change whose recall@k drops > `eval.recall_tolerance` or nDCG drops > `eval.ndcg_tolerance`, measured against the committed baseline for a dataset. Implemented in `services/regression.py`; wired into CI on 2026-11-02 (TASK-0244) as the `eval-gate` job — see §2 gate 6 for exactly what that job measures, what it pins, and what it does not cover.

**The verdict is three-valued**, in the same vocabulary `scripts/compare_nightly.py` uses for the nightly benchmark: `within-tolerance`, `regressed`, `not-comparable`. A gate that reported green because it could not compare anything would be the vacuity defect this repository treats as a bug, so `not-comparable` is its own outcome and fails CI exactly as `regressed` does. Only the reason, and therefore the fix, differs.

**Comparability is an enumerated fingerprint, not the lockfile hash.** One hash was answering two questions. `index.lock` asks *"has the index drifted from the config that built it?"*, where any difference counts because the index really was built under the old config — `core/identity.retrieval_config_hash` still answers that, unchanged and deliberately broad. A baseline asks *"is this result comparable to the recorded one?"*, where only settings that change what retrieval returns count. Under the broad hash, adding a knob whose default preserved behaviour retired every recorded baseline, which is how `chunking.token_counter`, `chars_per_token`, `semantic_percentile`, `retrieval.bm25_k1` and `bm25_b` — plus the identity-version-3 change to the digest function itself — retired the 2026-08-02 handbook baseline.

`services/regression.py` therefore records `COMPARABILITY_FIELDS` as **values rather than a digest**, so a refusal names the field that moved; a field a baseline does not mention is read as that field's current default, which is what lets a behaviour-preserving addition leave a baseline alive. The two errors are not symmetric — too broad retires good baselines, which is annoying and visible; too narrow silently compares runs that are not comparable, which is dangerous and invisible — so the enumeration errs broad, every deliberate omission is listed in `BEHAVIOUR_NEUTRAL_FIELDS`, and `tests/unit/services/test_comparability.py` fails the build when a setting appears in neither list. `COMPARABILITY_VERSION` covers the one case the "absent means default" rule cannot: a new field whose default does *not* preserve behaviour retires every older baseline when the version is bumped.

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
| Configuration mutated under a built index | Drift is reported and the moved field is named; a legal edit the index does not depend on is **not** reported. |
| Network partition to a provider (vector DB and LLM legs, plus a name that cannot resolve) | The call ends on *our* configured timeout rather than hanging; the failure is typed and retryable, renders as a problem document with a trace id and never a generic 500; the breaker opens and admits one probe after its reset timeout. (Closing on a healed peer is asserted in `tests/unit/core/test_breaker.py`, not here — no chaos case heals the partition and drives a real call through it.) |

Every scenario is a script in the repo; observed behavior is recorded in [failure-modes.md](failure-modes.md). Reliability anyone can re-run.

Every row injects its fault at the closest honest code seam, so the whole table runs on any machine — the partition row included, which is why it uses a local socket that accepts the connection and then never writes, rather than a container or a firewall rule. The first five rows live in `tests/chaos/test_chaos.py`; the last two are large enough to own their modules, `tests/chaos/test_config_mutation.py` and `tests/chaos/test_network_partition.py`. `tests/chaos/test_reindex_under_load.py` adds the reindex-under-continuous-query-load drill as a fourth module (8 cases, 2026-11-02; all three TASK-0241 ✅). The two environmental rows — stop-Qdrant and disk-full — additionally have **real-fault variants** in `tests/chaos/test_real_faults.py`, marked `integration` because they need a Docker daemon and skip cleanly without one: a throwaway Qdrant really stopped with `docker stop`, and the real journal writing onto a tmpfs with a hard size limit. The seam-level cases prove how fasterRag responds to those conditions; the variants prove that what the operating system actually reports is what it responds to.

The config-mutation row is worded narrowly on purpose. Its mutation set is derived from the live schema rather than listed by hand, and it is exhaustive over **single-field** edits of the subset the lock hashes; multi-field edits are not enumerated, and drift arriving by a route other than configuration is sampled with one scenario each. It does not establish "zero false negatives" in general, and [differentiators.md](differentiators.md) D1 states the same bound.

One honesty note about the suite's own audit: `test_chaos.py` carries a self-audit case
asserting that the suite is complete against this table, and it enumerates the five
scenarios defined in that module only. It therefore does not yet see the two newer ones, so
it would not notice if they were deleted. The scenarios run and pass; it is the audit that
is behind, and extending it is an open task.

### 1.10 Mutation testing (mutmut) — **NOT BUILT** (TASK-0243)

*Specified:* sampled on the two places where a subtle bug silently ruins quality — **chunk-boundary logic** and **retrieval-scoring/fusion logic**. Surviving mutants are triaged as missing-test bugs in [todo.md](todo.md).

*As built:* `mutmut` is not a dependency and has no configuration; the `mutation` marker is declared and unused. Mutation checking is done **by hand** today — several slices record deliberately breaking the code and confirming exactly one test fails (TASK-0193, TASK-0208, TASK-0211, TASK-0229, TASK-0232) — which is the practice this layer would automate, not a substitute for it.

### 1.11 Live provider checks (credential-gated)

`tests/eval/test_live_providers.py`, under the `eval` marker. Every other provider test substitutes a fake client, which proves the adapter issues the request it intends to and proves nothing about whether the vendor accepts it. [integrations.md](integrations.md) and the README advertise OpenAI, Anthropic, Cohere, Ollama and any OpenAI-compatible endpoint; before this layer existed, exactly one of those — the OpenAI generator — had ever been exercised against the real API, by hand, once.

Each advertised provider gets the smallest call that can keep the promise: one short generation, or one short embedding, asserting a well-formed typed `Completion`/`EmbeddingResult` and the token accounting the adapter claims to populate. One full run with every credential present costs **eight requests** of a few dozen tokens each — five generations (OpenAI, Anthropic, Cohere, Ollama, `openai_compatible`) and three embeddings (OpenAI, Cohere, Ollama); the two Ollama checks go to a local server and cost nothing. The HuggingFace embedder is local and is covered against real weights by `tests/eval/test_real_models.py` instead.

**What this layer establishes, and what it does not.**

- A provider counts as verified **only** when its own check ran and recorded evidence of a response. Verification is derived from the recorded responses, not from a counter kept beside them, so no provider that was never called can be counted.
- **A green `pytest -m eval` run with no credentials present proves nothing about any provider.** Every check skips, and a skip is not a pass. This is the misreading the trap registry calls T12, and it is the shape of the failure that hit the Redis suite on 2026-10-26 — a probe weaker than the assertion it gated, reported as green.
- Two things make that misreading structurally hard. Each provider has its own test, so pytest's own `SKIPPED` line is the record and there is no aggregate that could launder it; and the run's summary test is green **only** when every provider was reached, skipping with the full per-provider table otherwise. Run with `-rs` to see it.
- A missing credential never fails the suite. The maintainer can run with one key, with all of them, or with none. A credential that is *present but rejected* does fail — that is a real finding, not an absence.
- Gating: hosted providers gate on their key variable (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `COHERE_API_KEY`); Ollama has no credential, so it gates on `OLLAMA_HOST` naming a running server. Gating Ollama on nothing would mean a connection attempt, and a failure rather than a skip, on every machine without it. `openai_compatible` gates on three variables — `FASTERRAG_LIVE_OPENAI_COMPATIBLE_URL`, `..._KEY` and `FASTERRAG_LIVE_MODEL_OPENAI_COMPATIBLE_LLM` — because the schema requires both a base URL and a key, and a gateway names its own models, so there is no default model id to guess; it skips rather than guessing one. A missing provider SDK skips the same way, naming the extra to install. Every skip reason names both the provider and the variable, so a skipped run is legible rather than invisible. **Status (2026-10-29): none of the eight live bodies has ever been executed against a real provider** — no credential has been available to any run, so a green `pytest -m eval` proves only that the gating held. The gating logic itself is pinned by `tests/eval/test_live_provider_gating.py`, which deliberately carries no `eval` marker so it runs in the `not integration and not eval` job — the job that cannot skip (T12).
- Model ids go stale. Each check's model can be overridden with `FASTERRAG_LIVE_MODEL_<PROVIDER>_<LLM|EMBEDDINGS>` so a retired id is a configuration change, not a code change.

**Known gap recorded by this layer rather than papered over:** the Cohere and Ollama embedding adapters surface no token usage, so their checks assert `total_tokens is None` — pinning the current behaviour instead of asserting an accounting that does not exist. Cohere does return billed units that `CohereEmbedder` discards.

## 2. CI quality gates (all blocking)

1. `ruff check` + `ruff format --check` clean.
2. `mypy --strict` zero errors.
3. Unit tests green; coverage ≥ 85% on `core/`, `adapters/`, `workers/`.
4. Integration tests green (testcontainers).
5. Adapter contract suite green for all in-repo adapters.
6. Eval harness runs (`pytest -m eval`, the `models` job). CI holds no provider credentials, so the live provider checks (§1.11) skip there and this gate verifies no provider against any vendor API. **The D7 tolerance gate is wired as its own `eval-gate` job** (TASK-0244, 2026-11-02) — see the paragraph below for precisely what it does and does not establish.
7. Secret scanning clean; dependency lockfile up to date ([security.md](security.md)).
8. Commit-message rule enforced (single line, no trailers) — locally by the `commit-msg` hook on every commit, and in CI by the `commit-messages` job, which runs **on pull requests only** because it checks a base..head range that a bare push does not supply.
9. Docs updated in the same change for any behavior change (reviewed, not automated).
10. The repository's own documents must not contradict the tree. Four scripts run in the `quality` job, each against a failure that has actually occurred here: `check_doc_truth.py` (`CLAUDE.md` or `README.md` claiming a documentation-only repository while `src/fasterrag` holds an implementation), `check_openapi_drift.py` (a route the app serves but [api-reference.md](api-reference.md) does not document, or the reverse), `check_blockers.py` ([blockers.md](blockers.md) citing a `TASK-` id [todo.md](todo.md) does not hold, or presenting a ticked task as outstanding), and `check_claims.py` (a document calling a capability unbuilt while the task it cites is ticked). CI-only, not commit hooks. Each is narrow by design and none of them reads prose for truth — see each script's module docstring for what it explicitly cannot catch.
11. `fasterrag doctor --json` runs against the CI environment itself, after `fasterrag config init` in an empty directory — dogfooding D10 and the packaged-template path on every push.
12. The fast suite runs on **windows-latest** as well as ubuntu. Two shipped defects were Windows-only (a path-case document id, a UTF-8 BOM the config loader rejected), and neither is reachable from an ubuntu-only matrix.
13. `scripts/stamp_release.py --check` (the `quality` job): the tree is wholly in the development state or wholly in the released one, so a half-applied version stamp cannot be committed ([release.md](release.md) §3 step 3).
14. **The built wheel installs and imports as a consumer receives it** — `scripts/check_package.py`, its own `packaging` job. It builds wheel and sdist, installs the wheel into a clean virtual environment, and asserts every symbol [python-api.md](python-api.md) calls SemVer-stable imports there, the `fasterrag` console script runs, the packaged `config.yaml` is byte-identical to the repository's, and `torch`/`transformers`/`sentence_transformers` are **absent** from a core install. No other gate can see any of this: `pythonpath = ["src", ...]` means the whole suite imports the checkout rather than the package, so a wheel missing a subpackage, the packaged template, or the entry point stays green everywhere else. First run end to end on 2026-10-29 — passed, ~95s on a Windows dev machine.

**On gate 3's scope.** The threshold is applied to `core/`, `adapters/`, and `workers/` specifically, not to the repository. A tree-wide number is diluted by CLI plumbing and API wiring, so it can sit comfortably above the line while the retrieval and adapter code the gate exists to protect regresses underneath it. Measured at **87.11% branch coverage over 3525 statements** (2026-08-02) when the gate was set at 85 to leave working room without letting it drift. **Re-measured at 83% over 4554 statements on a Windows dev machine (2026-08-10, release-readiness review)** — below the threshold CI enforces, with the reconciliation tracked as TASK-0252 in [todo.md](todo.md): either the Linux CI measurement differs by platform, CI on main is red, or the gate's scope drifted; whichever it is, the answer is restoring headroom, not lowering the gate.

**On gate 6's scope — what the `eval-gate` job establishes, and what it does not.** The job starts a Qdrant service container, ingests `tests/eval/datasets/handbook/corpus` with the real `BAAI/bge-small-en-v1.5` embedder, scores it with `fasterrag benchmark --suite eval`, and acts on the three-valued verdict: `within-tolerance` passes, `regressed` and `not-comparable` both fail, with the measured report printed either way. It does **not** read the exit code alone — code 5 cannot tell "recall fell" from "nothing was comparable", and those need different answers from a human.

Three settings are pinned rather than inherited, each of them a value `tests/eval/datasets/handbook/baseline.json` records field by field. `retrieval.rerank=false`, because the shipped default names a 2.2 GB cross-encoder and a push gate cannot download one per run — **so the gate covers retrieval and fusion and does not cover the reranker.** `retrieval.top_k=5`, because the baseline measured recall@5. `chunking.token_counter=estimate`, because `auto` uses the embedding model's tokenizer when it happens to be cached on the machine and silently estimates when it is not, which would make chunk boundaries depend on a runner's cache state. The gate therefore measures a pinned eval configuration and not the shipped defaults, and the `policies` dataset is not scored by it at all.

**Baseline provenance, stated exactly.** The recorded numbers are the 2026-08-02 measurement and have not been re-measured; no machine available to the 2026-11-02 change could run the harness end to end (no Docker daemon, so no Qdrant), and inventing numbers is not an option the provable-claims policy leaves open. What *was* recovered is the configuration behind them: the baseline carried only an opaque `retrieval_config_hash`, and a search over candidate configurations found the single one whose hash under that commit's digest function is the recorded `17c2897…` — `retrieval.top_k=5`, `retrieval.rerank=false`, otherwise that commit's defaults, with `token_counter` necessarily `estimate` because `core/chunking/tokenizer.py` did not yet exist. A 256-bit preimage match is proof, not inference, so the fingerprint now in the file is **derived rather than re-recorded**, and the file's `config_hash` is left at its original value as the provenance it came from. The `policies` baseline (`ae2c592…`) was recovered the same way and differs only in `retrieval.top_k`, which is why one pinned configuration cannot serve both datasets. The gate has never run on a runner, so whether current code still reproduces those numbers is unknown until its first CI run; a `regressed` verdict there is a finding to investigate, not a reason to relax the tolerance.

**`metadata` on a golden record is a retrieval *filter*, not annotation.** The harness passes it straight to `retrieve(filters=...)`, so a descriptive key that no chunk payload carries filters every candidate away and the run scores a flat `0.0` — indistinguishable from total retrieval failure. Keep it `{}` unless you are deliberately scoping the query.

## 3. Test taxonomy and markers

| Marker | Runs | Needs |
|---|---|---|
| (none) unit | every push, < 2 min | nothing external |
| `integration` | every PR | Docker (testcontainers) |
| `eval` | PRs touching retrieval paths + nightly | small datasets in repo; the live provider checks (§1.11) additionally need provider credentials and skip without them |
| `chaos` | runnable now against a local stack | Docker |
| `load`, `soak` | *planned* — no test carries either marker yet (TASK-0243); results would go to the ledger | reference hardware (TASK-0084) |
| `mutation` | *planned* — no test carries the marker; `mutmut` is not installed (TASK-0243) | — |

## 4. Definition of done for any implemented feature

Code with docstrings only (comment policy) · unit + integration tests green · coverage gate met on touched packages · typing and lint clean · benchmark ledger entry if performance-relevant · affected docs updated in the same change · [todo.md](todo.md) ticked with date · single-line commits.
