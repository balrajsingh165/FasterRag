# cli-reference.md — Terminal Commands

The CLI is one half of the control plane (the other is the [REST API](api-reference.md)); both call the same service layer, so behavior is identical. Root command: `fasterrag`.

Global flags (valid on every command):

| Flag | Default | Description |
|---|---|---|
| `--config PATH` | `./config.yaml` | Config file to load. |
| `--collection NAME` | config default | Target collection. |
| `--json` | off | Machine-readable JSON output (stable schemas; intended for scripts and CI). |
| `--quiet` / `-q` | off | Errors only. |
| `--verbose` / `-v` | off | Debug logging. |

Exit codes: `0` success · `1` generic failure · `2` usage/validation error · `3` dependency unreachable · `4` doctor/preflight failure · `5` regression gate blocked.

---

## `fasterrag serve`

Run the API server (`app.host`/`app.port` from config).

| Flag | Description |
|---|---|
| `--reload` | Dev auto-reload. |
| `--host`, `--port` | Override config bind address. |

## `fasterrag worker`

Run the pipeline worker pools (CPU pool + embedding pool + indexer) for the configured queues.

| Flag | Description |
|---|---|
| `--pools cpu,embed,index` | Restrict which pools this process runs (default: all). |
| `--cpu-workers N`, `--embed-workers N` | Override `workers.*` sizes. |

## `fasterrag ingest <path|url> [...]`

Submit sources for ingestion (async job; same path as `POST /v1/ingest`).

| Flag | Description |
|---|---|
| `--metadata KEY=VALUE` | Repeatable; merged into chunk metadata. |
| `--priority-class NAME` | Tiered-embedding routing class (D9). |
| `--recursive` | Recurse into directories. |
| `--watch` | Follow the job and print per-stage progress until completion. |
| `--dry-run` | Parse + chunk only; report what would be indexed (no embedding, no writes). |

## `fasterrag query "<question>"`

Run a query (same path as `POST /v1/query`).

| Flag | Description |
|---|---|
| `--top-k N` | Override `retrieval.top_k`. |
| `--filter KEY=VALUE` | Repeatable metadata filter. |
| `--no-stream` | Wait for the full answer instead of streaming tokens. |
| `--show-chunks` | Print retrieved chunks with scores (retrieval debugging). |
| `--show-timings` | Print per-stage latency breakdown. |

## `fasterrag index <subcommand>`

| Subcommand | Description |
|---|---|
| `list` | Collections with vector counts, embedding model+version, drift status. |
| `create NAME` | Create a collection (`--distance`, `--shards`, `--replicas`). |
| `delete NAME` | Drop a collection (`--force` required if alias target). |
| `reembed NAME` | D2 zero-downtime re-embed: blue/green build → eval gate → alias swap (`--no-eval-gate` for dev; `--watch`). |
| `rollback NAME` | Alias flip back to the retained previous collection (within `index.reindex.rollback_retention_hours`). |
| `lock verify [NAME]` | D1: verify `index.lock` against current config/corpus; report any drift (model version, config hash, content hashes). Exit 1 on drift. |

## `fasterrag provision <tool>`

Config-driven provisioning for `qdrant`, `langfuse`, `grafana` (idempotent; **doctor-gated**; zero code changes). Prints the running URL on success (Langfuse: `http://<host>:3000`).

| Flag | Description |
|---|---|
| `--status` | Show provisioning/health state instead of provisioning. |
| `--down` | Stop the managed containers for the tool (data volumes preserved). |

## `fasterrag status`

One-screen system status: API reachability, worker pools, queue depths, DLQ depth, vector DB health per adapter, circuit-breaker states, cache hit rates, active jobs.

## `fasterrag doctor`

D10 preflight diagnostics. Checks: Docker present and running · required ports free (API, dashboard, 6333 **and** 6334 for Qdrant, Langfuse stack ports) · disk space · RAM/GPU availability · vector DB reachable in the configured mode (docker / external local / remote host:port) · API keys valid (non-destructive provider ping) · config schema valid. **Every failed check prints a concrete fix-it instruction.** `doctor` must pass before any auto-provisioning runs.

| Flag | Description |
|---|---|
| `--fix` | Apply safe automatic fixes (e.g. create missing named volume). |
| `--json` | Machine-readable report (same schema as `GET /v1/admin/doctor`). |

## `fasterrag estimate <path|url> [...]`

D9 preflight cost estimator — BEFORE ingestion: document/token counts, projected embedding cost per configured provider, projected wall-clock time given current worker config.

| Flag | Description |
|---|---|
| `--provider NAME` | Compare against a specific provider instead of the configured one. |
| `--all-providers` | Cost table across all configured/known providers. |

## `fasterrag replay --trace <id>`

D8 time-travel replay: re-execute a past query under a candidate config and show a side-by-side diff of retrieval sets and answers.

| Flag | Description |
|---|---|
| `--config candidate.yaml` | Candidate config to replay under (required). |
| `--diff-only` | Only show what changed (added/removed/reordered chunks, answer diff). |

## `fasterrag benchmark`

Run the benchmark suite from [performance.md](performance.md) and print/append ledger-formatted results ([benchmarks.md](benchmarks.md)).

| Flag | Description |
|---|---|
| `--suite ingest\|query\|eval\|all` | Which suite (default `all`). |
| `--dataset NAME` | Named dataset fixture. |
| `--ledger` | Emit a ready-to-commit benchmark ledger entry (includes hardware fingerprint + commit hash). |

## `fasterrag export --out <archive>`

D11 portability: export documents, chunks, metadata, and the index manifest to a portable archive (vendor-neutral format).

| Flag | Description |
|---|---|
| `--include-vectors` | Include raw vectors (enables direct vector copy on import where dimensions match). |

## `fasterrag import <archive>`

D11: import a previously exported archive.

| Flag | Description |
|---|---|
| `--reembed` | Re-embed with the currently configured model instead of copying vectors. |
| `--target-collection NAME` | Import into a specific collection. |

## `fasterrag config validate`

Validate `config.yaml` + `.env` presence of referenced env vars without starting anything. Exit 0/2. Same fail-fast checks as startup ([config-reference.md](config-reference.md) cross-field rules).

## `fasterrag autopilot run`

D6 eval-driven auto-tuning (requires `autopilot.enabled: true`). Generates a golden Q&A set from the corpus, searches chunk size / top_k / hybrid weights / rerank settings, and writes a **suggested config diff with measured deltas** (e.g. recall@10 before/after) to stdout and `autopilot-suggestion.yaml`. **Never applies changes** — a human reviews and applies the diff.

| Flag | Description |
|---|---|
| `--budget-minutes N` | Search time budget. |
| `--golden-set PATH` | Reuse an existing golden set instead of generating one. |
