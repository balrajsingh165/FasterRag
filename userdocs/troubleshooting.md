# Troubleshooting

**Start with `fasterrag doctor`.** It checks Docker, ports, disk, keys and the config, and prints
a fix for every failure. When reporting a problem, include the `trace_id` and `code` from the
error and the output of `fasterrag status`.

## Setup

| Symptom | Fix |
|---|---|
| The process exits at start naming a config key | Fix that key, then run `fasterrag config validate`. |
| It exits naming an environment variable | Add the variable to `.env` (copy `.env.example` if you have none). |
| An error names a missing extra, or `fasterrag serve` exits with code 2 | Install it, for example `pip install -e ".[server]"`. |
| A config change does nothing | The config is read at start-up. Restart the process. |

## Vector database

| Symptom | Fix |
|---|---|
| `/readyz` returns 503 listing `vector_db` | Start the database (`fasterrag provision qdrant`); fasterRag reconnects on its own. |
| Works locally, fails from another machine | Qdrant needs both ports 6333 and 6334 reachable. |
| Data gone after a container restart on Windows/WSL | Use a named Docker volume (`vector_db.docker.volume`), not a bind mount. |
| Dimension mismatch on upsert | The collection was built with another embedding model. Rebuild with `fasterrag index reembed <collection> <paths>`. |
| `404 NOT_FOUND`: no collection named … | `fasterrag index list` shows what exists; ingesting into a name creates it. |

## Ingestion

| Symptom | Fix |
|---|---|
| Job finished `partial` | Some files failed. `GET /v1/ingest/{job_id}/documents?status=dead_lettered` gives a `reason_code` for each; fix them, then `POST /v1/ingest/{job_id}/retry-dlq`. |
| `429 QUEUE_FULL` | The pipeline is full. Retry after the `Retry-After` header, or raise `workers.queue_depth`. |
| Re-ingesting adds nothing | Unchanged files are skipped on purpose. Change the file, or rebuild with `index reembed`. |
| `PDF_ENCRYPTED` / `ENCODING_UNREADABLE` / `PARSE_TIMEOUT` | Remove the PDF password / convert the file to UTF-8 / raise `ingestion.parse_timeout_seconds`. |
| A job stayed `running` after a restart | It is marked failed within `reliability.stale_job_seconds`; continue it with `fasterrag ingest --resume <job_id>`. |

## Answers

| Symptom | Fix |
|---|---|
| Exact names or IDs are not found | Keep `retrieval.hybrid: true` (on by default). Turning it on for an existing collection needs `index reembed`. |
| Right document, wrong passage | Lower `chunking.chunk_size` or try `chunking.strategy: layout`, then rebuild. |
| `INSUFFICIENT_EVIDENCE` instead of an answer | `generation.grounded_or_refuse` refused weak evidence. Lower `generation.faithfulness_threshold` or improve retrieval. |
| Response has `degraded: true` | A component was down: `mode` is `hybrid_only` (reranker) or `extractive` (LLM). Fix it; recovery is automatic. |
| A stream ends without `done` | The answer is incomplete; read the `error` event. |
| `USAGE_ERROR ... has not been started` in Python | Call methods inside `async with FasterRag.from_config(...) as rag:` (or `with` for `fasterrag.sync`). |

## Full guide

Every known symptom, cause and fix:
[troubleshooting.md on GitHub](https://github.com/balrajsingh165/fasterrag/blob/main/docs/troubleshooting.md).
