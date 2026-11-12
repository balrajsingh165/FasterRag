# CLI

Everything is under one command, `fasterrag`. Run `fasterrag <command> --help` for every flag.

## Everyday commands

| Command | What it does |
|---|---|
| `fasterrag config init [--profile local]` | write `config.yaml` and `.env.example` |
| `fasterrag config validate` | check the config and the env vars it names |
| `fasterrag config show [--changed]` | print every setting with its value and default |
| `fasterrag doctor [--fix]` | check Docker, ports, disk, keys and config; print a fix for each failure |
| `fasterrag provision qdrant` | start a managed Qdrant container (`--status`, `--down`) |
| `fasterrag estimate <paths> [--recursive]` | token count and projected cost before ingesting |
| `fasterrag ingest <paths> [--recursive] [--watch]` | parse, chunk, embed and index files or URLs |
| `fasterrag ingest --resume <job_id>` | continue an interrupted job from its checkpoint |
| `fasterrag query "<question>"` | answer a question with citations |
| `fasterrag serve [--host H] [--port P]` | run the REST API (needs the `server` extra) |
| `fasterrag status` | one-screen summary of the running system |

## Inspecting results

```bash
fasterrag query "..." --show-chunks --show-timings   # retrieved chunks and per-stage latency
fasterrag query "..." --filter department=legal     # metadata filter
fasterrag query "..." --top-k 5 --no-stream
fasterrag traces list                               # recent query traces
fasterrag traces show <trace_id>
fasterrag replay --trace <trace_id> --candidate new-config.yaml   # rerun a past query under another config
```

## Managing data

| Command | What it does |
|---|---|
| `fasterrag index list` | collections, vector counts and embedding model |
| `fasterrag index create <name>` / `index delete <name>` | create or drop a collection |
| `fasterrag index reembed <name> <paths>` | rebuild a collection beside the live one, then switch to it |
| `fasterrag index rollback <name>` | switch back to the previous build |
| `fasterrag documents list` | documents in a collection |
| `fasterrag documents delete <document_id>` | remove one document |
| `fasterrag backup <dir>` / `fasterrag restore <dir>` | back up or restore a deployment |
| `fasterrag export --out <file>` / `fasterrag import <file>` | move a collection between deployments |

## Options on every command

| Flag | Meaning |
|---|---|
| `--config PATH` | config file to load (default `./config.yaml`) |
| `--collection NAME` | collection to work on |
| `--set KEY=VALUE` | override one config key, repeatable |
| `--json` | machine-readable output |
| `--quiet` / `--verbose` | errors only / debug logging |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | success |
| `1` | the command failed |
| `2` | bad usage or invalid config |
| `3` | a backend could not be reached |
| `4` | a `doctor` check failed |
| `5` | the quality regression gate failed |

## Full reference

Every command and flag:
[cli-reference.md on GitHub](https://github.com/balrajsingh165/fasterrag/blob/main/docs/cli-reference.md).
