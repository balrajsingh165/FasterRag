# Providers & deployment

Every provider is chosen in `config.yaml` and installed as an extra. Keys go in `.env`; the
config names the variable.

## Vector databases

| `vector_db.provider` | Extra | Notes |
|---|---|---|
| `qdrant` (default) | `qdrant` | `mode: docker` lets `fasterrag provision qdrant` run it; `mode: external` uses yours. Ports 6333 and 6334 must both be reachable. |
| `chroma` | `chroma` | embedded, no server and no key; data in `vector_db.chroma.path` |
| `pgvector` | `pgvector` | set `vector_db.pgvector.dsn_env` to the env var holding the PostgreSQL DSN |
| `milvus` | `milvus` | set `vector_db.port: 19530` and `mode: external` |
| `weaviate` | `weaviate` | set `vector_db.port: 8080`, `vector_db.grpc_port: 50051` and `mode: external` |

```yaml
vector_db:
  provider: qdrant
  mode: external
  host: 10.0.0.42
  api_key_env: QDRANT_API_KEY
```

## Embedding models

| `embeddings.provider` | Extra | Key |
|---|---|---|
| `huggingface` (default, runs locally) | `huggingface` | none |
| `openai` | `openai` | `OPENAI_API_KEY` |
| `cohere` | `cohere` | `COHERE_API_KEY` |
| `ollama` (local server) | `ollama` | none |

Changing the embedding model of a collection that already holds data needs a rebuild:
`fasterrag index reembed <collection> <paths>`.

## LLMs

| `llm.provider` | Extra | Key |
|---|---|---|
| `openai` (default) | `openai` | `OPENAI_API_KEY` |
| `anthropic` | `anthropic` | `ANTHROPIC_API_KEY` |
| `cohere` | `cohere` | `COHERE_API_KEY` |
| `ollama` (local) | `ollama` | none; `llm.base_url` names the server |
| `openai_compatible` | `openai` | set `llm.base_url`; works with vLLM, LM Studio, llama.cpp server, TGI and most gateways |

```yaml
llm:
  provider: anthropic
  model: claude-sonnet-5
  api_key_env: ANTHROPIC_API_KEY
```

For every option and the plugin interface for your own provider, see
[integrations.md on GitHub](https://github.com/balrajsingh165/fasterrag/blob/main/docs/integrations.md).

## Deployment

**One process is the whole service.** `fasterrag serve` answers the API and runs ingestion
jobs. Scale by running more replicas behind a load balancer; set `cache.backend: redis` and
`security.rate_limit_backend: redis` so they share a cache and a rate limit.

**Docker Compose.** The repository ships a `Dockerfile` and a `docker-compose.yml`:

```bash
docker compose --profile core --profile qdrant up -d
```

Sizing presets add resource limits:

```bash
docker compose -f docker-compose.yml -f deploy/compose.medium.yml --profile core up -d
```

Ports are bound to `127.0.0.1`. Before you expose the API:

- put a TLS-terminating reverse proxy in front of it;
- set `security.auth: true` and `FASTERRAG_API_KEY` in `.env`;
- set `ingestion.allowed_source_roots`;
- point the readiness probe at `/readyz`, not `/healthz`.

**Observability (optional).** `observability.otel: true` exports traces over OTLP (`otel` extra);
`observability.dashboard: true` turns on a read-only dashboard on port 8080;
`fasterrag provision langfuse` and `fasterrag provision grafana` start those stacks. `/metrics`
serves Prometheus metrics.

More: [deployment.md](https://github.com/balrajsingh165/fasterrag/blob/main/docs/deployment.md),
[security.md](https://github.com/balrajsingh165/fasterrag/blob/main/docs/security.md),
[observability.md](https://github.com/balrajsingh165/fasterrag/blob/main/docs/observability.md)
on GitHub.
