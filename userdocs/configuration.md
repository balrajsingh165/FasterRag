# Configuration

fasterRag reads two files:

- **`config.yaml`** — every setting. Safe to commit. Create it with `fasterrag config init`.
- **`.env`** — secrets only. Never commit it. The config never holds a key; it names the
  environment variable to read, for example `llm.api_key_env: OPENAI_API_KEY`.

The config is validated when it loads. A bad value stops the process and names the key.

```bash
fasterrag config validate          # check the file and the env vars it references
fasterrag config show --changed    # list only what differs from the defaults
fasterrag --set retrieval.top_k=5 query "..."   # override any key for one command
```

Use `--config path/to/config.yaml` to load a file other than `./config.yaml`. In containers,
the `FASTERRAG_SET` environment variable takes the same `key=value` overrides as `--set`.

## The keys most people change

### Pick your stack

```yaml
vector_db:
  provider: qdrant              # qdrant | pgvector | chroma | milvus | weaviate
  mode: docker                  # docker (fasterRag runs it) | external (you run it)
  host: localhost
  port: 6333
  api_key_env: QDRANT_API_KEY
  collection:
    default_name: default

embeddings:
  provider: huggingface         # huggingface | openai | cohere | ollama
  model: BAAI/bge-small-en-v1.5
  api_key_env: null             # e.g. OPENAI_API_KEY for a hosted model

llm:
  provider: openai              # openai | anthropic | cohere | ollama | openai_compatible
  model: gpt-4o-mini
  api_key_env: OPENAI_API_KEY
  base_url: null                # required for openai_compatible
  temperature: null
  max_tokens: 1024
```

Per-provider details are on [Providers & deployment](providers.md).

### Chunking and retrieval

| Key | Default | What it does |
|---|---|---|
| `chunking.strategy` | `recursive` | `recursive`, `fixed`, `layout`, `late` or `semantic` (`semantic` needs `embeddings.provider: huggingface`) |
| `chunking.chunk_size` | `768` | target chunk size in tokens (64–2500) |
| `chunking.overlap` | `64` | tokens shared between neighbouring chunks |
| `retrieval.top_k` | `10` | chunks passed to the answer (1–100) |
| `retrieval.hybrid` | `true` | run keyword (BM25) search beside vector search |
| `retrieval.rerank` | `true` | rerank candidates with a cross-encoder (needs the `rerank` extra) |
| `retrieval.reranker_model` | `BAAI/bge-reranker-v2-m3` | the cross-encoder to use |
| `retrieval.rerank_top_n` | `100` | how many candidates the reranker sees |

Changing `chunking.*`, the embedding model or `retrieval.hybrid` on a collection that already
holds data needs a rebuild: `fasterrag index reembed <collection> <sources>`.

### Answers

| Key | Default | What it does |
|---|---|---|
| `generation.citations` | `true` | attach source citations to every answer |
| `generation.grounded_or_refuse` | `false` | refuse to answer when the evidence is too weak |
| `generation.faithfulness_threshold` | `0.7` | how strict that refusal is |
| `generation.context_budget_tokens` | `4000` | most tokens of retrieved text sent to the LLM |
| `cache.semantic` | `false` | reuse answers for near-identical questions |

### Running the server

| Key | Default | What it does |
|---|---|---|
| `app.host` / `app.port` | `127.0.0.1` / `8000` | where `fasterrag serve` listens |
| `security.auth` | `false` | require `Authorization: Bearer <key>` on every request |
| `security.api_key_env` | `FASTERRAG_API_KEY` | env var holding the API key(s) |
| `security.rate_limit_per_minute` | `600` | per-key request limit when auth is on |
| `ingestion.allowed_source_roots` | `[]` | directories a `path` source may read from; empty means anywhere, and is refused when `security.auth` is on |
| `workers.cpu_pool_size` | `0` (one per core) | processes that parse and chunk |
| `cost.per_query_token_budget` | `0` (off) | refuse a query that would spend more tokens |

## Full reference

Every key, its type, range and default:
[config-reference.md on GitHub](https://github.com/balrajsingh165/fasterrag/blob/main/docs/config-reference.md).
Ready-made setups (fully local, cost-optimised, multi-tenant):
[cookbook.md](https://github.com/balrajsingh165/fasterrag/blob/main/docs/cookbook.md).
