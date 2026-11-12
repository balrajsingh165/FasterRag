# Quickstart

From nothing to an answered question. Pick the CLI, Python or REST path below; all three use the
same `config.yaml`.

## Install

fasterRag needs Python 3.12 or newer. It is not on PyPI yet, so install from a checkout:

```bash
git clone https://github.com/balrajsingh165/fasterrag
cd fasterrag
pip install --pre "fasterrag[all]"
```

`pip install --pre fasterrag` alone gives you the library and the CLI, with Markdown, text, CSV and JSON
parsing. Add only the extras you need:

| Extra | Adds |
|---|---|
| `server` | the REST API (`fasterrag serve`) |
| `qdrant`, `pgvector`, `chroma`, `milvus`, `weaviate` | that vector database |
| `huggingface` | local embeddings (the default) |
| `rerank` | the cross-encoder reranker (off by default; set `retrieval.rerank: true`) |
| `openai`, `anthropic`, `cohere`, `ollama` | that embedding or LLM provider |
| `pdf`, `docx`, `html`, `ocr` | those document formats (`ocr` also needs the `tesseract` binary) |
| `redis` | a shared cache and rate limit across replicas |
| `otel` | OpenTelemetry export |
| `all` | everything above |

Example: `pip install --pre "fasterrag[server,qdrant,huggingface,rerank,pdf,openai]"`. A missing extra is
reported by name when you use the feature that needs it.

## Try it with no service and no key

```bash
pip install --pre "fasterrag[chroma,huggingface]"
fasterrag config init --profile local
fasterrag ingest ./notes.md
```

The `local` profile stores vectors in an embedded Chroma directory and runs the embedding model
in-process (downloaded once on first use). Nothing has to be running and no `.env` is needed.
`fasterrag query` also needs an LLM: the profile points at a local Ollama with `llama3.1`.

## The full setup

**1. Create the config.**

```bash
fasterrag config init        # writes config.yaml and .env.example
cp .env.example .env         # put your API keys here
fasterrag config validate    # names the bad key if something is wrong
```

`config.yaml` holds settings and is safe to commit. `.env` holds secrets and is never committed;
the config only names the environment variable to read. See [Configuration](configuration.md).

**2. Check your machine.**

```bash
fasterrag doctor
```

Checks Docker, ports, disk, memory, provider keys and the config, and prints a fix for every
failure. `fasterrag doctor --fix` applies the repairs that are safe to apply.

**3. Start the vector database** (skip if you already run one, or use Chroma).

```bash
fasterrag provision qdrant
```

**4. Ingest and ask.**

=== "CLI"

    ```bash
    fasterrag estimate ./my-docs/ --recursive    # optional: token count and cost first
    fasterrag ingest ./my-docs/ --recursive
    fasterrag query "What does the contract say about termination?"
    ```

=== "Python"

    ```python
    import asyncio
    from fasterrag import FasterRag


    async def main() -> None:
        async with FasterRag.from_config("config.yaml") as rag:
            job = await rag.ingest(["./my-docs/"], recursive=True)
            print(job.status, job.counts)

            result = await rag.query("What does the contract say about termination?")
            print(result.answer)
            for c in result.citations:
                print(c.source, c.page, c.score)


    asyncio.run(main())
    ```

    For blocking code use `from fasterrag.sync import FasterRag` with a plain `with`.

=== "REST"

    ```bash
    pip install --pre "fasterrag[server]"
    fasterrag serve                  # http://127.0.0.1:8000

    curl -X POST localhost:8000/v1/ingest \
      -H 'Content-Type: application/json' \
      -d '{"sources":[{"type":"path","value":"/data/docs"}]}'

    curl -X POST localhost:8000/v1/query \
      -H 'Content-Type: application/json' \
      -d '{"query":"What does the contract say about termination?","stream":false}'
    ```

    `POST /v1/ingest` returns `202` with a `job_id`; poll `GET /v1/ingest/{job_id}` until it
    settles.

## Next

- Tune the settings that matter: [Configuration](configuration.md)
- Swap the database or model: [Providers & deployment](providers.md)
- Something failed: [Troubleshooting](troubleshooting.md)

Full walkthrough:
[quickstart.md on GitHub](https://github.com/balrajsingh165/fasterrag/blob/main/docs/quickstart.md).
