# Python package

`FasterRag` is the whole engine in one object, driven by the same `config.yaml` as the CLI and
the API.

## Async

```python
import asyncio
from fasterrag import FasterRag


async def main() -> None:
    async with FasterRag.from_config("config.yaml") as rag:
        job = await rag.ingest(["./my-docs/"], recursive=True)
        print(job.job_id, job.status, job.counts)

        result = await rag.query("What are the payment terms?", top_k=5)
        print(result.answer, result.trace_id)
        for c in result.citations:
            print(c.source, c.page, c.score)


asyncio.run(main())
```

## Blocking

```python
from fasterrag.sync import FasterRag

with FasterRag.from_config("config.yaml") as rag:
    rag.ingest(["./my-docs/"], recursive=True)
    print(rag.query("What are the payment terms?").answer)
```

Same methods, same arguments, no `await`. It works in notebooks and scripts alike.

## Methods

| Method | Returns |
|---|---|
| `ingest(sources, *, collection=None, metadata=None, recursive=False, idempotency_key=None)` | the finished job: `job_id`, `status`, `counts` |
| `query(text, *, collection=None, top_k=None, filters=None)` | an answer: `answer`, `citations`, `mode`, `degraded`, `truncated`, `trace_id` |
| `query_stream(text, ...)` | the answer as events (`meta`, `token`, `citations`, `usage`, `done`), each with `type` and `data` |
| `retrieve(text, *, collection=None, top_k=None, filters=None)` | ranked chunks: `text`, `source`, `document_id`, `final_rank` |
| `documents(collection=None, *, offset=0, limit=...)` | the documents in a collection |
| `delete_document(document_id, *, collection=None)` | removes one document |
| `collections()` / `create_collection(name)` / `drop_collection(name)` | collection management |
| `estimate(sources)` | token count and projected cost (not async) |
| `doctor()` | the same checks as `fasterrag doctor` |
| `replay(trace_id)` | reruns a stored query |

Build from a file with `FasterRag.from_config(path)`, or from settings you constructed in code
with `FasterRag.from_settings(settings)`. Call methods inside the `async with` / `with` block;
outside it they raise a `USAGE_ERROR`.

fasterRag's own errors are subclasses of `fasterrag.errors.FasterRagError` and carry the
same `code`, `retryable` and `trace_id` the REST API returns.

## Retrieval only

Bring your own LLM step:

```python
chunks = await rag.retrieve("termination clause", top_k=10, filters={"department": "legal"})
context = "\n\n".join(chunk.text for chunk in chunks)
```

## Full reference

Every class, field and the plugin interface for adding your own provider:
[python-api.md on GitHub](https://github.com/balrajsingh165/fasterrag/blob/main/docs/python-api.md).
