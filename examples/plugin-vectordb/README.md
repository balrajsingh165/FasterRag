# fasterrag-memory-vectordb

A complete fasterRag plugin: an in-process vector database registered as `memory`. It is the
worked example for [docs/plugins.md](../../docs/plugins.md), and CI installs it with pip and runs
fasterRag's shared vector-database contract suite against it on every push.

```bash
pip install ./examples/plugin-vectordb
```

```yaml
# config.yaml
vector_db:
  provider: memory
  api_key_env: null   # a plugin's key name is always written out, here "needs none"
```

The data lives in the process's memory and is gone when it exits, so this store suits tests and
demonstrations, not deployments.

| File | Role |
|---|---|
| `pyproject.toml` | The `fasterrag.vectordb` entry point: `memory = "fasterrag_memory_vectordb:MemoryVectorDB"` |
| `src/fasterrag_memory_vectordb/adapter.py` | `MemoryVectorDB(VectorDBAdapter)`, declaring `adapter_api_version = "1.0"` |
| `tests/test_contract.py` | The unmodified contract suite, against the adapter as installed |

Run the contract suite from a fasterRag checkout, where `tests/contract/vectordb.py` lives:

```bash
python -m pytest examples/plugin-vectordb/tests
```
