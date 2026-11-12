# Examples

Small scripts that each show one thing, every one of them executed by CI's `examples` job on
every push. They use the Python API (`from fasterrag import FasterRag`) and the sample
documents in [`data/`](data/).

| Script | What it shows | Needs an LLM |
|---|---|---|
| [`retrieve_only.py`](retrieve_only.py) | Ingest a folder, then retrieve ranked chunks without generating | no |
| [`ingest_and_ask.py`](ingest_and_ask.py) | Ingest a folder and ask a question: answer, citations, trace id | yes |
| [`stream_answer.py`](stream_answer.py) | Stream an answer token by token through `query_stream` | yes |
| [`swap_provider.py`](swap_provider.py) | Run the same code against two vector databases by changing one config key | no |
| [`replay_trace.py`](replay_trace.py) | Replay a stored query under a changed configuration and print the diff | yes |
| [`plugin-vectordb/`](plugin-vectordb/) | A complete third-party vector-database plugin; see [docs/plugins.md](../docs/plugins.md) | no |
| [`scripted-llm/`](scripted-llm/) | The offline stand-in LLM CI uses for the generation scripts | no |

## Running them yourself

Every script reads `config.yaml` from the current directory (or the path given as its first
argument). The local profile needs no service and no key:

```bash
pip install "fasterrag[chroma,huggingface,ollama]"
mkdir rag-demo && cd rag-demo
fasterrag config init --profile local
python ../examples/retrieve_only.py
```

The first run downloads the embedding model (`BAAI/bge-small-en-v1.5`, about 130 MB). The three
scripts marked "needs an LLM" generate with the local profile's Ollama: install Ollama and run
`ollama pull llama3.1`, or point `llm` in `config.yaml` at any provider you have a key for.
`swap_provider.py` needs the sample plugin installed: `pip install ./examples/plugin-vectordb`.

## How CI runs them without a model or a key

The retrieval scripts run on the local profile exactly as you would: real embedded Chroma, the
real embedding model. Only the LLM is replaced, and it is replaced the way any provider is, by
configuration. [`scripted-llm/`](scripted-llm/) is a tiny package that registers an LLM named
`scripted` under the `fasterrag.llm` entry-point group. It answers extractively: the first
sentence of the first context passage in the prompt, cited with that passage's marker. So the
answer, its citation, the stream and the stored trace are all real pipeline output; only the
wording is not a model's. CI installs both sample packages and selects the double with the
documented environment override, leaving the scripts and `config.yaml` untouched:

```bash
pip install ./examples/plugin-vectordb ./examples/scripted-llm
export FASTERRAG_SET="llm.provider=scripted"
```

Each script exits non-zero when its result is not what it demonstrates (no chunk, no citation,
an empty stream), so a regression in the API it uses fails the job rather than printing
something wrong. The same job installs the sample plugin with pip and runs the shared
vector-database contract suite against it (`python -m pytest examples/plugin-vectordb/tests`).
