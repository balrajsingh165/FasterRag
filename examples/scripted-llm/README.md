# fasterrag-scripted-llm

A deterministic, offline stand-in LLM registered as `scripted` under the `fasterrag.llm`
entry-point group. It answers with the first sentence of the first context passage in the
prompt and cites that passage. CI uses it to run the generation examples with no API key and no
running model; see [../README.md](../README.md). It is not a model and must not be used to judge
answer quality.

```bash
pip install ./examples/scripted-llm
FASTERRAG_SET="llm.provider=scripted" python examples/ingest_and_ask.py
```
