# fasterrag-scripted-llm

A deterministic, offline stand-in LLM registered as `scripted` under the `fasterrag.llm`
entry-point group. It answers with the first sentence of the first context passage in the
prompt and cites that passage. CI uses it to run the generation examples with no API key and no
running model; see [../README.md](../README.md). It is not a model and must not be used to judge
a model's answer quality.

It also answers the P3 faithfulness prompt: each sentence of the answer is a claim, supported
when its words occur contiguously in the context passages, and the reply is the JSON verdict
P3 requires. The nightly's generation gate (`fasterrag benchmark --suite eval --faithfulness`)
uses it so that gate is deterministic and needs no key: it measures whether the pipeline gets
the retrieved evidence into the answer and its citation, not how well a model writes.

```bash
pip install ./examples/scripted-llm
FASTERRAG_SET="llm.provider=scripted" python examples/ingest_and_ask.py
```
