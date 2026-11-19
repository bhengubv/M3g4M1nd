# Tokenizer artefact

`tokenizer.json` here is a **placeholder**, trained on the 15-line fixture
corpus in `tests/fixtures/corpus.jsonl`. It has 733 entries, not the 6,400 the
model config reserves, because fifteen sentences cannot produce 6,400 useful
merges.

That is fine and deliberate: the extra embedding rows are simply unused, the
chat template is complete and tested, and every test in the suite runs against
this artefact without downloading anything. It is stated here rather than left
for someone to discover, because a vocabulary that silently fails to train is
exactly how MiniMind once shipped one with no merges at all.

Rebuild it from a real corpus with:

```python
from m3g4m1nd.tokenizer import train_tokenizer
train_tokenizer(texts, "model", vocab_size=6400)
```

`tokenizer_config.json` takes `model_max_length` from `M3Config`, so the two
cannot drift. `tests/test_tokenizer.py` asserts that, and asserts the trained
vocabulary fits inside the model's embedding.
