# Tokenizer

Byte-level BPE, **6,400 entries**, matching the model's reserved embedding rows
exactly. 465 KB.

Trained on 2026-11-20 from the first 200 MiB of MiniMind's
`pretrain_t2t_mini.jsonl` — 434,099 rows, 76.7M characters of mixed Chinese and
English — in 429 seconds on 4 CPU cores. The corpus itself is not committed
(`dataset/` is gitignored); it is a byte range of a 1.16 GB public dataset and
is re-fetchable.

Measured compression, characters per token:

| | |
|---|---|
| English | 2.53 |
| Chinese | 1.69 |
| Mixed | 2.14 |
| Code | 1.56 |

The Chinese figure is the useful check: MiniMind documents 1.5–1.7 chars/token
for Chinese with this vocabulary size, and an independently trained tokenizer
landing at 1.69 is evidence the merges are real rather than a vocabulary that
silently failed to train. `tests/test_tokenizer.py` asserts a floor on this,
because a BPE with no merges returns roughly 1.0 and reports success — which is
how MiniMind once shipped an empty one.

`tokenizer_config.json` takes `model_max_length` from `M3Config`, so the two
cannot drift.

## Rebuilding

```bash
python -m m3g4m1nd tokenizer --data dataset/corpus.jsonl --vocab-size 6400
```

Or from a sliced corpus, tolerating the truncated final record:

```python
from m3g4m1nd import train_tokenizer, texts_from_jsonl
train_tokenizer(texts_from_jsonl("dataset/slice.jsonl"), "model", vocab_size=6400)
```
