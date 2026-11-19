# M3g4M1nd

A complete language-model training stack in a few thousand lines of readable
PyTorch. The name is the joke: it is tiny. **63,912,192 parameters** — about
1/2700th of GPT-3 — trainable from nothing on one consumer GPU.

Inspired by, and rebuilt from, [MiniMind](https://github.com/jingyaogong/minimind).
See [CREDITS.md](CREDITS.md) — that project is the reason this one exists.

## Why rebuild it

MiniMind was read in full on 19 November 2026. The read found twelve defects,
every one of which was invisible to a reader and visible only to a test. The
root cause was not any line of code: it was that MiniMind accepted community
bug fixes and then **deleted the regression tests that came with them** — six
tests, 303 lines, none needing a GPU, removed across two commits.

So M3g4M1nd inverts that. The audit *is* the test suite.

```
A test is never deleted, never skipped, never marked xfail.
If a test is wrong, fix the test and say why in the commit message.
If a feature is dropped, its test goes to tests/retired/ with a note.
```

The six deleted MiniMind tests are restored in `tests/recovered/` and run on
every commit.

## The twelve, and what each one cost

| Defect in MiniMind | What it caused | Guarded here by |
|---|---|---|
| Rollout engine rebound the live model, so the importance ratio was always 1 | The clip that stops a training step changing the model too violently never engaged. `epsilon` was decoration. | `test_rl.py` |
| Stop token excluded from the trained mask | Model could never learn when to stop talking — and a second safeguard that trims at the stop token found nothing to trim, becoming dead code | `test_agent.py` |
| Value head normalised its input twice | Distorted value estimates, silently | `test_rl.py` |
| Missing scores padded with "100% certain" | Invented numbers fed straight into the training maths | `test_rollout.py` |
| `max_seq_len` accepted and ignored after a rename | Four call sites tuning a dial wired to nothing | `test_model_core.py`, `test_tokenizer.py` |
| LoRA targets chosen by shape coincidence | Fine-tuned 2 of 7 projections; mismatched adapters loaded partially and silently | `test_lora.py` |
| Export hardcoded `head_dim` | Non-default head dimensions exported wrong | `test_export.py` |
| Mixture-of-experts balancing term kept a stray axis | Wrong for any top-k above 1 | `test_model_core.py` |
| Server bound `0.0.0.0`, unbounded request sizes, leaked exception text, no lock | Open to the network; one request could pin the GPU | `test_serve.py` |
| `torch.load` without `weights_only` | Downloaded weights could execute code | `test_hygiene.py` |
| `torch` commented out of requirements; 22 of 34 pins unused | Documented setup produced an environment that cannot start | `test_hygiene.py` |
| Tokenizer declared 131,072 against a 32,768 model | — | `test_tokenizer.py` |

Resume correctness, label alignment, the two attention paths and the chat
template round-trip were never tested in MiniMind at all. They are tested here.

`tests/test_hygiene.py` checks that every test file named in this table exists
— added after this table spent a day naming twelve files that did not.

## Install

Requires a Python with a `torch` wheel. Verified on **Python 3.14.7 / torch 2.15.0 (CPU)**.

```bash
py -3.14 -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python -m pip install -r requirements.txt
```

`torch` is in `requirements.txt`, with a floor of `>=2.6` — which is also the
version from which `torch.load` defaults to refusing to execute code.

## Test

The whole suite runs on **CPU, with no dataset, no GPU and no network**:

```bash
.venv/Scripts/python -m pytest
```

That constraint is deliberate. All six recovered MiniMind tests needed only a
CPU, which is exactly why deleting them was indefensible.

## Status

**32/32 acceptance rows. 497 tests, all passing, ~55 s on CPU** — no dataset,
no network, no GPU. `docs/ACCEPTANCE.md` holds every row, the defect it
guards, and which kind of red state it had before it went green.

A model has been trained end to end — 536,896 parameters, loss 6.618 → 0.169 —
but on the fifteen-sentence fixture corpus, which it memorised. The pipeline
learns; there is no useful model yet, and there will not be one without a real
corpus.

Two gaps remain and neither is code: a real training corpus, and a first
commit so CI can run. Everything else listed as missing is built and tested.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
