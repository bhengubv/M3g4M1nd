# Credits

We always give credit, even when we do not have to.

## MiniMind

M3g4M1nd exists because of **[MiniMind](https://github.com/jingyaogong/minimind)**
by **jingyaogong**, Apache-2.0. MiniMind is the best teaching text of its kind:
a complete language-model training pipeline in under 5,000 lines of readable
PyTorch, trainable from nothing on one consumer GPU. Every architectural idea
in this repository traces back to it.

M3g4M1nd is not a fork. It is a recreation written after reading MiniMind in
full at commit `cc9629e5f464ef46ef9411bfac916364e0a6dff1` (18 November 2026),
with the twelve defects that read found designed out and turned into tests.

Credit specifically owed:

- The 364-line model core, whose shape M3g4M1nd follows closely.
- The decision to publish failing results — MiniMind's own README prints its
  reinforcement-learning runs collapsing into repetition, and its cheapest
  model producing garbled English. Most projects crop that. jingyaogong did
  not, and that honesty is what made a useful audit possible at all.
- The `safe_math_eval` AST sandbox, which is the correct solution to letting a
  model evaluate arithmetic it wrote itself, and which M3g4M1nd reimplements
  rather than improves on.

## Recovered regression tests

Six tests in `tests/recovered/` were contributed to MiniMind by community
members alongside their bug fixes, and were subsequently deleted from that
repository. They are restored here, under Apache-2.0, because they are good
tests and because deleting them was the root cause of most of what we fixed.

| Test | Guards | Source |
|---|---|---|
| `test_attention_bool_mask.py` | boolean padding masks in manual attention | MiniMind PR #907 (`Afloat16`) |
| `test_api_inference_event_loop.py` | synchronous inference off the async event loop | MiniMind PR #905 (`Afloat16`) |
| `test_generation_greedy_temperature.py` | temperature ignored for greedy decoding | MiniMind PR #912 (`Afloat16`) |
| `test_distillation_empty_supervision.py` | finite loss on fully-masked distillation batches | MiniMind PR #909 (`Afloat16`) |
| `test_lora_merge_fp32.py` | LoRA deltas accumulated before rounding | MiniMind PR #919 (`Afloat16`) |
| `test_stream_tool_calls.py` | streamed tool calls across chunk boundaries | MiniMind PR #883 (`cyanseek`) |

If any attribution above is wrong, it is our error and we will correct it on
being told.

## Upstream

- **PyTorch** — BSD-3-Clause
- **Hugging Face Transformers / Tokenizers** — Apache-2.0
- **FastAPI / Starlette / Pydantic / Uvicorn** — MIT / BSD-3-Clause
- **MobileLLM** (Meta) — the small-model depth-over-width finding that sets our
  default shape. Paper: arXiv:2402.14905.
- **Qwen3** (Alibaba) — the configuration layout we export to, so our weights
  load in `llama.cpp`, `vllm` and `ollama` without custom code.
