# Recovered MiniMind tests — verbatim originals

Six regression tests that MiniMind contributors submitted alongside their bug
fixes, and which were then deleted from that repository:

- commit `cc9629e` (18 Nov 2026) — five tests, 156 lines, titled
  "[update] fix decoding, drop tests"
- commit `233d00f` (27 Oct 2026) — one test, 147 lines, titled
  "[update] trim merged pr leftovers"

They are kept here **unmodified**, as provenance and attribution. They import
MiniMind's own module paths, so they do not run against this repository.

The live versions are the ports in `tests/`, which assert the same behaviour
against M3g4M1nd:

| Original | Ported to |
|---|---|
| `test_attention_bool_mask.py` | `tests/test_recovered_attention_and_generation.py` |
| `test_generation_greedy_temperature.py` | `tests/test_recovered_attention_and_generation.py` |
| `test_distillation_empty_supervision.py` | `tests/test_losses.py` |
| `test_lora_merge_fp32.py` | `tests/test_lora.py` |
| `test_api_inference_event_loop.py` | `tests/test_serve.py` |
| `test_stream_tool_calls.py` | `tests/test_stream_tool_calls.py` |

Authorship and licence: Apache-2.0, MiniMind contributors. See `../../CREDITS.md`.

All six are ported. The only case not carried across is the original's
`test_real_text_streamer_and_openai_client`, which drives the server through
the `openai` client package; the wire behaviour it checks is covered by
`test_a_real_token_streamer_round_trips_through_the_tokenizer`, and adding a
dependency to assert it twice was not worth it. Said here rather than left to
be noticed.
