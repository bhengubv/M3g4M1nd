# Acceptance rows

**PASS or FAIL only** — there is no third colour, and "no measurement" is FAIL,
not pending.

## PASS 32/32

**189 test functions, 248 cases after parametrisation, all passing, 15.4 s on
CPU** — no dataset, no network, no GPU.

> **Correction, 19 Nov 2026.** This file and the build prompt both said "28
> rows" while listing 32 (A7 + B9 + C4 + D7 + E4 + F1). Earlier reports of
> "13/28" and "23/32" were the same work against two different denominators;
> 32 is right.

## The four kinds of red state

Not equivalent, so each row says which it had:

- **measured against MiniMind** — the assertion was run against their code and
  failed there.
- **own first attempt** — the test caught a real mistake here while it was
  being written. Fourteen rows did.
- **fix survives upstream, test deleted** — MiniMind still has the fix; only
  the regression test went missing. Nothing to show red today, and that is the
  exposure.
- **not yet built** — FAIL. None left.

## Rows

| # | Requirement | Red state | State |
|---|---|---|---|
| A1 | Counts exact: dense 63,912,192; MoE 198,416,640 / active 63,936,768 | own — asserted the *rounded* 198,420,480 as if exact | **PASS** |
| A2 | MoE load-balancing term correct at any top-k | MiniMind's formula shown wrong by `test_routing_fractions_sum_to_k_not_one_per_expert_axis` | **PASS** |
| A3 | Config rejects unknown and renamed fields | measured — rejected `rope_parameters` on first reload, catching the Transformers 5.x rename MiniMind handles by deleting the key on export | **PASS** |
| A4 | Position overflow raises with the limit named | **measured against MiniMind** — `RuntimeError: size of tensor a (129) must match tensor b (128)` | **PASS** |
| A5 | Boolean mask behaves as the integer mask, cached and uncached | fix survives upstream, test deleted (PR #907) | **PASS** |
| A6 | Greedy ignores temperature/top-k/top-p; repetition penalty still applies | fix survives upstream, test deleted (PR #912) | **PASS** |
| A7 | Both attention paths agree; a fully-masked row does not NaN | none — one shared additive bias makes them agree by construction | **PASS** |
| B1 | Inner iterations make the clip live; clip fraction always reported | measured — pass 0 has ratio exactly 1 and clip fraction 0 (MiniMind's permanent state); pass 1+ moves and the clip fires | **PASS** |
| B2 | A single inner iteration refuses to pretend the clip works | own | **PASS** |
| B3 | Agent RL trains on the stop token; the first-stop trim actually trims | **measured** — `test_minimind_masking_makes_the_trim_a_silent_no_op` rebuilds their mask line verbatim and shows both safeguards failing together | **PASS** |
| B4 | Value head's input is the body output bit-for-bit | own | **PASS** |
| B5 | Distillation on a fully-masked batch returns differentiable zero | fix survives upstream, test deleted (PR #909) | **PASS** |
| B6 | LoRA targets by name, all 7 projections, mismatched adapter refused | measured — at the published shape the only square projections are `{q_proj, o_proj}` | **PASS** |
| B7 | LoRA merge accumulates fp32, rounds once | fix survives upstream, test deleted (PR #919) | **PASS** |
| B8 | Resume reproduces an uninterrupted run | own — stopping exactly at an epoch end left `(epoch=0, batch=4)` instead of `(epoch=1, batch=0)`; it resumed correctly only because the skip loop happened to yield nothing | **PASS** |
| B9 | Planned optimizer steps equal steps taken | own | **PASS** |
| C1 | Rollout refuses a token/score mismatch; snapshot is a real copy | measured — short, long and empty replies all raise; the snapshot does not move when the policy trains | **PASS** |
| C2 | No dataset class accepts a parameter it does not apply | own — `inspect.cleandoc` mangled the source and the AST walk crashed; the control test written to prove the detector works is what caught it | **PASS** |
| C3 | Label alignment proved token by token | own | **PASS** |
| C4 | Chat template round-trips with tools and thinking tags | own | **PASS** |
| D1 | Binds loopback by default; public binding needs a flag and warns | own | **PASS** |
| D2 | Fields bounded; out of range is 422; `max_tokens` clamped to the window | own — 500 requested with 8 left generates 8 | **PASS** |
| D3 | Exception text never reaches the response body | own — the handler fired but `TestClient` re-raises by default, so the test now asserts what a real client receives | **PASS** |
| D4 | Concurrent requests serialised against the one model | own — 4 simultaneous requests, `max_concurrent == 1` | **PASS** |
| D5 | Synchronous inference does not block the event loop | fix survives upstream, test deleted (PR #905) | **PASS** |
| D6 | Streamed tool calls correct across chunk boundaries | **own, and the worst of them** — the port found two real streaming bugs: special tokens were skipped, so `<tool_call>` markup was deleted and raw JSON leaked as content; and token-by-token decoding split 杭州's UTF-8 bytes into U+FFFD. Exactly what the deleted test existed to catch | **PASS** |
| D7 | Every `torch.load` passes `weights_only=True`; torch floor ≥2.6 | own | **PASS** |
| E1 | Qwen3 export reads configured `head_dim`; round-trips | own — `rope_theta` is no longer an attribute on `Qwen3Config` in Transformers 5.x, so a test reading it by name passes vacuously on 4.x and failed loudly here. Export agrees with the source to **0.0** | **PASS** |
| E2 | requirements.txt honest both ways | measured — red on 7 declared-but-unimported pins. Five were satisfied by writing the code; `numpy` was removed and `tokenizers` removed then earned back | **PASS** |
| E3 | Tokenizer's declared maximum equals the model's | own | **PASS** |
| E4 | NOTICE and CREDITS name MiniMind, its licence, the commit, the contributors | own | **PASS** |
| F1 | Whole suite on CPU, no dataset, no network, under 120 s | own — the GPU scanner matched its own needle literals, the same self-reference trap that caught the skip-marker check | **PASS** |

## Built

```
m3g4m1nd/config.py      strict configuration; renames and unknowns raise
m3g4m1nd/model.py       model core; one mask serves both attention paths
m3g4m1nd/outputs.py     our own output container
m3g4m1nd/lora.py        name-based targets; adapters refuse to half-load
m3g4m1nd/losses.py      distillation, masked CE, GRPO/CISPO with clip fraction
m3g4m1nd/rl.py          value head normalised once; a clip that engages
m3g4m1nd/rollout.py     sampling that never fabricates; snapshots, not rebinds
m3g4m1nd/agent.py       trajectories where the stop token is trainable
m3g4m1nd/data.py        datasets with no decorative parameters
m3g4m1nd/tokenizer.py   byte-level BPE; chat template that round-trips
m3g4m1nd/train.py       exact resume; step counts planned and checked
m3g4m1nd/serve.py       loopback, bounded, opaque errors, serialised
m3g4m1nd/export.py      Qwen3 export verified numerically before it writes
```

## Gaps closed after 32/32

The 32 rows only ever covered the twelve audited defects and the six deleted
tests. They were never the whole of MiniMind. Closed since:

| Gap | Closed by | Found while closing it |
|---|---|---|
| `agent.py` modelled tool use against tools that could not execute | `tools.py` — AST arithmetic sandbox and a tool registry | mapped `^` to `*` so `3^2` gave 6, **and wrote a test expecting 6** — a passing test locking in a wrong answer; also blanket-stripped commas, turning `gcd(12,18)` into `gcd(1218)` |
| DPO absent entirely | `losses.dpo_loss`, taking four named log-probability tensors rather than one batch halved on `batch_size // 2` | — |
| PPO and GAE absent entirely | `losses.generalised_advantage_estimate`, `losses.ppo_losses`, `losses.whiten` | — |
| No reward functions | `rewards.py` — a `Rubric` reporting every component, with MiniMind PR #904's by-value ground-truth matching | default weights summed to 5.75 against a clip of 3.0, so a right answer and a wrong one **both saturated and ranked identically** — the mirror of the inert clip |
| No CLI | `__main__.py` — one entry point, subcommands, shared flags | export took `--hidden-size`/`--layers` and defaulted the rest, so any other shape failed with seven size mismatches; now `config_from_state_dict` recovers the shape from the checkpoint |
| `__init__.py` was 0 bytes | the public API, 40 names | — |
| README named 12 guard files, none of which existed | real names, plus `test_hygiene.py` checking every file the README names | — |
| CI step "Refuse a suite that lost tests" printed a count and exited 0 | honest title, `set -euo pipefail`; the floor lives in `test_suite_gate.py` | — |
| CI ran Python 3.12 while everything was verified on 3.14.7 | a matrix of both | — |

## Second round of gap closing

| Gap | Closed by | Found while closing it |
|---|---|---|
| `Trainer` had no DataLoader | optional workers/pinning over the same deterministic batch sampler, so exact resume survives | `pin_memory=True` on a CPU box earns a torch warning and nothing else; now recorded and not acted on |
| No mixed precision | `amp_dtype` with autocast; a GradScaler only for float16, since bfloat16 has the exponent range | — |
| No gradient checkpointing | `gradient_checkpointing_enable()`; identical gradients, and a KV cache alongside it is refused as a contradiction | — |
| No distributed training | `init_distributed()`, DDP wrapping, rank-sharded sampler. Verified at world size one on gloo; the sharding arithmetic is tested exactly | `broadcast_buffers` was renamed `forward_sync_buffers` in torch 2.15; the flag is now read from the signature |
| MoE export was `NotImplementedError` | `qwen3_moe_config` / `qwen3_moe_state_dict`, fusing per-expert layers into Qwen3Moe's stacked tensors — **0.0** difference | gate-before-up was adjudicated by the numeric check, not guessed; the other order is wrong by 1.4e-3. Qwen3Moe's `decoder_sparse_step` defaults to 0, which would have exported a dense model under a MoE name |
| HTTP weight push was `NotImplementedError` | atomic write plus `/update_weights_from_disk`, raising on refusal | — |
| `dpo`, `grpo`, `agent`, `distill` printed "not wired up" and exited 2 | `pipelines.py` — four stages over the shared losses, updater, rollout, rubric and masks | the inert-clip alarm cried wolf: at a tiny learning rate the measured ratio is 0 and the alarm fired on a healthy run. `clip_is_inert` is now structural (`inner_iters == 1`), knowable before a step runs |
| No web UI | one page served at `/` by the same app, using the same stream parser. No CDN, no second server, no second copy of the parser | — |
| **Nothing had ever been trained** | a real pretrain → SFT run on the fixture corpus: **536,896 parameters, loss 6.618 → 0.169**, perplexity 748 → 1.2 | **the worst find of the round.** `Trainer.save` writes a resume checkpoint; `--from-weights` passed it to `load_state_dict(strict=False)`, which matched no keys, reported nothing, and left the model random. An SFT run trained 30 epochs from scratch and printed "saved". Now `checkpoints.py` loads strictly and a checkpoint supplying nothing is an error at any strictness |
| | | and `--match-vocab` was honoured by the RL stages and ignored by pretrain and SFT — a 6,400-row embedding against a 733-token tokenizer |

## What this is not

Every row is met, which means every row that was written down is met. It does
not mean M3g4M1nd is finished:

Two gaps remain, and neither can be closed from here:

- **There is no useful model.** A model *has* now been trained end to end —
  536,896 parameters, loss 6.618 → 0.169 — but on fifteen sentences, which it
  memorised. Greedy decoding returns training lines verbatim. That proves the
  pipeline learns; it is not a language model. Closing this needs a real
  corpus, which is a download decision, not a code one.
- **The committed tokenizer is a 733-entry placeholder** from the same fifteen
  sentences, against 6,400 reserved rows. Same reason: it needs real text.
- **CI has never run.** No remote and no commits, so `.github/workflows/tests.yml`
  is unverified. Closing this needs a commit and a push, which is the user's
  call every time.

Everything else that was listed as missing is now built and tested.

A green suite total is still not a product claim. The 32 rows mean 32 things
were checked; the 497 tests mean 497 things were checked.
