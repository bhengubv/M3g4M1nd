"""Rows A5 and A6 — recovered from MiniMind, ported to our names.

Provenance, stated plainly: in MiniMind both of these fixes are *still in the
code*. Only the tests were deleted (commit cc9629e, 18 Nov 2026). So there is
no red state to demonstrate against upstream today — the defect these guard
against is a future regression that nothing upstream would now catch.

That is the whole argument for keeping them. A fix without a test is a fix
waiting to be undone by the next refactor.

Original authorship: MiniMind PR #907 and #912 (`Afloat16`), Apache-2.0.
See CREDITS.md.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch

from tests.conftest import tiny_model


# ---------------------------------------------------------------- A5
# MiniMind PR #907: a boolean padding mask must behave exactly like the integer
# mask, in the manual attention path, with and without a KV cache.

def test_boolean_mask_matches_integer_mask():
    model = tiny_model()
    ids = torch.tensor([[0, 0, 3, 4], [0, 5, 6, 7]])
    mask = ids.ne(0)

    for cache in (False, True):
        with torch.no_grad():
            expected = model(ids, attention_mask=mask.long(), use_cache=cache)
            actual = model(ids, attention_mask=mask, use_cache=cache)
        torch.testing.assert_close(actual.logits, expected.logits)
        assert bool(actual.logits.isfinite().all())

        if cache:
            token = torch.tensor([[8], [9]])
            extended = torch.cat((mask, torch.ones(2, 1, dtype=torch.bool)), dim=1)
            with torch.no_grad():
                a = model(token, attention_mask=extended, past_key_values=actual.past_key_values)
                b = model(
                    token, attention_mask=extended.long(), past_key_values=expected.past_key_values
                )
            torch.testing.assert_close(a.logits, b.logits)


def test_boolean_mask_agrees_across_both_attention_paths():
    """Our addition: PR #907 only covered the manual path, because that is where
    MiniMind's bug was. Since we have one mask serving two paths, check both."""
    fused, manual = tiny_model(flash_attn=True), tiny_model(flash_attn=False)
    manual.load_state_dict(fused.state_dict())
    ids = torch.tensor([[0, 0, 3, 4], [0, 5, 6, 7]])
    with torch.no_grad():
        a = fused(ids, attention_mask=ids.ne(0)).logits
        b = manual(ids, attention_mask=ids.ne(0).long()).logits
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------- A6
# MiniMind PR #912: greedy decoding divided by temperature before checking
# whether it was sampling, so temperature=0 produced inf/NaN, and top-k/top-p
# silently reshaped greedy requests.

def _fixed_logits(**scores):
    logits = torch.zeros(1, 1, 64)
    for token, value in scores.items():
        logits[0, 0, int(token.lstrip("t"))] = value
    return logits


def test_greedy_ignores_temperature_and_sampling_filters():
    model = tiny_model()
    logits = _fixed_logits(t3=1.0, t7=2.0)
    for temperature in (0.0, -1.0, 0.7):
        with patch.object(
            model, "forward",
            return_value=SimpleNamespace(logits=logits.clone(), past_key_values=None),
        ):
            result = model.generate(
                torch.tensor([[4]]), max_new_tokens=1, do_sample=False,
                temperature=temperature, top_k=1000, top_p=0.0, eos_token_id=None,
            )
        assert result[0, -1].item() == 7, f"greedy went astray at temperature={temperature}"


def test_repetition_penalty_still_applies_to_greedy():
    """A repetition penalty is not a sampling control, so greedy still honours it.
    Token 7 scores higher but has been seen, so halving it hands the pick to 3."""
    model = tiny_model()
    logits = _fixed_logits(t3=1.5, t7=2.0)
    with patch.object(
        model, "forward", return_value=SimpleNamespace(logits=logits, past_key_values=None)
    ):
        result = model.generate(
            torch.tensor([[7]]), max_new_tokens=1, do_sample=False,
            temperature=0.0, repetition_penalty=2.0, eos_token_id=None,
        )
    assert result[0, -1].item() == 3


def test_sampling_with_a_degenerate_temperature_is_refused_up_front():
    """Our addition. MiniMind now raises for this too, but only after a year of
    returning NaN; the check belongs before the first forward pass, not inside it."""
    import pytest

    model = tiny_model()
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and positive"):
            model.generate(torch.tensor([[4]]), max_new_tokens=1, do_sample=True, temperature=bad)
