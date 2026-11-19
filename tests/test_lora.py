"""Rows B6 and B7 — low-rank adaptation.

B7 is recovered from MiniMind PR #919 (`Afloat16`), Apache-2.0, deleted in
commit cc9629e one commit after the fix it guards was merged. See CREDITS.md.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from torch import nn

from m3g4m1nd.config import M3Config
from m3g4m1nd.lora import DEFAULT_TARGETS, apply_lora, load_lora, merge_lora, save_lora
from m3g4m1nd.model import M3ForCausalLM
from tests.conftest import tiny_config


class LinearModel(nn.Module):
    """A single square projection, named so the default target list finds it."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)


# ---------------------------------------------------------------- B6
# MiniMind selected targets by `in_features == out_features`, which at its own
# default shape reached 2 of 7 projections per layer and said nothing.

def test_all_seven_projections_are_adapted_by_default():
    model = M3ForCausalLM(tiny_config())
    adapted = apply_lora(model, rank=2)
    leaves = {name.split(".")[-1] for name in adapted}
    assert leaves == set(DEFAULT_TARGETS), f"missed {set(DEFAULT_TARGETS) - leaves}"
    assert len(adapted) == 7 * tiny_config().num_hidden_layers


def test_the_shape_heuristic_minimind_used_would_have_reached_only_two():
    """Nails the actual consequence, at the real published shape."""
    model = M3ForCausalLM(M3Config())
    square = {
        name.split(".")[-1]
        for name, m in model.named_modules()
        if isinstance(m, nn.Linear) and m.in_features == m.out_features
    }
    assert square == {"q_proj", "o_proj"}, (
        "if this changes, the point the test is making has changed too"
    )
    assert len(set(DEFAULT_TARGETS) - square) == 5


def test_targets_are_configurable_and_narrowing_them_works():
    model = M3ForCausalLM(tiny_config())
    adapted = apply_lora(model, rank=2, targets=("q_proj", "v_proj"))
    assert {n.split(".")[-1] for n in adapted} == {"q_proj", "v_proj"}


def test_an_impossible_target_list_raises_rather_than_doing_nothing():
    model = M3ForCausalLM(tiny_config())
    with pytest.raises(ValueError, match="no layers matched"):
        apply_lora(model, rank=2, targets=("not_a_projection",))


def test_a_fresh_adapter_does_not_change_the_output():
    torch.manual_seed(5)
    model = M3ForCausalLM(tiny_config()).eval()
    ids = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        before = model(ids).logits.clone()
    apply_lora(model, rank=2)
    with torch.no_grad():
        after = model(ids).logits
    torch.testing.assert_close(before, after)


def test_adapter_built_for_other_targets_is_refused_not_half_loaded():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "adapter.pth"
        narrow = M3ForCausalLM(tiny_config())
        apply_lora(narrow, rank=2, targets=("q_proj",))
        save_lora(narrow, path)

        wide = M3ForCausalLM(tiny_config())
        apply_lora(wide, rank=2)  # all seven
        with pytest.raises(ValueError, match="was built for targets"):
            load_lora(wide, path)


def test_adapter_of_a_different_rank_is_refused():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "adapter.pth"
        a = M3ForCausalLM(tiny_config())
        apply_lora(a, rank=2)
        save_lora(a, path)

        b = M3ForCausalLM(tiny_config())
        apply_lora(b, rank=4)
        with pytest.raises(ValueError, match="rank"):
            load_lora(b, path)


def test_round_trip_restores_the_adapter_exactly():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "adapter.pth"
        torch.manual_seed(9)
        src = M3ForCausalLM(tiny_config())
        apply_lora(src, rank=2)
        for _, m in src.named_modules():
            if hasattr(m, "lora"):
                m.lora.B.weight.data.normal_(std=0.01)
        save_lora(src, path)

        # Same seed, so the base weights are identical and any difference in
        # the output can only come from the adapter.
        torch.manual_seed(9)
        dst = M3ForCausalLM(tiny_config())
        apply_lora(dst, rank=2)
        load_lora(dst, path)

        ids = torch.randint(0, 64, (1, 5))
        with torch.no_grad():
            torch.testing.assert_close(src.eval()(ids).logits, dst.eval()(ids).logits)


# ---------------------------------------------------------------- B7
# Recovered: MiniMind PR #919. Accumulate in fp32, round once.

def test_merge_rounds_once_after_adding_delta():
    model = LinearModel()
    model.q_proj.weight.data.fill_(0.9997)
    apply_lora(model, rank=1, targets=("q_proj",))
    model.q_proj.lora.A.weight.data.fill_(1.0)
    model.q_proj.lora.B.weight.data.fill_(0.0002)

    with tempfile.TemporaryDirectory() as d:
        adapter, merged = Path(d) / "adapter.pth", Path(d) / "merged.pth"
        save_lora(model, adapter)
        state = torch.load(adapter, weights_only=True)
        expected = (
            model.q_proj.weight.float()
            + state["q_proj.lora.B.weight"].float() @ state["q_proj.lora.A.weight"].float()
        ).half()

        merge_lora(model, adapter, merged)
        actual = torch.load(merged, weights_only=True)

        torch.testing.assert_close(actual["q_proj.weight"], expected, rtol=0, atol=0)
        assert actual["q_proj.weight"].dtype == torch.float16
        assert not any(".lora." in key for key in actual)


def test_rounding_twice_would_have_lost_the_delta():
    """Shows the defect is real rather than theoretical, at these magnitudes."""
    base, delta = torch.tensor([0.9997]), torch.tensor([0.0002])
    once = (base + delta).half()
    twice = (base.half().float() + delta).half()
    assert once.item() != twice.item(), "pick magnitudes where the difference shows"


def test_merged_weights_reproduce_the_adapted_model():
    torch.manual_seed(13)
    model = M3ForCausalLM(tiny_config()).eval()
    apply_lora(model, rank=2)
    for _, m in model.named_modules():
        if hasattr(m, "lora"):
            m.lora.B.weight.data.normal_(std=0.02)

    ids = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        adapted = model(ids).logits.clone()

    with tempfile.TemporaryDirectory() as d:
        adapter, merged = Path(d) / "a.pth", Path(d) / "m.pth"
        save_lora(model, adapter)
        merge_lora(model, adapter, merged)

        plain = M3ForCausalLM(tiny_config()).eval()
        plain.load_state_dict(torch.load(merged, weights_only=True), strict=True)
        with torch.no_grad():
            folded = plain(ids).logits

    torch.testing.assert_close(adapted, folded, rtol=2e-3, atol=2e-3)
