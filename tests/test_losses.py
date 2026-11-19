"""Rows B1, B2, B5 and C3 at the loss level.

B5 is recovered from MiniMind PR #909 (`Afloat16`), Apache-2.0, deleted in
commit cc9629e. See CREDITS.md.

B1/B2 are the inert-brake defect. The clip itself is arithmetic and can be
checked here without a GPU or a training run: given a ratio, does the clip
engage, and does the function say so? MiniMind's clip never engaged and nothing
reported that, which is why `grpo_per_token_loss` returns the clip fraction as
a required second value rather than an optional metric.
"""

from __future__ import annotations

import math

import pytest
import torch

from m3g4m1nd.losses import distillation_loss, grpo_per_token_loss, masked_cross_entropy


# ---------------------------------------------------------------- B5
def test_all_ignored_labels_produce_differentiable_zero():
    student = torch.randn(2, 4, 8, requires_grad=True)
    teacher = torch.randn_like(student)
    valid = torch.zeros(2, 4, dtype=torch.bool)

    loss = distillation_loss(student[valid], teacher[valid], temperature=1.5)
    assert bool(loss.isfinite())
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(student.grad, torch.zeros_like(student))


def test_nonempty_identical_logits_keep_zero_loss():
    student = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
    loss = distillation_loss(student, student.detach().clone(), temperature=1.5)
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    loss.backward()
    assert bool(student.grad.isfinite().all())


def test_distillation_rejects_a_nonpositive_temperature():
    s = torch.randn(2, 4)
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError, match="temperature must be positive"):
            distillation_loss(s, s, temperature=bad)


def test_temperature_squared_keeps_the_gradient_scale():
    """Without the T^2 factor the gradient shrinks as T grows, which is the
    whole reason Hinton's formulation carries it."""
    grads = {}
    for t in (1.0, 4.0):
        student = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
        teacher = torch.tensor([[0.0, 1.0, 0.0]])
        distillation_loss(student, teacher, temperature=t).backward()
        grads[t] = student.grad.abs().sum().item()
    assert grads[4.0] > grads[1.0] / 4, "T^2 compensation appears to be missing"


# ---------------------------------------------------------------- B1 / B2
def _logps(ratio: float, rows: int = 2, tokens: int = 3):
    old = torch.zeros(rows, tokens)
    new = torch.full((rows, tokens), math.log(ratio))
    ref = torch.zeros(rows, tokens)
    return new, old, ref


def test_an_identical_policy_gives_ratio_one_and_no_clipping():
    """This is MiniMind's actual state: rollout and update use the same weights,
    so the ratio is 1 everywhere and the clip is mathematically inert."""
    new, old, ref = _logps(1.0)
    _, clipped = grpo_per_token_loss(
        new, old, ref, torch.tensor([1.0, -1.0]), beta=0.1, loss_type="grpo", epsilon=0.2
    )
    assert clipped.sum().item() == 0.0


@pytest.mark.parametrize("ratio,should_clip", [(1.0, False), (1.1, False), (1.5, True), (0.5, True)])
def test_grpo_clip_engages_exactly_when_the_ratio_leaves_the_band(ratio, should_clip):
    new, old, ref = _logps(ratio)
    _, clipped = grpo_per_token_loss(
        new, old, ref, torch.tensor([1.0, -1.0]), beta=0.0, loss_type="grpo", epsilon=0.2
    )
    assert bool(clipped.all().item()) is should_clip


def test_clipping_actually_bounds_the_loss_it_is_meant_to_bound():
    """Not just reported — the clipped loss must be smaller in magnitude than
    the unclipped one for a positive advantage and a runaway ratio."""
    new, old, ref = _logps(4.0)
    adv = torch.tensor([1.0, 1.0])
    clipped_loss, _ = grpo_per_token_loss(
        new, old, ref, adv, beta=0.0, loss_type="grpo", epsilon=0.2
    )
    unclipped = -(torch.exp(new - old) * adv.unsqueeze(1))
    assert clipped_loss.mean().item() > unclipped.mean().item()


def test_cispo_clips_from_above_only():
    for ratio, expect in ((6.0, True), (3.0, False), (0.1, False)):
        new, old, ref = _logps(ratio)
        _, clipped = grpo_per_token_loss(
            new, old, ref, torch.tensor([1.0, 1.0]), beta=0.0,
            loss_type="cispo", epsilon_high=5.0,
        )
        assert bool(clipped.all().item()) is expect


def test_an_unknown_loss_type_is_refused():
    new, old, ref = _logps(1.0)
    with pytest.raises(ValueError, match="unknown loss_type"):
        grpo_per_token_loss(new, old, ref, torch.tensor([0.0, 0.0]), beta=0.1, loss_type="ppo2")


def test_kl_penalty_is_zero_when_policy_equals_reference():
    new, old, ref = _logps(1.0)
    loss, _ = grpo_per_token_loss(
        new, old, ref, torch.zeros(2), beta=1.0, loss_type="grpo"
    )
    assert loss.abs().max().item() == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------- C3
def test_masked_cross_entropy_trains_only_supervised_positions():
    torch.manual_seed(2)
    logits = torch.randn(1, 5, 7, requires_grad=True)
    labels = torch.tensor([[-100, -100, 3, 4, -100]])

    loss = masked_cross_entropy(logits, labels)
    assert bool(loss.isfinite())
    loss.backward()

    grad = logits.grad[0]
    # Shift convention: logit t predicts label t+1. Labels 3 and 4 sit at
    # positions 2 and 3, so logits 1 and 2 carry gradient and nothing else does.
    assert grad[1].abs().sum() > 0 and grad[2].abs().sum() > 0
    assert grad[0].abs().sum() == 0 and grad[3].abs().sum() == 0 and grad[4].abs().sum() == 0


def test_masked_cross_entropy_on_a_fully_ignored_batch_is_finite():
    logits = torch.randn(1, 4, 5, requires_grad=True)
    loss = masked_cross_entropy(logits, torch.full((1, 4), -100))
    assert bool(loss.isfinite()) and loss.item() == 0.0
    loss.backward()
