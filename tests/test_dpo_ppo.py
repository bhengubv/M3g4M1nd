"""DPO, GAE and the PPO losses — the two algorithms that were absent entirely.

MiniMind has both. M3g4M1nd had neither until now: `losses.py` carried GRPO and
CISPO only, so "parity minus the bugs" was overstated by two whole methods.
"""

from __future__ import annotations

import math

import pytest
import torch

from m3g4m1nd.losses import (
    dpo_loss,
    generalised_advantage_estimate as gae,
    ppo_losses,
    sequence_logps,
    whiten,
)


def logp(value: float, rows: int = 2) -> torch.Tensor:
    return torch.full((rows,), value)


# ------------------------------------------------------------------ DPO
def test_a_policy_matching_its_reference_sits_at_the_break_even_loss():
    """With no preference expressed either way the loss is -log(0.5)."""
    loss, accuracy = dpo_loss(logp(-2.0), logp(-3.0), logp(-2.0), logp(-3.0), beta=0.1)
    assert loss.item() == pytest.approx(-math.log(0.5), abs=1e-6)
    assert accuracy.item() == 0.0


def test_preferring_the_chosen_response_more_than_the_reference_lowers_the_loss():
    better, _ = dpo_loss(logp(-1.0), logp(-3.0), logp(-2.0), logp(-3.0), beta=0.1)
    worse, _ = dpo_loss(logp(-3.0), logp(-1.0), logp(-2.0), logp(-3.0), beta=0.1)
    assert better.item() < -math.log(0.5) < worse.item()


def test_accuracy_reports_the_ranking_separately_from_the_loss():
    """DPO loss falls whether or not the ranking improves, so the ranking is
    reported on its own."""
    _, accuracy = dpo_loss(
        torch.tensor([-1.0, -5.0]), torch.tensor([-3.0, -1.0]),
        torch.tensor([-2.0, -2.0]), torch.tensor([-3.0, -3.0]),
        beta=0.1,
    )
    assert accuracy.item() == pytest.approx(0.5)


def test_the_loss_is_monotonic_in_beta():
    losses = [dpo_loss(logp(-1.0), logp(-3.0), logp(-2.0), logp(-3.0), beta=b)[0].item()
              for b in (0.05, 0.1, 0.5)]
    assert losses == sorted(losses, reverse=True)


def test_dpo_is_differentiable():
    chosen = torch.tensor([-1.0], requires_grad=True)
    loss, _ = dpo_loss(chosen, logp(-3.0, 1), logp(-2.0, 1), logp(-3.0, 1))
    loss.backward()
    assert chosen.grad is not None and bool(chosen.grad.isfinite().all())


def test_a_nonpositive_beta_is_refused():
    for bad in (0.0, -0.1):
        with pytest.raises(ValueError, match="beta must be positive"):
            dpo_loss(logp(-1.0), logp(-2.0), logp(-1.0), logp(-2.0), beta=bad)


def test_mismatched_shapes_are_refused_rather_than_broadcast():
    """MiniMind cut one concatenated batch in half on `batch_size // 2`, which
    mispairs every example if the caller stacked them any other way. Four named
    arguments of one shape make that impossible."""
    with pytest.raises(ValueError, match="must have one shape"):
        dpo_loss(logp(-1.0, 2), logp(-2.0, 3), logp(-1.0, 2), logp(-2.0, 2))


def test_sequence_logps_sums_only_the_supervised_span():
    per_token = torch.tensor([[-1.0, -2.0, -3.0], [-1.0, -1.0, -1.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    assert sequence_logps(per_token, mask).tolist() == [-3.0, -2.0]


def test_sequence_logps_refuses_a_mask_of_the_wrong_shape():
    with pytest.raises(ValueError, match="must match"):
        sequence_logps(torch.zeros(2, 3), torch.ones(2, 4))


# ------------------------------------------------------------------ GAE
def test_a_single_terminal_reward_propagates_backwards():
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.zeros(1, 3)
    mask = torch.ones(1, 3)

    advantages, returns = gae(rewards, values, mask, gamma=1.0, lam=1.0)

    # With zero values and lam=1, every step sees the whole future reward.
    assert advantages.tolist() == [[1.0, 1.0, 1.0]]
    torch.testing.assert_close(returns, advantages)


def test_lambda_discounts_how_far_the_credit_reaches():
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.zeros(1, 3)
    mask = torch.ones(1, 3)

    full = gae(rewards, values, mask, gamma=1.0, lam=1.0)[0]
    partial = gae(rewards, values, mask, gamma=1.0, lam=0.5)[0]

    assert partial[0, 2].item() == pytest.approx(1.0)
    assert partial[0, 0].item() < full[0, 0].item()
    assert partial[0, 0].item() == pytest.approx(0.25)


def test_a_perfect_value_function_gives_no_advantage():
    rewards = torch.tensor([[0.0, 0.0, 1.0]])
    values = torch.tensor([[1.0, 1.0, 1.0]])
    advantages, _ = gae(rewards, values, torch.ones(1, 3), gamma=1.0, lam=1.0)
    assert advantages.abs().max().item() == pytest.approx(0.0, abs=1e-6)


def test_masked_positions_contribute_nothing_and_stop_the_trace():
    """The value past a row's own end is zero, not whatever padding holds."""
    rewards = torch.tensor([[0.0, 1.0, 0.0]])
    values = torch.tensor([[0.5, 0.5, 99.0]])        # junk in the padded slot
    mask = torch.tensor([[1.0, 1.0, 0.0]])

    advantages, returns = gae(rewards, values, mask, gamma=1.0, lam=1.0)

    assert advantages[0, 2].item() == 0.0
    assert returns[0, 2].item() == 0.0
    assert bool(advantages.isfinite().all())
    assert abs(advantages[0, 0].item()) < 10, "padding junk leaked into the estimate"


def test_rows_of_different_lengths_are_independent():
    rewards = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    values = torch.zeros(2, 3)
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])

    advantages, _ = gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert advantages.tolist() == [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]


@pytest.mark.parametrize("gamma,lam", [(-0.1, 0.95), (1.1, 0.95), (1.0, -0.1), (1.0, 1.1)])
def test_out_of_range_discounts_are_refused(gamma, lam):
    with pytest.raises(ValueError, match="must be in"):
        gae(torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(1, 2), gamma=gamma, lam=lam)


def test_gae_refuses_mismatched_shapes():
    with pytest.raises(ValueError, match="must all match"):
        gae(torch.zeros(1, 3), torch.zeros(1, 4), torch.ones(1, 3))


def test_whiten_centres_and_scales_over_live_positions_only():
    values = torch.tensor([[1.0, 3.0, 1000.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    whitened = whiten(values, mask)
    assert whitened[0, 2].item() == 0.0
    assert (whitened * mask).sum().item() == pytest.approx(0.0, abs=1e-5)


# ------------------------------------------------------------------ PPO
def _ppo(ratio: float, advantage: float = 1.0, clip: float = 0.2):
    logps = torch.full((1, 3), math.log(ratio))
    old = torch.zeros(1, 3)
    return ppo_losses(
        logps, old,
        values=torch.zeros(1, 3), old_values=torch.zeros(1, 3),
        advantages=torch.full((1, 3), advantage), returns=torch.zeros(1, 3),
        mask=torch.ones(1, 3), clip_epsilon=clip,
    )


def test_an_unmoved_policy_gives_no_clipping_and_no_kl():
    policy, value, clip_fraction, kl = _ppo(1.0)
    assert clip_fraction.item() == 0.0
    assert kl.item() == pytest.approx(0.0, abs=1e-9)
    assert policy.item() == pytest.approx(-1.0)


@pytest.mark.parametrize("ratio,clips", [(1.0, False), (1.1, False), (1.5, True), (0.5, True)])
def test_the_clip_engages_exactly_outside_the_band(ratio, clips):
    _, _, clip_fraction, _ = _ppo(ratio)
    assert (clip_fraction.item() > 0.0) is clips


def test_clipping_bounds_the_improvement_it_is_meant_to_bound():
    """A runaway ratio with a positive advantage must not buy unbounded gain."""
    bounded, _, _, _ = _ppo(4.0, advantage=1.0, clip=0.2)
    assert bounded.item() == pytest.approx(-1.2)


def test_a_negative_advantage_is_clipped_from_the_other_side():
    bounded, _, _, _ = _ppo(0.25, advantage=-1.0, clip=0.2)
    assert bounded.item() == pytest.approx(0.8)


def test_the_value_loss_is_clipped_against_the_old_estimate():
    logps = torch.zeros(1, 2)
    _, value_loss, _, _ = ppo_losses(
        logps, logps,
        values=torch.tensor([[5.0, 5.0]]),          # a wild move
        old_values=torch.tensor([[0.0, 0.0]]),
        advantages=torch.zeros(1, 2), returns=torch.tensor([[0.0, 0.0]]),
        mask=torch.ones(1, 2), clip_value=0.2,
    )
    # Unclipped would be 0.5 * 25 == 12.5; clipped takes the max of the two
    # squared errors, which is still the large one — the clip bounds the
    # *gradient path*, not the reported number.
    assert value_loss.item() == pytest.approx(12.5)


def test_masked_positions_do_not_contribute_to_either_loss():
    logps = torch.tensor([[0.0, 5.0]])
    old = torch.zeros(1, 2)
    mask = torch.tensor([[1.0, 0.0]])
    policy, _, clip_fraction, kl = ppo_losses(
        logps, old,
        values=torch.zeros(1, 2), old_values=torch.zeros(1, 2),
        advantages=torch.ones(1, 2), returns=torch.zeros(1, 2),
        mask=mask,
    )
    assert clip_fraction.item() == 0.0, "a masked position was counted as clipped"
    assert kl.item() == pytest.approx(0.0, abs=1e-9)
    assert policy.item() == pytest.approx(-1.0)


def test_approx_kl_grows_with_the_distance_moved():
    kls = [_ppo(r)[3].item() for r in (1.0, 1.2, 2.0)]
    assert kls == sorted(kls)
    assert kls[0] < kls[-1]


def test_the_ppo_losses_are_differentiable():
    logps = torch.zeros(1, 3, requires_grad=True)
    values = torch.zeros(1, 3, requires_grad=True)
    policy, value, _, _ = ppo_losses(
        logps, torch.zeros(1, 3), values, torch.zeros(1, 3),
        advantages=torch.ones(1, 3), returns=torch.ones(1, 3), mask=torch.ones(1, 3),
    )
    (policy + value).backward()
    assert bool(logps.grad.isfinite().all()) and bool(values.grad.isfinite().all())
