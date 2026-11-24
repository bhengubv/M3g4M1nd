"""Rows B1, B2 and B4.

B1/B2 are the inert-brake defect, tested where it actually bites: across inner
iterations of a real optimiser on a real tiny model, not in the abstract.

B4 is the double-normalisation defect in the value head.
"""

from __future__ import annotations

import pytest
import torch

from m3g4m1nd.config import M3Config
from m3g4m1nd.model import M3ForCausalLM, M3Model
from m3g4m1nd.rl import GRPOUpdater, UpdateStats, ValueHead
from tests.conftest import tiny_config, tiny_model


# ---------------------------------------------------------------- B4
def test_value_head_sees_the_body_output_unnormalised_a_second_time():
    torch.manual_seed(4)
    config = tiny_config()
    critic = ValueHead(config).eval()
    ids = torch.randint(0, config.vocab_size, (2, 5))

    with torch.no_grad():
        from_body, _, _ = critic.body(ids)
        used_by_head = critic.hidden_states(ids)

    torch.testing.assert_close(used_by_head, from_body, rtol=0, atol=0)


def test_normalising_twice_would_have_changed_the_input():
    """Proves the defect is real at these weights rather than a no-op."""
    torch.manual_seed(4)
    critic = ValueHead(tiny_config()).eval()
    critic.body.norm.weight.data.normal_(mean=1.0, std=0.3)
    ids = torch.randint(0, 64, (2, 5))

    with torch.no_grad():
        once = critic.hidden_states(ids)
        twice = critic.body.norm(once)

    assert not torch.allclose(once, twice, rtol=1e-4, atol=1e-4), (
        "pick norm weights where the second application actually shows"
    )


def test_value_head_returns_one_scalar_per_position():
    config = tiny_config()
    critic = ValueHead(config).eval()
    ids = torch.randint(0, config.vocab_size, (3, 7))
    with torch.no_grad():
        values = critic(ids)
    assert values.shape == (3, 7)
    assert bool(values.isfinite().all())


def test_critic_can_share_a_body_with_the_policy_without_a_dangling_lm_head():
    config = tiny_config()
    body = M3Model(config)
    critic = ValueHead(config, body=body)
    assert critic.body is body
    assert not hasattr(critic, "lm_head")


# ---------------------------------------------------------------- B2
# Refusing to pretend. MiniMind ran one pass per rollout and still advertised
# epsilon as a working control.

def test_a_single_inner_iteration_is_refused_by_default():
    model = tiny_model()
    with pytest.raises(ValueError, match="mathematically inert"):
        GRPOUpdater(optimizer=torch.optim.SGD(model.parameters(), lr=1e-3), inner_iters=1)


def test_the_refusal_names_the_two_ways_out():
    model = tiny_model()
    with pytest.raises(ValueError) as caught:
        GRPOUpdater(optimizer=torch.optim.SGD(model.parameters(), lr=1e-3), inner_iters=1)
    message = str(caught.value)
    assert "raise inner_iters" in message and "allow_inert_clip=True" in message


def test_a_single_iteration_is_allowed_if_you_say_it_out_loud():
    model = tiny_model()
    with pytest.warns(UserWarning, match="clip is inert"):
        updater = GRPOUpdater(
            optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
            inner_iters=1, allow_inert_clip=True,
        )
    assert updater.inner_iters == 1


def test_zero_iterations_is_simply_wrong():
    model = tiny_model()
    with pytest.raises(ValueError, match="must be >= 1"):
        GRPOUpdater(optimizer=torch.optim.SGD(model.parameters(), lr=1e-3), inner_iters=0)


# ---------------------------------------------------------------- B1
def _rollout_batch(model, rows=4, length=6):
    torch.manual_seed(21)
    ids = torch.randint(3, 60, (rows, length))
    mask = torch.ones(rows, length - 1)
    with torch.no_grad():
        old = GRPOUpdater.token_logps(model, ids)
    rewards = torch.tensor([1.0, -1.0, 0.5, -0.5])
    advantages = GRPOUpdater.group_advantages(rewards, num_generations=2)
    return ids, mask, old, advantages


def test_first_pass_has_ratio_one_and_later_passes_do_not():
    """The heart of it. Pass 0 compares the policy with itself, so the ratio is
    1 and the clip cannot fire — which is MiniMind's permanent state. From pass
    1 the policy has moved and the ratio is real."""
    torch.manual_seed(21)
    model = tiny_model().train()
    reference = tiny_model().eval()
    reference.load_state_dict(model.state_dict())
    ids, mask, old, adv = _rollout_batch(model)

    updater = GRPOUpdater(
        optimizer=torch.optim.SGD(model.parameters(), lr=5.0),
        inner_iters=3, beta=0.0, epsilon=0.05,
    )
    stats = updater.step(model, reference, ids, mask, old, adv)

    assert stats[0].mean_abs_log_ratio == pytest.approx(0.0, abs=1e-6)
    assert stats[0].clip_fraction == 0.0
    assert stats[1].mean_abs_log_ratio > 0.0, "policy did not move between passes"


def test_the_clip_engages_once_the_policy_has_moved():
    torch.manual_seed(21)
    model = tiny_model().train()
    reference = tiny_model().eval()
    reference.load_state_dict(model.state_dict())
    ids, mask, old, adv = _rollout_batch(model)

    updater = GRPOUpdater(
        optimizer=torch.optim.SGD(model.parameters(), lr=20.0),
        inner_iters=4, beta=0.0, epsilon=0.02,
    )
    updater.step(model, reference, ids, mask, old, adv)

    assert updater.clip_ever_engaged, (
        "the clip never fired across 4 inner iterations with epsilon=0.02 — "
        "the brake is decoration, which is exactly the defect being guarded"
    )
    assert max(s.clip_fraction for s in updater.history) > 0.0


def test_a_tight_epsilon_clips_more_than_a_loose_one():
    def run(epsilon):
        torch.manual_seed(21)
        model = tiny_model().train()
        reference = tiny_model().eval()
        reference.load_state_dict(model.state_dict())
        ids, mask, old, adv = _rollout_batch(model)
        updater = GRPOUpdater(
            optimizer=torch.optim.SGD(model.parameters(), lr=20.0),
            inner_iters=4, beta=0.0, epsilon=epsilon,
        )
        updater.step(model, reference, ids, mask, old, adv)
        return max(s.clip_fraction for s in updater.history)

    assert run(0.02) >= run(0.5), "epsilon is not controlling anything"


def test_clip_fraction_is_reported_on_every_iteration():
    torch.manual_seed(21)
    model = tiny_model().train()
    reference = tiny_model().eval()
    reference.load_state_dict(model.state_dict())
    ids, mask, old, adv = _rollout_batch(model)

    updater = GRPOUpdater(
        optimizer=torch.optim.SGD(model.parameters(), lr=1.0), inner_iters=3, beta=0.1
    )
    stats = updater.step(model, reference, ids, mask, old, adv)

    assert len(stats) == 3
    assert all(isinstance(s, UpdateStats) for s in stats)
    for s in stats:
        assert 0.0 <= s.clip_fraction <= 1.0
        for value in (s.loss, s.mean_abs_log_ratio, s.kl_to_reference):
            assert value == value and abs(value) != float("inf")


def test_group_advantages_centre_within_each_group():
    rewards = torch.tensor([2.0, 0.0, 10.0, 10.0])
    adv = GRPOUpdater.group_advantages(rewards, num_generations=2)
    assert adv[0].item() > 0 > adv[1].item()
    # A group with no spread yields no signal rather than a division blow-up.
    assert abs(adv[2].item()) < 1e-3 and abs(adv[3].item()) < 1e-3


def test_group_advantages_refuse_a_ragged_batch():
    with pytest.raises(ValueError, match="do not divide into groups"):
        GRPOUpdater.group_advantages(torch.zeros(5), num_generations=2)


# ------------------------------------------------- telling the two zeros apart
def _run(lr, epsilon, inner_iters=3):
    torch.manual_seed(21)
    model = tiny_model().train()
    reference = tiny_model().eval()
    reference.load_state_dict(model.state_dict())
    ids, mask, old, adv = _rollout_batch(model)
    updater = GRPOUpdater(
        optimizer=torch.optim.SGD(model.parameters(), lr=lr),
        inner_iters=inner_iters, beta=0.0, epsilon=epsilon,
    )
    updater.step(model, reference, ids, mask, old, adv)
    return updater


def test_a_tiny_learning_rate_leaves_the_clip_unfired_but_not_inert():
    """The case that made the diagnostic cry wolf: the policy barely moved, so
    the clip was never needed. That is not the defect."""
    updater = _run(lr=1e-9, epsilon=0.2)
    assert not updater.clip_ever_engaged

    # The ratio moved by an amount at or below float32 resolution. Whether
    # that measures as exactly 0 or as 2.4e-08 is the platform's arithmetic,
    # not a property of this code: Windows gave 0 and Linux gave 2.4e-08, and
    # the first version of this test asserted the Windows value and failed CI.
    # What matters is that the movement is negligible.
    assert max(s.mean_abs_log_ratio for s in updater.history) < 1e-6

    assert not updater.clip_is_inert, "a healthy run must not raise the alarm"


def test_a_single_pass_is_the_case_that_is_genuinely_inert():
    """One inner iteration: the ratio is identically 1 and cannot move, so the
    clip is incapable of firing rather than merely idle."""
    torch.manual_seed(21)
    model = tiny_model().train()
    reference = tiny_model().eval()
    reference.load_state_dict(model.state_dict())
    ids, mask, old, adv = _rollout_batch(model)

    with pytest.warns(UserWarning):
        updater = GRPOUpdater(
            optimizer=torch.optim.SGD(model.parameters(), lr=1.0),
            inner_iters=1, allow_inert_clip=True, beta=0.0,
        )
    updater.step(model, reference, ids, mask, old, adv)

    # One pass recomputes the log-probabilities from the same weights that drew
    # the samples, so the ratio is 1 to within arithmetic noise on any platform.
    assert max(s.mean_abs_log_ratio for s in updater.history) < 1e-6
    assert updater.clip_is_inert, "this is the condition worth alarming on"
    assert not updater.clip_ever_engaged


def test_inertness_is_knowable_before_a_single_step_runs():
    """Structural, not measured: it depends on inner_iters alone."""
    model = tiny_model()
    with pytest.warns(UserWarning):
        one = GRPOUpdater(optimizer=torch.optim.SGD(model.parameters(), lr=1.0),
                          inner_iters=1, allow_inert_clip=True)
    many = GRPOUpdater(optimizer=torch.optim.SGD(model.parameters(), lr=1.0),
                       inner_iters=2)
    assert one.clip_is_inert and not many.clip_is_inert
    assert one.history == [] and many.history == []


def test_a_large_learning_rate_fires_the_clip_and_is_not_inert():
    updater = _run(lr=20.0, epsilon=0.02, inner_iters=4)
    assert updater.clip_ever_engaged
    assert updater.ratio_ever_moved
    assert not updater.clip_is_inert


def test_a_default_updater_is_not_inert():
    model = tiny_model()
    updater = GRPOUpdater(optimizer=torch.optim.SGD(model.parameters(), lr=1e-3))
    assert updater.inner_iters >= 2
    assert not updater.clip_is_inert


def test_the_alarm_does_not_depend_on_platform_float_behaviour():
    """Regression for the first CI failure this repository ever had.

    `ratio_ever_moved` is a measurement and measurements differ between
    platforms: at lr=1e-9 Windows reported a log-ratio of exactly 0 and Linux
    reported 2.4e-08. A test asserting either value pins one machine's
    arithmetic as a property of the code. `clip_is_inert` is structural and
    identical everywhere, which is why the alarm is built on it.
    """
    model = tiny_model()
    with pytest.warns(UserWarning):
        inert = GRPOUpdater(
            optimizer=torch.optim.SGD(model.parameters(), lr=1e-3),
            inner_iters=1, allow_inert_clip=True,
        )
    live = GRPOUpdater(optimizer=torch.optim.SGD(model.parameters(), lr=1e-3), inner_iters=2)

    # No history, no arithmetic, no platform dependence — and still a verdict.
    assert inert.history == [] and live.history == []
    assert inert.clip_is_inert is True
    assert live.clip_is_inert is False
