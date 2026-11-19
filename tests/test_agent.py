"""Row B3 — the stop token is trainable, and the trim that depends on it works.

The pair of assertions MiniMind could not have passed, because its mask erased
the thing its trim was looking for.
"""

from __future__ import annotations

import pytest
import torch

from m3g4m1nd.agent import (
    GENERATED,
    OBSERVED,
    Turn,
    pad_trajectories,
    stop_is_trainable,
    stop_is_trainable as _stop_is_trainable,
    trajectory_mask,
    trim_after_first_stop,
)

STOP = 2


# ---------------------------------------------------------------- B3, first half
def test_the_stop_token_carries_loss():
    tokens, mask = trajectory_mask([Turn(generated=[7, 8, STOP])], prompt_len=3)
    ids = torch.tensor([[1, 1, 1] + tokens])
    loss_mask = torch.tensor([mask])

    assert stop_is_trainable(ids, loss_mask, STOP), (
        "the model is never scored on its own stop token, so it cannot learn to stop"
    )


def test_observations_do_not_carry_loss():
    tokens, mask = trajectory_mask(
        [Turn(generated=[7, 8], observation=[50, 51, 52])], prompt_len=2
    )
    assert mask == [OBSERVED, OBSERVED, GENERATED, GENERATED, OBSERVED, OBSERVED, OBSERVED]
    assert sum(mask) == 2


def test_a_stop_token_inside_an_observation_is_not_trainable():
    """A tool result that happens to contain the stop id must not be scored."""
    tokens, mask = trajectory_mask([Turn(generated=[7], observation=[STOP, 9])], prompt_len=0)
    ids = torch.tensor([tokens])
    loss_mask = torch.tensor([mask])
    assert not stop_is_trainable(ids, loss_mask, STOP)


def test_the_prompt_never_carries_loss():
    _, mask = trajectory_mask([Turn(generated=[5])], prompt_len=4)
    assert mask[:4] == [OBSERVED] * 4


def test_multiple_turns_each_contribute_their_own_generated_span():
    tokens, mask = trajectory_mask(
        [
            Turn(generated=[7, 8], observation=[60]),
            Turn(generated=[9, STOP]),
        ],
        prompt_len=1,
    )
    assert tokens == [7, 8, 60, 9, STOP]
    assert mask == [OBSERVED, GENERATED, GENERATED, OBSERVED, GENERATED, GENERATED]


def test_an_empty_turn_is_refused():
    with pytest.raises(ValueError, match="not a turn"):
        Turn(generated=[])


# ---------------------------------------------------------------- B3, second half
def test_the_trim_actually_trims():
    ids = torch.tensor([[7, 8, STOP, 9, 10]])
    mask = torch.tensor([[1, 1, 1, 1, 1]])

    trimmed = trim_after_first_stop(ids, mask, STOP)

    assert trimmed.tolist() == [[1, 1, 1, 0, 0]]
    assert trimmed.sum().item() < mask.sum().item(), "the trim changed nothing"


def test_the_trim_keeps_the_stop_token_itself():
    ids = torch.tensor([[7, STOP, 9]])
    mask = torch.ones(1, 3, dtype=torch.long)
    assert trim_after_first_stop(ids, mask, STOP).tolist() == [[1, 1, 0]]


def test_only_the_first_stop_counts():
    ids = torch.tensor([[7, STOP, 9, STOP, 11]])
    mask = torch.ones(1, 5, dtype=torch.long)
    assert trim_after_first_stop(ids, mask, STOP).tolist() == [[1, 1, 0, 0, 0]]


def test_a_row_with_no_stop_is_left_alone():
    ids = torch.tensor([[7, 8, 9]])
    mask = torch.ones(1, 3, dtype=torch.long)
    assert trim_after_first_stop(ids, mask, STOP).tolist() == [[1, 1, 1]]


def test_rows_are_trimmed_independently():
    ids = torch.tensor([[7, STOP, 9, 10], [7, 8, 9, STOP]])
    mask = torch.ones(2, 4, dtype=torch.long)
    assert trim_after_first_stop(ids, mask, STOP).tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]


def test_the_trim_refuses_a_shape_mismatch():
    with pytest.raises(ValueError, match="must match"):
        trim_after_first_stop(torch.zeros(1, 4, dtype=torch.long), torch.ones(1, 3), STOP)


# ------------------------------------------------- the two defects together
def test_minimind_masking_makes_the_trim_a_silent_no_op():
    """The control. Build MiniMind's mask and watch both safeguards fail at once:
    the stop token is not trainable, and the trim that looks for it does nothing."""
    generated = [7, 8, STOP, 9, 10]
    minimind_mask = [int(t != STOP) for t in generated]   # the offending line, verbatim

    ids = torch.tensor([generated])
    mask = torch.tensor([minimind_mask])

    assert not _stop_is_trainable(ids, mask, STOP), "control is wrong; rewrite it"
    trimmed = trim_after_first_stop(ids, mask, STOP)
    assert trimmed.tolist() == mask.tolist(), (
        "with stop tokens masked away the trim cannot find one, so it changes nothing — "
        "which is exactly what MiniMind's dead code did every step"
    )
    assert trimmed[0, 3:].sum().item() == 2, "tokens after the stop are still being trained on"


def test_our_masking_lets_both_safeguards_work():
    """The same trajectory through this module: stop trainable, trim effective."""
    tokens, mask = trajectory_mask([Turn(generated=[7, 8, STOP, 9, 10])], prompt_len=0)
    ids = torch.tensor([tokens])
    loss_mask = torch.tensor([mask])

    assert stop_is_trainable(ids, loss_mask, STOP)
    trimmed = trim_after_first_stop(ids, loss_mask, STOP)
    assert trimmed.tolist() == [[1, 1, 1, 0, 0]]
    assert trimmed[0, 3:].sum().item() == 0


# ------------------------------------------------- padding
def test_ragged_trajectories_pad_with_masks_that_agree():
    ids, loss_mask, attention = pad_trajectories(
        [([7, 8, STOP], [1, 1, 1]), ([7], [1])], pad_token_id=0
    )
    assert ids.tolist() == [[7, 8, STOP], [7, 0, 0]]
    assert loss_mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    assert attention.tolist() == [[1, 1, 1], [1, 0, 0]]


def test_padding_refuses_a_mask_that_does_not_match_its_tokens():
    with pytest.raises(ValueError, match="mask of 2 for 3 tokens"):
        pad_trajectories([([7, 8, 9], [1, 1])], pad_token_id=0)


def test_padding_refuses_an_empty_batch():
    with pytest.raises(ValueError, match="no trajectories"):
        pad_trajectories([], pad_token_id=0)
