"""Multi-turn tool-use trajectories.

This module exists because of the most consequential MiniMind defect, and it is
worth stating exactly, because two safeguards there cancelled each other out.

MiniMind's agent trainer built its response mask as:

    response_mask.extend([int(t != tokenizer.eos_token_id) for t in new_ids])

— zeroing every stop token, conflating "do not train on filler" with "do not
train on the end of my own turn". The model therefore received no gradient on
*when to stop talking*.

Then, a hundred lines later, it tried to trim each trajectory at its first stop
token:

    is_eos = (input_ids[:, 1:] == eos) & completion_mask.bool()

But `completion_mask` is the mask above, which has already zeroed every stop
token — so `is_eos` was always False, `has_eos` was always False, and the trim
was dead code that ran every step and did nothing.

Here the two are separated. `trajectory_mask` marks what the policy generated,
stop tokens included. `trim_after_first_stop` then truncates. A test asserts
the stop token is trainable, and another asserts the trim actually trims — the
pair that MiniMind could not have passed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

GENERATED = 1
OBSERVED = 0


@dataclass
class Turn:
    """One step of a trajectory: what the policy said, then what came back.

    `observation` is the environment's reply — a tool result rendered back into
    the context. It is part of the sequence the model reads and no part of what
    it is scored on.
    """

    generated: list[int]
    observation: list[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.generated:
            raise ValueError("a turn with no generated tokens is not a turn")


def trajectory_mask(turns: list[Turn], prompt_len: int) -> tuple[list[int], list[int]]:
    """Flatten turns into (token_ids, mask) over the whole trajectory.

    The mask is 1 for everything the policy produced — **stop tokens
    included** — and 0 for the prompt and for every observation.
    """
    if prompt_len < 0:
        raise ValueError(f"prompt_len must be non-negative, got {prompt_len}")

    tokens: list[int] = []
    mask: list[int] = [OBSERVED] * prompt_len

    for turn in turns:
        tokens.extend(turn.generated)
        mask.extend([GENERATED] * len(turn.generated))
        tokens.extend(turn.observation)
        mask.extend([OBSERVED] * len(turn.observation))

    if len(mask) != prompt_len + len(tokens):
        raise AssertionError("mask and token count diverged")
    return tokens, mask


def trim_after_first_stop(
    token_ids: torch.Tensor, mask: torch.Tensor, stop_token_id: int
) -> torch.Tensor:
    """Zero the mask after each row's first generated stop token.

    Depends on stop tokens being *inside* the mask, which is why the two
    operations are separate functions with separate tests. If something
    upstream masked them away, this becomes a silent no-op — and
    `stop_is_trainable` below is how a caller checks before relying on it.
    """
    if token_ids.shape != mask.shape:
        raise ValueError(
            f"token_ids {tuple(token_ids.shape)} and mask {tuple(mask.shape)} must match"
        )

    is_stop = (token_ids == stop_token_id) & mask.bool()
    has_stop = is_stop.any(dim=1)
    width = token_ids.size(1)

    first = torch.full((token_ids.size(0),), width - 1, dtype=torch.long)
    first[has_stop] = is_stop.int().argmax(dim=1)[has_stop]

    positions = torch.arange(width).unsqueeze(0)
    return mask * (positions <= first.unsqueeze(1)).to(mask.dtype)


def stop_is_trainable(token_ids: torch.Tensor, mask: torch.Tensor, stop_token_id: int) -> bool:
    """True if any stop token carries loss.

    Run this once on a real batch before trusting anything downstream that
    looks for a stop token. False means the model cannot learn to stop, and
    any trim that depends on finding one is doing nothing.
    """
    return bool(((token_ids == stop_token_id) & mask.bool()).any())


def pad_trajectories(
    rows: list[tuple[list[int], list[int]]], pad_token_id: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad ragged trajectories. Returns (token_ids, loss_mask, attention_mask)."""
    if not rows:
        raise ValueError("no trajectories to pad")
    width = max(len(tokens) for tokens, _ in rows)

    token_ids, loss_mask, attention = [], [], []
    for tokens, mask in rows:
        if len(mask) != len(tokens):
            raise ValueError(f"mask of {len(mask)} for {len(tokens)} tokens")
        filler = width - len(tokens)
        token_ids.append(tokens + [pad_token_id] * filler)
        loss_mask.append(mask + [OBSERVED] * filler)
        attention.append([1] * len(tokens) + [0] * filler)

    return (
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor(loss_mask, dtype=torch.long),
        torch.tensor(attention, dtype=torch.long),
    )
