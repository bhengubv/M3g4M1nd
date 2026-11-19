"""Reinforcement-learning pieces: a value head, and a policy updater whose
clip actually engages.

Two MiniMind defects live here.

**The value head normalised twice.** MiniMind's critic subclassed the causal LM
and called `self.model(...)` followed by `self.model.norm(...)` — but the body
already ends with that norm, so the value head saw `w·norm(w·norm(x))`. Here
the critic takes the body's output unchanged, and a test asserts it is the same
tensor object's values, bit for bit.

**The clip never engaged.** MiniMind's rollout engine rebound the live model, so
the sampling policy and the updating policy were the same weights with no
update between them. The importance ratio was therefore 1 everywhere and the
clip — the brake that stops one step moving the policy too far — was
mathematically inert. `epsilon` and `epsilon_high` did nothing, and no metric
said so.

The fix is not a patch, it is the missing loop: a batch of samples is drawn
once and then updated over several inner iterations, exactly as the algorithm
is defined. From the second iteration the ratio is genuinely away from 1 and
the clip has something to bite on. `GRPOUpdater` refuses to run with a single
inner iteration unless the caller says out loud that it knows the clip is
inert.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from .config import M3Config
from .losses import grpo_per_token_loss
from .model import M3Model


class ValueHead(nn.Module):
    """Scalar value per position, from the body's already-normalised output.

    Takes an `M3Model`, not an `M3ForCausalLM`, so there is no language-model
    head to leave dangling and no second norm to apply by accident.
    """

    def __init__(self, config: M3Config, body: M3Model | None = None):
        super().__init__()
        self.config = config
        self.body = body if body is not None else M3Model(config)
        self.value_head = nn.Linear(config.hidden_size, 1)

    def hidden_states(self, input_ids, attention_mask=None):
        hidden, _, _ = self.body(input_ids, attention_mask=attention_mask)
        return hidden

    def forward(self, input_ids, attention_mask=None):
        # The body applies the final norm. We do not apply it again.
        return self.value_head(self.hidden_states(input_ids, attention_mask)).squeeze(-1)


@dataclass
class UpdateStats:
    """What one inner iteration did. The clip fraction is not optional.

    If `clip_fraction` is 0.0 on every iteration the brake is not working, and
    whoever is running the job should be able to see that from the log rather
    than discover it a year later in an audit.
    """

    iteration: int
    loss: float
    mean_abs_log_ratio: float
    clip_fraction: float
    kl_to_reference: float


@dataclass
class GRPOUpdater:
    """Group-relative policy optimisation over a fixed batch of rollouts.

    `inner_iters` is the parameter MiniMind did not have. With one iteration
    the ratio is identically 1 and the clip cannot fire; with more, the second
    and later passes measure the policy against the weights that actually drew
    the samples, which is what the ratio is for.
    """

    optimizer: torch.optim.Optimizer
    beta: float = 0.1
    loss_type: str = "grpo"
    epsilon: float = 0.2
    epsilon_high: float = 5.0
    inner_iters: int = 2
    allow_inert_clip: bool = False
    grad_clip: float = 1.0
    history: list[UpdateStats] = field(default_factory=list)

    def __post_init__(self):
        if self.inner_iters < 1:
            raise ValueError(f"inner_iters must be >= 1, got {self.inner_iters}")
        if self.inner_iters == 1 and not self.allow_inert_clip:
            raise ValueError(
                "inner_iters=1 means the sampling policy and the updating policy are "
                "the same weights, so the importance ratio is identically 1 and "
                f"{self.loss_type!r} clipping is mathematically inert — epsilon has no "
                "effect. This is the MiniMind defect this class exists to prevent. "
                "Either raise inner_iters, or pass allow_inert_clip=True to say you "
                "know the clip does nothing."
            )
        if self.inner_iters == 1 and self.allow_inert_clip:
            warnings.warn(
                "inner_iters=1: the clip is inert and epsilon has no effect. "
                "The policy loss reduces to REINFORCE with a group baseline.",
                stacklevel=2,
            )

    @staticmethod
    def token_logps(model, input_ids, attention_mask=None) -> torch.Tensor:
        """log p of each token given its prefix. Position t scores token t+1."""
        logits = model(input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        return F.log_softmax(logits, dim=-1).gather(2, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def group_advantages(rewards: torch.Tensor, num_generations: int) -> torch.Tensor:
        """Centre and scale rewards within each prompt's group of samples."""
        if rewards.numel() % num_generations:
            raise ValueError(
                f"{rewards.numel()} rewards do not divide into groups of {num_generations}"
            )
        grouped = rewards.view(-1, num_generations)
        mean = grouped.mean(dim=1).repeat_interleave(num_generations)
        std = grouped.std(dim=1, unbiased=False).repeat_interleave(num_generations)
        return (rewards - mean) / (std + 1e-4)

    def step(
        self,
        model,
        reference_model,
        input_ids: torch.Tensor,
        completion_mask: torch.Tensor,
        old_per_token_logps: torch.Tensor,
        advantages: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> list[UpdateStats]:
        """Update on one batch of rollouts, over `inner_iters` passes."""
        with torch.no_grad():
            ref_logps = self.token_logps(reference_model, input_ids, attention_mask)

        stats: list[UpdateStats] = []
        for iteration in range(self.inner_iters):
            logps = self.token_logps(model, input_ids, attention_mask)
            per_token_loss, clipped = grpo_per_token_loss(
                logps, old_per_token_logps, ref_logps, advantages,
                beta=self.beta, loss_type=self.loss_type,
                epsilon=self.epsilon, epsilon_high=self.epsilon_high,
            )
            denom = completion_mask.sum(dim=1).clamp(min=1)
            loss = ((per_token_loss * completion_mask).sum(dim=1) / denom).mean()

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip)
            self.optimizer.step()

            with torch.no_grad():
                live = completion_mask.sum().clamp(min=1)
                record = UpdateStats(
                    iteration=iteration,
                    loss=loss.item(),
                    mean_abs_log_ratio=(
                        ((logps - old_per_token_logps).abs() * completion_mask).sum() / live
                    ).item(),
                    clip_fraction=((clipped * completion_mask).sum() / live).item(),
                    kl_to_reference=(((ref_logps - logps) * completion_mask).sum() / live).item(),
                )
            stats.append(record)
            self.history.append(record)

        return stats

    @property
    def clip_ever_engaged(self) -> bool:
        """False after a whole run means the brake was decoration."""
        return any(s.clip_fraction > 0.0 for s in self.history)

    @property
    def ratio_ever_moved(self) -> bool:
        """Whether the importance ratio was ever anything but exactly 1.

        This is the distinction that makes `clip_ever_engaged` usable. A run
        can finish with the clip never firing for two completely different
        reasons:

        - the learning rate is small, the policy barely moved, and the clip
          simply was not needed — fine;
        - the sampling policy and the updating policy are the same weights, so
          the ratio is identically 1 and the clip *cannot* fire — the MiniMind
          defect.

        Without this, a diagnostic for the second case fires constantly on the
        first and gets ignored, which is how a warning stops being read.
        """
        return any(s.mean_abs_log_ratio > 0.0 for s in self.history)

    @property
    def clip_is_inert(self) -> bool:
        """Whether the clip *cannot* fire, as opposed to merely did not.

        This is structural, not measured, and that was arrived at the hard
        way. The first version asked whether the ratio had ever moved — but at
        a learning rate of 1e-9 the policy moves by less than float32 can
        represent, so the measured ratio is exactly 0 and the alarm fired on a
        perfectly healthy run. A diagnostic that cries wolf stops being read,
        which is how MiniMind's real defect survived a year of green runs.

        With one inner iteration the sampling policy and the updating policy
        are the same weights, so the ratio is identically 1 whatever the
        learning rate. That is the condition worth alarming on, and it is
        knowable before a single step runs.
        """
        return self.inner_iters == 1
