"""Loss functions shared across the training scripts.

`distillation_loss` carries the recovered MiniMind fix (PR #909): a batch with
no supervised positions must return a finite, differentiable zero rather than
NaN. It happens routinely — a short conversation truncated so that the
assistant's turn falls outside the window leaves nothing to learn from, and
`kl_div` with `reduction='batchmean'` over an empty batch divides by zero.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """Temperature-scaled KL from teacher to student.

    Multiplied by T^2 so the gradient magnitude does not collapse as the
    temperature rises (Hinton et al., 2015).
    """
    if temperature <= 0:
        raise ValueError(f"distillation temperature must be positive, got {temperature}")

    if student_logits.numel() == 0:
        # Differentiable zero: keeps the graph intact so .backward() still runs
        # and produces a zero gradient rather than an error.
        return student_logits.sum() * 0.0

    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return (temperature**2) * kl


def masked_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100
) -> torch.Tensor:
    """Next-token cross-entropy, averaged over supervised positions only.

    Shift convention, stated once so no caller has to guess: the logit at
    position t predicts the label at t+1.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
        reduction="none",
    )
    supervised = (shift_labels.view(-1) != ignore_index).float()
    return (per_token * supervised).sum() / supervised.sum().clamp(min=1.0)


def grpo_per_token_loss(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    ref_per_token_logps: torch.Tensor,
    advantages: torch.Tensor,
    beta: float,
    loss_type: str = "grpo",
    epsilon: float = 0.2,
    epsilon_high: float = 5.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token policy loss, plus the clip fraction for logging.

    The clip fraction is returned, not optional, because MiniMind's clip never
    engaged and nothing reported that. If this number is always zero, the brake
    is not working and the operator should be able to see it.
    """
    kl = ref_per_token_logps - per_token_logps
    per_token_kl = torch.exp(kl) - kl - 1.0
    ratio = torch.exp(per_token_logps - old_per_token_logps)

    if loss_type == "cispo":
        clamped = torch.clamp(ratio, max=epsilon_high).detach()
        loss = -(clamped * advantages.unsqueeze(1) * per_token_logps - beta * per_token_kl)
        clipped = (ratio > epsilon_high).float()
    elif loss_type == "grpo":
        unclipped = ratio * advantages.unsqueeze(1)
        clamped = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages.unsqueeze(1)
        loss = -(torch.min(unclipped, clamped) - beta * per_token_kl)
        clipped = ((ratio - 1.0).abs() > epsilon).float()
    else:
        raise ValueError(f"unknown loss_type {loss_type!r}; expected 'grpo' or 'cispo'")

    return loss, clipped


def sequence_logps(
    per_token_logps: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Sum per-token log-probabilities over the supervised span of each row."""
    if per_token_logps.shape != mask.shape:
        raise ValueError(
            f"logps {tuple(per_token_logps.shape)} and mask {tuple(mask.shape)} must match"
        )
    return (per_token_logps * mask).sum(dim=1)


def dpo_loss(
    policy_chosen: torch.Tensor,
    policy_rejected: torch.Tensor,
    reference_chosen: torch.Tensor,
    reference_rejected: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Direct preference optimisation. Returns (loss, accuracy).

    Takes the four sequence log-probabilities explicitly rather than one batch
    to be cut in half. MiniMind concatenated chosen and rejected rows and split
    on `batch_size // 2`, which is correct only if the caller stacked them in
    that exact order and silently mispairs every example otherwise. Naming the
    four arguments makes that impossible.

    The accuracy returned alongside is the fraction of pairs the policy already
    ranks correctly. It is not optional: DPO loss falls whether or not the
    ranking improves, so a run with a falling loss and a flat accuracy is a run
    doing nothing useful, and the operator should be able to see that.
    """
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    shapes = {t.shape for t in (policy_chosen, policy_rejected, reference_chosen, reference_rejected)}
    if len(shapes) != 1:
        raise ValueError(f"all four log-probability tensors must have one shape, got {shapes}")

    policy_margin = policy_chosen - policy_rejected
    reference_margin = reference_chosen - reference_rejected
    logits = policy_margin - reference_margin

    loss = -F.logsigmoid(beta * logits).mean()
    accuracy = (policy_margin > reference_margin).float().mean()
    return loss, accuracy


def generalised_advantage_estimate(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE over a batch of equal-width, variably-masked sequences.

    Returns (advantages, returns). The value beyond each row's own final
    supervised position is treated as zero — a finished response has nothing
    after it — rather than reading whatever padding happens to hold, which is
    the quiet way a value head learns to predict junk.
    """
    if not rewards.shape == values.shape == mask.shape:
        raise ValueError(
            f"rewards {tuple(rewards.shape)}, values {tuple(values.shape)} and mask "
            f"{tuple(mask.shape)} must all match"
        )
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be in [0, 1], got {gamma}")
    if not 0.0 <= lam <= 1.0:
        raise ValueError(f"lam must be in [0, 1], got {lam}")

    values = values * mask
    width = rewards.size(1)
    advantages = torch.zeros_like(rewards)
    running = torch.zeros_like(rewards[:, 0])

    for step in reversed(range(width)):
        next_value = values[:, step + 1] if step + 1 < width else torch.zeros_like(running)
        live = mask[:, step]
        delta = rewards[:, step] + gamma * next_value - values[:, step]
        running = delta + gamma * lam * running
        # A masked position contributes nothing and does not carry the trace
        # past the end of its own sequence.
        running = running * live
        advantages[:, step] = running

    return advantages * mask, (advantages + values) * mask


def whiten(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Centre and scale over the live positions only."""
    live = mask.sum().clamp(min=1)
    mean = (values * mask).sum() / live
    variance = (((values - mean) ** 2) * mask).sum() / live
    return (values - mean) * torch.rsqrt(variance + eps) * mask


def ppo_losses(
    per_token_logps: torch.Tensor,
    old_per_token_logps: torch.Tensor,
    values: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    clip_epsilon: float = 0.2,
    clip_value: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clipped policy and value losses.

    Returns (policy_loss, value_loss, clip_fraction, approx_kl). As in
    `grpo_per_token_loss`, the clip fraction is a return value rather than a
    nice-to-have: a clip that never engages is indistinguishable from no clip
    at all, and that went unnoticed in MiniMind for a year.
    """
    live = mask.sum().clamp(min=1)
    log_ratio = per_token_logps - old_per_token_logps
    ratio = torch.exp(log_ratio)

    unclipped = -advantages * ratio
    clipped = -advantages * torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss = (torch.max(unclipped, clipped) * mask).sum() / live

    value_clipped = torch.clamp(values, old_values - clip_value, old_values + clip_value)
    value_loss = 0.5 * (
        torch.max((values - returns) ** 2, (value_clipped - returns) ** 2) * mask
    ).sum() / live

    clip_fraction = (((ratio - 1.0).abs() > clip_epsilon).float() * mask).sum() / live
    approx_kl = (0.5 * (log_ratio**2) * mask).sum() / live

    return policy_loss, value_loss, clip_fraction, approx_kl
