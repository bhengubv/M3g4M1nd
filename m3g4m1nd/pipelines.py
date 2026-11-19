"""The four stages beyond pretraining and SFT, each joined up end to end.

Until now the pieces existed and nothing wired them: `dpo`, `grpo`, `agent`
and `distill` printed what was missing and exited 2. Honest, but a gap.

Each function here is the plumbing between parts already tested elsewhere —
the losses in `losses.py`, the updater in `rl.py`, the rollout in
`rollout.py`, the rubric in `rewards.py`, the trajectory masks in `agent.py`.
None of them reimplements any of that, which is the whole point: MiniMind's
seven trainers each carried their own copy and the copies drifted apart.

Every function takes a `StageResult` back rather than printing, so the tests
can assert what happened instead of parsing a log.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .agent import Turn, pad_trajectories, stop_is_trainable, trajectory_mask, trim_after_first_stop
from .losses import distillation_loss, dpo_loss, masked_cross_entropy, sequence_logps
from .rewards import Rubric, deterministic_scorer
from .rl import GRPOUpdater
from .rollout import TorchRolloutEngine
from .tools import ToolRegistry


@dataclass
class StageResult:
    """What a stage did. Numbers, not prose, so a test can check them."""

    stage: str
    steps: int = 0
    losses: list[float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def final_loss(self) -> float:
        return self.losses[-1] if self.losses else float("nan")


def _token_logps(model, input_ids, mask=None) -> torch.Tensor:
    """log p of each token given its prefix, aligned to positions 1..n."""
    logits = model(input_ids, attention_mask=mask).logits[:, :-1, :]
    return torch.log_softmax(logits.float(), dim=-1).gather(
        2, input_ids[:, 1:].unsqueeze(-1)
    ).squeeze(-1)


# ------------------------------------------------------------------ DPO
def run_dpo(
    model,
    reference_model,
    dataset,
    optimizer,
    *,
    epochs: int = 1,
    batch_size: int = 2,
    beta: float = 0.1,
    grad_clip: float = 1.0,
) -> StageResult:
    """Direct preference optimisation over chosen/rejected pairs."""
    result = StageResult("dpo")
    model.train()
    reference_model.eval()
    accuracies: list[float] = []

    for _ in range(epochs):
        for start in range(0, len(dataset), batch_size):
            rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            batch = {key: torch.stack([row[key] for row in rows]) for key in rows[0]}

            # The pair never gets concatenated, so it cannot get mispaired.
            sides = {}
            for side in ("chosen", "rejected"):
                ids, mask = batch[f"{side}_ids"], batch[f"{side}_mask"][:, 1:]
                sides[f"policy_{side}"] = sequence_logps(_token_logps(model, ids), mask)
                with torch.no_grad():
                    sides[f"reference_{side}"] = sequence_logps(
                        _token_logps(reference_model, ids), mask
                    )

            loss, accuracy = dpo_loss(
                sides["policy_chosen"], sides["policy_rejected"],
                sides["reference_chosen"], sides["reference_rejected"],
                beta=beta,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            result.steps += 1
            result.losses.append(loss.item())
            accuracies.append(accuracy.item())

    result.metrics["preference_accuracy"] = sum(accuracies) / max(len(accuracies), 1)
    return result


# ------------------------------------------------------------------ distillation
def run_distillation(
    student,
    teacher,
    dataset,
    optimizer,
    *,
    epochs: int = 1,
    batch_size: int = 2,
    temperature: float = 1.5,
    alpha: float = 0.5,
    grad_clip: float = 1.0,
) -> StageResult:
    """Student learns the teacher's distribution, weighted against the labels."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if student.config.vocab_size != teacher.config.vocab_size:
        raise ValueError(
            f"student vocabulary ({student.config.vocab_size}) differs from the "
            f"teacher's ({teacher.config.vocab_size}); their logits are not comparable"
        )

    result = StageResult("distill")
    student.train()
    teacher.eval()
    teacher.requires_grad_(False)

    for _ in range(epochs):
        for start in range(0, len(dataset), batch_size):
            rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            input_ids = torch.stack([row[0] for row in rows])
            labels = torch.stack([row[1] for row in rows])

            student_logits = student(input_ids).logits
            with torch.no_grad():
                teacher_logits = teacher(input_ids).logits

            supervised = (labels[..., 1:] != -100).reshape(-1)
            flat = student_logits[..., :-1, :].reshape(-1, student_logits.size(-1))
            flat_teacher = teacher_logits[..., :-1, :].reshape(-1, teacher_logits.size(-1))

            cross_entropy = masked_cross_entropy(student_logits, labels)
            kl = distillation_loss(
                flat[supervised], flat_teacher[supervised], temperature=temperature
            )
            loss = alpha * cross_entropy + (1.0 - alpha) * kl

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()

            result.steps += 1
            result.losses.append(loss.item())
            result.metrics["cross_entropy"] = cross_entropy.item()
            result.metrics["kl"] = kl.detach().item()

    return result


# ------------------------------------------------------------------ GRPO
def run_grpo(
    model,
    reference_model,
    dataset,
    optimizer,
    tokenizer,
    *,
    epochs: int = 1,
    batch_size: int = 2,
    num_generations: int = 4,
    max_new_tokens: int = 16,
    inner_iters: int = 2,
    beta: float = 0.1,
    epsilon: float = 0.2,
    loss_type: str = "grpo",
    rubric: Rubric | None = None,
    engine: TorchRolloutEngine | None = None,
) -> StageResult:
    """Sample a group per prompt, score it, and update against the group mean."""
    result = StageResult("grpo")
    rubric = rubric or Rubric(scorer=deterministic_scorer)
    engine = engine or TorchRolloutEngine(model, tokenizer)
    updater = GRPOUpdater(
        optimizer=optimizer, beta=beta, loss_type=loss_type,
        epsilon=epsilon, inner_iters=inner_iters,
    )
    model.train()

    for _ in range(epochs):
        for start in range(0, len(dataset), batch_size):
            rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            width = max(row["prompt_ids"].numel() for row in rows)
            prompt_ids = torch.stack([
                torch.nn.functional.pad(
                    row["prompt_ids"], (width - row["prompt_ids"].numel(), 0),
                    value=tokenizer.pad_token_id,
                )
                for row in rows
            ])
            attention = (prompt_ids != tokenizer.pad_token_id).long()

            rollout = engine.rollout(
                prompt_ids, attention, num_generations=num_generations,
                max_new_tokens=max_new_tokens, temperature=0.9,
            )

            rewards = torch.tensor([
                rubric.score(rows[index // num_generations]["prompt"], completion,
                             rows[index // num_generations]["gt"])
                for index, completion in enumerate(rollout.completions)
            ], dtype=torch.float32)
            advantages = GRPOUpdater.group_advantages(rewards, num_generations)

            prompt_width = prompt_ids.size(1)
            full_mask = (rollout.output_ids != tokenizer.pad_token_id).long()
            completion_mask = rollout.completion_mask
            old = rollout.per_token_logps

            # Align the per-token scores to the whole sequence, zeroing the prompt.
            pad_left = prompt_width - 1
            padded_old = torch.nn.functional.pad(old, (pad_left, 0))
            padded_mask = torch.nn.functional.pad(completion_mask, (pad_left, 0))

            stats = updater.step(
                model, reference_model, rollout.output_ids, padded_mask,
                padded_old, advantages, attention_mask=full_mask,
            )
            engine.note_policy_update()

            result.steps += len(stats)
            result.losses.extend(s.loss for s in stats)
            result.metrics["mean_reward"] = rewards.mean().item()
            result.metrics["clip_fraction"] = max(s.clip_fraction for s in stats)
            result.metrics["kl_to_reference"] = stats[-1].kl_to_reference

    result.metrics["clip_ever_engaged"] = float(updater.clip_ever_engaged)
    # Zero above is benign when the ratio moved and the clip was simply not
    # needed. It is the defect only when the ratio never moved at all.
    result.metrics["ratio_ever_moved"] = float(updater.ratio_ever_moved)
    result.metrics["clip_is_inert"] = float(updater.clip_is_inert)
    return result


# ------------------------------------------------------------------ agent RL
def run_agent(
    model,
    reference_model,
    dataset,
    optimizer,
    tokenizer,
    *,
    epochs: int = 1,
    batch_size: int = 2,
    num_generations: int = 2,
    max_new_tokens: int = 12,
    max_turns: int = 2,
    inner_iters: int = 2,
    beta: float = 0.1,
    registry: ToolRegistry | None = None,
    rubric: Rubric | None = None,
    engine: TorchRolloutEngine | None = None,
) -> StageResult:
    """Multi-turn tool use: sample, execute, feed the result back, score the whole
    trajectory.

    The masking is `agent.py`'s, so the stop token is trainable and the trim at
    the first stop actually trims — the pair MiniMind's own code disabled.
    """
    result = StageResult("agent")
    registry = registry or ToolRegistry()
    rubric = rubric or Rubric(scorer=deterministic_scorer)
    engine = engine or TorchRolloutEngine(model, tokenizer)
    updater = GRPOUpdater(
        optimizer=optimizer, beta=beta, inner_iters=inner_iters, loss_type="cispo"
    )
    model.train()
    tool_calls_executed = 0

    for _ in range(epochs):
        for start in range(0, len(dataset), batch_size):
            rows = [dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]
            trajectories, rewards = [], []

            for row in rows:
                for _ in range(num_generations):
                    prompt = row["prompt_ids"].unsqueeze(0)
                    turns: list[Turn] = []
                    transcript = ""

                    for _turn in range(max_turns):
                        rollout = engine.rollout(
                            prompt, torch.ones_like(prompt), num_generations=1,
                            max_new_tokens=max_new_tokens, temperature=0.9,
                        )
                        generated = rollout.completion_ids[0].tolist()
                        text = rollout.completions[0]
                        transcript += text

                        observation: list[int] = []
                        if "<tool_call>" in text and "</tool_call>" in text:
                            import json
                            import re

                            body = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
                            try:
                                call = json.loads(body.group(1))
                                outcome = registry.execute(
                                    call.get("name", ""), call.get("arguments", {})
                                )
                            except Exception:
                                outcome = {"error": "unparseable tool call"}
                            tool_calls_executed += 1
                            rendered = tokenizer(
                                f"<tool_response>\n{outcome}\n</tool_response>",
                                add_special_tokens=False,
                            ).input_ids
                            observation = rendered

                        turns.append(Turn(generated=generated, observation=observation))
                        if not observation:
                            break
                        prompt = torch.cat(
                            [prompt, torch.tensor([generated + observation])], dim=1
                        )

                    tokens, mask = trajectory_mask(turns, prompt_len=row["prompt_ids"].numel())
                    trajectories.append((row["prompt_ids"].tolist() + tokens, [0] * 0 + mask))
                    rewards.append(rubric.score(row["prompt"], transcript, row["gt"]))

            input_ids, loss_mask, attention = pad_trajectories(
                trajectories, tokenizer.pad_token_id
            )
            shifted_ids, shifted_mask = input_ids[:, 1:], loss_mask[:, 1:].float()
            trimmed = trim_after_first_stop(shifted_ids, shifted_mask, tokenizer.eos_token_id)

            reward_tensor = torch.tensor(rewards, dtype=torch.float32)
            advantages = GRPOUpdater.group_advantages(reward_tensor, num_generations)

            with torch.no_grad():
                old = _token_logps(model, input_ids, attention)

            stats = updater.step(
                model, reference_model, input_ids, trimmed, old, advantages,
                attention_mask=attention,
            )
            engine.note_policy_update()

            result.steps += len(stats)
            result.losses.extend(s.loss for s in stats)
            result.metrics["mean_reward"] = reward_tensor.mean().item()
            result.metrics["stop_is_trainable"] = float(
                stop_is_trainable(shifted_ids, shifted_mask, tokenizer.eos_token_id)
            )

    result.metrics["tool_calls_executed"] = float(tool_calls_executed)
    result.metrics["clip_ever_engaged"] = float(updater.clip_ever_engaged)
    # Zero above is benign when the ratio moved and the clip was simply not
    # needed. It is the defect only when the ratio never moved at all.
    result.metrics["ratio_ever_moved"] = float(updater.ratio_ever_moved)
    result.metrics["clip_is_inert"] = float(updater.clip_is_inert)
    return result
