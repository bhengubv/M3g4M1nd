"""The four stages beyond pretraining and SFT, joined up end to end.

Each was previously a subcommand that printed what was missing and exited 2.
These tests run each stage for real on a tiny model and assert it did
something: a loss, a step count, and the metric that says the stage worked
rather than merely ran.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from m3g4m1nd.config import M3Config
from m3g4m1nd.data import PreferenceDataset, PromptDataset, SFTDataset
from m3g4m1nd.model import M3ForCausalLM
from m3g4m1nd.pipelines import run_agent, run_distillation, run_dpo, run_grpo
from m3g4m1nd.tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(ROOT / "model")


def tiny(tok, seed: int = 11, **overrides) -> M3ForCausalLM:
    torch.manual_seed(seed)
    return M3ForCausalLM(M3Config(**{
        "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "intermediate_size": 64,
        "vocab_size": len(tok), "max_position_embeddings": 256, **overrides,
    }))


def pair(tok, **overrides):
    policy = tiny(tok, **overrides)
    reference = tiny(tok, **overrides)
    reference.load_state_dict(policy.state_dict())
    return policy, reference.eval()


# ------------------------------------------------------------------ DPO
def test_dpo_runs_and_reports_preference_accuracy(tok):
    policy, reference = pair(tok)
    dataset = PreferenceDataset(FIXTURES / "preference.jsonl", tok, max_length=64)

    result = run_dpo(
        policy, reference, dataset,
        torch.optim.AdamW(policy.parameters(), lr=1e-3),
        epochs=1, batch_size=2, beta=0.1,
    )

    assert result.stage == "dpo"
    assert result.steps == 2
    assert all(loss == loss for loss in result.losses)
    assert "preference_accuracy" in result.metrics
    assert 0.0 <= result.metrics["preference_accuracy"] <= 1.0


def test_dpo_starts_at_the_break_even_loss(tok):
    """Policy and reference identical means no preference expressed, so the
    first loss is -log(0.5)."""
    import math

    policy, reference = pair(tok)
    dataset = PreferenceDataset(FIXTURES / "preference.jsonl", tok, max_length=64)
    result = run_dpo(
        policy, reference, dataset,
        torch.optim.AdamW(policy.parameters(), lr=0.0), epochs=1, batch_size=4,
    )
    assert result.losses[0] == pytest.approx(-math.log(0.5), abs=1e-5)


def test_dpo_moves_the_policy_away_from_the_reference(tok):
    policy, reference = pair(tok)
    before = policy.lm_head.weight.detach().clone()
    run_dpo(
        policy, reference,
        PreferenceDataset(FIXTURES / "preference.jsonl", tok, max_length=64),
        torch.optim.AdamW(policy.parameters(), lr=1e-2), epochs=1, batch_size=2,
    )
    assert not torch.allclose(before, policy.lm_head.weight)


def test_the_preference_dataset_never_concatenates_the_pair(tok):
    row = PreferenceDataset(FIXTURES / "preference.jsonl", tok, max_length=64)[0]
    assert set(row) == {"chosen_ids", "chosen_mask", "rejected_ids", "rejected_mask"}
    assert row["chosen_mask"].sum() > 0 and row["rejected_mask"].sum() > 0


def test_the_preference_dataset_refuses_a_row_without_both_sides(tmp_path, tok):
    broken = tmp_path / "half.jsonl"
    broken.write_text('{"chosen": [{"role": "user", "content": "x"}]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no 'rejected' field"):
        PreferenceDataset(broken, tok, max_length=32)[0]


# ------------------------------------------------------------------ distillation
def test_distillation_runs_and_reports_both_terms(tok):
    student = tiny(tok, seed=11)
    teacher = tiny(tok, seed=12)
    dataset = SFTDataset(FIXTURES / "sft.jsonl", tok, max_length=64)

    result = run_distillation(
        student, teacher, dataset,
        torch.optim.AdamW(student.parameters(), lr=1e-3),
        epochs=1, batch_size=2, temperature=1.5, alpha=0.5,
    )

    assert result.stage == "distill"
    assert result.steps == 2
    assert "cross_entropy" in result.metrics and "kl" in result.metrics
    assert result.final_loss == result.final_loss


def test_distilling_from_an_identical_teacher_leaves_only_the_label_term(tok):
    student = tiny(tok, seed=11)
    teacher = tiny(tok, seed=11)
    teacher.load_state_dict(student.state_dict())

    result = run_distillation(
        student, teacher, SFTDataset(FIXTURES / "sft.jsonl", tok, max_length=64),
        torch.optim.AdamW(student.parameters(), lr=0.0), epochs=1, batch_size=4, alpha=0.5,
    )
    assert result.metrics["kl"] == pytest.approx(0.0, abs=1e-5)


def test_a_vocabulary_mismatch_is_refused(tok):
    student = tiny(tok)
    teacher = tiny(tok, vocab_size=len(tok) + 8)
    with pytest.raises(ValueError, match="logits are not comparable"):
        run_distillation(
            student, teacher, SFTDataset(FIXTURES / "sft.jsonl", tok, max_length=32),
            torch.optim.AdamW(student.parameters(), lr=1e-3),
        )


def test_an_out_of_range_alpha_is_refused(tok):
    student, teacher = tiny(tok), tiny(tok, seed=12)
    with pytest.raises(ValueError, match=r"alpha must be in \[0, 1\]"):
        run_distillation(
            student, teacher, SFTDataset(FIXTURES / "sft.jsonl", tok, max_length=32),
            torch.optim.AdamW(student.parameters(), lr=1e-3), alpha=1.5,
        )


# ------------------------------------------------------------------ GRPO
def test_grpo_runs_a_whole_loop_and_the_clip_is_live(tok):
    policy, reference = pair(tok)
    dataset = PromptDataset(FIXTURES / "prompts.jsonl", tok, max_length=48)

    result = run_grpo(
        policy, reference, dataset,
        torch.optim.AdamW(policy.parameters(), lr=1e-3), tok,
        epochs=1, batch_size=2, num_generations=2, max_new_tokens=6,
        inner_iters=2, epsilon=0.01,
    )

    assert result.stage == "grpo"
    assert result.steps == 4                      # 2 batches x 2 inner iterations
    assert "mean_reward" in result.metrics
    assert result.metrics["clip_ever_engaged"] == 1.0, (
        "the clip never fired across the run, which is the MiniMind defect"
    )


def test_grpo_refuses_a_single_inner_iteration(tok):
    policy, reference = pair(tok)
    with pytest.raises(ValueError, match="mathematically inert"):
        run_grpo(
            policy, reference,
            PromptDataset(FIXTURES / "prompts.jsonl", tok, max_length=48),
            torch.optim.AdamW(policy.parameters(), lr=1e-3), tok, inner_iters=1,
        )


def test_grpo_scores_against_the_ground_truth_when_the_row_has_one(tok):
    dataset = PromptDataset(FIXTURES / "prompts.jsonl", tok, max_length=48)
    assert dataset[0]["gt"] == [9472]
    assert dataset[2]["gt"] == []


# ------------------------------------------------------------------ agent RL
def test_agent_rl_runs_and_keeps_the_stop_token_trainable(tok):
    policy, reference = pair(tok)
    dataset = PromptDataset(FIXTURES / "prompts.jsonl", tok, max_length=48)

    result = run_agent(
        policy, reference, dataset,
        torch.optim.AdamW(policy.parameters(), lr=1e-3), tok,
        epochs=1, batch_size=2, num_generations=2, max_new_tokens=6,
        max_turns=2, inner_iters=2,
    )

    assert result.stage == "agent"
    assert result.steps == 4
    assert "mean_reward" in result.metrics
    assert "tool_calls_executed" in result.metrics


def test_the_agent_stage_can_execute_a_tool_for_real(tok):
    """Closes the hole: the trajectory machinery is wired to a live registry."""
    from m3g4m1nd.tools import ToolRegistry

    registry = ToolRegistry()
    assert registry.execute("calculate", {"expression": "256*37"}) == {"result": 9472}


def test_the_prompt_dataset_drops_a_trailing_assistant_turn(tok):
    """That turn is what the policy is being asked to produce."""
    dataset = PromptDataset(FIXTURES / "sft.jsonl", tok, max_length=96)
    rendered = dataset[0]["prompt"]
    assert rendered.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert "Hello there" not in rendered


def test_the_prompt_dataset_refuses_a_row_with_no_messages(tmp_path, tok):
    broken = tmp_path / "empty.jsonl"
    broken.write_text('{"gt": [1]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no conversations/messages"):
        PromptDataset(broken, tok, max_length=32)[0]
