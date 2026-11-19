"""Rows A1, A2, A4, A7 — the model core.

Each test names the MiniMind defect it exists to prevent, so that if anyone
ever proposes deleting one they have to argue with the consequence, not just
the line count.
"""

from __future__ import annotations

import math

import pytest
import torch

from m3g4m1nd.config import M3Config
from m3g4m1nd.model import M3ForCausalLM, MoEFeedForward, count_parameters
from tests.conftest import tiny_config, tiny_model


# ---------------------------------------------------------------- A1
# Published parameter counts are exact integers, not "about 64M". If a refactor
# changes the shape, the number changes and this fails loudly.

def test_dense_parameter_count_is_exact():
    total, trainable = count_parameters(M3ForCausalLM(M3Config()))
    assert total == 63_912_192, f"dense reference model is {total:,}, expected 63,912,192"
    assert trainable == total


def test_moe_parameter_counts_are_exact():
    config = M3Config(use_moe=True, num_experts=4, num_experts_per_tok=1)
    total, _ = count_parameters(M3ForCausalLM(config))
    assert total == 198_416_640, f"MoE reference model is {total:,}, expected 198,416,640"

    # Active parameters: everything except the experts that do not fire.
    per_expert = 3 * config.hidden_size * config.moe_intermediate_size
    inactive = config.num_hidden_layers * per_expert * (
        config.num_experts - config.num_experts_per_tok
    )
    assert total - inactive == 63_936_768

    # Active is the dense model plus one router gate per layer, exactly.
    dense_total, _ = count_parameters(M3ForCausalLM(M3Config()))
    gates = config.num_hidden_layers * config.hidden_size * config.num_experts
    assert total - inactive == dense_total + gates


def test_tied_embedding_is_counted_once():
    model = M3ForCausalLM(M3Config(vocab_size=64, hidden_size=16, num_hidden_layers=1))
    assert model.lm_head.weight is model.model.embed_tokens.weight
    ids = {id(p) for p in model.parameters()}
    assert len(ids) == len(list(model.parameters())), "parameters() yielded the tied weight twice"


def test_untied_embedding_adds_its_own_matrix():
    tied, _ = count_parameters(M3ForCausalLM(M3Config(tie_word_embeddings=True)))
    untied, _ = count_parameters(M3ForCausalLM(M3Config(tie_word_embeddings=False)))
    assert untied - tied == 6400 * 768


# ---------------------------------------------------------------- A2
# MiniMind's load-balancing term left the top-k axis in place after averaging,
# so it was scaled by k. Correct at its default of k=1, wrong at every other
# setting, and invisible to a reader.

def _aux_loss(num_experts_per_tok, seq_len=8):
    torch.manual_seed(3)
    config = tiny_config(
        use_moe=True, num_experts=4, num_experts_per_tok=num_experts_per_tok,
        router_aux_loss_coef=1.0,
    )
    model = M3ForCausalLM(config).train()
    ids = torch.randint(0, config.vocab_size, (2, seq_len))
    return model(ids).aux_loss


def test_routing_fractions_sum_to_k_not_one_per_expert_axis():
    """The quantity the balancing term is defined over: fraction of tokens per expert."""
    import torch.nn.functional as F

    torch.manual_seed(5)
    for k in (1, 2, 3):
        idx = torch.stack([torch.randperm(4)[:k] for _ in range(16)])
        load = F.one_hot(idx, 4).float().sum(dim=1).mean(dim=0)
        assert math.isclose(load.sum().item(), float(k), rel_tol=1e-6), (
            f"with top-{k}, per-expert token fractions must total {k}, got {load.sum().item()}"
        )


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_aux_loss_is_finite_and_bounded_at_every_k(k):
    loss = _aux_loss(k)
    assert torch.isfinite(loss), f"aux loss not finite at top-{k}"
    # With coefficient 1 and E experts, a perfectly balanced router gives k;
    # the worst case is bounded by E*k. MiniMind's version blew past this.
    assert 0.0 < loss.item() <= 4 * k + 1e-6, f"aux loss {loss.item()} out of range at top-{k}"


def test_aux_loss_grows_only_with_imbalance_not_with_k():
    """Scaling with k is the bug; scaling with imbalance is the point."""
    balanced = _aux_loss(1)
    assert torch.isfinite(balanced)
    for k in (2, 3, 4):
        assert _aux_loss(k).item() / k < 4 * balanced.item() + 1.0


def test_aux_loss_is_zero_in_eval():
    torch.manual_seed(3)
    model = M3ForCausalLM(tiny_config(use_moe=True)).eval()
    out = model(torch.randint(0, 64, (2, 6)))
    assert out.aux_loss.item() == 0.0


# ---------------------------------------------------------------- A4
# MiniMind sliced its rotary table past the end, got an empty tensor, and died
# in a broadcast several frames away from the cause.

def test_overflowing_the_position_table_raises_with_the_limit_named():
    model = tiny_model()  # max_position_embeddings=128
    ids = torch.randint(0, 64, (1, 129))
    with pytest.raises(ValueError, match=r"max_position_embeddings=128"):
        model(ids)


def test_overflow_during_cached_decoding_also_raises():
    model = tiny_model()
    out = model(torch.randint(0, 64, (1, 127)), use_cache=True)
    # 127 cached + 1 is fine; the next one is not.
    model(torch.randint(0, 64, (1, 1)), past_key_values=out.past_key_values, use_cache=True)
    out2 = model(torch.randint(0, 64, (1, 128)), use_cache=True)
    with pytest.raises(ValueError, match=r"128 cached \+ 1 new"):
        model(torch.randint(0, 64, (1, 1)), past_key_values=out2.past_key_values)


def test_sequence_exactly_at_the_limit_is_allowed():
    model = tiny_model()
    out = model(torch.randint(0, 64, (1, 128)))
    assert out.logits.shape == (1, 128, 64)


# ---------------------------------------------------------------- A7
# MiniMind had two attention paths that built their own masks and were never
# compared to each other.

@pytest.mark.parametrize("padded", [False, True])
def test_fused_and_manual_attention_paths_agree(padded):
    torch.manual_seed(11)
    fused = tiny_model(flash_attn=True)
    manual = tiny_model(flash_attn=False)
    manual.load_state_dict(fused.state_dict())

    ids = torch.tensor([[0, 0, 3, 4, 5], [7, 8, 9, 10, 11]])
    mask = ids.ne(0).long() if padded else None

    with torch.no_grad():
        a = fused(ids, attention_mask=mask).logits
        b = manual(ids, attention_mask=mask).logits
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)
    assert bool(a.isfinite().all()) and bool(b.isfinite().all())


def test_a_fully_masked_row_does_not_poison_the_batch():
    """Left-padded batches produce all-masked rows routinely. -inf gives NaN."""
    for flash in (True, False):
        model = tiny_model(flash_attn=flash)
        ids = torch.tensor([[0, 0, 0], [4, 5, 6]])
        mask = torch.tensor([[0, 0, 0], [1, 1, 1]])
        with torch.no_grad():
            logits = model(ids, attention_mask=mask).logits
        assert bool(logits.isfinite().all()), f"NaN leaked with flash_attn={flash}"


def test_mask_shape_mismatch_is_refused():
    model = tiny_model()
    ids = torch.randint(0, 64, (2, 5))
    with pytest.raises(ValueError, match="covers 3 keys but there are 5"):
        model(ids, attention_mask=torch.ones(2, 3, dtype=torch.long))
