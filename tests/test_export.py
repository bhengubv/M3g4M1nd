"""Row E1 — the Qwen3 export reads every field and is verified numerically."""

from __future__ import annotations

import pytest
import torch

from m3g4m1nd.config import M3Config
from m3g4m1nd.export import compare, export_qwen3, qwen3_config, qwen3_state_dict
from m3g4m1nd.model import M3ForCausalLM

SMALL = dict(
    hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
    num_key_value_heads=2, intermediate_size=64, vocab_size=64,
    max_position_embeddings=128,
)


def small(**overrides) -> M3Config:
    return M3Config(**{**SMALL, **overrides})


# ---------------------------------------------------------------- E1
def test_head_dim_is_read_from_the_config_not_recomputed():
    """MiniMind wrote `hidden_size // num_attention_heads` here. With a
    non-default head_dim that is silently the wrong number, and head_dim decides
    how every attention weight is interpreted."""
    config = small(head_dim=16)          # not 32//4 == 8
    assert config.head_dim == 16
    assert qwen3_config(config).head_dim == 16


def test_the_default_case_that_hid_the_defect_still_agrees():
    config = small()
    assert config.head_dim == config.hidden_size // config.num_attention_heads
    assert qwen3_config(config).head_dim == config.head_dim


@pytest.mark.parametrize(
    "field", ["vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
              "num_attention_heads", "num_key_value_heads", "head_dim",
              "max_position_embeddings", "rms_norm_eps",
              "tie_word_embeddings", "hidden_act"],
)
def test_every_translated_field_matches_the_source(field):
    config = small(head_dim=16, rms_norm_eps=1e-5, rope_theta=5e5)
    assert getattr(qwen3_config(config), field) == getattr(config, field)


def test_rope_theta_crosses_over_inside_rope_parameters():
    """Transformers 5.x moved it. `Qwen3Config` has no `rope_theta` attribute at
    all now, so a test reading it by that name passes vacuously on 4.x and fails
    loudly here — which is how this was caught."""
    config = small(rope_theta=5e5)
    exported = qwen3_config(config)
    assert not hasattr(exported, "rope_theta")
    assert exported.rope_parameters["rope_theta"] == config.rope_theta == 5e5
    assert exported.rope_parameters["rope_type"] == "default"


def test_the_state_dict_fits_qwen3_with_nothing_left_over():
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(5)
    ours = M3ForCausalLM(small()).eval()
    theirs = Qwen3ForCausalLM(qwen3_config(small())).eval()

    missing, unexpected = theirs.load_state_dict(qwen3_state_dict(ours), strict=False)
    assert [k for k in missing if "inv_freq" not in k] == []
    assert sorted(unexpected) == []


def test_the_rotary_buffers_are_not_exported():
    torch.manual_seed(5)
    state = qwen3_state_dict(M3ForCausalLM(small()))
    assert not [k for k in state if "rope_cos" in k or "rope_sin" in k]


@pytest.mark.parametrize("head_dim", [8, 16])
def test_the_exported_model_produces_the_same_logits(head_dim):
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(5)
    config = small(head_dim=head_dim)
    ours = M3ForCausalLM(config).eval()
    theirs = Qwen3ForCausalLM(qwen3_config(config)).eval()
    theirs.load_state_dict(qwen3_state_dict(ours), strict=False)

    ids = torch.randint(0, config.vocab_size, (2, 12))
    assert compare(ours, theirs, ids) < 2e-4


def test_export_writes_a_loadable_checkpoint(tmp_path):
    from transformers import Qwen3ForCausalLM

    torch.manual_seed(5)
    ours = M3ForCausalLM(small()).eval()
    out = export_qwen3(ours, tmp_path / "exported")

    assert (out / "config.json").exists()
    reloaded = Qwen3ForCausalLM.from_pretrained(out).eval()

    ids = torch.randint(0, 64, (2, 10))
    assert compare(ours, reloaded, ids) < 2e-4


def test_export_includes_the_tokenizer_when_asked(tmp_path):
    from pathlib import Path

    torch.manual_seed(5)
    ours = M3ForCausalLM(small()).eval()
    out = export_qwen3(
        ours, tmp_path / "with_tok",
        tokenizer_dir=Path(__file__).resolve().parent.parent / "model",
    )
    assert (out / "tokenizer.json").exists()
    assert (out / "tokenizer_config.json").exists()


def test_export_refuses_rather_than_writing_a_mismatch(tmp_path):
    """A tolerance of zero makes any drift fatal. The point is that the exporter
    checks at all: MiniMind's wrote the file and reported success."""
    torch.manual_seed(5)
    ours = M3ForCausalLM(small()).eval()

    with pytest.raises(ValueError, match="disagrees with the source"):
        export_qwen3(ours, tmp_path / "strict", tolerance=-1.0)
    assert not (tmp_path / "strict").exists(), "a rejected export still wrote files"


def test_moe_export_is_no_longer_refused():
    """**Replaces** a test that asserted `NotImplementedError`.

    Said out loud rather than deleted quietly, because this repository's rule
    is that a test is never removed without a reason on the record. The reason
    here is that the limitation it guarded was lifted: MoE export is
    implemented, and the four tests below cover it. The assertion it used to
    make — that the exporter refuses rather than silently flattening a MoE
    model into a dense one — is still worth keeping, inverted.
    """
    config = M3Config(**{**SMALL, "use_moe": True})
    exported = qwen3_config(config)
    moe_exported = __import__(
        "m3g4m1nd.export", fromlist=["qwen3_moe_config"]
    ).qwen3_moe_config(config)

    # The dense translation drops the experts, so it must never be the one used
    # for a MoE model. That is now enforced by export_qwen3 choosing the target.
    assert not hasattr(exported, "num_experts")
    assert moe_exported.num_experts == config.num_experts


# ------------------------------------------------- mixture of experts
def _moe_config(**overrides):
    return M3Config(**{
        **SMALL, "use_moe": True, "num_experts": 4, "num_experts_per_tok": 1, **overrides
    })


def test_moe_export_targets_qwen3moe_and_matches_numerically():
    """Was a NotImplementedError. The fused layout concatenates gate before up
    per expert; the numeric check adjudicated that rather than a guess — the
    other order is wrong by 1.4e-3, not by nothing."""
    from transformers import Qwen3MoeForCausalLM

    from m3g4m1nd.export import qwen3_moe_config, qwen3_moe_state_dict

    torch.manual_seed(7)
    config = _moe_config()
    ours = M3ForCausalLM(config).eval()
    theirs = Qwen3MoeForCausalLM(qwen3_moe_config(config)).eval()

    missing, unexpected = theirs.load_state_dict(qwen3_moe_state_dict(ours), strict=False)
    assert [k for k in missing if "inv_freq" not in k] == []
    assert sorted(unexpected) == []

    ids = torch.randint(0, config.vocab_size, (2, 10))
    assert compare(ours, theirs, ids) < 1e-5


def test_swapping_the_fused_halves_is_detectably_wrong():
    """Proves the ordering is verified, not assumed."""
    from transformers import Qwen3MoeForCausalLM

    from m3g4m1nd.export import qwen3_moe_config, qwen3_moe_state_dict

    torch.manual_seed(7)
    config = _moe_config()
    ours = M3ForCausalLM(config).eval()

    state = qwen3_moe_state_dict(ours)
    for key in list(state):
        if key.endswith("gate_up_proj"):
            gate, up = state[key].chunk(2, dim=1)
            state[key] = torch.cat([up, gate], dim=1)

    wrong = Qwen3MoeForCausalLM(qwen3_moe_config(config)).eval()
    wrong.load_state_dict(state, strict=False)
    assert compare(ours, wrong, torch.randint(0, 64, (2, 10))) > 1e-4


def test_every_layer_is_sparse_in_the_exported_config():
    """Qwen3Moe's `decoder_sparse_step` defaults to 0, which makes no layer
    sparse — a MoE model exported as a dense one under a MoE name."""
    from m3g4m1nd.export import qwen3_moe_config

    exported = qwen3_moe_config(_moe_config())
    assert exported.decoder_sparse_step == 1
    assert exported.mlp_only_layers == []


def test_top1_routing_does_not_inherit_weight_normalisation():
    """Ours keeps the raw softmax probability at k=1; Qwen3Moe with
    norm_topk_prob=True would divide it by itself and pass exactly 1.0, which
    is a different model."""
    from m3g4m1nd.export import qwen3_moe_config

    assert qwen3_moe_config(_moe_config(num_experts_per_tok=1)).norm_topk_prob is False
    assert qwen3_moe_config(_moe_config(num_experts_per_tok=2)).norm_topk_prob is True


def test_moe_export_writes_a_loadable_checkpoint(tmp_path):
    from transformers import Qwen3MoeForCausalLM

    torch.manual_seed(7)
    ours = M3ForCausalLM(_moe_config()).eval()
    out = export_qwen3(ours, tmp_path / "moe")

    reloaded = Qwen3MoeForCausalLM.from_pretrained(out).eval()
    assert compare(ours, reloaded, torch.randint(0, 64, (2, 8))) < 1e-5
