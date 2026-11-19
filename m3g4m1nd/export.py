"""Export to the Qwen3 layout, so the weights load in the wider ecosystem.

The MiniMind defect here is one line. Its exporter built the target config with

    "head_dim": lm_config.hidden_size // lm_config.num_attention_heads

— recomputing a value the source config already holds and allows you to set
independently. At the default shape the two agree, so it works; set `head_dim`
to anything else and the export is silently wrong in the one field that decides
how every attention weight is interpreted.

So: read the field, never recompute it, and prove the export numerically rather
than trusting that the names line up. `export_qwen3` ends by running both
models on the same input and refusing to write a mismatch.
"""

from __future__ import annotations

from pathlib import Path

import torch

from .config import M3Config
from .model import M3ForCausalLM


def qwen3_config(config: M3Config):
    """Translate our config into Qwen3's, reading every field rather than
    rederiving any of it."""
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        # The line MiniMind got wrong. Read, not recomputed.
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        # Transformers 5.x folds rope_theta into rope_parameters and no longer
        # exposes it as an attribute. Passing it the canonical way rather than
        # relying on the compatibility shim that accepts the old keyword.
        rope_parameters=dict(config.rope_parameters),
        tie_word_embeddings=config.tie_word_embeddings,
        hidden_act=config.hidden_act,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
    )


def qwen3_state_dict(model: M3ForCausalLM) -> dict[str, torch.Tensor]:
    """Our parameter names already mirror Qwen3's, so this is a copy with the
    rotary buffers dropped. Asserting that rather than assuming it."""
    state = {}
    for key, value in model.state_dict().items():
        if key.endswith(("rope_cos", "rope_sin")):
            continue  # derived, non-persistent
        state[key] = value.detach().cpu().clone()
    return state


def compare(ours: M3ForCausalLM, theirs, input_ids: torch.Tensor) -> float:
    """Largest absolute logit difference on the same input."""
    ours.eval()
    theirs.eval()
    with torch.no_grad():
        mine = ours(input_ids).logits.float()
        yours = theirs(input_ids).logits.float()
    return (mine - yours).abs().max().item()


def qwen3_moe_config(config: M3Config):
    """Translate our MoE config into Qwen3Moe's.

    One semantic difference has to be handled rather than hoped over. Our
    router normalises the top-k weights only when k > 1 — at k = 1 the single
    expert keeps its raw softmax probability. Qwen3Moe with
    `norm_topk_prob=True` would divide that probability by itself and feed the
    expert a weight of exactly 1.0, which is a different model. So the flag is
    translated, not copied.
    """
    from transformers import Qwen3MoeConfig

    return Qwen3MoeConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_parameters=dict(config.rope_parameters),
        tie_word_embeddings=config.tie_word_embeddings,
        hidden_act=config.hidden_act,
        attention_bias=False,
        use_sliding_window=False,
        sliding_window=None,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        moe_intermediate_size=config.moe_intermediate_size,
        norm_topk_prob=config.norm_topk_prob and config.num_experts_per_tok > 1,
        # Every layer is sparse. The default of 0 makes none of them sparse,
        # which would quietly export a dense model under a MoE name.
        decoder_sparse_step=1,
        mlp_only_layers=[],
    )


def qwen3_moe_state_dict(model: M3ForCausalLM) -> dict[str, torch.Tensor]:
    """Fuse our per-expert linear layers into Qwen3Moe's stacked tensors.

    Theirs holds, per layer:
        experts.gate_up_proj  (num_experts, 2 * moe_intermediate, hidden)
        experts.down_proj     (num_experts, hidden, moe_intermediate)

    Ours holds one `FeedForward` per expert. The gate half comes first in the
    fused tensor; `export_qwen3` runs both models on the same tokens and
    refuses to write a mismatch, so that ordering is verified rather than
    assumed.
    """
    config = model.config
    source = {k: v for k, v in model.state_dict().items()
              if not k.endswith(("rope_cos", "rope_sin"))}

    fused: dict[str, torch.Tensor] = {}
    for key, value in source.items():
        if ".mlp.experts." not in key:
            fused[key] = value.detach().cpu().clone()

    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}.mlp.experts"
        gate = torch.stack([
            source[f"{prefix}.{e}.gate_proj.weight"] for e in range(config.num_experts)
        ])
        up = torch.stack([
            source[f"{prefix}.{e}.up_proj.weight"] for e in range(config.num_experts)
        ])
        down = torch.stack([
            source[f"{prefix}.{e}.down_proj.weight"] for e in range(config.num_experts)
        ])
        fused[f"{prefix}.gate_up_proj"] = torch.cat([gate, up], dim=1).detach().cpu().clone()
        fused[f"{prefix}.down_proj"] = down.detach().cpu().clone()

    return fused


def export_qwen3(
    model: M3ForCausalLM,
    out_dir: str | Path,
    tokenizer_dir: str | Path | None = None,
    tolerance: float = 2e-4,
) -> Path:
    """Write a Qwen3-shaped checkpoint, verifying it before it lands.

    MiniMind's exporter wrote the file and told you it had worked. This one
    loads what it is about to write, runs both models on the same tokens, and
    raises rather than leave a plausible-looking checkpoint that decodes to
    rubbish.
    """
    from transformers import AutoTokenizer, Qwen3ForCausalLM, Qwen3MoeForCausalLM

    out_dir = Path(out_dir)
    if model.config.use_moe:
        target = Qwen3MoeForCausalLM(qwen3_moe_config(model.config))
        state = qwen3_moe_state_dict(model)
    else:
        target = Qwen3ForCausalLM(qwen3_config(model.config))
        state = qwen3_state_dict(model)

    missing, unexpected = target.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith("rotary_emb.inv_freq")]
    if real_missing or unexpected:
        raise ValueError(
            f"state dict does not fit Qwen3. missing={real_missing[:6]} "
            f"unexpected={sorted(unexpected)[:6]}"
        )

    probe = torch.randint(0, model.config.vocab_size, (2, 16))
    drift = compare(model, target, probe)
    if drift > tolerance:
        raise ValueError(
            f"exported model disagrees with the source by {drift:.2e} (tolerance "
            f"{tolerance:.0e}). Writing it would produce a checkpoint that loads "
            f"everywhere and is wrong."
        )

    target.save_pretrained(out_dir)
    if tokenizer_dir is not None:
        AutoTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True).save_pretrained(
            out_dir
        )
    return out_dir


def config_from_state_dict(state: dict) -> M3Config:
    """Recover the model shape from the weights themselves.

    The CLI used to ask for `--hidden-size` and `--layers` and derive the rest
    from defaults, so exporting anything else failed with seven size
    mismatches. The checkpoint already knows its own shape; asking the operator
    to retype it correctly is a question with a wrong answer available.
    """
    def need(key: str):
        if key not in state:
            raise ValueError(
                f"checkpoint is missing {key!r}, so its shape cannot be recovered. "
                f"Keys present include: {sorted(state)[:5]}"
            )
        return state[key]

    vocab_size, hidden_size = need("model.embed_tokens.weight").shape

    layers = {
        int(key.split(".")[2])
        for key in state
        if key.startswith("model.layers.") and len(key.split(".")) > 2
    }
    if not layers:
        raise ValueError("checkpoint contains no decoder layers")
    num_layers = max(layers) + 1

    head_dim = need("model.layers.0.self_attn.q_norm.weight").shape[0]
    q_rows = need("model.layers.0.self_attn.q_proj.weight").shape[0]
    k_rows = need("model.layers.0.self_attn.k_proj.weight").shape[0]

    expert_keys = [k for k in state if ".mlp.experts." in k]
    if expert_keys:
        experts = {int(k.split(".mlp.experts.")[1].split(".")[0]) for k in expert_keys}
        intermediate = state["model.layers.0.mlp.experts.0.gate_proj.weight"].shape[0]
        moe = {"use_moe": True, "num_experts": max(experts) + 1,
               "moe_intermediate_size": intermediate}
    else:
        intermediate = need("model.layers.0.mlp.gate_proj.weight").shape[0]
        moe = {"use_moe": False}

    return M3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=q_rows // head_dim,
        num_key_value_heads=k_rows // head_dim,
        head_dim=head_dim,
        intermediate_size=intermediate,
        tie_word_embeddings="lm_head.weight" not in state
        or state["lm_head.weight"].data_ptr() == state["model.embed_tokens.weight"].data_ptr(),
        **moe,
    )
