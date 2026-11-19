"""Model configuration.

The one rule this file exists to enforce: **a setting we do not recognise is an
error, not a note-to-self.**

MiniMind renamed `max_seq_len` to `max_position_embeddings` in April 2025 and
missed four call sites. Because its config subclassed a base class that accepts
arbitrary keyword arguments, those four sites kept passing a setting that was
silently filed away and never read by the model. Anyone tuning memory through
that knob was turning a dial wired to nothing, and nothing anywhere said so.

So: every field is named explicitly, renamed fields raise and say what to use
instead, and anything genuinely unknown raises too.
"""

from __future__ import annotations

import math

from transformers import PretrainedConfig

# Settings that used to exist under another name. Passing one is a hard error
# with a pointer, not a silent no-op. Add to this when you rename something.
RENAMED = {
    "max_seq_len": "max_position_embeddings",
    "n_layers": "num_hidden_layers",
    "dim": "hidden_size",
    "n_heads": "num_attention_heads",
    "n_kv_heads": "num_key_value_heads",
    "n_routed_experts": "num_experts",
    "n_shared_experts": None,  # removed outright; MoE here has no shared expert
    "aux_loss_alpha": "router_aux_loss_coef",
    # Transformers 5.x renamed this and serialises the new name into config.json.
    # Our strict check caught the mismatch on the first reload, so we use the
    # new name natively rather than deleting the key on export the way MiniMind
    # had to (scripts/convert_model.py: `del config['rope_parameters']`).
    "rope_scaling": "rope_parameters",
}

# Keys the Transformers machinery itself round-trips through config.json. These
# are permitted but are not ours; anything outside this set and our own fields
# is rejected.
_TRANSFORMERS_KEYS = frozenset({
    "architectures", "model_type", "torch_dtype", "dtype", "transformers_version",
    "tokenizer_class", "auto_map", "pad_token_id", "sep_token_id", "unk_token_id",
    "decoder_start_token_id", "is_encoder_decoder", "is_decoder", "add_cross_attention",
    "tie_encoder_decoder", "prefix", "task_specific_params", "problem_type",
    "id2label", "label2id", "num_labels", "finetuning_task", "output_hidden_states",
    "output_attentions", "return_dict", "use_return_dict", "chunk_size_feed_forward",
    "pruned_heads", "tf_legacy_loss", "use_bfloat16", "_commit_hash",
    "_attn_implementation", "_attn_implementation_autoset", "attn_implementation",
})


class M3Config(PretrainedConfig):
    """Configuration for an M3g4M1nd causal language model.

    Defaults reproduce the 63,912,192-parameter reference model: 8 layers of
    width 768, 8 query heads over 4 key/value heads, a 6,400-token vocabulary
    tied between the embedding and the output projection.
    """

    model_type = "m3g4m1nd"

    def __init__(
        self,
        *,
        hidden_size: int = 768,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 8,
        num_key_value_heads: int = 4,
        head_dim: int | None = None,
        intermediate_size: int | None = None,
        vocab_size: int = 6400,
        max_position_embeddings: int = 32768,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1e6,
        dropout: float = 0.0,
        flash_attn: bool = True,
        tie_word_embeddings: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        inference_rope_scaling: bool = False,
        rope_parameters: dict | None = None,
        # mixture-of-experts; ignored unless use_moe is true
        use_moe: bool = False,
        num_experts: int = 4,
        num_experts_per_tok: int = 1,
        moe_intermediate_size: int | None = None,
        norm_topk_prob: bool = True,
        router_aux_loss_coef: float = 5e-4,
        **kwargs,
    ):
        self._reject_unknown(kwargs)

        if num_attention_heads % num_key_value_heads:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be a multiple of "
                f"num_key_value_heads ({num_key_value_heads})"
            )
        if use_moe and not 1 <= num_experts_per_tok <= num_experts:
            raise ValueError(
                f"num_experts_per_tok ({num_experts_per_tok}) must be between 1 and "
                f"num_experts ({num_experts})"
            )

        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_attention_heads
        # MiniMind's width heuristic, kept because it is what makes the reference
        # parameter count what it is: pi * hidden, rounded up to a multiple of 64.
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else math.ceil(hidden_size * math.pi / 64) * 64
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.flash_attn = flash_attn
        self.inference_rope_scaling = inference_rope_scaling

        # Canonical shape for Transformers 5.x: one dict carrying rope_type and
        # rope_theta together. `rope_theta` is also kept as a plain attribute
        # because our own attention code reads it directly.
        if rope_parameters is not None:
            self.rope_parameters = dict(rope_parameters)
            self.rope_theta = self.rope_parameters.get("rope_theta", rope_theta)
        elif inference_rope_scaling:
            self.rope_parameters = {
                "rope_type": "yarn",
                "rope_theta": rope_theta,
                "factor": 16,
                "beta_fast": 32,
                "beta_slow": 1,
                "attention_factor": 1.0,
                "original_max_position_embeddings": 2048,
            }
        else:
            self.rope_parameters = {"rope_type": "default", "rope_theta": rope_theta}

        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = (
            moe_intermediate_size if moe_intermediate_size is not None else self.intermediate_size
        )
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef

        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @staticmethod
    def _reject_unknown(kwargs: dict) -> None:
        for key in sorted(kwargs):
            if key in RENAMED:
                replacement = RENAMED[key]
                if replacement is None:
                    raise TypeError(
                        f"{key!r} was removed from M3Config and has no replacement. "
                        f"It is not read by the model; passing it would do nothing."
                    )
                raise TypeError(
                    f"{key!r} was renamed to {replacement!r}. Passing {key!r} would be "
                    f"silently ignored by the model, which is the bug this check exists "
                    f"to prevent."
                )
            if key.startswith("_") or key in _TRANSFORMERS_KEYS:
                continue
            raise TypeError(
                f"M3Config got an unexpected keyword argument {key!r}. If this is a "
                f"real setting, add it as a named parameter; if it is a rename, add it "
                f"to RENAMED. It must not be accepted and ignored."
            )

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads
