"""The M3g4M1nd causal language model.

Decoder-only transformer: pre-norm RMSNorm, grouped-query attention with
per-head query/key normalisation, SwiGLU feed-forward, rotary position
embeddings with optional YaRN extrapolation, and an optional mixture-of-experts
feed-forward. Laid out to match the Qwen3 configuration so exported weights
load in llama.cpp, vllm and ollama without custom code.

Four of the twelve MiniMind defects live in this file's territory, and each one
is answered by a design decision rather than a patch:

1.  **One mask, two paths.** MiniMind branched between fused attention and a
    manual fallback, each building its own mask, and never compared them. Here
    a single additive bias carries both causality and padding, and both paths
    consume it unchanged — so they agree by construction, and a test proves it.

2.  **A finite floor, not -inf.** A fully-masked row — which a left-padded
    batch produces routinely — softmaxes to NaN under -inf. The bias uses
    finfo.min/2, so such a row yields a harmless uniform distribution that the
    output mask then discards.

3.  **Greedy means greedy.** Temperature, top-k and top-p are sampling
    controls and are not consulted when do_sample is false. Repetition penalty
    is not a sampling control and still applies.

4.  **Running off the position table is an error.** MiniMind sliced its rotary
    table past the end, got an empty tensor, and died in a broadcast several
    frames later. Here it raises and says what the limit was.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN

from .config import M3Config
from .outputs import CausalOutput


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = x.float()
        normed = normed * torch.rsqrt(normed.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * normed).type_as(x)


def build_rope_cache(
    head_dim: int,
    end: int,
    rope_theta: float = 1e6,
    rope_parameters: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables, optionally stretched by YaRN.

    YaRN: f'(i) = f(i)((1 - g) + g/s) where g is a linear ramp over the
    dimensions whose wavelength falls between beta_fast and beta_slow.
    """
    freqs = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2)[: head_dim // 2].float() / head_dim))
    attn_factor = 1.0

    if rope_parameters and rope_parameters.get("rope_type") == "yarn":
        orig_max = rope_parameters.get("original_max_position_embeddings", 2048)
        factor = rope_parameters.get("factor", 16)
        beta_fast = rope_parameters.get("beta_fast", 32.0)
        beta_slow = rope_parameters.get("beta_slow", 1.0)
        attn_factor = rope_parameters.get("attention_factor", 1.0)
        if end / orig_max > 1.0:
            def inv_dim(b):
                return (head_dim * math.log(orig_max / (b * 2 * math.pi))) / (2 * math.log(rope_theta))

            low = max(math.floor(inv_dim(beta_fast)), 0)
            high = min(math.ceil(inv_dim(beta_slow)), head_dim // 2 - 1)
            ramp = torch.clamp(
                (torch.arange(head_dim // 2).float() - low) / max(high - low, 0.001), 0, 1
            )
            freqs = freqs * (1 - ramp + ramp / factor)

    angles = torch.outer(torch.arange(end), freqs).float()
    cos = torch.cat([angles.cos(), angles.cos()], dim=-1) * attn_factor
    sin = torch.cat([angles.sin(), angles.sin()], dim=-1) * attn_factor
    return cos, sin


def apply_rotary(q, k, cos, sin):
    def rotate_half(x):
        half = x.shape[-1] // 2
        return torch.cat((-x[..., half:], x[..., :half]), dim=-1)

    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (
        ((q * cos) + (rotate_half(q) * sin)).to(q.dtype),
        ((k * cos) + (rotate_half(k) * sin)).to(k.dtype),
    )


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bs, slen, n_kv, hd = x.shape
    return x[:, :, :, None, :].expand(bs, slen, n_kv, n_rep, hd).reshape(bs, slen, n_kv * n_rep, hd)


def attention_bias(
    attention_mask: torch.Tensor | None,
    batch: int,
    q_len: int,
    past_len: int,
    device,
    dtype,
) -> torch.Tensor:
    """One additive bias carrying causality *and* padding, for both paths.

    Masked entries get finfo(dtype).min / 2 rather than -inf: a row with every
    key masked — ordinary in a left-padded batch — would otherwise softmax to
    NaN and poison the whole batch. With a finite floor it becomes a uniform
    distribution, which the caller's output mask discards anyway.
    """
    k_len = past_len + q_len
    q_pos = torch.arange(past_len, past_len + q_len, device=device).unsqueeze(1)
    k_pos = torch.arange(k_len, device=device).unsqueeze(0)
    allowed = k_pos <= q_pos  # [q_len, k_len]
    allowed = allowed.unsqueeze(0).unsqueeze(0).expand(batch, 1, q_len, k_len)

    if attention_mask is not None:
        if attention_mask.dim() != 2:
            raise ValueError(
                f"attention_mask must be [batch, key_length]; got {tuple(attention_mask.shape)}"
            )
        if attention_mask.size(1) != k_len:
            raise ValueError(
                f"attention_mask covers {attention_mask.size(1)} keys but there are {k_len} "
                f"({past_len} cached + {q_len} new)"
            )
        # Accept bool or integer masks identically by normalising once, here.
        pad = attention_mask.to(torch.bool)[:, None, None, :]
        allowed = allowed & pad

    bias = torch.zeros(batch, 1, q_len, k_len, device=device, dtype=dtype)
    return bias.masked_fill_(~allowed, torch.finfo(dtype).min / 2)


class Attention(nn.Module):
    def __init__(self, config: M3Config):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.n_rep = config.num_key_value_groups
        self.head_dim = config.head_dim
        self.dropout = config.dropout
        self.use_fused = config.flash_attn and hasattr(F, "scaled_dot_product_attention")

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, bias=None):
        bsz, q_len, _ = x.shape
        q = self.q_proj(x).view(bsz, q_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, q_len, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, q_len, self.n_kv_heads, self.head_dim)
        q, k = self.q_norm(q), self.k_norm(k)

        cos, sin = position_embeddings
        q, k = apply_rotary(q, k, cos, sin)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)
        present = (k, v) if use_cache else None

        q = q.transpose(1, 2)
        k = repeat_kv(k, self.n_rep).transpose(1, 2)
        v = repeat_kv(v, self.n_rep).transpose(1, 2)

        if self.use_fused:
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=bias, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if bias is not None:
                scores = scores + bias
            out = self.attn_dropout(F.softmax(scores, dim=-1).type_as(q)) @ v

        out = out.transpose(1, 2).reshape(bsz, q_len, -1)
        return self.resid_dropout(self.o_proj(out)), present


class FeedForward(nn.Module):
    def __init__(self, config: M3Config, intermediate_size: int | None = None):
        super().__init__()
        inter = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoEFeedForward(nn.Module):
    """Top-k routed experts, no shared expert.

    The load-balancing term is the MiniMind defect this class answers. There,
    `one_hot(idx, E).float().mean(0)` left the k axis in place, so the term was
    multiplied by k for any top-k above 1 — correct only at the default of 1,
    and wrong in a way no reader would see. Here the k axis is summed away
    first, giving the fraction of tokens routed to each expert, which is what
    the balancing term is defined over.
    """

    def __init__(self, config: M3Config):
        super().__init__()
        self.config = config
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            FeedForward(config, intermediate_size=config.moe_intermediate_size)
            for _ in range(config.num_experts)
        )
        self.aux_loss = None

    def forward(self, x):
        bsz, seq_len, hidden = x.shape
        flat = x.view(-1, hidden)
        scores = F.softmax(self.gate(flat), dim=-1)
        topk_w, topk_idx = torch.topk(scores, k=self.config.num_experts_per_tok, dim=-1, sorted=False)

        if self.config.norm_topk_prob and self.config.num_experts_per_tok > 1:
            topk_w = topk_w / (topk_w.sum(dim=-1, keepdim=True) + 1e-20)

        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            hit = topk_idx == i
            if hit.any():
                tokens = hit.any(dim=-1).nonzero().flatten()
                weight = topk_w[hit].view(-1, 1)
                out.index_add_(0, tokens, (expert(flat[tokens]) * weight).to(out.dtype))
            elif self.training:
                # Keep every expert in the autograd graph so DDP does not trip
                # over a parameter that produced no gradient this step.
                out = out + 0.0 * sum(p.sum() for p in expert.parameters())

        if self.training and self.config.router_aux_loss_coef > 0:
            n_exp = self.config.num_experts
            # fraction of tokens routed to each expert (k axis summed, not averaged)
            load = F.one_hot(topk_idx, n_exp).float().sum(dim=1).mean(dim=0)
            importance = scores.mean(dim=0)
            self.aux_loss = (load * importance).sum() * n_exp * self.config.router_aux_loss_coef
        else:
            self.aux_loss = scores.new_zeros(())

        return out.view(bsz, seq_len, hidden)


class Block(nn.Module):
    def __init__(self, config: M3Config):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = MoEFeedForward(config) if config.use_moe else FeedForward(config)

    def forward(self, hidden, position_embeddings, past_key_value=None, use_cache=False, bias=None):
        attn_out, present = self.self_attn(
            self.input_layernorm(hidden), position_embeddings, past_key_value, use_cache, bias
        )
        hidden = hidden + attn_out
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden, present


class M3Model(nn.Module):
    """The transformer body. Returns normalised hidden states.

    It applies the final norm itself. That is stated here because MiniMind's
    PPO critic assumed the opposite and normalised a second time.
    """

    def __init__(self, config: M3Config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(Block(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        cos, sin = build_rope_cache(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.rope_parameters,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def _rope_slice(self, past_len: int, q_len: int, device):
        limit = self.config.max_position_embeddings
        if past_len + q_len > limit:
            raise ValueError(
                f"sequence of {past_len + q_len} tokens exceeds max_position_embeddings="
                f"{limit} ({past_len} cached + {q_len} new). Raise "
                f"max_position_embeddings, or enable inference_rope_scaling to extrapolate."
            )
        if self.rope_cos.device != device:
            self.rope_cos = self.rope_cos.to(device)
            self.rope_sin = self.rope_sin.to(device)
        return self.rope_cos[past_len : past_len + q_len], self.rope_sin[past_len : past_len + q_len]

    def forward(self, input_ids, attention_mask=None, past_key_values=None, use_cache=False):
        bsz, q_len = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        past_len = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = self._rope_slice(past_len, q_len, hidden.device)
        bias = attention_bias(
            attention_mask, bsz, q_len, past_len, hidden.device, hidden.dtype
        )

        presents = []
        for layer, past in zip(self.layers, past_key_values):
            if self.gradient_checkpointing and self.training:
                if use_cache:
                    raise ValueError(
                        "gradient checkpointing recomputes the forward pass, so a KV "
                        "cache built during it would be discarded. Disable one."
                    )
                hidden, present = torch.utils.checkpoint.checkpoint(
                    layer, hidden, position_embeddings, past, use_cache, bias,
                    use_reentrant=False,
                )
            else:
                hidden, present = layer(hidden, position_embeddings, past, use_cache, bias)
            presents.append(present)

        hidden = self.norm(hidden)
        aux = sum(
            (layer.mlp.aux_loss for layer in self.layers if isinstance(layer.mlp, MoEFeedForward)),
            hidden.new_zeros(()),
        )
        return hidden, presents, aux


class M3ForCausalLM(PreTrainedModel):
    config_class = M3Config
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    supports_gradient_checkpointing = True

    def __init__(self, config: M3Config | None = None):
        config = config or M3Config()
        super().__init__(config)
        self.model = M3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def gradient_checkpointing_enable(self, **_) -> None:
        """Trade compute for memory: recompute each block's forward in the
        backward pass instead of holding its activations."""
        self.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.model.gradient_checkpointing = False

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self.model.gradient_checkpointing

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
        labels=None,
        logits_to_keep=0,
        **_,
    ):
        hidden, presents, aux_loss = self.model(
            input_ids, attention_mask, past_key_values, use_cache
        )
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden[:, keep, :])

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return CausalOutput(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=presents,
            hidden_states=hidden,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids=None,
        attention_mask=None,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.85,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        eos_token_id: int | None = None,
        use_cache: bool = True,
        num_return_sequences: int = 1,
        streamer=None,
        **kwargs,
    ):
        """Token-at-a-time decoding.

        Sampling controls are consulted only when sampling. MiniMind divided
        logits by temperature before checking whether it was sampling, so a
        greedy request with temperature=0 produced inf/NaN, and greedy requests
        were silently reshaped by top-k and top-p. Repetition penalty is not a
        sampling control and applies either way.
        """
        if input_ids is None:
            input_ids = kwargs.pop("inputs", None)
        if input_ids is None:
            raise ValueError("generate needs input_ids")
        if do_sample and not 0.0 < temperature < math.inf:
            raise ValueError(
                f"sampling temperature must be finite and positive, got {temperature!r}. "
                f"For greedy decoding pass do_sample=False."
            )
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        input_ids = input_ids.repeat(num_return_sequences, 1)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat(num_return_sequences, 1)
        else:
            attention_mask = torch.ones_like(input_ids)

        past_key_values = None
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        if streamer is not None:
            streamer.put(input_ids.cpu())

        for _ in range(max_new_tokens):
            past_len = past_key_values[0][0].shape[1] if past_key_values else 0
            out = self.forward(
                input_ids[:, past_len:],
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
            )
            logits = out.logits[:, -1].float()

            if repetition_penalty != 1.0:
                for row in range(input_ids.size(0)):
                    seen = torch.unique(input_ids[row])
                    score = logits[row, seen]
                    logits[row, seen] = torch.where(
                        score > 0, score / repetition_penalty, score * repetition_penalty
                    )

            if do_sample:
                logits = logits / temperature
                if top_k and top_k > 0:
                    kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                    logits = logits.masked_fill(logits < kth, -float("inf"))
                if top_p < 1.0:
                    ordered, order = torch.sort(logits, descending=True)
                    cutoff = torch.cumsum(F.softmax(ordered, dim=-1), dim=-1) > top_p
                    cutoff = F.pad(cutoff[..., :-1], (1, 0))
                    logits = logits.masked_fill(cutoff.scatter(1, order, cutoff), -float("inf"))
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_token_id is not None:
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    next_token.new_full((next_token.size(0), 1), eos_token_id),
                    next_token,
                )

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones(attention_mask.size(0), 1)], dim=-1
            )
            past_key_values = out.past_key_values if use_cache else None
            if streamer is not None:
                streamer.put(next_token.cpu())

            if eos_token_id is not None:
                finished |= next_token.squeeze(-1).eq(eos_token_id)
                if bool(finished.all()):
                    break

        if streamer is not None:
            streamer.end()
        return input_ids


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """(total, trainable). Tied weights counted once, as named_parameters dedupes."""
    seen = {id(p): p for p in model.parameters()}
    total = sum(p.numel() for p in seen.values())
    trainable = sum(p.numel() for p in seen.values() if p.requires_grad)
    return total, trainable
