"""Low-rank adaptation.

Two MiniMind defects live here.

**Targets were chosen by a shape coincidence.** `apply_lora` selected every
linear layer where `in_features == out_features`. That was shorthand for "the
attention projections", but at the default shape it catches only the query and
output projections — the key and value projections are 768->384 and the whole
feed-forward is 768->2432. So "fine-tune the model cheaply" quietly adapted 2
of 7 matrices per layer, and if you changed `num_key_value_heads` to 8, two
more silently joined in. Here targets are selected by name, the default list is
explicit, and it is configurable.

**Adapters loaded partially and silently.** `load_lora` used
`strict=False` with substring name matching, so an adapter built for a
different target list, rank, or model would load whatever happened to match and
leave the rest at initialisation — a wrong model that runs. Here the adapter
records what it was built for, and a mismatch raises.
"""

from __future__ import annotations

import torch
from torch import nn

# Every projection in the block. MiniMind's shape test reached only q and o.
DEFAULT_TARGETS: tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)

_META_TARGETS = "__lora_targets__"
_META_RANK = "__lora_rank__"
_META_ALPHA = "__lora_alpha__"


class LoRA(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float | None = None):
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        self.rank = rank
        self.alpha = float(rank if alpha is None else alpha)
        self.register_buffer("scaling", torch.tensor(self.alpha / rank, dtype=torch.float32))
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()  # so the adapted model starts identical to the base

    def forward(self, x):
        return self.B(self.A(x)) * self.scaling

    def delta(self) -> torch.Tensor:
        """The weight change this adapter represents, in fp32."""
        return (self.B.weight.data.float() @ self.A.weight.data.float()) * self.scaling.float()


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float | None = None,
    targets: tuple[str, ...] | None = None,
) -> list[str]:
    """Attach adapters to the named projections. Returns the layers adapted."""
    targets = tuple(targets or DEFAULT_TARGETS)
    device = next(model.parameters()).device
    adapted: list[str] = []

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name.split(".")[-1] not in targets:
            continue
        lora = LoRA(module.in_features, module.out_features, rank, alpha).to(device)
        module.add_module("lora", lora)
        original_forward = module.forward

        def forward_with_lora(x, _base=original_forward, _lora=lora):
            return _base(x) + _lora(x)

        module.forward = forward_with_lora
        adapted.append(name)

    if not adapted:
        raise ValueError(
            f"no layers matched targets {targets}. Available leaf names include: "
            f"{sorted({n.split('.')[-1] for n, m in model.named_modules() if isinstance(m, nn.Linear)})}"
        )
    model._lora_targets = targets
    model._lora_rank = rank
    return adapted


def _unwrap(model: nn.Module) -> nn.Module:
    model = getattr(model, "module", model)
    return getattr(model, "_orig_mod", model)


def save_lora(model: nn.Module, path) -> None:
    """Write adapter weights plus the metadata needed to refuse a bad load."""
    raw = _unwrap(model)
    state: dict = {}
    for name, module in raw.named_modules():
        lora = getattr(module, "lora", None)
        if lora is None:
            continue
        for key, value in lora.state_dict().items():
            state[f"{name}.lora.{key}"] = value.cpu()
    if not state:
        raise ValueError("model has no LoRA adapters; call apply_lora first")

    state[_META_TARGETS] = list(getattr(raw, "_lora_targets", DEFAULT_TARGETS))
    state[_META_RANK] = int(getattr(raw, "_lora_rank", 0))
    state[_META_ALPHA] = float(next(
        m.lora.alpha for _, m in raw.named_modules() if getattr(m, "lora", None) is not None
    ))
    torch.save(state, path)


def load_lora(model: nn.Module, path) -> None:
    """Load an adapter, refusing anything that does not match this model.

    MiniMind loaded whatever matched and left the rest at initialisation, which
    produces a model that runs and is wrong. A mismatch here is an error.
    """
    raw = _unwrap(model)
    state = torch.load(path, map_location="cpu", weights_only=True)

    saved_targets = tuple(state.pop(_META_TARGETS, ()) or ())
    saved_rank = state.pop(_META_RANK, None)
    state.pop(_META_ALPHA, None)

    live_targets = tuple(getattr(raw, "_lora_targets", ()) or ())
    live_rank = getattr(raw, "_lora_rank", None)

    if saved_targets and live_targets and saved_targets != live_targets:
        raise ValueError(
            f"adapter was built for targets {saved_targets} but this model was adapted for "
            f"{live_targets}. Loading it would leave the difference at initialisation."
        )
    if saved_rank and live_rank and saved_rank != live_rank:
        raise ValueError(f"adapter rank {saved_rank} does not match model rank {live_rank}")

    holders = {name: m for name, m in raw.named_modules() if getattr(m, "lora", None) is not None}
    if not holders:
        raise ValueError("model has no LoRA adapters; call apply_lora before load_lora")

    expected = {f"{n}.lora.{k}" for n in holders for k in holders[n].lora.state_dict()}
    missing, unexpected = expected - set(state), set(state) - expected
    if missing or unexpected:
        raise ValueError(
            f"adapter does not fit this model. missing={sorted(missing)[:4]} "
            f"unexpected={sorted(unexpected)[:4]}"
        )

    for name, module in holders.items():
        module.lora.load_state_dict(
            {k.removeprefix(f"{name}.lora."): v for k, v in state.items()
             if k.startswith(f"{name}.lora.")},
            strict=True,
        )


def merge_lora(model: nn.Module, lora_path, save_path, dtype=torch.float16) -> None:
    """Fold adapters into the base weights.

    Accumulate in fp32, round **once** at the end. MiniMind rounded the base
    weight to fp16, added the delta, and rounded again — losing precision twice
    on a quantity that is deliberately small. (MiniMind PR #919, whose test we
    also keep.)
    """
    load_lora(model, lora_path)
    raw = _unwrap(model)

    merged: dict[str, torch.Tensor] = {}
    for key, value in raw.state_dict().items():
        if ".lora." not in key:
            merged[key] = value.detach().cpu().float()

    for name, module in raw.named_modules():
        lora = getattr(module, "lora", None)
        if lora is None:
            continue
        key = f"{name}.weight"
        if key not in merged:
            raise KeyError(f"adapted module {name} has no base weight in the state dict")
        merged[key] = merged[key] + lora.delta().cpu()

    torch.save({k: v.to(dtype) for k, v in merged.items()}, save_path)
