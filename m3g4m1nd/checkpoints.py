"""Loading weights, strictly.

This module exists because of a silent failure found by actually running the
pipeline end to end rather than testing its parts.

`Trainer.save` writes a resume checkpoint — a dict of `model`, `optimizer`,
`epoch`, `plan` and so on. The CLI's `--from-weights` passed that straight to
`load_state_dict(..., strict=False)`, which found none of the keys it wanted,
reported nothing, and left the model at its random initialisation. The SFT
stage ran for thirty epochs from scratch and printed "saved".

`strict=False` forgives missing keys. It does not distinguish "this checkpoint
is a slightly different architecture" from "this checkpoint is not weights at
all", and that is the entire problem. So:

- `load_weights` accepts either format, unwrapping a resume checkpoint, and
  loads strictly by default;
- a mismatch raises, with the counts, rather than loading a fraction;
- `--from-weights` cannot silently do nothing again.
"""

from __future__ import annotations

from pathlib import Path

import torch

RESUME_KEYS = {"model", "optimizer", "epoch", "plan"}


def is_resume_checkpoint(payload) -> bool:
    return isinstance(payload, dict) and RESUME_KEYS.issubset(payload)


def read_state_dict(path: str | Path) -> dict:
    """Read either a bare state dict or a resume checkpoint, returning weights."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if is_resume_checkpoint(payload):
        return payload["model"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{path} does not contain a state dict")
    non_tensor = [k for k, v in payload.items() if not torch.is_tensor(v)]
    if non_tensor:
        raise ValueError(
            f"{path} is not weights: {len(non_tensor)} of {len(payload)} entries are "
            f"not tensors (for example {non_tensor[:3]}). If it is a resume "
            f"checkpoint it is missing one of {sorted(RESUME_KEYS)}."
        )
    return payload


def load_weights(model, path: str | Path, strict: bool = True) -> None:
    """Load weights into `model`, refusing a checkpoint that does not fit.

    With `strict=False` the mismatch is reported rather than swallowed: a
    checkpoint that supplies none of the model's parameters is an error at any
    strictness, because it means nothing was loaded.
    """
    state = read_state_dict(path)
    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
    except RuntimeError as error:
        # strict=False forgives absent keys but not wrong shapes, and torch's
        # message buries the cause in a wall of tensor sizes.
        raise ValueError(
            f"{path} has parameters of the wrong shape for this model. The "
            f"checkpoint's own shape can be recovered with "
            f"m3g4m1nd.export.config_from_state_dict. Detail: {error}"
        ) from None

    supplied = len(model.state_dict()) - len(missing)
    if supplied == 0:
        raise ValueError(
            f"{path} supplied none of the model's {len(model.state_dict())} "
            f"parameters, so nothing would be loaded. It has "
            f"{len(unexpected)} keys the model does not want, for example "
            f"{sorted(unexpected)[:3]}."
        )
    if strict and (missing or unexpected):
        raise ValueError(
            f"{path} does not fit this model: {len(missing)} parameters missing, "
            f"{len(unexpected)} unexpected. missing={sorted(missing)[:3]} "
            f"unexpected={sorted(unexpected)[:3]}"
        )
