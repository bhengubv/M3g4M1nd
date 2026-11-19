"""Model output container.

A plain dataclass rather than a Transformers output class. The Transformers
output hierarchy and its KV-cache objects have moved twice in two major
versions; MiniMind carries workarounds for exactly that churn, including a line
that silently discards a cache object it does not recognise and recomputes the
whole prefix. Owning the container costs twenty lines and removes that risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class CausalOutput:
    logits: torch.Tensor
    past_key_values: list | None = None
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None
    hidden_states: torch.Tensor | None = None

    def __getitem__(self, key):
        # Tolerate tuple-style access, which some training loops still use.
        return getattr(self, key) if isinstance(key, str) else (self.logits, self.past_key_values)[key]
