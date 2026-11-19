"""Shared fixtures.

Every model built here is tiny and runs on a CPU in milliseconds. No test in
this repository downloads a dataset, reaches the network, or needs a GPU — the
constraint that made deleting MiniMind's six tests indefensible, since all six
had the same property.
"""

from __future__ import annotations

import pytest
import torch

from m3g4m1nd.config import M3Config
from m3g4m1nd.model import M3ForCausalLM

TINY = dict(
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=1,
    head_dim=8,
    intermediate_size=32,
    vocab_size=64,
    max_position_embeddings=128,
)


def tiny_config(**overrides):
    return M3Config(**{**TINY, **overrides})


def tiny_model(**overrides):
    torch.manual_seed(17)
    return M3ForCausalLM(tiny_config(**overrides)).eval()


@pytest.fixture
def tiny():
    return tiny_model()


@pytest.fixture
def reference_config():
    """The 63,912,192-parameter published shape."""
    return M3Config()
