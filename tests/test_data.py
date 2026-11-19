"""Row C2 — every constructor parameter changes behaviour.

MiniMind's `RLAIFDataset` and `AgentRLDataset` each took a `max_length` they
stored and never applied. The knob looked live and was not, and the config
detour feeding it was dead too. This file checks both ways: the AST, for
parameters that are stored and never read, and the behaviour, at the boundary.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import pytest
import torch

from m3g4m1nd import data
from m3g4m1nd.data import IGNORE, PretrainDataset, SFTDataset, read_jsonl
from m3g4m1nd.tokenizer import load_tokenizer

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"
DATASET_CLASSES = (PretrainDataset, SFTDataset)


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(ROOT / "model")


# ---------------------------------------------------------------- C2, statically
def _stored_but_unread(cls) -> set[str]:
    """Attributes assigned in __init__ and never read anywhere in the class."""
    # dedent, not cleandoc: cleandoc normalises docstring whitespace and
    # destroys code indentation. A nested class is indented, so it needs this.
    tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    class_node = tree.body[0]

    assigned, read = set(), set()
    for node in ast.walk(class_node):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id != "self":
                continue
            if isinstance(node.ctx, ast.Store):
                assigned.add(node.attr)
            else:
                read.add(node.attr)
    return assigned - read


@pytest.mark.parametrize("cls", DATASET_CLASSES, ids=lambda c: c.__name__)
def test_no_dataset_stores_a_value_it_never_reads(cls):
    orphans = _stored_but_unread(cls)
    assert not orphans, (
        f"{cls.__name__} stores {sorted(orphans)} and never reads it. That is the "
        f"MiniMind defect: a knob that looks live and is not."
    )


@pytest.mark.parametrize("cls", DATASET_CLASSES, ids=lambda c: c.__name__)
def test_every_constructor_parameter_is_stored(cls):
    parameters = set(inspect.signature(cls.__init__).parameters) - {"self"}
    source = inspect.getsource(cls)
    for name in parameters:
        assert f"self.{name}" in source or name in ("path", "tokenizer"), (
            f"{cls.__name__}.{name} is accepted and goes nowhere"
        )


def test_the_detector_would_have_caught_miniminds_version():
    """A control: build the defect and confirm the detector sees it."""

    class Offender:
        def __init__(self, max_length):
            self.max_length = max_length

        def __getitem__(self, index):
            return index  # never consults self.max_length

    assert "max_length" in _stored_but_unread(Offender)


# ---------------------------------------------------------------- C2, behaviourally
def test_pretrain_max_length_is_exact_at_the_boundary(tok):
    for width in (8, 16, 64):
        dataset = PretrainDataset(FIXTURES / "corpus.jsonl", tok, max_length=width)
        ids, labels = dataset[1]
        assert ids.shape == (width,), f"max_length={width} produced {tuple(ids.shape)}"
        assert labels.shape == (width,)


def test_pretrain_truncation_actually_truncates_a_long_row(tok):
    long_text = " ".join(["scattering"] * 400)
    path = FIXTURES / "tmp_long.jsonl"
    path.write_text('{"text": "' + long_text + '"}\n', encoding="utf-8")
    try:
        narrow = PretrainDataset(path, tok, max_length=12)
        wide = PretrainDataset(path, tok, max_length=64)
        assert narrow[0][0].shape == (12,)
        assert wide[0][0].shape == (64,)
        # The narrow one must have no padding: the row overflows it.
        assert (narrow[0][1] == IGNORE).sum() == 0
    finally:
        path.unlink()


def test_pretrain_brackets_each_row_with_bos_and_eos(tok):
    dataset = PretrainDataset(FIXTURES / "corpus.jsonl", tok, max_length=64)
    ids, labels = dataset[0]
    assert ids[0].item() == tok.bos_token_id
    end = (labels != IGNORE).sum().item()
    assert ids[end - 1].item() == tok.eos_token_id


def test_pretrain_masks_padding_out_of_the_loss(tok):
    dataset = PretrainDataset(FIXTURES / "corpus.jsonl", tok, max_length=96)
    ids, labels = dataset[0]
    padding = ids == tok.pad_token_id
    assert padding.any(), "widen max_length until this row pads"
    assert (labels[padding] == IGNORE).all()


def test_pretrain_refuses_a_max_length_with_no_room_for_content(tok):
    with pytest.raises(ValueError, match="room for bos"):
        PretrainDataset(FIXTURES / "corpus.jsonl", tok, max_length=2)


def test_sft_supervises_only_the_assistant_turns(tok):
    dataset = SFTDataset(FIXTURES / "sft.jsonl", tok, max_length=128)
    ids, labels = dataset[0]
    supervised = labels != IGNORE
    assert supervised.any()

    text = tok.decode(ids[supervised], skip_special_tokens=False)
    assert "Hello there" in text
    assert "Hi" not in text.replace("<|im_end|>", ""), "the user's turn is being trained on"


def test_sft_trains_on_the_turn_ending_token(tok):
    """The stop token must be learnable. MiniMind's agent trainer masked it."""
    dataset = SFTDataset(FIXTURES / "sft.jsonl", tok, max_length=128)
    ids, labels = dataset[0]
    supervised_ids = ids[labels != IGNORE].tolist()
    assert tok.eos_token_id in supervised_ids, (
        "the model is never shown where its own turn ends, so it cannot learn to stop"
    )


def test_sft_supervises_every_assistant_turn_in_a_multi_turn_row(tok):
    dataset = SFTDataset(FIXTURES / "sft.jsonl", tok, max_length=192)
    ids, labels = dataset[1]
    text = tok.decode(ids[labels != IGNORE], skip_special_tokens=False)
    assert "Rayleigh" in text and "Longer path" in text


def test_train_on_last_turn_only_narrows_the_supervision(tok):
    both = SFTDataset(FIXTURES / "sft.jsonl", tok, max_length=192)
    last = SFTDataset(FIXTURES / "sft.jsonl", tok, max_length=192, train_on_last_turn_only=True)

    text_both = tok.decode(both[1][0][both[1][1] != IGNORE], skip_special_tokens=False)
    text_last = tok.decode(last[1][0][last[1][1] != IGNORE], skip_special_tokens=False)

    assert "Rayleigh" in text_both
    assert "Rayleigh" not in text_last
    assert "Longer path" in text_last


def test_sft_max_length_is_exact_and_truncation_bites(tok):
    narrow = SFTDataset(FIXTURES / "sft.jsonl", tok, max_length=16)
    wide = SFTDataset(FIXTURES / "sft.jsonl", tok, max_length=192)
    assert narrow[1][0].shape == (16,)
    assert wide[1][0].shape == (192,)
    assert (narrow[1][1] != IGNORE).sum() <= (wide[1][1] != IGNORE).sum()


def test_sft_handles_a_tool_calling_row(tok):
    dataset = SFTDataset(FIXTURES / "sft.jsonl", tok, max_length=192)
    ids, labels = dataset[2]
    text = tok.decode(ids[labels != IGNORE], skip_special_tokens=False)
    assert "27 degrees" in text
    assert "tool_response" not in text, "the tool's own output is being trained on"


def test_jsonl_reader_names_the_offending_line(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"text": "ok"}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        read_jsonl(bad)


def test_an_empty_file_is_refused(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contains no rows"):
        read_jsonl(empty)
