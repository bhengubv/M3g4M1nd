"""Rows C4 and E3 — the chat template and the declared length.

E3 is the smallest MiniMind defect and the easiest to have caught: its
tokenizer config declared a 131,072-token maximum against a 32,768-token
model. Nothing read the first number, so nothing noticed.

C4 was never tested in MiniMind at all. A chat template that does not
round-trip corrupts training data silently — the model is taught to predict
text that nobody can reproduce at inference.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from m3g4m1nd.config import M3Config
from m3g4m1nd.tokenizer import (
    CHAT_TEMPLATE,
    MARKUP_TOKENS,
    load_tokenizer,
    tokenizer_config,
    train_tokenizer,
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "model"
CORPUS = ROOT / "tests" / "fixtures" / "corpus.jsonl"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(MODEL_DIR)


def _roundtrips(tokenizer, text: str) -> bool:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return tokenizer.decode(ids, skip_special_tokens=False) == text


# ---------------------------------------------------------------- E3
def test_declared_maximum_equals_the_model_window(tok):
    assert tok.model_max_length == M3Config().max_position_embeddings == 32768


def test_the_declared_maximum_is_derived_not_typed():
    """It comes from the config object, so the two cannot drift apart."""
    narrow = tokenizer_config(M3Config(max_position_embeddings=2048))
    assert narrow["model_max_length"] == 2048


def test_the_committed_config_matches_what_the_code_would_generate(tok):
    on_disk = json.loads((MODEL_DIR / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert on_disk["model_max_length"] == tokenizer_config()["model_max_length"]
    assert on_disk["chat_template"] == CHAT_TEMPLATE


def test_the_trained_vocabulary_fits_inside_the_model_embedding(tok):
    """733 entries against 6,400 reserved rows: the spare rows are unused, which
    is fine and is stated in model/README.md. What must never happen is the
    tokenizer being larger than the embedding."""
    assert len(tok) <= M3Config().vocab_size


def test_the_special_tokens_are_the_ones_the_template_emits(tok):
    assert tok.bos_token == "<|im_start|>"
    assert tok.eos_token == "<|im_end|>"
    assert tok.pad_token == "<|endoftext|>"
    for token in MARKUP_TOKENS:
        assert len(tok(token, add_special_tokens=False).input_ids) == 1, (
            f"{token} should be a single token, not split into pieces"
        )


# ---------------------------------------------------------------- C4
def test_a_plain_conversation_round_trips(tok):
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        tokenize=False, add_generation_prompt=False,
    )
    assert _roundtrips(tok, rendered)
    assert rendered.startswith("<|im_start|>user\nHi<|im_end|>")


def test_a_conversation_with_a_system_turn_round_trips(tok):
    rendered = tok.apply_chat_template(
        [
            {"role": "system", "content": "Be brief."},
            {"role": "user", "content": "Why is the sky blue?"},
            {"role": "assistant", "content": "Rayleigh scattering."},
        ],
        tokenize=False, add_generation_prompt=False,
    )
    assert _roundtrips(tok, rendered)
    assert "<|im_start|>system\nBe brief.<|im_end|>" in rendered


def test_tools_thinking_and_tool_results_all_round_trip(tok):
    messages = [
        {"role": "system", "content": "You may use tools.", "tools": TOOLS},
        {"role": "user", "content": "Weather in Durban?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I should call the tool.",
            "tool_calls": [{"name": "get_weather", "arguments": {"city": "Durban"}}],
        },
        {"role": "tool", "content": '{"temperature": "27C"}'},
        {"role": "assistant", "content": "27 degrees and humid."},
    ]
    rendered = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, tools=TOOLS
    )
    assert _roundtrips(tok, rendered), "tool/thinking markup did not survive a round trip"
    for fragment in ("<tools>", "</tools>", "<tool_call>", "</tool_call>",
                     "<tool_response>", "<think>", "</think>", "get_weather", "Durban"):
        assert fragment in rendered, f"{fragment} missing from the rendered conversation"


def test_the_generation_prompt_opens_thinking_when_asked(tok):
    messages = [{"role": "user", "content": "Hi"}]
    opened = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, open_thinking=True
    )
    closed = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, open_thinking=False
    )
    assert opened.endswith("<|im_start|>assistant\n<think>\n")
    assert closed.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert _roundtrips(tok, opened) and _roundtrips(tok, closed)


def test_non_ascii_and_emoji_survive_byte_level_encoding(tok):
    for text in ("杭州今天天气怎么样？", "Je peux manger du verre", "🙂 ok", "a\tb\nc"):
        assert _roundtrips(tok, text), f"{text!r} did not round trip"


def test_an_empty_corpus_is_refused_rather_than_yielding_an_empty_vocabulary(tmp_path):
    """MiniMind trained on the wrong data format, got a vocabulary with no
    merges, and printed 'Tokenizer training completed.'"""
    with pytest.raises(ValueError, match="no usable text"):
        train_tokenizer(["", "   ", "\n"], tmp_path / "empty", vocab_size=100)


def test_the_tokenizer_can_be_rebuilt_from_the_committed_corpus(tmp_path):
    texts = [
        json.loads(line)["text"]
        for line in CORPUS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert texts, "the fixture corpus is empty"
    out = train_tokenizer(texts, tmp_path / "model", vocab_size=2000)
    assert (out / "tokenizer.json").exists()
    assert (out / "tokenizer_config.json").exists()

    rebuilt = load_tokenizer(out)
    assert rebuilt.model_max_length == 32768
    assert _roundtrips(rebuilt, "<|im_start|>user\nHi<|im_end|>\n")


# ---------------------------------------------- streaming a real corpus
def test_the_streaming_reader_tolerates_a_truncated_tail(tmp_path):
    """A corpus fetched as a byte range ends mid-record. One unparseable final
    line must not discard the preceding two hundred megabytes."""
    from m3g4m1nd.tokenizer import texts_from_jsonl

    path = tmp_path / "sliced.jsonl"
    path.write_text(
        '{"text": "first"}\n{"text": "second"}\n{"text": "thi',
        encoding="utf-8",
    )
    assert list(texts_from_jsonl(path)) == ["first", "second"]


def test_the_streaming_reader_handles_both_corpus_shapes(tmp_path):
    from m3g4m1nd.tokenizer import texts_from_jsonl

    path = tmp_path / "mixed.jsonl"
    path.write_text(
        '{"text": "plain text row"}\n'
        '{"conversations": [{"role": "user", "content": "a"}, '
        '{"role": "assistant", "content": "b"}]}\n',
        encoding="utf-8",
    )
    assert list(texts_from_jsonl(path)) == ["plain text row", "a\nb"]


def test_the_streaming_reader_honours_a_limit(tmp_path):
    from m3g4m1nd.tokenizer import texts_from_jsonl

    path = tmp_path / "many.jsonl"
    path.write_text("".join(f'{{"text": "row {i}"}}\n' for i in range(50)), encoding="utf-8")
    assert len(list(texts_from_jsonl(path, limit=7))) == 7


def test_a_corpus_of_nothing_but_rubbish_is_refused(tmp_path):
    from m3g4m1nd.tokenizer import texts_from_jsonl

    path = tmp_path / "rubbish.jsonl"
    path.write_text("not json\nalso not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="yielded no usable text"):
        list(texts_from_jsonl(path))


def test_blank_and_empty_rows_are_skipped_not_counted(tmp_path):
    from m3g4m1nd.tokenizer import texts_from_jsonl

    path = tmp_path / "sparse.jsonl"
    path.write_text(
        '\n{"text": ""}\n{"text": "   "}\n{"text": "real"}\n\n', encoding="utf-8"
    )
    assert list(texts_from_jsonl(path)) == ["real"]


# ---------------------------------------------- the vocabulary is real
def test_the_vocabulary_fills_the_space_the_model_reserves(tok):
    """A 733-entry placeholder trained on fifteen sentences used to sit here."""
    assert len(tok) == M3Config().vocab_size == 6400


@pytest.mark.parametrize(
    "language,text,floor",
    [
        ("English", "Attention is a weighted average where the weights are learned "
                    "from the content itself.", 2.0),
        ("Chinese", "人工智能是计算机科学的一个分支，它企图了解智能的实质。", 1.4),
        ("Mixed", "Python 是一种高级编程语言，widely used in data science.", 1.8),
    ],
)
def test_compression_proves_the_merges_are_real(tok, language, text, floor):
    """The test that would have caught MiniMind's empty vocabulary.

    A byte-level BPE with no merges emits one token per byte and reports
    success. English then compresses at roughly 1.0 and Chinese below that.
    Real merges put English above 2 and Chinese near 1.7, which is also the
    range MiniMind documents for this vocabulary size.
    """
    tokens = len(tok.encode(text, add_special_tokens=False))
    ratio = len(text) / tokens
    assert ratio >= floor, (
        f"{language} compresses at {ratio:.2f} chars/token, below the {floor} floor. "
        f"A vocabulary with no merges sits near 1.0."
    )


def test_a_vocabulary_with_no_merges_would_fail_that_floor(tmp_path):
    """The control: build the degenerate case and confirm the floor catches it."""
    out = train_tokenizer(["abcdefghij"], tmp_path / "degenerate", vocab_size=300)
    degenerate = load_tokenizer(out)
    text = "Attention is a weighted average where the weights are learned."
    ratio = len(text) / len(degenerate.encode(text, add_special_tokens=False))
    assert ratio < 2.0, f"control is wrong: a near-empty vocabulary compressed at {ratio:.2f}"
