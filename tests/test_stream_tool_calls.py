"""Row D6 — streamed tool calls, ported from MiniMind PR #883 (`cyanseek`).

Apache-2.0, deleted from MiniMind in commit 233d00f ("trim merged pr
leftovers"), 147 lines. Ported here against our own server. See CREDITS.md.

The original's cases are all kept: split markers including character by
character, several calls in one stream, tool markup appearing *inside* the
reasoning block where it must stay verbatim, malformed and unfinished calls
preserved as text, an empty stream, incremental delivery before generation
finishes, and a real model producing real tokens.
"""

from __future__ import annotations

import json
import threading
from unittest.mock import patch

import pytest
import torch
from fastapi.testclient import TestClient

from m3g4m1nd import serve
from m3g4m1nd.config import M3Config
from m3g4m1nd.model import M3ForCausalLM
from m3g4m1nd.tokenizer import load_tokenizer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = '<tool_call>{"name":"weather","arguments":{"city":"杭州"}}</tool_call>'
TOOLS = [{
    "type": "function",
    "function": {
        "name": "weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]
ASK = {"model": "m3g4m1nd", "messages": [{"role": "user", "content": "Hi"}]}


class ScriptedModel:
    """Emits pre-set chunks through the streamer, as a real generate would."""

    class config:
        max_position_embeddings = 256

    def __init__(self, chunks, tokenize: bool = False, tokenizer=None):
        self.chunks, self.tokenize, self.tokenizer = chunks, tokenize, tokenizer

    def generate(self, input_ids, streamer=None, **_):
        if self.tokenize:
            for token in self.tokenizer.encode("".join(self.chunks), add_special_tokens=False):
                streamer.put(torch.tensor([token]))
            streamer.end()
        else:
            for chunk in self.chunks:
                streamer.on_finalized_text(chunk)
            streamer.on_finalized_text("", stream_end=True)
        return torch.tensor([[3, 4]])


@pytest.fixture(scope="module")
def tok():
    return load_tokenizer(ROOT / "model")


@pytest.fixture
def http(tok):
    patcher = patch.multiple(
        serve, model=ScriptedModel([]), tokenizer=tok, device="cpu", create=True
    )
    patcher.start()
    with TestClient(serve.app) as client:
        yield client
    patcher.stop()


def frames(http, chunks, thinking: bool = False, **extra):
    serve.model = ScriptedModel(chunks)
    response = http.post(
        "/v1/chat/completions",
        json={**ASK, "stream": True, "open_thinking": thinking, **extra},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    ]


def assert_output(collected, content, count, reasoning=""):
    assert not any("error" in frame for frame in collected), collected
    deltas = [frame["choices"][0]["delta"] for frame in collected]

    assert "".join(d.get("content", "") for d in deltas) == content
    assert "".join(d.get("reasoning_content", "") for d in deltas) == reasoning

    calls = [call for d in deltas for call in d.get("tool_calls", [])]
    assert len(calls) == count
    assert [call["index"] for call in calls] == list(range(count))
    assert len({call["id"] for call in calls}) == count
    for call in calls:
        assert call["function"]["name"] == "weather"
        assert json.loads(call["function"]["arguments"]) == {"city": "杭州"}

    expected_finish = "tool_calls" if count else "stop"
    assert collected[-1]["choices"][0]["finish_reason"] == expected_finish


# ---------------------------------------------------------------- D6
def test_split_markers_and_multiple_calls(http):
    text = "Before " + TOOL + " between " + TOOL + " after"
    for chunks in ([text], list(text), ["Before <tool_", text[len("Before <tool_"):]]):
        assert_output(frames(http, chunks), "Before  between  after", 2)


def test_thinking_boundary_and_tool_text_inside_reasoning(http):
    """Tool markup before the boundary is reasoning text, not a call."""
    reasoning = "Consider " + TOOL + "\n"
    text = reasoning + "</think>\n\n" + TOOL + " answer"
    assert_output(frames(http, list(text), thinking=True), " answer", 1, reasoning)


@pytest.mark.parametrize(
    "text",
    ["plain < text", "end <tool_", '<tool_call>{"name":',
     "<tool_call>bad json</tool_call>", "<tool_call>[]</tool_call>"],
)
def test_invalid_and_unfinished_calls_preserve_text(http, text):
    assert_output(frames(http, list(text)), text, 0)


def test_an_invalid_call_followed_by_a_valid_one(http):
    invalid = "<tool_call>bad json</tool_call>"
    assert_output(frames(http, [invalid, TOOL]), invalid, 1)


def test_empty_and_unfinished_thinking(http):
    assert_output(frames(http, []), "", 0)
    assert_output(frames(http, ["reason </thi"], thinking=True), "", 0, "reason </thi")


def test_content_and_tool_call_arrive_before_generation_ends(http):
    """Delivery is incremental: a frame reaches the caller while the model is
    still blocked inside generate."""
    carry_on, finish = threading.Event(), threading.Event()

    class Staged(ScriptedModel):
        def generate(self, input_ids, streamer=None, **_):
            streamer.on_finalized_text("first ")
            assert carry_on.wait(5)
            streamer.on_finalized_text(TOOL)
            assert finish.wait(5)
            streamer.on_finalized_text(" last", stream_end=True)
            return torch.tensor([[3, 4]])

    serve.model = Staged([])
    request = serve.ChatRequest(**{**ASK, "stream": True})
    inputs = serve._render(request)
    stream = serve._stream(inputs, request)

    try:
        first = json.loads(next(stream)[6:])
        assert first["choices"][0]["delta"]["content"] == "first "

        carry_on.set()
        call = json.loads(next(stream)[6:])
        assert call["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "weather"

        finish.set()
        rest = [
            json.loads(line[6:])
            for line in stream
            if line.startswith("data: ") and line[6:].strip() != "[DONE]"
        ]
    finally:
        carry_on.set()
        finish.set()

    assert_output([first, call] + rest, "first  last", 1)


def test_a_real_token_streamer_round_trips_through_the_tokenizer(http, tok):
    """The original's `tokenize=True` path: the server decodes real token ids
    rather than being handed ready-made strings."""
    serve.model = ScriptedModel(["Before ", TOOL, " after"], tokenize=True, tokenizer=tok)
    response = http.post("/v1/chat/completions", json={**ASK, "stream": True, "tools": TOOLS})
    collected = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    ]
    assert_output(collected, "Before  after", 1)


def test_a_real_tiny_model_streams_without_error(http, tok):
    """Real weights, real sampling, four tokens. Nothing is asserted about what
    it says — only that the stream is well formed and finishes."""
    torch.manual_seed(42)
    serve.model = M3ForCausalLM(M3Config(
        hidden_size=32, num_hidden_layers=1, num_attention_heads=4,
        num_key_value_heads=2, intermediate_size=64,
        vocab_size=len(tok), max_position_embeddings=256,
    )).eval()

    response = http.post("/v1/chat/completions", json={**ASK, "stream": True, "max_tokens": 4})
    collected = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    ]
    assert not any("error" in frame for frame in collected), collected
    assert collected[-1]["choices"][0]["finish_reason"] == "stop"


def test_the_stream_terminates_with_done(http):
    serve.model = ScriptedModel(["hello"])
    response = http.post("/v1/chat/completions", json={**ASK, "stream": True})
    assert response.text.rstrip().endswith("data: [DONE]")
