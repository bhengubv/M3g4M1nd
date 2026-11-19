"""Rows D1, D2, D3, D4 and D5 — the serving layer.

D5 is recovered from MiniMind PR #905 (`Afloat16`), Apache-2.0, deleted in
commit cc9629e. See CREDITS.md.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from unittest.mock import patch

import httpx
import pytest
import torch
from fastapi.testclient import TestClient

from m3g4m1nd import serve
from m3g4m1nd.serve import LOOPBACK, ChatRequest, parse_response


class Inputs(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self, prompt_tokens: int = 1):
        self.prompt_tokens = prompt_tokens

    def apply_chat_template(self, *_, **__):
        return "prompt"

    def __call__(self, *_, **__):
        width = self.prompt_tokens
        return Inputs(
            input_ids=torch.full((1, width), 3, dtype=torch.long),
            attention_mask=torch.ones(1, width, dtype=torch.long),
        )

    def decode(self, *_, **__):
        return "answer"


class FakeModel:
    class config:
        max_position_embeddings = 128

    def __init__(self, delay: float = 0.0, text: str = "answer"):
        self.delay, self.text = delay, text
        self.calls: list[int] = []
        self.concurrent, self.max_concurrent = 0, 0
        self._guard = threading.Lock()

    def generate(self, input_ids, max_new_tokens=1, streamer=None, **_):
        with self._guard:
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            self.calls.append(max_new_tokens)
            if self.delay:
                time.sleep(self.delay)
            if streamer is not None:
                streamer.on_finalized_text(self.text)
                streamer.on_finalized_text("", stream_end=True)
            return torch.tensor([[3, 4]])
        finally:
            with self._guard:
                self.concurrent -= 1


def client(model=None, tokenizer=None, as_real_client=False):
    """`as_real_client=True` stops TestClient re-raising server exceptions, so
    the test sees the response an actual HTTP caller would receive."""
    patcher = patch.multiple(
        serve,
        model=model or FakeModel(),
        tokenizer=tokenizer or FakeTokenizer(),
        device="cpu",
        create=True,
    )
    patcher.start()
    return (
        TestClient(serve.app, raise_server_exceptions=not as_real_client),
        patcher,
    )


ASK = {"model": "m3g4m1nd", "messages": [{"role": "user", "content": "Hi"}], "stream": False}


# ---------------------------------------------------------------- D1
def test_serve_refuses_to_bind_every_interface_by_default():
    with pytest.raises(ValueError, match="refusing to bind"):
        serve.serve(host="0.0.0.0")


def test_the_refusal_says_why():
    with pytest.raises(ValueError) as caught:
        serve.serve(host="0.0.0.0")
    message = str(caught.value)
    assert "no authentication" in message and "allow_public=True" in message


def test_the_default_host_is_loopback():
    import inspect

    assert inspect.signature(serve.serve).parameters["host"].default == LOOPBACK


def test_binding_publicly_is_possible_but_warns():
    with patch("uvicorn.run") as run:
        with pytest.warns(UserWarning, match="no authentication"):
            serve.serve(host="0.0.0.0", allow_public=True)
    run.assert_called_once()
    assert run.call_args.kwargs["host"] == "0.0.0.0"


# ---------------------------------------------------------------- D2
@pytest.mark.parametrize(
    "field,value",
    [
        ("temperature", -0.5), ("temperature", 99.0),
        ("top_p", 0.0), ("top_p", 1.5),
        ("top_k", -1),
        ("max_tokens", 0), ("max_tokens", 10_000_000),
        ("repetition_penalty", 0.0),
        ("messages", []),
    ],
)
def test_out_of_range_fields_are_rejected_with_422(field, value):
    http, patcher = client()
    try:
        response = http.post("/v1/chat/completions", json={**ASK, field: value})
    finally:
        patcher.stop()
    assert response.status_code == 422, f"{field}={value} was accepted"


def test_a_request_longer_than_the_window_is_refused_not_crashed():
    """MiniMind let max_tokens run past the position table and died in a slice."""
    http, patcher = client(tokenizer=FakeTokenizer(prompt_tokens=128))
    try:
        response = http.post("/v1/chat/completions", json={**ASK, "max_tokens": 10})
    finally:
        patcher.stop()
    assert response.status_code == 422
    assert "context window" in response.text


def test_max_tokens_is_clamped_to_what_is_left_of_the_window():
    model = FakeModel()
    http, patcher = client(model=model, tokenizer=FakeTokenizer(prompt_tokens=120))
    try:
        assert http.post("/v1/chat/completions", json={**ASK, "max_tokens": 500}).status_code == 200
    finally:
        patcher.stop()
    assert model.calls == [8], f"asked for 500 with 8 left, generated {model.calls}"


def test_temperature_zero_means_greedy_and_is_not_an_error():
    assert ChatRequest(messages=[{"role": "user", "content": "x"}], temperature=0.0).sampling is False
    assert ChatRequest(messages=[{"role": "user", "content": "x"}], temperature=0.7).sampling is True


# ---------------------------------------------------------------- D3
def test_an_internal_failure_returns_an_opaque_error_with_an_incident_id():
    class Exploding(FakeModel):
        def generate(self, *_, **__):
            raise RuntimeError("secret: /home/user/weights/full_sft_768.pth is corrupt")

    http, patcher = client(model=Exploding(), as_real_client=True)
    try:
        response = http.post("/v1/chat/completions", json=ASK)
    finally:
        patcher.stop()

    assert response.status_code == 500
    body = response.text
    assert "secret" not in body and "full_sft_768" not in body and "RuntimeError" not in body
    assert response.json()["error"]["incident"]


def test_validation_errors_are_not_turned_into_500s():
    http, patcher = client()
    try:
        response = http.post("/v1/chat/completions", json={**ASK, "top_p": 9.0})
    finally:
        patcher.stop()
    assert response.status_code == 422


# ---------------------------------------------------------------- D4
def test_concurrent_requests_are_serialised_against_the_one_model():
    model = FakeModel(delay=0.05)
    http, patcher = client(model=model)
    results: list[int] = []

    def fire():
        results.append(http.post("/v1/chat/completions", json=ASK).status_code)

    try:
        threads = [threading.Thread(target=fire) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
    finally:
        patcher.stop()

    assert results == [200] * 4
    assert model.max_concurrent == 1, (
        f"{model.max_concurrent} generations overlapped on one shared model"
    )


def test_every_concurrent_caller_gets_a_complete_answer():
    model = FakeModel(delay=0.02)
    http, patcher = client(model=model)
    bodies: list[str] = []

    def fire():
        bodies.append(http.post("/v1/chat/completions", json=ASK).json()["choices"][0]["message"]["content"])

    try:
        threads = [threading.Thread(target=fire) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
    finally:
        patcher.stop()

    assert bodies == ["answer"] * 3


# ---------------------------------------------------------------- D5
# Recovered: MiniMind PR #905. A blocking forward pass must not stall the loop.

def test_other_coroutines_run_during_generation():
    """Ported from the deleted test. The endpoint is a plain `def`, so Starlette
    runs it in a worker thread; if it were `async def`, the sleep below could
    never fire and `observed` would be [False]."""
    release = threading.Event()
    observed: list[bool] = []

    class Blocking(FakeModel):
        def generate(self, *_, **__):
            observed.append(release.wait(timeout=5))
            return torch.tensor([[3, 4]])

    async def exercise():
        async def release_from_event_loop():
            await asyncio.sleep(0.05)
            release.set()

        transport = httpx.ASGITransport(app=serve.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
            response, _ = await asyncio.gather(
                http.post("/v1/chat/completions", json={**ASK, "max_tokens": 1}),
                release_from_event_loop(),
            )
        return response

    patcher = patch.multiple(
        serve, model=Blocking(), tokenizer=FakeTokenizer(), device="cpu", create=True
    )
    patcher.start()
    try:
        response = asyncio.run(exercise())
    finally:
        patcher.stop()

    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "answer"
    assert observed == [True], "generation blocked the coroutine needed to release it"


# ------------------------------------------------- supporting behaviour
def test_health_reports_whether_a_model_is_loaded():
    http, patcher = client()
    try:
        assert http.get("/health").json() == {"status": "ok", "model_loaded": True}
    finally:
        patcher.stop()


def test_parse_response_splits_reasoning_content_and_tool_calls():
    text = (
        "<think>weighing it up</think>here you go"
        '<tool_call>{"name":"weather","arguments":{"city":"Durban"}}</tool_call>'
    )
    content, reasoning, calls = parse_response(text)
    assert reasoning == "weighing it up"
    assert content == "here you go"
    assert calls[0]["function"]["name"] == "weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Durban"}


def test_malformed_tool_json_is_dropped_not_raised():
    content, _, calls = parse_response("text<tool_call>{not json}</tool_call>")
    assert calls is None
    assert "not json" in content or content == "text"


def test_streaming_emits_a_tool_call_split_across_two_chunks():
    """The marker arrives in pieces; the stream must not leak the fragment."""
    model = FakeModel()
    chunks = ['before<tool_c', 'all>{"name":"weather","arguments":{}}</tool_call>after']

    class Chunked(FakeModel):
        def generate(self, input_ids, max_new_tokens=1, streamer=None, **_):
            for chunk in chunks:
                streamer.on_finalized_text(chunk)
            streamer.on_finalized_text("", stream_end=True)
            return torch.tensor([[3, 4]])

    http, patcher = client(model=Chunked())
    try:
        response = http.post("/v1/chat/completions", json={**ASK, "stream": True})
        frames = [
            json.loads(line[6:])
            for line in response.text.splitlines()
            if line.startswith("data: ") and not line.endswith("[DONE]")
        ]
    finally:
        patcher.stop()

    assert response.status_code == 200
    contents = "".join(
        f["choices"][0]["delta"].get("content", "") for f in frames if "choices" in f
    )
    assert "<tool_c" not in contents and "tool_call" not in contents
    assert contents == "beforeafter"
    tool_frames = [f for f in frames if f["choices"][0]["delta"].get("tool_calls")]
    assert len(tool_frames) == 1
    assert tool_frames[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["name"] == "weather"
    assert frames[-1]["choices"][0]["finish_reason"] == "tool_calls"


# ------------------------------------------------------------------ web UI
def test_the_chat_page_is_served_from_the_same_process():
    """MiniMind's demo is a separate Streamlit app with its own copy of the
    stream parser. This is one page, one server, one parser."""
    http, patcher = client()
    try:
        response = http.get("/")
    finally:
        patcher.stop()

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    body = response.text
    assert "<title>M3g4M1nd</title>" in body
    assert "v1/chat/completions" in body, "the page must use the API it ships with"


def test_the_page_handles_light_and_dark():
    from m3g4m1nd.webui import PAGE

    assert "prefers-color-scheme: dark" in PAGE
    assert ':root:not([data-theme="light"])' in PAGE


def test_the_page_renders_reasoning_and_tool_calls_separately():
    from m3g4m1nd.webui import PAGE

    for field in ("reasoning_content", "tool_calls", "[DONE]"):
        assert field in PAGE, f"the page ignores {field} from the stream"


def test_the_page_needs_no_external_resources():
    """No CDN, no font host, nothing to fetch. It works offline, like the tests."""
    from m3g4m1nd.webui import PAGE

    for scheme in ("http://", "https://", "//cdn", "<script src"):
        assert scheme not in PAGE, f"the page reaches out via {scheme}"


def test_the_page_uses_relative_urls_so_it_works_behind_a_prefix():
    from m3g4m1nd.webui import PAGE

    assert "fetch('health')" in PAGE
    assert "fetch('v1/chat/completions'" in PAGE
