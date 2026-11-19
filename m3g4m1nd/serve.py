"""An OpenAI-compatible HTTP endpoint.

Five MiniMind defects live here.

**It bound every interface.** `uvicorn.run(app, host="0.0.0.0", port=8998)`,
with no authentication, no rate limit and no way to change it short of editing
the file. Anyone who could reach the machine could use the GPU. Here the
default is loopback, and opening it up takes an explicit flag and prints a
warning.

**Request sizes were unbounded.** `max_tokens` defaulted to 8192 with no
ceiling, against a decoding loop of `for _ in range(max_new_tokens)`. One
request could occupy the GPU indefinitely, and anything past the model's
position table crashed it. Every field is bounded here, and `max_tokens` is
checked against what is actually left of the window.

**Exception text went to the client.** `detail=str(e)` handed internal state to
whoever asked. Here the detail is logged and the caller gets an opaque error
with an id to quote.

**One model, unbounded threads, no lock.** Every request spawned a thread
against one shared model. Here generation is serialised, so concurrent callers
get correct, non-interleaved output.

**Synchronous inference blocked the event loop.** Fixed upstream in PR #905,
whose test was then deleted. The endpoints here are plain `def`, so Starlette
runs them in a worker thread and the loop stays responsive.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import uuid
import warnings
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

log = logging.getLogger("m3g4m1nd.serve")

LOOPBACK = "127.0.0.1"

# Module-level handles, patched in tests and set by create_app.
model: Any = None
tokenizer: Any = None
device: str = "cpu"

# One model, so one generation at a time. MiniMind spawned a thread per request
# against a shared model with no lock.
_generation_lock = threading.Lock()


class ChatRequest(BaseModel):
    """Every numeric field is bounded. Out of range is a 422, not a 500."""

    model: str = "m3g4m1nd"
    messages: list = Field(min_length=1)
    temperature: float = Field(default=0.7, ge=0.0, le=5.0, allow_inf_nan=False)
    top_p: float = Field(default=0.92, gt=0.0, le=1.0, allow_inf_nan=False)
    top_k: int = Field(default=50, ge=0, le=100_000)
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=10.0, allow_inf_nan=False)
    max_tokens: int = Field(default=512, ge=1, le=32_768)
    stream: bool = True
    tools: list = Field(default_factory=list)
    open_thinking: bool = False

    @property
    def sampling(self) -> bool:
        """Temperature 0 means greedy, which is not an error."""
        return self.temperature > 0.0


# Turn delimiters, which the caller never wants to see. Markup tokens such as
# <tool_call> and <think> are deliberately not in this list: they carry meaning
# the stream parser depends on.
CONTROL_TOKEN_TEXT = ("<|im_start|>", "<|im_end|>", "<|endoftext|>")


def _drop_control_tokens(text: str) -> str:
    for token in CONTROL_TOKEN_TEXT:
        text = text.replace(token, "")
    return text


def parse_response(text: str) -> tuple[str, str | None, list | None]:
    """Split a completion into content, reasoning and tool calls."""
    reasoning = None
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if match:
        reasoning = match.group(1).strip()
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    elif "</think>" in text:
        head, _, tail = text.partition("</think>")
        reasoning, text = head.strip(), tail.strip()

    calls = []
    for index, body in enumerate(re.findall(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)):
        try:
            payload = json.loads(body.strip())
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            # `<tool_call>[]</tool_call>` is valid JSON and not a call. Treating
            # it as one would reach `.get` on a list.
            continue
        calls.append({
            "id": f"call_{index}",
            "type": "function",
            "function": {
                "name": payload.get("name", ""),
                "arguments": json.dumps(payload.get("arguments", {}), ensure_ascii=False),
            },
        })
    if calls:
        text = re.sub(r"<tool_call>.*?</tool_call>", "", text, flags=re.DOTALL)

    return text.strip(), reasoning, calls or None


def _window_left(prompt_tokens: int) -> int:
    limit = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    return (limit - prompt_tokens) if limit else 1 << 30


def _render(request: ChatRequest):
    prompt = tokenizer.apply_chat_template(
        request.messages,
        tokenize=False,
        add_generation_prompt=True,
        tools=request.tools or None,
        open_thinking=request.open_thinking,
    )
    return tokenizer(prompt, return_tensors="pt", truncation=True).to(device)


def _generate(inputs, request: ChatRequest, streamer=None):
    budget = min(request.max_tokens, _window_left(inputs["input_ids"].shape[1]))
    if budget < 1:
        raise HTTPException(
            status_code=422,
            detail=(
                "the prompt already fills the model's context window; "
                "no room left to generate"
            ),
        )
    with _generation_lock:
        return model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=budget,
            do_sample=request.sampling,
            temperature=request.temperature if request.sampling else 1.0,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            eos_token_id=getattr(tokenizer, "eos_token_id", None),
            streamer=streamer,
        )


def create_app() -> FastAPI:
    app = FastAPI(title="M3g4M1nd")

    @app.exception_handler(Exception)
    def opaque_errors(request: Request, exc: Exception):
        """Log the detail, give the caller an id. MiniMind returned str(e)."""
        incident = uuid.uuid4().hex[:12]
        log.exception("request failed [%s]", incident)
        return JSONResponse(
            status_code=500,
            content={"error": {
                "message": "internal error", "type": "internal_error", "incident": incident,
            }},
        )

    @app.get("/", response_class=HTMLResponse)
    def index():
        """The chat page, from the same process and the same stream parser."""
        from .webui import PAGE

        return HTMLResponse(PAGE)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": model is not None}

    # Plain `def`, not `async def`: Starlette runs it in a worker thread, so a
    # slow forward pass cannot stall the event loop. (MiniMind PR #905.)
    @app.post("/v1/chat/completions")
    def chat_completions(request: ChatRequest):
        inputs = _render(request)

        if request.stream:
            return StreamingResponse(
                _stream(inputs, request), media_type="text/event-stream"
            )

        generated = _generate(inputs, request)
        answer = tokenizer.decode(
            generated[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )
        content, reasoning, calls = parse_response(answer)
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            message["reasoning_content"] = reasoning
        if calls:
            message["tool_calls"] = calls
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:16]}",
            "object": "chat.completion",
            "model": request.model,
            "choices": [
                {"index": 0, "message": message,
                 "finish_reason": "tool_calls" if calls else "stop"}
            ],
        }

    return app


def _stream(inputs, request: ChatRequest):
    """Server-sent events, emitting tool calls as whole blocks.

    A marker can straddle two chunks, so text is held back until either the
    marker completes or it is clear none is forming.
    """
    from queue import Queue

    queue: Queue = Queue()

    class Streamer:
        """Buffers token ids until they decode to something complete.

        Two things here are load-bearing, and both were found by the test this
        file's recovered counterpart guards.

        **Special tokens are kept.** `<tool_call>` is a single token and it *is*
        the protocol. Decoding with `skip_special_tokens=True` deletes the
        markup and the raw JSON leaks out as ordinary content. Only the three
        control tokens are dropped.

        **Bytes are buffered.** A byte-level vocabulary splits one character
        across several tokens, so decoding each token on arrival yields U+FFFD
        replacement characters mid-word. Hold the ids back until the decode is
        clean.
        """

        def __init__(self):
            self.cache: list[int] = []

        def put(self, value):
            if not hasattr(value, "tolist"):
                if str(value):
                    queue.put(str(value))
                return

            self.cache.extend(int(t) for t in value.flatten().tolist())
            text = tokenizer.decode(self.cache, skip_special_tokens=False)
            if "�" in text:
                return  # mid-character; wait for the rest of the bytes
            self.cache.clear()
            text = _drop_control_tokens(text)
            if text:
                queue.put(text)

        def end(self):
            if self.cache:
                leftover = _drop_control_tokens(
                    tokenizer.decode(self.cache, skip_special_tokens=False)
                )
                self.cache.clear()
                if leftover:
                    queue.put(leftover)
            queue.put(None)

        def on_finalized_text(self, text: str, stream_end: bool = False):
            if text:
                queue.put(text)
            if stream_end:
                queue.put(None)

    streamer = Streamer()

    def run():
        try:
            _generate(inputs, request, streamer=streamer)
        except Exception:
            incident = uuid.uuid4().hex[:12]
            log.exception("streamed generation failed [%s]", incident)
            queue.put({"error": {"message": "internal error", "incident": incident}})
            queue.put(None)

    threading.Thread(target=run, daemon=True).start()

    def frame(field: str, value) -> str:
        return f'data: {json.dumps({"choices": [{"delta": {field: value}}]}, ensure_ascii=False)}\n\n'

    pending, in_tool, emitted = "", False, 0
    # Before the thinking boundary, text is reasoning and tool markup inside it
    # is just text. After it, tool markup is a call.
    thinking_ended = not request.open_thinking
    strip_newlines = False

    while True:
        chunk = queue.get()
        if chunk is None:
            break
        if isinstance(chunk, dict):
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            continue
        pending += chunk

        while pending:
            if strip_newlines:
                pending = pending.lstrip("\n")
                if not pending:
                    break
                strip_newlines = False

            if in_tool:
                end = pending.find("</tool_call>")
                if end < 0:
                    break
                end += len("</tool_call>")
                block, pending = pending[:end], pending[end:]
                _, _, calls = parse_response(block)
                if calls:
                    call = dict(calls[0], index=emitted, id=f"call_{uuid.uuid4().hex[:10]}")
                    emitted += 1
                    yield frame("tool_calls", [call])
                else:
                    # Malformed or unfinished: it is text, and the caller sees it.
                    yield frame("content", block)
                in_tool = False
                continue

            marker = "<tool_call>" if thinking_ended else "</think>"
            field = "content" if thinking_ended else "reasoning_content"
            position = pending.find(marker)

            if position < 0:
                # Hold back any suffix that could still grow into the marker.
                keep = next(
                    (n for n in range(len(marker) - 1, 0, -1) if pending.endswith(marker[:n])), 0
                )
                cut = len(pending) - keep
                if cut:
                    yield frame(field, pending[:cut])
                    pending = pending[cut:]
                break

            if position:
                yield frame(field, pending[:position])
            pending = pending[position:]

            if thinking_ended:
                in_tool = True
            else:
                pending = pending[len(marker):]
                thinking_ended = True
                strip_newlines = True

    if pending:
        yield frame("content" if thinking_ended else "reasoning_content", pending)
    tail = {"choices": [{"delta": {}, "finish_reason": "tool_calls" if emitted else "stop"}]}
    yield f"data: {json.dumps(tail, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


app = create_app()


def serve(host: str = LOOPBACK, port: int = 8998, allow_public: bool = False) -> None:
    """Run the server. Loopback unless told otherwise, out loud."""
    import uvicorn

    if host not in {LOOPBACK, "localhost", "::1"} and not allow_public:
        raise ValueError(
            f"refusing to bind {host!r}: this endpoint has no authentication and no rate "
            f"limit, so exposing it hands your model and your hardware to the network. "
            f"Pass allow_public=True if that is genuinely what you want."
        )
    if allow_public and host not in {LOOPBACK, "localhost", "::1"}:
        warnings.warn(
            f"serving on {host}:{port} with no authentication and no rate limit.",
            stacklevel=2,
        )
    uvicorn.run(app, host=host, port=port)
