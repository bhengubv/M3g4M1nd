"""Tokenizer construction and the chat template.

Byte-level BPE over a 6,400-token vocabulary. Small on purpose: at 768 hidden
units the embedding and output matrix are a tied 4.9M parameters, which is
7.7% of the whole model. A 151k vocabulary would be more than half of it.

One MiniMind defect lives here: its `tokenizer_config.json` declared
`model_max_length: 131072` while the model's position table was 32,768. Nothing
read the first number, so nothing caught it — until a tool that trusts the
tokenizer's claim hands the model a sequence it cannot index. Here the declared
length comes from the model config, and a test asserts the two agree.

The chat template is written to round-trip: rendering a conversation and then
tokenising and detokenising it returns the same string. MiniMind's equivalent
was never tested for that.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .config import M3Config

# Reserved first, so ids stay stable as the vocabulary grows.
CONTROL_TOKENS = [
    "<|endoftext|>",   # pad and unk
    "<|im_start|>",    # bos
    "<|im_end|>",      # eos
]
MARKUP_TOKENS = [
    "<think>", "</think>",
    "<tool_call>", "</tool_call>",
    "<tool_response>", "</tool_response>",
]
RESERVED_SLOTS = 32

CHAT_TEMPLATE = (
    "{%- if tools %}"
    "{{- '<|im_start|>system\\n' }}"
    "{%- if messages[0].role == 'system' %}{{- messages[0].content + '\\n\\n' }}{%- endif %}"
    "{{- '# Tools\\n\\nYou may call one or more of these functions.\\n<tools>\\n' }}"
    "{%- for tool in tools %}{{- tool | tojson }}{{- '\\n' }}{%- endfor %}"
    "{{- '</tools>\\n\\nEmit each call as <tool_call>{\"name\": ..., \"arguments\": ...}"
    "</tool_call>.<|im_end|>\\n' }}"
    "{%- elif messages[0].role == 'system' %}"
    "{{- '<|im_start|>system\\n' + messages[0].content + '<|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- for message in messages %}"
    "{%- if message.role == 'user' or (message.role == 'system' and not loop.first) %}"
    "{{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>\\n' }}"
    "{%- elif message.role == 'assistant' %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if message.reasoning_content %}"
    "{{- '<think>\\n' + message.reasoning_content + '\\n</think>\\n\\n' }}"
    "{%- endif %}"
    "{{- message.content }}"
    "{%- if message.tool_calls %}"
    "{%- for call in message.tool_calls %}"
    "{{- '\\n<tool_call>' }}{{- call | tojson }}{{- '</tool_call>' }}"
    "{%- endfor %}{%- endif %}"
    "{{- '<|im_end|>\\n' }}"
    "{%- elif message.role == 'tool' %}"
    "{{- '<|im_start|>tool\\n<tool_response>\\n' + message.content "
    "+ '\\n</tool_response><|im_end|>\\n' }}"
    "{%- endif %}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}"
    "{{- '<|im_start|>assistant\\n' }}"
    "{%- if open_thinking %}{{- '<think>\\n' }}{%- else %}{{- '<think>\\n\\n</think>\\n\\n' }}{%- endif %}"
    "{%- endif %}"
)


def special_tokens(reserved: int = RESERVED_SLOTS) -> list[str]:
    named = CONTROL_TOKENS + MARKUP_TOKENS
    spare = max(reserved - len(named), 0)
    return named + [f"<|reserved_{i}|>" for i in range(spare)]


def tokenizer_config(config: M3Config | None = None) -> dict:
    """The sidecar config. `model_max_length` comes from the model, not a guess."""
    config = config or M3Config()
    return {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        # The number MiniMind got wrong: it claimed 131072 against a 32768 model.
        "model_max_length": config.max_position_embeddings,
        "add_bos_token": False,
        "add_eos_token": False,
        "clean_up_tokenization_spaces": False,
        "spaces_between_special_tokens": False,
        "additional_special_tokens": MARKUP_TOKENS,
        "chat_template": CHAT_TEMPLATE,
    }


def train_tokenizer(
    texts: Iterable[str],
    out_dir: str | Path,
    vocab_size: int = 6400,
    config: M3Config | None = None,
) -> Path:
    """Train a byte-level BPE and write it beside its config.

    Refuses an empty corpus. MiniMind's version, pointed at the wrong data
    format, produced a vocabulary with no merges at all and reported success —
    a bug its own source comment now documents.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = [t for t in texts if t and t.strip()]
    if not corpus:
        raise ValueError(
            "no usable text in the corpus. Training would yield a vocabulary with no "
            "merges and report success, which is how MiniMind shipped an empty one."
        )

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        corpus,
        trainer=trainers.BpeTrainer(
            vocab_size=vocab_size,
            show_progress=False,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=special_tokens(),
        ),
    )

    tokenizer.save(str(out_dir / "tokenizer.json"))
    (out_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out_dir


def texts_from_jsonl(path: str | Path, limit: int | None = None):
    """Stream training text out of a jsonl corpus, tolerating a truncated tail.

    A corpus fetched as a byte range ends mid-record, and one unparseable final
    line should not lose the preceding two hundred megabytes. Malformed lines
    anywhere are skipped and counted rather than raised, because a corpus is
    data and data is never as clean as its schema.
    """
    import json

    used = skipped = 0
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if limit is not None and used >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if "text" in row:
                text = str(row["text"])
            else:
                text = "\n".join(
                    m.get("content", "") for m in row.get("conversations", []) or []
                )
            if text.strip():
                used += 1
                yield text
    if not used:
        raise ValueError(f"{path} yielded no usable text ({skipped} malformed lines)")


def load_tokenizer(path: str | Path = "model"):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(path), local_files_only=True)
