"""Datasets.

The MiniMind defect here is small and tells on everything else: two of its
dataset classes took a `max_length` they never applied. `RLAIFDataset` and
`AgentRLDataset` both stored it and neither truncated anything, so a caller
tuning sequence length through them was tuning nothing — and the config detour
that fed them was itself a dead knob (see config.py).

So the rule in this module: **every constructor parameter changes behaviour, and
a test demonstrates it at the boundary.** `tests/test_data.py` also walks the
AST and fails on any parameter that is stored and never read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

IGNORE = -100


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{number} is not valid JSON: {error}") from None
    if not rows:
        raise ValueError(f"{path} contains no rows")
    return rows


class PretrainDataset(Dataset):
    """Plain next-token prediction over `{"text": ...}` rows."""

    def __init__(self, path, tokenizer, max_length: int = 512):
        if max_length < 3:
            raise ValueError(f"max_length must leave room for bos+token+eos, got {max_length}")
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        text = str(self.rows[index]["text"])
        body = self.tokenizer(
            text, add_special_tokens=False, truncation=True, max_length=self.max_length - 2
        ).input_ids
        tokens = [self.tokenizer.bos_token_id, *body, self.tokenizer.eos_token_id]

        pad_id = self.tokenizer.pad_token_id
        input_ids = tokens + [pad_id] * (self.max_length - len(tokens))
        assert len(input_ids) == self.max_length  # truncation above makes this exact

        ids = torch.tensor(input_ids, dtype=torch.long)
        labels = ids.clone()
        labels[len(tokens):] = IGNORE
        return ids, labels


class SFTDataset(Dataset):
    """Supervised fine-tuning over `{"conversations": [...]}` rows.

    Only the assistant's own tokens carry loss, including the turn-ending token
    so the model learns where to stop — the thing MiniMind's agent trainer
    masked away.
    """

    def __init__(self, path, tokenizer, max_length: int = 1024, train_on_last_turn_only: bool = False):
        if max_length < 4:
            raise ValueError(f"max_length too small to hold a turn, got {max_length}")
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_on_last_turn_only = train_on_last_turn_only

        self.turn_open = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        self.turn_close = tokenizer("<|im_end|>", add_special_tokens=False).input_ids

    def __len__(self) -> int:
        return len(self.rows)

    def render(self, row: dict[str, Any]) -> str:
        messages = row["conversations"]
        tools = row.get("tools")
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, tools=tools
        )

    def supervised_spans(self, input_ids: list[int]) -> list[tuple[int, int]]:
        """Half-open spans of assistant tokens, end inclusive of the close token."""
        spans: list[tuple[int, int]] = []
        open_len, close_len = len(self.turn_open), len(self.turn_close)
        position = 0
        while position < len(input_ids):
            if input_ids[position : position + open_len] != self.turn_open:
                position += 1
                continue
            start = position + open_len
            cursor = start
            while cursor < len(input_ids):
                if input_ids[cursor : cursor + close_len] == self.turn_close:
                    break
                cursor += 1
            spans.append((start, min(cursor + close_len, len(input_ids))))
            position = cursor + close_len
        if self.train_on_last_turn_only and spans:
            return spans[-1:]
        return spans

    def __getitem__(self, index: int):
        rendered = self.render(self.rows[index])
        input_ids = self.tokenizer(
            rendered, add_special_tokens=False, truncation=True, max_length=self.max_length
        ).input_ids

        labels = [IGNORE] * len(input_ids)
        for start, end in self.supervised_spans(input_ids):
            for position in range(start, end):
                labels[position] = input_ids[position]

        pad_id = self.tokenizer.pad_token_id
        padding = self.max_length - len(input_ids)
        input_ids = input_ids + [pad_id] * padding
        labels = labels + [IGNORE] * padding

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


class PreferenceDataset(Dataset):
    """Chosen/rejected pairs for DPO, over `{"chosen": [...], "rejected": [...]}`.

    Returns a dict rather than a tuple, and never concatenates the two sides.
    MiniMind stacked chosen and rejected into one batch and split it on
    `batch_size // 2`, which mispairs every example unless the caller stacked
    them in exactly that order. Keeping the pair together makes that
    impossible.
    """

    def __init__(self, path, tokenizer, max_length: int = 512):
        if max_length < 4:
            raise ValueError(f"max_length too small to hold a turn, got {max_length}")
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.turn_open = tokenizer("<|im_start|>assistant\n", add_special_tokens=False).input_ids
        self.turn_close = tokenizer("<|im_end|>", add_special_tokens=False).input_ids

    def __len__(self) -> int:
        return len(self.rows)

    def _encode(self, messages) -> tuple[list[int], list[int]]:
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = self.tokenizer(
            rendered, add_special_tokens=False, truncation=True, max_length=self.max_length
        ).input_ids

        mask = [0] * len(input_ids)
        open_len, close_len = len(self.turn_open), len(self.turn_close)
        position = 0
        while position < len(input_ids):
            if input_ids[position : position + open_len] != self.turn_open:
                position += 1
                continue
            start = position + open_len
            cursor = start
            while cursor < len(input_ids):
                if input_ids[cursor : cursor + close_len] == self.turn_close:
                    break
                cursor += 1
            for index in range(start, min(cursor + close_len, len(input_ids))):
                mask[index] = 1
            position = cursor + close_len

        pad = self.max_length - len(input_ids)
        pad_id = self.tokenizer.pad_token_id
        return input_ids + [pad_id] * pad, mask + [0] * pad

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        for side in ("chosen", "rejected"):
            if side not in row:
                raise ValueError(f"row {index} has no {side!r} field")

        chosen_ids, chosen_mask = self._encode(row["chosen"])
        rejected_ids, rejected_mask = self._encode(row["rejected"])
        return {
            "chosen_ids": torch.tensor(chosen_ids, dtype=torch.long),
            "chosen_mask": torch.tensor(chosen_mask, dtype=torch.float32),
            "rejected_ids": torch.tensor(rejected_ids, dtype=torch.long),
            "rejected_mask": torch.tensor(rejected_mask, dtype=torch.float32),
        }


class PromptDataset(Dataset):
    """Prompts to roll out from, for the RL stages.

    Carries `gt` — the expected answer or answers — when the row has it, so
    the rubric can verify rather than only score style. Rows may also name the
    tools the model is allowed to call.
    """

    def __init__(self, path, tokenizer, max_length: int = 512, open_thinking: bool = False):
        if max_length < 4:
            raise ValueError(f"max_length too small to hold a prompt, got {max_length}")
        self.rows = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.open_thinking = open_thinking

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        messages = row.get("conversations") or row.get("messages")
        if not messages:
            raise ValueError(f"row {index} has no conversations/messages field")

        # The last assistant turn is what the policy is being asked to produce,
        # so it is not part of the prompt.
        if messages[-1].get("role") == "assistant":
            messages = messages[:-1]

        tools = row.get("tools")
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            tools=tools, open_thinking=self.open_thinking,
        )
        encoded = self.tokenizer(
            rendered, add_special_tokens=False, truncation=True, max_length=self.max_length
        ).input_ids

        expected = row.get("gt", [])
        if expected and not isinstance(expected, list):
            expected = [expected]

        return {
            "prompt": rendered,
            "prompt_ids": torch.tensor(encoded, dtype=torch.long),
            "gt": expected,
            "tools": tools or [],
        }
