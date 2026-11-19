"""Trajectory sampling, decoupled from the training side.

Two MiniMind defects live here, and they are the same defect twice: **the code
accommodated a mismatch instead of refusing it.**

**It padded missing scores with certainty.** When the HTTP backend returned
fewer per-token log-probabilities than tokens, MiniMind padded the gap with
`0.0` — log p = 0, meaning the model was 100% certain — and fed those invented
numbers straight into the importance ratio. Thirty lines away, in the function
that *calls* the engine, the same mismatch raises. Here it always raises.

**It rebound the live model and called that a snapshot.** `update_policy` did
`self.policy_model = model`, so the engine sampled with the very weights being
updated. "Training-inference separation" was a diagram, not a fact, and the
consequence was that the importance ratio was identically 1 and the clip was
inert. Here `snapshot()` takes a real copy, and `stale_by` says how many
updates behind it is.
"""

from __future__ import annotations

import copy
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass

import torch


@dataclass
class RolloutResult:
    """One batch of sampled trajectories.

    Every field is required. MiniMind's equivalent allowed the scores to be
    shorter than the tokens and silently repaired it.
    """

    output_ids: torch.Tensor        # [rows, prompt + completion]
    completion_ids: torch.Tensor    # [rows, completion]
    per_token_logps: torch.Tensor   # [rows, completion]
    completion_mask: torch.Tensor   # [rows, completion]
    prompt_lens: torch.Tensor       # [rows]
    completions: list[str]

    def __post_init__(self):
        rows, length = self.completion_ids.shape
        for name in ("per_token_logps", "completion_mask"):
            got = tuple(getattr(self, name).shape)
            if got != (rows, length):
                raise ValueError(
                    f"{name} is {got} but there are {rows} rows of {length} completion "
                    f"tokens. A score per token is not optional: padding the gap is how "
                    f"invented numbers reach the importance ratio."
                )
        if len(self.completions) != rows:
            raise ValueError(f"{len(self.completions)} decoded strings for {rows} rows")
        if self.prompt_lens.numel() != rows:
            raise ValueError(f"{self.prompt_lens.numel()} prompt lengths for {rows} rows")


class RolloutEngine(ABC):
    @abstractmethod
    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature):
        ...

    @abstractmethod
    def update_policy(self, model) -> None:
        ...


class TorchRolloutEngine(RolloutEngine):
    """Samples with a snapshot of the policy, not with the policy itself.

    `stale_by` counts updates since the snapshot was taken. It is exposed so a
    training loop can assert what it believes: zero means the sampler and the
    learner are the same weights and the importance ratio is 1.
    """

    def __init__(self, policy_model, tokenizer, device: str = "cpu", snapshot: bool = True):
        self.tokenizer = tokenizer
        self.device = device
        self.use_snapshot = snapshot
        self._policy = None
        self.stale_by = 0
        self.update_policy(policy_model)

    def update_policy(self, model) -> None:
        raw = getattr(model, "module", model)
        raw = getattr(raw, "_orig_mod", raw)
        if self.use_snapshot:
            self._policy = copy.deepcopy(raw).eval()
            for p in self._policy.parameters():
                p.requires_grad_(False)
        else:
            # Explicitly opted into MiniMind's behaviour; see stale_by.
            self._policy = raw
        self.stale_by = 0

    def note_policy_update(self) -> None:
        """Called by the trainer after each optimiser step."""
        self.stale_by += 1

    @property
    def is_live_policy(self) -> bool:
        return not self.use_snapshot

    def rollout(
        self, prompt_ids, attention_mask=None, num_generations: int = 1,
        max_new_tokens: int = 32, temperature: float = 0.8,
    ) -> RolloutResult:
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids)
        prompts = prompt_ids.repeat_interleave(num_generations, dim=0)
        masks = attention_mask.repeat_interleave(num_generations, dim=0)
        prompt_len = prompts.size(1)

        with torch.no_grad():
            output_ids = self._policy.generate(
                input_ids=prompts, attention_mask=masks,
                max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
                eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
            ).clone()

        completion_ids = output_ids[:, prompt_len:]
        full_mask = torch.cat(
            [masks, masks.new_ones(output_ids.size(0), completion_ids.size(1))], dim=1
        )
        with torch.no_grad():
            logits = self._policy(output_ids, attention_mask=full_mask).logits[:, :-1, :]
            all_logps = torch.log_softmax(logits.float(), dim=-1).gather(
                2, output_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps = all_logps[:, prompt_len - 1 :]

        completions = _decode(self.tokenizer, completion_ids)
        return RolloutResult(
            output_ids=output_ids,
            completion_ids=completion_ids,
            per_token_logps=per_token_logps,
            completion_mask=torch.ones_like(completion_ids, dtype=torch.float32),
            prompt_lens=torch.full((output_ids.size(0),), prompt_len, dtype=torch.long),
            completions=completions,
        )


class HttpRolloutEngine(RolloutEngine):
    """Sampling delegated to an external inference server (SGLang-style).

    The server's reply is untrusted input. A short score list is an error, not
    something to pad.
    """

    def __init__(
        self,
        base_url: str,
        tokenizer,
        timeout: int = 120,
        session=None,
        shared_path: str | Path | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.tokenizer = tokenizer
        self.timeout = timeout
        if session is None:
            # Imported here, not at module scope: `import m3g4m1nd` should not
            # require an HTTP library to hand you a tensor. The same reasoning
            # keeps `serve` and `webui` out of the package's __init__.
            import requests

            session = requests
        self.http = session
        # Where weights are written for the server to pick up. It must be a
        # path the server can also read; on one machine that is any directory,
        # across machines it has to be shared storage.
        self.shared_path = Path(shared_path) if shared_path else Path("./rollout_weights")
        self.updates = 0

    def rollout(
        self, prompt_ids, attention_mask=None, num_generations: int = 1,
        max_new_tokens: int = 32, temperature: float = 0.8,
    ) -> RolloutResult:
        if attention_mask is None:
            attention_mask = torch.ones_like(prompt_ids)
        rows = []
        for ids, mask in zip(prompt_ids, attention_mask):
            kept = ids[mask.bool()].tolist()
            rows.extend([kept] * num_generations)

        response = self.http.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": rows,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                },
                "return_logprob": True,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            payload = [payload]
        if len(payload) != len(rows):
            raise ValueError(f"asked for {len(rows)} samples, server returned {len(payload)}")

        return self.assemble(rows, payload, self.tokenizer)

    @staticmethod
    def assemble(prompts: list[list[int]], payload: list[dict], tokenizer) -> RolloutResult:
        """Turn a server reply into a result, refusing anything incomplete."""
        completions_ids, logps_rows, texts = [], [], []

        for index, item in enumerate(payload):
            meta = item.get("meta_info", item)
            tokens = list(meta.get("output_ids", item.get("output_ids", [])) or [])
            raw = meta.get("output_token_logprobs", []) or []
            scores = [float(x[0]) if isinstance(x, (list, tuple)) else float(x) for x in raw]

            if len(scores) != len(tokens):
                raise ValueError(
                    f"sample {index}: server returned {len(scores)} log-probabilities for "
                    f"{len(tokens)} tokens. Padding the difference would feed invented "
                    f"certainties into the importance ratio, so this is refused."
                )
            if not tokens:
                raise ValueError(f"sample {index}: server returned no tokens")

            completions_ids.append(tokens)
            logps_rows.append(scores)
            texts.append(_decode_one(tokenizer, tokens))

        width = max(len(t) for t in completions_ids)
        pad_id = getattr(tokenizer, "pad_token_id", 0) or 0

        def pad(rows, value, dtype):
            return torch.tensor(
                [row + [value] * (width - len(row)) for row in rows], dtype=dtype
            )

        completion_ids = pad(completions_ids, pad_id, torch.long)
        mask = torch.tensor(
            [[1.0] * len(r) + [0.0] * (width - len(r)) for r in completions_ids],
            dtype=torch.float32,
        )
        out_width = max(len(p) + len(c) for p, c in zip(prompts, completions_ids))
        output_ids = torch.tensor(
            [
                p + c + [pad_id] * (out_width - len(p) - len(c))
                for p, c in zip(prompts, completions_ids)
            ],
            dtype=torch.long,
        )

        return RolloutResult(
            output_ids=output_ids,
            completion_ids=completion_ids,
            per_token_logps=pad(logps_rows, 0.0, torch.float32),
            completion_mask=mask,
            prompt_lens=torch.tensor([len(p) for p in prompts], dtype=torch.long),
            completions=texts,
        )

    def update_policy(self, model) -> None:
        """Push the trained weights to the inference server.

        Weights go to disk and the server is told to reload from that path —
        the protocol SGLang and friends expose. Two details matter and are
        tested:

        **It raises on failure.** A silently failed push leaves the server
        sampling from stale weights while the trainer believes it refreshed
        them, which makes the importance ratio quietly meaningless. MiniMind
        printed a warning and set a flag.

        **The write is atomic.** A reload that races a half-written directory
        loads a mixture of two checkpoints, which is worse than either.
        """
        raw = getattr(model, "module", model)
        raw = getattr(raw, "_orig_mod", raw)

        target = self.shared_path.resolve()
        staging = target.with_name(target.name + ".incoming")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        raw.save_pretrained(
            staging,
            state_dict={k: v.detach().cpu() for k, v in raw.state_dict().items()},
        )
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(staging)

        if target.exists():
            shutil.rmtree(target)
        staging.replace(target)

        response = self.http.post(
            f"{self.base_url}/update_weights_from_disk",
            json={"model_path": str(target)},
            timeout=self.timeout,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"the inference server refused the weight update "
                f"({response.status_code}): {response.text[:200]}. The server is "
                f"still sampling from stale weights, so the importance ratio "
                f"would be meaningless."
            )
        body = response.json() if response.content else {}
        if isinstance(body, dict) and body.get("success") is False:
            raise RuntimeError(
                f"the inference server reported the update failed: "
                f"{body.get('message', body)}"
            )
        self.updates += 1

    def flush_cache(self) -> bool:
        return self.http.post(f"{self.base_url}/flush_cache", timeout=30).status_code == 200

    def health(self) -> bool:
        try:
            return self.http.get(f"{self.base_url}/health", timeout=5).status_code == 200
        except Exception:
            return False


def _decode(tokenizer, token_ids) -> list[str]:
    if hasattr(tokenizer, "batch_decode"):
        return tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    return [_decode_one(tokenizer, row.tolist()) for row in token_ids]


def _decode_one(tokenizer, tokens) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(tokens, skip_special_tokens=True)
    return " ".join(str(t) for t in tokens)
