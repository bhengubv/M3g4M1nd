"""Reward functions for the RL stages.

Without these, `rl.py` and `agent.py` take advantages from nowhere. MiniMind
computes rewards inline inside three separate training scripts, which is how
its three copies drifted: the GRPO script scores length over the whole
response, the agent script scores it over the last turn, and nothing says which
is intended.

One MiniMind fix is carried over deliberately. PR #904 — "match numeric gt by
value, not by substring" — fixed ground-truth checking that accepted "19" as a
hit for an expected 9, and 144000 as a hit for 1440. Substring matching on
numbers is wrong in a way that inflates every score it touches, and the fix is
reimplemented and tested here.

Every reward is a pure function of text. Nothing here loads a model, so the
whole suite still runs on a CPU with no downloads; a learned scorer plugs in
through `Rubric(scorer=...)`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Iterable

# A number, not part of an identifier and not a fragment of a longer number.
_NUMBER = re.compile(r"(?<![A-Za-z0-9_.])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_]|\.\d)")
_PURE_NUMBER = re.compile(r"^[-+]?\d+(?:\.\d+)?$")


def numbers_in(text: str) -> list[float]:
    """Every standalone number, thousands separators removed."""
    return [float(match) for match in _NUMBER.findall(str(text).replace(",", ""))]


def ground_truth_hits(text: str, expected: Iterable) -> set[str]:
    """Which expected answers appear. Numbers compared by value, not substring.

    This is MiniMind PR #904's fix. Substring matching says 9 is present in
    "19" and that 1440 is present in "144000"; both are false and both inflate
    the score.
    """
    text = str(text)
    found = numbers_in(text)
    lowered = text.lower()

    hits = set()
    for item in expected:
        needle = str(item).strip()
        if not needle:
            continue
        bare = needle.replace(",", "")
        if _PURE_NUMBER.match(bare):
            target = float(bare)
            if any(abs(target - value) < 1e-6 for value in found):
                hits.add(needle)
        elif needle.lower() in lowered:
            hits.add(needle)
    return hits


def repetition_penalty(text: str, n: int = 3, cap: float = 0.5) -> float:
    """How much of the text is repeated n-grams, scaled to at most `cap`."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    tokens = re.findall(r"\w+|[^\w\s]", str(text).lower())
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    repeated = len(grams) - len(set(grams))
    return min(cap, repeated * cap * 2 / len(grams))


def length_reward(text: str, low: int = 20, high: int = 800, inside: float = 0.5) -> float:
    """`inside` if the text is a sensible length, -`inside` otherwise."""
    if low > high:
        raise ValueError(f"low ({low}) must not exceed high ({high})")
    return inside if low <= len(str(text).strip()) <= high else -inside


def thinking_reward(text: str, low: int = 20, high: int = 300) -> float:
    """Score the reasoning block, if there is exactly one and it is sane."""
    text = str(text)
    if "</think>" not in text:
        return 0.0
    score = 0.25 if text.count("</think>") == 1 else -0.25
    reasoning = text.split("</think>", 1)[0].replace("<think>", "").strip()
    score += 1.0 if low <= len(reasoning) <= high else -0.5
    return score


def tool_tag_reward(text: str, penalty: float = 0.5) -> float:
    """Penalise unbalanced tool markup — an unclosed call is unusable."""
    text = str(text)
    imbalance = abs(text.count("<tool_call>") - text.count("</tool_call>"))
    return -penalty * imbalance


def answer_of(text: str) -> str:
    """The part after the reasoning block, which is what gets checked."""
    text = str(text)
    return text.split("</think>", 1)[-1].strip() if "</think>" in text else text.strip()


def deterministic_scorer(_prompt: str, response: str) -> float:
    """A stand-in for a learned reward model.

    Scores brevity and punctuation — deliberately trivial. Its purpose is that
    the tests, and anyone trying the RL path, need no 3.6 GB download. MiniMind
    defaulted to `../../internlm2-1_8b-reward`, a sibling directory it never
    documented as a prerequisite.
    """
    answer = answer_of(response)
    if not answer:
        return -1.0
    tidy = 0.5 if answer[-1] in ".!?。！？" else 0.0
    brevity = max(-1.0, min(1.0, (400 - len(answer)) / 400))
    return tidy + brevity


@dataclass
class Rubric:
    """A weighted sum of rewards, clipped, with the parts kept for inspection.

    `breakdown()` returns every component. A single number tells you a run is
    improving; it does not tell you the model found a way to earn points
    without answering, which is what happened to MiniMind's own published
    GRPO and agent models.
    """

    weight_length: float = 1.0
    weight_thinking: float = 1.0
    weight_tool_tags: float = 1.0
    weight_ground_truth: float = 2.5
    weight_repetition: float = 1.0
    weight_scorer: float = 1.0
    # A safety rail, not the normal operating point. The first version shipped
    # 3.0 against components summing to 5.75, so a correct answer and a wrong
    # one both saturated to 3.0 and ranked identically — the mirror of the
    # inert clip in rl.py, and just as invisible. `saturates` exists so this
    # cannot come back silently.
    clip: float = 6.0
    scorer: Callable[[str, str], float] | None = field(default=None)

    def breakdown(
        self, prompt: str, response: str, expected: Iterable | None = None
    ) -> dict[str, float]:
        answer = answer_of(response)
        parts = {
            "length": self.weight_length * length_reward(response),
            "thinking": self.weight_thinking * thinking_reward(response),
            "tool_tags": self.weight_tool_tags * tool_tag_reward(response),
            "repetition": -self.weight_repetition * repetition_penalty(answer),
        }
        expected = list(expected or [])
        if expected:
            hits = ground_truth_hits(answer, expected)
            parts["ground_truth"] = self.weight_ground_truth * len(hits) / len(expected)
        if self.scorer is not None:
            parts["scorer"] = self.weight_scorer * self.scorer(prompt, response)
        return parts

    def score(
        self, prompt: str, response: str, expected: Iterable | None = None
    ) -> float:
        total = sum(self.breakdown(prompt, response, expected).values())
        return max(-self.clip, min(self.clip, total))

    def saturates(
        self, prompt: str, response: str, expected: Iterable | None = None
    ) -> bool:
        """True if the clip changed this score.

        Two responses that both saturate are ranked identically no matter how
        differently they answered, so a rubric that saturates routinely has
        stopped measuring anything. Check it on a batch before trusting the
        advantages computed from it.
        """
        total = sum(self.breakdown(prompt, response, expected).values())
        return abs(total) > self.clip

    def score_batch(
        self, prompts: list[str], responses: list[str], expected: list | None = None
    ) -> list[float]:
        if len(prompts) != len(responses):
            raise ValueError(f"{len(prompts)} prompts for {len(responses)} responses")
        if expected is not None and len(expected) != len(responses):
            raise ValueError(f"{len(expected)} expected sets for {len(responses)} responses")
        return [
            self.score(prompt, response, None if expected is None else expected[index])
            for index, (prompt, response) in enumerate(zip(prompts, responses))
        ]
