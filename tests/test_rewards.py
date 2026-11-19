"""Reward functions.

The headline case is MiniMind PR #904 — ground truth matched by value rather
than by substring. Substring matching on numbers inflates every score it
touches, and it is the kind of wrongness that makes a training curve look
better than the model is.
"""

from __future__ import annotations

import pytest

from m3g4m1nd.rewards import (
    Rubric,
    answer_of,
    deterministic_scorer,
    ground_truth_hits,
    length_reward,
    numbers_in,
    repetition_penalty,
    thinking_reward,
    tool_tag_reward,
)


# ------------------------------------------------------------------ PR #904
def test_a_number_is_not_found_inside_a_longer_number():
    """9 does not appear in "19". MiniMind's substring check said it did."""
    assert ground_truth_hits("the answer is 19", [9]) == set()
    assert ground_truth_hits("the answer is 9", [9]) == {"9"}


def test_a_truncated_number_is_not_a_hit():
    assert ground_truth_hits("144000", [1440]) == set()
    assert ground_truth_hits("144000", [144000]) == {"144000"}


def test_numbers_are_compared_by_value_across_formatting():
    assert ground_truth_hits("that is 1,000 exactly", [1000]) == {"1000"}
    assert ground_truth_hits("that is 2.50", ["2.5"]) == {"2.5"}
    assert ground_truth_hits("-5 degrees", [-5]) == {"-5"}


def test_text_answers_still_match_case_insensitively():
    assert ground_truth_hits("It is Durban today", ["durban"]) == {"durban"}
    assert ground_truth_hits("It is Durban today", ["Cape Town"]) == set()


def test_several_expected_answers_are_counted_independently():
    assert ground_truth_hits("71 squared is 5041", [71, 5041]) == {"71", "5041"}
    assert ground_truth_hits("71 squared is 5041", [71, 9999]) == {"71"}


def test_empty_expectations_are_ignored():
    assert ground_truth_hits("anything", ["", "  ", None]) == set()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("3 and 4", [3.0, 4.0]),
        ("x1 and y2", []),                 # part of an identifier
        ("1,234", [1234.0]),
        ("version 2.5.1", []),             # not a plain number
        ("-7 +8", [-7.0, 8.0]),
    ],
)
def test_number_extraction_only_takes_standalone_numbers(text, expected):
    assert numbers_in(text) == expected


# ------------------------------------------------------------------ components
def test_length_reward_rewards_a_sensible_answer_and_punishes_the_rest():
    assert length_reward("x" * 100) == 0.5
    assert length_reward("hi") == -0.5
    assert length_reward("x" * 5000) == -0.5


def test_length_bounds_are_validated():
    with pytest.raises(ValueError, match="must not exceed"):
        length_reward("x", low=100, high=10)


def test_thinking_reward_requires_exactly_one_closed_block():
    good = "<think>" + "x" * 50 + "</think>answer"
    assert thinking_reward(good) == pytest.approx(1.25)

    doubled = "<think>" + "x" * 50 + "</think>mid</think>answer"
    assert thinking_reward(doubled) < thinking_reward(good)

    assert thinking_reward("no thinking at all") == 0.0


def test_thinking_reward_punishes_an_empty_or_rambling_block():
    assert thinking_reward("<think></think>answer") == pytest.approx(0.25 - 0.5)
    assert thinking_reward("<think>" + "x" * 5000 + "</think>a") == pytest.approx(0.25 - 0.5)


def test_tool_tags_must_balance():
    assert tool_tag_reward("<tool_call>{}</tool_call>") == 0.0
    assert tool_tag_reward("<tool_call>{}") == -0.5
    assert tool_tag_reward("<tool_call>{}<tool_call>{}</tool_call>") == -0.5


def test_repetition_penalty_scales_with_repetition_and_is_capped():
    assert repetition_penalty("one two three four five") == 0.0
    chant = "we dance together " * 20
    assert repetition_penalty(chant) == pytest.approx(0.5)
    assert repetition_penalty("") == 0.0


def test_repetition_penalty_validates_its_window():
    with pytest.raises(ValueError, match="n must be >= 1"):
        repetition_penalty("text", n=0)


def test_answer_of_strips_the_reasoning_block():
    assert answer_of("<think>musing</think>  the answer ") == "the answer"
    assert answer_of("just an answer") == "just an answer"


def test_the_default_scorer_needs_no_model():
    """MiniMind's RL scripts defaulted to a 3.6 GB sibling directory that was
    never listed as a prerequisite."""
    assert deterministic_scorer("q", "A short, tidy answer.") > 0
    assert deterministic_scorer("q", "") == -1.0


# ------------------------------------------------------------------ rubric
def test_the_rubric_reports_every_component_not_just_a_total():
    rubric = Rubric(scorer=deterministic_scorer)
    parts = rubric.breakdown("q", "<think>" + "x" * 50 + "</think>It is 42.", [42])
    assert set(parts) == {"length", "thinking", "tool_tags", "repetition",
                          "ground_truth", "scorer"}
    assert parts["ground_truth"] == pytest.approx(2.5)


def test_a_wrong_answer_scores_below_a_right_one():
    rubric = Rubric(scorer=deterministic_scorer)
    body = "<think>" + "x" * 50 + "</think>The answer is "
    assert rubric.score("q", body + "42.", [42]) > rubric.score("q", body + "19.", [42])


def test_the_rubric_catches_the_failure_mode_minimind_published():
    """Its released GRPO and agent models collapsed into chanting one phrase.
    A rubric that only summed a learned score would not have seen it; the
    repetition component does."""
    rubric = Rubric(scorer=deterministic_scorer)
    answer = "<think>" + "x" * 50 + "</think>"
    chanting = answer + "we dance together " * 30 + "42."
    plain = answer + "It is 42."

    assert rubric.breakdown("q", chanting, [42])["repetition"] == pytest.approx(-0.5)
    assert rubric.score("q", plain, [42]) > rubric.score("q", chanting, [42])


def test_the_total_is_clipped_both_ways():
    rubric = Rubric(clip=1.0, scorer=deterministic_scorer)
    best = rubric.score("q", "<think>" + "x" * 50 + "</think>It is 42.", [42])
    worst = rubric.score("q", "<tool_call>" * 20, [42])
    assert best <= 1.0 and worst >= -1.0


def test_ground_truth_is_omitted_when_nothing_is_expected():
    parts = Rubric().breakdown("q", "an answer", None)
    assert "ground_truth" not in parts


def test_the_scorer_is_omitted_when_none_is_supplied():
    assert "scorer" not in Rubric().breakdown("q", "an answer")


def test_batch_scoring_refuses_mismatched_inputs():
    rubric = Rubric()
    with pytest.raises(ValueError, match="2 prompts for 3 responses"):
        rubric.score_batch(["a", "b"], ["x", "y", "z"])
    with pytest.raises(ValueError, match="1 expected sets for 2 responses"):
        rubric.score_batch(["a", "b"], ["x", "y"], [[1]])


def test_batch_scoring_pairs_each_response_with_its_own_expectation():
    rubric = Rubric()
    scores = rubric.score_batch(
        ["q", "q"], ["It is 42.", "It is 42."], [[42], [99]]
    )
    assert scores[0] > scores[1]


def test_scores_feed_group_advantages_without_further_handling():
    """Joins up with rl.py: a rubric's output is the input to advantages."""
    import torch

    from m3g4m1nd.rl import GRPOUpdater

    rubric = Rubric(scorer=deterministic_scorer)
    body = "<think>" + "x" * 50 + "</think>The answer is "
    rewards = torch.tensor(
        rubric.score_batch(["q"] * 4, [body + s for s in ("42.", "19.", "42.", "7.")],
                           [[42]] * 4)
    )
    advantages = GRPOUpdater.group_advantages(rewards, num_generations=2)
    assert advantages.shape == (4,)
    assert advantages[0].item() > advantages[1].item()


def test_a_full_marks_answer_does_not_saturate_the_clip():
    """The fault the three tests above caught: with clip=3.0 against components
    summing to 5.75, a right answer and a wrong one both pinned at 3.0 and
    ranked the same. The clip is a rail, not the operating point."""
    rubric = Rubric(scorer=deterministic_scorer)
    best = "<think>" + "x" * 50 + "</think>The answer is 42."
    assert not rubric.saturates("q", best, [42]), (
        "a legitimately good answer is hitting the clip, so nothing better can "
        "be distinguished from it"
    )


def test_saturation_is_detectable_when_it_does_happen():
    tight = Rubric(clip=1.0, scorer=deterministic_scorer)
    best = "<think>" + "x" * 50 + "</think>The answer is 42."
    assert tight.saturates("q", best, [42])
    assert tight.score("q", best, [42]) == pytest.approx(1.0)


def test_two_saturating_responses_rank_identically_which_is_the_warning():
    """Demonstrates why `saturates` is worth checking rather than assumed."""
    tight = Rubric(clip=1.0, scorer=deterministic_scorer)
    body = "<think>" + "x" * 50 + "</think>The answer is "
    right = tight.score("q", body + "42.", [42])
    also_right = tight.score("q", body + "42 exactly.", [42])
    assert right == also_right == pytest.approx(1.0)
