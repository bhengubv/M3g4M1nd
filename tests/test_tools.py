"""The arithmetic sandbox and the tool registry.

Closes the hole in `agent.py`: it built tool-use trajectories against tools
that could not execute.

The sandbox is the piece credited to MiniMind's design rather than corrected,
so it is tested hardest. A model writes the expression; the expression must not
be able to reach anything.
"""

from __future__ import annotations

import math

import pytest

from m3g4m1nd.tools import (
    MAX_EXPRESSION_LENGTH,
    CALCULATOR,
    Tool,
    ToolRegistry,
    safe_eval,
)


# ------------------------------------------------------------------ arithmetic
@pytest.mark.parametrize(
    "expression,expected",
    [
        ("1+1", 2), ("256*37", 9472), ("2**10", 1024), ("(94)-35", 59),
        ("10/4", 2.5), ("7//2", 3), ("7%3", 1), ("-5", -5), ("+5", 5),
        ("sqrt(144)", 12), ("log10(1000)", 3), ("hypot(3,4)", 5),
        ("floor(2.7)", 2), ("ceil(2.1)", 3), ("fabs(-3)", 3), ("gcd(12,18)", 6),
        ("atan2(1,1)", math.pi/4), ("fmod(7,3)", 1),
        ("(20**3)*((198)/11)", 144000),
    ],
)
def test_ordinary_arithmetic_works(expression, expected):
    assert safe_eval(expression) == pytest.approx(expected)


def test_constants_are_available():
    assert safe_eval("pi") == pytest.approx(math.pi)
    assert safe_eval("2*pi") == pytest.approx(2 * math.pi)


@pytest.mark.parametrize(
    "written,expected",
    [("3^2", 9), ("6×7", 42), ("10÷4", 2.5), ("5−3", 2), ("（2+3）*2", 10), ("1,000+1", 1001), ("5²", 25), ("2³", 8),
     ("1,000,000/1000", 1000)],
)
def test_characters_a_model_actually_types_are_normalised(written, expected):
    assert safe_eval(written) == pytest.approx(expected)


# ------------------------------------------------------------------ the refusals
@pytest.mark.parametrize(
    "attack",
    [
        "__import__('os').system('echo hi')",
        "().__class__",
        "math.sqrt(4)",                 # attribute access, even to a safe module
        "open('/etc/passwd')",
        "exec('x=1')",
        "eval('1')",
        "[x for x in range(3)]",
        "lambda: 1",
        "{'a': 1}",
        "'string'",
        "f'{1}'",
        "globals()",
        "print(1)",
        "1 if True else 2",
        "(y := 4)",
        "1 < 2",
        "1 and 2",
        "~1",
        "1 << 3",
    ],
)
def test_everything_outside_arithmetic_is_refused(attack):
    with pytest.raises(ValueError):
        safe_eval(attack)


@pytest.mark.parametrize("bomb", ["9**9**9", "10**99999", "2**100000", "(10**50)**50"])
def test_short_expressions_that_would_never_finish_are_refused(bomb):
    with pytest.raises(ValueError, match="exponent|digits"):
        safe_eval(bomb)


def test_a_legitimately_large_power_still_works():
    assert safe_eval("2**64") == 2**64


def test_division_by_zero_is_an_error_not_a_crash():
    with pytest.raises(ValueError, match="division by zero"):
        safe_eval("1/0")


def test_an_overlong_expression_is_refused_before_parsing():
    with pytest.raises(ValueError, match="exceeds"):
        safe_eval("1+" * MAX_EXPRESSION_LENGTH + "1")


def test_empty_and_malformed_expressions_are_refused():
    for text in ("", "   ", "1+", ")(", "sqrt(", "1 2"):
        with pytest.raises(ValueError):
            safe_eval(text)


def test_unknown_names_and_functions_are_named_in_the_error():
    with pytest.raises(ValueError, match="unknown name 'banana'"):
        safe_eval("banana+1")
    with pytest.raises(ValueError, match="unknown function 'frobnicate'"):
        safe_eval("frobnicate(1)")


def test_keyword_arguments_are_refused():
    with pytest.raises(ValueError, match="keyword arguments"):
        safe_eval("hypot(3, y=4)")


def test_argument_separators_survive_thousands_normalisation():
    """A blanket comma strip made `gcd(12,18)` into `gcd(1218)` — a wrong
    answer rather than an error, which is the worse of the two."""
    assert safe_eval("gcd(12,18)") == 6
    assert safe_eval("hypot(3,4)") == 5
    assert safe_eval("1,000+1") == 1001


def test_caret_means_exponentiation_not_multiplication():
    """The first version mapped `^` to `*` and a test was written expecting 6."""
    assert safe_eval("3^2") == 9
    assert safe_eval("2^10") == 1024


def test_a_non_finite_result_is_refused():
    with pytest.raises(ValueError, match="not a finite number"):
        safe_eval("inf")
    with pytest.raises(ValueError, match="not a finite number|arithmetic failed"):
        safe_eval("inf-inf")


def test_a_complex_result_is_refused_rather_than_returned():
    with pytest.raises(ValueError):
        safe_eval("sqrt(-1)")


# ------------------------------------------------------------------ registry
def test_the_calculator_is_registered_by_default():
    registry = ToolRegistry()
    assert "calculate" in registry.tools
    assert registry.execute("calculate", {"expression": "2+2"}) == {"result": 4}


def test_the_schema_is_the_shape_the_chat_template_expects():
    schema = CALCULATOR.schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "calculate"
    assert schema["function"]["parameters"]["required"] == ["expression"]


def test_an_unknown_tool_returns_an_error_to_the_model():
    """A trajectory where the model misuses a tool and recovers is worth
    training on, so a bad call is an observation, not an exception."""
    result = ToolRegistry().execute("weather", {"city": "Durban"})
    assert "no such tool" in result["error"]
    assert result["available"] == ["calculate"]


def test_missing_required_arguments_are_named():
    result = ToolRegistry().execute("calculate", {})
    assert "missing required argument(s): expression" in result["error"]


def test_non_object_arguments_are_refused():
    result = ToolRegistry().execute("calculate", ["2+2"])
    assert "must be an object" in result["error"]


def test_a_bad_expression_comes_back_as_an_observation():
    result = ToolRegistry().execute("calculate", {"expression": "__import__('os')"})
    assert "error" in result and "result" not in result


def test_a_tool_that_raises_something_unexpected_does_not_kill_the_rollout():
    def explode(_arguments):
        raise RuntimeError("the tool itself is broken")

    registry = ToolRegistry(
        tools=[Tool("broken", "", {"type": "object", "properties": {}}, explode)]
    )
    result = registry.execute("broken", {})
    assert result["error"] == "RuntimeError: the tool itself is broken"


def test_registering_the_same_name_twice_is_refused():
    registry = ToolRegistry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(CALCULATOR)


def test_a_registry_can_be_built_empty_and_filled():
    registry = ToolRegistry(tools=[])
    assert registry.schemas() == []
    registry.register(CALCULATOR)
    assert len(registry.schemas()) == 1


# ------------------------------------------------- it now joins up with agent.py
def test_a_tool_result_becomes_an_observation_in_a_trajectory():
    """The hole this module closes: trajectories can now carry real results."""
    from m3g4m1nd.agent import Turn, trajectory_mask

    registry = ToolRegistry()
    observation = registry.execute("calculate", {"expression": "256*37"})
    assert observation == {"result": 9472}

    observation_tokens = [90, 91, 92]  # stands for the rendered result
    tokens, mask = trajectory_mask(
        [Turn(generated=[7, 8], observation=observation_tokens), Turn(generated=[9, 2])],
        prompt_len=1,
    )
    assert sum(mask) == 4, "only the policy's own tokens carry loss"
    assert tokens[2:5] == observation_tokens
