"""Tool execution, and a calculator the model cannot escape from.

`agent.py` builds multi-turn tool-use trajectories. Until this module existed
it built them for tools that could not run, which is a hole in something we
shipped rather than something we left out.

The arithmetic sandbox is the one piece of MiniMind this project copies in
spirit rather than correcting. Letting a model evaluate an expression it wrote
itself is a code-execution hole if you reach for `eval`, and MiniMind did the
right thing: parse to an AST, walk it with a whitelist, and guard the one
operator that can hang a process without being large. That design is credited
in CREDITS.md and reimplemented here.

What it must refuse, and does:

- attribute access, imports, comprehensions, calls to anything unlisted
- `9**9**9` and `10**99999`, which are short strings and unbounded work
- names it does not know, keyword arguments, lambdas, walrus, f-strings
"""

from __future__ import annotations

import ast
import math
import operator
import re
from dataclasses import dataclass
from typing import Any, Callable

MAX_EXPRESSION_LENGTH = 512
MAX_EXPONENT = 10_000
MAX_RESULT_DIGITS = 100

_SAFE_FUNCTIONS = {
    name: getattr(math, name)
    for name in (
        "sqrt exp log log2 log10 sin cos tan asin acos atan atan2 sinh cosh tanh "
        "floor ceil trunc fabs fmod hypot gcd degrees radians"
    ).split()
}
_SAFE_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}

# Typed arithmetic that a model writes by habit and Python does not accept.
#
# Two things here were got wrong first time and are worth the comment.
# `^` means exponentiation to anyone typing it, so it maps to `**`, not `*` —
# the first version returned 6 for `3^2` and a test had been written to expect
# exactly that, which is how a passing suite locks in a wrong answer.
# Superscripts expand rather than vanish, for the same reason.
_SUBSTITUTIONS = str.maketrans({
    "^": "**", "×": "*", "÷": "/", "−": "-", "–": "-",
    "²": "**2", "³": "**3", "（": "(", "）": ")",
})

# Thousands separators only — between digits, three at a time. A blanket comma
# strip turns `gcd(12,18)` into `gcd(1218)`, which is a wrong answer rather
# than an error.
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?:\D|$))")


def normalise(expression: str) -> str:
    return _THOUSANDS.sub("", str(expression)).translate(_SUBSTITUTIONS).strip()


def _guarded_pow(base: float, exponent: float) -> float:
    """`2**10` is fine. `9**9**9` is eleven characters and will not finish."""
    if abs(exponent) > MAX_EXPONENT:
        raise ValueError(f"exponent {exponent} exceeds the limit of {MAX_EXPONENT}")
    if abs(base) > 1 and abs(exponent) * math.log10(abs(base)) > MAX_RESULT_DIGITS:
        raise ValueError("result would have more digits than anyone wants")
    return base**exponent


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: _guarded_pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def safe_eval(expression: str) -> float:
    """Evaluate arithmetic. Everything not explicitly permitted is refused."""
    text = normalise(expression)
    if not text:
        raise ValueError("empty expression")
    if len(text) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"expression of {len(text)} characters exceeds {MAX_EXPRESSION_LENGTH}")

    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ValueError(f"not a valid expression: {text[:60]!r}") from None

    def resolve(node) -> str | None:
        """Accept `sqrt(4)`; refuse `math.sqrt(4)` and every other attribute."""
        return node.id if isinstance(node, ast.Name) else None

    def walk(node):
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                return node.value
            raise ValueError(f"only numbers are allowed, got {type(node.value).__name__}")
        if isinstance(node, ast.Name):
            if node.id in _SAFE_CONSTANTS:
                return _SAFE_CONSTANTS[node.id]
            raise ValueError(f"unknown name {node.id!r}")
        if isinstance(node, ast.UnaryOp):
            if type(node.op) not in _UNARY:
                raise ValueError(f"operator {type(node.op).__name__} is not allowed")
            return _UNARY[type(node.op)](walk(node.operand))
        if isinstance(node, ast.BinOp):
            if type(node.op) not in _BINARY:
                raise ValueError(f"operator {type(node.op).__name__} is not allowed")
            return _BINARY[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.Call):
            if node.keywords:
                raise ValueError("keyword arguments are not allowed")
            name = resolve(node.func)
            if name not in _SAFE_FUNCTIONS:
                raise ValueError(f"unknown function {name or type(node.func).__name__!r}")
            return _SAFE_FUNCTIONS[name](*(walk(arg) for arg in node.args))
        raise ValueError(f"{type(node).__name__} is not allowed in an expression")

    try:
        result = walk(tree.body)
    except ZeroDivisionError:
        raise ValueError("division by zero") from None
    except (TypeError, OverflowError) as error:
        raise ValueError(f"arithmetic failed: {error}") from None

    if isinstance(result, complex):
        raise ValueError("complex results are not supported")
    if isinstance(result, float) and (math.isnan(result) or math.isinf(result)):
        raise ValueError(f"result is not a finite number: {result}")
    return result


# ------------------------------------------------------------------ registry
@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    run: Callable[[dict], Any]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _calculate(arguments: dict) -> dict:
    return {"result": safe_eval(arguments["expression"])}


CALCULATOR = Tool(
    name="calculate",
    description="Evaluate an arithmetic expression.",
    parameters={
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    },
    run=_calculate,
)


class ToolRegistry:
    """What the model is allowed to call, and what happens when it tries.

    A failed call returns an error *to the model* rather than raising, because
    a trajectory where the model misuses a tool and recovers is a trajectory
    worth training on. A call to something unregistered is also an error
    returned, never a dispatch.
    """

    def __init__(self, tools: list[Tool] | None = None):
        self.tools: dict[str, Tool] = {}
        for tool in tools if tools is not None else [CALCULATOR]:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"{tool.name!r} is already registered")
        self.tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        return [tool.schema() for tool in self.tools.values()]

    def missing_arguments(self, tool: Tool, arguments: dict) -> list[str]:
        required = tool.parameters.get("required", [])
        return [name for name in required if name not in arguments]

    def execute(self, name: str, arguments: Any) -> dict:
        tool = self.tools.get(name)
        if tool is None:
            return {"error": f"no such tool: {name!r}", "available": sorted(self.tools)}
        if not isinstance(arguments, dict):
            return {"error": f"arguments must be an object, got {type(arguments).__name__}"}

        absent = self.missing_arguments(tool, arguments)
        if absent:
            return {"error": f"missing required argument(s): {', '.join(absent)}"}

        try:
            return tool.run(arguments)
        except ValueError as error:
            return {"error": str(error)}
        except Exception as error:  # a tool's own bug must not kill the rollout
            return {"error": f"{type(error).__name__}: {error}"}
