"""Safe arithmetic calculator using Python AST (never eval)."""

from __future__ import annotations

import ast
import math
import operator
import re
from typing import Any

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_SAFE_FUNCS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": lambda *xs: sum(xs),
    "sqrt": math.sqrt,
    "ceil": math.ceil,
    "floor": math.floor,
}

_SAFE_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
}


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_CONSTANTS:
            return _SAFE_CONSTANTS[node.id]
        raise ValueError(f"Unknown identifier: {node.id}")
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.BinOp):
        op = _BINARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent too large")
        return op(left, right)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
            raise ValueError("Function call not permitted")
        if node.keywords:
            raise ValueError("Keyword arguments not permitted")
        args = [_eval_node(a) for a in node.args]
        return _SAFE_FUNCS[node.func.id](*args)
    raise ValueError(f"Disallowed AST node: {type(node).__name__}")


def safe_calculate(expression: str) -> float | int:
    """Safely evaluate an arithmetic expression via AST."""
    if not isinstance(expression, str) or not expression.strip():
        raise ValueError("Expression must be a non-empty string")
    if len(expression) > 256:
        raise ValueError("Expression too long")
    cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", expression.strip())
    tree = ast.parse(cleaned, mode="eval")
    return _eval_node(tree)


def run_calculator(args: dict[str, Any], _context: dict[str, Any] | None = None) -> dict[str, Any]:
    """Tool entrypoint for calculator."""
    expr = args.get("expression") or args.get("expr") or ""
    try:
        result = safe_calculate(str(expr))
        return {"ok": True, "expression": expr, "result": result}
    except Exception as exc:
        return {"ok": False, "expression": expr, "error": str(exc)}
