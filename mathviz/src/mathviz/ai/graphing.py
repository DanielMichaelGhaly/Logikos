from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import re


@dataclass
class DesmosConfig:
    # Expressions to plot, e.g. y = sin(x)
    expressions: list[str]
    # Optional viewport settings
    xmin: float | None = None
    xmax: float | None = None
    ymin: float | None = None
    ymax: float | None = None


TRIG_KEYWORDS = ("sin", "cos", "tan", "cot", "sec", "csc")


def _extract_math_expression(text: str) -> Optional[str]:
    """Extract a math expression from natural language text.
    Handles patterns like:
      - y = sin(x)
      - f(x) = sin(x)
      - derivative of sin(x)
      - differentiate sin(x)
      - integral of sin(x)
      - integrate sin(x)
    Returns the RHS or the core function like 'sin(x)'.
    """
    t = (text or "").strip()
    if not t:
        return None

    # If prompt starts with an imperative like 'differentiate'/'derivative of', try to extract RHS after '=' first
    m0 = re.search(r"=\s*([^,\.!?]+)", t)
    if m0:
        rhs = m0.group(1).strip()
        if rhs:
            return rhs

    # If there's an equals sign, prefer RHS
    if "=" in t:
        parts = t.split("=", 1)
        rhs = parts[1].strip()
        return rhs if rhs else None

    # derivative/integral detection
    m = re.search(r"derivative of\s+([^,.!?]+)", t, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"differentiate\s+([^,.!?]+)", t, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"integral of\s+([^,.!?]+)", t, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"integrate\s+([^,.!?]+)", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # Fallback: if it looks like a trig function mention, try to pull token
    if any(k in t.lower() for k in TRIG_KEYWORDS):
        # naive extraction of token containing parentheses
        m2 = re.search(r"([a-zA-Z]+\s*\([^)]*\))", t)
        if m2:
            return m2.group(1).replace(" ", "").strip()

    return None


def maybe_make_desmos_config(solution: Dict[str, Any]) -> Optional[DesmosConfig]:
    """Heuristically derive a Desmos plot config from the solution/problem.

    Preference order:
      1) If final_answer.type == 'derivative', plot original and derivative
      2) Extract from problem text if it contains a recognizable expression
    """
    try:
        problem_text = (
            solution.get("problem", {}).get("text")
            or solution.get("problem", {}).get("problem_text")
            or ""
        )
        final_answer = solution.get("final_answer")

        # Case 1: derivative - plot function and its derivative
        if isinstance(final_answer, dict) and final_answer.get("type") == "derivative":
            orig = final_answer.get("original_expression")
            deriv = final_answer.get("derivative")
            exprs = [e for e in [orig, deriv] if isinstance(e, str) and e.strip()]
            if exprs:
                return DesmosConfig(expressions=exprs)

        # Case 2: extract from problem text
        if isinstance(problem_text, str):
            expr = _extract_math_expression(problem_text)
            if expr:
                return DesmosConfig(expressions=[expr])
    except Exception:
        pass

    return None
