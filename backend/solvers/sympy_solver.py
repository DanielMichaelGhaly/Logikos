"""
SymPy-based ground truth mathematical solver.
Provides ultra-accurate mathematical computation for confidence comparison.
"""

import time
import traceback
from typing import List, Optional

import sympy as sp
from sympy import symbols, solve, factor, expand, simplify, diff, integrate, limit, latex
from sympy.parsing.sympy_parser import parse_expr
from sympy.core.sympify import SympifyError

from shared.schemas import SympyResult, QuestionType, QuestionClassification


class SympySolver:
    """Ultra-accurate SymPy solver for ground truth mathematical computation."""

    def __init__(self):
        """Initialize the SymPy solver."""
        # Common symbols
        self.x, self.y, self.z = symbols('x y z')
        self.t = symbols('t')

    def solve_question(self, classification: QuestionClassification) -> SympyResult:
        """
        Solve a mathematical question using SymPy.

        Args:
            classification: Structured question from the classifier

        Returns:
            SympyResult with computation results and steps
        """
        start_time = time.time()

        try:
            # Parse the mathematical expression
            expr = self._parse_expression(classification.expression)

            # Solve based on question type
            if classification.type == QuestionType.SOLVE:
                result, steps = self._solve_equation(expr)
            elif classification.type == QuestionType.FACTOR:
                result, steps = self._factor_expression(expr)
            elif classification.type == QuestionType.EXPAND:
                result, steps = self._expand_expression(expr)
            elif classification.type == QuestionType.SIMPLIFY:
                result, steps = self._simplify_expression(expr)
            elif classification.type == QuestionType.DERIVATIVE:
                result, steps = self._compute_derivative(expr)
            elif classification.type == QuestionType.INTEGRAL:
                result, steps = self._compute_integral(expr)
            elif classification.type == QuestionType.LIMIT:
                result, steps = self._compute_limit(expr)
            elif classification.type == QuestionType.ROOTS:
                result, steps = self._find_roots(expr)
            else:
                result = f"Question type '{classification.type}' not supported by SymPy solver"
                steps = ["Unsupported question type"]

            execution_time = time.time() - start_time

            return SympyResult(
                success=True,
                result=str(result),
                steps=steps,
                latex=latex(result) if hasattr(result, '__iter__') == False else latex(result[0]) if result else None,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"SymPy error: {str(e)}"

            return SympyResult(
                success=False,
                error=error_msg,
                execution_time=execution_time
            )

    def _parse_expression(self, expression_str: str) -> sp.Basic:
        """Parse string expression into SymPy object."""
        # Clean up common formatting issues
        cleaned = expression_str.strip()

        # Convert ^ to ** for SymPy exponentiation
        cleaned = cleaned.replace('^', '**')

        # Add implicit multiplication for expressions like 2x
        import re
        cleaned = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', cleaned)

        # Handle equations (contains =)
        if '=' in cleaned:
            left, right = cleaned.split('=', 1)
            left_expr = parse_expr(left.strip())
            right_expr = parse_expr(right.strip())
            return sp.Eq(left_expr, right_expr)

        # Handle regular expressions
        return parse_expr(cleaned)

    def _solve_equation(self, expr) -> tuple:
        """Solve an equation or expression."""
        steps = []

        if isinstance(expr, sp.Eq):
            # It's an equation
            steps.append(f"Given equation: {expr}")
            solutions = solve(expr, self.x)
            steps.append(f"Solving for x: {solutions}")
        else:
            # Assume it's an expression to solve for zero
            steps.append(f"Given expression: {expr}")
            steps.append(f"Setting equal to zero: {expr} = 0")
            solutions = solve(expr, self.x)
            steps.append(f"Solutions: {solutions}")

        return solutions, steps

    def _factor_expression(self, expr) -> tuple:
        """Factor an expression."""
        steps = []
        steps.append(f"Original expression: {expr}")

        factored = factor(expr)
        steps.append(f"Factored form: {factored}")

        return factored, steps

    def _expand_expression(self, expr) -> tuple:
        """Expand an expression."""
        steps = []
        steps.append(f"Original expression: {expr}")

        expanded = expand(expr)
        steps.append(f"Expanded form: {expanded}")

        return expanded, steps

    def _simplify_expression(self, expr) -> tuple:
        """Simplify an expression."""
        steps = []
        steps.append(f"Original expression: {expr}")

        simplified = simplify(expr)
        steps.append(f"Simplified form: {simplified}")

        return simplified, steps

    def _compute_derivative(self, expr) -> tuple:
        """Compute derivative of an expression."""
        steps = []
        steps.append(f"Function: f(x) = {expr}")

        derivative = diff(expr, self.x)
        steps.append(f"Derivative: f'(x) = {derivative}")

        return derivative, steps

    def _compute_integral(self, expr) -> tuple:
        """Compute integral of an expression."""
        steps = []
        steps.append(f"Function: f(x) = {expr}")

        integral = integrate(expr, self.x)
        steps.append(f"Integral: ∫f(x)dx = {integral} + C")

        return integral, steps

    def _compute_limit(self, expr) -> tuple:
        """Compute limit of an expression."""
        steps = []
        steps.append(f"Function: f(x) = {expr}")

        # Default limit as x approaches 0
        limit_value = limit(expr, self.x, 0)
        steps.append(f"Limit as x→0: {limit_value}")

        return limit_value, steps

    def _find_roots(self, expr) -> tuple:
        """Find roots of an expression."""
        steps = []
        steps.append(f"Function: f(x) = {expr}")
        steps.append("Finding roots (zeros) of the function")

        roots = solve(expr, self.x)
        steps.append(f"Roots: {roots}")

        return roots, steps

    def extract_numerical_result(self, sympy_result: SympyResult) -> Optional[str]:
        """
        Extract numerical result for confidence comparison.

        Args:
            sympy_result: SymPy computation result

        Returns:
            String representation of numerical result or None
        """
        if not sympy_result.success or not sympy_result.result:
            return None

        try:
            # Handle different result types
            result_str = sympy_result.result

            # For lists (like multiple solutions)
            if result_str.startswith('[') and result_str.endswith(']'):
                return result_str

            # For single values
            return result_str

        except Exception:
            return None