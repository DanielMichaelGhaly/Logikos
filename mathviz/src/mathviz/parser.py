"""
Natural language to schema conversion using regex and rules.
"""

import re
from typing import List, Optional

from .schemas import Equation, MathProblem, Variable


class MathParser:
    """Parser for converting natural language math problems to structured schemas."""

    def __init__(self) -> None:
        """Initialize the parser with common patterns."""
        self.equation_patterns = [
            r"solve\s+for\s+\w+:\s*(.+?)\s*=\s*(.+)",  # "solve for x: equation" pattern first
            r"(.+?)\s*=\s*(.+)",  # Basic equation pattern
        ]
        
        self.variable_patterns = [
            r"\b([a-zA-Z])\b",  # Single letter variables
            r"\\([a-zA-Z]+)",  # LaTeX variables like \alpha
        ]

    def parse(self, problem_text: str) -> MathProblem:
        """Parse natural language text into a MathProblem schema."""
        problem_type = self._identify_problem_type(problem_text)
        variables = self._extract_variables(problem_text)
        equations = self._extract_equations(problem_text)
        goal = self._extract_goal(problem_text)
        
        return MathProblem(
            problem_text=problem_text,
            problem_type=problem_type,
            variables=variables,
            equations=equations,
            goal=goal
        )

    def _identify_problem_type(self, text: str) -> str:
        """Identify the type of mathematical problem."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["derivative", "differentiate", "d/dx"]):
            return "calculus"
        elif any(word in text_lower for word in ["integral", "integrate", "area under"]):
            return "calculus"
        elif any(word in text_lower for word in ["maximize", "minimize", "optimal"]):
            return "optimization"
        elif any(word in text_lower for word in ["solve", "equation", "=", "roots", "root", "zeros", "zeroes", "factors", "factor"]):
            return "algebraic"
        else:
            return "general"

    def _extract_variables(self, text: str) -> List[Variable]:
        """Extract variables from the problem text."""
        variables = []
        found_vars = set()
        
        for pattern in self.variable_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in found_vars and len(match) == 1:
                    variables.append(Variable(name=match))
                    found_vars.add(match)
        
        return variables

    def _extract_equations(self, text: str) -> List[Equation]:
        """Extract equations from the problem text, robustly handling instruction phrases and extracting only the math part."""
        equations = []

        # Remove all known instruction phrases
        cleaned_text = text
        instruction_patterns = [
            r"solve\s+for\s+\w+:\s*",
            r"find\s+the\s+roots\s+of\s+",
            r"find\s+roots\s+of\s+",
            r"roots\s+of\s+",
            r"find\s+the\s+zeros\s+of\s+",
            r"find\s+zeros\s+of\s+",
            r"zeros\s+of\s+",
        ]
        for pat in instruction_patterns:
            cleaned_text = re.sub(pat, "", cleaned_text, flags=re.IGNORECASE)

        # If the original text was a 'roots' or 'zeros' instruction, treat as equation = 0
        if any(re.match(pat, text, re.IGNORECASE) for pat in instruction_patterns[1:]):
            expr = cleaned_text.strip()
            if expr:
                equations.append(Equation(
                    left_side=expr,
                    right_side="0"
                ))
        else:
            # Extract the first valid equation (left = right) from the cleaned text
            # This regex matches anything (non-greedy) before and after the first '='
            eq_match = re.search(r"(.+?)\s*=\s*(.+)", cleaned_text)
            if eq_match:
                left_side = eq_match.group(1).strip()
                right_side = eq_match.group(2).strip()
                # Ensure left_side does not contain instruction text
                for pat in instruction_patterns:
                    left_side = re.sub(pat, "", left_side, flags=re.IGNORECASE).strip()
                equations.append(Equation(
                    left_side=left_side,
                    right_side=right_side
                ))

        return equations

    def _extract_goal(self, text: str) -> str:
        """Extract the goal/objective from the problem text."""
        text_lower = text.lower()
        
        # Look for "solve for" patterns
        solve_match = re.search(r"solve\s+for\s+(\w+)", text_lower)
        if solve_match:
            return f"solve for {solve_match.group(1)}"
        
        # Look for "find" patterns
        find_match = re.search(r"find\s+(\w+)", text_lower)
        if find_match:
            return f"find {find_match.group(1)}"
        
        # Default goal
        return "solve the problem"
