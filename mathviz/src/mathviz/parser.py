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
        elif any(word in text_lower for word in ["solve", "equation", "="]):
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
        """Extract equations from the problem text."""
        equations = []
        
        # Special handling for "solve for" pattern
        solve_pattern = r"solve\s+for\s+\w+:\s*(.+?)\s*=\s*(.+)$"
        solve_match = re.search(solve_pattern, text, re.IGNORECASE)
        
        if solve_match:
            left_side, right_side = solve_match.groups()
            equations.append(Equation(
                left_side=left_side.strip(),
                right_side=right_side.strip()
            ))
        else:
            # Try other patterns
            for pattern in self.equation_patterns:
                if pattern.startswith("solve"):  # Skip the solve pattern as we handled it above
                    continue
                matches = re.findall(pattern, text)
                for match in matches:
                    if len(match) == 2:
                        left_side, right_side = match
                        equations.append(Equation(
                            left_side=left_side.strip(),
                            right_side=right_side.strip()
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
