#!/usr/bin/env python3
"""
LaTeX Formatter

Formats mathematical expressions and solutions as LaTeX for display.
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class FormattedSolution:
    """Container for formatted mathematical solution"""
    problem_statement: str
    solution_latex: str
    step_by_step: List[str]
    verification_latex: str = ""
    metadata: Dict[str, Any] = None


class LaTeXFormatter:
    """Formats mathematical content as LaTeX"""
    
    def __init__(self):
        self.math_symbols = {
            'pi': r'\pi',
            'theta': r'\theta',
            'alpha': r'\alpha',
            'beta': r'\beta',
            'gamma': r'\gamma',
            'delta': r'\delta',
            'sqrt': r'\sqrt',
            'infinity': r'\infty',
            '+-': r'\pm',
            '<=': r'\leq',
            '>=': r'\geq',
            '!=': r'\neq',
            'integral': r'\int',
            'sum': r'\sum',
            'prod': r'\prod',
        }
    
    def format_problem_solution(self, 
                              problem: str, 
                              ai_explanation: str,
                              sympy_result: Any,
                              verification_status: str = "unknown") -> FormattedSolution:
        """Format a complete problem solution with AI and SymPy results"""
        
        # Format problem statement
        problem_latex = self._format_problem_statement(problem)
        
        # Extract and format the main solution
        solution_latex = self._format_main_solution(sympy_result)
        
        # Format step-by-step explanation
        steps = self._format_steps_from_explanation(ai_explanation)
        
        # Format verification
        verification_latex = self._format_verification(verification_status, sympy_result)
        
        return FormattedSolution(
            problem_statement=problem_latex,
            solution_latex=solution_latex,
            step_by_step=steps,
            verification_latex=verification_latex,
            metadata={
                'verification_status': verification_status,
                'total_steps': len(steps)
            }
        )
    
    def _format_problem_statement(self, problem: str) -> str:
        """Format the problem statement as LaTeX"""
        
        # Basic formatting
        formatted = problem.strip()
        
        # Convert common mathematical notation
        formatted = self._apply_math_symbols(formatted)
        
        # Handle equations
        if '=' in formatted:
            # Split on = and format each side
            parts = formatted.split('=')
            if len(parts) == 2:
                left = self._format_expression(parts[0].strip())
                right = self._format_expression(parts[1].strip())
                return f"$${left} = {right}$$"
        
        # Handle "solve" statements
        if formatted.lower().startswith('solve'):
            equation_part = formatted[5:].strip()  # Remove "solve"
            return f"\\text{{Solve: }} ${self._format_expression(equation_part)}$"
        
        # Handle "find" statements  
        if formatted.lower().startswith('find'):
            return f"\\text{{{formatted}}} "
            
        # Default formatting
        return f"\\text{{{formatted}}}"
    
    def _format_main_solution(self, sympy_result) -> str:
        """Format the main solution result as LaTeX"""
        
        if hasattr(sympy_result, 'latex_result') and sympy_result.latex_result:
            return sympy_result.latex_result
        
        if hasattr(sympy_result, 'result'):
            result = sympy_result.result
            
            if isinstance(result, list):
                if len(result) == 1:
                    return f"$$x = {self._sympy_to_latex(result[0])}$$"
                else:
                    solutions = [f"x = {self._sympy_to_latex(sol)}" for sol in result]
                    return f"$${', '.join(solutions)}$$"
            else:
                return f"$$x = {self._sympy_to_latex(result)}$$"
        
        return "$$\\text{Solution not available}$$"
    
    def _format_steps_from_explanation(self, explanation: str) -> List[str]:
        """Extract and format steps from AI explanation"""
        steps = []
        
        # Look for numbered steps
        step_pattern = re.compile(r'(Step \d+:.*?)(?=Step \d+:|$)', re.DOTALL | re.IGNORECASE)
        step_matches = step_pattern.findall(explanation)
        
        if step_matches:
            for i, step in enumerate(step_matches, 1):
                formatted_step = self._format_step_content(step.strip())
                steps.append(f"**Step {i}:** {formatted_step}")
        else:
            # Fallback: split by paragraphs or sentences
            paragraphs = [p.strip() for p in explanation.split('\n\n') if p.strip()]
            for i, paragraph in enumerate(paragraphs[:5], 1):  # Limit to 5 steps
                if len(paragraph) > 20:  # Skip very short paragraphs
                    formatted_step = self._format_step_content(paragraph)
                    steps.append(f"**Step {i}:** {formatted_step}")
        
        return steps
    
    def _format_step_content(self, content: str) -> str:
        """Format the content of a single step"""
        
        # Clean up the content
        content = content.strip()
        
        # Remove redundant "Step X:" prefixes
        content = re.sub(r'^Step \d+:\s*', '', content, flags=re.IGNORECASE)
        
        # Format mathematical expressions in the content
        content = self._format_inline_math(content)
        
        return content
    
    def _format_inline_math(self, text: str) -> str:
        """Format mathematical expressions within text"""
        
        # Pattern to find mathematical expressions
        math_patterns = [
            r'(\b\d+x\b)',           # 2x, 3x, etc.
            r'(\bx\s*=\s*[^,.\s]+)', # x = something
            r'([+-]?\d+/\d+)',       # fractions
            r'(\d+\^\d+)',           # exponents
            r'(x\^?\d+)',            # x^2, x2, etc.
        ]
        
        for pattern in math_patterns:
            def replacer(match):
                expr = match.group(1)
                # Convert to LaTeX
                expr = expr.replace('^', '^{') + '}' if '^' in expr else expr
                expr = self._apply_math_symbols(expr)
                return f'${expr}$'
            
            text = re.sub(pattern, replacer, text)
        
        return text
    
    def _format_verification(self, status: str, sympy_result) -> str:
        """Format verification information"""
        
        if status == "match":
            return r"\textcolor{green}{\checkmark \text{ Verified by SymPy}}"
        elif status == "mismatch":  
            return r"\textcolor{red}{\times \text{ Does not match SymPy result}}"
        elif status == "sympy_error":
            return r"\textcolor{orange}{\text{SymPy verification failed}}"
        else:
            return r"\textcolor{gray}{\text{Verification pending}}"
    
    def _format_expression(self, expr: str) -> str:
        """Format a mathematical expression"""
        
        expr = expr.strip()
        
        # Handle implicit multiplication: 2x -> 2x (already correct)
        # Handle exponents: x^2 -> x^{2}
        expr = re.sub(r'(\w)\^(\d+)', r'\1^{\2}', expr)
        
        # Apply symbol replacements
        expr = self._apply_math_symbols(expr)
        
        return expr
    
    def _apply_math_symbols(self, text: str) -> str:
        """Apply mathematical symbol replacements"""
        
        for symbol, latex in self.math_symbols.items():
            text = text.replace(symbol, latex)
        
        return text
    
    def _sympy_to_latex(self, sympy_expr) -> str:
        """Convert SymPy expression to LaTeX"""
        
        try:
            # If it's a SymPy object, use its latex method
            if hasattr(sympy_expr, 'latex'):
                return sympy_expr.latex()
            elif hasattr(sympy_expr, '_latex'):
                return sympy_expr._latex()
            else:
                # Convert to string and apply basic LaTeX formatting
                expr_str = str(sympy_expr)
                return self._format_expression(expr_str)
        except:
            return str(sympy_expr)


def test_latex_formatter():
    """Test the LaTeX formatter"""
    formatter = LaTeXFormatter()
    
    # Mock SymPy result
    class MockSymPyResult:
        def __init__(self, result, success=True):
            self.result = result
            self.success = success
            self.latex_result = f"$$x = {result}$$"
    
    print("🧪 Testing LaTeX Formatter")
    print("=" * 50)
    
    # Test case
    problem = "solve 2x+5=0"
    ai_explanation = """Step 1: Isolate the Variable Term
First, we need to isolate the term that contains the variable x. To do this, we subtract 5 from both sides: 2x + 5 - 5 = 0 - 5

Step 2: Solve for x  
Now divide both sides by 2: 2x/2 = -5/2
Therefore: x = -5/2"""
    
    sympy_result = MockSymPyResult([-5/2])
    
    formatted = formatter.format_problem_solution(
        problem, ai_explanation, sympy_result, "match"
    )
    
    print(f"Problem: {formatted.problem_statement}")
    print(f"Solution: {formatted.solution_latex}")
    print(f"Steps: {len(formatted.step_by_step)}")
    for i, step in enumerate(formatted.step_by_step, 1):
        print(f"  {i}. {step[:60]}...")
    print(f"Verification: {formatted.verification_latex}")


if __name__ == "__main__":
    test_latex_formatter()