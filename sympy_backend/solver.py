#!/usr/bin/env python3
"""
SymPy Mathematical Solver

Handles solving of parsed mathematical expressions using SymPy.
"""

import sympy as sp
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass

from .expression_parser import ParsedExpression


@dataclass
class SolutionResult:
    """Container for solution results"""
    success: bool
    original_problem: str
    problem_type: str
    operation: str
    result: Any
    steps: List[str]
    latex_result: str = ""
    error_message: str = ""
    metadata: Dict[str, Any] = None


class SymPySolver:
    """SymPy-based mathematical problem solver"""
    
    def __init__(self):
        self.solution_steps = []
    
    def solve(self, parsed: ParsedExpression) -> SolutionResult:
        """Solve a parsed mathematical expression"""
        self.solution_steps = []
        
        try:
            if parsed.problem_type == 'error':
                return SolutionResult(
                    success=False,
                    original_problem=parsed.original_text,
                    problem_type=parsed.problem_type,
                    operation=parsed.operation,
                    result=None,
                    steps=[],
                    error_message=parsed.metadata.get('error', 'Unknown parsing error')
                )
            
            # Route to appropriate solver method
            if parsed.operation == 'solve':
                return self._solve_equation(parsed)
            elif parsed.operation == 'differentiate':
                return self._differentiate(parsed)
            elif parsed.operation == 'integrate':
                return self._integrate(parsed)
            elif parsed.operation in ['simplify', 'evaluate']:
                return self._simplify_or_evaluate(parsed)
            else:
                return self._handle_general(parsed)
                
        except Exception as e:
            return SolutionResult(
                success=False,
                original_problem=parsed.original_text,
                problem_type=parsed.problem_type,
                operation=parsed.operation,
                result=None,
                steps=self.solution_steps,
                error_message=f"Solver error: {str(e)}"
            )
    
    def _solve_equation(self, parsed: ParsedExpression) -> SolutionResult:
        """Solve algebraic equations"""
        self._add_step(f"Parsing equation: {parsed.original_text}")
        
        equation = parsed.expressions[0]  # Should be an Eq object
        variables = parsed.variables
        
        self._add_step(f"Equation in symbolic form: {equation}")
        
        # Determine what to solve for
        if parsed.target_variable:
            target_var = sp.Symbol(parsed.target_variable)
            solve_for = target_var
        else:
            # If no specific target, solve for all variables
            solve_for = variables
        
        self._add_step(f"Solving for: {solve_for}")
        
        # Use SymPy's solve function
        solutions = sp.solve(equation, solve_for)
        
        if isinstance(solutions, list):
            if len(solutions) == 0:
                result_text = "No solutions found"
                latex_result = r"\text{No solutions}"
            else:
                result_text = f"Solutions: {solutions}"
                latex_result = self._format_solutions_latex(solve_for, solutions)
        elif isinstance(solutions, dict):
            result_text = f"Solutions: {solutions}"
            latex_result = self._format_dict_solutions_latex(solutions)
        else:
            result_text = f"Solution: {solutions}"
            latex_result = f"${sp.latex(solutions)}$"
        
        self._add_step(f"Final result: {result_text}")
        
        return SolutionResult(
            success=True,
            original_problem=parsed.original_text,
            problem_type=parsed.problem_type,
            operation=parsed.operation,
            result=solutions,
            steps=self.solution_steps,
            latex_result=latex_result,
            metadata={
                'equation': str(equation),
                'variables': [str(v) for v in variables],
                'target_variable': parsed.target_variable
            }
        )
    
    def _differentiate(self, parsed: ParsedExpression) -> SolutionResult:
        """Compute derivatives"""
        self._add_step(f"Computing derivative: {parsed.original_text}")
        
        expr = parsed.expressions[0]
        var_name = parsed.target_variable or 'x'
        var = sp.Symbol(var_name)
        
        self._add_step(f"Expression: {expr}")
        self._add_step(f"Differentiating with respect to: {var}")
        
        # Compute derivative
        derivative = sp.diff(expr, var)
        simplified = sp.simplify(derivative)
        
        if simplified != derivative:
            self._add_step(f"Derivative: {derivative}")
            self._add_step(f"Simplified: {simplified}")
            result = simplified
        else:
            self._add_step(f"Derivative: {derivative}")
            result = derivative
        
        latex_result = f"$\\frac{{d}}{{d{var}}} {sp.latex(expr)} = {sp.latex(result)}$"
        
        return SolutionResult(
            success=True,
            original_problem=parsed.original_text,
            problem_type=parsed.problem_type,
            operation=parsed.operation,
            result=result,
            steps=self.solution_steps,
            latex_result=latex_result,
            metadata={
                'original_expression': str(expr),
                'variable': var_name,
                'derivative': str(result)
            }
        )
    
    def _integrate(self, parsed: ParsedExpression) -> SolutionResult:
        """Compute integrals"""
        self._add_step(f"Computing integral: {parsed.original_text}")
        
        expr = parsed.expressions[0]
        var_name = parsed.target_variable or 'x'
        var = sp.Symbol(var_name)
        
        self._add_step(f"Expression: {expr}")
        self._add_step(f"Integrating with respect to: {var}")
        
        # Compute integral
        integral = sp.integrate(expr, var)
        simplified = sp.simplify(integral)
        
        if simplified != integral:
            self._add_step(f"Integral: {integral}")
            self._add_step(f"Simplified: {simplified}")
            result = simplified
        else:
            self._add_step(f"Integral: {integral}")
            result = integral
        
        # Add constant of integration note
        self._add_step("Note: Don't forget the constant of integration +C for indefinite integrals")
        
        latex_result = f"$\\int {sp.latex(expr)} \\, d{var} = {sp.latex(result)} + C$"
        
        return SolutionResult(
            success=True,
            original_problem=parsed.original_text,
            problem_type=parsed.problem_type,
            operation=parsed.operation,
            result=result,
            steps=self.solution_steps,
            latex_result=latex_result,
            metadata={
                'original_expression': str(expr),
                'variable': var_name,
                'integral': str(result)
            }
        )
    
    def _simplify_or_evaluate(self, parsed: ParsedExpression) -> SolutionResult:
        """Simplify or evaluate expressions"""
        self._add_step(f"Processing expression: {parsed.original_text}")
        
        expr = parsed.expressions[0]
        self._add_step(f"Expression: {expr}")
        
        if parsed.operation == 'simplify':
            result = sp.simplify(expr)
            self._add_step(f"Simplified: {result}")
        else:  # evaluate
            # Try to evaluate numerically if possible
            try:
                result = expr.evalf()
                self._add_step(f"Evaluated: {result}")
            except:
                result = sp.simplify(expr)
                self._add_step(f"Simplified: {result}")
        
        latex_result = f"${sp.latex(expr)} = {sp.latex(result)}$"
        
        return SolutionResult(
            success=True,
            original_problem=parsed.original_text,
            problem_type=parsed.problem_type,
            operation=parsed.operation,
            result=result,
            steps=self.solution_steps,
            latex_result=latex_result,
            metadata={'original_expression': str(expr), 'result': str(result)}
        )
    
    def _handle_general(self, parsed: ParsedExpression) -> SolutionResult:
        """Handle general expressions"""
        self._add_step(f"Processing general expression: {parsed.original_text}")
        
        if parsed.expressions:
            expr = parsed.expressions[0]
            result = sp.simplify(expr)
            self._add_step(f"Simplified: {result}")
            
            return SolutionResult(
                success=True,
                original_problem=parsed.original_text,
                problem_type=parsed.problem_type,
                operation='simplify',
                result=result,
                steps=self.solution_steps,
                latex_result=f"${sp.latex(result)}$",
                metadata={'expression': str(expr)}
            )
        else:
            return SolutionResult(
                success=False,
                original_problem=parsed.original_text,
                problem_type=parsed.problem_type,
                operation=parsed.operation,
                result=None,
                steps=self.solution_steps,
                error_message="No expressions to process"
            )
    
    def _format_solutions_latex(self, variable: sp.Symbol, solutions: List) -> str:
        """Format solution list as LaTeX"""
        if len(solutions) == 1:
            return f"${sp.latex(variable)} = {sp.latex(solutions[0])}$"
        else:
            solution_strs = [f"{sp.latex(variable)} = {sp.latex(sol)}" for sol in solutions]
            return f"${', '.join(solution_strs)}$"
    
    def _format_dict_solutions_latex(self, solutions: Dict) -> str:
        """Format solution dictionary as LaTeX"""
        solution_strs = [f"{sp.latex(var)} = {sp.latex(val)}" for var, val in solutions.items()]
        return f"${', '.join(solution_strs)}$"
    
    def _add_step(self, description: str):
        """Add a step to the solution process"""
        self.solution_steps.append(description)


def test_solver():
    """Test the SymPy solver with various problems"""
    from .expression_parser import EnhancedMathParser
    
    parser = EnhancedMathParser()
    solver = SymPySolver()
    
    test_cases = [
        "solve 2x+5=0",
        "find roots of x^2 - 4",
        "derivative of x^2 + 3x",
        "integral of x^2",
        "simplify (x^2 - 1)/(x - 1)",
    ]
    
    print("🧪 Testing SymPy Solver")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Problem: '{case}'")
        
        # Parse
        parsed = parser.parse(case)
        if parsed.problem_type == 'error':
            print(f"   ❌ Parse Error: {parsed.metadata.get('error')}")
            continue
        
        # Solve  
        result = solver.solve(parsed)
        
        if result.success:
            print(f"   ✅ Result: {result.result}")
            print(f"   📝 Steps: {len(result.steps)} steps")
            if result.latex_result:
                print(f"   📐 LaTeX: {result.latex_result}")
        else:
            print(f"   ❌ Error: {result.error_message}")


if __name__ == "__main__":
    test_solver()