"""
AI-Enhanced Mathematical Problem Solver with SymPy fallback and step tracing.
Supports advanced calculus, optimization, and AI-powered problem solving.
"""

import sympy as sp
import numpy as np
from typing import Dict, List, Any, Optional
import logging

from .schemas import MathProblem, MathSolution, Equation
from .trace import StepTrace, Step
from .reasoning import ReasoningGenerator
from .viz import Visualizer
try:
    from .ai_apis import solve_with_ai, AIResponse
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("AI APIs not available - using symbolic solver only")
try:
    from .advanced_calculus import AdvancedCalculus, OptimizationResult, GradientDescentResult
    ADVANCED_CALCULUS_AVAILABLE = True
except ImportError:
    ADVANCED_CALCULUS_AVAILABLE = False
    print("Advanced calculus module not available")

logger = logging.getLogger(__name__)


class MathSolver:
    """AI-Enhanced mathematical problem solver with SymPy fallback."""

    def __init__(self, use_ai: bool = True, ai_first: bool = True):
        self.reasoner = ReasoningGenerator()
        self.visualizer = Visualizer()
        self.step_counter = 0
        self.use_ai = use_ai and AI_AVAILABLE
        self.ai_first = ai_first  # Try AI first, then fallback to symbolic
        
        if ADVANCED_CALCULUS_AVAILABLE:
            self.advanced_calc = AdvancedCalculus()
        else:
            self.advanced_calc = None
            
        logger.info(f"MathSolver initialized - AI: {self.use_ai}, Advanced Calculus: {ADVANCED_CALCULUS_AVAILABLE}")

    def solve(self, problem: MathProblem) -> MathSolution:
        """Solve a mathematical problem with step-by-step tracing using AI and/or symbolic methods."""
        trace = StepTrace(problem_id=f"problem_{hash(problem.problem_text)}")
        self.step_counter = 0
        
        try:
            # Try AI first if enabled
            ai_solution = None
            if self.use_ai and self.ai_first:
                ai_solution = self._try_ai_solution(problem, trace)
            
            # Determine problem type and solve
            if problem.problem_type == "algebraic":
                solution_data = self._solve_algebraic(problem, trace, ai_solution)
            elif problem.problem_type == "calculus":
                solution_data = self._solve_calculus(problem, trace, ai_solution)
            elif problem.problem_type == "optimization":
                solution_data = self._solve_optimization(problem, trace, ai_solution)
            else:
                solution_data = self._solve_general(problem, trace, ai_solution)
            
            # Try AI as fallback if symbolic failed and AI not tried yet
            if (solution_data.get("status") == "not_implemented" or 
                solution_data.get("error")) and self.use_ai and not ai_solution:
                ai_fallback = self._try_ai_solution(problem, trace)
                if ai_fallback and ai_fallback.success:
                    solution_data["ai_fallback"] = ai_fallback.content
                    solution_data["ai_provider"] = ai_fallback.provider
            
            # Generate reasoning and visualization
            reasoning = self.reasoner.generate_reasoning(trace)
            visualization = self.visualizer.generate_latex(trace)
            
            return MathSolution(
                problem=problem,
                solution_steps=[step.__dict__ for step in trace.steps],
                final_answer=solution_data,
                reasoning=reasoning,
                visualization=visualization,
                metadata={"trace_id": trace.problem_id, "step_count": len(trace.steps)}
            )
            
        except Exception as e:
            trace.success = False
            trace.error_message = str(e)
            return MathSolution(
                problem=problem,
                solution_steps=[step.__dict__ for step in trace.steps],
                final_answer={"error": str(e)},
                reasoning=f"Error occurred during solving: {str(e)}",
                visualization="",
                metadata={"trace_id": trace.problem_id, "error": True}
            )
    
    def _try_ai_solution(self, problem: MathProblem, trace: StepTrace) -> Optional[Any]:
        """Attempt to solve problem using AI APIs."""
        if not self.use_ai:
            return None
        
        try:
            problem_type_map = {
                "calculus": "differentiation" if "derivative" in problem.problem_text.lower() else "calculus",
                "optimization": "optimization",
                "algebraic": "algebra"
            }
            
            ai_problem_type = problem_type_map.get(problem.problem_type, "general")
            
            self._add_step(trace, "ai_attempt", 
                         problem.problem_text,
                         "Attempting AI solution...",
                         f"Trying to solve using AI with problem type: {ai_problem_type}")
            
            ai_response = solve_with_ai(problem.problem_text, ai_problem_type)
            
            if ai_response.success:
                self._add_step(trace, "ai_solution",
                             problem.problem_text,
                             ai_response.content,
                             f"AI solution from {ai_response.provider} (confidence: {ai_response.confidence:.2f})")
                logger.info(f"AI solution successful: {ai_response.provider}")
                return ai_response
            else:
                self._add_step(trace, "ai_failed",
                             problem.problem_text,
                             f"AI failed: {ai_response.error}",
                             "AI solution attempt failed, falling back to symbolic methods")
                logger.warning(f"AI solution failed: {ai_response.error}")
                return None
                
        except Exception as e:
            logger.warning(f"AI solution attempt error: {e}")
            self._add_step(trace, "ai_error",
                         problem.problem_text,
                         f"AI error: {str(e)}",
                         "AI solution encountered an error")
            return None

    def _solve_algebraic(self, problem: MathProblem, trace: StepTrace, ai_solution: Optional[Any] = None) -> Dict[str, Any]:
        """Solve algebraic equations using SymPy with optional AI assistance."""
        
        # If AI provided a solution, incorporate it
        if ai_solution and ai_solution.success:
            self._add_step(trace, "ai_integration",
                         ai_solution.content,
                         "Integrating AI solution with symbolic verification",
                         f"Using AI solution from {ai_solution.provider} as guidance")
        
        if not problem.equations:
            raise ValueError("No equations found in algebraic problem")
        
        # Convert problem variables to SymPy symbols
        symbols = {}
        for var in problem.variables:
            if var.domain == "integer":
                symbols[var.name] = sp.Symbol(var.name, integer=True)
            elif var.domain == "positive":
                symbols[var.name] = sp.Symbol(var.name, positive=True)
            else:
                symbols[var.name] = sp.Symbol(var.name, real=True)
        
        trace.initial_state = f"Variables: {list(symbols.keys())}"
        
        # Process equations
        equations = []
        for eq in problem.equations:
            self._add_step(trace, "parse_equation", 
                         f"Original: {eq.left_side} = {eq.right_side}",
                         f"Parsing equation: {eq.left_side} = {eq.right_side}",
                         "Converting natural language equation to symbolic form")
            
            # Convert equation strings to SymPy expressions
            try:
                # Preprocess expressions to handle common mathematical notation
                left_processed = self._preprocess_expression(eq.left_side)
                right_processed = self._preprocess_expression(eq.right_side)
                
                left_expr = sp.sympify(left_processed, locals=symbols)
                right_expr = sp.sympify(right_processed, locals=symbols)
                sympy_eq = sp.Eq(left_expr, right_expr)
                equations.append(sympy_eq)
                
                self._add_step(trace, "symbolic_conversion",
                             f"{eq.left_side} = {eq.right_side}",
                             str(sympy_eq),
                             "Converted to symbolic equation")
            except Exception as e:
                raise ValueError(f"Failed to parse equation '{eq.left_side} = {eq.right_side}': {str(e)}")
        
        # Determine variables to solve for
        if "solve for" in problem.goal:
            target_var = problem.goal.split("solve for")[-1].strip()
            solve_vars = [symbols.get(target_var, sp.Symbol(target_var))]
        else:
            solve_vars = list(symbols.values())
        
        self._add_step(trace, "identify_target", 
                     f"Goal: {problem.goal}",
                     f"Solving for: {[str(v) for v in solve_vars]}",
                     "Identified target variables")
        
        # Solve the system of equations
        if len(equations) == 1 and len(solve_vars) == 1:
            solutions = sp.solve(equations[0], solve_vars[0])
        else:
            solutions = sp.solve(equations, solve_vars)
        
        self._add_step(trace, "solve",
                     f"System: {[str(eq) for eq in equations]}",
                     f"Solutions: {solutions}",
                     "Applied SymPy solver to find solutions")
        
        # Format solutions
        if isinstance(solutions, list):
            formatted_solutions = {str(solve_vars[0]): [sp.simplify(sol) for sol in solutions]}
        elif isinstance(solutions, dict):
            formatted_solutions = {str(k): sp.simplify(v) for k, v in solutions.items()}
        else:
            formatted_solutions = {str(solve_vars[0]): sp.simplify(solutions)}
        
        trace.final_state = f"Solutions: {formatted_solutions}"
        
        return {
            "type": "algebraic_solution",
            "solutions": formatted_solutions,
            "method": "sympy_solve",
            "equations_solved": len(equations)
        }

    def _solve_calculus(self, problem: MathProblem, trace: StepTrace, ai_solution: Optional[Any] = None) -> Dict[str, Any]:
        """Solve calculus problems using advanced calculus module and AI assistance."""
        
        # If AI provided a solution, incorporate it
        if ai_solution and ai_solution.success:
            self._add_step(trace, "ai_calculus_guidance",
                         ai_solution.content,
                         "AI calculus solution available",
                         f"AI guidance from {ai_solution.provider}")
        
        # Extract the main expression from problem text
        problem_text = problem.problem_text.lower()
        
        # Enhanced pattern matching for calculus operations
        if "derivative" in problem_text or "differentiate" in problem_text:
            if "partial" in problem_text and self.advanced_calc:
                return self._solve_partial_derivative(problem, trace, ai_solution)
            else:
                return self._solve_derivative(problem, trace, ai_solution)
        elif "gradient" in problem_text and self.advanced_calc:
            return self._solve_gradient(problem, trace, ai_solution)
        elif "integral" in problem_text or "integrate" in problem_text:
            return self._solve_integral(problem, trace, ai_solution)
        elif "optimize" in problem_text or "critical points" in problem_text:
            if self.advanced_calc:
                return self._solve_optimization(problem, trace, ai_solution)
            else:
                raise ValueError("Optimization requires advanced calculus module")
        else:
            raise ValueError("Unrecognized calculus problem type")
    
    def _solve_derivative(self, problem: MathProblem, trace: StepTrace, ai_solution: Optional[Any] = None) -> Dict[str, Any]:
        """Solve derivative problems."""
        # For now, handle simple cases - this could be expanded significantly
        import re
        
        # Try to extract expression from problem text
        # Look for patterns like "derivative of x^2 + 3x" or "d/dx(x^2)"
        text = problem.problem_text
        
        # Simple pattern: "derivative of [expression]"
        match = re.search(r"derivative of ([^,\.]+)", text, re.IGNORECASE)
        if not match:
            match = re.search(r"differentiate ([^,\.]+)", text, re.IGNORECASE)
        
        if not match:
            raise ValueError("Could not extract expression for differentiation")
        
        expr_str = match.group(1).strip()
        
        # Create symbol (assume x unless specified)
        x = sp.Symbol('x')
        
        self._add_step(trace, "parse_expression",
                     f"Original: {text}",
                     f"Expression to differentiate: {expr_str}",
                     "Extracted expression for differentiation")
        
        # Convert to SymPy expression with preprocessing
        try:
            processed_expr = self._preprocess_expression(expr_str)
            expr = sp.sympify(processed_expr, locals={'x': x})
        except:
            raise ValueError(f"Could not parse expression: {expr_str}")
        
        # Compute derivative
        derivative = sp.diff(expr, x)
        simplified = sp.simplify(derivative)
        
        self._add_step(trace, "differentiate",
                     f"f(x) = {expr}",
                     f"f'(x) = {derivative}",
                     "Applied differentiation rules")
        
        if simplified != derivative:
            self._add_step(trace, "simplify",
                         f"f'(x) = {derivative}",
                         f"f'(x) = {simplified}",
                         "Simplified the result")
        
        trace.final_state = f"Derivative: {simplified}"
        
        return {
            "type": "derivative",
            "original_expression": str(expr),
            "derivative": str(simplified),
            "variable": "x"
        }
    
    def _solve_integral(self, problem: MathProblem, trace: StepTrace, ai_solution: Optional[Any] = None) -> Dict[str, Any]:
        """Solve integration problems."""
        import re
        
        text = problem.problem_text
        
        # Simple pattern: "integral of [expression]"
        match = re.search(r"integral of ([^,\.]+)", text, re.IGNORECASE)
        if not match:
            match = re.search(r"integrate ([^,\.]+)", text, re.IGNORECASE)
        
        if not match:
            raise ValueError("Could not extract expression for integration")
        
        expr_str = match.group(1).strip()
        
        x = sp.Symbol('x')
        
        self._add_step(trace, "parse_expression",
                     f"Original: {text}",
                     f"Expression to integrate: {expr_str}",
                     "Extracted expression for integration")
        
        try:
            processed_expr = self._preprocess_expression(expr_str)
            expr = sp.sympify(processed_expr, locals={'x': x})
        except:
            raise ValueError(f"Could not parse expression: {expr_str}")
        
        # Compute integral
        integral = sp.integrate(expr, x)
        simplified = sp.simplify(integral)
        
        self._add_step(trace, "integrate",
                     f"∫ {expr} dx",
                     f"= {integral}",
                     "Applied integration rules")
        
        if simplified != integral:
            self._add_step(trace, "simplify",
                         f"{integral}",
                         f"{simplified}",
                         "Simplified the result")
        
        trace.final_state = f"Integral: {simplified} + C"
        
        return {
            "type": "integral",
            "original_expression": str(expr),
            "integral": str(simplified),
            "variable": "x",
            "note": "Don't forget the constant of integration +C"
        }

    def _solve_optimization(self, problem: MathProblem, trace: StepTrace, ai_solution: Optional[Any] = None) -> Dict[str, Any]:
        """Solve optimization problems."""
        # Placeholder for optimization - would implement constraint handling, Lagrange multipliers, etc.
        self._add_step(trace, "optimization_setup",
                     problem.problem_text,
                     "Setting up optimization problem",
                     "Optimization problems require more advanced implementation")
        
        return {
            "type": "optimization",
            "status": "not_implemented",
            "message": "Optimization solving not yet implemented"
        }

    def _solve_general(self, problem: MathProblem, trace: StepTrace, ai_solution: Optional[Any] = None) -> Dict[str, Any]:
        """Handle general mathematical problems."""
        self._add_step(trace, "general_analysis",
                     problem.problem_text,
                     "Analyzing general problem",
                     "General problem type - using heuristic approach")
        
        return {
            "type": "general",
            "status": "analyzed",
            "message": "General problem handling - may need manual intervention"
        }

    def _preprocess_expression(self, expr: str) -> str:
        """Preprocess mathematical expressions to handle common notation."""
        import re
        
        # Handle exponentiation: x^2 -> x**2
        expr = re.sub(r'([a-zA-Z0-9)])\^', r'\1**', expr)
        
        # Handle implicit multiplication like 2x -> 2*x
        # Pattern: digit followed by letter
        expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr)
        
        # Pattern: closing parenthesis followed by letter or opening parenthesis
        expr = re.sub(r'\)([a-zA-Z(])', r')*\1', expr)
        
        # Pattern: letter followed by opening parenthesis
        expr = re.sub(r'([a-zA-Z])\(', r'\1*(', expr)
        
        return expr

    def _add_step(self, trace: StepTrace, operation: str, before: str, after: str, justification: str) -> None:
        """Add a step to the solution trace."""
        step = Step(
            step_id=self.step_counter,
            operation=operation,
            expression_before=before,
            expression_after=after,
            justification=justification
        )
        trace.add_step(step)
        self.step_counter += 1
    
    def _solve_partial_derivative(self, problem: MathProblem, trace: StepTrace, ai_solution: Optional[Any] = None) -> Dict[str, Any]:
        """Solve partial derivative problems using advanced calculus."""
        if not self.advanced_calc:
            raise ValueError("Advanced calculus module not available")
        
        import re
        text = problem.problem_text
        
        # Extract expression and variable
        match = re.search(r"partial derivative of ([^,\.]+)(?:\s*with respect to|\s*w\.r\.t\.?)\s*([a-zA-Z]+)", text, re.IGNORECASE)
        if not match:
            raise ValueError("Could not extract expression and variable for partial differentiation")
        
        expr_str = match.group(1).strip()
        variable = match.group(2).strip()
        
        self._add_step(trace, "parse_partial_derivative",
                     text,
                     f"Expression: {expr_str}, Variable: {variable}",
                     "Extracted expression and variable for partial differentiation")
        
        # Use advanced calculus module
        result = self.advanced_calc.compute_partial_derivative(expr_str, variable)
        
        if result.get("error"):
            raise ValueError(result["error"])
        
        # Add advanced calculus steps to trace
        for step in result["steps"]:
            self._add_step(trace, step.step_type,
                         step.input_expr,
                         step.result,
                         step.description)
        
        trace.final_state = f"∂/∂{variable}({expr_str}) = {result['derivative']}"
        
        return {
            "type": "partial_derivative",
            "original_expression": expr_str,
            "derivative": result["derivative"],
            "variable": variable,
            "variables": result.get("variables", []),
            "method": "advanced_calculus"
        }
    
    def _solve_gradient(self, problem: MathProblem, trace: StepTrace, ai_solution: Optional[Any] = None) -> Dict[str, Any]:
        """Solve gradient problems using advanced calculus."""
        if not self.advanced_calc:
            raise ValueError("Advanced calculus module not available")
        
        import re
        text = problem.problem_text
        
        # Extract expression
        match = re.search(r"gradient of ([^,\.]+)", text, re.IGNORECASE)
        if not match:
            raise ValueError("Could not extract expression for gradient computation")
        
        expr_str = match.group(1).strip()
        
        self._add_step(trace, "parse_gradient",
                     text,
                     f"Expression for gradient: {expr_str}",
                     "Extracted expression for gradient computation")
        
        # Use advanced calculus module
        result = self.advanced_calc.compute_gradient(expr_str)
        
        if result.get("error"):
            raise ValueError(result["error"])
        
        # Add advanced calculus steps to trace
        for step in result["steps"]:
            self._add_step(trace, step.step_type,
                         step.input_expr,
                         step.result,
                         step.description)
        
        trace.final_state = f"∇({expr_str}) = {result['gradient']}"
        
        return {
            "type": "gradient",
            "original_expression": expr_str,
            "gradient": result["gradient"],
            "components": result["components"],
            "variables": result["variables"],
            "method": "advanced_calculus"
        }
    
    def _solve_critical_points(self, problem: MathProblem, trace: StepTrace, ai_solution: Optional[Any] = None) -> Dict[str, Any]:
        """Find critical points using advanced calculus."""
        if not self.advanced_calc:
            raise ValueError("Advanced calculus module not available")
        
        import re
        text = problem.problem_text
        
        # Extract expression
        match = re.search(r"critical points of ([^,\.]+)", text, re.IGNORECASE)
        if not match:
            match = re.search(r"optimize ([^,\.]+)", text, re.IGNORECASE)
        if not match:
            raise ValueError("Could not extract expression for critical points")
        
        expr_str = match.group(1).strip()
        
        self._add_step(trace, "parse_optimization",
                     text,
                     f"Function to optimize: {expr_str}",
                     "Extracted function for critical point analysis")
        
        # Use advanced calculus module
        result = self.advanced_calc.find_critical_points(expr_str)
        
        # Add advanced calculus steps to trace
        for step in result.steps:
            self._add_step(trace, step.step_type,
                         step.input_expr,
                         step.result,
                         step.description)
        
        if result.critical_points:
            points_str = ", ".join([str(point) for point in result.critical_points])
            trace.final_state = f"Critical points: {points_str}"
        else:
            trace.final_state = "No critical points found"
        
        return {
            "type": "critical_points",
            "original_expression": expr_str,
            "critical_points": result.critical_points,
            "function_values": result.function_values,
            "gradient": str(result.gradient) if result.gradient else None,
            "hessian": str(result.hessian) if result.hessian else None,
            "method": "advanced_calculus"
        }
