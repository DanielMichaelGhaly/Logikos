"""
Mathematical Step-by-Step Explanation System

This module provides detailed, pedagogical explanations for mathematical operations.
Instead of generic steps, it breaks down mathematical reasoning into educational steps
that explain WHY and HOW each transformation happens.

For derivatives: explains power rule, constant rule, sum rule, etc.
For algebra: explains combining terms, isolating variables, etc.
For integration: explains basic integration rules, substitution, etc.
"""

import sympy as sp
import re
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from .trace import Step


@dataclass
class MathStep:
    """A single mathematical step with detailed explanation."""
    step_number: int
    rule_name: str  # e.g., "Power Rule", "Chain Rule", "Combine Like Terms"
    before_expression: str  # LaTeX or string
    after_expression: str   # LaTeX or string  
    explanation: str       # Detailed pedagogical explanation
    rule_formula: Optional[str] = None  # e.g., "d/dx[x^n] = n·x^(n-1)"
    latex_before: Optional[str] = None
    latex_after: Optional[str] = None


class DerivativeStepExplainer:
    """Generates step-by-step explanations for derivatives."""
    
    def __init__(self):
        """Initialize with derivative rules and patterns."""
        self.rules = {
            'power_rule': "d/dx[x^n] = n·x^(n-1)",
            'constant_rule': "d/dx[c] = 0 (where c is a constant)",
            'constant_multiple': "d/dx[c·f(x)] = c·d/dx[f(x)]",
            'sum_rule': "d/dx[f(x) + g(x)] = d/dx[f(x)] + d/dx[g(x)]",
            'product_rule': "d/dx[f(x)·g(x)] = f'(x)·g(x) + f(x)·g'(x)",
            'chain_rule': "d/dx[f(g(x))] = f'(g(x))·g'(x)",
            'trig_sin': "d/dx[sin(x)] = cos(x)",
            'trig_cos': "d/dx[cos(x)] = -sin(x)",
            'exp_rule': "d/dx[e^x] = e^x",
            'ln_rule': "d/dx[ln(x)] = 1/x"
        }
    
    def explain_derivative(self, original_expr: sp.Expr, variable: sp.Symbol = None) -> List[MathStep]:
        """Generate detailed step-by-step explanation for taking a derivative."""
        if variable is None:
            variable = sp.Symbol('x')
        
        steps = []
        step_count = 1
        
        # Step 1: Identify the function and what we're differentiating
        steps.append(MathStep(
            step_number=step_count,
            rule_name="Problem Setup",
            before_expression=f"f(x) = {original_expr}",
            after_expression=f"Find d/dx[{original_expr}]",
            explanation=f"We need to find the derivative of f(x) = {original_expr} with respect to x.",
            latex_before=f"f(x) = {sp.latex(original_expr)}",
            latex_after=f"\\frac{{d}}{{dx}}\\left[{sp.latex(original_expr)}\\right]"
        ))
        step_count += 1
        
        # Step 2: Break down the expression by structure
        if original_expr.is_Add:
            # Handle sum/difference
            steps.extend(self._explain_sum_rule(original_expr, variable, step_count))
            step_count += len(steps) - 1
        elif original_expr.is_Mul:
            # Handle products
            steps.extend(self._explain_product_or_constant_multiple(original_expr, variable, step_count))
            step_count += len(steps) - 1
        elif original_expr.is_Pow:
            # Handle powers
            steps.extend(self._explain_power_rule(original_expr, variable, step_count))
            step_count += len(steps) - 1
        else:
            # Handle individual terms
            steps.extend(self._explain_single_term(original_expr, variable, step_count))
            step_count += len(steps) - 1
        
        # Final step: Show the complete result
        final_derivative = sp.diff(original_expr, variable)
        simplified = sp.simplify(final_derivative)
        
        if simplified != final_derivative:
            steps.append(MathStep(
                step_number=step_count,
                rule_name="Simplification",
                before_expression=str(final_derivative),
                after_expression=str(simplified),
                explanation="Simplify the final expression by combining like terms.",
                latex_before=sp.latex(final_derivative),
                latex_after=sp.latex(simplified)
            ))
            step_count += 1
        
        steps.append(MathStep(
            step_number=step_count,
            rule_name="Final Answer",
            before_expression=f"d/dx[{original_expr}]",
            after_expression=f"f'(x) = {simplified}",
            explanation=f"Therefore, the derivative of {original_expr} is {simplified}.",
            latex_before=f"\\frac{{d}}{{dx}}\\left[{sp.latex(original_expr)}\\right]",
            latex_after=f"f'(x) = {sp.latex(simplified)}"
        ))
        
        return steps
    
    def _explain_sum_rule(self, expr: sp.Expr, var: sp.Symbol, start_step: int) -> List[MathStep]:
        """Explain sum rule application."""
        steps = []
        terms = sp.Add.make_args(expr)
        
        # Explain sum rule
        steps.append(MathStep(
            step_number=start_step,
            rule_name="Sum Rule",
            before_expression=str(expr),
            after_expression=f"d/dx[{terms[0]}] + d/dx[{terms[1]}]" + (f" + d/dx[{terms[2]}]" if len(terms) > 2 else ""),
            explanation="Apply the sum rule: the derivative of a sum is the sum of the derivatives.",
            rule_formula=self.rules['sum_rule'],
            latex_before=sp.latex(expr),
            latex_after=" + ".join([f"\\frac{{d}}{{dx}}\\left[{sp.latex(term)}\\right]" for term in terms])
        ))
        
        # Explain each term
        current_step = start_step + 1
        term_derivatives = []
        
        for i, term in enumerate(terms):
            if term.is_number:
                # Constant term
                steps.append(MathStep(
                    step_number=current_step,
                    rule_name="Constant Rule",
                    before_expression=f"d/dx[{term}]",
                    after_expression="0",
                    explanation=f"The derivative of the constant {term} is 0.",
                    rule_formula=self.rules['constant_rule'],
                    latex_before=f"\\frac{{d}}{{dx}}\\left[{sp.latex(term)}\\right]",
                    latex_after="0"
                ))
                term_derivatives.append(0)
            elif term.has(var) and term.is_Mul:
                # Handle terms like 3x, -2x^2, etc.
                steps.extend(self._explain_constant_multiple_term(term, var, current_step))
                term_derivatives.append(sp.diff(term, var))
                current_step += 1  # Adjust for additional steps
            elif term.is_Pow and term.base == var:
                # Handle x^n terms
                steps.extend(self._explain_power_rule(term, var, current_step))
                term_derivatives.append(sp.diff(term, var))
            else:
                # General term
                derivative = sp.diff(term, var)
                steps.append(MathStep(
                    step_number=current_step,
                    rule_name="Differentiate Term",
                    before_expression=f"d/dx[{term}]",
                    after_expression=str(derivative),
                    explanation=f"Differentiate the term {term}.",
                    latex_before=f"\\frac{{d}}{{dx}}\\left[{sp.latex(term)}\\right]",
                    latex_after=sp.latex(derivative)
                ))
                term_derivatives.append(derivative)
            
            current_step += 1
        
        return steps
    
    def _explain_power_rule(self, expr: sp.Expr, var: sp.Symbol, start_step: int) -> List[MathStep]:
        """Explain power rule application for terms like x^n."""
        steps = []
        
        if expr.is_Pow and expr.base == var:
            # Pure power: x^n
            base = expr.base
            exponent = expr.exp
            
            steps.append(MathStep(
                step_number=start_step,
                rule_name="Power Rule",
                before_expression=f"d/dx[{expr}]",
                after_expression=f"{exponent}·{base}^{exponent-1}",
                explanation=f"Apply the power rule: bring down the exponent {exponent}, then subtract 1 from the exponent.",
                rule_formula=self.rules['power_rule'],
                latex_before=f"\\frac{{d}}{{dx}}\\left[{sp.latex(expr)}\\right]",
                latex_after=f"{sp.latex(exponent)} \\cdot {sp.latex(base)}^{{{sp.latex(exponent-1)}}}"
            ))
            
            # Simplify if needed
            result = exponent * base**(exponent-1)
            if result != exponent * base**(exponent-1):  # If it simplifies
                simplified = sp.simplify(result)
                steps.append(MathStep(
                    step_number=start_step + 1,
                    rule_name="Simplify",
                    before_expression=f"{exponent}·{base}^{exponent-1}",
                    after_expression=str(simplified),
                    explanation="Simplify the expression.",
                    latex_before=f"{sp.latex(exponent)} \\cdot {sp.latex(base)}^{{{sp.latex(exponent-1)}}}",
                    latex_after=sp.latex(simplified)
                ))
        
        return steps
    
    def _explain_constant_multiple_term(self, expr: sp.Expr, var: sp.Symbol, start_step: int) -> List[MathStep]:
        """Explain constant multiple rule for terms like 3x, -2x^2, etc."""
        steps = []
        
        # Extract constant and variable part
        constant_factors = []
        variable_part = 1
        
        for factor in expr.as_ordered_factors():
            if factor.has(var):
                variable_part *= factor
            else:
                constant_factors.append(factor)
        
        constant = sp.Mul(*constant_factors) if constant_factors else 1
        
        steps.append(MathStep(
            step_number=start_step,
            rule_name="Constant Multiple Rule",
            before_expression=f"d/dx[{expr}]",
            after_expression=f"{constant} · d/dx[{variable_part}]",
            explanation=f"Factor out the constant {constant}. The derivative of a constant times a function is the constant times the derivative of the function.",
            rule_formula=self.rules['constant_multiple'],
            latex_before=f"\\frac{{d}}{{dx}}\\left[{sp.latex(expr)}\\right]",
            latex_after=f"{sp.latex(constant)} \\cdot \\frac{{d}}{{dx}}\\left[{sp.latex(variable_part)}\\right]"
        ))
        
        # Now differentiate the variable part
        if variable_part == var:
            # Simple x term
            steps.append(MathStep(
                step_number=start_step + 1,
                rule_name="Power Rule",
                before_expression=f"d/dx[{variable_part}]",
                after_expression="1",
                explanation=f"d/dx[x] = 1 (since x = x^1, we get 1·x^0 = 1)",
                rule_formula="d/dx[x] = 1",
                latex_before=f"\\frac{{d}}{{dx}}\\left[{sp.latex(variable_part)}\\right]",
                latex_after="1"
            ))
            
            steps.append(MathStep(
                step_number=start_step + 2,
                rule_name="Multiply",
                before_expression=f"{constant} · 1",
                after_expression=str(constant),
                explanation=f"Multiply: {constant} × 1 = {constant}",
                latex_before=f"{sp.latex(constant)} \\cdot 1",
                latex_after=sp.latex(constant)
            ))
        elif variable_part.is_Pow and variable_part.base == var:
            # Power term like x^n
            exponent = variable_part.exp
            derivative_of_power = exponent * var**(exponent-1)
            
            steps.append(MathStep(
                step_number=start_step + 1,
                rule_name="Power Rule",
                before_expression=f"d/dx[{variable_part}]",
                after_expression=str(derivative_of_power),
                explanation=f"Apply power rule to {variable_part}: bring down {exponent}, subtract 1 from exponent",
                rule_formula=self.rules['power_rule'],
                latex_before=f"\\frac{{d}}{{dx}}\\left[{sp.latex(variable_part)}\\right]",
                latex_after=sp.latex(derivative_of_power)
            ))
            
            final_result = constant * derivative_of_power
            simplified = sp.simplify(final_result)
            
            steps.append(MathStep(
                step_number=start_step + 2,
                rule_name="Multiply",
                before_expression=f"{constant} · {derivative_of_power}",
                after_expression=str(simplified),
                explanation=f"Multiply: {constant} × {derivative_of_power} = {simplified}",
                latex_before=f"{sp.latex(constant)} \\cdot {sp.latex(derivative_of_power)}",
                latex_after=sp.latex(simplified)
            ))
        
        return steps
    
    def _explain_product_or_constant_multiple(self, expr: sp.Expr, var: sp.Symbol, start_step: int) -> List[MathStep]:
        """Handle product rule or constant multiples."""
        factors = expr.as_ordered_factors()
        
        # Check if it's just a constant multiple
        has_constant = any(not factor.has(var) for factor in factors)
        variable_factors = [f for f in factors if f.has(var)]
        
        if has_constant and len(variable_factors) == 1:
            # It's a constant multiple
            return self._explain_constant_multiple_term(expr, var, start_step)
        else:
            # TODO: Implement product rule for genuine products
            # For now, fall back to basic explanation
            derivative = sp.diff(expr, var)
            return [MathStep(
                step_number=start_step,
                rule_name="Product Rule (Advanced)",
                before_expression=str(expr),
                after_expression=str(derivative),
                explanation="This requires the product rule - a more advanced topic.",
                rule_formula=self.rules['product_rule']
            )]
    
    def _explain_single_term(self, expr: sp.Expr, var: sp.Symbol, start_step: int) -> List[MathStep]:
        """Handle single terms that don't fit other categories."""
        steps = []
        
        if expr.is_number:
            # Constant
            steps.append(MathStep(
                step_number=start_step,
                rule_name="Constant Rule",
                before_expression=f"d/dx[{expr}]",
                after_expression="0",
                explanation=f"The derivative of any constant ({expr}) is 0.",
                rule_formula=self.rules['constant_rule'],
                latex_before=f"\\frac{{d}}{{dx}}\\left[{sp.latex(expr)}\\right]",
                latex_after="0"
            ))
        elif expr == var:
            # Simple x
            steps.append(MathStep(
                step_number=start_step,
                rule_name="Power Rule",
                before_expression=f"d/dx[x]",
                after_expression="1",
                explanation="d/dx[x] = 1 (since x = x^1, applying power rule gives 1·x^0 = 1)",
                rule_formula="d/dx[x] = 1",
                latex_before="\\frac{d}{dx}[x]",
                latex_after="1"
            ))
        elif str(expr) == 'sin(x)':
            steps.append(MathStep(
                step_number=start_step,
                rule_name="Trigonometric Rule",
                before_expression="d/dx[sin(x)]",
                after_expression="cos(x)",
                explanation="The derivative of sin(x) is cos(x)",
                rule_formula=self.rules['trig_sin'],
                latex_before="\\frac{d}{dx}[\\sin(x)]",
                latex_after="\\cos(x)"
            ))
        elif str(expr) == 'cos(x)':
            steps.append(MathStep(
                step_number=start_step,
                rule_name="Trigonometric Rule", 
                before_expression="d/dx[cos(x)]",
                after_expression="-sin(x)",
                explanation="The derivative of cos(x) is -sin(x)",
                rule_formula=self.rules['trig_cos'],
                latex_before="\\frac{d}{dx}[\\cos(x)]",
                latex_after="-\\sin(x)"
            ))
        else:
            # Generic case
            derivative = sp.diff(expr, var)
            steps.append(MathStep(
                step_number=start_step,
                rule_name="Differentiate",
                before_expression=str(expr),
                after_expression=str(derivative),
                explanation=f"Differentiate {expr} with respect to {var}",
                latex_before=sp.latex(expr),
                latex_after=sp.latex(derivative)
            ))
        
        return steps


class AlgebraicStepExplainer:
    """Generates step-by-step explanations for algebraic operations."""
    
    def explain_equation_solving(self, equation: sp.Eq, variable: sp.Symbol) -> List[MathStep]:
        """Generate step-by-step explanation for solving equations."""
        # This would implement algebraic step explanations
        # For now, placeholder
        return [MathStep(
            step_number=1,
            rule_name="Algebraic Solving",
            before_expression=str(equation),
            after_expression="Solution process",
            explanation="Solve the equation step by step"
        )]


class IntegrationStepExplainer:
    """Generates step-by-step explanations for integration."""
    
    def explain_integration(self, expr: sp.Expr, variable: sp.Symbol = None) -> List[MathStep]:
        """Generate detailed step-by-step explanation for integration."""
        # This would implement integration step explanations
        # For now, placeholder
        return [MathStep(
            step_number=1,
            rule_name="Integration",
            before_expression=str(expr),
            after_expression="Integration process",
            explanation="Integrate the expression step by step"
        )]


class MasterStepExplainer:
    """Main class that coordinates all step explanations."""
    
    def __init__(self):
        """Initialize all specialized explainers."""
        self.derivative_explainer = DerivativeStepExplainer()
        self.algebraic_explainer = AlgebraicStepExplainer()
        self.integration_explainer = IntegrationStepExplainer()
    
    def explain_derivative(self, expression_str: str, variable: str = 'x') -> List[MathStep]:
        """Explain derivative step-by-step."""
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(expression_str.replace('^', '**'), locals={variable: var})
            return self.derivative_explainer.explain_derivative(expr, var)
        except Exception as e:
            return [MathStep(
                step_number=1,
                rule_name="Error",
                before_expression=expression_str,
                after_expression="",
                explanation=f"Could not parse expression: {e}"
            )]
    
    def explain_integration(self, expression_str: str, variable: str = 'x') -> List[MathStep]:
        """Explain integration step-by-step."""
        try:
            var = sp.Symbol(variable)
            expr = sp.sympify(expression_str.replace('^', '**'), locals={variable: var})
            return self.integration_explainer.explain_integration(expr, var)
        except Exception as e:
            return [MathStep(
                step_number=1,
                rule_name="Error",
                before_expression=expression_str,
                after_expression="",
                explanation=f"Could not parse expression: {e}"
            )]
    
    def explain_algebraic_solving(self, equation_str: str, variable: str = 'x') -> List[MathStep]:
        """Explain algebraic equation solving step-by-step."""
        try:
            var = sp.Symbol(variable)
            # Parse equation (assumes format "left = right")
            left, right = equation_str.split('=')
            equation = sp.Eq(sp.sympify(left.strip()), sp.sympify(right.strip()))
            return self.algebraic_explainer.explain_equation_solving(equation, var)
        except Exception as e:
            return [MathStep(
                step_number=1,
                rule_name="Error",
                before_expression=equation_str,
                after_expression="",
                explanation=f"Could not parse equation: {e}"
            )]