"""
Sanity checks for mathematical problems with unit handling and expression validation.
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
import sympy as sp
import pint
from .schemas import MathProblem, Variable, Equation


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


class MathValidator:
    """Comprehensive validator for mathematical problems."""

    def __init__(self):
        """Initialize the validator with unit registry and validation rules."""
        self.ureg = pint.UnitRegistry()
        self.validation_errors: List[str] = []
        
        # Common mathematical functions and constants
        self.allowed_functions = {
            'sin', 'cos', 'tan', 'asin', 'acos', 'atan',
            'sinh', 'cosh', 'tanh', 'exp', 'log', 'ln',
            'sqrt', 'abs', 'floor', 'ceil', 'round',
            'factorial', 'gamma', 'pi', 'e'
        }
        
        # Domain validation patterns
        self.domain_patterns = {
            'real': r'^real|R|ℝ$',
            'integer': r'^integer|int|Z|ℤ$',
            'natural': r'^natural|N|ℕ$',
            'positive': r'^positive|pos|\+$',
            'negative': r'^negative|neg|\-$',
            'complex': r'^complex|C|ℂ$'
        }

    def validate(self, problem: MathProblem) -> bool:
        """Validate a mathematical problem comprehensively."""
        self.validation_errors = []  # Reset errors
        
        try:
            # Basic problem structure validation
            self._validate_problem_structure(problem)
            
            # Variable validation
            self._validate_variables(problem.variables)
            
            # Equation validation
            self._validate_equations(problem.equations)
            
            # Constraint validation
            self._validate_constraints(problem.constraints)
            
            # Problem type consistency
            self._validate_problem_type_consistency(problem)
            
            # Goal validation
            self._validate_goal(problem)
            
            # If we have errors, validation failed
            if self.validation_errors:
                raise ValidationError("\n".join(self.validation_errors))
            
            return True
            
        except Exception as e:
            if isinstance(e, ValidationError):
                raise
            else:
                raise ValidationError(f"Validation failed with error: {str(e)}")

    def get_validation_errors(self) -> List[str]:
        """Get list of validation errors from last validation."""
        return self.validation_errors.copy()

    def _validate_problem_structure(self, problem: MathProblem) -> None:
        """Validate basic problem structure."""
        if not problem.problem_text or not problem.problem_text.strip():
            self.validation_errors.append("Problem text cannot be empty")
        
        if not problem.problem_type or not problem.problem_type.strip():
            self.validation_errors.append("Problem type must be specified")
        
        valid_problem_types = {'algebraic', 'calculus', 'optimization', 'general'}
        if problem.problem_type not in valid_problem_types:
            self.validation_errors.append(
                f"Invalid problem type '{problem.problem_type}'. "
                f"Must be one of: {', '.join(valid_problem_types)}"
            )

    def _validate_variables(self, variables: List[Variable]) -> None:
        """Validate variable definitions and constraints."""
        variable_names = set()
        
        for var in variables:
            # Check for duplicate variable names
            if var.name in variable_names:
                self.validation_errors.append(f"Duplicate variable name: {var.name}")
            variable_names.add(var.name)
            
            # Validate variable name format
            if not self._is_valid_variable_name(var.name):
                self.validation_errors.append(
                    f"Invalid variable name '{var.name}'. "
                    "Variable names must be valid mathematical identifiers."
                )
            
            # Validate domain
            if var.domain and not self._is_valid_domain(var.domain):
                self.validation_errors.append(
                    f"Invalid domain '{var.domain}' for variable '{var.name}'"
                )
            
            # Validate constraints
            for constraint in var.constraints:
                if not self._validate_constraint_expression(constraint, var.name):
                    self.validation_errors.append(
                        f"Invalid constraint '{constraint}' for variable '{var.name}'"
                    )

    def _validate_equations(self, equations: List[Equation]) -> None:
        """Validate equation expressions and syntax."""
        for i, eq in enumerate(equations):
            # Check that both sides are present
            if not eq.left_side.strip():
                self.validation_errors.append(f"Equation {i+1}: Left side cannot be empty")
            
            if not eq.right_side.strip():
                self.validation_errors.append(f"Equation {i+1}: Right side cannot be empty")
            
            # Validate expression syntax
            if not self._validate_expression(eq.left_side):
                self.validation_errors.append(
                    f"Equation {i+1}: Invalid left side expression '{eq.left_side}'"
                )
            
            if not self._validate_expression(eq.right_side):
                self.validation_errors.append(
                    f"Equation {i+1}: Invalid right side expression '{eq.right_side}'"
                )
            
            # Check for dimensional consistency if units are present
            self._validate_dimensional_consistency(eq.left_side, eq.right_side)

    def _validate_constraints(self, constraints: List[str]) -> None:
        """Validate constraint expressions."""
        for i, constraint in enumerate(constraints):
            if not self._validate_constraint_expression(constraint):
                self.validation_errors.append(
                    f"Invalid constraint {i+1}: '{constraint}'"
                )

    def _validate_problem_type_consistency(self, problem: MathProblem) -> None:
        """Validate that problem content matches declared problem type."""
        problem_text = problem.problem_text.lower()
        
        if problem.problem_type == "calculus":
            calculus_keywords = ['derivative', 'differentiate', 'integral', 'integrate', 'd/dx', 'limit']
            if not any(keyword in problem_text for keyword in calculus_keywords):
                self.validation_errors.append(
                    "Problem type 'calculus' but no calculus keywords found in problem text"
                )
        
        elif problem.problem_type == "algebraic":
            if not problem.equations and "=" not in problem_text:
                self.validation_errors.append(
                    "Problem type 'algebraic' but no equations found"
                )
        
        elif problem.problem_type == "optimization":
            opt_keywords = ['maximize', 'minimize', 'optimal', 'minimum', 'maximum']
            if not any(keyword in problem_text for keyword in opt_keywords):
                self.validation_errors.append(
                    "Problem type 'optimization' but no optimization keywords found"
                )

    def _validate_goal(self, problem: MathProblem) -> None:
        """Validate problem goal specification."""
        if not problem.goal or not problem.goal.strip():
            self.validation_errors.append("Problem goal must be specified")

    def _is_valid_variable_name(self, name: str) -> bool:
        """Check if variable name is valid mathematical identifier."""
        # Variable names should be letters, possibly with subscripts/superscripts
        pattern = r'^[a-zA-Z][a-zA-Z0-9_]*$|^[a-zA-Z]$|^\\[a-zA-Z]+$'
        return bool(re.match(pattern, name))

    def _is_valid_domain(self, domain: str) -> bool:
        """Check if domain specification is valid."""
        domain_lower = domain.lower()
        return any(re.match(pattern, domain_lower) for pattern in self.domain_patterns.values())

    def _validate_expression(self, expr: str) -> bool:
        """Validate mathematical expression syntax."""
        if not expr.strip():
            return False
        
        try:
            # Try to parse with SymPy
            # Remove units first for parsing
            expr_no_units = self._extract_expression_without_units(expr)
            sp.sympify(expr_no_units)
            return True
        except:
            # If SymPy fails, do basic validation
            return self._basic_expression_validation(expr)

    def _basic_expression_validation(self, expr: str) -> bool:
        """Basic validation for expressions that SymPy can't parse."""
        # Check for balanced parentheses
        if not self._has_balanced_parentheses(expr):
            return False
        
        # Check for invalid characters
        allowed_chars = set('abcdefghijklmnopqrstuvwxyz'
                           'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                           '0123456789'
                           '+-*/^()[]{}=<>!.,_ ')
        if not all(c in allowed_chars for c in expr):
            return False
        
        return True

    def _has_balanced_parentheses(self, expr: str) -> bool:
        """Check if parentheses are balanced in expression."""
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        
        for char in expr:
            if char in pairs:
                stack.append(char)
            elif char in pairs.values():
                if not stack:
                    return False
                if pairs[stack.pop()] != char:
                    return False
        
        return len(stack) == 0

    def _extract_expression_without_units(self, expr: str) -> str:
        """Extract mathematical expression by removing unit specifications."""
        # Simple approach: remove common unit patterns
        # This could be made more sophisticated
        unit_patterns = [
            r'\s*(m|kg|s|A|K|mol|cd)\b',  # SI base units
            r'\s*(mm|cm|km|g|mg|min|hr|°C|°F)\b',  # Common derived units
            r'\s*units?\s*\([^)]+\)',  # unit(something) patterns
        ]
        
        result = expr
        for pattern in unit_patterns:
            result = re.sub(pattern, '', result, flags=re.IGNORECASE)
        
        return result.strip()

    def _validate_dimensional_consistency(self, left_expr: str, right_expr: str) -> None:
        """Check dimensional consistency between equation sides."""
        try:
            # Extract units from both sides
            left_units = self._extract_units(left_expr)
            right_units = self._extract_units(right_expr)
            
            # If both sides have units, check compatibility
            if left_units and right_units:
                try:
                    left_quantity = self.ureg.Quantity(1, left_units)
                    right_quantity = self.ureg.Quantity(1, right_units)
                    
                    # Try to convert - if they're dimensionally inconsistent, this will fail
                    left_quantity.to(right_quantity.units)
                    
                except Exception:
                    self.validation_errors.append(
                        f"Dimensional inconsistency: '{left_units}' cannot be converted to '{right_units}'"
                    )
        
        except Exception:
            # If unit extraction fails, skip dimensional analysis
            pass

    def _extract_units(self, expr: str) -> Optional[str]:
        """Extract unit specification from expression."""
        # Look for common unit patterns
        unit_patterns = [
            r'\b(m|kg|s|A|K|mol|cd|mm|cm|km|g|mg|min|hr|°C|°F)\b',
            r'units?\s*\(([^)]+)\)',
        ]
        
        for pattern in unit_patterns:
            match = re.search(pattern, expr, re.IGNORECASE)
            if match:
                return match.group(1) if len(match.groups()) > 0 else match.group(0)
        
        return None

    def _validate_constraint_expression(self, constraint: str, variable: str = None) -> bool:
        """Validate constraint expressions like 'x > 0', 'x ∈ [0, 1]'."""
        if not constraint.strip():
            return False
        
        # Common constraint patterns
        constraint_patterns = [
            r'^[a-zA-Z_][a-zA-Z0-9_]*\s*[<>=!]+\s*[\d\-+.eE]+$',  # x > 0, x = 5, etc.
            r'^[a-zA-Z_][a-zA-Z0-9_]*\s*∈\s*\[[\d\-+.eE,\s]+\]$',  # x ∈ [0, 1]
            r'^[a-zA-Z_][a-zA-Z0-9_]*\s*(>|<|>=|<=)\s*[\d\-+.eE]+$',  # x >= 0
        ]
        
        constraint_lower = constraint.lower().strip()
        
        # Check against patterns
        for pattern in constraint_patterns:
            if re.match(pattern, constraint):
                return True
        
        # Check for mathematical validity
        try:
            # Try to parse as SymPy expression (for complex constraints)
            if variable:
                locals_dict = {variable: sp.Symbol(variable)}
            else:
                locals_dict = {}
            
            # Handle common constraint formats
            if '∈' in constraint or 'in' in constraint_lower:
                return True  # Set membership - accept for now
            
            # Try to parse as inequality
            for op in ['<=', '>=', '<', '>', '=', '!=']:
                if op in constraint:
                    parts = constraint.split(op)
                    if len(parts) == 2:
                        try:
                            sp.sympify(parts[0].strip(), locals=locals_dict)
                            sp.sympify(parts[1].strip(), locals=locals_dict)
                            return True
                        except:
                            continue
            
            return False
            
        except:
            return False
