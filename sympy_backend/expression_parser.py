#!/usr/bin/env python3
"""
Enhanced Mathematical Expression Parser

Handles natural language mathematical input and converts it to SymPy-compatible expressions.
Supports various formats like:
- "solve 2x+5=0"  
- "find roots of x^2-4"
- "derivative of sin(x)"
- "integrate cos(x)"
"""

import re
import sympy as sp
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class ParsedExpression:
    """Container for parsed mathematical expressions"""
    original_text: str
    problem_type: str  # 'equation', 'expression', 'calculus', etc.
    expressions: List[sp.Expr]  # SymPy expressions
    variables: List[sp.Symbol]  # Variables found
    operation: str  # 'solve', 'differentiate', 'integrate', 'simplify'
    target_variable: Optional[str] = None
    metadata: Dict[str, Any] = None


class EnhancedMathParser:
    """Enhanced parser for natural language mathematical expressions"""
    
    def __init__(self):
        # Precompile regex patterns for better performance
        self.patterns = {
            # Equation solving patterns
            'solve_equation': re.compile(r'solve\s+(.+?)\s*=\s*(.+?)(?:\s+for\s+([a-zA-Z]))?$', re.IGNORECASE),
            'solve_for': re.compile(r'solve\s+for\s+([a-zA-Z])\s*:\s*(.+?)\s*=\s*(.+?)$', re.IGNORECASE),
            
            # Root finding patterns  
            'find_roots': re.compile(r'(?:find\s+)?(?:the\s+)?roots?\s+of\s+(.+?)$', re.IGNORECASE),
            'find_zeros': re.compile(r'(?:find\s+)?(?:the\s+)?zeros?\s+of\s+(.+?)$', re.IGNORECASE),
            
            # Calculus patterns
            'derivative': re.compile(r'(?:derivative|differentiate)\s+(?:of\s+)?(.+?)(?:\s+(?:with\s+respect\s+to|w\.?r\.?t\.?)\s+([a-zA-Z]))?$', re.IGNORECASE),
            'integral': re.compile(r'(?:integral|integrate)\s+(?:of\s+)?(.+?)(?:\s+(?:with\s+respect\s+to|w\.?r\.?t\.?)\s+([a-zA-Z]))?$', re.IGNORECASE),
            
            # General patterns
            'simplify': re.compile(r'(?:simplify|expand|factor)\s+(.+?)$', re.IGNORECASE),
            'evaluate': re.compile(r'(?:evaluate|compute)\s+(.+?)$', re.IGNORECASE),
        }
        
        # Function name mappings for SymPy
        self.function_mappings = {
            'sin': sp.sin, 'cos': sp.cos, 'tan': sp.tan,
            'sec': sp.sec, 'csc': sp.csc, 'cot': sp.cot,
            'asin': sp.asin, 'acos': sp.acos, 'atan': sp.atan,
            'sinh': sp.sinh, 'cosh': sp.cosh, 'tanh': sp.tanh,
            'ln': sp.log, 'log': sp.log, 'log10': lambda x: sp.log(x, 10),
            'exp': sp.exp, 'sqrt': sp.sqrt, 'abs': sp.Abs,
            'pi': sp.pi, 'e': sp.E,
        }
    
    def parse(self, text: str) -> ParsedExpression:
        """Parse natural language mathematical text into structured format"""
        text = text.strip()
        
        # Try different parsing patterns
        for pattern_name, pattern in self.patterns.items():
            match = pattern.match(text)
            if match:
                return self._parse_with_pattern(text, pattern_name, match)
        
        # Fallback: try to parse as general expression
        return self._parse_general_expression(text)
    
    def _parse_with_pattern(self, text: str, pattern_name: str, match: re.Match) -> ParsedExpression:
        """Parse expression using a specific pattern"""
        
        if pattern_name == 'solve_equation':
            left_expr = match.group(1).strip()
            right_expr = match.group(2).strip()
            target_var = match.group(3) if match.group(3) else None
            
            # Process expressions
            left_sympy = self._text_to_sympy(left_expr)
            right_sympy = self._text_to_sympy(right_expr)
            equation = sp.Eq(left_sympy, right_sympy)
            
            variables = list(equation.free_symbols)
            
            return ParsedExpression(
                original_text=text,
                problem_type='equation',
                expressions=[equation],
                variables=variables,
                operation='solve',
                target_variable=target_var,
                metadata={'left': str(left_sympy), 'right': str(right_sympy)}
            )
        
        elif pattern_name == 'solve_for':
            target_var = match.group(1)
            left_expr = match.group(2).strip()
            right_expr = match.group(3).strip()
            
            left_sympy = self._text_to_sympy(left_expr)
            right_sympy = self._text_to_sympy(right_expr)
            equation = sp.Eq(left_sympy, right_sympy)
            
            variables = list(equation.free_symbols)
            
            return ParsedExpression(
                original_text=text,
                problem_type='equation',
                expressions=[equation],
                variables=variables,
                operation='solve',
                target_variable=target_var,
                metadata={'left': str(left_sympy), 'right': str(right_sympy)}
            )
        
        elif pattern_name in ['find_roots', 'find_zeros']:
            expr_text = match.group(1).strip()
            expr = self._text_to_sympy(expr_text)
            
            # Convert to equation = 0
            equation = sp.Eq(expr, 0)
            variables = list(expr.free_symbols)
            
            return ParsedExpression(
                original_text=text,
                problem_type='equation',
                expressions=[equation],
                variables=variables,
                operation='solve',
                metadata={'expression': str(expr)}
            )
        
        elif pattern_name == 'derivative':
            expr_text = match.group(1).strip()
            var_name = match.group(2) if match.group(2) else 'x'
            
            expr = self._text_to_sympy(expr_text)
            var = sp.Symbol(var_name)
            variables = [var] if var not in expr.free_symbols else list(expr.free_symbols)
            
            return ParsedExpression(
                original_text=text,
                problem_type='calculus',
                expressions=[expr],
                variables=variables,
                operation='differentiate',
                target_variable=var_name,
                metadata={'expression': str(expr), 'variable': var_name}
            )
        
        elif pattern_name == 'integral':
            expr_text = match.group(1).strip()
            var_name = match.group(2) if match.group(2) else 'x'
            
            expr = self._text_to_sympy(expr_text)
            var = sp.Symbol(var_name)
            variables = [var] if var not in expr.free_symbols else list(expr.free_symbols)
            
            return ParsedExpression(
                original_text=text,
                problem_type='calculus',
                expressions=[expr],
                variables=variables,
                operation='integrate',
                target_variable=var_name,
                metadata={'expression': str(expr), 'variable': var_name}
            )
        
        elif pattern_name in ['simplify', 'evaluate']:
            expr_text = match.group(1).strip()
            expr = self._text_to_sympy(expr_text)
            variables = list(expr.free_symbols)
            
            return ParsedExpression(
                original_text=text,
                problem_type='expression',
                expressions=[expr],
                variables=variables,
                operation=pattern_name,
                metadata={'expression': str(expr)}
            )
        
        # Should not reach here, but fallback
        return self._parse_general_expression(text)
    
    def _parse_general_expression(self, text: str) -> ParsedExpression:
        """Parse as a general mathematical expression"""
        try:
            expr = self._text_to_sympy(text)
            variables = list(expr.free_symbols)
            
            return ParsedExpression(
                original_text=text,
                problem_type='expression',
                expressions=[expr],
                variables=variables,
                operation='evaluate',
                metadata={'expression': str(expr)}
            )
        except Exception as e:
            # If all else fails, create an error result
            return ParsedExpression(
                original_text=text,
                problem_type='error',
                expressions=[],
                variables=[],
                operation='none',
                metadata={'error': str(e)}
            )
    
    def _text_to_sympy(self, text: str) -> sp.Expr:
        """Convert text mathematical expression to SymPy expression"""
        # Clean up the text
        text = text.strip()
        
        # Apply preprocessing transformations
        text = self._preprocess_mathematical_text(text)
        
        # Parse with SymPy
        try:
            # Create a local namespace with mathematical functions
            local_dict = self.function_mappings.copy()
            
            # Add common variables
            for var in 'abcdefghijklmnopqrstuvwxyz':
                local_dict[var] = sp.Symbol(var)
            
            return sp.sympify(text, locals=local_dict)
            
        except Exception as e:
            raise ValueError(f"Could not parse mathematical expression '{text}': {e}")
    
    def _preprocess_mathematical_text(self, text: str) -> str:
        """Preprocess text to handle common mathematical notation"""
        
        # Handle exponentiation: x^2 → x**2
        text = re.sub(r'([a-zA-Z0-9\)])(\^)', r'\1**', text)
        
        # Handle implicit multiplication: 2x → 2*x (but avoid breaking function names)
        # First, protect common function names
        protected_functions = ['sin', 'cos', 'tan', 'sec', 'csc', 'cot', 'asin', 'acos', 'atan', 
                              'sinh', 'cosh', 'tanh', 'ln', 'log', 'exp', 'sqrt', 'abs']
        
        # Replace digit followed by letter (but not function names)
        for func in protected_functions:
            # Temporarily replace function names with placeholders
            text = text.replace(func, f'FUNC_{func.upper()}')
        
        # Now handle implicit multiplication: 2x → 2*x
        text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
        
        # Handle coefficient-variable multiplication: 2(x+1) → 2*(x+1)
        text = re.sub(r'(\d)\(', r'\1*(', text)
        
        # Handle closing paren followed by variable: (x+1)y → (x+1)*y
        text = re.sub(r'\)([a-zA-Z])', r')*\1', text)
        
        # Restore protected function names
        for func in protected_functions:
            text = text.replace(f'FUNC_{func.upper()}', func)
        
        # Handle sqrt notation: sqrt x → sqrt(x)
        text = re.sub(r'sqrt\s+([a-zA-Z0-9]+)(?![a-zA-Z0-9\(])', r'sqrt(\1)', text)
        
        # Handle pi and e
        text = re.sub(r'\bpi\b', 'pi', text)
        text = re.sub(r'\be\b(?![a-zA-Z])', 'e', text)  # e but not exp or other words
        
        return text


def test_parser():
    """Test the enhanced parser with various inputs"""
    parser = EnhancedMathParser()
    
    test_cases = [
        "solve 2x+5=0",
        "solve 2x + 5 = 0 for x", 
        "solve for x: 2x + 5 = 0",
        "find roots of x^2 - 4",
        "find the zeros of x^2 + 2x + 1",
        "derivative of sin(x)",
        "differentiate x^2 + 3x",
        "derivative of cos(x) with respect to x",
        "integral of x^2",
        "integrate sin(x) w.r.t. x",
        "simplify (x^2 - 1)/(x - 1)",
        "evaluate 2^3 + 4*5",
    ]
    
    print("🧪 Testing Enhanced Math Parser")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. Input: '{case}'")
        try:
            result = parser.parse(case)
            print(f"   Type: {result.problem_type}")
            print(f"   Operation: {result.operation}")
            if result.expressions:
                print(f"   Expressions: {[str(expr) for expr in result.expressions]}")
            print(f"   Variables: {[str(var) for var in result.variables]}")
            if result.target_variable:
                print(f"   Target Variable: {result.target_variable}")
        except Exception as e:
            print(f"   ❌ Error: {e}")


if __name__ == "__main__":
    test_parser()