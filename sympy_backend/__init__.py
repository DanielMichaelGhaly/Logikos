"""
SymPy Backend Module

Handles mathematical verification and symbolic computation.
"""

from .expression_parser import EnhancedMathParser, ParsedExpression
from .solver import SymPySolver
from .verifier import SolutionVerifier

__all__ = ['EnhancedMathParser', 'ParsedExpression', 'SymPySolver', 'SolutionVerifier']