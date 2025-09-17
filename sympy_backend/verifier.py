#!/usr/bin/env python3
"""
Solution Verifier

Compares AI solutions with SymPy results to ensure mathematical accuracy.
"""

import re
import sympy as sp
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from .solver import SolutionResult


class VerificationStatus(Enum):
    """Status of verification"""
    MATCH = "match"
    MISMATCH = "mismatch"
    PARTIAL_MATCH = "partial_match"
    AI_ERROR = "ai_error"
    SYMPY_ERROR = "sympy_error"
    INCONCLUSIVE = "inconclusive"


@dataclass
class VerificationResult:
    """Result of verifying an AI solution against SymPy"""
    status: VerificationStatus
    ai_solution: str
    sympy_solution: Any
    explanation: str
    confidence: float  # 0.0 to 1.0
    details: Dict[str, Any] = None


class SolutionVerifier:
    """Verifies AI solutions against SymPy computations"""
    
    def __init__(self):
        self.tolerance = 1e-10  # Numerical comparison tolerance
    
    def verify(self, ai_response: str, sympy_result: SolutionResult) -> VerificationResult:
        """Verify AI response against SymPy result"""
        
        if not sympy_result.success:
            return VerificationResult(
                status=VerificationStatus.SYMPY_ERROR,
                ai_solution=ai_response,
                sympy_solution=None,
                explanation=f"SymPy failed to solve: {sympy_result.error_message}",
                confidence=0.0,
                details={'sympy_error': sympy_result.error_message}
            )
        
        # Extract solution from AI response
        ai_solution = self._extract_ai_solution(ai_response, sympy_result.problem_type, sympy_result.operation)
        
        if not ai_solution:
            return VerificationResult(
                status=VerificationStatus.AI_ERROR,
                ai_solution=ai_response,
                sympy_solution=sympy_result.result,
                explanation="Could not extract mathematical solution from AI response",
                confidence=0.0,
                details={'extraction_failed': True}
            )
        
        # Compare based on problem type
        if sympy_result.operation == 'solve':
            return self._verify_equation_solution(ai_solution, sympy_result)
        elif sympy_result.operation == 'differentiate':
            return self._verify_derivative(ai_solution, sympy_result)
        elif sympy_result.operation == 'integrate':
            return self._verify_integral(ai_solution, sympy_result)
        elif sympy_result.operation in ['simplify', 'evaluate']:
            return self._verify_expression(ai_solution, sympy_result)
        else:
            return VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                ai_solution=ai_solution,
                sympy_solution=sympy_result.result,
                explanation=f"Verification not implemented for operation: {sympy_result.operation}",
                confidence=0.0
            )
    
    def _extract_ai_solution(self, ai_response: str, problem_type: str, operation: str) -> Optional[str]:
        """Extract the mathematical solution from AI response text"""
        
        # Common patterns for different types of solutions
        patterns = {
            'solve': [
                r'x\s*=\s*([-+]?\d*\.?\d+(?:/\d+)?)',  # x = number or fraction
                r'x\s*=\s*[-+]?\s*\d*\.?\d*\s*/\s*\d+',  # x = fraction with spaces
                r'x\s*=\s*([-+]?\d*\.?\d+)',  # x = decimal
                r'solution.*?x\s*=\s*([-+]?\d*\.?\d+(?:/\d+)?)',  # "solution is x = ..."
                r'answer.*?x\s*=\s*([-+]?\d*\.?\d+(?:/\d+)?)',   # "answer is x = ..."
            ],
            'differentiate': [
                r"derivative.*?(?:is|=).*?([+-]?\d*\.?\d*\*?[a-zA-Z]?[+-]?\d*\.?\d*)",
                r"f'.*?=.*?([+-]?\d*\.?\d*\*?[a-zA-Z]?[+-]?\d*\.?\d*)",
                r"(?:result|answer).*?([+-]?\d*\.?\d*\*?[a-zA-Z]?[+-]?\d*\.?\d*)"
            ],
            'integrate': [
                r"integral.*?(?:is|=).*?([^.]+)",
                r"∫.*?=.*?([^.]+)",
                r"result.*?([^.]+\+\s*C)"
            ]
        }
        
        # Try operation-specific patterns first
        if operation in patterns:
            for pattern in patterns[operation]:
                match = re.search(pattern, ai_response, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        # Fallback: look for common mathematical expressions
        math_patterns = [
            r'[-+]?\d*\.?\d+/\d+',  # fractions
            r'[-+]?\d*\.?\d+',      # numbers
            r'[-+]?\d*\*?[a-zA-Z]+[-+]?\d*',  # simple algebraic expressions
        ]
        
        for pattern in math_patterns:
            matches = re.findall(pattern, ai_response)
            if matches:
                # Return the last match (often the final answer)
                return matches[-1].strip()
        
        return None
    
    def _verify_equation_solution(self, ai_solution: str, sympy_result: SolutionResult) -> VerificationResult:
        """Verify equation solutions"""
        try:
            # Convert AI solution to comparable format
            ai_value = self._parse_numeric_value(ai_solution)
            
            # Get SymPy solution
            sympy_solutions = sympy_result.result
            
            if isinstance(sympy_solutions, list):
                sympy_values = [self._sympy_to_float(sol) for sol in sympy_solutions]
            else:
                sympy_values = [self._sympy_to_float(sympy_solutions)]
            
            # Check if AI solution matches any SymPy solution
            for sympy_val in sympy_values:
                if sympy_val is not None and ai_value is not None:
                    if abs(ai_value - sympy_val) < self.tolerance:
                        return VerificationResult(
                            status=VerificationStatus.MATCH,
                            ai_solution=ai_solution,
                            sympy_solution=sympy_result.result,
                            explanation=f"AI solution {ai_solution} matches SymPy result {sympy_val}",
                            confidence=0.95,
                            details={
                                'ai_numeric': ai_value,
                                'sympy_numeric': sympy_val,
                                'tolerance': self.tolerance
                            }
                        )
            
            return VerificationResult(
                status=VerificationStatus.MISMATCH,
                ai_solution=ai_solution,
                sympy_solution=sympy_result.result,
                explanation=f"AI solution {ai_solution} does not match SymPy result {sympy_solutions}",
                confidence=0.9,
                details={
                    'ai_numeric': ai_value,
                    'sympy_numeric': sympy_values,
                    'tolerance': self.tolerance
                }
            )
            
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                ai_solution=ai_solution,
                sympy_solution=sympy_result.result,
                explanation=f"Error during verification: {str(e)}",
                confidence=0.0,
                details={'error': str(e)}
            )
    
    def _verify_derivative(self, ai_solution: str, sympy_result: SolutionResult) -> VerificationResult:
        """Verify derivative computations"""
        try:
            # For derivatives, we need to compare expressions, not just numbers
            # This is more complex and would require expression parsing
            
            # Simple check: if AI mentions key terms from the SymPy result
            sympy_str = str(sympy_result.result)
            
            # Check if AI solution contains key components
            if self._expressions_similar(ai_solution, sympy_str):
                return VerificationResult(
                    status=VerificationStatus.PARTIAL_MATCH,
                    ai_solution=ai_solution,
                    sympy_solution=sympy_result.result,
                    explanation="AI derivative appears to contain correct terms",
                    confidence=0.7,
                    details={'similarity_check': True}
                )
            else:
                return VerificationResult(
                    status=VerificationStatus.INCONCLUSIVE,
                    ai_solution=ai_solution,
                    sympy_solution=sympy_result.result,
                    explanation="Derivative verification requires more sophisticated expression comparison",
                    confidence=0.3,
                    details={'needs_expression_parsing': True}
                )
                
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                ai_solution=ai_solution,
                sympy_solution=sympy_result.result,
                explanation=f"Error during derivative verification: {str(e)}",
                confidence=0.0,
                details={'error': str(e)}
            )
    
    def _verify_integral(self, ai_solution: str, sympy_result: SolutionResult) -> VerificationResult:
        """Verify integral computations"""
        # Similar to derivative verification
        return VerificationResult(
            status=VerificationStatus.INCONCLUSIVE,
            ai_solution=ai_solution,
            sympy_solution=sympy_result.result,
            explanation="Integral verification requires expression comparison",
            confidence=0.3,
            details={'needs_expression_parsing': True}
        )
    
    def _verify_expression(self, ai_solution: str, sympy_result: SolutionResult) -> VerificationResult:
        """Verify expression simplification/evaluation"""
        try:
            ai_value = self._parse_numeric_value(ai_solution)
            sympy_value = self._sympy_to_float(sympy_result.result)
            
            if ai_value is not None and sympy_value is not None:
                if abs(ai_value - sympy_value) < self.tolerance:
                    return VerificationResult(
                        status=VerificationStatus.MATCH,
                        ai_solution=ai_solution,
                        sympy_solution=sympy_result.result,
                        explanation=f"AI result {ai_solution} matches SymPy result {sympy_value}",
                        confidence=0.9,
                        details={'ai_numeric': ai_value, 'sympy_numeric': sympy_value}
                    )
                else:
                    return VerificationResult(
                        status=VerificationStatus.MISMATCH,
                        ai_solution=ai_solution,
                        sympy_solution=sympy_result.result,
                        explanation=f"AI result {ai_solution} differs from SymPy result {sympy_value}",
                        confidence=0.8,
                        details={'ai_numeric': ai_value, 'sympy_numeric': sympy_value}
                    )
            else:
                return VerificationResult(
                    status=VerificationStatus.INCONCLUSIVE,
                    ai_solution=ai_solution,
                    sympy_solution=sympy_result.result,
                    explanation="Could not convert solutions to comparable numeric values",
                    confidence=0.2,
                    details={'ai_numeric': ai_value, 'sympy_numeric': sympy_value}
                )
                
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.INCONCLUSIVE,
                ai_solution=ai_solution,
                sympy_solution=sympy_result.result,
                explanation=f"Error during expression verification: {str(e)}",
                confidence=0.0,
                details={'error': str(e)}
            )
    
    def _parse_numeric_value(self, text: str) -> Optional[float]:
        """Parse numeric value from text, handling fractions and decimals"""
        try:
            # Handle fractions like -5/2
            if '/' in text:
                # Extract fraction pattern
                match = re.search(r'[-+]?\d+/\d+', text)
                if match:
                    fraction_str = match.group()
                    numerator, denominator = fraction_str.split('/')
                    return float(numerator) / float(denominator)
            
            # Handle decimals and integers
            match = re.search(r'[-+]?\d*\.?\d+', text)
            if match:
                return float(match.group())
            
            return None
        except:
            return None
    
    def _sympy_to_float(self, sympy_expr) -> Optional[float]:
        """Convert SymPy expression to float if possible"""
        try:
            if hasattr(sympy_expr, 'evalf'):
                return float(sympy_expr.evalf())
            else:
                return float(sympy_expr)
        except:
            return None
    
    def _expressions_similar(self, ai_expr: str, sympy_expr: str) -> bool:
        """Basic similarity check between expressions"""
        # Simple heuristic: check if they contain similar terms
        ai_terms = re.findall(r'[a-zA-Z0-9]+', ai_expr.lower())
        sympy_terms = re.findall(r'[a-zA-Z0-9]+', sympy_expr.lower())
        
        if not ai_terms or not sympy_terms:
            return False
        
        common_terms = set(ai_terms) & set(sympy_terms)
        similarity = len(common_terms) / max(len(set(ai_terms)), len(set(sympy_terms)))
        
        return similarity > 0.5


def test_verifier():
    """Test the solution verifier"""
    from .expression_parser import EnhancedMathParser
    from .solver import SymPySolver
    
    parser = EnhancedMathParser()
    solver = SymPySolver()
    verifier = SolutionVerifier()
    
    # Test cases: (problem, AI response)
    test_cases = [
        ("solve 2x+5=0", "The solution is x = -5/2 or -2.5"),
        ("solve 2x+5=0", "x = -3"),  # Wrong answer
        ("solve x^2-4=0", "The solutions are x = 2 and x = -2"),
    ]
    
    print("🧪 Testing Solution Verifier")
    print("=" * 50)
    
    for i, (problem, ai_response) in enumerate(test_cases, 1):
        print(f"\n{i}. Problem: '{problem}'")
        print(f"   AI Response: '{ai_response}'")
        
        # Parse and solve with SymPy
        parsed = parser.parse(problem)
        sympy_result = solver.solve(parsed)
        
        # Verify
        verification = verifier.verify(ai_response, sympy_result)
        
        print(f"   SymPy Result: {sympy_result.result}")
        print(f"   Status: {verification.status.value}")
        print(f"   Confidence: {verification.confidence:.2f}")
        print(f"   Explanation: {verification.explanation}")


if __name__ == "__main__":
    test_verifier()