"""
Confidence comparison system between SymPy ground truth and AI reasoning.
Only shows confidence for supported question types with parseable results.
"""

import re
from typing import Optional

from shared.schemas import (
    ConfidenceComparison,
    SympyResult,
    AIResult,
    QuestionType,
    QuestionClassification
)


class ConfidenceComparator:
    """Compares SymPy and AI results to determine confidence level."""

    def __init__(self):
        """Initialize the confidence comparator."""
        # Question types where we can meaningfully compare results
        self.supported_types = {
            QuestionType.SOLVE,
            QuestionType.FACTOR,
            QuestionType.SIMPLIFY,
            QuestionType.DERIVATIVE,
            QuestionType.INTEGRAL,
            QuestionType.ROOTS
        }

    def compare_results(
        self,
        classification: QuestionClassification,
        sympy_result: SympyResult,
        ai_result: AIResult
    ) -> ConfidenceComparison:
        """
        Compare SymPy and AI results and determine confidence.

        Args:
            classification: Original question classification
            sympy_result: SymPy ground truth result
            ai_result: AI reasoning result

        Returns:
            ConfidenceComparison with confidence level and details
        """
        # Check if this question type supports confidence comparison
        if classification.type not in self.supported_types:
            return ConfidenceComparison(
                confidence_level="no_comparison",
                comparison_details=f"Confidence comparison not supported for question type: {classification.type}",
                show_to_user=False
            )

        # Extract comparable results
        sympy_comparable = self._extract_sympy_result(sympy_result)
        ai_comparable = self._extract_ai_result(ai_result)

        # If we can't extract results from either, don't show comparison
        if not sympy_comparable or not ai_comparable:
            return ConfidenceComparison(
                sympy_result=sympy_comparable,
                ai_result=ai_comparable,
                confidence_level="no_comparison",
                comparison_details="Could not extract comparable results from one or both solvers",
                show_to_user=False
            )

        # Compare the results
        confidence_level, details = self._compare_mathematical_results(
            sympy_comparable,
            ai_comparable,
            classification.type
        )

        return ConfidenceComparison(
            sympy_result=sympy_comparable,
            ai_result=ai_comparable,
            confidence_level=confidence_level,
            comparison_details=details,
            show_to_user=confidence_level != "no_comparison"
        )

    def _extract_sympy_result(self, sympy_result: SympyResult) -> Optional[str]:
        """Extract comparable result from SymPy computation."""
        if not sympy_result.success or not sympy_result.result:
            return None

        result = sympy_result.result.strip()

        # Clean up SymPy formatting
        result = self._normalize_mathematical_expression(result)

        return result

    def _extract_ai_result(self, ai_result: AIResult) -> Optional[str]:
        """Extract comparable result from AI reasoning."""
        if not ai_result.success or not ai_result.mathematical_result:
            return None

        result = ai_result.mathematical_result.strip()

        # Remove common prefixes
        result = re.sub(r'^(Answer|Result|Solution|Final answer):\s*', '', result, flags=re.IGNORECASE)

        # Clean up AI formatting
        result = self._normalize_mathematical_expression(result)

        return result

    def _normalize_mathematical_expression(self, expr: str) -> str:
        """Normalize mathematical expressions for comparison."""
        # Remove extra whitespace
        expr = re.sub(r'\s+', ' ', expr).strip()

        # Normalize list formatting [1, 2] vs [1,2]
        expr = re.sub(r'\s*,\s*', ',', expr)

        # Normalize fractions: -5/2 vs -2.5
        # This is basic - more sophisticated normalization could be added

        # Sort lists for comparison if it looks like a list
        if expr.startswith('[') and expr.endswith(']'):
            try:
                # Extract numbers from list and sort
                numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', expr)
                if numbers:
                    # Simple sort by string representation
                    sorted_nums = sorted(numbers)
                    expr = '[' + ', '.join(sorted_nums) + ']'
            except:
                pass  # Keep original if sorting fails

        return expr

    def _compare_mathematical_results(
        self,
        sympy_result: str,
        ai_result: str,
        question_type: QuestionType
    ) -> tuple:
        """
        Compare mathematical results and determine confidence.

        Returns:
            tuple of (confidence_level, details)
        """
        # Exact match
        if sympy_result == ai_result:
            return "high", f"Exact match: {sympy_result}"

        # Numerical equivalence checks
        confidence, details = self._check_numerical_equivalence(sympy_result, ai_result)
        if confidence != "no_comparison":
            return confidence, details

        # Algebraic equivalence checks (basic)
        confidence, details = self._check_algebraic_equivalence(sympy_result, ai_result, question_type)
        if confidence != "no_comparison":
            return confidence, details

        # No match found
        return "low", f"Results differ: SymPy='{sympy_result}' vs AI='{ai_result}'"

    def _check_numerical_equivalence(self, sympy_result: str, ai_result: str) -> tuple:
        """Check if results are numerically equivalent."""
        try:
            # Handle lists of numbers
            if sympy_result.startswith('[') and ai_result.startswith('['):
                sympy_nums = self._extract_numbers_from_list(sympy_result)
                ai_nums = self._extract_numbers_from_list(ai_result)

                if len(sympy_nums) == len(ai_nums):
                    # Compare each number with tolerance
                    all_close = True
                    for s_num, a_num in zip(sorted(sympy_nums), sorted(ai_nums)):
                        if abs(s_num - a_num) > 0.0001:
                            all_close = False
                            break

                    if all_close:
                        return "high", f"Numerically equivalent lists: {sympy_result} ≈ {ai_result}"
                    else:
                        return "medium", f"Lists have same size but different values: {sympy_result} vs {ai_result}"

            # Handle single numerical values
            sympy_num = self._extract_single_number(sympy_result)
            ai_num = self._extract_single_number(ai_result)

            if sympy_num is not None and ai_num is not None:
                if abs(sympy_num - ai_num) < 0.0001:
                    return "high", f"Numerically equivalent: {sympy_result} ≈ {ai_result}"
                elif abs(sympy_num - ai_num) < 0.01:
                    return "medium", f"Close numerically: {sympy_result} ≈ {ai_result}"

        except Exception as e:
            pass  # Fall through to no comparison

        return "no_comparison", ""

    def _check_algebraic_equivalence(self, sympy_result: str, ai_result: str, question_type: QuestionType) -> tuple:
        """Check for basic algebraic equivalence."""
        # For factoring, check if both results are factored forms
        if question_type == QuestionType.FACTOR:
            if ('(' in sympy_result and ')' in sympy_result and
                '(' in ai_result and ')' in ai_result):
                # Both look like factored forms - could do more sophisticated checking
                return "medium", f"Both are factored forms: {sympy_result} vs {ai_result}"

        # For equations, check if solutions are equivalent in different forms
        if question_type == QuestionType.SOLVE:
            # Handle cases like x = -5/2 vs x = -2.5
            sympy_clean = re.sub(r'[x=\s]', '', sympy_result)
            ai_clean = re.sub(r'[x=\s]', '', ai_result)

            try:
                # Try to evaluate as fractions/decimals
                if '/' in sympy_clean:
                    sympy_val = eval(sympy_clean)  # Simple evaluation for fractions
                    if abs(float(ai_clean) - sympy_val) < 0.0001:
                        return "high", f"Equivalent forms: {sympy_result} = {ai_result}"
            except:
                pass

        return "no_comparison", ""

    def _extract_numbers_from_list(self, list_str: str) -> list:
        """Extract numerical values from a list string like '[1, -2, 3]'."""
        numbers = []
        # Find all number patterns including fractions
        matches = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', list_str)

        for match in matches:
            try:
                if '/' in match:
                    # Handle fractions
                    num, denom = match.split('/')
                    numbers.append(float(num) / float(denom))
                else:
                    numbers.append(float(match))
            except:
                continue

        return numbers

    def _extract_single_number(self, result_str: str) -> Optional[float]:
        """Extract a single numerical value from result string."""
        # Remove common prefixes and equals signs
        cleaned = re.sub(r'^[x=\s]*', '', result_str)

        try:
            # Handle fractions
            if '/' in cleaned and len(cleaned.split('/')) == 2:
                num, denom = cleaned.split('/')
                return float(num) / float(denom)

            # Handle regular numbers
            match = re.search(r'-?\d+(?:\.\d+)?', cleaned)
            if match:
                return float(match.group(0))

        except:
            pass

        return None