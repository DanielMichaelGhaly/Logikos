#!/usr/bin/env python3
"""
AI Response Parser

Extracts mathematical solutions and explanations from AI model responses.
"""

import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass


@dataclass
class ParsedResponse:
    """Container for parsed AI response"""
    problem_type: str
    final_answer: str
    explanation: str
    steps: List[str]
    confidence: float
    metadata: Dict[str, Any] = None


class ResponseParser:
    """Parser for AI mathematical responses"""
    
    def __init__(self):
        # Patterns for extracting solutions
        self.solution_patterns = {
            'algebraic': [
                r'x\s*=\s*([-+]?\d*\.?\d+(?:/\d+)?)',  # x = number/fraction
                r'solution\s+is\s+x\s*=\s*([-+]?\d*\.?\d+(?:/\d+)?)',
                r'answer\s+is\s+x\s*=\s*([-+]?\d*\.?\d+(?:/\d+)?)',
                r'therefore,?\s+x\s*=\s*([-+]?\d*\.?\d+(?:/\d+)?)',
                r'final\s+answer:?\s+x\s*=\s*([-+]?\d*\.?\d+(?:/\d+)?)',
            ],
            'calculus': [
                r'derivative\s+is\s+([^.\n]+)',
                r"f'\\(x\\)\s*=\s*([^.\n]+)",
                r'result\s+is\s+([^.\n]+)',
                r'answer\s+is\s+([^.\n]+)',
            ],
            'general': [
                r'answer\s+is\s+([^.\n]+)',
                r'result\s+is\s+([^.\n]+)',
                r'solution\s+is\s+([^.\n]+)',
                r'therefore,?\s+([^.\n]+)',
            ]
        }
    
    def parse_response(self, response: str, problem_type: str = "general") -> ParsedResponse:
        """Parse AI response to extract solution and explanation"""
        
        # Extract final answer
        final_answer = self._extract_final_answer(response, problem_type)
        
        # Extract steps
        steps = self._extract_steps(response)
        
        # Calculate confidence based on response quality
        confidence = self._calculate_confidence(response, final_answer, steps)
        
        # Clean explanation (remove redundant parts)
        explanation = self._clean_explanation(response)
        
        return ParsedResponse(
            problem_type=problem_type,
            final_answer=final_answer or "Could not extract final answer",
            explanation=explanation,
            steps=steps,
            confidence=confidence,
            metadata={
                'response_length': len(response),
                'steps_count': len(steps),
                'has_final_answer': bool(final_answer)
            }
        )
    
    def _extract_final_answer(self, response: str, problem_type: str) -> Optional[str]:
        """Extract the final answer from the response"""
        
        # Try problem-type specific patterns first
        patterns = self.solution_patterns.get(problem_type, self.solution_patterns['general'])
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # Fallback: look at the end of the response for common answer formats
        final_section = response[-200:]  # Last 200 characters
        
        fallback_patterns = [
            r'x\s*=\s*([-+]?\d*\.?\d+(?:/\d+)?)',
            r'=\s*([-+]?\d*\.?\d+(?:/\d+)?)\s*$',
            r'answer:?\s*([^.\n]+)',
            r'solution:?\s*([^.\n]+)',
        ]
        
        for pattern in fallback_patterns:
            match = re.search(pattern, final_section, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_steps(self, response: str) -> List[str]:
        """Extract solution steps from the response"""
        steps = []
        
        # Look for numbered steps (1., 2., etc.)
        numbered_steps = re.findall(r'(\d+\.?\s+[^1-9\n]+)', response, re.MULTILINE)
        if numbered_steps:
            steps.extend([step.strip() for step in numbered_steps])
        
        # Look for bullet points or step indicators
        step_indicators = ['step', 'first', 'next', 'then', 'finally', 'therefore']
        
        sentences = re.split(r'[.!?]+', response)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Ignore very short sentences
                for indicator in step_indicators:
                    if indicator in sentence.lower() and sentence not in steps:
                        steps.append(sentence)
                        break
        
        # Remove duplicates while preserving order
        seen = set()
        unique_steps = []
        for step in steps:
            if step not in seen:
                seen.add(step)
                unique_steps.append(step)
        
        return unique_steps[:10]  # Limit to 10 steps max
    
    def _calculate_confidence(self, response: str, final_answer: Optional[str], steps: List[str]) -> float:
        """Calculate confidence score based on response quality"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence for having a final answer
        if final_answer:
            confidence += 0.2
        
        # Boost confidence for having multiple steps
        if len(steps) >= 2:
            confidence += 0.1
        if len(steps) >= 4:
            confidence += 0.1
        
        # Boost confidence for mathematical keywords
        math_keywords = ['equation', 'solve', 'substitute', 'simplify', 'isolate', 
                        'derivative', 'integral', 'calculate']
        keyword_count = sum(1 for keyword in math_keywords if keyword in response.lower())
        confidence += min(0.15, keyword_count * 0.03)
        
        # Reduce confidence for very short responses
        if len(response) < 100:
            confidence -= 0.2
        
        # Reduce confidence for errors or uncertainty indicators
        uncertainty_words = ['not sure', 'might be', 'possibly', 'unclear', 'error']
        if any(word in response.lower() for word in uncertainty_words):
            confidence -= 0.2
        
        # Ensure confidence is between 0 and 1
        return max(0.0, min(1.0, confidence))
    
    def _clean_explanation(self, response: str) -> str:
        """Clean up the explanation by removing redundant parts"""
        
        # Remove common prefixes
        prefixes_to_remove = [
            r"certainly[!.]?\s*",
            r"of course[!.]?\s*",
            r"let's solve this step by step[.]\s*",
            r"i'll help you solve this[.]\s*",
        ]
        
        cleaned = response
        for prefix in prefixes_to_remove:
            cleaned = re.sub(prefix, "", cleaned, flags=re.IGNORECASE)
        
        # Remove excessive whitespace and newlines
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # Multiple newlines -> double newline
        cleaned = re.sub(r' +', ' ', cleaned)  # Multiple spaces -> single space
        
        return cleaned.strip()


def test_response_parser():
    """Test the response parser with sample AI responses"""
    parser = ResponseParser()
    
    # Test responses
    test_cases = [
        (
            "algebraic",
            """Certainly! Let's solve the equation 2x + 5 = 0 step-by-step.

Step 1: Isolate the Variable Term
First, we need to isolate the term that contains the variable x. To do this, we subtract 5 from both sides of the equation. This will move the constant on the left-hand side over to the right-hand side:
2x + 5 - 5 = 0 - 5

Simplifying this gives us:
2x = -5

Step 2: Solve for x
Now that we have isolated the term with x, our goal is to find out what x equals. To do this, divide both sides of the equation by 2:
2x/2 = -5/2

Simplifying this gives us:
x = -5/2

Final Answer
So, the solution to the equation 2x + 5 = 0 is:
x = -5/2 or in decimal form, -2.5."""
        ),
        (
            "calculus",
            """The derivative of sin(x) is cos(x).

This is one of the fundamental derivatives in calculus. Using the definition of derivative and limit processes, we can show that d/dx[sin(x)] = cos(x).

Therefore, f'(x) = cos(x)."""
        )
    ]
    
    print("🧪 Testing Response Parser")
    print("=" * 50)
    
    for i, (prob_type, response) in enumerate(test_cases, 1):
        print(f"\n{i}. Problem type: {prob_type}")
        print(f"   Response length: {len(response)} characters")
        
        parsed = parser.parse_response(response, prob_type)
        
        print(f"   ✅ Final Answer: {parsed.final_answer}")
        print(f"   📝 Steps found: {len(parsed.steps)}")
        print(f"   🎯 Confidence: {parsed.confidence:.2f}")
        print(f"   📊 Explanation length: {len(parsed.explanation)} chars")


if __name__ == "__main__":
    test_response_parser()