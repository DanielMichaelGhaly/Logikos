"""
AI-Powered Natural Language Math Parser

Uses AI models to intelligently parse natural language math problems into
structured schemas, handling all kinds of natural language variations.
"""

import json
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from .schemas import Equation, MathProblem, Variable

# Try to import OpenAI - graceful fallback if not available
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available - using fallback parser")

# Try to import other AI providers as fallbacks
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


@dataclass
class ParseResult:
    """Result from AI parsing with confidence score."""
    success: bool
    problem: Optional[MathProblem]
    confidence: float
    raw_response: str
    error: Optional[str] = None


class AIMathParser:
    """AI-powered parser for natural language math problems."""
    
    def __init__(self, preferred_provider: str = "auto"):
        """
        Initialize the AI parser.
        
        Args:
            preferred_provider: "openai", "anthropic", "ollama", or "auto"
        """
        self.preferred_provider = preferred_provider
        self.providers_available = {
            "openai": OPENAI_AVAILABLE,
            "anthropic": ANTHROPIC_AVAILABLE,
            "ollama": OLLAMA_AVAILABLE
        }
        
        # Initialize clients
        self.openai_client = None
        self.anthropic_client = None
        
        if OPENAI_AVAILABLE:
            try:
                self.openai_client = OpenAI()
            except:
                print("OpenAI client initialization failed")
        
        if ANTHROPIC_AVAILABLE:
            try:
                self.anthropic_client = anthropic.Anthropic()
            except:
                print("Anthropic client initialization failed")
    
    def parse(self, problem_text: str) -> ParseResult:
        """Parse natural language math problem using AI."""
        
        # Try providers in order of preference
        if self.preferred_provider == "auto":
            providers_to_try = ["openai", "anthropic", "ollama"]
        else:
            providers_to_try = [self.preferred_provider]
        
        for provider in providers_to_try:
            if self.providers_available.get(provider):
                try:
                    result = self._parse_with_provider(problem_text, provider)
                    if result.success:
                        return result
                except Exception as e:
                    print(f"Failed to parse with {provider}: {e}")
                    continue
        
        # Fallback to regex-based parsing
        return self._fallback_parse(problem_text)
    
    def _parse_with_provider(self, problem_text: str, provider: str) -> ParseResult:
        """Parse using specific AI provider."""
        
        prompt = self._create_parsing_prompt(problem_text)
        
        if provider == "openai":
            return self._parse_with_openai(prompt, problem_text)
        elif provider == "anthropic":
            return self._parse_with_anthropic(prompt, problem_text)
        elif provider == "ollama":
            return self._parse_with_ollama(prompt, problem_text)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _create_parsing_prompt(self, problem_text: str) -> str:
        """Create a structured prompt for AI parsing."""
        return f"""
You are a mathematical problem parser. Your task is to analyze natural language math problems and extract structured information.

Given this problem: "{problem_text}"

Please provide a JSON response with the following structure:
{{
    "problem_type": "algebraic|calculus|optimization|general",
    "variables": [
        {{"name": "x", "domain": "real"}},
        {{"name": "y", "domain": "real"}}
    ],
    "equations": [
        {{"left_side": "2x^2 - 8x + 6", "right_side": "0"}}
    ],
    "goal": "solve for x",
    "confidence": 0.95,
    "reasoning": "Brief explanation of your interpretation"
}}

Guidelines:
1. For "Solve: equation", convert to proper equation format (expr = 0 if no explicit = sign)
2. For "Find derivative of expr", set problem_type to "calculus" and goal appropriately
3. For "Integrate expr", set problem_type to "calculus"
4. Extract all variables mentioned (x, y, z, etc.)
5. Convert natural language equations to mathematical form
6. Clean up mathematical notation (convert unicode symbols to ASCII when needed)
7. Be permissive - if unsure, make reasonable assumptions

Examples:
- "Solve: 2x² - 8x + 6 = 0" → equation with left_side="2x^2 - 8x + 6", right_side="0"
- "Find derivative of x³ + 2x" → problem_type="calculus", goal="find derivative"
- "Solve for y: 3y + 5 = 14" → equation with left_side="3y + 5", right_side="14"

Return ONLY valid JSON, no additional text.
"""
    
    def _parse_with_openai(self, prompt: str, problem_text: str) -> ParseResult:
        """Parse using OpenAI GPT."""
        if not self.openai_client:
            raise Exception("OpenAI client not available")
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a precise mathematical problem parser. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            raw_response = response.choices[0].message.content
            return self._process_ai_response(raw_response, problem_text, "openai")
            
        except Exception as e:
            return ParseResult(
                success=False,
                problem=None,
                confidence=0.0,
                raw_response="",
                error=f"OpenAI parsing failed: {e}"
            )
    
    def _parse_with_anthropic(self, prompt: str, problem_text: str) -> ParseResult:
        """Parse using Anthropic Claude."""
        if not self.anthropic_client:
            raise Exception("Anthropic client not available")
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=0.1,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            raw_response = response.content[0].text
            return self._process_ai_response(raw_response, problem_text, "anthropic")
            
        except Exception as e:
            return ParseResult(
                success=False,
                problem=None,
                confidence=0.0,
                raw_response="",
                error=f"Anthropic parsing failed: {e}"
            )
    
    def _parse_with_ollama(self, prompt: str, problem_text: str) -> ParseResult:
        """Parse using Ollama local models."""
        if not OLLAMA_AVAILABLE:
            raise Exception("Ollama not available")
        
        try:
            response = ollama.chat(
                model="llama3.2",  # or another available model
                messages=[
                    {"role": "system", "content": "You are a precise mathematical problem parser. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            
            raw_response = response['message']['content']
            return self._process_ai_response(raw_response, problem_text, "ollama")
            
        except Exception as e:
            return ParseResult(
                success=False,
                problem=None,
                confidence=0.0,
                raw_response="",
                error=f"Ollama parsing failed: {e}"
            )
    
    def _process_ai_response(self, raw_response: str, problem_text: str, provider: str) -> ParseResult:
        """Process the AI response and create structured result."""
        try:
            # Clean up the response - sometimes AI adds extra text
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = raw_response.strip()
            
            # Parse JSON
            parsed_data = json.loads(json_str)
            
            # Create Variables
            variables = []
            for var_data in parsed_data.get("variables", []):
                variables.append(Variable(
                    name=var_data["name"],
                    domain=var_data.get("domain", "real")
                ))
            
            # Create Equations
            equations = []
            for eq_data in parsed_data.get("equations", []):
                equations.append(Equation(
                    left_side=eq_data["left_side"],
                    right_side=eq_data["right_side"]
                ))
            
            # Create MathProblem
            problem = MathProblem(
                problem_text=problem_text,
                problem_type=parsed_data.get("problem_type", "general"),
                variables=variables,
                equations=equations,
                goal=parsed_data.get("goal", "solve the problem")
            )
            
            return ParseResult(
                success=True,
                problem=problem,
                confidence=parsed_data.get("confidence", 0.8),
                raw_response=raw_response,
                error=None
            )
            
        except Exception as e:
            return ParseResult(
                success=False,
                problem=None,
                confidence=0.0,
                raw_response=raw_response,
                error=f"Failed to process AI response: {e}"
            )
    
    def _fallback_parse(self, problem_text: str) -> ParseResult:
        """Fallback parsing using improved regex patterns."""
        try:
            # Enhanced parsing logic for common cases
            problem_type = self._identify_problem_type(problem_text)
            variables = self._extract_variables(problem_text)
            equations = self._extract_equations_enhanced(problem_text)
            goal = self._extract_goal(problem_text)
            
            problem = MathProblem(
                problem_text=problem_text,
                problem_type=problem_type,
                variables=variables,
                equations=equations,
                goal=goal
            )
            
            return ParseResult(
                success=True,
                problem=problem,
                confidence=0.6,  # Lower confidence for regex fallback
                raw_response="Fallback regex parsing",
                error=None
            )
            
        except Exception as e:
            return ParseResult(
                success=False,
                problem=None,
                confidence=0.0,
                raw_response="",
                error=f"Fallback parsing failed: {e}"
            )
    
    def _extract_equations_enhanced(self, text: str) -> List[Equation]:
        """Enhanced equation extraction with better pattern matching."""
        equations = []
        
        # Handle "Solve: expression = value" pattern
        solve_colon_pattern = r"solve\s*:\s*(.+?)(?:=(.+?))?(?:\s|$)"
        match = re.search(solve_colon_pattern, text, re.IGNORECASE)
        
        if match:
            left_part = match.group(1).strip()
            right_part = match.group(2).strip() if match.group(2) else "0"
            
            # Clean up mathematical notation
            left_part = self._clean_math_expression(left_part)
            right_part = self._clean_math_expression(right_part)
            
            equations.append(Equation(
                left_side=left_part,
                right_side=right_part
            ))
        else:
            # Standard equation patterns
            eq_patterns = [
                r"(.+?)\s*=\s*(.+)",  # Basic equation
                r"find\s+(?:the\s+)?(?:roots|zeros)\s+of\s+(.+)",  # Find roots
            ]
            
            for pattern in eq_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) == 2:
                        left, right = match.groups()
                        equations.append(Equation(
                            left_side=self._clean_math_expression(left.strip()),
                            right_side=self._clean_math_expression(right.strip())
                        ))
                    elif len(match.groups()) == 1:
                        # "find roots of expr" case
                        expr = match.group(1).strip()
                        equations.append(Equation(
                            left_side=self._clean_math_expression(expr),
                            right_side="0"
                        ))
        
        return equations
    
    def _clean_math_expression(self, expr: str) -> str:
        """Clean up mathematical expressions."""
        # Convert unicode symbols to ASCII
        replacements = {
            '²': '^2',
            '³': '^3',
            '×': '*',
            '÷': '/',
            '−': '-',
            '∫': 'integral',
            '∂': 'd',
            '∆': 'Delta',
            'π': 'pi',
            'α': 'alpha',
            'β': 'beta',
            'γ': 'gamma',
            'θ': 'theta',
            'λ': 'lambda',
        }
        
        for old, new in replacements.items():
            expr = expr.replace(old, new)
        
        return expr.strip()
    
    def _identify_problem_type(self, text: str) -> str:
        """Identify the type of mathematical problem."""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ["derivative", "differentiate", "d/dx", "gradient"]):
            return "calculus"
        elif any(word in text_lower for word in ["integral", "integrate", "area under", "∫"]):
            return "calculus"
        elif any(word in text_lower for word in ["maximize", "minimize", "optimal", "critical points"]):
            return "optimization"
        elif any(word in text_lower for word in ["solve", "equation", "=", "roots", "root", "zeros", "factor"]):
            return "algebraic"
        else:
            return "general"
    
    def _extract_variables(self, text: str) -> List[Variable]:
        """Extract variables from the problem text."""
        variables = []
        found_vars = set()
        
        # Look for single letter variables
        var_pattern = r'\b([a-zA-Z])\b'
        matches = re.findall(var_pattern, text)
        
        for match in matches:
            if match not in found_vars and match.lower() not in ['a', 'an', 'is', 'or', 'of', 'if', 'to', 'in']:
                variables.append(Variable(name=match, domain="real"))
                found_vars.add(match)
        
        return variables
    
    def _extract_goal(self, text: str) -> str:
        """Extract the goal from the problem text."""
        text_lower = text.lower()
        
        # Look for specific goal patterns
        if "solve" in text_lower:
            if "for" in text_lower:
                match = re.search(r"solve.*for\s+(\w+)", text_lower)
                if match:
                    return f"solve for {match.group(1)}"
            return "solve"
        
        if "find" in text_lower:
            if "derivative" in text_lower:
                return "find derivative"
            elif "integral" in text_lower:
                return "find integral"
            elif any(word in text_lower for word in ["roots", "zeros"]):
                return "find roots"
        
        return "solve the problem"


# Create a global instance for easy import
ai_parser = AIMathParser()


def parse_math_problem(problem_text: str) -> ParseResult:
    """Convenient function to parse a math problem using AI."""
    return ai_parser.parse(problem_text)