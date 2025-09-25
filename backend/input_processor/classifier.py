"""
AI-powered question classifier using Ollama.
Converts natural language mathematical questions into structured JSON format.
"""

import json
import re
import time
from typing import Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from shared.schemas import QuestionClassification, QuestionType
from backend.config import config


class QuestionClassifier:
    """AI-powered question classifier using Ollama."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the classifier with Ollama model."""
        self.model_name = model_name or config.ollama_model
        self.ollama_available = OLLAMA_AVAILABLE
        self._model_initialized = False

        if not self.ollama_available:
            print("Warning: Ollama not available. Falling back to rule-based classification.")
        else:
            self._test_model_availability()

    def _test_model_availability(self):
        """Test if the Ollama model is available and responding."""
        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": "Test connection. Reply with 'OK'."}],
                options={"num_predict": 10}
            )
            if response and response.get('message', {}).get('content'):
                self._model_initialized = True
                print(f"✅ Ollama model '{self.model_name}' is available and responding")
            else:
                print(f"⚠️ Ollama model '{self.model_name}' available but not responding properly")
        except Exception as e:
            print(f"❌ Ollama model '{self.model_name}' not available: {e}")
            self.ollama_available = False

    def _initialize_model_with_system_prompt(self):
        """Initialize the model with the system prompt if not already done."""
        if not self._model_initialized or not self.ollama_available:
            return

        try:
            # Send system prompt to prepare the model for mathematical tasks
            ollama.chat(
                model=self.model_name,
                messages=[{"role": "system", "content": config.system_prompt}],
                options=config.get_model_initialization_options()
            )
            print(f"🤖 Model '{self.model_name}' initialized with math tutor system prompt")
        except Exception as e:
            print(f"Warning: Could not initialize model with system prompt: {e}")

    def classify_question(self, user_input: str) -> QuestionClassification:
        """
        Classify a natural language mathematical question.

        Args:
            user_input: Raw user input like "factor x^2-4 and explain clearly"

        Returns:
            QuestionClassification with structured data
        """
        start_time = time.time()

        if self.ollama_available and self._model_initialized:
            try:
                # Initialize with system prompt on first use
                self._initialize_model_with_system_prompt()

                classification = self._classify_with_ai(user_input)
                classification.confidence = 0.9  # High confidence for AI classification
                return classification
            except Exception as e:
                print(f"AI classification failed: {e}. Falling back to rule-based.")

        # Fallback to rule-based classification
        classification = self._classify_with_rules(user_input)
        classification.confidence = 0.6  # Medium confidence for rule-based
        return classification

    def _classify_with_ai(self, user_input: str) -> QuestionClassification:
        """Use Ollama AI to classify the question."""
        prompt = config.classification_prompt_template.format(user_input=user_input)

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": config.temperature,
                "num_predict": 512,  # Limit response length for classification
                "top_p": 0.9,
            }
        )

        # Parse AI response
        ai_response = response['message']['content'].strip()

        # Extract JSON from response (handle cases where AI adds extra text)
        json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in AI response: {json_str}. Error: {e}")
        else:
            raise ValueError(f"No valid JSON found in AI response: {ai_response}")

        # Validate the required fields
        if "type" not in data or "expression" not in data:
            raise ValueError(f"Missing required fields in AI response: {data}")

        # Validate question type
        try:
            question_type = QuestionType(data.get("type", "unknown"))
        except ValueError:
            print(f"Warning: Unknown question type '{data.get('type')}', using 'unknown'")
            question_type = QuestionType.UNKNOWN

        return QuestionClassification(
            type=question_type,
            expression=data.get("expression", ""),
            context=data.get("context"),
            domain=data.get("domain"),
            visualization_hint=data.get("visualization_hint", False),
            raw_input=user_input
        )

    def _classify_with_rules(self, user_input: str) -> QuestionClassification:
        """Rule-based fallback classification."""
        input_lower = user_input.lower()

        # Determine question type
        question_type = QuestionType.UNKNOWN
        visualization_hint = False

        if any(word in input_lower for word in ["solve", "find solution", "what is"]):
            question_type = QuestionType.SOLVE
        elif any(word in input_lower for word in ["factor", "factorize"]):
            question_type = QuestionType.FACTOR
        elif any(word in input_lower for word in ["expand", "expand out"]):
            question_type = QuestionType.EXPAND
        elif any(word in input_lower for word in ["simplify", "reduce"]):
            question_type = QuestionType.SIMPLIFY
        elif any(word in input_lower for word in ["derivative", "differentiate", "d/dx"]):
            question_type = QuestionType.DERIVATIVE
        elif any(word in input_lower for word in ["integral", "integrate", "antiderivative"]):
            question_type = QuestionType.INTEGRAL
        elif any(word in input_lower for word in ["limit", "approaches"]):
            question_type = QuestionType.LIMIT
        elif any(word in input_lower for word in ["plot", "graph", "draw", "visualize"]):
            question_type = QuestionType.PLOT
            visualization_hint = True
        elif any(word in input_lower for word in ["roots", "zeros", "find roots"]):
            question_type = QuestionType.ROOTS

        # Extract mathematical expression (simple pattern matching)
        expression = self._extract_expression(user_input)

        # Extract context
        context = None
        if any(phrase in input_lower for phrase in ["explain", "step by step", "clearly", "detail"]):
            context = "explain clearly"

        # Extract domain
        domain = None
        domain_match = re.search(r'from\s+([^to]+)\s+to\s+([^,\s]+)', input_lower)
        if domain_match:
            domain = f"{domain_match.group(1).strip()} to {domain_match.group(2).strip()}"

        return QuestionClassification(
            type=question_type,
            expression=expression,
            context=context,
            domain=domain,
            visualization_hint=visualization_hint,
            raw_input=user_input
        )

    def _extract_expression(self, text: str) -> str:
        """Extract mathematical expression from text using patterns."""
        # Remove common instruction words
        cleaned = re.sub(r'\b(solve|factor|plot|graph|find|derivative|integral|simplify|expand)\b', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'\b(and|please|can you|help me|explain|clearly|step by step)\b', '', cleaned, flags=re.IGNORECASE)

        # Look for mathematical expressions
        # Match common patterns like equations, functions, polynomials
        patterns = [
            r'[a-zA-Z0-9\+\-\*\/\^\(\)\=\s]+',  # General mathematical expression
            r'f\([^)]+\)\s*=\s*[^,\s]+',        # Function definition
            r'[a-zA-Z]+\([^)]*\)',              # Function calls
        ]

        for pattern in patterns:
            matches = re.findall(pattern, cleaned)
            if matches:
                # Return the longest match (likely the main expression)
                expression = max(matches, key=len).strip()
                if len(expression) > 2:  # Avoid single characters
                    return expression

        # Fallback: return cleaned text
        return cleaned.strip() or text.strip()