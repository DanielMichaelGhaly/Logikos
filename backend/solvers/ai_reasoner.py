"""
AI reasoning engine using Ollama for educational step-by-step explanations.
Provides pedagogical explanations and determines visualization representability.
"""

import time
import re
import json
from typing import List, Optional

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

from shared.schemas import AIResult, QuestionClassification, QuestionType
from backend.config import config


class AIReasoner:
    """AI-powered reasoning engine for educational mathematical explanations."""

    def __init__(self, model_name: Optional[str] = None):
        """Initialize the AI reasoner with Ollama model."""
        self.model_name = model_name or config.ollama_model
        self.ollama_available = OLLAMA_AVAILABLE
        self._model_initialized = False

        if not self.ollama_available:
            print("Warning: Ollama not available. AI reasoning disabled.")
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
                print(f"✅ AI Reasoner: Ollama model '{self.model_name}' is available")
            else:
                print(f"⚠️ AI Reasoner: Ollama model '{self.model_name}' available but not responding properly")
        except Exception as e:
            print(f"❌ AI Reasoner: Ollama model '{self.model_name}' not available: {e}")
            self.ollama_available = False

    def _initialize_model_with_system_prompt(self):
        """Initialize the model with the system prompt for mathematical tutoring."""
        if not self._model_initialized or not self.ollama_available:
            return

        try:
            # Send system prompt to prepare the model for mathematical tutoring
            ollama.chat(
                model=self.model_name,
                messages=[{"role": "system", "content": config.system_prompt}],
                options=config.get_model_initialization_options()
            )
            print(f"🤖 AI Reasoner: Model '{self.model_name}' initialized with math tutor system prompt")
        except Exception as e:
            print(f"Warning: Could not initialize AI reasoner with system prompt: {e}")

    def reason_about_question(self, classification: QuestionClassification) -> AIResult:
        """
        Provide educational reasoning for a mathematical question.

        Args:
            classification: Structured question from the classifier

        Returns:
            AIResult with explanations and representability analysis
        """
        start_time = time.time()

        if not self.ollama_available or not self._model_initialized:
            return AIResult(
                success=False,
                error="Ollama not available or not initialized",
                execution_time=time.time() - start_time
            )

        try:
            # Initialize with system prompt on first use
            self._initialize_model_with_system_prompt()

            # Generate educational explanation
            explanation, steps, math_result = self._generate_explanation(classification)

            # Determine if problem is representable
            representable, viz_type = self._analyze_representability(classification)

            execution_time = time.time() - start_time

            return AIResult(
                success=True,
                explanation=explanation,
                steps=steps,
                mathematical_result=math_result,
                representable=representable,
                visualization_type=viz_type,
                execution_time=execution_time
            )

        except Exception as e:
            execution_time = time.time() - start_time
            return AIResult(
                success=False,
                error=f"AI reasoning error: {str(e)}",
                execution_time=execution_time
            )

    def _generate_explanation(self, classification: QuestionClassification) -> tuple:
        """Generate step-by-step educational explanation."""
        prompt = self._build_explanation_prompt(classification)

        response = ollama.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": config.temperature,
                "num_predict": 1024,  # Allow longer responses for explanations
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            }
        )

        ai_response = response['message']['content']

        # Parse response for explanation, steps, and mathematical result
        explanation = ai_response.strip()
        steps = self._extract_steps(ai_response)
        math_result = self._extract_mathematical_result(ai_response)

        return explanation, steps, math_result

    def _build_explanation_prompt(self, classification: QuestionClassification) -> str:
        """Build educational prompt based on question type."""
        type_specific_guidance = config.get_type_specific_guidance(classification.type.value)

        prompt = config.reasoning_prompt_template.format(
            question_type=classification.type.value,
            expression=classification.expression,
            context=classification.context or "standard explanation",
            type_specific_guidance=type_specific_guidance,
            raw_input=classification.raw_input
        )

        return prompt

    def _extract_steps(self, ai_response: str) -> List[str]:
        """Extract step-by-step breakdown from AI response."""
        steps = []

        # Look for numbered steps
        step_patterns = [
            r'(\d+[\.\)]\s*[^\n]+)',  # "1. Step description" or "1) Step description"
            r'(Step \d+[:\.]?\s*[^\n]+)',  # "Step 1: description"
            r'(First[,:]?\s*[^\n]+)',  # "First, ..."
            r'(Next[,:]?\s*[^\n]+)',   # "Next, ..."
            r'(Then[,:]?\s*[^\n]+)',   # "Then, ..."
            r'(Finally[,:]?\s*[^\n]+)', # "Finally, ..."
        ]

        for pattern in step_patterns:
            matches = re.findall(pattern, ai_response, re.IGNORECASE)
            steps.extend([match.strip() for match in matches])

        # If no clear steps found, split by sentences and take meaningful ones
        if not steps:
            sentences = re.split(r'[.!?]+', ai_response)
            steps = [s.strip() for s in sentences if len(s.strip()) > 20 and any(word in s.lower() for word in ['solve', 'factor', 'differentiate', 'integrate', 'substitute', 'simplify'])]

        return steps[:10]  # Limit to 10 steps for readability

    def _extract_mathematical_result(self, ai_response: str) -> Optional[str]:
        """Extract the final mathematical result from AI response."""
        # Look for explicit final result markers
        result_patterns = [
            r'FINAL RESULT:\s*([^\n]+)',
            r'Final answer:\s*([^\n]+)',
            r'Answer:\s*([^\n]+)',
            r'Result:\s*([^\n]+)',
            r'Solution:\s*([^\n]+)',
        ]

        for pattern in result_patterns:
            match = re.search(pattern, ai_response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Look for mathematical expressions at the end
        lines = ai_response.strip().split('\n')
        for line in reversed(lines):
            line = line.strip()
            if line and any(char in line for char in ['=', '+', '-', '*', '/', '^', '(', ')']):
                # Likely a mathematical expression
                if len(line) < 100:  # Reasonable length
                    return line

        return None

    def _analyze_representability(self, classification: QuestionClassification) -> tuple:
        """Determine if the problem can be visualized and what type."""
        representable = False
        viz_type = None

        # Explicit visualization requests
        if classification.visualization_hint or classification.type in [QuestionType.PLOT, QuestionType.GRAPH]:
            representable = True
            viz_type = "plot"
            return representable, viz_type

        # Use AI to determine representability
        try:
            prompt = config.representability_prompt_template.format(
                question_type=classification.type.value,
                expression=classification.expression
            )

            response = ollama.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1,  # Low temperature for consistent analysis
                    "num_predict": 256,
                    "top_p": 0.9,
                }
            )

            ai_response = response['message']['content'].strip()

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(0))
                    representable = data.get("representable", False)
                    viz_type = data.get("visualization_type", "none")
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON in representability response: {json_match.group(0)}")

        except Exception as e:
            print(f"Representability analysis failed: {e}")

        return representable, viz_type

    def extract_mathematical_result_for_comparison(self, ai_result: AIResult) -> Optional[str]:
        """
        Extract mathematical result for confidence comparison with SymPy.

        Args:
            ai_result: AI reasoning result

        Returns:
            String representation of mathematical result or None
        """
        if not ai_result.success or not ai_result.mathematical_result:
            return None

        result = ai_result.mathematical_result.strip()

        # Clean up common formatting
        result = re.sub(r'^(Answer|Result|Solution):\s*', '', result, flags=re.IGNORECASE)
        result = result.strip()

        # Extract mathematical expressions
        # Look for patterns like x = 2, [-2, 2], (x-2)(x+2), etc.
        math_patterns = [
            r'x\s*=\s*[^,\s]+',          # x = value
            r'\[.*\]',                   # [list of values]
            r'\([^)]+\)\([^)]+\)',       # factored form like (x-2)(x+2)
            r'[+-]?\d*\.?\d+[^,\s]*',    # numerical values
        ]

        for pattern in math_patterns:
            match = re.search(pattern, result)
            if match:
                return match.group(0).strip()

        return result if len(result) < 100 else None