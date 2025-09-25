"""
Configuration management for Logikos Ollama integration.
Handles model selection, prompts, and settings.
"""

import os
from typing import Optional


class LogikosConfig:
    """Configuration for Logikos mathematical chat assistant."""

    def __init__(self):
        """Initialize configuration with environment variables and defaults."""
        # Model Configuration
        self.ollama_model = os.getenv("LOGIKOS_MODEL", "Randomblock1/nemotron-nano:latest")
        self.ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")

        # Performance Settings
        self.request_timeout = float(os.getenv("LOGIKOS_TIMEOUT", "30.0"))
        self.temperature = float(os.getenv("LOGIKOS_TEMPERATURE", "0.3"))

        # System Prompts
        self.system_prompt = self._get_system_prompt()
        self.classification_prompt_template = self._get_classification_template()
        self.reasoning_prompt_template = self._get_reasoning_template()
        self.representability_prompt_template = self._get_representability_template()

    def _get_system_prompt(self) -> str:
        """Get the main system prompt for initializing the model."""
        return """You are an expert mathematics tutor and problem solver with the following capabilities:

1. **Clear Step-by-Step Solutions**: You solve mathematical problems with detailed, easy-to-follow steps
2. **Educational Explanations**: You provide context that helps students understand concepts, assuming they have basic mathematical knowledge but need guidance
3. **Mathematical Precision**: You are precise with mathematical notation and terminology
4. **Pedagogical Approach**: You explain WHY each step is valid, not just WHAT to do
5. **Structured Responses**: You format your responses clearly with logical progression

Your goal is to help users understand mathematics through clear reasoning and step-by-step problem solving. Always show your work and explain the mathematical concepts involved."""

    def _get_classification_template(self) -> str:
        """Get the prompt template for question classification."""
        return """You are a mathematical question classifier. Convert the user's natural language input into a structured JSON format.

User input: "{user_input}"

Return ONLY a valid JSON object with these fields:
- "type": one of [solve, factor, expand, simplify, derivative, integral, limit, plot, graph, roots]
- "expression": the mathematical expression to work with
- "context": any additional context like "explain clearly", "step by step", etc.
- "domain": any specified domain like "from 0 to 2π"
- "visualization_hint": true if user wants a plot/graph, false otherwise

Examples:
Input: "factor x^2-4 and explain clearly"
Output: {{"type": "factor", "expression": "x^2-4", "context": "explain clearly", "domain": null, "visualization_hint": false}}

Input: "plot sin(x) from 0 to 2π"
Output: {{"type": "plot", "expression": "sin(x)", "context": null, "domain": "0 to 2π", "visualization_hint": true}}

Input: "solve 2x + 5 = 0 step by step"
Output: {{"type": "solve", "expression": "2x + 5 = 0", "context": "step by step", "domain": null, "visualization_hint": false}}

Your response (JSON only):"""

    def _get_reasoning_template(self) -> str:
        """Get the prompt template for educational reasoning."""
        return """You are an expert mathematics tutor. Provide a clear, educational explanation for this mathematical problem.

Question type: {question_type}
Expression: {expression}
Context: {context}

Guidelines:
1. Provide step-by-step explanations that someone with basic mathematical knowledge can follow
2. Explain the mathematical concepts and rules involved
3. Show your work clearly and justify each step
4. At the end, clearly state the final mathematical result in the format "FINAL RESULT: [your answer]"
5. Be educational and help the student understand WHY each step works

{type_specific_guidance}

Problem: {raw_input}

Your educational explanation:"""

    def _get_representability_template(self) -> str:
        """Get the prompt template for visualization representability analysis."""
        return """Analyze this mathematical problem and determine if it can be visualized:

Question type: {question_type}
Expression: {expression}

Can this problem be represented visually (plot, graph, contour map)?
Respond with ONLY a JSON object:
{{"representable": true/false, "visualization_type": "plot/contour/3d/none", "reason": "brief explanation"}}

Guidelines:
- Functions like f(x) = x^2 can be plotted → representable: true, type: "plot"
- Equations like 2x + 5 = 0 are not visual → representable: false, type: "none"
- Multivariable functions can be contour maps → representable: true, type: "contour"
- 3D surfaces for functions of two variables → representable: true, type: "3d"

Your response (JSON only):"""

    def get_type_specific_guidance(self, question_type: str) -> str:
        """Get specific guidance based on question type."""
        guidance_map = {
            "solve": """
For solving equations:
- Identify the type of equation (linear, quadratic, etc.)
- Show each algebraic step clearly
- Explain why each step is valid (addition property, multiplication property, etc.)
- Verify the solution by substitution if helpful
- Express the solution in its simplest form""",

            "factor": """
For factoring:
- Look for common factors first
- Identify special patterns (difference of squares, perfect square trinomials, etc.)
- Show the factoring process step by step
- Explain which factoring techniques you're using and why
- Verify by expanding the factored form if helpful""",

            "derivative": """
For derivatives:
- Identify which differentiation rules apply (power rule, product rule, chain rule, etc.)
- Show each step of the differentiation process
- Explain the rules being used and why they apply
- Simplify the final result
- Include the derivative notation clearly""",

            "integral": """
For integration:
- Identify the integration method needed (direct integration, substitution, parts, etc.)
- Show substitutions step by step if applicable
- Explain integration rules being used
- Don't forget the constant of integration (+C)
- Verify by differentiation if helpful""",

            "simplify": """
For simplification:
- Identify what needs to be simplified (fractions, radicals, expressions, etc.)
- Show each simplification step
- Explain the algebraic rules being applied
- Work toward the simplest possible form
- Ensure the final form is equivalent to the original""",

            "expand": """
For expansion:
- Identify the algebraic pattern (distributive property, FOIL, binomial theorem, etc.)
- Show each step of the expansion
- Explain which algebraic rules you're applying
- Combine like terms systematically
- Present the final expanded form clearly"""
        }

        return guidance_map.get(question_type, "Provide clear, step-by-step mathematical reasoning.")

    def get_model_initialization_options(self) -> dict:
        """Get options for model initialization."""
        return {
            "temperature": self.temperature,
            "num_predict": 2048,  # Max tokens for response
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        }


# Global configuration instance
config = LogikosConfig()