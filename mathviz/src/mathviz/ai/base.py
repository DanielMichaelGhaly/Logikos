from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class StructuredProblem:
    problem_text: str
    # An optional normalized problem type hint (e.g., "algebra", "calculus")
    problem_type: Optional[str] = None
    # Optional flags for visualization
    visualize: bool = True


@dataclass
class AIChatResult:
    reply_text: str
    structured_problem: StructuredProblem
    # Optional additional metadata from the AI pass (e.g., difficulty, tags)
    metadata: Dict[str, Any] | None = None


@dataclass
class AIStepExplanation:
    # An optional textual explanation of the steps, post-solution
    reasoning: Optional[str]


class AIProvider(Protocol):
    def generate_structured_problem(self, message: str) -> AIChatResult:
        """Turn a free-form user message into a structured math problem.

        This should produce a normalized prompt suitable for the mathviz pipeline
        (parser/validator/solver). It may include a lightweight problem_type hint
        and whether a graph is desired.
        """

    def refine_problem(self, message: str, last_error: Optional[str] = None) -> AIChatResult:
        """Validate/repair a user message into a better normalized form.

        Use this when the first attempt failed parsing/solving. Providers can
        either produce a clarified normalized_problem_text or a reply_text asking
        the user to clarify missing pieces.
        """

    def explain_steps(self, solution: Dict[str, Any]) -> AIStepExplanation:
        """Optionally enhance or rewrite step explanations using AI."""


def provider_from_env() -> AIProvider:
    """Factory: choose AI provider based on env vars.

    Set MATHVIZ_AI_PROVIDER=ollama to use a local Ollama model (e.g., Qwen).
    Otherwise, default to the stub provider.
    """
    provider = os.environ.get("MATHVIZ_AI_PROVIDER", "stub").lower()
    if provider == "ollama":
        from .providers.ollama import OllamaAIProvider
        return OllamaAIProvider()
    else:
        from .providers.stub import StubAIProvider
        return StubAIProvider()
