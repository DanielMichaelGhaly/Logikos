from __future__ import annotations

from typing import Any, Dict
from ..base import AIProvider, StructuredProblem, AIChatResult, AIStepExplanation


class StubAIProvider:
    """A minimal provider that simulates an AI pass.

    This provider does two things:
      1) Detects whether the message looks like a math question and creates a
         StructuredProblem with optional visualization hints (detects common
         keywords like plot/graph/sin/cos/tan).
      2) After solving, it lightly reformats steps/reasoning (placeholder).

    Swap with a real provider by implementing the AIProvider Protocol.
    """

    def generate_structured_problem(self, message: str) -> AIChatResult:
        import re
        text = message.strip()

        # Heuristic normalization for common patterns
        norm = re.sub(r"^\s*(solve|find|compute|determine)\b[:,-]*\s*", "", text, flags=re.IGNORECASE)
        # remove trailing question mark/spaces
        norm = re.sub(r"[?\s]+$", "", norm)
        # trim constructions like "for x" / "for y"
        norm = re.sub(r"\bfor\s+[a-zA-Z][a-zA-Z0-9_]*\b", "", norm, flags=re.IGNORECASE).strip()

        visualize = any(k in norm.lower() for k in ["plot", "graph", "desmos", "sin", "cos", "tan"])

        # naive type hint
        problem_type = None
        if any(k in norm.lower() for k in ["derivative", "differentiate", "integrate", "integral"]):
            problem_type = "calculus"
        elif any(k in norm.lower() for k in ["solve", "roots", "factor", "=", "quadratic"]):
            problem_type = "algebra"

        reply = "Got it — I’ll parse your question, solve it with the math engine, and then explain the steps."
        return AIChatResult(
            reply_text=reply,
            structured_problem=StructuredProblem(problem_text=norm or text, problem_type=problem_type, visualize=visualize),
            metadata={"provider": "stub"},
        )

    def refine_problem(self, message: str, last_error: Optional[str] = None) -> AIChatResult:
        # Simple refinement: strip common prefixes and re-run normalization
        text = message.strip()
        text = re.sub(r"^\s*(solve for|solve|differentiate|derivative of|integrate|integral of)\b[:,-]*\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*f\s*\(\s*x\s*\)\s*=\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*y\s*=\s*", "", text, flags=re.IGNORECASE)
        # If we removed directive verbs, keep a reply that we refined input
        reply = "I refined the input to a solver-friendly form."
        return AIChatResult(
            reply_text=reply,
            structured_problem=StructuredProblem(problem_text=text, visualize=True),
            metadata={"provider": "stub", "refined": True, "last_error": last_error},
        )

    def explain_steps(self, solution: Dict[str, Any]) -> AIStepExplanation:
        # In a real provider, you would rephrase or enhance the steps.
        # Here we just pass-through any provided reasoning.
        reasoning = None
        if isinstance(solution, dict):
            reasoning = solution.get("reasoning")
        return AIStepExplanation(reasoning=reasoning)
