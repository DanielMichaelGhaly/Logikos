"""
Step text generation from trace (rule-based).
"""

from .trace import StepTrace


class ReasoningGenerator:
    """Generate human-readable reasoning from solution traces."""

    def generate_reasoning(self, trace: StepTrace) -> str:
        """Generate reasoning text from a step trace."""
        # TODO: Implement reasoning generation
        return "Solution reasoning goes here."
