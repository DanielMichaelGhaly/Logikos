"""
LaTeX/HTML visualization from trace.
"""

from .trace import StepTrace


class Visualizer:
    """Generate LaTeX/HTML from solution traces."""

    def generate_latex(self, trace: StepTrace) -> str:
        """Generate LaTeX output."""
        return r"\begin{align} x = 42 \end{align}"

    def generate_html(self, trace: StepTrace) -> str:
        """Generate HTML output."""
        return "<div>Solution visualization</div>"
