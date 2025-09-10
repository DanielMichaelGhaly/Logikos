"""
MathViz: Mathematical problem visualization and solving pipeline.
"""

from .pipeline import MathVizPipeline
from .schemas import MathProblem, MathSolution
from .trace import Step, StepTrace

__version__ = "0.1.0"
__all__ = ["MathVizPipeline", "MathProblem", "MathSolution", "Step", "StepTrace"]
