"""
Main orchestration pipeline.
"""

from .parser import MathParser
from .validator import MathValidator
from .solver import MathSolver
from .reasoning import ReasoningGenerator
from .viz import Visualizer
from .schemas import MathSolution


class MathVizPipeline:
    """Main pipeline orchestrating the math problem solving process."""

    def __init__(self):
        self.parser = MathParser()
        self.validator = MathValidator()
        self.solver = MathSolver()
        self.reasoner = ReasoningGenerator()
        self.visualizer = Visualizer()

    def process(self, problem_text: str) -> MathSolution:
        """Process a math problem through the complete pipeline."""
        # Parse natural language to structured problem
        problem = self.parser.parse(problem_text)
        
        # Validate the problem
        if not self.validator.validate(problem):
            raise ValueError("Invalid problem")
        
        # Solve the problem
        solution = self.solver.solve(problem)
        
        return solution
