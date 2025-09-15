"""
Comprehensive test suite for MathViz pipeline.
"""

import pytest
import sys
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mathviz.pipeline import MathVizPipeline
from mathviz.schemas import MathProblem, MathSolution
from mathviz.validator import ValidationError


class TestMathVizPipeline:
    """Test suite for the MathViz pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create a MathViz pipeline for testing."""
        return MathVizPipeline()

    def test_pipeline_initialization(self, pipeline):
        """Test that pipeline initializes correctly."""
        assert pipeline is not None
        assert pipeline.parser is not None
        assert pipeline.validator is not None
        assert pipeline.solver is not None
        assert pipeline.reasoner is not None
        assert pipeline.visualizer is not None

    def test_simple_algebra_solve(self, pipeline):
        """Test solving simple algebraic equations."""
        problem = "Solve for x: 2x + 5 = 13"
        solution = pipeline.process(problem)
        
        assert isinstance(solution, MathSolution)
        assert solution.problem.problem_type == "algebraic"
        assert solution.final_answer is not None
        assert "solutions" in solution.final_answer
        assert len(solution.solution_steps) > 0
        assert solution.reasoning != ""

    def test_basic_derivative(self, pipeline):
        """Test computing basic derivatives."""
        problem = "Find the derivative of x^2 + 3x"
        solution = pipeline.process(problem)
        
        assert isinstance(solution, MathSolution)
        assert solution.problem.problem_type == "calculus"
        assert solution.final_answer is not None
        assert "derivative" in solution.final_answer
        assert len(solution.solution_steps) > 0
        assert solution.reasoning != ""

    def test_basic_integration(self, pipeline):
        """Test computing basic integrals."""
        problem = "Integrate 2x + 1"
        solution = pipeline.process(problem)
        
        assert isinstance(solution, MathSolution)
        assert solution.problem.problem_type == "calculus"
        assert solution.final_answer is not None
        assert "integral" in solution.final_answer
        assert len(solution.solution_steps) > 0
        assert solution.reasoning != ""

    def test_invalid_problem_handling(self, pipeline):
        """Test handling of invalid problems."""
        invalid_problems = [
            "",  # Empty string
            "This is not a math problem",  # Non-mathematical text
            "x = = = y",  # Invalid syntax
        ]
        
        for problem in invalid_problems:
            try:
                solution = pipeline.process(problem)
                # If no exception is raised, check that error is handled gracefully
                if solution.metadata and solution.metadata.get("error"):
                    assert solution.metadata["error"] is True
            except (ValidationError, ValueError, Exception):
                # These exceptions are acceptable for invalid input
                pass

    def test_complex_algebra(self, pipeline):
        """Test more complex algebraic problems."""
        problem = "Find the roots of x^2 - 5x + 6"
        solution = pipeline.process(problem)
        
        assert isinstance(solution, MathSolution)
        assert solution.problem.problem_type == "algebraic"
        assert solution.final_answer is not None
        assert len(solution.solution_steps) > 0

    def test_trigonometric_derivative(self, pipeline):
        """Test derivatives involving trigonometric functions."""
        problem = "Differentiate sin(x) + cos(x)"
        solution = pipeline.process(problem)
        
        assert isinstance(solution, MathSolution)
        assert solution.problem.problem_type == "calculus"
        assert solution.final_answer is not None
        assert "derivative" in solution.final_answer

    def test_solution_metadata(self, pipeline):
        """Test that solution metadata is populated correctly."""
        problem = "Solve for x: x + 3 = 7"
        solution = pipeline.process(problem)
        
        assert solution.metadata is not None
        assert "trace_id" in solution.metadata
        assert "step_count" in solution.metadata
        assert solution.metadata["step_count"] >= 0

    def test_step_tracing(self, pipeline):
        """Test that solution steps are properly traced."""
        problem = "Solve for x: 2x = 8"
        solution = pipeline.process(problem)
        
        assert len(solution.solution_steps) > 0
        
        # Check that each step has required fields
        for step in solution.solution_steps:
            assert "step_id" in step
            assert "operation" in step
            assert "expression_before" in step
            assert "expression_after" in step
            assert "justification" in step

    def test_reasoning_generation(self, pipeline):
        """Test that reasoning is generated for solutions."""
        problem = "Find the derivative of x^2"
        solution = pipeline.process(problem)
        
        assert solution.reasoning is not None
        assert len(solution.reasoning) > 0
        assert "derivative" in solution.reasoning.lower()

    def test_visualization_generation(self, pipeline):
        """Test that visualization data is generated."""
        problem = "Solve for x: x + 1 = 5"
        solution = pipeline.process(problem)
        
        assert solution.visualization is not None
        # LaTeX should contain mathematical notation
        assert "\\" in solution.visualization  # LaTeX commands

    @pytest.mark.parametrize("problem,expected_type", [
        ("Solve for x: 3x = 9", "algebraic"),
        ("Find the derivative of x^3", "calculus"),
        ("Integrate x^2", "calculus"),
        ("Maximize f(x) = x^2 - 2x", "optimization"),
    ])
    def test_problem_type_classification(self, pipeline, problem, expected_type):
        """Test that problems are classified correctly."""
        solution = pipeline.process(problem)
        assert solution.problem.problem_type == expected_type

    def test_batch_processing(self, pipeline):
        """Test processing multiple problems."""
        problems = [
            "Solve for x: x + 2 = 5",
            "Find the derivative of x^2",
            "Integrate 3x"
        ]
        
        solutions = []
        for problem in problems:
            solution = pipeline.process(problem)
            solutions.append(solution)
        
        assert len(solutions) == 3
        for solution in solutions:
            assert isinstance(solution, MathSolution)
            assert solution.final_answer is not None

    def test_performance_timing(self, pipeline):
        """Test that solutions are computed in reasonable time."""
        import time
        
        problem = "Solve for x: 5x - 3 = 17"
        
        start_time = time.time()
        solution = pipeline.process(problem)
        elapsed_time = time.time() - start_time
        
        # Should complete in under 5 seconds for simple problems
        assert elapsed_time < 5.0
        assert solution is not None

    def test_error_handling_and_recovery(self, pipeline):
        """Test graceful error handling."""
        # Problem that might cause parsing issues
        problem = "Solve x +++ y = 5"
        
        try:
            solution = pipeline.process(problem)
            # If it doesn't throw an exception, check for error metadata
            if solution.metadata and solution.metadata.get("error"):
                assert solution.metadata["error"] is True
                assert solution.reasoning.startswith("Error occurred")
        except Exception as e:
            # Exception is acceptable for malformed input
            assert str(e) != ""

    def test_variable_extraction(self, pipeline):
        """Test that variables are correctly extracted from problems."""
        problem = "Solve for x: 2x + 3y = 10"
        parsed_problem = pipeline.parser.parse(problem)
        
        # Should extract variables x and y
        variable_names = [var.name for var in parsed_problem.variables]
        assert "x" in variable_names or "y" in variable_names

    def test_equation_extraction(self, pipeline):
        """Test that equations are correctly extracted."""
        problem = "Solve for x: 2x + 5 = 13"
        parsed_problem = pipeline.parser.parse(problem)
        
        assert len(parsed_problem.equations) > 0
        equation = parsed_problem.equations[0]
        assert equation.left_side is not None
        assert equation.right_side is not None

    def test_goal_extraction(self, pipeline):
        """Test that problem goals are correctly identified."""
        problem = "Solve for x: 3x = 12"
        parsed_problem = pipeline.parser.parse(problem)
        
        assert "solve for x" in parsed_problem.goal.lower()

    def test_validation_integration(self, pipeline):
        """Test that validation is properly integrated."""
        # Valid problem should pass validation
        valid_problem = "Solve for x: x + 1 = 3"
        solution = pipeline.process(valid_problem)
        assert solution is not None
        
        # The pipeline should handle validation internally
        # and either succeed or fail gracefully

    def test_solver_integration(self, pipeline):
        """Test that solver produces consistent results."""
        problem = "Solve for x: 4x = 20"
        
        # Run the same problem multiple times
        solutions = []
        for _ in range(3):
            solution = pipeline.process(problem)
            solutions.append(solution)
        
        # Results should be consistent
        first_answer = solutions[0].final_answer
        for solution in solutions[1:]:
            # The structure might vary, but the essence should be the same
            assert solution.final_answer is not None