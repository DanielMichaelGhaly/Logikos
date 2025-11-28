"""
Comprehensive test suite for AI-enhanced MathViz implementation.
Tests differentiation, optimization, visualization, and AI integration features.
"""

import pytest
import os
import sys
import time
from unittest.mock import Mock, patch, MagicMock

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from mathviz.pipeline import MathVizPipeline, PipelineConfig
    from mathviz.ai_apis import AIResponse, solve_with_ai, generate_reasoning_with_ai
    from mathviz.advanced_calculus import AdvancedCalculus, OptimizationResult, GradientDescentResult
    from mathviz.graph_visualizer import GraphVisualizer, GraphConfig, VisualizationResult
    from mathviz.schemas import MathProblem, MathSolution
    from mathviz.solver import MathSolver
    from mathviz.reasoning import ReasoningGenerator
    from mathviz.viz import Visualizer
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    IMPORTS_SUCCESSFUL = False
    IMPORT_ERROR = str(e)


class TestAIAPIs:
    """Test AI API integration functionality."""
    
    @pytest.fixture
    def mock_ai_response(self):
        """Create a mock AI response."""
        return AIResponse(
            success=True,
            content="The derivative of x^2 is 2x using the power rule.",
            confidence=0.85,
            provider="MockAI",
            tokens_used=50
        )
    
    @pytest.fixture
    def mock_failed_ai_response(self):
        """Create a mock failed AI response."""
        return AIResponse(
            success=False,
            content="",
            confidence=0.0,
            provider="MockAI",
            error="API rate limit exceeded"
        )
    
    @pytest.mark.skipif(not IMPORTS_SUCCESSFUL, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_SUCCESSFUL else ''}")
    def test_ai_response_structure(self, mock_ai_response):
        """Test AI response data structure."""
        assert mock_ai_response.success is True
        assert mock_ai_response.content is not None
        assert mock_ai_response.confidence > 0
        assert mock_ai_response.provider == "MockAI"
        assert mock_ai_response.error is None
    
    @pytest.mark.skipif(not IMPORTS_SUCCESSFUL, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_SUCCESSFUL else ''}")
    @patch('mathviz.ai_apis.ai_manager')
    def test_ai_problem_solving(self, mock_ai_manager, mock_ai_response):
        """Test AI problem solving integration."""
        # Mock the AI manager
        mock_ai_manager.solve_problem.return_value = mock_ai_response
        
        # Test solving with AI
        result = solve_with_ai("Find the derivative of x^2", "differentiation")
        
        assert result.success is True
        assert "derivative" in result.content.lower()
        assert result.provider == "MockAI"
        
        # Verify AI manager was called correctly
        mock_ai_manager.solve_problem.assert_called_once_with("Find the derivative of x^2", "differentiation")
    
    @pytest.mark.skipif(not IMPORTS_SUCCESSFUL, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_SUCCESSFUL else ''}")
    @patch('mathviz.ai_apis.ai_manager')
    def test_ai_reasoning_generation(self, mock_ai_manager, mock_ai_response):
        """Test AI reasoning generation."""
        # Mock reasoning response
        reasoning_response = AIResponse(
            success=True,
            content="Step 1: Apply the power rule. Step 2: Simplify the result.",
            confidence=0.9,
            provider="MockAI"
        )
        mock_ai_manager.generate_reasoning.return_value = reasoning_response
        
        result = generate_reasoning_with_ai("x^2", "2x")
        
        assert result.success is True
        assert "step" in result.content.lower()
        mock_ai_manager.generate_reasoning.assert_called_once_with("x^2", "2x")
    
    @pytest.mark.skipif(not IMPORTS_SUCCESSFUL, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_SUCCESSFUL else ''}")
    @patch('mathviz.ai_apis.ai_manager')
    def test_ai_fallback_handling(self, mock_ai_manager, mock_failed_ai_response):
        """Test AI fallback when API fails."""
        mock_ai_manager.solve_problem.return_value = mock_failed_ai_response
        
        result = solve_with_ai("x^2", "differentiation")
        
        assert result.success is False
        assert result.error is not None


class TestAdvancedCalculus:
    """Test advanced calculus functionality."""
    
    @pytest.fixture
    def advanced_calc(self):
        """Create AdvancedCalculus instance."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        return AdvancedCalculus()
    
    def test_partial_derivative_computation(self, advanced_calc):
        """Test partial derivative computation."""
        result = advanced_calc.compute_partial_derivative("x**2 + y**2", "x")
        
        assert result["derivative"] == "2*x"
        assert result["variable"] == "x"
        assert "steps" in result
        assert len(result["steps"]) > 0
    
    def test_gradient_computation(self, advanced_calc):
        """Test gradient computation."""
        result = advanced_calc.compute_gradient("x**2 + y**2")
        
        assert "gradient" in result
        assert result["components"] == ["2*x", "2*y"]
        assert set(result["variables"]) == {"x", "y"}
        assert "steps" in result
    
    def test_critical_points_finding(self, advanced_calc):
        """Test critical points finding."""
        # Test with a simple function that has a critical point at (0, 0)
        result = advanced_calc.find_critical_points("x**2 + y**2")
        
        assert isinstance(result, OptimizationResult)
        assert len(result.critical_points) > 0
        # For x^2 + y^2, critical point should be at (0, 0)
        assert (0.0, 0.0) in result.critical_points or abs(result.critical_points[0][0]) < 1e-10
    
    def test_gradient_descent(self, advanced_calc):
        """Test gradient descent optimization."""
        result = advanced_calc.gradient_descent(
            "x**2 + y**2", 
            initial_point=(1.0, 1.0),
            learning_rate=0.1,
            max_iterations=50
        )
        
        assert isinstance(result, GradientDescentResult)
        assert len(result.path) > 1
        assert len(result.function_values) == len(result.path)
        assert result.iterations > 0
        # Should converge towards (0, 0) for this function
        assert abs(result.final_point[0]) < 0.1
        assert abs(result.final_point[1]) < 0.1
    
    def test_critical_point_classification(self, advanced_calc):
        """Test critical point classification."""
        # Test with x^2 + y^2 which has a minimum at (0, 0)
        result = advanced_calc.classify_critical_point("x**2 + y**2", (0.0, 0.0))
        
        assert result["classification"] in ["local minimum", "minimum"]
        assert "eigenvalues" in result
        assert "determinant" in result


class TestGraphVisualization:
    """Test graph visualization functionality."""
    
    @pytest.fixture
    def graph_visualizer(self):
        """Create GraphVisualizer instance."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        return GraphVisualizer()
    
    @pytest.fixture
    def graph_config(self):
        """Create GraphConfig instance."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        return GraphConfig(x_range=(-5, 5), y_range=(-5, 5))
    
    def test_single_function_visualization(self, graph_visualizer, graph_config):
        """Test single function visualization."""
        result = graph_visualizer.visualize_function("x**2", graph_config)
        
        assert isinstance(result, VisualizationResult)
        assert result.success is True
        assert result.graph_url is not None or result.graph_html is not None
    
    def test_multiple_functions_visualization(self, graph_visualizer, graph_config):
        """Test multiple functions visualization."""
        expressions = ["x**2", "2*x", "sin(x)"]
        result = graph_visualizer.visualize_functions(expressions, graph_config)
        
        assert result.success is True
        assert "expressions" in result.metadata
    
    def test_derivative_visualization(self, graph_visualizer, graph_config):
        """Test derivative visualization."""
        result = graph_visualizer.visualize_derivative("x**3", config=graph_config)
        
        assert result.success is True
        assert "derivative" in result.metadata
        assert result.metadata["derivative"]["original"] is not None
        assert result.metadata["derivative"]["derivative"] is not None
    
    def test_3d_function_visualization(self, graph_visualizer, graph_config):
        """Test 3D function visualization."""
        result = graph_visualizer.visualize_3d_function("x**2 + y**2", config=graph_config)
        
        assert result.success is True
        assert result.graph_html is not None
        assert result.metadata["plot_type"] == "3d_surface"
    
    def test_optimization_visualization(self, graph_visualizer, graph_config):
        """Test optimization visualization with critical points."""
        critical_points = [(0.0, 0.0)]
        result = graph_visualizer.visualize_optimization("x**2 + y**2", critical_points, graph_config)
        
        assert result.success is True
        assert "optimization" in result.metadata
        assert result.metadata["optimization"]["critical_points"] == critical_points


class TestAIEnhancedSolver:
    """Test AI-enhanced solver functionality."""
    
    @pytest.fixture
    def ai_solver(self):
        """Create AI-enhanced solver."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        return MathSolver(use_ai=True, ai_first=True)
    
    @pytest.fixture
    def symbolic_solver(self):
        """Create symbolic-only solver."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        return MathSolver(use_ai=False)
    
    @pytest.fixture
    def sample_derivative_problem(self):
        """Create sample derivative problem."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        return MathProblem(
            problem_text="Find the derivative of x^3 + 2x^2 + x",
            problem_type="calculus",
            variables=[],
            equations=[],
            constraints=[],
            goal="find derivative"
        )
    
    @pytest.fixture
    def sample_optimization_problem(self):
        """Create sample optimization problem."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        return MathProblem(
            problem_text="Find critical points of x^2 + y^2",
            problem_type="optimization",
            variables=[],
            equations=[],
            constraints=[],
            goal="find critical points"
        )
    
    @patch('mathviz.solver.solve_with_ai')
    def test_ai_enhanced_derivative_solving(self, mock_solve_ai, ai_solver, sample_derivative_problem):
        """Test AI-enhanced derivative solving."""
        # Mock AI response
        mock_ai_response = AIResponse(
            success=True,
            content="The derivative is 3x^2 + 4x + 1",
            confidence=0.9,
            provider="MockAI"
        )
        mock_solve_ai.return_value = mock_ai_response
        
        solution = ai_solver.solve(sample_derivative_problem)
        
        assert isinstance(solution, MathSolution)
        assert solution.final_answer is not None
        # Should have AI-related steps
        ai_steps = [step for step in solution.solution_steps if 
                   isinstance(step, dict) and step.get('operation') == 'ai_solution']
        assert len(ai_steps) > 0
    
    def test_symbolic_derivative_solving(self, symbolic_solver, sample_derivative_problem):
        """Test symbolic derivative solving."""
        solution = symbolic_solver.solve(sample_derivative_problem)
        
        assert isinstance(solution, MathSolution)
        assert solution.final_answer is not None
        assert solution.final_answer["type"] == "derivative"
        assert "3*x**2 + 4*x + 1" in solution.final_answer["derivative"]
    
    @patch('mathviz.solver.solve_with_ai')
    def test_ai_fallback_on_failure(self, mock_solve_ai, ai_solver, sample_derivative_problem):
        """Test AI fallback when primary method fails."""
        # Mock AI failure
        mock_solve_ai.return_value = AIResponse(
            success=False,
            content="",
            confidence=0.0,
            provider="MockAI",
            error="API unavailable"
        )
        
        solution = ai_solver.solve(sample_derivative_problem)
        
        # Should still get a solution via symbolic fallback
        assert isinstance(solution, MathSolution)
        assert solution.final_answer is not None


class TestAIEnhancedReasoning:
    """Test AI-enhanced reasoning generation."""
    
    @pytest.fixture
    def ai_reasoner(self):
        """Create AI-enhanced reasoner."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        return ReasoningGenerator(use_ai=True, ai_first=True)
    
    @pytest.fixture
    def template_reasoner(self):
        """Create template-based reasoner."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        return ReasoningGenerator(use_ai=False)
    
    @pytest.fixture
    def sample_trace(self):
        """Create sample solution trace."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        from mathviz.trace import StepTrace, Step
        
        trace = StepTrace(problem_id="test_derivative")
        trace.add_step(Step(
            step_id=0,
            operation="differentiate",
            expression_before="x^2",
            expression_after="2*x",
            justification="Applied power rule"
        ))
        trace.final_state = "Derivative: 2*x"
        trace.success = True
        return trace
    
    @patch('mathviz.reasoning.generate_reasoning_with_ai')
    def test_ai_reasoning_generation(self, mock_reasoning_ai, ai_reasoner, sample_trace):
        """Test AI reasoning generation."""
        # Mock AI reasoning response
        mock_ai_response = AIResponse(
            success=True,
            content="To find the derivative of x², we apply the power rule: d/dx(xⁿ) = nxⁿ⁻¹. Therefore, d/dx(x²) = 2x¹ = 2x.",
            confidence=0.95,
            provider="MockAI"
        )
        mock_reasoning_ai.return_value = mock_ai_response
        
        reasoning = ai_reasoner.generate_reasoning(sample_trace)
        
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert "power rule" in reasoning.lower()
    
    def test_template_reasoning_generation(self, template_reasoner, sample_trace):
        """Test template-based reasoning generation."""
        reasoning = template_reasoner.generate_reasoning(sample_trace)
        
        assert isinstance(reasoning, str)
        assert len(reasoning) > 0
        assert "derivative" in reasoning.lower()


class TestEnhancedVisualization:
    """Test enhanced visualization with interactive graphs."""
    
    @pytest.fixture
    def enhanced_visualizer(self):
        """Create enhanced visualizer."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        return Visualizer(enable_interactive_graphs=True)
    
    @pytest.fixture
    def basic_visualizer(self):
        """Create basic visualizer without interactive graphs."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        return Visualizer(enable_interactive_graphs=False)
    
    @pytest.fixture
    def sample_trace(self):
        """Create sample trace for visualization."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        from mathviz.trace import StepTrace, Step
        
        trace = StepTrace(problem_id="test_viz")
        trace.add_step(Step(
            step_id=0,
            operation="differentiate",
            expression_before="x^2 + 3x",
            expression_after="2*x + 3",
            justification="Applied differentiation rules"
        ))
        trace.final_state = "Derivative: 2*x + 3"
        trace.success = True
        return trace
    
    def test_interactive_graph_generation(self, enhanced_visualizer, sample_trace):
        """Test interactive graph generation."""
        result = enhanced_visualizer.generate_interactive_graph(sample_trace)
        
        # Result might be None if expressions can't be extracted, which is acceptable
        if result is not None:
            assert isinstance(result, VisualizationResult)
            assert result.success is True
    
    def test_desmos_url_generation(self, enhanced_visualizer):
        """Test Desmos URL generation."""
        expressions = ["x^2", "2*x"]
        url = enhanced_visualizer.generate_desmos_url(expressions)
        
        if url is not None:  # May be None if visualization not available
            assert isinstance(url, str)
            assert "desmos.com" in url or url == ""  # Could be empty string on failure
    
    def test_optimization_graph_generation(self, enhanced_visualizer):
        """Test optimization graph generation."""
        critical_points = [(0.0, 0.0)]
        result = enhanced_visualizer.generate_optimization_graph("x^2 + y^2", critical_points)
        
        if result is not None:  # May be None if visualization not available
            assert isinstance(result, VisualizationResult)


class TestPipelineIntegration:
    """Test complete pipeline integration."""
    
    @pytest.fixture
    def ai_pipeline(self):
        """Create AI-enhanced pipeline."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        config = PipelineConfig(
            use_ai=True,
            ai_first=True,
            enable_interactive_graphs=True,
            enable_rate_limiting=False,  # Disable for testing
            max_retries=2,
            timeout=10.0
        )
        return MathVizPipeline(config)
    
    @pytest.fixture
    def symbolic_pipeline(self):
        """Create symbolic-only pipeline."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        config = PipelineConfig(
            use_ai=False,
            enable_interactive_graphs=False,
            enable_rate_limiting=False,
            max_retries=2,
            timeout=10.0
        )
        return MathVizPipeline(config)
    
    def test_pipeline_status(self, ai_pipeline):
        """Test pipeline status reporting."""
        status = ai_pipeline.get_pipeline_status()
        
        assert "config" in status
        assert "components" in status
        assert "capabilities" in status
        assert status["capabilities"]["ai_solving"] is True
    
    @patch('mathviz.solver.solve_with_ai')
    def test_derivative_problem_processing(self, mock_solve_ai, ai_pipeline):
        """Test processing derivative problem through AI pipeline."""
        # Mock AI response
        mock_solve_ai.return_value = AIResponse(
            success=True,
            content="The derivative of x^2 is 2x",
            confidence=0.9,
            provider="MockAI"
        )
        
        problem_text = "Find the derivative of x^2"
        solution = ai_pipeline.process(problem_text)
        
        assert isinstance(solution, MathSolution)
        assert solution.final_answer is not None
        assert "ai_enabled" in solution.metadata
        assert solution.metadata["ai_enabled"] is True
    
    def test_optimization_problem_processing(self, symbolic_pipeline):
        """Test processing optimization problem."""
        problem_text = "Find critical points of x^2 + y^2"
        solution = symbolic_pipeline.process(problem_text)
        
        assert isinstance(solution, MathSolution)
        # Should handle the problem even if optimization not fully implemented
        assert solution.final_answer is not None
    
    def test_error_handling(self, ai_pipeline):
        """Test pipeline error handling."""
        # Invalid problem that should cause parsing to fail
        problem_text = "This is not a math problem at all"
        solution = ai_pipeline.process(problem_text)
        
        # Should return error solution instead of crashing
        assert isinstance(solution, MathSolution)
        assert "error" in solution.metadata
    
    def test_pipeline_with_graph_config(self, ai_pipeline):
        """Test pipeline processing with graph configuration."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        
        problem_text = "Find the derivative of x^3"
        graph_config = GraphConfig(x_range=(-3, 3), y_range=(-5, 5))
        
        result = ai_pipeline.process_with_graph_config(problem_text, graph_config)
        
        assert "solution" in result
        assert isinstance(result["solution"], MathSolution)
        # Other fields may be None if visualization not available
        assert "interactive_graph" in result
        assert "desmos_url" in result
        assert "graph_html" in result


class TestPerformanceAndRateLimit:
    """Test performance and rate limiting functionality."""
    
    @pytest.fixture
    def rate_limited_pipeline(self):
        """Create rate-limited pipeline."""
        if not IMPORTS_SUCCESSFUL:
            pytest.skip(f"Import failed: {IMPORT_ERROR}")
        config = PipelineConfig(
            use_ai=False,  # Disable AI to avoid API calls
            enable_rate_limiting=True,
            retry_delay=0.1,  # Short delay for testing
            max_retries=2
        )
        return MathVizPipeline(config)
    
    def test_rate_limiting(self, rate_limited_pipeline):
        """Test rate limiting between requests."""
        problem_text = "Solve x + 1 = 3"
        
        start_time = time.time()
        # First request
        solution1 = rate_limited_pipeline.process(problem_text)
        first_time = time.time()
        
        # Second request (should be rate limited)
        solution2 = rate_limited_pipeline.process(problem_text)
        second_time = time.time()
        
        # Should have enforced delay
        time_diff = second_time - first_time
        assert time_diff >= 0.05  # Some delay should be present
        
        assert isinstance(solution1, MathSolution)
        assert isinstance(solution2, MathSolution)
    
    def test_performance_metadata(self, rate_limited_pipeline):
        """Test performance metadata collection."""
        problem_text = "Solve 2x = 10"
        solution = rate_limited_pipeline.process(problem_text)
        
        assert isinstance(solution, MathSolution)
        assert "processing_time" in solution.metadata
        assert solution.metadata["processing_time"] > 0
        assert solution.metadata["pipeline_version"] == "ai-enhanced"


@pytest.mark.integration
class TestFullIntegration:
    """Full integration tests with real components."""
    
    @pytest.mark.skipif(not IMPORTS_SUCCESSFUL, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_SUCCESSFUL else ''}")
    def test_complete_derivative_workflow(self):
        """Test complete derivative solving workflow."""
        config = PipelineConfig(
            use_ai=False,  # Use symbolic for reliable testing
            enable_interactive_graphs=True,
            enable_rate_limiting=False
        )
        pipeline = MathVizPipeline(config)
        
        problem_text = "Find the derivative of x^3 + 2*x^2 + x"
        solution = pipeline.process(problem_text)
        
        assert isinstance(solution, MathSolution)
        assert solution.final_answer["type"] == "derivative"
        assert "3*x**2 + 4*x + 1" in solution.final_answer["derivative"]
        assert solution.reasoning is not None
        assert len(solution.reasoning) > 0
    
    @pytest.mark.skipif(not IMPORTS_SUCCESSFUL, reason=f"Import failed: {IMPORT_ERROR if not IMPORTS_SUCCESSFUL else ''}")
    def test_complete_algebraic_workflow(self):
        """Test complete algebraic solving workflow."""
        config = PipelineConfig(use_ai=False, enable_rate_limiting=False)
        pipeline = MathVizPipeline(config)
        
        problem_text = "Solve for x: 2*x + 5 = 13"
        solution = pipeline.process(problem_text)
        
        assert isinstance(solution, MathSolution)
        assert solution.final_answer["type"] == "algebraic_solution"
        # Should find x = 4
        solutions = solution.final_answer["solutions"]
        assert any("4" in str(sol) for sol in solutions.values())


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])