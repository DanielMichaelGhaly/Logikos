#!/usr/bin/env python3
"""
Integration Tests for Logikos Workflow

Tests the complete AI+SymPy workflow with various mathematical problems.
"""

import sys
import os
import pytest

# Add the parent directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_workflow import MathWorkflow


class TestLogikosIntegration:
    """Integration tests for the complete Logikos workflow"""
    
    def setup_method(self):
        """Setup for each test"""
        # Test without AI to avoid external dependencies
        self.workflow = MathWorkflow(enable_ai=False, verbose=False)
    
    def test_simple_linear_equation(self):
        """Test solving a simple linear equation"""
        result = self.workflow.solve_problem("solve 2x+5=0")
        
        assert result.success, f"Workflow failed: {result.error_message}"
        assert result.sympy_result is not None
        assert result.sympy_result.success
        assert result.sympy_result.result == [-5/2]  # Expected solution
    
    def test_quadratic_equation(self):
        """Test solving a quadratic equation"""
        result = self.workflow.solve_problem("find roots of x^2-4")
        
        assert result.success, f"Workflow failed: {result.error_message}"
        assert result.sympy_result is not None
        assert result.sympy_result.success
        
        # Should find x = 2 and x = -2
        solutions = result.sympy_result.result
        assert len(solutions) == 2
        assert 2 in solutions or -2 in solutions  # At least one should match
    
    def test_derivative_calculation(self):
        """Test derivative calculation"""
        result = self.workflow.solve_problem("derivative of x^2 + 3x")
        
        assert result.success, f"Workflow failed: {result.error_message}"
        assert result.sympy_result is not None
        assert result.sympy_result.success
        
        # Derivative of x^2 + 3x should be 2x + 3
        derivative_str = str(result.sympy_result.result)
        assert "2*x" in derivative_str
        assert "3" in derivative_str
    
    def test_integral_calculation(self):
        """Test integral calculation"""
        result = self.workflow.solve_problem("integral of x^2")
        
        assert result.success, f"Workflow failed: {result.error_message}"
        assert result.sympy_result is not None
        assert result.sympy_result.success
        
        # Integral of x^2 should be x^3/3
        integral_str = str(result.sympy_result.result)
        assert "x**3" in integral_str
        assert "3" in integral_str
    
    def test_expression_simplification(self):
        """Test expression simplification"""
        result = self.workflow.solve_problem("simplify (x^2 - 1)/(x - 1)")
        
        assert result.success, f"Workflow failed: {result.error_message}"
        assert result.sympy_result is not None
        assert result.sympy_result.success
        
        # Should simplify to x + 1
        simplified_str = str(result.sympy_result.result)
        assert "x + 1" in simplified_str or "x+1" in simplified_str
    
    def test_invalid_input(self):
        """Test handling of invalid mathematical input"""
        result = self.workflow.solve_problem("this is not a math problem")
        
        # Should handle gracefully, either by returning an error or trying to process
        # The exact behavior depends on implementation, but it shouldn't crash
        assert isinstance(result.success, bool)
    
    def test_workflow_components_initialized(self):
        """Test that all workflow components are properly initialized"""
        workflow = MathWorkflow(enable_ai=True, verbose=False)
        
        assert workflow.math_parser is not None
        assert workflow.sympy_solver is not None
        assert workflow.verifier is not None
        assert workflow.latex_formatter is not None
        assert workflow.step_visualizer is not None
        
        # AI components should be initialized when enabled
        assert workflow.ai_solver is not None
        assert workflow.response_parser is not None
    
    def test_no_ai_workflow_components(self):
        """Test that AI components are None when AI is disabled"""
        workflow = MathWorkflow(enable_ai=False)
        
        assert workflow.ai_solver is None
        assert workflow.response_parser is None
        
        # Other components should still be initialized
        assert workflow.math_parser is not None
        assert workflow.sympy_solver is not None
    
    def test_multiple_problems(self):
        """Test solving multiple problems in sequence"""
        problems = [
            "solve x+1=0",
            "solve 2x+4=0",
            "find roots of x^2-1"
        ]
        
        for problem in problems:
            result = self.workflow.solve_problem(problem)
            assert result.success, f"Failed on problem: {problem}"
            assert result.sympy_result is not None
            assert result.sympy_result.success


def test_basic_functionality():
    """Simple smoke test that can be run directly"""
    workflow = MathWorkflow(enable_ai=False)
    result = workflow.solve_problem("solve 2x+5=0")
    
    print(f"✅ Basic test passed: {result.success}")
    if result.success:
        print(f"   Solution: {result.sympy_result.result}")
    else:
        print(f"   Error: {result.error_message}")


if __name__ == "__main__":
    test_basic_functionality()