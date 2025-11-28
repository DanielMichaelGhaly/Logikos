"""
Educational Visualization Integration

This module integrates with educational-friendly APIs and tools for visualizing
mathematical solutions in an interactive, student-friendly way.

Supported platforms:
- Desmos Graphing Calculator (for function plotting)
- GeoGebra (for geometry and interactive math)
- Function Plot.js (for embedded web visualizations)
- MathJax (for LaTeX rendering)
- Plotly (for interactive plots)

The goal is to make math visualization accessible and educational,
helping students understand step-by-step solutions through visual aids.
"""

import json
import urllib.parse
import requests
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import re
import sympy as sp
import numpy as np

from .schemas import MathProblem, MathSolution
from .trace import StepTrace


@dataclass
class VisualizationConfig:
    """Configuration for educational visualizations."""
    # Desmos settings
    desmos_enabled: bool = True
    desmos_colors: List[str] = None
    
    # GeoGebra settings
    geogebra_enabled: bool = True
    
    # Function Plot settings
    function_plot_enabled: bool = True
    plot_resolution: int = 1000
    
    # General settings
    interactive: bool = True
    show_grid: bool = True
    show_axes: bool = True
    x_range: Tuple[float, float] = (-10, 10)
    y_range: Tuple[float, float] = (-10, 10)
    
    def __post_init__(self):
        if self.desmos_colors is None:
            self.desmos_colors = [
                '#c74440',  # Red
                '#2d70b3',  # Blue
                '#388c46',  # Green
                '#6042a6',  # Purple
                '#000000',  # Black
                '#fa7e19'   # Orange
            ]


class DesmosIntegration:
    """Integration with Desmos Graphing Calculator."""
    
    def __init__(self):
        self.base_url = "https://www.desmos.com/calculator"
        
    def create_graph_url(self, expressions: List[str], config: VisualizationConfig) -> str:
        """Create a Desmos graph URL with the given expressions."""
        # Convert expressions to Desmos-compatible LaTeX
        desmos_expressions = []
        
        for i, expr in enumerate(expressions):
            try:
                # Convert SymPy expression to LaTeX
                sympy_expr = sp.sympify(expr)
                latex_expr = sp.latex(sympy_expr)
                
                # Get color for this expression
                color = config.desmos_colors[i % len(config.desmos_colors)]
                
                desmos_expressions.append({
                    "id": f"expr_{i}",
                    "latex": latex_expr,
                    "color": color
                })
                
            except Exception as e:
                print(f"Warning: Could not convert expression '{expr}' to LaTeX: {e}")
                continue
        
        # Create Desmos state
        state = {
            "version": 10,
            "randomSeed": "abc123",
            "graph": {
                "viewport": {
                    "xmin": config.x_range[0],
                    "ymin": config.y_range[0],
                    "xmax": config.x_range[1],
                    "ymax": config.y_range[1]
                }
            },
            "expressions": {
                "list": desmos_expressions
            }
        }
        
        # URL encode the state
        state_json = json.dumps(state, separators=(',', ':'))
        encoded_state = urllib.parse.quote(state_json)
        
        return f"{self.base_url}?embed=true&state={encoded_state}"
    
    def create_step_visualization(self, trace: StepTrace) -> Optional[str]:
        """Create visualization for step-by-step solution."""
        expressions = self._extract_expressions_from_trace(trace)
        
        if not expressions:
            return None
        
        config = VisualizationConfig()
        return self.create_graph_url(expressions, config)
    
    def _extract_expressions_from_trace(self, trace: StepTrace) -> List[str]:
        """Extract plottable expressions from solution trace."""
        expressions = []
        
        for step in trace.steps:
            # Look for expressions that can be plotted
            expr_before = step.expression_before  # Uses legacy property
            expr_after = step.expression_after    # Uses legacy property
            
            # Also check input and output states for expressions
            input_expr = step.input_state.get('expression', '') if hasattr(step, 'input_state') else ''
            output_expr = step.output_state.get('expression', '') if hasattr(step, 'output_state') else ''
            
            # Try to find function expressions
            for expr in [expr_before, expr_after, input_expr, output_expr]:
                if expr and self._is_plottable_expression(expr):
                    if expr not in expressions:  # Avoid duplicates
                        expressions.append(expr)
        
        return expressions
    
    def _is_plottable_expression(self, expr: str) -> bool:
        """Check if an expression can be plotted."""
        if not expr or len(expr.strip()) == 0:
            return False
        
        try:
            # Simple heuristics for plottable expressions
            expr_lower = expr.lower()
            
            # Skip constants and single numbers
            if expr.strip().replace('.', '').replace('-', '').isdigit():
                return False
            
            # Check for variables
            if 'x' in expr_lower or 'y' in expr_lower:
                # Try to parse with SymPy
                sympy_expr = sp.sympify(expr)
                return len(sympy_expr.free_symbols) > 0
                
        except:
            pass
        
        return False


class GeoGebraIntegration:
    """Integration with GeoGebra for interactive math."""
    
    def __init__(self):
        self.base_url = "https://www.geogebra.org/graphing"
    
    def create_commands(self, expressions: List[str]) -> List[str]:
        """Create GeoGebra commands for expressions."""
        commands = []
        
        for i, expr in enumerate(expressions):
            try:
                # Convert to GeoGebra syntax
                geogebra_expr = self._convert_to_geogebra(expr)
                commands.append(f"f_{i}(x) = {geogebra_expr}")
                
            except Exception as e:
                print(f"Warning: Could not convert '{expr}' to GeoGebra format: {e}")
                continue
        
        return commands
    
    def _convert_to_geogebra(self, expr: str) -> str:
        """Convert expression to GeoGebra syntax."""
        # Basic conversions
        geogebra_expr = expr.replace('^', '**')  # Power operator
        geogebra_expr = geogebra_expr.replace('**', '^')  # Back to GeoGebra format
        
        # Handle common functions
        function_mapping = {
            'ln': 'log',
            'exp': 'e^',
            'sqrt': 'sqrt'
        }
        
        for sympy_func, geogebra_func in function_mapping.items():
            geogebra_expr = geogebra_expr.replace(sympy_func, geogebra_func)
        
        return geogebra_expr


class FunctionPlotIntegration:
    """Integration with Function Plot.js for web visualizations."""
    
    def generate_html(self, expressions: List[str], config: VisualizationConfig) -> str:
        """Generate HTML with Function Plot visualization."""
        # Convert expressions to Function Plot format
        plot_data = []
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12', '#34495e']
        
        for i, expr in enumerate(expressions):
            try:
                # Convert to Function Plot syntax
                plot_expr = self._convert_to_function_plot(expr)
                color = colors[i % len(colors)]
                
                plot_data.append({
                    "fn": plot_expr,
                    "color": color,
                    "graphType": "polyline"
                })
                
            except Exception as e:
                print(f"Warning: Could not convert '{expr}' for Function Plot: {e}")
                continue
        
        # Generate HTML
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>MathViz Function Plot</title>
            <script src="https://unpkg.com/function-plot@1.23.3/dist/function-plot.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                #plot {{ width: 800px; height: 600px; margin: 0 auto; }}
                .info {{ text-align: center; margin: 20px; }}
                .controls {{ text-align: center; margin: 10px; }}
            </style>
        </head>
        <body>
            <div class="info">
                <h2>Mathematical Visualization</h2>
                <p>Interactive plot showing: {', '.join(expressions)}</p>
            </div>
            <div id="plot"></div>
            <div class="controls">
                <p>Drag to pan, scroll to zoom. Double-click to reset view.</p>
            </div>
            
            <script>
                try {{
                    functionPlot({{
                        target: '#plot',
                        width: 800,
                        height: 600,
                        grid: {str(config.show_grid).lower()},
                        xAxis: {{
                            domain: [{config.x_range[0]}, {config.x_range[1]}]
                        }},
                        yAxis: {{
                            domain: [{config.y_range[0]}, {config.y_range[1]}]
                        }},
                        data: {json.dumps(plot_data)}
                    }});
                }} catch (error) {{
                    document.getElementById('plot').innerHTML = 
                        '<p style="color: red; text-align: center;">Error rendering plot: ' + error.message + '</p>';
                }}
            </script>
        </body>
        </html>
        """
        
        return html_template.strip()
    
    def _convert_to_function_plot(self, expr: str) -> str:
        """Convert expression to Function Plot syntax."""
        # Handle power operator
        plot_expr = expr.replace('**', '^')
        
        # Handle common functions
        function_mapping = {
            'ln': 'log',
            'exp': 'exp'
        }
        
        for sympy_func, plot_func in function_mapping.items():
            plot_expr = plot_expr.replace(sympy_func, plot_func)
        
        return plot_expr


class EducationalVisualizer:
    """Main educational visualization coordinator."""
    
    def __init__(self, config: VisualizationConfig = None):
        self.config = config or VisualizationConfig()
        self.desmos = DesmosIntegration()
        self.geogebra = GeoGebraIntegration()
        self.function_plot = FunctionPlotIntegration()
    
    def create_solution_visualization(self, solution: MathSolution) -> Dict[str, Any]:
        """Create comprehensive visualization for a solution."""
        result = {
            "success": False,
            "visualizations": {},
            "expressions": [],
            "metadata": {}
        }
        
        # Extract expressions from the solution
        expressions = self._extract_expressions_from_solution(solution)
        
        if not expressions:
            result["metadata"]["warning"] = "No plottable expressions found"
            return result
        
        result["expressions"] = expressions
        result["success"] = True
        
        # Generate different visualization formats
        try:
            # Desmos integration
            if self.config.desmos_enabled:
                desmos_url = self.desmos.create_graph_url(expressions, self.config)
                result["visualizations"]["desmos"] = {
                    "url": desmos_url,
                    "expressions": expressions,
                    "type": "interactive_graph"
                }
            
            # GeoGebra integration
            if self.config.geogebra_enabled:
                geogebra_commands = self.geogebra.create_commands(expressions)
                result["visualizations"]["geogebra"] = {
                    "commands": geogebra_commands,
                    "expressions": expressions,
                    "type": "interactive_geometry"
                }
            
            # Function Plot HTML
            if self.config.function_plot_enabled:
                html_plot = self.function_plot.generate_html(expressions, self.config)
                result["visualizations"]["function_plot"] = {
                    "html": html_plot,
                    "expressions": expressions,
                    "type": "embedded_plot"
                }
            
            result["metadata"].update({
                "expression_count": len(expressions),
                "visualization_types": list(result["visualizations"].keys()),
                "config": self.config.__dict__
            })
            
        except Exception as e:
            result["success"] = False
            result["metadata"]["error"] = str(e)
        
        return result
    
    def create_step_by_step_visualization(self, trace: StepTrace) -> Dict[str, Any]:
        """Create visualizations showing step-by-step progression."""
        result = {
            "success": False,
            "step_visualizations": [],
            "expressions_by_step": {},
            "metadata": {}
        }
        
        try:
            # Extract expressions for each step
            for i, step in enumerate(trace.steps):
                step_expressions = []
                
                # Try to extract plottable expressions from this step
                for expr_field in ['expression_before', 'expression_after']:
                    expr = getattr(step, expr_field, '')
                    if self._is_plottable(expr):
                        step_expressions.append(expr)
                
                if step_expressions:
                    result["expressions_by_step"][f"step_{i}"] = step_expressions
                    
                    # Create visualization for this step
                    if self.config.desmos_enabled:
                        step_viz = {
                            "step_id": step.step_id,
                            "operation": step.operation,
                            "expressions": step_expressions,
                            "desmos_url": self.desmos.create_graph_url(step_expressions, self.config)
                        }
                        result["step_visualizations"].append(step_viz)
            
            result["success"] = len(result["step_visualizations"]) > 0
            result["metadata"]["total_steps"] = len(trace.steps)
            result["metadata"]["visualizable_steps"] = len(result["step_visualizations"])
            
        except Exception as e:
            result["metadata"]["error"] = str(e)
        
        return result
    
    def _extract_expressions_from_solution(self, solution: MathSolution) -> List[str]:
        """Extract plottable expressions from solution."""
        expressions = []
        
        # Check the problem text for expressions
        problem_text = solution.problem.problem_text
        if self._contains_plottable_content(problem_text):
            extracted = self._extract_from_text(problem_text)
            expressions.extend(extracted)
        
        # Check solution steps
        for step_dict in solution.solution_steps:
            for field in ['expression_before', 'expression_after']:
                expr = step_dict.get(field, '')
                if self._is_plottable(expr):
                    if expr not in expressions:
                        expressions.append(expr)
        
        # Check final answer
        final_answer = solution.final_answer
        if isinstance(final_answer, dict):
            for key, value in final_answer.items():
                if isinstance(value, str) and self._is_plottable(value):
                    if value not in expressions:
                        expressions.append(value)
        
        return expressions[:5]  # Limit to 5 expressions for clarity
    
    def _extract_from_text(self, text: str) -> List[str]:
        """Extract mathematical expressions from text."""
        expressions = []
        
        # Look for expressions after keywords
        patterns = [
            r"differentiate[:\s]+([^,\.\n]+)",
            r"derivative of[:\s]+([^,\.\n]+)",
            r"integrate[:\s]+([^,\.\n]+)",
            r"integral of[:\s]+([^,\.\n]+)",
            r"plot[:\s]+([^,\.\n]+)",
            r"graph[:\s]+([^,\.\n]+)",
            r"f\(x\)\s*=\s*([^,\.\n]+)",
            r"y\s*=\s*([^,\.\n]+)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                expr = match.strip()
                if self._is_plottable(expr):
                    expressions.append(expr)
        
        return expressions
    
    def _contains_plottable_content(self, text: str) -> bool:
        """Check if text contains plottable mathematical content."""
        keywords = [
            'differentiate', 'derivative', 'integrate', 'integral',
            'plot', 'graph', 'function', 'f(x)', 'y =', 'x^', 'sin', 'cos'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in keywords)
    
    def _is_plottable(self, expr: str) -> bool:
        """Check if expression can be plotted."""
        if not expr or len(expr.strip()) < 2:
            return False
        
        try:
            # Quick checks
            if expr.strip().replace('.', '').replace('-', '').isdigit():
                return False
            
            # Must contain variable
            if 'x' not in expr.lower():
                return False
            
            # Try to parse
            sympy_expr = sp.sympify(expr)
            return len(sympy_expr.free_symbols) > 0
            
        except:
            return False
    
    def get_student_friendly_url(self, expressions: List[str]) -> str:
        """Get the most student-friendly visualization URL."""
        # Prefer Desmos for its educational focus
        if self.config.desmos_enabled and expressions:
            return self.desmos.create_graph_url(expressions, self.config)
        
        return None
    
    def create_educational_summary(self, solution: MathSolution) -> Dict[str, str]:
        """Create educational summary with visualization links."""
        viz_result = self.create_solution_visualization(solution)
        
        summary = {
            "problem_type": solution.problem.problem_type,
            "visualization_available": viz_result["success"],
            "educational_notes": []
        }
        
        if viz_result["success"]:
            summary["primary_visualization"] = self.get_student_friendly_url(viz_result["expressions"])
            
            # Add educational context based on problem type
            if solution.problem.problem_type == "calculus":
                summary["educational_notes"].append(
                    "📊 Visualization shows the function and its derivative/integral relationship"
                )
            elif solution.problem.problem_type == "algebraic":
                summary["educational_notes"].append(
                    "📊 Visualization helps see where the function crosses the x-axis (roots)"
                )
            
            summary["expressions"] = viz_result["expressions"]
        
        return summary