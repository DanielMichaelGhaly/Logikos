"""
Comprehensive visualization module supporting LaTeX, HTML, interactive graphs, Plotly visualization, and animation preparation.
Integrates with graph visualization APIs for enhanced mathematical visualization.
"""

import re
import json
from typing import Dict, List, Any, Optional, Tuple
import sympy as sp
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

from .trace import StepTrace, Step
from .schemas import MathSolution
try:
    from .graph_visualizer import GraphVisualizer, GraphConfig, VisualizationResult
    from .advanced_calculus import create_3d_function_plot, create_gradient_descent_animation
    GRAPH_VISUALIZATION_AVAILABLE = True
except ImportError:
    GRAPH_VISUALIZATION_AVAILABLE = False
    print("Graph visualization not available")

logger = logging.getLogger(__name__)


class Visualizer:
    """Comprehensive visualization generator for mathematical solutions with interactive graph capabilities."""

    def __init__(self, enable_interactive_graphs: bool = True):
        """Initialize with visualization templates, settings, and graph visualization capabilities."""
        self.enable_interactive_graphs = enable_interactive_graphs and GRAPH_VISUALIZATION_AVAILABLE
        
        if self.enable_interactive_graphs:
            self.graph_visualizer = GraphVisualizer()
        else:
            self.graph_visualizer = None
            
        logger.info(f"Visualizer initialized - Interactive graphs: {self.enable_interactive_graphs}")
        self.latex_templates = {
            'equation_solve': r'''
\begin{{align}}
{steps}
\end{{align}}
''',
            'calculus': r'''
\begin{{align}}
{expression} &= {result} \\
\text{{Explanation: }} &\text{{{explanation}}}
\end{{align}}
''',
            'step_by_step': r'''
\begin{{alignat}}{{2}}
{steps}
\end{{alignat}}
'''
        }
        
        # Color schemes for different types of visualizations
        self.color_schemes = {
            'steps': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
            'function': '#1f77b4',
            'derivative': '#ff7f0e',
            'integral': '#2ca02c'
        }

    def generate_latex(self, trace: StepTrace) -> str:
        """Generate comprehensive LaTeX output from solution trace."""
        if not trace.steps:
            return r"\text{No solution steps available}"
        
        problem_type = self._identify_problem_type(trace)
        
        if problem_type == 'algebraic':
            return self._generate_algebraic_latex(trace)
        elif problem_type in ['derivative', 'integral']:
            return self._generate_calculus_latex(trace)
        else:
            return self._generate_general_latex(trace)

    def generate_html(self, trace: StepTrace) -> str:
        """Generate comprehensive HTML output with CSS styling."""
        if not trace.steps:
            return "<div class='no-steps'>No solution steps available</div>"
        
        html_parts = [
            "<div class='math-visualization'>",
            self._generate_css_styles(),
            "<div class='solution-container'>"
        ]
        
        # Add step-by-step visualization
        html_parts.append(self._generate_steps_html(trace))
        
        # Add mathematical expressions with MathJax
        html_parts.append(self._generate_mathjax_html(trace))
        
        # Add interactive elements if applicable
        html_parts.append(self._generate_interactive_elements(trace))
        
        html_parts.extend(["</div>", "</div>"])
        
        return "\n".join(html_parts)
    
    def generate_interactive_graph(self, trace: StepTrace, solution: Optional[MathSolution] = None,
                                 config: Optional[GraphConfig] = None) -> Optional[VisualizationResult]:
        """Generate interactive graph visualization using graph visualization APIs."""
        if not self.enable_interactive_graphs or not self.graph_visualizer:
            logger.warning("Interactive graph visualization not available")
            return None
        
        try:
            problem_type = self._identify_problem_type(trace)
            expressions = self._extract_expressions_from_trace(trace, solution)
            
            if not expressions:
                logger.warning("No expressions found for graph visualization")
                return None
            
            if config is None:
                config = GraphConfig()
            
            # Generate appropriate visualization based on problem type
            if problem_type == 'derivative':
                return self._generate_derivative_graph(expressions, config)
            elif problem_type == 'integral':
                return self._generate_integral_graph(expressions, config)
            elif len(expressions) == 1 and self._is_multivariable_function(expressions[0]):
                return self._generate_3d_function_graph(expressions[0], config)
            else:
                return self._generate_function_graph(expressions, config)
                
        except Exception as e:
            logger.error(f"Error generating interactive graph: {e}")
            return None
    
    def generate_optimization_graph(self, expression: str, critical_points: List[Tuple[float, float]] = None,
                                  config: Optional[GraphConfig] = None) -> Optional[VisualizationResult]:
        """Generate interactive graph for optimization problems."""
        if not self.enable_interactive_graphs or not self.graph_visualizer:
            return None
        
        try:
            if config is None:
                config = GraphConfig()
            
            return self.graph_visualizer.visualize_optimization(expression, critical_points, config)
            
        except Exception as e:
            logger.error(f"Error generating optimization graph: {e}")
            return None
    
    def generate_desmos_url(self, expressions: List[str], config: Optional[GraphConfig] = None) -> Optional[str]:
        """Generate Desmos URL for expressions."""
        if not self.enable_interactive_graphs or not self.graph_visualizer:
            return None
        
        try:
            result = self.graph_visualizer.visualize_functions(expressions, config, "desmos")
            return result.graph_url if result.success else None
        except Exception as e:
            logger.error(f"Error generating Desmos URL: {e}")
            return None
    
    def _extract_expressions_from_trace(self, trace: StepTrace, solution: Optional[MathSolution] = None) -> List[str]:
        """Extract mathematical expressions suitable for graphing from trace."""
        expressions = []
        
        # Look for expressions in the trace steps
        for step in trace.steps:
            # For derivative problems, include both original and derivative
            if step.operation == 'differentiate':
                # Try to extract clean expressions
                before_expr = self._clean_expression_for_graphing(step.expression_before)
                after_expr = self._clean_expression_for_graphing(step.expression_after)
                
                if before_expr:
                    expressions.append(before_expr)
                if after_expr:
                    expressions.append(after_expr)
            
            # For parsing steps, look for main expressions
            elif step.operation in ['parse_expression', 'symbolic_conversion']:
                expr = self._clean_expression_for_graphing(step.expression_after)
                if expr:
                    expressions.append(expr)
        
        # If no expressions found in steps, try to extract from solution
        if not expressions and solution:
            if hasattr(solution, 'final_answer'):
                if isinstance(solution.final_answer, dict):
                    for key, value in solution.final_answer.items():
                        if 'expression' in key.lower() or 'derivative' in key.lower():
                            expr = self._clean_expression_for_graphing(str(value))
                            if expr:
                                expressions.append(expr)
        
        # Remove duplicates while preserving order
        unique_expressions = []
        for expr in expressions:
            if expr not in unique_expressions:
                unique_expressions.append(expr)
        
        return unique_expressions
    
    def _clean_expression_for_graphing(self, expression: str) -> Optional[str]:
        """Clean and validate expression for graphing."""
        if not expression or not isinstance(expression, str):
            return None
        
        # Remove common text patterns that aren't mathematical expressions
        skip_patterns = [
            'step', 'solution', 'equation', 'original', 'parsing', 'converting', 
            'applying', 'simplifying', '=', 'solve', 'find', 'derivative of'
        ]
        
        cleaned = expression.lower().strip()
        for pattern in skip_patterns:
            if pattern in cleaned and len(cleaned) < 50:  # Likely descriptive text
                return None
        
        # Extract mathematical expressions from text
        # Look for patterns like "f(x) = x^2" or just "x^2"
        import re
        
        # Pattern for function definition
        func_pattern = r'f\(x\)\s*=\s*(.+)'
        match = re.search(func_pattern, expression, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Pattern for expressions that look mathematical
        math_pattern = r'([x\^\+\-\*\/\(\)\d\w\s]+)'
        if re.search(r'[x\^\+\-\*\/]', expression):  # Contains mathematical operators
            # Try to clean up the expression
            cleaned = re.sub(r'\s+', ' ', expression)  # Normalize whitespace
            if len(cleaned) < 100 and 'x' in cleaned:  # Reasonable length and contains variable
                return cleaned
        
        return None
    
    def _is_multivariable_function(self, expression: str) -> bool:
        """Check if expression contains multiple variables for 3D plotting."""
        try:
            expr = sp.sympify(expression)
            variables = expr.free_symbols
            return len(variables) >= 2
        except:
            # Simple heuristic check
            return 'y' in expression.lower() and 'x' in expression.lower()
    
    def _generate_derivative_graph(self, expressions: List[str], config: GraphConfig) -> Optional[VisualizationResult]:
        """Generate graph for derivative problems."""
        if len(expressions) >= 2:
            # Assume first is original function, second is derivative
            original_expr = expressions[0]
            return self.graph_visualizer.visualize_derivative(original_expr, config=config)
        elif len(expressions) == 1:
            # Try to compute derivative and visualize
            return self.graph_visualizer.visualize_derivative(expressions[0], config=config)
        else:
            return None
    
    def _generate_integral_graph(self, expressions: List[str], config: GraphConfig) -> Optional[VisualizationResult]:
        """Generate graph for integral problems."""
        if expressions:
            # For integrals, visualize the integrand
            return self.graph_visualizer.visualize_function(expressions[0], config)
        return None
    
    def _generate_function_graph(self, expressions: List[str], config: GraphConfig) -> Optional[VisualizationResult]:
        """Generate graph for general functions."""
        return self.graph_visualizer.visualize_functions(expressions, config)
    
    def _generate_3d_function_graph(self, expression: str, config: GraphConfig) -> Optional[VisualizationResult]:
        """Generate 3D graph for multivariable functions."""
        return self.graph_visualizer.visualize_3d_function(expression, config=config)

    def generate_plotly_figure(self, trace: StepTrace, solution: Optional[MathSolution] = None) -> Optional[go.Figure]:
        """Generate interactive Plotly visualization."""
        try:
            problem_type = self._identify_problem_type(trace)
            
            if problem_type == 'algebraic':
                return self._create_algebraic_plot(trace, solution)
            elif problem_type == 'derivative':
                return self._create_derivative_plot(trace, solution)
            elif problem_type == 'integral':
                return self._create_integral_plot(trace, solution)
            else:
                return self._create_generic_plot(trace)
                
        except Exception as e:
            # Return error visualization
            fig = go.Figure()
            fig.add_annotation(
                text=f"Visualization not available: {str(e)}",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig

    def generate_animation_data(self, trace: StepTrace) -> Dict[str, Any]:
        """Generate data structure for animation (e.g., for Manim)."""
        animation_data = {
            'type': self._identify_problem_type(trace),
            'steps': [],
            'timeline': [],
            'visual_elements': []
        }
        
        for i, step in enumerate(trace.steps):
            step_data = {
                'step_id': step.step_id,
                'operation': step.operation,
                'before': step.expression_before,
                'after': step.expression_after,
                'duration': 2.0,  # Default duration in seconds
                'animation_type': self._get_animation_type(step.operation),
                'visual_cues': self._get_visual_cues(step)
            }
            animation_data['steps'].append(step_data)
            animation_data['timeline'].append(i * 2.0)
        
        return animation_data

    def _identify_problem_type(self, trace: StepTrace) -> str:
        """Identify the type of problem from the trace."""
        operations = [step.operation for step in trace.steps]
        
        if any(op == 'differentiate' for op in operations):
            return 'derivative'
        elif any(op == 'integrate' for op in operations):
            return 'integral'
        elif any(op in ['solve', 'isolate'] for op in operations):
            return 'algebraic'
        else:
            return 'general'

    def _generate_algebraic_latex(self, trace: StepTrace) -> str:
        """Generate LaTeX for algebraic problem solutions."""
        latex_steps = []
        
        for i, step in enumerate(trace.steps):
            if step.operation in ['solve', 'isolate', 'simplify']:
                # Clean up expressions for LaTeX
                before = self._format_for_latex(step.expression_before)
                after = self._format_for_latex(step.expression_after)
                
                if step.operation == 'solve':
                    latex_steps.append(f"{before} &\\Rightarrow {after}")
                else:
                    latex_steps.append(f"{before} &= {after}")
                
                # Add step annotation
                if step.justification:
                    latex_steps.append(f"&\\quad \\text{{({step.justification})}}")
        
        if latex_steps:
            return self.latex_templates['step_by_step'].format(
                steps=" \\\\ ".join(latex_steps)
            )
        else:
            return r"\text{No algebraic steps to display}"

    def _generate_calculus_latex(self, trace: StepTrace) -> str:
        """Generate LaTeX for calculus problems."""
        # Find the main calculus step
        calc_step = None
        for step in trace.steps:
            if step.operation in ['differentiate', 'integrate']:
                calc_step = step
                break
        
        if not calc_step:
            return r"\text{No calculus operation found}"
        
        expression = self._format_for_latex(calc_step.expression_before)
        result = self._format_for_latex(calc_step.expression_after)
        
        if calc_step.operation == 'differentiate':
            return rf"""
\begin{{align}}
\frac{{d}}{{dx}}\left({expression}\right) &= {result} \\
\text{{Method: }} &\text{{Differentiation rules applied}}
\end{{align}}
"""
        else:  # integrate
            return rf"""
\begin{{align}}
\int {expression} \, dx &= {result} + C \\
\text{{Method: }} &\text{{Integration rules applied}}
\end{{align}}
"""

    def _generate_general_latex(self, trace: StepTrace) -> str:
        """Generate LaTeX for general problems."""
        if trace.final_state:
            final_formatted = self._format_for_latex(trace.final_state)
            return rf"""
\begin{{align}}
\text{{Result: }} &{final_formatted}
\end{{align}}
"""
        else:
            return r"\text{Solution completed}"

    def _format_for_latex(self, expression: str) -> str:
        """Format mathematical expression for LaTeX rendering."""
        if not expression:
            return ""
        
        # Basic formatting for common mathematical notation
        formatted = expression
        
        # Replace common patterns
        replacements = {
            '**': '^',
            'sqrt(': r'\sqrt{',
            'sin(': r'\sin(',
            'cos(': r'\cos(',
            'tan(': r'\tan(',
            'log(': r'\log(',
            'ln(': r'\ln(',
            'pi': r'\pi',
            'inf': r'\infty',
            '<=': r'\leq',
            '>=': r'\geq',
            '!=': r'\neq'
        }
        
        for pattern, replacement in replacements.items():
            formatted = formatted.replace(pattern, replacement)
        
        # Handle fractions (simple cases)
        fraction_pattern = r'([^/]+)/([^/\s]+)'
        formatted = re.sub(fraction_pattern, r'\\frac{\1}{\2}', formatted)
        
        return formatted

    def _generate_css_styles(self) -> str:
        """Generate CSS styles for HTML visualization."""
        return """
<style>
.math-visualization {
    font-family: 'Times New Roman', serif;
    max-width: 800px;
    margin: 0 auto;
    padding: 20px;
    background-color: #fafafa;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.solution-container {
    background: white;
    padding: 20px;
    border-radius: 6px;
    margin-bottom: 20px;
}

.step-visualization {
    margin: 15px 0;
    padding: 15px;
    border-left: 4px solid #1f77b4;
    background-color: #f8f9fa;
}

.step-number {
    font-weight: bold;
    color: #1f77b4;
    margin-bottom: 5px;
}

.expression {
    font-family: 'Courier New', monospace;
    background-color: #e9ecef;
    padding: 8px 12px;
    border-radius: 4px;
    display: inline-block;
    margin: 5px;
}

.step-transition {
    text-align: center;
    font-size: 24px;
    color: #28a745;
    margin: 10px 0;
}

.final-answer {
    background-color: #d4edda;
    border: 2px solid #28a745;
    border-radius: 6px;
    padding: 15px;
    margin-top: 20px;
    text-align: center;
}

.mathjax-container {
    margin: 20px 0;
    text-align: center;
}
</style>
"""

    def _generate_steps_html(self, trace: StepTrace) -> str:
        """Generate HTML for step-by-step visualization."""
        html_parts = ["<div class='steps-visualization'>"]
        
        for i, step in enumerate(trace.steps):
            html_parts.append(f"""
<div class='step-visualization'>
    <div class='step-number'>Step {i+1}: {step.operation.replace('_', ' ').title()}</div>
    <div class='step-content'>
        <div class='expression'>{step.expression_before}</div>
        <div class='step-transition'>↓</div>
        <div class='expression'>{step.expression_after}</div>
        <div class='step-justification'>{step.justification}</div>
    </div>
</div>
""")
        
        html_parts.append("</div>")
        
        if trace.final_state:
            html_parts.append(f"""
<div class='final-answer'>
    <strong>Final Answer:</strong> {trace.final_state}
</div>
""")
        
        return "\n".join(html_parts)

    def _generate_mathjax_html(self, trace: StepTrace) -> str:
        """Generate MathJax-enabled HTML for mathematical expressions."""
        latex_content = self.generate_latex(trace)
        
        return f"""
<div class='mathjax-container'>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <div class='latex-display'>
        $$
        {latex_content}
        $$
    </div>
</div>
"""

    def _generate_interactive_elements(self, trace: StepTrace) -> str:
        """Generate interactive HTML elements."""
        return """
<div class='interactive-controls'>
    <button onclick='toggleSteps()' class='btn btn-primary'>Toggle Step Details</button>
    <button onclick='exportLatex()' class='btn btn-secondary'>Export LaTeX</button>
</div>
<script>
function toggleSteps() {
    const steps = document.querySelectorAll('.step-visualization');
    steps.forEach(step => {
        step.style.display = step.style.display === 'none' ? 'block' : 'none';
    });
}

function exportLatex() {
    const latex = document.querySelector('.latex-display').textContent;
    navigator.clipboard.writeText(latex).then(() => {
        alert('LaTeX copied to clipboard!');
    });
}
</script>
"""

    def _create_algebraic_plot(self, trace: StepTrace, solution: Optional[MathSolution]) -> go.Figure:
        """Create Plotly visualization for algebraic problems."""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['Solution Steps', 'Verification'],
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}]]
        )
        
        # Steps visualization
        step_numbers = list(range(1, len(trace.steps) + 1))
        step_names = [step.operation.replace('_', ' ').title() for step in trace.steps]
        
        fig.add_trace(
            go.Scatter(
                x=step_numbers,
                y=[1] * len(step_numbers),
                mode='markers+text',
                text=step_names,
                textposition='top center',
                marker=dict(size=15, color=self.color_schemes['steps'][:len(step_numbers)]),
                name='Solution Steps'
            ),
            row=1, col=1
        )
        
        # Try to create a verification plot if we have a solution
        if solution and solution.final_answer.get('solutions'):
            self._add_verification_plot(fig, solution, row=1, col=2)
        
        fig.update_layout(
            title='Algebraic Problem Solution Visualization',
            showlegend=False
        )
        
        return fig

    def _create_derivative_plot(self, trace: StepTrace, solution: Optional[MathSolution]) -> go.Figure:
        """Create Plotly visualization for derivative problems."""
        fig = go.Figure()
        
        # Try to extract and plot the original function and its derivative
        try:
            # Find the differentiation step
            diff_step = None
            for step in trace.steps:
                if step.operation == 'differentiate':
                    diff_step = step
                    break
            
            if diff_step:
                # Create a simple visualization
                x = np.linspace(-5, 5, 100)
                
                # This is a simplified example - in practice, you'd parse the expressions
                # For now, create a generic function plot
                y_original = x**2  # Placeholder
                y_derivative = 2*x  # Placeholder
                
                fig.add_trace(go.Scatter(
                    x=x, y=y_original,
                    mode='lines',
                    name='Original Function',
                    line=dict(color=self.color_schemes['function'])
                ))
                
                fig.add_trace(go.Scatter(
                    x=x, y=y_derivative,
                    mode='lines',
                    name='Derivative',
                    line=dict(color=self.color_schemes['derivative'])
                ))
            
            fig.update_layout(
                title='Function and Its Derivative',
                xaxis_title='x',
                yaxis_title='y'
            )
            
        except Exception:
            fig.add_annotation(
                text="Function visualization not available for this expression",
                x=0.5, y=0.5,
                xref="paper", yref="paper",
                showarrow=False
            )
        
        return fig

    def _create_integral_plot(self, trace: StepTrace, solution: Optional[MathSolution]) -> go.Figure:
        """Create Plotly visualization for integral problems."""
        fig = go.Figure()
        
        # Similar to derivative plot but for integration
        x = np.linspace(-5, 5, 100)
        y_integrand = 2*x  # Placeholder
        y_integral = x**2  # Placeholder
        
        fig.add_trace(go.Scatter(
            x=x, y=y_integrand,
            mode='lines',
            name='Integrand',
            line=dict(color=self.color_schemes['function'])
        ))
        
        fig.add_trace(go.Scatter(
            x=x, y=y_integral,
            mode='lines',
            name='Integral',
            line=dict(color=self.color_schemes['integral'])
        ))
        
        # Add area under curve for visual effect
        fig.add_trace(go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y_integrand, np.zeros(len(x))]),
            fill='tonexty',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Area Under Curve'
        ))
        
        fig.update_layout(
            title='Integration Visualization',
            xaxis_title='x',
            yaxis_title='y'
        )
        
        return fig

    def _create_generic_plot(self, trace: StepTrace) -> go.Figure:
        """Create a generic visualization for any problem type."""
        fig = go.Figure()
        
        # Create a simple step progression chart
        step_numbers = list(range(1, len(trace.steps) + 1))
        operations = [step.operation for step in trace.steps]
        
        fig.add_trace(go.Bar(
            x=step_numbers,
            y=[1] * len(step_numbers),
            text=operations,
            textposition='auto',
            marker_color=self.color_schemes['steps'][:len(step_numbers)]
        ))
        
        fig.update_layout(
            title='Solution Steps Overview',
            xaxis_title='Step Number',
            yaxis_title='Progress',
            showlegend=False
        )
        
        return fig

    def _add_verification_plot(self, fig: go.Figure, solution: MathSolution, row: int, col: int) -> None:
        """Add verification plot to existing figure."""
        # This would create a verification visualization
        # For now, just add a placeholder
        fig.add_trace(
            go.Scatter(
                x=[1], y=[1],
                mode='markers+text',
                text=['Solution Verified'],
                textposition='middle center',
                marker=dict(size=50, color='green', symbol='check'),
                name='Verification'
            ),
            row=row, col=col
        )

    def _get_animation_type(self, operation: str) -> str:
        """Get animation type for a given operation."""
        animation_types = {
            'parse_equation': 'highlight',
            'solve': 'transform',
            'differentiate': 'calculus_rule',
            'integrate': 'calculus_rule',
            'simplify': 'simplify',
            'substitute': 'substitute'
        }
        return animation_types.get(operation, 'default')

    def _get_visual_cues(self, step: Step) -> Dict[str, Any]:
        """Get visual cues for animation of a step."""
        return {
            'highlight_before': True,
            'transition_type': 'fade',
            'emphasis_color': '#ff7f0e',
            'duration': 2.0
        }
