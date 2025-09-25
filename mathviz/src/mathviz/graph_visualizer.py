"""
Graph Visualization API Integration for MathViz
===============================================

Integration with free graph visualization services including:
- Desmos API for interactive mathematical graphs
- Plotly for local visualization
- GeoGebra API integration
- FunctionPlot.js integration via HTML generation
"""

import json
import requests
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import base64
import urllib.parse
import numpy as np
import re
import sympy as sp
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging

logger = logging.getLogger(__name__)

@dataclass
class GraphConfig:
    """Configuration for graph visualization"""
    x_range: Tuple[float, float] = (-10, 10)
    y_range: Tuple[float, float] = (-10, 10)
    resolution: int = 1000
    interactive: bool = True
    show_grid: bool = True
    show_axes: bool = True
    theme: str = "default"  # "default", "dark", "minimal"

@dataclass
class VisualizationResult:
    """Result of graph visualization"""
    success: bool
    graph_url: Optional[str] = None
    graph_html: Optional[str] = None
    graph_image: Optional[bytes] = None
    interactive_url: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class DesmosAPI:
    """Integration with Desmos graphing calculator API"""
    
    def __init__(self):
        self.base_url = "https://www.desmos.com/calculator"
        self.api_url = "https://www.desmos.com/api/v1.6/calculator"
    
    def create_graph_url(self, expressions: List[str], config: GraphConfig = None) -> VisualizationResult:
        """Create Desmos graph URL for given expressions"""
        try:
            if config is None:
                config = GraphConfig()
            
            # Format expressions for Desmos
            desmos_expressions = []
            for i, expr in enumerate(expressions):
                # Convert from SymPy/Python format to Desmos format
                desmos_expr = self._convert_to_desmos_format(expr)
                desmos_expressions.append({
                    "id": f"expr_{i}",
                    "latex": desmos_expr,
                    "color": self._get_color(i)
                })
            
            # Create state object for Desmos
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
            
            # Encode state for URL
            state_json = json.dumps(state)
            encoded_state = urllib.parse.quote(state_json)
            
            # Create shareable URL
            graph_url = f"{self.base_url}?embed=true&state={encoded_state}"
            
            return VisualizationResult(
                success=True,
                graph_url=graph_url,
                interactive_url=graph_url,
                metadata={
                    "expressions": expressions,
                    "desmos_expressions": desmos_expressions,
                    "viewport": config.__dict__
                }
            )
            
        except Exception as e:
            return VisualizationResult(
                success=False,
                error=f"Error creating Desmos graph: {str(e)}"
            )
    
    def _convert_to_desmos_format(self, expression: str) -> str:
        """Convert expression to Desmos LaTeX format"""
        try:
            # Parse with SymPy
            # Normalize common 'f(x) =' or 'y =' prefixes
            cleaned = re.sub(r"^\s*f\s*\(\s*x\s*\)\s*=\s*", "", expression)
            cleaned = re.sub(r"^\s*y\s*=\s*", "", cleaned)
            expr = sp.sympify(cleaned)
            
            # Convert to LaTeX
            latex = sp.latex(expr)
            
            # Desmos-specific adjustments
            latex = latex.replace("\\log", "\\ln")  # Desmos uses ln for natural log
            latex = latex.replace("^{", "^\\left{")
            latex = latex.replace("}", "\\right}")
            
            return latex
            
        except:
            # Fallback to basic conversion
            return expression.replace("**", "^").replace("log", "ln")
    
    def _get_color(self, index: int) -> str:
        """Get color for expression based on index"""
        colors = ["#c74440", "#2d70b3", "#388c46", "#6042a6", "#000000", 
                 "#3a8b8c", "#9950ae", "#a85c00", "#5f7c8d", "#895d63"]
        return colors[index % len(colors)]

class GeoGebraAPI:
    """Integration with GeoGebra API"""
    
    def __init__(self):
        self.base_url = "https://www.geogebra.org/graphing"
        self.embed_url = "https://www.geogebra.org/calculator"
    
    def create_graph_url(self, expressions: List[str], config: GraphConfig = None) -> VisualizationResult:
        """Create GeoGebra graph URL"""
        try:
            if config is None:
                config = GraphConfig()
            
            # Convert expressions to GeoGebra format
            ggb_commands = []
            for i, expr in enumerate(expressions):
                ggb_expr = self._convert_to_geogebra_format(expr)
                ggb_commands.append(f"f_{i}(x) = {ggb_expr}")
            
            # Create URL with commands
            commands_str = ";".join(ggb_commands)
            encoded_commands = urllib.parse.quote(commands_str)
            
            graph_url = f"{self.embed_url}?command={encoded_commands}"
            
            return VisualizationResult(
                success=True,
                graph_url=graph_url,
                interactive_url=graph_url,
                metadata={
                    "expressions": expressions,
                    "geogebra_commands": ggb_commands
                }
            )
            
        except Exception as e:
            return VisualizationResult(
                success=False,
                error=f"Error creating GeoGebra graph: {str(e)}"
            )
    
    def _convert_to_geogebra_format(self, expression: str) -> str:
        """Convert expression to GeoGebra format"""
        try:
            cleaned = re.sub(r"^\s*f\s*\(\s*x\s*\)\s*=\s*", "", expression)
            cleaned = re.sub(r"^\s*y\s*=\s*", "", cleaned)
            expr = sp.sympify(cleaned)
            # GeoGebra uses standard mathematical notation
            return str(expr).replace("**", "^")
        except:
            return expression.replace("**", "^")

class FunctionPlotGenerator:
    """Generate HTML with function-plot.js for visualization"""
    
    def __init__(self):
        self.template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MathViz Function Plot</title>
            <script src="https://unpkg.com/function-plot@1.23.3/dist/function-plot.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                #plot {{ width: 800px; height: 600px; margin: 0 auto; }}
                .info {{ text-align: center; margin: 20px; }}
            </style>
        </head>
        <body>
            <div class="info">
                <h2>Mathematical Function Visualization</h2>
                <p>Functions: {functions_display}</p>
            </div>
            <div id="plot"></div>
            <script>
                try {{
                    functionPlot({{
                        target: '#plot',
                        width: 800,
                        height: 600,
                        grid: {grid},
                        xAxis: {{
                            domain: [{x_min}, {x_max}]
                        }},
                        yAxis: {{
                            domain: [{y_min}, {y_max}]
                        }},
                        data: {data}
                    }});
                }} catch (e) {{
                    document.getElementById('plot').innerHTML = 
                        '<div style="text-align:center; padding:50px; color:red;">Error plotting function: ' + e.message + '</div>';
                }}
            </script>
        </body>
        </html>
        """
    
    def generate_html(self, expressions: List[str], config: GraphConfig = None) -> VisualizationResult:
        """Generate HTML for function plotting"""
        try:
            if config is None:
                config = GraphConfig()
            
            data = []
            functions_display = []
            
            for i, expr in enumerate(expressions):
                # Convert expression for function-plot.js
                js_expr = self._convert_to_js_format(expr)
                
                data.append({
                    "fn": js_expr,
                    "color": self._get_color(i),
                    "graphType": "polyline"
                })
                
                functions_display.append(f"f{i+1}(x) = {expr}")
            
            html = self.template.format(
                functions_display=", ".join(functions_display),
                grid="true" if config.show_grid else "false",
                x_min=config.x_range[0],
                x_max=config.x_range[1],
                y_min=config.y_range[0],
                y_max=config.y_range[1],
                data=json.dumps(data)
            )
            
            return VisualizationResult(
                success=True,
                graph_html=html,
                metadata={
                    "expressions": expressions,
                    "js_data": data
                }
            )
            
        except Exception as e:
            return VisualizationResult(
                success=False,
                error=f"Error generating function plot HTML: {str(e)}"
            )
    
    def _convert_to_js_format(self, expression: str) -> str:
        """Convert expression to JavaScript format for function-plot.js"""
        try:
            expr = sp.sympify(expression)
            
            # Convert to JavaScript-compatible string
            js_expr = str(expr)
            
            # Common conversions
            js_expr = js_expr.replace("**", "^")
            js_expr = js_expr.replace("log(", "log(")  # Natural log
            js_expr = js_expr.replace("exp(", "exp(")
            js_expr = js_expr.replace("sin(", "sin(")
            js_expr = js_expr.replace("cos(", "cos(")
            js_expr = js_expr.replace("tan(", "tan(")
            js_expr = js_expr.replace("pi", "PI")
            js_expr = js_expr.replace("e", "E")
            
            return js_expr
            
        except:
            return expression.replace("**", "^")
    
    def _get_color(self, index: int) -> str:
        """Get color for function based on index"""
        colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", 
                 "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#795548"]
        return colors[index % len(colors)]

class GraphVisualizer:
    """Main graph visualization coordinator"""
    
    def __init__(self):
        self.desmos = DesmosAPI()
        self.geogebra = GeoGebraAPI()
        self.function_plot = FunctionPlotGenerator()

    
    def visualize_function(self, expression: str, config: GraphConfig = None,
                          provider: str = "auto") -> VisualizationResult:
        """Visualize a single mathematical function"""
        return self.visualize_functions([expression], config, provider)
    
    def visualize_functions(self, expressions: List[str], config: GraphConfig = None,
                           provider: str = "auto") -> VisualizationResult:
        """Visualize multiple mathematical functions"""
        if config is None:
            config = GraphConfig()
        
        results = {}
        
        # Try different providers based on preference
        if provider in ["auto", "desmos"]:
            try:
                results["desmos"] = self.desmos.create_graph_url(expressions, config)
            except Exception as e:
                logger.warning(f"Desmos visualization failed: {e}")
        
        if provider in ["auto", "geogebra"]:
            try:
                results["geogebra"] = self.geogebra.create_graph_url(expressions, config)
            except Exception as e:
                logger.warning(f"GeoGebra visualization failed: {e}")
        
        if provider in ["auto", "functionplot", "html"]:
            try:
                results["functionplot"] = self.function_plot.generate_html(expressions, config)
            except Exception as e:
                logger.warning(f"Function plot HTML generation failed: {e}")
        
        # Combine results or return best available
        if provider != "auto":
            return results.get(provider, VisualizationResult(success=False, error=f"Provider {provider} not available"))
        
        # Return combined result for auto mode
        combined_result = VisualizationResult(success=True)
        
        for provider_name, result in results.items():
            if result and result.success:
                if result.graph_url:
                    combined_result.graph_url = combined_result.graph_url or result.graph_url
                    combined_result.interactive_url = combined_result.interactive_url or result.interactive_url
                if result.graph_html:
                    combined_result.graph_html = result.graph_html
                
                combined_result.metadata[provider_name] = result.metadata
        
        if not any(result.success for result in results.values()):
            combined_result.success = False
            combined_result.error = "All visualization providers failed"
        
        return combined_result
    
    def visualize_contour(self, expression: str, variables: list = None, config: GraphConfig = None, levels: int = 10) -> VisualizationResult:
        """Visualize a contour map for a function of two variables using Gnuplot if available, else Plotly."""
        try:
            import shutil
            if config is None:
                config = GraphConfig()

            logger.info(f"Generating contour map for expression: {expression}")

            # Parse expression and detect variables
            expr = sp.sympify(expression)
            if variables is None:
                variables = list(expr.free_symbols)
                variables = [str(v) for v in sorted(variables, key=str)[:2]]
                logger.info(f"Auto-detected variables: {variables}")

            if len(variables) < 2:
                return VisualizationResult(
                    success=False,
                    error=f"Need at least 2 variables for contour visualization, found: {variables}"
                )

            # Check if gnuplot is available
            gnuplot_path = shutil.which('gnuplot')
            logger.info(f"Gnuplot path: {gnuplot_path}")

            if gnuplot_path:
                # Use Gnuplot for lightweight contour plotting
                import tempfile, os
                logger.info("Using gnuplot for contour generation")

                # Use the detected variables instead of hardcoded x, y
                var_symbols = [sp.Symbol(var) for var in variables[:2]]
                x_vals = np.linspace(config.x_range[0], config.x_range[1], min(config.resolution, 100))
                y_vals = np.linspace(config.y_range[0], config.y_range[1], min(config.resolution, 100))
                X, Y = np.meshgrid(x_vals, y_vals)

                try:
                    f_lambd = sp.lambdify(var_symbols, expr, modules=['numpy'])
                    Z = f_lambd(X, Y)
                    logger.info(f"Successfully computed function values for contour plot")
                except Exception as e:
                    logger.error(f"Failed to compute function values: {e}")
                    raise
                # Write data to temp file
                with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.dat') as datafile:
                    for i in range(len(x_vals)):
                        for j in range(len(y_vals)):
                            datafile.write(f"{x_vals[i]} {y_vals[j]} {Z[j, i]}\n")
                        datafile.write("\n")
                    datafile_path = datafile.name
                # Write gnuplot script
                with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.plt') as scriptfile:
                    scriptfile.write(f"set term pngcairo size 800,600\n")
                    scriptfile.write(f"set output '{scriptfile.name}.png'\n")
                    scriptfile.write(f"set title 'Contour map of {expression}'\n")
                    scriptfile.write(f"set xlabel '{variables[0]}'\nset ylabel '{variables[1]}'\n")
                    scriptfile.write(f"set contour base\nset view map\n")
                    scriptfile.write(f"set cntrparam levels {levels}\n")
                    scriptfile.write(f"splot '{datafile_path}' with lines\n")
                    scriptfile_path = scriptfile.name

                # Run gnuplot with proper error handling
                import subprocess
                try:
                    result = subprocess.run([gnuplot_path, scriptfile_path],
                                         check=True,
                                         capture_output=True,
                                         text=True,
                                         timeout=30)
                    logger.info("Gnuplot executed successfully")
                except subprocess.TimeoutExpired:
                    logger.error("Gnuplot execution timed out")
                    raise Exception("Gnuplot execution timed out after 30 seconds")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Gnuplot execution failed: {e.stderr}")
                    raise Exception(f"Gnuplot failed: {e.stderr}")
                # Read image
                img_path = scriptfile_path + '.png'
                with open(img_path, 'rb') as imgf:
                    img_bytes = imgf.read()
                # Clean up temp files
                os.remove(datafile_path)
                os.remove(scriptfile_path)
                os.remove(img_path)
                # Return image as base64 html
                import base64
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                html = f"<img src='data:image/png;base64,{img_b64}' alt='Contour map'/>"
                return VisualizationResult(
                    success=True,
                    graph_html=html,
                    metadata={"expression": expression, "type": "contour", "engine": "gnuplot"}
                )
            else:
                # Fallback to Plotly
                logger.info("Gnuplot not available, falling back to Plotly")
                var_symbols = [sp.Symbol(var) for var in variables[:2]]
                x_vals = np.linspace(config.x_range[0], config.x_range[1], min(config.resolution, 100))
                y_vals = np.linspace(config.y_range[0], config.y_range[1], min(config.resolution, 100))
                X, Y = np.meshgrid(x_vals, y_vals)

                try:
                    f_lambd = sp.lambdify(var_symbols, expr, modules=['numpy'])
                    Z = f_lambd(X, Y)

                    fig = go.Figure(data=go.Contour(
                        z=Z,
                        x=x_vals,
                        y=y_vals,
                        colorscale='Viridis',
                        ncontours=levels
                    ))
                    fig.update_layout(
                        title=f"Contour map of {expression}",
                        xaxis_title=variables[0],
                        yaxis_title=variables[1],
                        width=800,
                        height=600
                    )
                    graph_html = fig.to_html(include_plotlyjs='cdn')
                    logger.info("Plotly contour map generated successfully")

                    return VisualizationResult(
                        success=True,
                        graph_html=graph_html,
                        metadata={
                            "expression": expression,
                            "variables": variables,
                            "type": "contour",
                            "engine": "plotly",
                            "levels": levels
                        }
                    )
                except Exception as e:
                    logger.error(f"Plotly fallback failed: {e}")
                    raise
        except Exception as e:
            return VisualizationResult(success=False, error=f"Error generating contour map: {str(e)}")
    
    def visualize_derivative(self, expression: str, variable: str = "x", 
                           config: GraphConfig = None) -> VisualizationResult:
        """Visualize function and its derivative"""
        try:
            expr = sp.sympify(expression)
            var = sp.Symbol(variable)
            derivative = sp.diff(expr, var)
            
            expressions = [
                str(expr),
                str(derivative)
            ]
            
            result = self.visualize_functions(expressions, config)
            result.metadata["derivative"] = {
                "original": str(expr),
                "derivative": str(derivative),
                "variable": variable
            }
            
            return result
            
        except Exception as e:
            return VisualizationResult(
                success=False,
                error=f"Error visualizing derivative: {str(e)}"
            )
    
    def visualize_optimization(self, expression: str, critical_points: List[Tuple[float, float]] = None,
                             config: GraphConfig = None) -> VisualizationResult:
        """Visualize function with critical points for optimization"""
        try:
            # Create basic function plot
            result = self.visualize_function(expression, config)
            
            if critical_points and result.success:
                # Add critical points information to metadata
                result.metadata["optimization"] = {
                    "critical_points": critical_points,
                    "function": expression
                }
                
                # Create enhanced HTML with critical points
                if result.graph_html:
                    # Inject critical points into the HTML
                    enhanced_html = self._add_critical_points_to_html(
                        result.graph_html, expression, critical_points
                    )
                    result.graph_html = enhanced_html
            
            return result
            
        except Exception as e:
            return VisualizationResult(
                success=False,
                error=f"Error visualizing optimization: {str(e)}"
            )
    
    def visualize_3d_function(self, expression: str, variables: List[str] = None,
                            config: GraphConfig = None) -> VisualizationResult:
        """Create 3D visualization using Plotly"""
        try:
            if config is None:
                config = GraphConfig()
            
            expr = sp.sympify(expression)
            
            if variables is None:
                variables = list(expr.free_symbols)
                variables = [str(v) for v in sorted(variables, key=str)[:2]]
            
            if len(variables) < 2:
                return VisualizationResult(
                    success=False,
                    error="Need at least 2 variables for 3D visualization"
                )
            
            # Create mesh
            x_vals = np.linspace(config.x_range[0], config.x_range[1], 50)
            y_vals = np.linspace(config.y_range[0], config.y_range[1], 50)
            X, Y = np.meshgrid(x_vals, y_vals)
            
            # Convert to numerical function
            sym_vars = [sp.Symbol(var) for var in variables[:2]]
            func = sp.lambdify(sym_vars, expr, 'numpy')
            
            Z = func(X, Y)
            
            # Create 3D surface plot
            fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
            
            fig.update_layout(
                title=f'3D Plot of {expression}',
                scene=dict(
                    xaxis_title=variables[0],
                    yaxis_title=variables[1],
                    zaxis_title='f(x,y)'
                ),
                width=800,
                height=600
            )
            
            # Convert to HTML
            html = fig.to_html(include_plotlyjs='cdn')
            
            return VisualizationResult(
                success=True,
                graph_html=html,
                metadata={
                    "expression": expression,
                    "variables": variables,
                    "plot_type": "3d_surface"
                }
            )
            
        except Exception as e:
            return VisualizationResult(
                success=False,
                error=f"Error creating 3D visualization: {str(e)}"
            )
    
    def _add_critical_points_to_html(self, html: str, expression: str, 
                                   critical_points: List[Tuple[float, float]]) -> str:
        """Add critical points visualization to existing HTML"""
        try:
            # Create additional script for critical points
            points_data = []
            for i, point in enumerate(critical_points):
                points_data.append({
                    "points": [[point[0], point[1]]],
                    "fnType": "points",
                    "color": "red",
                    "attr": {"r": 5}
                })
            
            # Insert critical points into the data array
            points_json = json.dumps(points_data)
            
            # Find the data array in HTML and append points
            import re
            pattern = r'(data:\s*\[)(.*?)(\])'
            
            def replacement(match):
                existing_data = match.group(2)
                if existing_data.strip():
                    return f"{match.group(1)}{existing_data},{points_json[1:-1]}{match.group(3)}"
                else:
                    return f"{match.group(1)}{points_json[1:-1]}{match.group(3)}"
            
            enhanced_html = re.sub(pattern, replacement, html, flags=re.DOTALL)
            return enhanced_html
            
        except Exception as e:
            logger.warning(f"Could not add critical points to HTML: {e}")
            return html


def create_graph_url(expression: str, provider: str = "desmos", 
                    config: GraphConfig = None) -> str:
    """Convenience function to create graph URL"""
    visualizer = GraphVisualizer()
    result = visualizer.visualize_function(expression, config, provider)
    return result.graph_url or result.interactive_url or ""

def create_interactive_graph(expressions: List[str], config: GraphConfig = None) -> str:
    """Create interactive graph HTML"""
    visualizer = GraphVisualizer()
    result = visualizer.visualize_functions(expressions, config)
    return result.graph_html or ""


if __name__ == "__main__":
    # Test graph visualization
    visualizer = GraphVisualizer()
    
    # Test single function
    print("Testing single function visualization...")
    result = visualizer.visualize_function("x**2 + 2*x + 1")
    print(f"Desmos URL: {result.graph_url}")
    print(f"HTML available: {result.graph_html is not None}")
    
    # Test multiple functions
    print("\nTesting multiple functions...")
    result = visualizer.visualize_functions(["x**2", "2*x + 1", "sin(x)"])
    print(f"Success: {result.success}")
    
    # Test derivative visualization
    print("\nTesting derivative visualization...")
    result = visualizer.visualize_derivative("x**3 - 3*x**2 + 2*x")
    print(f"Success: {result.success}")
    print(f"Derivative metadata: {result.metadata.get('derivative', {})}")
    
    # Test 3D visualization
    print("\nTesting 3D visualization...")
    result = visualizer.visualize_3d_function("x**2 + y**2")
    print(f"3D Success: {result.success}")
    print(f"3D HTML length: {len(result.graph_html) if result.graph_html else 0}")