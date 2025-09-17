"""
Advanced Calculus Module for MathViz
====================================

Enhanced differentiation and optimization capabilities including:
- Partial derivatives and gradients
- Optimization problem solving
- Critical point analysis
- Gradient descent visualization
- Multi-variable calculus
"""

import numpy as np
import sympy as sp
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .trace import Step, StepTrace

@dataclass
class OptimizationResult:
    """Result of optimization problem"""
    critical_points: List[Tuple[float, ...]]
    function_values: List[float]
    optimization_type: str  # "minimize", "maximize", "saddle"
    gradient: Optional[sp.Expr] = None
    hessian: Optional[sp.Matrix] = None
    steps: List[Step] = field(default_factory=list)

@dataclass
class GradientDescentResult:
    """Result of gradient descent optimization"""
    path: List[Tuple[float, ...]]
    function_values: List[float]
    gradients: List[Tuple[float, ...]]
    converged: bool
    iterations: int
    final_point: Tuple[float, ...]
    final_value: float

class AdvancedCalculus:
    """Advanced calculus operations"""
    
    def __init__(self):
        self.steps = []
    
    def _add_step(self, operation: str, before: str, after: str, justification: str) -> None:
        """Helper method to add a step with consistent format."""
        self.steps.append(Step(
            step_id=str(len(self.steps)),
            description=justification,
            operation=operation,
            input_state={"expression": before},
            output_state={"expression": after},
            reasoning=justification
        ))
    
    def compute_partial_derivative(self, expression: str, variable: str, 
                                 variables: List[str] = None) -> Dict[str, Any]:
        """Compute partial derivative of expression with respect to variable"""
        self.steps = []
        
        try:
            # Parse expression and variables
            if variables is None:
                variables = list(sp.sympify(expression).free_symbols)
                variables = [str(v) for v in variables]
            
            expr = sp.sympify(expression)
            var = sp.Symbol(variable)
            
            self._add_step(
                "setup",
                expression,
                str(expr),
                f"Computing partial derivative of {expression} with respect to {variable}"
            )
            
            # Compute partial derivative
            partial_deriv = sp.diff(expr, var)
            
            self._add_step(
                "differentiation",
                str(expr),
                str(partial_deriv),
                "Apply partial differentiation rules"
            )
            
            # Simplify result
            simplified = sp.simplify(partial_deriv)
            
            if simplified != partial_deriv:
                self._add_step(
                    "simplification",
                    str(partial_deriv),
                    str(simplified),
                    "Simplify the result"
                )
            
            return {
                "derivative": str(simplified),
                "expression": simplified,
                "variable": variable,
                "variables": variables,
                "steps": self.steps
            }
            
        except Exception as e:
            self._add_step(
                "error",
                expression,
                "",
                f"Error computing partial derivative: {str(e)}"
            )
            return {
                "derivative": None,
                "error": str(e),
                "steps": self.steps
            }
    
    def compute_gradient(self, expression: str, variables: List[str] = None) -> Dict[str, Any]:
        """Compute gradient vector of multivariate function"""
        self.steps = []
        
        try:
            expr = sp.sympify(expression)
            
            if variables is None:
                variables = list(expr.free_symbols)
                variables = [str(v) for v in sorted(variables, key=str)]
            
            self._add_step(
                "setup",
                expression,
                str(expr),
                f"Computing gradient of {expression} with respect to variables {variables}"
            )
            
            # Compute partial derivatives for each variable
            gradient_components = []
            for var_name in variables:
                var = sp.Symbol(var_name)
                partial = sp.diff(expr, var)
                gradient_components.append(partial)
                
                self._add_step(
                    "partial_derivative",
                    str(expr),
                    str(partial),
                    f"∂f/∂{var_name} = {partial}"
                )
            
            # Create gradient vector
            gradient = sp.Matrix(gradient_components)
            
            self._add_step(
                "gradient",
                str(expr),
                str(gradient),
                f"Gradient vector: ∇f = {gradient}"
            )
            
            return {
                "gradient": str(gradient),
                "components": [str(comp) for comp in gradient_components],
                "variables": variables,
                "gradient_matrix": gradient,
                "steps": self.steps
            }
            
        except Exception as e:
            self._add_step(
                "error",
                expression,
                "",
                f"Error computing gradient: {str(e)}"
            )
            return {
                "gradient": None,
                "error": str(e),
                "steps": self.steps
            }
    
    def find_critical_points(self, expression: str, variables: List[str] = None) -> OptimizationResult:
        """Find critical points by setting gradient equal to zero"""
        self.steps = []
        
        try:
            expr = sp.sympify(expression)
            
            if variables is None:
                variables = list(expr.free_symbols)
                variables = [str(v) for v in sorted(variables, key=str)]
            
            step = Step(
                step_id=str(len(self.steps)),
                description=f"Finding critical points of {expression}",
                operation="setup",
                input_state={"expression": expression},
                output_state={"expression": str(expr)},
                reasoning=f"Finding critical points of {expression}"
            )
            self.steps.append(step)
            
            # Compute gradient
            gradient_result = self.compute_gradient(expression, variables)
            if gradient_result.get("error"):
                raise ValueError(gradient_result["error"])
            
            gradient = gradient_result["gradient_matrix"]
            
            # Set gradient equal to zero and solve
            step = Step(
                step_id=str(len(self.steps)),
                description="Set gradient equal to zero: ∇f = 0",
                operation="critical_point_setup",
                input_state={"expression": str(gradient)},
                output_state={"expression": "System of equations to solve"},
                reasoning="Set gradient equal to zero: ∇f = 0"
            )
            self.steps.append(step)
            
            # Solve the system of equations
            sym_vars = [sp.Symbol(var) for var in variables]
            equations = [comp for comp in gradient]
            solutions = sp.solve(equations, sym_vars)
            
            critical_points = []
            function_values = []
            
            if solutions:
                if isinstance(solutions, dict):
                    solutions = [solutions]
                elif not isinstance(solutions, list):
                    solutions = [dict(zip(sym_vars, solutions))]
                
                for sol in solutions:
                    if isinstance(sol, dict):
                        point = tuple(float(sol.get(var, 0)) for var in sym_vars)
                        critical_points.append(point)
                        
                        # Evaluate function at critical point
                        func_val = float(expr.subs(sol))
                        function_values.append(func_val)
                        
                        step = Step(
                            step_id=str(len(self.steps)),
                            description=f"Critical point found: {point} with f = {func_val:.4f}",
                            operation="critical_point",
                            input_state={"expression": str(equations)},
                            output_state={"expression": f"Point: {point}, Value: {func_val:.4f}"},
                            reasoning=f"Critical point found: {point} with f = {func_val:.4f}"
                        )
                        self.steps.append(step)
            
            # Compute Hessian for classification
            hessian = None
            if len(variables) <= 3:  # Only for manageable dimensions
                try:
                    hessian = sp.hessian(expr, sym_vars)
                    step = Step(
                        step_id=str(len(self.steps)),
                        description=f"Hessian matrix computed: {hessian}",
                        operation="hessian",
                        input_state={"expression": str(expr)},
                        output_state={"expression": str(hessian)},
                        reasoning=f"Hessian matrix computed: {hessian}"
                    )
                    self.steps.append(step)
                except:
                    pass
            
            return OptimizationResult(
                critical_points=critical_points,
                function_values=function_values,
                optimization_type="critical_points",
                gradient=gradient,
                hessian=hessian,
                steps=self.steps
            )
            
        except Exception as e:
            step = Step(
                step_id=str(len(self.steps)),
                description=f"Error finding critical points: {str(e)}",
                operation="error",
                input_state={"expression": expression},
                output_state={"expression": ""},
                reasoning=f"Error finding critical points: {str(e)}"
            )
            self.steps.append(step)
            return OptimizationResult(
                critical_points=[],
                function_values=[],
                optimization_type="error",
                steps=self.steps
            )
    
    def gradient_descent(self, expression: str, variables: List[str] = None,
                        initial_point: Tuple[float, ...] = None,
                        learning_rate: float = 0.1, max_iterations: int = 100,
                        tolerance: float = 1e-6) -> GradientDescentResult:
        """Perform gradient descent optimization"""
        self.steps = []
        
        try:
            expr = sp.sympify(expression)
            
            if variables is None:
                variables = list(expr.free_symbols)
                variables = [str(v) for v in sorted(variables, key=str)]
            
            if initial_point is None:
                initial_point = tuple(1.0 for _ in variables)
            
            self.steps.append(Step(
                step_id=str(len(self.steps)), 
                description=f"Starting gradient descent on {expression} from point {initial_point}", 
                operation="setup", 
                input_state={"expression": expression}, 
                output_state={"expression": f"Initial point: {initial_point}"}, 
                reasoning=f"Starting gradient descent on {expression} from point {initial_point}"
            ))
            
            # Compute gradient functions
            gradient_result = self.compute_gradient(expression, variables)
            if gradient_result.get("error"):
                raise ValueError(gradient_result["error"])
            
            gradient_exprs = [sp.sympify(comp) for comp in gradient_result["components"]]
            sym_vars = [sp.Symbol(var) for var in variables]
            
            # Convert to numerical functions
            func = sp.lambdify(sym_vars, expr, 'numpy')
            grad_funcs = [sp.lambdify(sym_vars, grad_expr, 'numpy') for grad_expr in gradient_exprs]
            
            # Initialize tracking
            path = [initial_point]
            function_values = [float(func(*initial_point))]
            gradients = []
            current_point = np.array(initial_point)
            
            self.steps.append(Step(
                step_id=str(len(self.steps)), 
                description=f"Iteration 0: f({initial_point}) = {function_values[0]:.6f}", 
                operation="iteration_start", 
                input_state={"expression": str(initial_point)}, 
                output_state={"expression": str(function_values[0])}, 
                reasoning=f"Iteration 0: f({initial_point}) = {function_values[0]:.6f}"
            ))
            
            for iteration in range(max_iterations):
                # Compute gradient at current point
                current_grad = np.array([grad_func(*current_point) for grad_func in grad_funcs])
                gradients.append(tuple(current_grad))
                
                # Check convergence
                grad_norm = np.linalg.norm(current_grad)
                if grad_norm < tolerance:
                    self.steps.append(Step(
                        step_id=str(len(self.steps)), 
                        description=f"Converged at iteration {iteration}: ||∇f|| = {grad_norm:.6f}", 
                        operation="convergence", 
                        input_state={"expression": str(current_point)}, 
                        output_state={"expression": "Convergence achieved"}, 
                        reasoning=f"Converged at iteration {iteration}: ||∇f|| = {grad_norm:.6f}"
                    ))
                    break
                
                # Update point
                current_point = current_point - learning_rate * current_grad
                current_value = float(func(*current_point))
                
                path.append(tuple(current_point))
                function_values.append(current_value)
                
                if iteration % 10 == 0 or iteration < 5:
                    self.steps.append(Step(
                        step_id=str(len(self.steps)), 
                        description=f"Iteration {iteration + 1}: f({current_point}) = {current_value:.6f}", 
                        operation="iteration", 
                        input_state={"expression": str(current_point)}, 
                        output_state={"expression": f"Value: {current_value:.6f}, Gradient norm: {grad_norm:.6f}"}, 
                        reasoning=f"Iteration {iteration + 1}: f({current_point}) = {current_value:.6f}"
                    ))
            
            converged = grad_norm < tolerance if 'grad_norm' in locals() else False
            
            return GradientDescentResult(
                path=path,
                function_values=function_values,
                gradients=gradients,
                converged=converged,
                iterations=len(path) - 1,
                final_point=tuple(current_point),
                final_value=function_values[-1]
            )
            
        except Exception as e:
            self.steps.append(Step(
                step_id=str(len(self.steps)), 
                description=f"Error in gradient descent: {str(e)}", 
                operation="error", 
                input_state={"expression": expression}, 
                output_state={"expression": ""}, 
                reasoning=f"Error in gradient descent: {str(e)}"
            ))
            return GradientDescentResult(
                path=[],
                function_values=[],
                gradients=[],
                converged=False,
                iterations=0,
                final_point=(0,),
                final_value=0.0
            )
    
    def classify_critical_point(self, expression: str, point: Tuple[float, ...], 
                              variables: List[str] = None) -> Dict[str, Any]:
        """Classify critical point using second derivative test"""
        try:
            expr = sp.sympify(expression)
            
            if variables is None:
                variables = list(expr.free_symbols)
                variables = [str(v) for v in sorted(variables, key=str)]
            
            sym_vars = [sp.Symbol(var) for var in variables]
            
            # Compute Hessian matrix
            hessian = sp.hessian(expr, sym_vars)
            
            # Evaluate Hessian at the point
            point_dict = dict(zip(sym_vars, point))
            hessian_at_point = hessian.subs(point_dict)
            
            # Convert to numerical matrix
            hessian_numeric = np.array(hessian_at_point.evalf()).astype(float)
            
            # Compute eigenvalues
            eigenvalues = np.linalg.eigvals(hessian_numeric)
            
            # Classify based on eigenvalues
            if len(variables) == 1:
                # Single variable case
                second_deriv = float(hessian_at_point[0, 0])
                if second_deriv > 0:
                    classification = "local minimum"
                elif second_deriv < 0:
                    classification = "local maximum"
                else:
                    classification = "inconclusive"
            else:
                # Multi-variable case
                if all(eig > 0 for eig in eigenvalues):
                    classification = "local minimum"
                elif all(eig < 0 for eig in eigenvalues):
                    classification = "local maximum"
                elif any(eig > 0 for eig in eigenvalues) and any(eig < 0 for eig in eigenvalues):
                    classification = "saddle point"
                else:
                    classification = "inconclusive"
            
            return {
                "classification": classification,
                "hessian": str(hessian_at_point),
                "eigenvalues": eigenvalues.tolist(),
                "point": point,
                "determinant": float(np.linalg.det(hessian_numeric)) if len(variables) > 1 else float(hessian_at_point[0, 0])
            }
            
        except Exception as e:
            return {
                "classification": "error",
                "error": str(e),
                "point": point
            }
    
    def lagrange_multipliers(self, objective: str, constraints: List[str], 
                           variables: List[str] = None) -> Dict[str, Any]:
        """Solve constrained optimization using Lagrange multipliers"""
        self.steps = []
        
        try:
            obj_expr = sp.sympify(objective)
            constraint_exprs = [sp.sympify(constraint) for constraint in constraints]
            
            if variables is None:
                all_symbols = set()
                all_symbols.update(obj_expr.free_symbols)
                for constraint in constraint_exprs:
                    all_symbols.update(constraint.free_symbols)
                variables = [str(v) for v in sorted(all_symbols, key=str)]
            
            sym_vars = [sp.Symbol(var) for var in variables]
            
            # Create Lagrange multipliers
            lambdas = [sp.Symbol(f'lambda_{i}') for i in range(len(constraints))]
            
            self.steps.append(Step(
                step_id=str(len(self.steps)), 
                description=f"Setting up Lagrangian for objective {objective} with constraints {constraints}", 
                operation="setup", 
                input_state={"expression": objective}, 
                output_state={"expression": f"Variables: {variables}, Multipliers: {[str(l) for l in lambdas]}"}, 
                reasoning=f"Setting up Lagrangian for objective {objective} with constraints {constraints}"
            ))
            
            # Build Lagrangian
            lagrangian = obj_expr
            for i, constraint in enumerate(constraint_exprs):
                lagrangian += lambdas[i] * constraint
            
            self.steps.append(Step(
                step_id=str(len(self.steps)), 
                description=f"Lagrangian: L = {lagrangian}", 
                operation="lagrangian", 
                input_state={"expression": str(lagrangian)}, 
                output_state={"expression": str(lagrangian)}, 
                reasoning=f"Lagrangian: L = {lagrangian}"
            ))
            
            # Create system of equations
            equations = []
            
            # Partial derivatives with respect to original variables
            for var in sym_vars:
                eq = sp.diff(lagrangian, var)
                equations.append(eq)
                self.steps.append(Step(
                    step_id=str(len(self.steps)), 
                    description=f"∂L/∂{var} = 0: {eq} = 0", 
                    operation="gradient_condition", 
                    input_state={"expression": str(lagrangian)}, 
                    output_state={"expression": str(eq)}, 
                    reasoning=f"∂L/∂{var} = 0: {eq} = 0"
                ))
            
            # Constraint equations
            for constraint in constraint_exprs:
                equations.append(constraint)
                self.steps.append(Step(
                    step_id=str(len(self.steps)), 
                    description=f"Constraint: {constraint} = 0", 
                    operation="constraint", 
                    input_state={"expression": str(constraint)}, 
                    output_state={"expression": str(constraint)}, 
                    reasoning=f"Constraint: {constraint} = 0"
                ))
            
            # Solve system
            all_vars = sym_vars + lambdas
            solutions = sp.solve(equations, all_vars)
            
            results = []
            if solutions:
                if isinstance(solutions, dict):
                    solutions = [solutions]
                
                for sol in solutions:
                    if isinstance(sol, dict):
                        point = tuple(float(sol.get(var, 0)) for var in sym_vars)
                        multiplier_values = [float(sol.get(lam, 0)) for lam in lambdas]
                        objective_value = float(obj_expr.subs(sol))
                        
                        results.append({
                            "point": point,
                            "objective_value": objective_value,
                            "multipliers": multiplier_values,
                            "solution_dict": sol
                        })
                        
                        self.steps.append(Step(
                            step_id=str(len(self.steps)), 
                            description=f"Critical point: {point}, f = {objective_value:.4f}, λ = {multiplier_values}", 
                            operation="solution", 
                            input_state={"expression": str(equations)}, 
                            output_state={"expression": f"Point: {point}, Value: {objective_value:.4f}"}, 
                            reasoning=f"Critical point: {point}, f = {objective_value:.4f}, λ = {multiplier_values}"
                        ))
            
            return {
                "solutions": results,
                "lagrangian": str(lagrangian),
                "equations": [str(eq) for eq in equations],
                "steps": self.steps
            }
            
        except Exception as e:
            self.steps.append(Step(
                step_id=str(len(self.steps)), 
                description=f"Error in Lagrange multipliers: {str(e)}", 
                operation="error", 
                input_state={"expression": objective}, 
                output_state={"expression": ""}, 
                reasoning=f"Error in Lagrange multipliers: {str(e)}"
            ))
            return {
                "solutions": [],
                "error": str(e),
                "steps": self.steps
            }


def create_3d_function_plot(expression: str, variables: List[str] = None, 
                          x_range: Tuple[float, float] = (-5, 5),
                          y_range: Tuple[float, float] = (-5, 5),
                          points: List[Tuple[float, float]] = None) -> go.Figure:
    """Create 3D plot of function with optional critical points"""
    try:
        expr = sp.sympify(expression)
        
        if variables is None:
            variables = list(expr.free_symbols)
            variables = [str(v) for v in sorted(variables, key=str)[:2]]  # Take first 2
        
        if len(variables) < 2:
            raise ValueError("Need at least 2 variables for 3D plot")
        
        # Create mesh
        x = np.linspace(x_range[0], x_range[1], 50)
        y = np.linspace(y_range[0], y_range[1], 50)
        X, Y = np.meshgrid(x, y)
        
        # Convert to numerical function
        sym_vars = [sp.Symbol(var) for var in variables[:2]]
        func = sp.lambdify(sym_vars, expr, 'numpy')
        
        Z = func(X, Y)
        
        # Create surface plot
        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Viridis')])
        
        # Add critical points if provided
        if points:
            for i, point in enumerate(points):
                if len(point) >= 2:
                    z_val = float(func(point[0], point[1]))
                    fig.add_trace(go.Scatter3d(
                        x=[point[0]], y=[point[1]], z=[z_val],
                        mode='markers',
                        marker=dict(size=10, color='red'),
                        name=f'Critical Point {i+1}'
                    ))
        
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
        
        return fig
        
    except Exception as e:
        # Return empty figure on error
        return go.Figure().add_annotation(
            text=f"Error creating 3D plot: {str(e)}",
            x=0.5, y=0.5, showarrow=False
        )


def create_gradient_descent_animation(gd_result: GradientDescentResult, 
                                    expression: str, variables: List[str] = None) -> go.Figure:
    """Create animated visualization of gradient descent path"""
    try:
        if len(gd_result.path) == 0:
            return go.Figure()
        
        # Create base function plot
        fig = create_3d_function_plot(expression, variables)
        
        # Add gradient descent path
        path_array = np.array(gd_result.path)
        
        if path_array.shape[1] >= 2:
            # Get function values for path points
            expr = sp.sympify(expression)
            if variables is None:
                variables = list(expr.free_symbols)
                variables = [str(v) for v in sorted(variables, key=str)[:2]]
            
            sym_vars = [sp.Symbol(var) for var in variables[:2]]
            func = sp.lambdify(sym_vars, expr, 'numpy')
            
            z_path = [func(point[0], point[1]) for point in gd_result.path]
            
            # Add path trace
            fig.add_trace(go.Scatter3d(
                x=path_array[:, 0],
                y=path_array[:, 1],
                z=z_path,
                mode='lines+markers',
                line=dict(color='red', width=5),
                marker=dict(size=5, color='red'),
                name='Gradient Descent Path'
            ))
            
            # Add starting point
            fig.add_trace(go.Scatter3d(
                x=[path_array[0, 0]],
                y=[path_array[0, 1]],
                z=[z_path[0]],
                mode='markers',
                marker=dict(size=12, color='green'),
                name='Start'
            ))
            
            # Add ending point
            fig.add_trace(go.Scatter3d(
                x=[path_array[-1, 0]],
                y=[path_array[-1, 1]],
                z=[z_path[-1]],
                mode='markers',
                marker=dict(size=12, color='blue'),
                name='End'
            ))
        
        fig.update_layout(title=f'Gradient Descent on {expression}')
        return fig
        
    except Exception as e:
        return go.Figure().add_annotation(
            text=f"Error creating gradient descent animation: {str(e)}",
            x=0.5, y=0.5, showarrow=False
        )


if __name__ == "__main__":
    # Test advanced calculus features
    calc = AdvancedCalculus()
    
    # Test partial derivatives
    print("Testing partial derivatives...")
    result = calc.compute_partial_derivative("x**2 + y**2", "x")
    print(f"∂/∂x (x² + y²) = {result['derivative']}")
    
    # Test gradient
    print("\nTesting gradient...")
    grad_result = calc.compute_gradient("x**2 + y**2")
    print(f"∇(x² + y²) = {grad_result['gradient']}")
    
    # Test critical points
    print("\nTesting critical points...")
    opt_result = calc.find_critical_points("x**2 + y**2")
    print(f"Critical points: {opt_result.critical_points}")
    
    # Test gradient descent
    print("\nTesting gradient descent...")
    gd_result = calc.gradient_descent("x**2 + y**2", initial_point=(2.0, 2.0))
    print(f"Final point: {gd_result.final_point}, Final value: {gd_result.final_value}")