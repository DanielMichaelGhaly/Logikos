"""
Pydantic models for mathematical problems and solutions.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Variable(BaseModel):
    """A mathematical variable with constraints."""
    name: str
    domain: str = "real"  # real, integer, complex, etc.
    constraints: List[str] = Field(default_factory=list)


class Equation(BaseModel):
    """A mathematical equation."""
    left_side: str
    right_side: str
    equation_type: str = "algebraic"  # algebraic, differential, etc.


class MathProblem(BaseModel):
    """A structured mathematical problem."""
    problem_text: str
    problem_type: str  # equation, optimization, calculus, etc.
    variables: List[Variable] = Field(default_factory=list)
    equations: List[Equation] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    goal: str  # What we're solving for
    context: Optional[Dict[str, Any]] = None


class MathSolution(BaseModel):
    """A complete solution to a mathematical problem."""
    problem: MathProblem
    solution_steps: List[Dict[str, Any]]
    final_answer: Dict[str, Any]
    reasoning: str
    visualization: str  # LaTeX or HTML
    metadata: Optional[Dict[str, Any]] = None
    trace: Optional[Any] = None  # StepTrace object from solver
    
    class Config:
        arbitrary_types_allowed = True  # Allow non-Pydantic types like StepTrace
