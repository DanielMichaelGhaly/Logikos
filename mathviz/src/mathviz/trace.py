"""
Step and StepTrace dataclasses for solution tracking.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Step:
    """A single step in a mathematical solution."""
    step_id: int
    operation: str  # add, subtract, multiply, divide, substitute, etc.
    expression_before: str
    expression_after: str
    justification: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StepTrace:
    """A complete trace of solution steps."""
    problem_id: str
    steps: List[Step] = field(default_factory=list)
    initial_state: str = ""
    final_state: str = ""
    success: bool = True
    error_message: Optional[str] = None
    
    def add_step(self, step: Step) -> None:
        """Add a step to the trace."""
        self.steps.append(step)
    
    def get_step(self, step_id: int) -> Optional[Step]:
        """Get a step by its ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_step_count(self) -> int:
        """Get the total number of steps."""
        return len(self.steps)
