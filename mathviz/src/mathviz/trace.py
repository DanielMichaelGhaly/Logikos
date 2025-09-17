"""
Step and StepTrace dataclasses for solution tracking.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Step:
    """A single step in a mathematical solution."""
    step_id: str
    description: str
    operation: str  # add, subtract, multiply, divide, substitute, etc.
    input_state: Dict[str, Any]
    output_state: Dict[str, Any]
    reasoning: str
    rule_formula: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Legacy support for old interface
    @property
    def expression_before(self) -> str:
        """Legacy property for backward compatibility."""
        return self.input_state.get("expression", "")
    
    @property
    def expression_after(self) -> str:
        """Legacy property for backward compatibility."""
        return self.output_state.get("expression", "")
    
    @property
    def justification(self) -> str:
        """Legacy property for backward compatibility."""
        return self.reasoning


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
    
    def get_step(self, step_id: str) -> Optional[Step]:
        """Get a step by its ID."""
        for step in self.steps:
            if step.step_id == step_id:
                return step
        return None
    
    def get_step_count(self) -> int:
        """Get the total number of steps."""
        return len(self.steps)
