"""
Shared schemas for the Logikos mathematical chat assistant.
Defines the JSON structures for question classification and processing.
"""

from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel


class QuestionType(str, Enum):
    """Supported mathematical question types."""
    SOLVE = "solve"
    FACTOR = "factor"
    EXPAND = "expand"
    SIMPLIFY = "simplify"
    DERIVATIVE = "derivative"
    INTEGRAL = "integral"
    LIMIT = "limit"
    PLOT = "plot"
    GRAPH = "graph"
    ROOTS = "roots"
    UNKNOWN = "unknown"


class QuestionClassification(BaseModel):
    """Structured representation of a classified mathematical question."""
    type: QuestionType
    expression: str
    context: Optional[str] = None
    domain: Optional[str] = None
    visualization_hint: bool = False
    confidence: float = 0.0
    raw_input: str


class SympyResult(BaseModel):
    """Result from SymPy ground truth computation."""
    success: bool
    result: Optional[str] = None
    steps: List[str] = []
    latex: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0


class AIResult(BaseModel):
    """Result from AI reasoning engine."""
    success: bool
    explanation: Optional[str] = None
    steps: List[str] = []
    mathematical_result: Optional[str] = None
    representable: bool = False
    visualization_type: Optional[str] = None
    error: Optional[str] = None
    execution_time: float = 0.0


class ConfidenceComparison(BaseModel):
    """Confidence comparison between SymPy and AI results."""
    sympy_result: Optional[str] = None
    ai_result: Optional[str] = None
    confidence_level: str  # "high", "medium", "low", "no_comparison"
    comparison_details: Optional[str] = None
    show_to_user: bool = False


class VisualizationData(BaseModel):
    """Data for embedded visualizations."""
    visualization_type: str  # "plot", "contour", "3d", "none"
    plot_data: Optional[str] = None  # Base64 encoded or HTML
    interactive: bool = False
    error: Optional[str] = None


class ChatResponse(BaseModel):
    """Complete response for the chat interface."""
    question_classification: QuestionClassification
    sympy_result: SympyResult
    ai_result: AIResult
    confidence: ConfidenceComparison
    visualization: VisualizationData
    response_time: float = 0.0