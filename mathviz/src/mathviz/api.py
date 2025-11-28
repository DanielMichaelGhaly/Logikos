"""
FastAPI backend endpoints for MathViz - REST API for mathematical problem solving.
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import traceback
import uvicorn
from datetime import datetime
from .ai.base import provider_from_env
from .ai.graphing import maybe_make_desmos_config

# Import MathViz components
from .pipeline import MathVizPipeline
from .schemas import MathProblem, MathSolution
from .validator import ValidationError
from .trace import StepTrace, Step

# Initialize FastAPI app
app = FastAPI(
    title="MathViz API",
    description="AI-powered mathematical problem solver with step-by-step explanations and visualizations",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MathViz pipeline
pipeline = MathVizPipeline()
ai_provider = provider_from_env()

# Request/Response Models
class ProblemRequest(BaseModel):
    """Request model for problem solving."""
    problem_text: str = Field(..., description="Natural language mathematical problem", min_length=1)
    include_steps: bool = Field(True, description="Include step-by-step solution")
    include_reasoning: bool = Field(True, description="Include detailed reasoning")
    include_visualization: bool = Field(True, description="Include visualization data")
    problem_type: Optional[str] = Field(None, description="Optional problem type hint")

class ValidationRequest(BaseModel):
    """Request model for problem validation."""
    problem_text: str = Field(..., description="Natural language mathematical problem", min_length=1)

class APIResponse(BaseModel):
    """Base response model."""
    success: bool
    message: str
    timestamp: datetime = Field(default_factory=datetime.now)

class ValidationResponse(APIResponse):
    """Response model for validation."""
    is_valid: bool
    parsed_problem: Optional[Dict[str, Any]] = None
    validation_errors: Optional[List[str]] = None

class SolutionResponse(APIResponse):
    """Response model for problem solving."""
    solution: Optional[Dict[str, Any]] = None
    execution_time_ms: Optional[float] = None
    error_details: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.now)
    components: Dict[str, str]

# Exception handlers
@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc: ValidationError):
    """Handle validation errors."""
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "success": False,
            "message": f"Validation failed: {str(exc)}",
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "message": f"Internal server error: {str(exc)}",
            "timestamp": datetime.now().isoformat(),
            "error_type": type(exc).__name__
        }
    )

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    return """
    <html>
        <head>
            <title>MathViz API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { color: #333; }
                .endpoint { background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .method { color: #007bff; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1 class="header">🧮 MathViz API</h1>
            <p>AI-powered mathematical problem solver with step-by-step explanations and visualizations.</p>
            
            <h2>Available Endpoints:</h2>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/health</code> - Health check
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/validate</code> - Validate a mathematical problem
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <code>/solve</code> - Solve a mathematical problem
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/examples</code> - Get example problems
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/docs</code> - Interactive API documentation
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <code>/redoc</code> - Alternative API documentation
            </div>
            
            <h2>Quick Example:</h2>
            <pre>
POST /solve
{
    "problem_text": "Solve for x: 2x + 5 = 13",
    "include_steps": true,
    "include_reasoning": true
}
            </pre>
        </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Test basic pipeline functionality
        test_problem = "2 + 2"
        parsed = pipeline.parser.parse(test_problem)
        
        components = {
            "parser": "healthy",
            "validator": "healthy", 
            "solver": "healthy",
            "reasoner": "healthy",
            "visualizer": "healthy"
        }
        
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            components=components
        )
    except Exception as e:
        return HealthResponse(
            status="degraded",
            version="0.1.0",
            components={"error": str(e)}
        )

@app.post("/validate", response_model=ValidationResponse)
async def validate_problem(request: ValidationRequest):
    """Validate a mathematical problem."""
    try:
        # Parse the problem
        parsed_problem = pipeline.parser.parse(request.problem_text)
        
        # Validate the parsed problem
        is_valid = pipeline.validator.validate(parsed_problem)
        
        if is_valid:
            return ValidationResponse(
                success=True,
                message="Problem is valid",
                is_valid=True,
                parsed_problem={
                    "problem_type": parsed_problem.problem_type,
                    "goal": parsed_problem.goal,
                    "variables": [{"name": v.name, "domain": v.domain} for v in parsed_problem.variables],
                    "equations": [{"left": eq.left_side, "right": eq.right_side} for eq in parsed_problem.equations],
                    "constraints": parsed_problem.constraints
                }
            )
        else:
            return ValidationResponse(
                success=True,
                message="Problem has validation issues",
                is_valid=False,
                validation_errors=pipeline.validator.get_validation_errors()
            )
            
    except ValidationError as e:
        return ValidationResponse(
            success=False,
            message=f"Validation failed: {str(e)}",
            is_valid=False,
            validation_errors=pipeline.validator.get_validation_errors()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during validation: {str(e)}"
        )

@app.post("/solve", response_model=SolutionResponse)
async def solve_problem(request: ProblemRequest):
    """Solve a mathematical problem."""
    start_time = datetime.now()
    
    try:
        # Solve the problem using the pipeline
        solution = pipeline.process(request.problem_text)
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Prepare response data
        solution_data = {
            "problem": {
                "text": solution.problem.problem_text,
                "type": solution.problem.problem_type,
                "goal": solution.problem.goal
            },
            "final_answer": solution.final_answer,
            "metadata": solution.metadata or {}
        }
        
        # Add optional components based on request flags
        if request.include_steps and solution.solution_steps:
            solution_data["steps"] = solution.solution_steps
        
        if request.include_reasoning and solution.reasoning:
            solution_data["reasoning"] = solution.reasoning
        
        if request.include_visualization and solution.visualization:
            solution_data["visualization"] = solution.visualization
        
        return SolutionResponse(
            success=True,
            message="Problem solved successfully",
            solution=solution_data,
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        error_details = traceback.format_exc()
        
        return SolutionResponse(
            success=False,
            message=f"Error solving problem: {str(e)}",
            execution_time_ms=execution_time,
            error_details=error_details
        )

class ChatRequest(BaseModel):
    """Chat-style request model that expects a math question."""
    message: str = Field(..., description="User message containing a math question", min_length=1)
    include_steps: bool = Field(True, description="Include step-by-step solution")
    include_reasoning: bool = Field(True, description="Include detailed reasoning")
    include_visualization: bool = Field(True, description="Include visualization data")

class DesmosConfigModel(BaseModel):
    expressions: List[str]
    xmin: Optional[float] = None
    xmax: Optional[float] = None
    ymin: Optional[float] = None
    ymax: Optional[float] = None

class ChatResponse(APIResponse):
    reply_text: str
    solution: Optional[Dict[str, Any]] = None
    desmos_url: Optional[str] = None
    desmos_config: Optional[DesmosConfigModel] = None

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """AI-first chat endpoint: message -> AI structuring -> solve -> AI steps -> optional graph."""
    start_time = datetime.now()

    try:
        # 1) Use AI provider to structure the problem from free-form chat
        ai_result = ai_provider.generate_structured_problem(request.message)
        normalized_text = ai_result.structured_problem.problem_text

        # 2) Solve using existing pipeline
        solved = pipeline.process(normalized_text)

        # 3) Build a solution payload that's frontend-friendly
        solution_payload: Dict[str, Any] = {
            "problem": {
                "text": getattr(solved.problem, "problem_text", normalized_text),
                "type": getattr(solved.problem, "problem_type", None),
                "goal": getattr(solved.problem, "goal", None),
            },
            "final_answer": getattr(solved, "final_answer", None),
            "metadata": getattr(solved, "metadata", {}) or {},
        }

        # Optional components
        if request.include_steps and getattr(solved, "solution_steps", None):
            solution_payload["steps"] = solved.solution_steps
        if request.include_reasoning and getattr(solved, "reasoning", None):
            # Optionally enhance with AI provider
            ai_steps = ai_provider.explain_steps({
                "reasoning": solved.reasoning
            })
            solution_payload["reasoning"] = ai_steps.reasoning or solved.reasoning
        if request.include_visualization and getattr(solved, "visualization", None):
            solution_payload["visualization"] = solved.visualization

        # 4) Heuristically provide Desmos config if appropriate
        desmos_config_model: Optional[DesmosConfigModel] = None
        desmos_config = maybe_make_desmos_config(solution_payload)
        if desmos_config:
            desmos_config_model = DesmosConfigModel(
                expressions=desmos_config.expressions,
                xmin=desmos_config.xmin,
                xmax=desmos_config.xmax,
                ymin=desmos_config.ymin,
                ymax=desmos_config.ymax,
            )

        _ = (datetime.now() - start_time).total_seconds() * 1000  # execution time (unused in response)

        return ChatResponse(
            success=True,
            message="Chat processed successfully",
            reply_text=ai_result.reply_text,
            solution=solution_payload,
            desmos_config=desmos_config_model,
        )

    except Exception as e:
        return ChatResponse(
            success=False,
            message=f"Error processing chat: {str(e)}",
            reply_text="Sorry, I ran into an error processing that question.",
        )

@app.get("/examples")
async def get_examples():
    """Get example mathematical problems."""
    examples = [
        {
            "category": "Algebra",
            "problems": [
                {
                    "text": "Solve for x: 2x + 5 = 13",
                    "difficulty": "easy",
                    "description": "Simple linear equation"
                },
                {
                    "text": "Find the roots of x^2 - 5x + 6",
                    "difficulty": "medium",
                    "description": "Quadratic equation"
                },
                {
                    "text": "Solve the system: x + y = 5, 2x - y = 1",
                    "difficulty": "medium",
                    "description": "System of linear equations"
                }
            ]
        },
        {
            "category": "Calculus",
            "problems": [
                {
                    "text": "Find the derivative of x^2 + 3x",
                    "difficulty": "easy",
                    "description": "Basic polynomial differentiation"
                },
                {
                    "text": "Differentiate sin(x) + cos(x)",
                    "difficulty": "easy",
                    "description": "Trigonometric differentiation"
                },
                {
                    "text": "Integrate 2x + 1",
                    "difficulty": "easy",
                    "description": "Basic polynomial integration"
                },
                {
                    "text": "Find the derivative of x^3 * sin(x)",
                    "difficulty": "medium",
                    "description": "Product rule application"
                }
            ]
        },
        {
            "category": "Advanced",
            "problems": [
                {
                    "text": "Find the limit of (sin(x)/x) as x approaches 0",
                    "difficulty": "hard",
                    "description": "L'Hôpital's rule or standard limit"
                },
                {
                    "text": "Integrate x * e^x",
                    "difficulty": "medium",
                    "description": "Integration by parts"
                }
            ]
        }
    ]
    
    return {
        "success": True,
        "message": "Example problems retrieved successfully",
        "examples": examples,
        "total_categories": len(examples),
        "total_problems": sum(len(cat["problems"]) for cat in examples)
    }

@app.get("/solve/{problem_id}")
async def get_solution_by_id(problem_id: str):
    """Get a cached solution by ID (placeholder for future implementation)."""
    return {
        "success": False,
        "message": "Solution caching not yet implemented",
        "problem_id": problem_id
    }

@app.post("/solve/batch")
async def solve_batch_problems(problems: List[ProblemRequest]):
    """Solve multiple problems in batch."""
    if len(problems) > 10:  # Limit batch size
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size limited to 10 problems"
        )
    
    results = []
    for i, problem_request in enumerate(problems):
        try:
            # Solve individual problem
            solution = pipeline.process(problem_request.problem_text)
            
            result = {
                "index": i,
                "success": True,
                "problem_text": problem_request.problem_text,
                "final_answer": solution.final_answer,
                "reasoning": solution.reasoning if problem_request.include_reasoning else None
            }
        except Exception as e:
            result = {
                "index": i,
                "success": False,
                "problem_text": problem_request.problem_text,
                "error": str(e)
            }
        
        results.append(result)
    
    return {
        "success": True,
        "message": f"Processed {len(problems)} problems",
        "results": results,
        "total_problems": len(problems),
        "successful_solutions": sum(1 for r in results if r["success"])
    }

@app.get("/statistics")
async def get_statistics():
    """Get API usage statistics (placeholder)."""
    return {
        "success": True,
        "message": "Statistics retrieved",
        "stats": {
            "total_requests": 0,  # Would track in production
            "successful_solutions": 0,
            "error_rate": 0.0,
            "average_response_time_ms": 0.0,
            "supported_problem_types": ["algebraic", "calculus", "optimization", "general"]
        }
    }

# Utility functions for development and testing
@app.post("/debug/parse")
async def debug_parse_problem(request: ValidationRequest):
    """Debug endpoint to see how a problem is parsed."""
    try:
        parsed_problem = pipeline.parser.parse(request.problem_text)
        
        return {
            "success": True,
            "parsed_problem": {
                "problem_text": parsed_problem.problem_text,
                "problem_type": parsed_problem.problem_type,
                "variables": [{"name": v.name, "domain": v.domain, "constraints": v.constraints} 
                            for v in parsed_problem.variables],
                "equations": [{"left_side": eq.left_side, "right_side": eq.right_side, "type": eq.equation_type} 
                            for eq in parsed_problem.equations],
                "constraints": parsed_problem.constraints,
                "goal": parsed_problem.goal,
                "context": parsed_problem.context
            }
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }

def create_app() -> FastAPI:
    """Factory function to create the FastAPI app."""
    return app

def run_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI server."""
    uvicorn.run(
        "mathviz.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    run_server(reload=True)