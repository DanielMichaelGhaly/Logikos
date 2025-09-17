"""
FastAPI Backend for MathViz React Frontend

This provides a clean REST API for the React frontend to communicate with
the MathViz pipeline, including CORS handling and proper error responses.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import sys
import os
from pathlib import Path
from datetime import datetime
import traceback

# Add the mathviz source to Python path
mathviz_src = Path(__file__).parent.parent / "mathviz" / "src"
if str(mathviz_src) not in sys.path:
    sys.path.insert(0, str(mathviz_src))

try:
    from mathviz.pipeline import MathVizPipeline, PipelineConfig
    MATHVIZ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: MathViz pipeline not available: {e}")
    MATHVIZ_AVAILABLE = False

# Create FastAPI app
app = FastAPI(
    title="MathViz API",
    description="REST API for MathViz mathematical problem solving",
    version="1.0.0"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400,
)

# Initialize MathViz pipeline
if MATHVIZ_AVAILABLE:
    config = PipelineConfig(
        use_ai=False,  # Disable AI inside pipeline; we orchestrate AI at the API layer
        ai_first=False,
        use_ai_parser=True,  # Use enhanced parser
        enable_interactive_graphs=True,
        enable_rate_limiting=False,  # Disable for development
        timeout=120.0  # Allow longer processing for local LLMs
    )
    pipeline = MathVizPipeline(config)
else:
    pipeline = None

# Initialize AI provider (stub or real based on env)
try:
    from mathviz.ai.base import provider_from_env
    from mathviz.ai.graphing import maybe_make_desmos_config
    ai_provider = provider_from_env()
except Exception as e:
    print(f"Warning: AI provider not available: {e}")
    ai_provider = None

# Request/Response models
class SolveProblemRequest(BaseModel):
    problem: str

class SolveProblemResponse(BaseModel):
    success: bool
    solution: dict = None
    desmos_url: str = None
    error: str = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    mathviz_available: bool

# API Routes

class ChatRequest(BaseModel):
    message: str
    include_steps: bool = True
    include_reasoning: bool = True
    include_visualization: bool = True

class DesmosConfigModel(BaseModel):
    expressions: List[str]
    xmin: Optional[float] = None
    xmax: Optional[float] = None
    ymin: Optional[float] = None
    ymax: Optional[float] = None

class ChatResponse(BaseModel):
    success: bool
    message: str
    reply_text: str
    solution: Optional[Dict[str, Any]] = None
    desmos_url: Optional[str] = None
    desmos_config: Optional[DesmosConfigModel] = None

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """AI-first chat endpoint orchestrating AI parse -> pipeline solve -> AI steps -> optional graph."""
    if not MATHVIZ_AVAILABLE:
        raise HTTPException(status_code=503, detail="MathViz pipeline is not available")

    # 1) Initial AI normalization
    if ai_provider is None:
        normalized_text = request.message.strip()
        reply = "Got it — solving your math question."
    else:
        ai_result = ai_provider.generate_structured_problem(request.message)
        normalized_text = ai_result.structured_problem.problem_text
        reply = ai_result.reply_text

    # Attempt solve with up to one refinement if it fails
    attempts = 0
    last_error: Optional[str] = None
    solution_obj = None
    while attempts < 2 and solution_obj is None:
        try:
            solution_obj = pipeline.process(normalized_text)
        except Exception as e:
            last_error = str(e)
            solution_obj = None
        if solution_obj is None:
            attempts += 1
            if ai_provider is not None:
                refined = ai_provider.refine_problem(normalized_text, last_error)
                normalized_text = refined.structured_problem.problem_text
                reply = refined.reply_text or reply
            else:
                break

    if solution_obj is None:
        # Failed after refinement
        return ChatResponse(
            success=False,
            message=f"Unable to process the question. {last_error or ''}",
            reply_text="I couldn't parse that yet. Could you rephrase the problem?",
        )

    try:
        # Convert solution to dict
        solution_dict: Dict[str, Any] = {
            "problem": {
                "problem_text": solution_obj.problem.problem_text,
                "problem_type": solution_obj.problem.problem_type,
                "variables": [{"name": v.name, "domain": v.domain} for v in solution_obj.problem.variables],
                "equations": [{"left_side": e.left_side, "right_side": e.right_side} for e in solution_obj.problem.equations],
                "goal": solution_obj.problem.goal,
            },
            "solution_steps": solution_obj.solution_steps,
            "final_answer": solution_obj.final_answer,
            "reasoning": solution_obj.reasoning,
            "visualization": solution_obj.visualization,
            "metadata": solution_obj.metadata or {},
        }


        # Enhance reasoning via AI provider if requested
        if request.include_reasoning and ai_provider is not None and solution_obj.reasoning:
            try:
                ai_steps = ai_provider.explain_steps({"reasoning": solution_obj.reasoning})
                solution_dict["reasoning"] = ai_steps.reasoning or solution_obj.reasoning
            except Exception:
                pass

        # Optional Desmos config
        desmos_conf_model: Optional[DesmosConfigModel] = None
        if request.include_visualization:
            try:
                desmos_conf = maybe_make_desmos_config({
                    "problem": {"text": solution_obj.problem.problem_text},
                    "final_answer": solution_obj.final_answer,
                })
                if desmos_conf:
                    desmos_conf_model = DesmosConfigModel(
                        expressions=desmos_conf.expressions,
                        xmin=desmos_conf.xmin,
                        xmax=desmos_conf.xmax,
                        ymin=desmos_conf.ymin,
                        ymax=desmos_conf.ymax,
                    )
            except Exception:
                desmos_conf_model = None

        # Extract Desmos URL from metadata if present
        desmos_url = None
        if getattr(solution_obj, 'metadata', None):
            desmos_url = solution_obj.metadata.get('desmos_url') if isinstance(solution_obj.metadata, dict) else None

        return ChatResponse(
            success=True,
            message="Chat processed successfully",
            reply_text=reply,
            solution=solution_dict,
            desmos_url=desmos_url,
            desmos_config=desmos_conf_model,
        )
    except Exception as e:
        print(f"Error processing chat: {e}")
        print(traceback.format_exc())
        return ChatResponse(
            success=False,
            message=str(e),
            reply_text="Sorry, I ran into an error processing that question.",
        )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        mathviz_available=MATHVIZ_AVAILABLE
    )

@app.post("/solve", response_model=SolveProblemResponse)
async def solve_problem(request: SolveProblemRequest):
    """Solve a mathematical problem"""
    if not MATHVIZ_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="MathViz pipeline is not available"
        )
    
    if not request.problem.strip():
        raise HTTPException(
            status_code=400,
            detail="Problem text cannot be empty"
        )
    
    try:
        # Process the problem
        solution = pipeline.process(request.problem.strip())
        
        # Convert solution to dict format for JSON serialization
        solution_dict = {
            "problem": {
                "problem_text": solution.problem.problem_text,
                "problem_type": solution.problem.problem_type,
                "variables": [{"name": v.name, "domain": v.domain} for v in solution.problem.variables],
                "equations": [{"left_side": e.left_side, "right_side": e.right_side} for e in solution.problem.equations],
                "goal": solution.problem.goal
            },
            "solution_steps": solution.solution_steps,
            "final_answer": solution.final_answer,
            "reasoning": solution.reasoning,
            "visualization": solution.visualization,
            "metadata": solution.metadata or {}
        }
        
        # Try to extract Desmos URL if available
        desmos_url = None
        if hasattr(solution, 'metadata') and solution.metadata:
            desmos_url = solution.metadata.get('desmos_url')
        
        return SolveProblemResponse(
            success=True,
            solution=solution_dict,
            desmos_url=desmos_url
        )
        
    except Exception as e:
        print(f"Error solving problem: {e}")
        print(traceback.format_exc())
        
        return SolveProblemResponse(
            success=False,
            error=str(e)
        )

@app.post("/solve-with-graph", response_model=SolveProblemResponse)
async def solve_problem_with_visualization(request: SolveProblemRequest):
    """Solve a problem with enhanced visualization"""
    if not MATHVIZ_AVAILABLE:
        raise HTTPException(
            status_code=503, 
            detail="MathViz pipeline is not available"
        )
    
    try:
        # Use the enhanced pipeline method that includes graphs
        result = pipeline.process_with_graph_config(request.problem.strip())
        
        solution_dict = {
            "problem": {
                "problem_text": result["solution"].problem.problem_text,
                "problem_type": result["solution"].problem.problem_type,
                "variables": [{"name": v.name, "domain": v.domain} for v in result["solution"].problem.variables],
                "equations": [{"left_side": e.left_side, "right_side": e.right_side} for e in result["solution"].problem.equations],
                "goal": result["solution"].problem.goal
            },
            "solution_steps": result["solution"].solution_steps,
            "final_answer": result["solution"].final_answer,
            "reasoning": result["solution"].reasoning,
            "visualization": result["solution"].visualization,
            "metadata": result["solution"].metadata or {}
        }
        
        return SolveProblemResponse(
            success=True,
            solution=solution_dict,
            desmos_url=result.get("desmos_url")
        )
        
    except Exception as e:
        print(f"Error solving problem with visualization: {e}")
        print(traceback.format_exc())
        
        return SolveProblemResponse(
            success=False,
            error=str(e)
        )

@app.get("/problem-types")
async def get_problem_types():
    """Get available problem types"""
    return {
        "types": ["algebraic", "calculus", "optimization", "general"]
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "MathViz API is running",
        "version": "1.0.0",
        "mathviz_available": MATHVIZ_AVAILABLE,
        "endpoints": {
            "health": "/health",
            "solve": "/solve",
            "solve_with_graph": "/solve-with-graph",
            "problem_types": "/problem-types",
            "chat": "/chat"
        }
    }

if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting MathViz API server...")
    print(f"📊 MathViz Pipeline Available: {MATHVIZ_AVAILABLE}")
    print("🌐 API will be available at: http://localhost:8000")
    print("📚 API docs will be available at: http://localhost:8000/docs")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )