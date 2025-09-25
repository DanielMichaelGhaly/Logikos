"""
Main FastAPI application for the Logikos mathematical chat assistant.
Coordinates input processing, dual solving, and confidence comparison.
"""

import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.input_processor.classifier import QuestionClassifier
from backend.solvers.sympy_solver import SympySolver
from backend.solvers.ai_reasoner import AIReasoner
from backend.solvers.confidence import ConfidenceComparator
from shared.schemas import ChatResponse, VisualizationData


app = FastAPI(
    title="Logikos Mathematical Chat Assistant",
    description="AI-enhanced mathematical problem solver with confidence comparison",
    version="2.0.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
classifier = QuestionClassifier()
sympy_solver = SympySolver()
ai_reasoner = AIReasoner()
confidence_comparator = ConfidenceComparator()


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Logikos Mathematical Chat Assistant API v2.0"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint that processes mathematical questions.

    Args:
        request: Chat request with user message

    Returns:
        ChatResponse with complete processing results
    """
    start_time = time.time()

    try:
        # Step 1: Classify the question
        classification = classifier.classify_question(request.message)

        # Step 2: Solve with SymPy (ground truth)
        sympy_result = sympy_solver.solve_question(classification)

        # Step 3: Reason with AI (educational)
        ai_result = ai_reasoner.reason_about_question(classification)

        # Step 4: Compare results for confidence
        confidence = confidence_comparator.compare_results(
            classification, sympy_result, ai_result
        )

        # Step 5: Handle visualization (placeholder for now)
        visualization = VisualizationData(
            visualization_type="none",  # Will be implemented in Phase 4
            plot_data=None,
            interactive=False
        )

        # If AI determined the problem is representable, note it
        if ai_result.success and ai_result.representable:
            visualization.visualization_type = ai_result.visualization_type or "plot"

        response_time = time.time() - start_time

        return ChatResponse(
            question_classification=classification,
            sympy_result=sympy_result,
            ai_result=ai_result,
            confidence=confidence,
            visualization=visualization,
            response_time=response_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.post("/classify")
async def classify_question(request: ChatRequest):
    """Endpoint to test question classification."""
    try:
        classification = classifier.classify_question(request.message)
        return classification
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")


@app.post("/sympy-solve")
async def sympy_solve_endpoint(request: ChatRequest):
    """Endpoint to test SymPy solving."""
    try:
        classification = classifier.classify_question(request.message)
        result = sympy_solver.solve_question(classification)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SymPy solving error: {str(e)}")


@app.post("/ai-reason")
async def ai_reason_endpoint(request: ChatRequest):
    """Endpoint to test AI reasoning."""
    try:
        classification = classifier.classify_question(request.message)
        result = ai_reasoner.reason_about_question(classification)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI reasoning error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)