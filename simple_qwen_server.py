#!/usr/bin/env python3
"""
Simple Qwen-powered math server
"""

import json
import urllib.request
import urllib.error
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sympy as sp
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

app = FastAPI(title="Simple Qwen Math Server", version="1.0.0")

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Qwen configuration
OLLAMA_HOST = "http://127.0.0.1:11434"
QWEN_MODEL = "lly/qwen2.5-32b-instruct-iq3_m:latest"

class MathRequest(BaseModel):
    problem: str

class MathResponse(BaseModel):
    success: bool
    problem: str
    ai_explanation: str
    solution: str
    error: str = None

def call_qwen(prompt: str) -> str:
    """Direct call to Qwen model via Ollama"""
    math_prompt = f"""You are a helpful math tutor. Solve this step by step and explain clearly:

{prompt}

Please provide a clear explanation and the final answer."""
    
    data = {
        "model": QWEN_MODEL,
        "prompt": math_prompt,
        "stream": False,
    }
    
    print(f"🤖 Calling Qwen with prompt: {prompt}")
    print(f"🌐 Ollama URL: {OLLAMA_HOST}/api/generate")
    print(f"📊 Model: {QWEN_MODEL}")
    
    try:
        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        
        print("⏳ Sending request to Qwen...")
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            response_text = result.get("response", "").strip()
            print(f"✅ Qwen responded: {response_text[:100]}...")
            return response_text
    
    except urllib.error.HTTPError as e:
        error_msg = f"HTTP Error {e.code}: {e.reason}"
        print(f"❌ HTTP Error calling Qwen: {error_msg}")
        return f"Qwen API Error: {error_msg}"
    except urllib.error.URLError as e:
        error_msg = f"Connection Error: {e.reason}"
        print(f"❌ Connection Error calling Qwen: {error_msg}")
        return f"Qwen Connection Error: {error_msg}"
    except Exception as e:
        print(f"❌ Unexpected error calling Qwen: {e}")
        return f"Qwen Error: {e}"

def solve_with_sympy(problem: str) -> str:
    """Simple SymPy backup solver"""
    try:
        # Try to parse as equation
        if "=" in problem:
            left, right = problem.split("=", 1)
            eq = sp.Eq(sp.sympify(left.strip()), sp.sympify(right.strip()))
            solution = sp.solve(eq)
            return f"Solution: {solution}"
        
        # Try as expression (derivative, etc)
        expr = sp.sympify(problem)
        if "derivative" in problem.lower() or "differentiate" in problem.lower():
            x = sp.Symbol('x')
            result = sp.diff(expr, x)
            return f"Derivative: {result}"
        
        return f"Simplified: {sp.simplify(expr)}"
    
    except Exception as e:
        return f"SymPy error: {e}"

@app.get("/")
async def root():
    return HTMLResponse(content="""
<!DOCTYPE html>
<html>
<head>
    <title>Simple Qwen Math Solver</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .container { background: #f5f5f5; padding: 20px; border-radius: 10px; }
        input { width: 100%; padding: 10px; font-size: 16px; margin: 10px 0; }
        button { background: #007cba; color: white; padding: 12px 20px; border: none; border-radius: 5px; font-size: 16px; cursor: pointer; }
        button:hover { background: #005a87; }
        .result { margin-top: 20px; padding: 15px; background: white; border-radius: 5px; white-space: pre-wrap; }
        .loading { color: #666; font-style: italic; }
        .error { color: red; }
        .success { color: green; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🧮 Simple Qwen Math Solver</h1>
        <p>Powered by Qwen 2.5 32B model via Ollama</p>
        
        <input type="text" id="problem" placeholder="Enter math problem (e.g., 'solve 2x + 5 = 13' or 'derivative of x^2 + 3x')" />
        <button onclick="solveProblem()">Solve with Qwen AI</button>
        
        <div id="result" class="result" style="display:none;"></div>
    </div>

    <script>
        async function solveProblem() {
            const problem = document.getElementById('problem').value;
            const resultDiv = document.getElementById('result');
            
            if (!problem) {
                alert('Please enter a math problem');
                return;
            }
            
            resultDiv.style.display = 'block';
            resultDiv.className = 'result loading';
            resultDiv.textContent = 'Asking Qwen to solve this problem...';
            
            try {
                const response = await fetch('/solve', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ problem: problem })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    resultDiv.className = 'result success';
                    resultDiv.textContent = `Problem: ${data.problem}\n\nQwen's Explanation:\n${data.ai_explanation}\n\nSymPy Verification:\n${data.solution}`;
                } else {
                    resultDiv.className = 'result error';
                    resultDiv.textContent = `Error: ${data.error}`;
                }
            } catch (error) {
                resultDiv.className = 'result error';
                resultDiv.textContent = `Network error: ${error.message}`;
            }
        }
        
        document.getElementById('problem').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                solveProblem();
            }
        });
    </script>
</body>
</html>
    """)

@app.post("/solve", response_model=MathResponse)
async def solve_math(request: MathRequest):
    """Solve math problem using Qwen AI + SymPy verification"""
    try:
        print(f"Solving: {request.problem}")
        
        # Get AI explanation from Qwen
        ai_response = call_qwen(request.problem)
        
        # Get SymPy solution as verification
        sympy_result = solve_with_sympy(request.problem)
        
        return MathResponse(
            success=True,
            problem=request.problem,
            ai_explanation=ai_response,
            solution=sympy_result
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return MathResponse(
            success=False,
            problem=request.problem,
            ai_explanation="",
            solution="",
            error=str(e)
        )

@app.post("/chat")
async def chat(request: dict):
    """Chat endpoint compatible with React frontend"""
    try:
        message = request.get("message", "")
        print(f"Chat request: {message}")
        
        # Get AI response from Qwen
        ai_response = call_qwen(message)
        
        # Try to extract any math and solve with SymPy
        sympy_result = solve_with_sympy(message)
        
        return {
            "success": True,
            "message": "Chat processed successfully",
            "reply_text": ai_response,
            "solution": {
                "problem": {
                    "problem_text": message,
                    "problem_type": "general",
                    "variables": [],
                    "equations": [],
                    "goal": "solve"
                },
                "solution_steps": ["AI Solution"],
                "final_answer": sympy_result,
                "reasoning": ai_response,
                "visualization": "",
                "metadata": {"ai_model": QWEN_MODEL}
            },
            "desmos_url": None,
            "desmos_config": None
        }
        
    except Exception as e:
        print(f"Chat error: {e}")
        return {
            "success": False,
            "message": str(e),
            "reply_text": f"Error: {e}",
            "solution": None
        }

@app.get("/health")
async def health():
    return {"status": "healthy", "qwen_model": QWEN_MODEL}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Simple Qwen Math Server...")
    print(f"🤖 Using model: {QWEN_MODEL}")
    print("🌐 Server: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)