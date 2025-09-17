#!/usr/bin/env python3
"""
Qwen AI Solver Interface

Handles communication with Qwen model via Ollama for mathematical problem solving.
"""

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class AIResponse:
    """Container for AI model response"""
    success: bool
    content: str
    provider: str = "qwen"
    confidence: float = 0.8
    error: str = ""
    metadata: Dict[str, Any] = None


class QwenSolver:
    """Interface for Qwen AI model via Ollama"""
    
    def __init__(self, 
                 ollama_host: str = "http://localhost:11434",
                 model_name: str = "qwen2.5:32b"):
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.timeout = 120  # seconds
    
    def solve_problem(self, problem: str, problem_type: str = "general") -> AIResponse:
        """Solve a mathematical problem using Qwen AI"""
        
        # Build appropriate prompt based on problem type
        prompt = self._build_prompt(problem, problem_type)
        
        try:
            logger.info(f"Requesting Qwen solution for: {problem}")
            response_text = self._call_ollama(prompt)
            
            return AIResponse(
                success=True,
                content=response_text,
                provider="qwen",
                confidence=0.8,
                metadata={
                    'problem_type': problem_type,
                    'model': self.model_name,
                    'prompt_length': len(prompt)
                }
            )
            
        except Exception as e:
            logger.error(f"Qwen API error: {e}")
            return AIResponse(
                success=False,
                content="",
                provider="qwen",
                confidence=0.0,
                error=str(e),
                metadata={'problem_type': problem_type}
            )
    
    def _build_prompt(self, problem: str, problem_type: str) -> str:
        """Build an appropriate prompt for the given problem type"""
        
        base_instruction = "You are a helpful math tutor. Solve this step by step and explain clearly:"
        
        if problem_type == "algebraic" or "solve" in problem.lower():
            specific_instruction = """
Please:
1. Identify the equation to solve
2. Show each algebraic step clearly
3. Verify your answer by substitution
4. State the final answer clearly as "x = [value]"
"""
        elif problem_type == "calculus" or any(word in problem.lower() for word in ["derivative", "integral"]):
            specific_instruction = """
Please:
1. Identify the function and operation
2. Apply the appropriate calculus rules
3. Show each step of differentiation/integration
4. Simplify the final result
5. State the final answer clearly
"""
        else:
            specific_instruction = """
Please:
1. Identify what type of problem this is
2. Show your work step by step
3. Explain your reasoning
4. State the final answer clearly
"""
        
        return f"{base_instruction}\n\n{problem}\n{specific_instruction}"
    
    def _call_ollama(self, prompt: str) -> str:
        """Make HTTP request to Ollama API"""
        
        data = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
        }
        
        logger.debug(f"Calling Ollama API at {self.ollama_host}")
        
        try:
            req = urllib.request.Request(
                f"{self.ollama_host}/api/generate",
                data=json.dumps(data).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                response_text = result.get("response", "").strip()
                
                if not response_text:
                    raise ValueError("Empty response from Ollama API")
                
                logger.info(f"Qwen responded with {len(response_text)} characters")
                return response_text
        
        except urllib.error.HTTPError as e:
            error_msg = f"HTTP Error {e.code}: {e.reason}"
            raise Exception(f"Ollama API HTTP Error: {error_msg}")
        
        except urllib.error.URLError as e:
            raise Exception(f"Ollama Connection Error: {e.reason}")
        
        except json.JSONDecodeError as e:
            raise Exception(f"Ollama API returned invalid JSON: {e}")
        
        except Exception as e:
            raise Exception(f"Ollama API error: {e}")
    
    def check_availability(self) -> bool:
        """Check if Ollama service is available"""
        try:
            # Try a simple request to check if Ollama is running
            req = urllib.request.Request(f"{self.ollama_host}/api/tags", method="GET")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except:
            return False


def test_qwen_solver():
    """Test the Qwen solver"""
    solver = QwenSolver()
    
    print("🧪 Testing Qwen AI Solver")
    print("=" * 50)
    
    # Check availability first
    if not solver.check_availability():
        print("❌ Ollama service not available - please start Ollama first")
        return
    
    # Test cases
    test_cases = [
        ("solve 2x+5=0", "algebraic"),
        ("derivative of x^2", "calculus"),
    ]
    
    for i, (problem, prob_type) in enumerate(test_cases, 1):
        print(f"\n{i}. Problem: '{problem}' (Type: {prob_type})")
        
        result = solver.solve_problem(problem, prob_type)
        
        if result.success:
            print(f"   ✅ Success")
            print(f"   📝 Response: {result.content[:100]}...")
            print(f"   🎯 Confidence: {result.confidence}")
        else:
            print(f"   ❌ Error: {result.error}")


if __name__ == "__main__":
    test_qwen_solver()