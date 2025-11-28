"""
Custom model integration for MathViz - Support for user-trained mathematical reasoning models.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, Optional
import os
import sys
import re

from .schemas import MathProblem, MathSolution
from .trace import StepTrace, Step
from .reasoning import ReasoningGenerator
from .viz import Visualizer


class CustomMathModel:
    """Wrapper for custom trained mathematical reasoning models."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the custom model.
        
        Args:
            model_path: Path to the fine-tuned model directory
            device: Device to load the model on ("auto", "cpu", "cuda")
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.tokenizer = None
        self.fallback_available = True
        
        try:
            self._load_model()
        except Exception as e:
            print(f"⚠️ Warning: Failed to load custom model from {model_path}: {e}")
            print("📝 Will use fallback SymPy solver instead")
            self.fallback_available = False
    
    def _load_model(self):
        """Load the tokenizer and model from the specified path."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model directory not found: {self.model_path}")
        
        print(f"🔄 Loading custom model from {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate device settings
        device_map = None
        torch_dtype = torch.float32
        
        if self.device == "auto" and torch.cuda.is_available():
            device_map = "auto"
            torch_dtype = torch.float16
        elif self.device == "cuda" and torch.cuda.is_available():
            device_map = {"": 0}
            torch_dtype = torch.float16
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True
        )
        
        if device_map is None:  # CPU usage
            self.model.to("cpu")
        
        print(f"✅ Custom model loaded successfully")
    
    def is_available(self) -> bool:
        """Check if the custom model is available and loaded."""
        return self.model is not None and self.tokenizer is not None
    
    def generate_solution(self, problem_text: str) -> Dict[str, Any]:
        """
        Generate a solution using the custom model.
        
        Args:
            problem_text: The mathematical problem to solve
            
        Returns:
            Dictionary containing the solution text and metadata
        """
        if not self.is_available():
            raise RuntimeError("Custom model not available")
        
        # Format the prompt
        prompt = self._format_prompt(problem_text)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )
        
        # Move to appropriate device
        if torch.cuda.is_available() and self.device != "cpu":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode the response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the model's response (after "### Response:")
        if "### Response:" in full_response:
            solution_text = full_response.split("### Response:")[-1].strip()
        else:
            solution_text = full_response.strip()
        
        return {
            "solution_text": solution_text,
            "model_name": "Custom MathViz Model",
            "model_path": self.model_path,
            "confidence": self._estimate_confidence(solution_text)
        }
    
    def _format_prompt(self, problem_text: str) -> str:
        """Format the problem text into the expected prompt format."""
        return f"""### Instruction:
Solve the following mathematical problem step by step:

### Input:
{problem_text}

### Response:
"""
    
    def _estimate_confidence(self, solution_text: str) -> float:
        """
        Estimate confidence in the solution based on content analysis.
        This is a simple heuristic - could be improved with proper uncertainty quantification.
        """
        confidence = 0.5  # Base confidence
        
        # Higher confidence if solution contains structured steps
        if "Step" in solution_text:
            confidence += 0.2
        
        # Higher confidence if solution contains mathematical expressions
        if re.search(r'[=+\-*/]', solution_text):
            confidence += 0.1
        
        # Higher confidence if solution has a clear final answer
        if "Final Answer:" in solution_text or "Answer:" in solution_text:
            confidence += 0.2
        
        # Lower confidence if solution is too short or too long
        if len(solution_text) < 20:
            confidence -= 0.2
        elif len(solution_text) > 1000:
            confidence -= 0.1
        
        return min(max(confidence, 0.1), 0.95)  # Clamp between 0.1 and 0.95


class CustomModelPipeline:
    """
    Enhanced MathViz pipeline that can use custom trained models.
    Falls back to SymPy solver if custom model is not available.
    """
    
    def __init__(self, custom_model_path: Optional[str] = None, use_custom_model: bool = True):
        """
        Initialize the pipeline with optional custom model support.
        
        Args:
            custom_model_path: Path to the custom model directory
            use_custom_model: Whether to attempt using the custom model
        """
        # Initialize standard components
        from .parser import MathParser
        from .validator import MathValidator
        from .solver import MathSolver
        
        self.parser = MathParser()
        self.validator = MathValidator()
        self.sympy_solver = MathSolver()  # Fallback solver
        self.reasoner = ReasoningGenerator()
        self.visualizer = Visualizer()
        
        # Initialize custom model if requested
        self.custom_model = None
        self.use_custom_model = use_custom_model
        
        if use_custom_model and custom_model_path:
            try:
                self.custom_model = CustomMathModel(custom_model_path)
            except Exception as e:
                print(f"⚠️ Failed to load custom model: {e}")
                print("📝 Will use SymPy solver as fallback")
    
    def process(self, problem_text: str) -> MathSolution:
        """
        Process a mathematical problem using custom model or fallback to SymPy.
        
        Args:
            problem_text: Natural language mathematical problem
            
        Returns:
            MathSolution object with complete solution
        """
        # Parse the problem
        try:
            problem = self.parser.parse(problem_text)
        except Exception as e:
            # If parsing fails, create a minimal problem object
            problem = MathProblem(
                problem_text=problem_text,
                problem_type="general",
                variables=[],
                equations=[],
                goal="solve the problem"
            )
        
        # Try custom model first if available
        if self.use_custom_model and self.custom_model and self.custom_model.is_available():
            try:
                return self._process_with_custom_model(problem)
            except Exception as e:
                print(f"⚠️ Custom model failed: {e}")
                print("📝 Falling back to SymPy solver")
        
        # Fallback to traditional SymPy pipeline
        return self._process_with_sympy(problem)
    
    def _process_with_custom_model(self, problem: MathProblem) -> MathSolution:
        """Process problem using the custom model."""
        # Generate solution with custom model
        result = self.custom_model.generate_solution(problem.problem_text)
        
        # Parse the AI-generated solution to extract steps
        solution_steps = self._parse_ai_solution(result["solution_text"])
        
        # Extract final answer from solution
        final_answer = self._extract_final_answer(result["solution_text"])
        
        return MathSolution(
            problem=problem,
            solution_steps=solution_steps,
            final_answer=final_answer,
            reasoning=result["solution_text"],
            visualization=self._generate_ai_visualization(result["solution_text"]),
            metadata={
                "solver_type": "custom_ai_model",
                "model_name": result["model_name"],
                "confidence": result["confidence"],
                "model_path": result["model_path"]
            }
        )
    
    def _process_with_sympy(self, problem: MathProblem) -> MathSolution:
        """Process problem using the traditional SymPy solver."""
        return self.sympy_solver.solve(problem)
    
    def _parse_ai_solution(self, solution_text: str) -> list:
        """Parse AI-generated solution text to extract steps."""
        steps = []
        step_counter = 0
        
        # Look for step patterns
        step_matches = re.finditer(r'Step (\d+):\s*([^\n]+)', solution_text, re.IGNORECASE)
        
        for match in step_matches:
            step_num = match.group(1)
            step_desc = match.group(2)
            
            steps.append({
                "step_id": step_counter,
                "operation": f"ai_step_{step_num}",
                "expression_before": "",
                "expression_after": "",
                "justification": step_desc
            })
            step_counter += 1
        
        # If no structured steps found, create a single step
        if not steps:
            steps.append({
                "step_id": 0,
                "operation": "ai_solution",
                "expression_before": "",
                "expression_after": "",
                "justification": solution_text[:200] + "..." if len(solution_text) > 200 else solution_text
            })
        
        return steps
    
    def _extract_final_answer(self, solution_text: str) -> dict:
        """Extract the final answer from AI-generated solution."""
        # Look for final answer patterns
        answer_patterns = [
            r'Final Answer:\s*([^\n]+)',
            r'Answer:\s*([^\n]+)',
            r'Therefore,?\s*([^\n]+)',
            r'So,?\s*([^\n]+)'
        ]
        
        for pattern in answer_patterns:
            match = re.search(pattern, solution_text, re.IGNORECASE)
            if match:
                return {
                    "type": "ai_generated",
                    "answer": match.group(1).strip(),
                    "extraction_method": "pattern_matching"
                }
        
        # If no clear answer pattern, return the last line
        lines = solution_text.strip().split('\n')
        if lines:
            return {
                "type": "ai_generated",
                "answer": lines[-1].strip(),
                "extraction_method": "last_line"
            }
        
        return {
            "type": "ai_generated", 
            "answer": "Solution provided above",
            "extraction_method": "fallback"
        }
    
    def _generate_ai_visualization(self, solution_text: str) -> str:
        """Generate basic LaTeX visualization for AI solutions."""
        # This is a simplified version - could be enhanced
        return f"\\text{{AI Solution: {solution_text[:50]}...}}"


def list_available_models(models_dir: str = "models") -> list:
    """List available custom models in the models directory."""
    available_models = []
    
    if os.path.exists(models_dir):
        for item in os.listdir(models_dir):
            model_path = os.path.join(models_dir, item)
            if os.path.isdir(model_path):
                # Check if it looks like a model directory
                if os.path.exists(os.path.join(model_path, "config.json")):
                    available_models.append({
                        "name": item,
                        "path": model_path,
                        "has_tokenizer": os.path.exists(os.path.join(model_path, "tokenizer.json"))
                    })
    
    return available_models


def test_custom_model(model_path: str):
    """Test a custom model with some sample problems."""
    print(f"🧪 Testing custom model: {model_path}")
    
    test_problems = [
        "Solve for x: 3x + 7 = 22",
        "Find the derivative of x^2 + 5x",
        "What is 20% of 150?"
    ]
    
    try:
        model = CustomMathModel(model_path)
        
        for problem in test_problems:
            print(f"\n📝 Problem: {problem}")
            result = model.generate_solution(problem)
            print(f"🤖 Solution: {result['solution_text'][:200]}...")
            print(f"📊 Confidence: {result['confidence']:.2f}")
    
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    # Example usage and testing
    import argparse
    
    parser = argparse.ArgumentParser(description="Test custom mathematical reasoning models")
    parser.add_argument("--model-path", required=True, help="Path to the custom model")
    parser.add_argument("--test-problem", help="Specific problem to test")
    
    args = parser.parse_args()
    
    if args.test_problem:
        model = CustomMathModel(args.model_path)
        result = model.generate_solution(args.test_problem)
        print(f"Problem: {args.test_problem}")
        print(f"Solution: {result['solution_text']}")
    else:
        test_custom_model(args.model_path)