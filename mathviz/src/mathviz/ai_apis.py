"""
AI API Integration Module for MathViz
=====================================

This module provides integration with free AI APIs for mathematical problem solving.
Supports multiple providers with fallback mechanisms and rate limiting.
"""

import os
import json
import time
import requests
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

@dataclass
class AIResponse:
    """Response from AI API"""
    success: bool
    content: str
    confidence: float
    provider: str
    tokens_used: Optional[int] = None
    error: Optional[str] = None

class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    def __init__(self, name: str, api_key: Optional[str] = None):
        self.name = name
        self.api_key = api_key
        self.rate_limit_delay = 1.0  # seconds between requests
        self.last_request_time = 0.0
    
    @abstractmethod
    def solve_problem(self, problem: str, problem_type: str = "general") -> AIResponse:
        """Solve a mathematical problem"""
        pass
    
    @abstractmethod
    def generate_reasoning(self, problem: str, solution: str) -> AIResponse:
        """Generate step-by-step reasoning for a solution"""
        pass
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting between API calls"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self.last_request_time = time.time()


class HuggingFaceProvider(AIProvider):
    """Hugging Face Inference API provider (free tier available)"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("HuggingFace", api_key or os.getenv("HUGGINGFACE_API_KEY"))
        self.base_url = "https://api-inference.huggingface.co/models"
        self.math_model = "microsoft/DialoGPT-medium"  # Free model
        self.rate_limit_delay = 2.0  # HF free tier is slower
    
    def solve_problem(self, problem: str, problem_type: str = "general") -> AIResponse:
        """Solve problem using HuggingFace model"""
        self._enforce_rate_limit()
        
        prompt = self._format_solve_prompt(problem, problem_type)
        
        try:
            response = requests.post(
                f"{self.base_url}/{self.math_model}",
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                json={"inputs": prompt, "parameters": {"max_length": 200, "temperature": 0.3}},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    content = result[0].get("generated_text", "").replace(prompt, "").strip()
                    return AIResponse(
                        success=True,
                        content=content,
                        confidence=0.7,
                        provider=self.name
                    )
            
            return AIResponse(
                success=False,
                content="",
                confidence=0.0,
                provider=self.name,
                error=f"API error: {response.status_code}"
            )
            
        except Exception as e:
            return AIResponse(
                success=False,
                content="",
                confidence=0.0,
                provider=self.name,
                error=str(e)
            )
    
    def generate_reasoning(self, problem: str, solution: str) -> AIResponse:
        """Generate reasoning using HuggingFace model"""
        self._enforce_rate_limit()
        
        prompt = f"""Explain how to solve this step by step:
Problem: {problem}
Solution: {solution}

Step-by-step explanation:
1."""
        
        try:
            response = requests.post(
                f"{self.base_url}/{self.math_model}",
                headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
                json={"inputs": prompt, "parameters": {"max_length": 300, "temperature": 0.2}},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    content = result[0].get("generated_text", "").replace(prompt, "").strip()
                    return AIResponse(
                        success=True,
                        content=f"1.{content}",
                        confidence=0.7,
                        provider=self.name
                    )
            
            return AIResponse(
                success=False,
                content="",
                confidence=0.0,
                provider=self.name,
                error=f"API error: {response.status_code}"
            )
            
        except Exception as e:
            return AIResponse(
                success=False,
                content="",
                confidence=0.0,
                provider=self.name,
                error=str(e)
            )
    
    def _format_solve_prompt(self, problem: str, problem_type: str) -> str:
        """Format problem for solving"""
        if problem_type == "differentiation":
            return f"Find the derivative: {problem}\nSolution:"
        elif problem_type == "optimization":
            return f"Solve this optimization problem: {problem}\nSolution:"
        elif problem_type == "algebra":
            return f"Solve for the unknown: {problem}\nSolution:"
        else:
            return f"Solve: {problem}\nSolution:"


class GroqProvider(AIProvider):
    """Groq API provider (has free tier)"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__("Groq", api_key or os.getenv("GROQ_API_KEY"))
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama3-8b-8192"  # Fast and free
        self.rate_limit_delay = 1.0
    
    def solve_problem(self, problem: str, problem_type: str = "general") -> AIResponse:
        """Solve problem using Groq API"""
        self._enforce_rate_limit()
        
        system_prompt = self._get_system_prompt(problem_type)
        
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                } if self.api_key else {"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": problem}
                    ],
                    "temperature": 0.2,
                    "max_tokens": 300
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                tokens_used = result.get("usage", {}).get("total_tokens", 0)
                
                return AIResponse(
                    success=True,
                    content=content,
                    confidence=0.8,
                    provider=self.name,
                    tokens_used=tokens_used
                )
            
            return AIResponse(
                success=False,
                content="",
                confidence=0.0,
                provider=self.name,
                error=f"API error: {response.status_code}"
            )
            
        except Exception as e:
            return AIResponse(
                success=False,
                content="",
                confidence=0.0,
                provider=self.name,
                error=str(e)
            )
    
    def generate_reasoning(self, problem: str, solution: str) -> AIResponse:
        """Generate reasoning using Groq API"""
        self._enforce_rate_limit()
        
        prompt = f"""Given this mathematical problem and its solution, provide a clear step-by-step explanation:

Problem: {problem}
Solution: {solution}

Please explain the solution process step by step."""
        
        try:
            response = requests.post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                } if self.api_key else {"Content-Type": "application/json"},
                json={
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a math tutor. Explain solutions clearly and step-by-step."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 400
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                tokens_used = result.get("usage", {}).get("total_tokens", 0)
                
                return AIResponse(
                    success=True,
                    content=content,
                    confidence=0.8,
                    provider=self.name,
                    tokens_used=tokens_used
                )
            
            return AIResponse(
                success=False,
                content="",
                confidence=0.0,
                provider=self.name,
                error=f"API error: {response.status_code}"
            )
            
        except Exception as e:
            return AIResponse(
                success=False,
                content="",
                confidence=0.0,
                provider=self.name,
                error=str(e)
            )
    
    def _get_system_prompt(self, problem_type: str) -> str:
        """Get system prompt based on problem type"""
        if problem_type == "differentiation":
            return "You are a calculus expert. Solve differentiation problems step by step, showing all work."
        elif problem_type == "optimization":
            return "You are an optimization specialist. Solve optimization problems by finding critical points and determining maxima/minima."
        elif problem_type == "algebra":
            return "You are an algebra expert. Solve algebraic equations step by step, isolating variables clearly."
        else:
            return "You are a mathematics expert. Solve problems step by step with clear explanations."


class FallbackProvider(AIProvider):
    """Fallback provider using basic mathematical templates"""
    
    def __init__(self):
        super().__init__("Fallback")
    
    def solve_problem(self, problem: str, problem_type: str = "general") -> AIResponse:
        """Provide basic problem solving using templates"""
        try:
            if "derivative" in problem.lower() or problem_type == "differentiation":
                content = "To find the derivative, apply differentiation rules step by step."
            elif "optimize" in problem.lower() or "maximum" in problem.lower() or problem_type == "optimization":
                content = "To optimize: 1) Find the derivative, 2) Set equal to zero, 3) Solve for critical points, 4) Test second derivative."
            elif "solve" in problem.lower() and "x" in problem:
                content = "To solve for x: 1) Isolate x on one side, 2) Perform inverse operations, 3) Simplify the result."
            else:
                content = "This problem requires step-by-step mathematical analysis. Please break it down into smaller parts."
            
            return AIResponse(
                success=True,
                content=content,
                confidence=0.3,
                provider=self.name
            )
            
        except Exception as e:
            return AIResponse(
                success=False,
                content="Unable to process this problem.",
                confidence=0.0,
                provider=self.name,
                error=str(e)
            )
    
    def generate_reasoning(self, problem: str, solution: str) -> AIResponse:
        """Generate basic reasoning template"""
        content = f"""Step-by-step approach:
1. Analyze the given problem: {problem}
2. Apply appropriate mathematical principles
3. Work through the calculations systematically
4. Verify the solution: {solution}"""
        
        return AIResponse(
            success=True,
            content=content,
            confidence=0.3,
            provider=self.name
        )


class AIManager:
    """Manages multiple AI providers with fallback"""
    
    def __init__(self):
        self.providers: List[AIProvider] = []
        self._setup_providers()
    
    def _setup_providers(self):
        """Setup available AI providers in order of preference"""
        # Add Groq if API key available
        if os.getenv("GROQ_API_KEY"):
            self.providers.append(GroqProvider())
            logger.info("Groq provider initialized")
        
        # Add HuggingFace if API key available (or use free tier)
        self.providers.append(HuggingFaceProvider())
        logger.info("HuggingFace provider initialized")
        
        # Always add fallback provider
        self.providers.append(FallbackProvider())
        logger.info("Fallback provider initialized")
    
    def solve_problem(self, problem: str, problem_type: str = "general") -> AIResponse:
        """Try to solve problem using available providers"""
        for provider in self.providers:
            try:
                logger.info(f"Attempting to solve with {provider.name}")
                response = provider.solve_problem(problem, problem_type)
                if response.success and response.confidence > 0.4:
                    logger.info(f"Successfully solved with {provider.name}")
                    return response
            except Exception as e:
                logger.warning(f"Provider {provider.name} failed: {e}")
                continue
        
        # Return the last attempt even if low confidence
        return response
    
    def generate_reasoning(self, problem: str, solution: str) -> AIResponse:
        """Generate reasoning using available providers"""
        for provider in self.providers:
            try:
                logger.info(f"Generating reasoning with {provider.name}")
                response = provider.generate_reasoning(problem, solution)
                if response.success and response.confidence > 0.4:
                    logger.info(f"Successfully generated reasoning with {provider.name}")
                    return response
            except Exception as e:
                logger.warning(f"Provider {provider.name} failed: {e}")
                continue
        
        return response
    
    def get_available_providers(self) -> List[str]:
        """Get list of available provider names"""
        return [p.name for p in self.providers]


# Global AI manager instance
ai_manager = AIManager()


def solve_with_ai(problem: str, problem_type: str = "general") -> AIResponse:
    """Convenience function to solve problem with AI"""
    return ai_manager.solve_problem(problem, problem_type)


def generate_reasoning_with_ai(problem: str, solution: str) -> AIResponse:
    """Convenience function to generate reasoning with AI"""
    return ai_manager.generate_reasoning(problem, solution)


if __name__ == "__main__":
    # Test the AI integration
    print("Testing AI Integration...")
    
    # Test problem solving
    test_problem = "Find the derivative of x^2 + 3x + 5"
    response = solve_with_ai(test_problem, "differentiation")
    print(f"Problem: {test_problem}")
    print(f"AI Solution ({response.provider}): {response.content}")
    print(f"Confidence: {response.confidence}")
    print()
    
    # Test reasoning generation
    reasoning_response = generate_reasoning_with_ai(test_problem, "2x + 3")
    print(f"AI Reasoning ({reasoning_response.provider}): {reasoning_response.content}")