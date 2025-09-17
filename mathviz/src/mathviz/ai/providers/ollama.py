from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

import urllib.request
import urllib.error

from ..base import AIProvider, AIChatResult, AIStepExplanation, StructuredProblem


OLLAMA_DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("MATHVIZ_AI_MODEL", "qwen2.5:7b-instruct")


SYSTEM_INSTRUCTION_PARSE = (
    "You are an assistant that converts natural language math questions into a normalized form. "
    "Output JSON only, with keys: normalized_problem_text (string), problem_type (string|optional), visualize (bool), reply_text (string). "
    "Avoid extra commentary."
)

SYSTEM_INSTRUCTION_EXPLAIN = (
    "You are an assistant that rewrites mathematical solution steps for clarity and brevity while preserving correctness. "
    "Output JSON only with key: reasoning (string)."
)


def _ollama_generate(prompt: str, model: Optional[str] = None, system: Optional[str] = None) -> str:
    data = {
        "model": model or OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        data["system"] = system

    req = urllib.request.Request(
        f"{OLLAMA_DEFAULT_HOST}/api/generate",
        data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return payload.get("response", "").strip()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"Ollama connection error: {e}")


def _extract_json_block(text: str) -> Dict[str, Any]:
    # Try direct JSON first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Fallback: find the first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    raise ValueError("No valid JSON object found in model output")


class OllamaAIProvider:
    def generate_structured_problem(self, message: str) -> AIChatResult:
        user_prompt = (
            f"Instruction: Convert the following prompt into normalized math:\n"
            f"Prompt: {message}\n\n"
            f"Respond as JSON."
        )
        raw = _ollama_generate(user_prompt, system=SYSTEM_INSTRUCTION_PARSE)
        obj = _extract_json_block(raw)
        norm = obj.get("normalized_problem_text") or message
        problem_type = obj.get("problem_type")
        visualize = bool(obj.get("visualize", True))
        reply = obj.get("reply_text") or "Got it — I’ll parse your question, solve it, and explain the steps."
        return AIChatResult(
            reply_text=reply,
            structured_problem=StructuredProblem(problem_text=norm, problem_type=problem_type, visualize=visualize),
            metadata={"provider": "ollama", "model": OLLAMA_MODEL},
        )

    def refine_problem(self, message: str, last_error: Optional[str] = None) -> AIChatResult:
        hint = f"Last error: {last_error}" if last_error else ""
        user_prompt = (
            "Instruction: Repair the following user message into a normalized math expression/question suitable for a symbolic solver.\n"
            "Return JSON with keys: normalized_problem_text (string), problem_type (string|optional), visualize (bool), reply_text (string).\n"
            f"Prompt: {message}\n{hint}\nRespond as JSON."
        )
        raw = _ollama_generate(user_prompt, system=SYSTEM_INSTRUCTION_PARSE)
        obj = _extract_json_block(raw)
        norm = obj.get("normalized_problem_text") or message
        problem_type = obj.get("problem_type")
        visualize = bool(obj.get("visualize", True))
        reply = obj.get("reply_text") or "I refined the input to a solver-friendly form."
        return AIChatResult(
            reply_text=reply,
            structured_problem=StructuredProblem(problem_text=norm, problem_type=problem_type, visualize=visualize),
            metadata={"provider": "ollama", "model": OLLAMA_MODEL, "refined": True},
        )

    def explain_steps(self, solution: Dict[str, Any]) -> AIStepExplanation:
        original = ""
        if isinstance(solution, dict):
            original = solution.get("reasoning") or ""
        if not original:
            return AIStepExplanation(reasoning=None)

        user_prompt = (
            "Instruction: Rewrite solution reasoning for clarity; keep math precise.\n" 
            f"Reasoning: {original}\n\nRespond as JSON."
        )
        raw = _ollama_generate(user_prompt, system=SYSTEM_INSTRUCTION_EXPLAIN)
        obj = _extract_json_block(raw)
        return AIStepExplanation(reasoning=obj.get("reasoning") or original)
