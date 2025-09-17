# Logikos

AI-first math assistant that parses natural-language questions, solves symbolically with SymPy, generates human-readable steps, and visualizes results. Designed for clarity, extensibility, and collaboration.

## Overview

Flow:
1) Chat input → /chat
2) AI normalization and validation
   - AI decides problem type and normalizes text
   - If initial solving fails, AI refines input (validator loop) and retries once
3) Solve/visualize
   - If solvable by SymPy → solve and produce structured result
   - If graphable → produce Desmos config (and HTML alternatives)
   - If both → do both
4) Step generation
   - Convert solver trace into readable steps
   - AI can rewrite reasoning for clarity
5) Math formatting (KaTeX on frontend)
6) Output rendering
   - Steps + final answer
   - Embedded graph when available

## Repos & Apps

- mathviz-backend/ (FastAPI) — REST API used by the frontend
- mathviz/ (Python library) — pipeline and AI integrations
- mathviz-frontend/ (React+TS) — chat UI and visualization

## Key Endpoints

- POST /chat — preferred chat endpoint (AI-first)
- POST /solve — legacy solver endpoint
- POST /solve-with-graph — legacy solve + graph
- GET /health — health check

## AI Providers

- Stub (default): simple heuristics for local dev
- Ollama (local, free): set environment variables
  - MATHVIZ_AI_PROVIDER=ollama
  - MATHVIZ_AI_MODEL=qwen2.5:7b-instruct (or your own)
  - OLLAMA_HOST=http://127.0.0.1:11434

Provider interface (mathviz/src/mathviz/ai/base.py):
- generate_structured_problem(message) → normalized text + reply
- refine_problem(message, last_error?) → validator/refiner
- explain_steps(solution_dict) → improved reasoning

## Backend Architecture

- mathviz-backend/main.py
  - FastAPI app, CORS, /chat orchestration
  - One refinement retry on error (AI validator loop)
- mathviz/src/mathviz/
  - pipeline.py — orchestrates parser → validator → solver → reasoning → viz
  - schemas.py — Pydantic models for problems/solutions
  - parser.py — NL → structured problem
  - validator.py — domain/unit checks (hooked by pipeline)
  - solver.py — SymPy-based solving; calculus helpers
  - reasoning.py / step_explainer.py — turns traces into readable steps
  - viz.py / graph_visualizer.py — Desmos/HTML/GeoGebra integration
  - ai/ — AI interfaces and providers (stub, ollama) and graphing helpers

## Frontend Architecture

- mathviz-frontend/src/
  - services/api.ts — API client (120s timeout for local LLMs)
  - components/
    - MathVizChat.tsx — chat layout (now branded Logikos)
    - MessageBubble.tsx — message styling
    - SolutionDisplay.tsx — steps and KaTeX
    - DesmosGraph.tsx — JS API embed (with iframe fallback)
  - types/ — shared TS types

## Development

1) Backend
```bash
# Optional (local free AI):
ollama serve
ollama pull qwen2.5:7b-instruct
export MATHVIZ_AI_PROVIDER=ollama
export MATHVIZ_AI_MODEL='qwen2.5:7b-instruct'
export OLLAMA_HOST='http://127.0.0.1:11434'

python mathviz-backend/main.py
```

2) Frontend
```bash
cd mathviz-frontend
npm install
npm start
# open http://localhost:3000
```

## Contributing

- Code style: Python (Black/Ruff), TypeScript (ESLint via CRA defaults)
- Keep changes small and focused; update docs when modifying flows
- Add tests for new problem types and parsers

## Current Priorities

- Factorization path: normalize “Factor …” to show factors or convert to equation when solving for zeros.
- Model selector in UI + latency display
- Richer graph specs (bounds, points, overlays)

---

Questions? See docs/AI_PROVIDERS.md for provider setup and usage.
