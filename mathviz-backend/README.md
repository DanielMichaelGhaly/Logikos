# MathViz Backend (FastAPI)

This backend serves the MathViz API that the React frontend calls.

Key Endpoints:
- `GET /health` – health check
- `POST /solve` – legacy solve endpoint
- `POST /solve-with-graph` – legacy solve with graph config
- `POST /chat` – AI-first chat endpoint (preferred)

## Run (development)

```bash
# Optional: start Ollama for local AI provider
ollama serve
ollama pull lly/qwen2.5-32b-instruct-iq3_m:latest

# Choose AI provider (ollama or stub)
export MATHVIZ_AI_PROVIDER=ollama
export MATHVIZ_AI_MODEL='lly/qwen2.5-32b-instruct-iq3_m:latest'
export OLLAMA_HOST='http://127.0.0.1:11434'

# Start the server
python main.py
```

## Test

```bash
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Differentiate sin(x)"}' | jq
```

See also: ../docs/AI_PROVIDERS.md
