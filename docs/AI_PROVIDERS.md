# AI Providers for MathViz Chat

This project supports pluggable AI providers for the /chat endpoint. The AI is used to:
- Normalize/structure the math question to a solver-friendly form
- Optionally rewrite step explanations for clarity

The actual solving is done by SymPy in the MathViz pipeline.

## Providers

- Local Ollama (recommended and free):
  - Requirements: macOS, [Ollama](https://ollama.com) installed
  - Start server: `ollama serve`
  - Pull a model, e.g. a Qwen instruct variant:
    ```bash
    ollama pull lly/qwen2.5-32b-instruct-iq3_m:latest
    # or lighter:
    # ollama pull qwen2.5:7b-instruct
    # ollama pull qwen2.5:3b-instruct
    ```
  - Env vars:
    ```bash
    export MATHVIZ_AI_PROVIDER=ollama
    export MATHVIZ_AI_MODEL='lly/qwen2.5-32b-instruct-iq3_m:latest'
    export OLLAMA_HOST='http://127.0.0.1:11434'
    ```

- Stub (default):
  - No external requirements; simple heuristics for development.

## Switching Providers

The backend selects the provider using `MATHVIZ_AI_PROVIDER`:
- `ollama` → uses local Ollama server with `MATHVIZ_AI_MODEL` (default `qwen2.5:7b-instruct`).
- anything else → falls back to stub.

## Testing

With the backend running on :8000:
```bash
curl -s -X POST http://localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"message":"Differentiate sin(x)"}' | jq
```

You should see `reply_text`, a `solution` payload, and `desmos_config` for graphable prompts.
