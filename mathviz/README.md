# MathViz

A mathematical problem visualization and solving pipeline that transforms natural language math problems into structured solutions with step-by-step reasoning and visual output.

## Overview

MathViz provides a complete pipeline for:
- Parsing natural language math problems into structured schemas
- Validating problem inputs and constraints
- Solving problems using SymPy/NumPy with step-by-step tracing
- Generating human-readable reasoning from solution traces
- Visualizing solutions in LaTeX/HTML format

## Project Structure

```
mathviz/
├─ pyproject.toml      # Project configuration and dependencies
├─ README.md           # This file
└─ src/
   └─ mathviz/
      ├─ __init__.py        # Package initialization
      ├─ pipeline.py        # Main orchestration logic
      ├─ schemas.py         # Pydantic problem models
      ├─ parser.py          # Natural language → schema conversion
      ├─ validator.py       # Problem sanity checks
      ├─ solver.py          # SymPy/NumPy ground-truth + step trace
      ├─ trace.py           # Step & StepTrace dataclasses
      ├─ reasoning.py       # Step text generation from trace
      └─ viz.py             # LaTeX/HTML output from trace
```

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e .[dev]
```

## Usage

```python
from mathviz import MathVizPipeline

# Create pipeline
pipeline = MathVizPipeline()

# Process a math problem
problem = "Solve for x: 2x + 5 = 13"
result = pipeline.process(problem)

# Access the solution steps, reasoning, and visualization
print(result.solution)
print(result.reasoning)
print(result.visualization)
```

## Development

Run tests:
```bash
pytest
```

Format code:
```bash
black src/
```

Type checking:
```bash
mypy src/
```

Linting:
```bash
ruff check src/
```
