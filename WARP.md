# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Logikos is a mathematical problem-solving project containing **MathViz** - a comprehensive pipeline for transforming natural language math problems into structured solutions with step-by-step reasoning and visual output.

## Architecture

### Core Pipeline Flow
The MathViz system follows a clear 5-stage pipeline architecture:

1. **Parser** (`parser.py`) - Converts natural language math problems into structured Pydantic schemas
2. **Validator** (`validator.py`) - Performs sanity checks on parsed problems
3. **Solver** (`solver.py`) - Uses SymPy/NumPy to solve problems with step-by-step tracing
4. **Reasoning Generator** (`reasoning.py`) - Converts solution traces into human-readable explanations
5. **Visualizer** (`viz.py`) - Generates LaTeX/HTML output from solution traces

### Key Components

- **Pipeline Orchestration**: `MathVizPipeline` class in `pipeline.py` coordinates all stages
- **Data Models**: Pydantic schemas in `schemas.py` define `MathProblem`, `MathSolution`, `Variable`, `Equation`
- **Step Tracing**: `Step` and `StepTrace` dataclasses in `trace.py` track solution progression
- **Package Entry**: `__init__.py` exports main public APIs

### Project Structure
```
Logikos/
├── .venv/                    # Virtual environment with all dependencies
├── mathviz/                  # Main package directory
│   ├── pyproject.toml        # Project config with dependencies and dev tools
│   ├── README.md             # Package documentation
│   └── src/mathviz/          # Source code
│       ├── __init__.py       # Public API exports
│       ├── pipeline.py       # Main orchestration logic
│       ├── schemas.py        # Pydantic data models
│       ├── parser.py         # Natural language → schema conversion
│       ├── validator.py      # Problem validation logic
│       ├── solver.py         # SymPy/NumPy solving with tracing
│       ├── trace.py          # Step tracking dataclasses
│       ├── reasoning.py      # Trace → human text generation
│       └── viz.py            # LaTeX/HTML output generation
```

## Development Environment

### Virtual Environment Setup
The project uses a Python virtual environment located at `.venv/`. Always activate it before development:

```bash
source .venv/bin/activate
```

### Installation Commands
Install in development mode:
```bash
cd mathviz
pip install -e .[dev]
```

For production installation:
```bash
cd mathviz
pip install -e .
```

### Testing
Run all tests:
```bash
cd mathviz
pytest
```

Run a specific test file:
```bash
cd mathviz
pytest tests/test_specific_file.py
```

### Code Quality

**Formatting:**
```bash
cd mathviz
black src/
```

**Type checking:**
```bash
cd mathviz
mypy src/
```

**Linting:**
```bash
cd mathviz
ruff check src/
```

Fix linting issues automatically:
```bash
cd mathviz
ruff check src/ --fix
```

## Development Patterns

### Module Naming
Following user preferences, simulation-related modules use naming patterns like `sim_runner`, `sim_validation` for consistency. The current architecture uses descriptive names (`solver.py`, `validator.py`) that align with mathematical domain terminology.

### Dependencies
- **Core**: pydantic>=2.0.0, sympy>=1.12, numpy>=1.24.0
- **Development**: pytest, black, mypy, ruff
- **Python**: Requires >=3.8

### Current Implementation Status
Most core modules contain TODO placeholders for implementation:
- `solver.py`: Main solving logic needs implementation
- `validator.py`: Validation logic needs implementation  
- `reasoning.py`: Reasoning generation needs implementation
- Basic structure and schemas are in place

## Usage Patterns

### Basic Usage
```python
from mathviz import MathVizPipeline

pipeline = MathVizPipeline()
problem = "Solve for x: 2x + 5 = 13"
result = pipeline.process(problem)

# Access components
print(result.solution_steps)
print(result.reasoning)
print(result.visualization)
```

### Working with Individual Components
```python
from mathviz.parser import MathParser
from mathviz.schemas import MathProblem

parser = MathParser()
problem = parser.parse("Find the derivative of x^2 + 3x")
```

## Configuration

### Code Style
- Line length: 88 characters (Black/Ruff)
- Target Python version: 3.8+
- Type checking: Strict mode enabled (disallow_untyped_defs=true)
- Import sorting: Enabled through Ruff

### Build System
Uses modern Python packaging with setuptools and pyproject.toml configuration.