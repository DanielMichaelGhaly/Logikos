# Logikos

# MathViz

A comprehensive mathematical problem visualization and solving pipeline that transforms natural language math problems into structured solutions with step-by-step reasoning and visual output.

## 🎯 Overview

MathViz provides an end-to-end pipeline for processing mathematical problems:

- **Natural Language Parsing**: Convert plain English math problems into structured schemas
- **Input Validation**: Sanity checks and constraint verification for problem inputs
- **Symbolic Solving**: Ground-truth solutions using SymPy/NumPy with complete step tracing
- **Reasoning Generation**: Human-readable explanations from solution traces
- **Visual Output**: LaTeX and HTML formatting for professional presentation

## ✨ Features

- **Multi-Domain Support**: Handles algebraic equations, calculus problems, and optimization tasks
- **Step-by-Step Tracing**: Complete audit trail of every transformation in the solution process
- **Type-Safe Schemas**: Pydantic models ensure data integrity throughout the pipeline
- **Extensible Architecture**: Modular design allows easy addition of new problem types
- **Multiple Output Formats**: Generate LaTeX for academic papers or HTML for web display

## 🏗️ Architecture

```
mathviz/
├─ pyproject.toml      # Project configuration and dependencies
├─ README.md           # Documentation
└─ src/
   └─ mathviz/
      ├─ __init__.py        # Package initialization
      ├─ pipeline.py        # Main orchestration logic
      ├─ schemas.py         # Pydantic problem/solution models
      ├─ parser.py          # NL → structured schema conversion
      ├─ validator.py       # Problem validation and sanity checks
      ├─ solver.py          # SymPy/NumPy solver with tracing
      ├─ trace.py           # Step & StepTrace dataclasses
      ├─ reasoning.py       # Natural language step generation
      └─ viz.py             # LaTeX/HTML visualization
```

## 📦 Installation

### Basic Installation

```bash
pip install -e .
```

### Development Installation

```bash
pip install -e .[dev]
```

### Requirements

- Python >= 3.8
- pydantic >= 2.0.0
- sympy >= 1.12
- numpy >= 1.24.0

## 🚀 Quick Start

### Basic Usage

```python
from mathviz import MathVizPipeline

# Initialize the pipeline
pipeline = MathVizPipeline()

# Solve a simple equation
problem = "Solve for x: 2x + 5 = 13"
result = pipeline.process(problem)

# Access results
print("Solution:", result.final_answer)
print("Steps:", result.solution_steps)
print("Reasoning:", result.reasoning)
print("LaTeX:", result.visualization)
```

### Advanced Usage

```python
from mathviz import MathParser, MathSolver, Visualizer
from mathviz.schemas import MathProblem, Variable, Equation

# Manual problem construction
problem = MathProblem(
    problem_text="Find the derivative of x^2 + 3x",
    problem_type="calculus",
    variables=[Variable(name="x", domain="real")],
    equations=[],
    goal="differentiate expression"
)

# Solve and visualize
solver = MathSolver()
solution = solver.solve(problem)

visualizer = Visualizer()
latex_output = visualizer.generate_latex(solution.trace)
```

## 📚 Supported Problem Types

### Algebraic Equations
```python
pipeline.process("Solve for x: 3x - 7 = 2x + 5")
pipeline.process("Find x: x^2 - 4x + 4 = 0")
```

### Calculus Problems
```python
pipeline.process("Find the derivative of x^3 + 2x^2")
pipeline.process("Integrate x^2 from 0 to 5")
```

### Optimization
```python
pipeline.process("Maximize f(x) = -x^2 + 4x + 1")
```

## 🔧 Core Components

### Parser (`parser.py`)
Converts natural language to structured schemas using regex patterns and rule-based extraction.

**Key Features:**
- Equation extraction with multiple pattern matching
- Variable identification (single letters and LaTeX)
- Problem type classification
- Goal extraction

### Validator (`validator.py`)
Performs sanity checks on problem structure and constraints.

**Validation Steps:**
- Variable domain verification
- Equation consistency checks
- Constraint feasibility analysis

### Solver (`solver.py`)
Computes ground-truth solutions using SymPy symbolic mathematics.

**Capabilities:**
- Symbolic equation solving
- Numerical computation fallback
- Complete step tracing
- Error handling and recovery

### Trace System (`trace.py`)
Records every transformation in the solution process.

**Step Tracking:**
- Operation type (add, multiply, substitute, etc.)
- Expression before/after transformation
- Mathematical justification
- Metadata for debugging

### Reasoning Generator (`reasoning.py`)
Converts solution traces into human-readable explanations.

**Output Style:**
- Natural language descriptions
- Mathematical justifications
- Pedagogical clarity

### Visualizer (`viz.py`)
Renders solutions in professional formats.

**Output Formats:**
- LaTeX for academic papers
- HTML for web display
- Step-by-step presentation

## 📊 Data Models

### MathProblem
```python
{
    "problem_text": "Solve for x: 2x + 5 = 13",
    "problem_type": "algebraic",
    "variables": [{"name": "x", "domain": "real"}],
    "equations": [{"left_side": "2x + 5", "right_side": "13"}],
    "goal": "solve for x"
}
```

### MathSolution
```python
{
    "problem": MathProblem(...),
    "solution_steps": [...],
    "final_answer": {"x": 4},
    "reasoning": "Step-by-step explanation...",
    "visualization": "LaTeX or HTML output"
}
```

## 🧪 Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black src/
```

### Type Checking

```bash
mypy src/
```

### Linting

```bash
ruff check src/
```

## 🛣️ Roadmap

- [ ] Complete solver implementation for all problem types
- [ ] Enhanced reasoning generation with templates
- [ ] Interactive HTML visualizations
- [ ] Support for systems of equations
- [ ] Matrix operations and linear algebra
- [ ] Graphing capabilities for functions
- [ ] API endpoint for web integration
- [ ] CLI tool for command-line usage

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Parser Enhancement**: Add more sophisticated NLP patterns
2. **Solver Extensions**: Implement additional problem types
3. **Visualization**: Create interactive graphs and animations
4. **Testing**: Expand test coverage for edge cases
5. **Documentation**: Add more examples and tutorials


## 🙏 Acknowledgments

Built with:
- [SymPy](https://www.sympy.org/) - Symbolic mathematics
- [NumPy](https://numpy.org/) - Numerical computing
- [Pydantic](https://pydantic.dev/) - Data validation


---

**Note**: This project is under active development. Some features are still being implemented (marked with TODO in the codebase).
