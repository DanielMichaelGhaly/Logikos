# 🧮 MathViz

**AI-powered mathematical problem solver with step-by-step explanations and interactive visualizations**

MathViz is a comprehensive framework that transforms natural language math problems into structured solutions with detailed reasoning, step-by-step breakdowns, and multiple visualization formats including LaTeX, interactive plots, and web interfaces.

## ✨ Features

### 🧠 **Intelligent Problem Solving**
- **Natural Language Processing**: Parse mathematical problems written in plain English
- **Multi-Domain Support**: Algebra, calculus, optimization, and more
- **Step-by-Step Solutions**: Detailed traces of every solution step
- **Symbolic & Numeric Computation**: Powered by SymPy and NumPy

### 📊 **Rich Visualizations** 
- **LaTeX Output**: Publication-ready mathematical notation
- **Interactive Plots**: Dynamic visualizations with Plotly
- **HTML Rendering**: Web-friendly formatted solutions
- **Animation Support**: Preparation for Manim-based animations

### 🌐 **Multiple Interfaces**
- **Streamlit Web App**: Beautiful, interactive web interface
- **FastAPI Backend**: RESTful API for integration
- **Command Line**: Direct CLI problem solving
- **Python Library**: Programmatic access

### 🔧 **Robust Architecture**
- **Modular Design**: Clean separation of parsing, validation, solving, and visualization
- **Comprehensive Validation**: Unit checking, constraint validation, expression parsing
- **Error Handling**: Graceful failure with detailed error messages
- **Extensible**: Easy to add new problem types and solution methods

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd mathviz

# Install in development mode with all dependencies
source .venv/bin/activate  # Activate virtual environment
cd mathviz
pip install -e .[dev,web,viz]
```

### Basic Usage

```python
from mathviz import MathVizPipeline

# Create the pipeline
pipeline = MathVizPipeline()

# Solve a problem
result = pipeline.process("Solve for x: 2x + 5 = 13")

# Access results
print(f"Answer: {result.final_answer}")
print(f"Steps: {len(result.solution_steps)}")
print(f"Reasoning: {result.reasoning}")
```

### Web Interface (Recommended)

```bash
# Launch the Streamlit web app
python run_mathviz.py --streamlit

# Open your browser to http://localhost:8501
```

### API Server

```bash
# Start the FastAPI server
python run_mathviz.py --api

# Visit http://localhost:8000/docs for interactive API documentation
```

### Command Line

```bash
# Solve a problem directly
python run_mathviz.py --solve "Find the derivative of x^2 + 3x"

# Run interactive demo
python run_mathviz.py --examples interactive

# Run comprehensive examples
python run_mathviz.py --examples

# Check system status
python run_mathviz.py --status
```

## 📋 Supported Problem Types

### Algebra
- ✅ Linear equations: `Solve for x: 2x + 5 = 13`
- ✅ Quadratic equations: `Find the roots of x^2 - 5x + 6`
- ✅ Systems of equations: `Solve: x + y = 5, 2x - y = 1`
- ✅ Polynomial factoring and expansion

### Calculus
- ✅ Derivatives: `Find the derivative of x^2 + 3x`
- ✅ Integrals: `Integrate 2x + 1`
- ✅ Trigonometric functions: `Differentiate sin(x) + cos(x)`
- ✅ Product rule, chain rule applications

### Advanced (Planned)
- 🔄 Optimization problems
- 🔄 Differential equations
- 🔄 Vector calculus
- 🔄 Complex analysis

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    MathViz Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│  Natural Language  →  Parser  →  Validator  →  Solver      │
│       Input                                        ↓        │
│                                                            │
│  Visualization  ←  Reasoning  ←  Step Tracer  ←            │
└─────────────────────────────────────────────────────────────┘

┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Web Interface  │  │   API Server    │  │   CLI Tool     │
│   (Streamlit)   │  │   (FastAPI)     │  │   (Direct)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Core Components

- **🔍 Parser** (`parser.py`): Natural language → structured problem representation
- **✅ Validator** (`validator.py`): Input validation, unit checking, constraint verification
- **🧮 Solver** (`solver.py`): SymPy-based symbolic and numeric problem solving
- **📝 Reasoning** (`reasoning.py`): Natural language explanation generation
- **🎨 Visualizer** (`viz.py`): LaTeX, HTML, and interactive plot generation
- **📊 Tracer** (`trace.py`): Step-by-step solution tracking

## 🧪 Testing

```bash
# Run the full test suite
python run_mathviz.py --tests

# Or use pytest directly
cd mathviz
pytest tests/ -v

# Run specific test categories
pytest tests/test_pipeline.py -v
```

## 📚 Examples

### Comprehensive Demo
```bash
# Run full demonstration with all problem categories
python examples.py

# Quick demo with essential problems
python examples.py quick

# Interactive problem-solving session
python examples.py interactive
```

### API Usage

```python
import requests

# Solve a problem via API
response = requests.post("http://localhost:8000/solve", json={
    "problem_text": "Solve for x: 3x - 7 = 14",
    "include_steps": True,
    "include_reasoning": True
})

result = response.json()
print(result["solution"]["final_answer"])
```

### Streamlit Features

- 🎯 **Interactive Problem Input**: Text area with validation
- 📊 **Real-time Visualization**: Dynamic plots and LaTeX rendering
- 📝 **Step-by-Step Breakdown**: Expandable solution steps
- 📚 **Example Library**: Pre-loaded problem templates
- 💾 **Solution History**: Keep track of solved problems
- 🎨 **Customizable Display**: Toggle reasoning, steps, and visualizations

## 🛠️ Development

### Project Structure
```
mathviz/
├── 📄 pyproject.toml          # Dependencies and build config
├── 📄 README.md               # This documentation
├── 📄 examples.py             # Comprehensive demonstrations
├── 📄 run_mathviz.py          # CLI runner for all functionality
├── 📁 src/mathviz/            # Main package
│   ├── 🐍 __init__.py         # Package exports
│   ├── 🔧 pipeline.py         # Main orchestration
│   ├── 📋 schemas.py          # Data models (Pydantic)
│   ├── 🔍 parser.py           # Natural language parsing
│   ├── ✅ validator.py        # Input validation
│   ├── 🧮 solver.py           # Mathematical solving (SymPy)
│   ├── 📝 reasoning.py        # Text generation
│   ├── 🎨 viz.py              # Visualization generation
│   ├── 📊 trace.py            # Solution step tracking
│   ├── 🌐 streamlit_app.py    # Web interface
│   └── 🔌 api.py              # REST API endpoints
└── 📁 tests/                  # Comprehensive test suite
    └── 🧪 test_pipeline.py    # Main test cases
```

### Code Quality

```bash
# Format code
cd mathviz && black src/

# Type checking
cd mathviz && mypy src/

# Linting
cd mathviz && ruff check src/

# Fix linting issues
cd mathviz && ruff check src/ --fix
```

### Adding New Problem Types

1. **Extend Parser**: Add pattern recognition in `parser.py`
2. **Update Solver**: Implement solving logic in `solver.py`
3. **Add Validation**: Include validation rules in `validator.py`
4. **Enhance Reasoning**: Add explanation templates in `reasoning.py`
5. **Create Visualizations**: Add visualization support in `viz.py`

## 🎯 Use Cases

### 🎓 **Education**
- **Student Learning**: Step-by-step problem breakdown
- **Teacher Tools**: Generate worked examples
- **Homework Help**: Detailed solution explanations

### 🔬 **Research & Development**
- **Mathematical Modeling**: Rapid prototyping of equations
- **Algorithm Verification**: Check symbolic computations
- **Documentation**: Generate LaTeX for papers

### 💼 **Professional**
- **Engineering Calculations**: Verify design computations
- **Data Science**: Mathematical model development
- **API Integration**: Embed math solving in applications

## 🤝 Contributing

1. **Fork the Repository**: Create your own copy
2. **Create Feature Branch**: `git checkout -b feature/amazing-feature`
3. **Make Changes**: Implement your improvements
4. **Add Tests**: Ensure comprehensive test coverage
5. **Submit Pull Request**: Describe your changes

### Development Setup

```bash
# Clone and setup development environment
git clone <your-fork>
cd mathviz
source .venv/bin/activate
cd mathviz
pip install -e .[dev,web,viz]

# Run tests to verify setup
python run_mathviz.py --tests

# Start development
python run_mathviz.py --streamlit
```

## 📈 Performance

- **Typical Response Time**: < 1 second for basic problems
- **Memory Usage**: ~50MB baseline + problem complexity
- **Scalability**: Stateless design supports horizontal scaling
- **Batch Processing**: API supports multiple problems

## 🔧 Configuration

### Environment Variables
- `MATHVIZ_LOG_LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `MATHVIZ_TIMEOUT`: Problem solving timeout (default: 30s)
- `MATHVIZ_CACHE_SIZE`: Solution cache size (default: 100)

### Advanced Usage

```python
from mathviz import MathVizPipeline
from mathviz.schemas import MathProblem, Variable

# Custom pipeline configuration
pipeline = MathVizPipeline()

# Manual problem construction
problem = MathProblem(
    problem_text="Solve for x: ax + b = c",
    problem_type="algebraic",
    variables=[Variable(name="x", domain="real")],
    goal="solve for x"
)

result = pipeline.solver.solve(problem)
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **SymPy**: Symbolic mathematics library
- **Streamlit**: Web application framework  
- **FastAPI**: Modern API framework
- **Plotly**: Interactive visualization library
- **Pydantic**: Data validation library

---

**🚀 Ready to solve some math problems? Start with:**

```bash
python run_mathviz.py --streamlit
```

**Visit the web interface and try solving: "Find the derivative of x² + 3x + 2"**
