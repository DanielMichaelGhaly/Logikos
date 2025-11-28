=======
# Logikos - AI-Enhanced Mathematical Problem Solver

**🔧 Restructured Architecture** - Clean separation of AI, SymPy verification, and visualization components for better collaboration and maintainability.

## Overview

Logikos combines AI-powered mathematical explanations with SymPy's symbolic computation for accurate and educational problem solving. The system provides:

- **🤖 AI Explanations**: Step-by-step solutions using Nemotron model
- **🔢 SymPy Verification**: Mathematical accuracy guaranteed by symbolic computation
- **✅ Cross-Validation**: AI solutions are automatically verified against SymPy results
- **📊 Rich Visualization**: LaTeX formatting and interactive displays

## Quick Start

### Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- Ollama (for AI features, optional)

### Installation

1. **Activate virtual environment**:
   ```bash
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the application**:
   ```bash
   # Command-line interface
   ./start.sh

   # Or test directly
   python run_workflow.py "solve 2x+5=0" --no-ai

   # Full web application (if available)
   ./start.sh --full-app
   ```

## Usage Examples

### Basic Problem Solving
```bash
# Solve algebraic equations
python run_workflow.py "solve 2x+5=0"

# Find roots of polynomials  
python run_workflow.py "find roots of x^2-4"

# Calculus operations
python run_workflow.py "derivative of x^2 + 3x"
python run_workflow.py "integral of sin(x)"

# Expression simplification
python run_workflow.py "simplify (x^2-1)/(x-1)"
```

### Advanced Options
```bash
# Disable AI (SymPy only)
python run_workflow.py "solve x^2-9=0" --no-ai

# Verbose output with detailed steps
python run_workflow.py "derivative of cos(x)" --verbose

# Save HTML visualization
python run_workflow.py "solve 3x-7=0" --save-html solution.html

# Save JSON results  
python run_workflow.py "integral of x^2" --save-json results.json
```

## Architecture

```
Logikos/
├── run_workflow.py          # Main entry point
├── requirements.txt         # Dependencies
├── README.md               # This file
├── ai/                     # AI processing components
│   ├── ai_solver.py        # AI model interface
│   └── response_parser.py  # Parse AI responses
├── sympy_backend/          # Mathematical verification
│   ├── expression_parser.py # Enhanced math parsing
│   ├── solver.py          # SymPy solving logic
│   └── verifier.py        # AI/SymPy cross-validation
├── visualization/          # Display and formatting
│   ├── latex_formatter.py # LaTeX output
│   └── step_visualizer.py # Step-by-step display
└── tests/                 # Integration tests
    └── test_integration.py
```

### Key Components

#### 1. Enhanced Math Parser (`sympy_backend/expression_parser.py`)
**✅ FIXED**: Now correctly handles natural language input like "solve 2x+5=0"

- Converts natural language to SymPy expressions
- Supports various input formats (equations, derivatives, integrals)
- Robust preprocessing for mathematical notation

#### 2. SymPy Solver (`sympy_backend/solver.py`) 
- Pure symbolic computation for mathematical accuracy
- Detailed step-by-step solving process
- LaTeX output generation

#### 3. AI Integration (`ai/ai_solver.py`)
- Interface to AI models via Ollama (currently Nemotron)
- Contextual prompts for different problem types
- Graceful fallback when AI unavailable

#### 4. Verification System (`sympy_backend/verifier.py`)
**✅ NEW**: Cross-validates AI solutions against SymPy results

- Extracts numerical solutions from AI responses
- Compares against SymPy results with tolerance
- Confidence scoring and error detection

## Problem Types Supported

| Type | Example Input | SymPy Support | AI Support |
|------|---------------|---------------|------------|
| **Linear Equations** | `solve 2x+5=0` | ✅ | ✅ |
| **Quadratic Equations** | `solve x^2-4=0` | ✅ | ✅ |
| **Root Finding** | `find roots of x^3-8` | ✅ | ✅ |
| **Derivatives** | `derivative of x^2+sin(x)` | ✅ | ✅ |
| **Integrals** | `integral of cos(x)` | ✅ | ✅ |
| **Simplification** | `simplify (x^2-1)/(x-1)` | ✅ | ✅ |

## Verification System

The system automatically cross-validates AI solutions:

```
🤖 AI: "x = -2.5"
🔢 SymPy: [-5/2]
✅ Verification: MATCH (confidence: 0.95)
```

Status indicators:
- ✅ **MATCH**: AI solution matches SymPy exactly
- ❌ **MISMATCH**: Solutions differ (potential AI error)
- ⚠️ **PARTIAL_MATCH**: Similar but not exact
- ❓ **INCONCLUSIVE**: Cannot verify (complex expressions)

## Development

### Running Tests
```bash
# Basic integration test
python tests/test_integration.py

# Full test suite (if pytest installed)
pytest tests/ -v
```

### Project Structure Benefits
- **🔧 Modular**: Each component can be developed independently
- **🧰 Testable**: Clear interfaces enable comprehensive testing
- **🚀 Scalable**: Easy to add new AI models or math operations
- **👥 Collaborative**: Team members can work on specific components

### Adding New Features

#### New Math Operations
1. Add parsing patterns to `sympy_backend/expression_parser.py`
2. Implement solver logic in `sympy_backend/solver.py`
3. Add verification patterns to `sympy_backend/verifier.py`

#### New AI Models
1. Create new solver class in `ai/` directory
2. Implement same interface as `AISolver`
3. Update `run_workflow.py` to support new model

## Troubleshooting

### Common Issues

**"SymPy error: Sympify of expression 'could not parse'"**
- ✅ **FIXED**: Enhanced parser now handles natural language input
- Use proper mathematical notation: `2*x` instead of `2x` in complex expressions

**"Ollama service not available"**
- Start Ollama: `ollama serve`
- Or use `--no-ai` flag for SymPy-only mode

**Import errors**
- Ensure virtual environment is activated: `source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`

### Getting Help
- Check the integration tests for usage examples
- Use `--verbose` flag for detailed debugging output
- Each component has standalone test functions

## Original vs. New Architecture

### Before (Issues)
- ❌ Parser failed on "solve 2x+5=0"
- ❌ Multiple overlapping scripts and directories
- ❌ No systematic AI-SymPy verification
- ❌ Difficult for team collaboration

### After (Solutions)
- ✅ Enhanced parser handles natural language
- ✅ Clean modular architecture
- ✅ Automatic AI-SymPy cross-validation
- ✅ Clear separation of concerns for collaboration

---

**🎯 Ready to use**: The core SymPy functionality works immediately. AI features require Ollama setup but gracefully degrade when unavailable.
>>>>>>> 4f8d194d9e58e27369b5b7f39aa4fb0455cf55fc
