# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Logikos is being transformed into a streamlined AI-enhanced mathematical chat assistant. The new vision focuses on a ChatGPT-like interface with intelligent question processing, dual verification (SymPy + AI), and embedded visualizations. The system uses Ollama exclusively for AI features and provides high-confidence mathematical solutions with educational explanations.

## 🚀 Current Transformation (September 2025)

**Status**: Major architectural transformation in progress
**Goal**: Clean, professional mathematical chat assistant with embedded visualizations
**AI Model**: Ollama (nvidia model) only - no OpenAI fallbacks

## 🎯 New Vision: Streamlined Mathematical Chat Assistant

### Target Architecture
```
User Input → AI Classifier → [SymPy Ground Truth + AI Reasoning] → Confidence Compare → Chat UI
                                        ↓
                            Visualization Detector → Plot Generator → Embedded Display
```

### Key Features
1. **Smart Question Classification**: AI-powered conversion of natural language to structured JSON
2. **Dual Processing**: SymPy (ground truth) + AI reasoning (educational explanations)
3. **Confidence System**: Compare results and show confidence only for supported question types
4. **Embedded Visualizations**: Auto-detect representable problems and embed interactive plots
5. **ChatGPT-like Interface**: Professional chat UI with LaTeX rendering

## 📋 Transformation Phases

### Phase 1: Architecture Simplification ⏳
**Status**: Pending
**Goal**: Clean up existing complexity, establish clear structure
- Remove redundant directories and components
- Consolidate into 4 core modules: input_processor, solvers, visualization, frontend
- Update dependencies (Ollama only, remove OpenAI)
- Establish clear data flow

### Phase 2: Smart Input Processing ⏳
**Status**: Pending
**Goal**: AI-powered question classification to structured JSON
- Build InputClassifier using Ollama
- Define JSON schema for question types (solve, factor, derivative, integral, plot)
- Create question type detection patterns
- Handle context and visualization hints

### Phase 3: Dual Processing Engine ⏳
**Status**: Pending
**Goal**: Parallel SymPy (ground truth) + AI reasoning with confidence
- Enhanced SymPy backend for ultra-accurate mathematical solving
- AI reasoning engine for educational step-by-step explanations
- Confidence system that compares results only for supported question types
- High/Medium/Low confidence display scale

### Phase 4: Smart Visualization System ⏳
**Status**: Pending
**Goal**: Auto-detect and embed visualizable problems
- Representability detector (AI decides if problem is visualizable)
- Support for function plots, contour maps, 3D surfaces
- Gnuplot integration for embeddable plots
- Chat-compatible visualization boxes

### Phase 5: ChatGPT-like Frontend ⏳
**Status**: Pending
**Goal**: Professional chat interface with mathematical rendering
- React-based chat UI similar to ChatGPT
- LaTeX rendering for mathematical expressions
- Embedded visualization boxes in chat
- Clean, professional design (no emojis)
- Responsive and accessible interface

## 🛠️ Development Commands

### Current Development (Legacy System)
**Note**: These commands work with the existing system during transformation
```bash
# Test existing SymPy functionality
python run_workflow.py "solve 2x+5=0" --no-ai
python run_workflow.py "derivative of x^2" --no-ai

# Start existing backend
cd mathviz-backend && python main.py
```

### New System (Post-Transformation)
**Coming**: New streamlined commands for the chat interface
```bash
# Start new chat assistant (planned)
python start_chat.py

# Start new backend API (planned)
python backend/main.py

# Test new input classifier (planned)
python test_classifier.py "factor x^2-4 and explain clearly"
```

### Environment Setup
```bash
# Activate environment
source .venv/bin/activate

# Install dependencies (will be updated during transformation)
pip install -r requirements.txt

# Ollama setup (nvidia model only)
ollama serve
# Pull your nvidia model as needed
```

### Code Quality
```bash
# Format code
black .

# Type checking (when available)
mypy .

# Linting (when available)
ruff check .
```

## 🏗️ New Architecture (Post-Transformation)

### Streamlined Structure
```
logikos/
├── backend/
│   ├── input_processor/     # AI question classification
│   │   ├── classifier.py    # Ollama-powered input parsing
│   │   └── schemas.py       # JSON question schemas
│   ├── solvers/            # Dual processing engines
│   │   ├── sympy_solver.py  # Ground truth mathematical solver
│   │   ├── ai_reasoner.py   # Ollama educational explanations
│   │   └── confidence.py    # Result comparison system
│   ├── visualization/      # Embedded plot generation
│   │   ├── detector.py      # Representability analysis
│   │   ├── generators.py    # Gnuplot/plotting integration
│   │   └── embedder.py      # Chat-compatible visualizations
│   └── api/               # FastAPI endpoints
│       ├── main.py         # API server
│       └── routes.py       # Chat endpoints
├── frontend/              # React chat interface
│   ├── src/
│   │   ├── components/     # Chat UI components
│   │   ├── latex/         # Mathematical rendering
│   │   └── visualizations/ # Embedded plot display
│   └── package.json
└── shared/               # Common utilities and schemas
```

### Core Components

#### 1. Input Processor
- **Purpose**: Convert natural language to structured questions
- **Technology**: Ollama (nvidia model)
- **Output**: JSON schema with question type, expression, context
- **Example**: "factor x^2-4 clearly" → `{"type": "factor", "expression": "x^2-4", "context": "explain clearly"}`

#### 2. Dual Solvers
- **SymPy Solver**: Ultra-accurate ground truth mathematical computation
- **AI Reasoner**: Educational step-by-step explanations using Ollama
- **Confidence System**: Compare results, show confidence only for supported types

#### 3. Visualization System
- **Detector**: AI determines if problem is representable
- **Generators**: Create plots using gnuplot/matplotlib
- **Embedder**: Integrate visualizations into chat interface

#### 4. Chat Interface
- **Design**: ChatGPT-like professional interface
- **Features**: LaTeX rendering, embedded plots, clean UX
- **Technology**: React with mathematical rendering libraries

## 📊 Question Type Examples

### Input Processing Examples
```bash
# User input → JSON classification
"Factor x^2-4 and explain clearly"
→ {"type": "factor", "expression": "x^2-4", "context": "explain clearly", "visualization_hint": false}

"Plot the function f(x) = sin(x) from 0 to 2π"
→ {"type": "plot", "expression": "sin(x)", "domain": "0 to 2π", "visualization_hint": true}

"Solve 2x + 5 = 0 step by step"
→ {"type": "solve", "expression": "2x + 5 = 0", "context": "step by step", "visualization_hint": false}
```

### Supported Operations
- **Algebraic**: solve, factor, expand, simplify
- **Calculus**: derivative, integral, limit
- **Visualization**: plot, graph, contour, 3d surface
- **Educational**: step-by-step explanations, concept clarification

## 🔧 Dependencies (Post-Transformation)

### Core Dependencies
- **SymPy**: Mathematical ground truth computation
- **Ollama**: AI model serving (nvidia model only)
- **FastAPI**: Backend API server
- **React**: Frontend chat interface
- **Gnuplot**: Visualization generation

### Removed Dependencies
- **OpenAI**: No longer used (Ollama exclusive)
- **Complex visualization libraries**: Simplified to gnuplot focus
- **Redundant parsing libraries**: Consolidated approach

## 🚨 Critical Requirements

1. **Ollama Only**: No OpenAI fallbacks or alternatives
2. **Confidence Only for Supported**: Parse AI responses, compare only for known question types
3. **Clean UI**: Professional, ChatGPT-like interface without emojis
4. **Embedded Plots**: Visualizations integrated seamlessly into chat
5. **Simplified Architecture**: Clear data flow, reduced complexity
6. **Educational Focus**: Step-by-step explanations with LaTeX rendering

## 📈 Success Metrics

- **Single Workflow**: Clear path from user input to final output
- **Confidence System**: Working comparison for supported question types
- **Embedded Visualizations**: Interactive plots within chat interface
- **Professional UI**: ChatGPT-quality interface design
- **Ollama Integration**: Smooth AI integration without fallbacks
- **Performance**: Fast response times for both SymPy and AI processing

## 🔄 Current Status

**Transformation Phase**: Architecture Simplification (Phase 1)
**Next Milestone**: Smart Input Processing (Phase 2)
**Estimated Completion**: Progressive implementation across all phases

This transformation will create a focused, professional mathematical chat assistant with intelligent question processing, dual verification, and embedded visualizations - all powered exclusively by Ollama AI integration.