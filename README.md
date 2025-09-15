# MathViz: Problem Solver AI

MathViz is an AI-powered framework for solving, explaining, and visualizing problems in **math, physics, and chemistry**.  
The system takes a natural language problem as input, parses it into a structured **intermediate representation (JSON schema)**, validates the data, solves it symbolically/numerically, generates step-by-step reasoning, and produces **visualizations and interactive simulations**.

---

## 🔹 Core Idea
The project bridges **textual problem descriptions** and **interactive visual understanding**:
1. **User Input (NL prompt)** → converted to **structured JSON schema**.
2. **Validation Layer** ensures physical/mathematical sanity (units, domains, negative numbers, etc.).
3. **Solver Layer** computes solutions using SymPy/NumPy and produces a **step trace**.
4. **Step Generator** transforms the solver trace into human-readable reasoning.
5. **Visualization Builder** creates LaTeX equations, 2D/3D graphs, and optional physics simulations.
6. **Frontend** renders answers, reasoning, and visualizations with interactive feedback (revise/resolve loop).

---

## 🔹 Architecture


mathviz/
├─ pyproject.toml
├─ README.md
└─ src/
└─ mathviz/
├─ init.py        # package init
├─ pipeline.py        # orchestration of all steps
├─ schemas.py         # Pydantic models (problem JSON schema)
├─ parser.py          # prompt → schema (rules/regex/LLM hook)
├─ validator.py       # sanity checks (domains, units)
├─ solver.py          # SymPy/NumPy solver + step trace
├─ trace.py           # Step + StepTrace dataclasses
├─ reasoning.py       # explanation generator from trace
└─ viz.py             # visualization builder (LaTeX/HTML/Plotly/Manim)

---

## 🔹 Pipeline

1. **Parsing & Schema**  
   - Convert user prompt → JSON schema.  
   - Enforce schema with Pydantic.  
   - Optionally use an LLM or lightweight regex/rule parser.

2. **Validation**  
   - Check numeric domains, units, and variable consistency.  
   - Python libraries: `pint` for units, custom domain checks.

3. **Solver**  
   - Symbolic math with **SymPy** (algebra, differentiation, integration).  
   - Numerical math with **NumPy**.  
   - Produces **step traces** for later reasoning.

4. **Step Generator**  
   - Rule-based text generator (future: LLM-enhanced).  
   - Converts solver traces → step-by-step reasoning.

5. **Visualization**  
   - **Math rendering:** LaTeX (KaTeX/MathJax).  
   - **2D Graphs:** Plotly.js.  
   - **Physics Simulations:** matter.js / planck.js.  
   - **Animations:** Manim or TikZ.  
   - Optional: 3D scenes with three.js.

6. **Frontend**  
   - Framework: **React + TypeScript + TailwindCSS**.  
   - Features:  
     - Problem editor with schema validation.  
     - Interactive visualizations.  
     - Simulator controls (play, pause, reset).  

7. **Backend**  
   - **FastAPI** orchestrating pipeline steps.  
   - Data storage: **Postgres** (structured problems) or **Vector DB** for retrieval-augmented generation (formulas, constants).  

---

## 🔹 Example Flow

1. User asks:  
   > "Find the roots of x² - 5x + 6 and plot the function."

2. System converts → JSON:
   ```json
   {
     "problem_type": "polynomial",
     "equation": "x^2 - 5x + 6",
     "goal": "roots",
     "visualize": true
   }

3.	Validation checks input.

4.	Solver uses SymPy → roots = [2, 3].

5.	Step Generator explains factoring:
    •	“We can factor x² - 5x + 6 as (x - 2)(x - 3). The roots are 2 and 3.”

6.	Visualization produces:
	•	Equation in LaTeX.
	•	2D plot with marked roots.

🔹 Roadmap
	•	Implement JSON schema parser.
	•	Add validation layer with units/domain checks.
	•	Build solver + step tracer with SymPy/NumPy.
	•	Generate rule-based reasoning text.
	•	Integrate LaTeX + Plotly.js visualization.
	•	Build React + Tailwind frontend.
	•	Add optional physics simulator (matter.js).
	•	Extend to chemistry and physics problem types.

⸻

🔹 Tech Stack
	•	Backend: FastAPI, Pydantic, SymPy, NumPy, Pint
	•	Frontend: React, TypeScript, Tailwind, Plotly.js, KaTeX/MathJax, Manim, matter.js
	•	Storage: Postgres + VectorDB (for RAG)
	•	Language Layer: Rule parser + optional LLM (fine-tuned or external API)

⸻

🔹 Output to User
	•	Answer (numeric/symbolic)
	•	Step-by-step reasoning
	•	Visualizations (graphs, diagrams, simulations)
	•	Interactive simulator (optional for physics/chemistry)

---

# 2. Initialization Prompt

Here’s a **prompt** you can paste into an AI assistant to bootstrap the project:

```plaintext
You are building a project called **MathViz**, an AI-powered problem solver for math, physics, and chemistry.  

Project requirements:
- Input: natural language problem.
- Parse into JSON schema (via regex/rules/LLM).
- Validate inputs (units, domains, numeric sanity).
- Solve problem using SymPy/NumPy (algebra, calculus, numerical).
- Generate step-by-step reasoning from solver trace.
- Visualize results using LaTeX (math), Plotly.js (2D plots), and optionally Manim/matter.js (animations, physics sims).
- Frontend: React + TypeScript + TailwindCSS.
- Backend: FastAPI with orchestration pipeline.
- Data storage: Postgres or Vector DB for RAG.

Project structure (Python backend):

mathviz/
├─ pyproject.toml
├─ README.md
└─ src/mathviz/
├─ init.py
├─ pipeline.py
├─ schemas.py
├─ parser.py
├─ validator.py
├─ solver.py
├─ trace.py
├─ reasoning.py
└─ viz.py

Tasks to initialize:
1. Generate `pyproject.toml` with dependencies (FastAPI, Pydantic, SymPy, NumPy, Pint).  
2. Scaffold `src/mathviz/` files with stub functions/classes.  
3. Add README.md with project description.  
4. Ensure all modules importable and pipeline skeleton runs with dummy flow.  

