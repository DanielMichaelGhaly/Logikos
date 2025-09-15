"""
Streamlit frontend application for MathViz - Interactive mathematical problem solving interface.
"""

import streamlit as st
import plotly.graph_objects as go
import json
from typing import Dict, Any, Optional
import traceback

# Import MathViz components
from .pipeline import MathVizPipeline
from .schemas import MathProblem, MathSolution
from .validator import ValidationError

# Configure Streamlit page
st.set_page_config(
    page_title="MathViz - AI Math Problem Solver",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .problem-input {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #007bff;
    }
    
    .solution-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    
    .step-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

class MathVizApp:
    """Main Streamlit application class for MathViz."""
    
    def __init__(self):
        """Initialize the MathViz application."""
        self.pipeline = MathVizPipeline()
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize session state variables."""
        if 'solution_history' not in st.session_state:
            st.session_state.solution_history = []
        if 'current_solution' not in st.session_state:
            st.session_state.current_solution = None
        if 'problem_examples' not in st.session_state:
            st.session_state.problem_examples = [
                "Solve for x: 2x + 5 = 13",
                "Find the derivative of x^2 + 3x",
                "Integrate 2x + 1",
                "Find the roots of x^2 - 5x + 6",
                "Differentiate sin(x) + cos(x)",
                "Solve the system: x + y = 5, 2x - y = 1"
            ]
    
    def run(self):
        """Run the main Streamlit application."""
        self.render_header()
        self.render_sidebar()
        self.render_main_content()
    
    def render_header(self):
        """Render the application header."""
        st.markdown('<h1 class="main-header">🧮 MathViz</h1>', unsafe_allow_html=True)
        st.markdown("""
        <p style="text-align: center; font-size: 1.2rem; color: #666;">
        AI-powered mathematical problem solver with step-by-step explanations and visualizations
        </p>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls and options."""
        with st.sidebar:
            st.header("🔧 Controls")
            
            # Problem examples
            st.subheader("📚 Example Problems")
            selected_example = st.selectbox(
                "Choose an example:",
                [""] + st.session_state.problem_examples,
                index=0
            )
            
            if selected_example and st.button("Load Example"):
                st.session_state.problem_text = selected_example
                st.experimental_rerun()
            
            st.divider()
            
            # Visualization options
            st.subheader("📊 Visualization")
            show_steps = st.checkbox("Show step-by-step breakdown", value=True)
            show_reasoning = st.checkbox("Show detailed reasoning", value=True)
            show_plot = st.checkbox("Show interactive plots", value=True)
            show_latex = st.checkbox("Show LaTeX output", value=False)
            
            # Store in session state
            st.session_state.viz_options = {
                'show_steps': show_steps,
                'show_reasoning': show_reasoning,
                'show_plot': show_plot,
                'show_latex': show_latex
            }
            
            st.divider()
            
            # Solution history
            st.subheader("📜 Recent Solutions")
            if st.session_state.solution_history:
                for i, hist_item in enumerate(reversed(st.session_state.solution_history[-5:])):
                    if st.button(f"📋 {hist_item['problem'][:30]}...", key=f"hist_{i}"):
                        st.session_state.current_solution = hist_item['solution']
                        st.experimental_rerun()
            else:
                st.write("No recent solutions")
            
            if st.button("🗑️ Clear History"):
                st.session_state.solution_history = []
                st.experimental_rerun()
    
    def render_main_content(self):
        """Render the main content area."""
        # Problem input section
        self.render_problem_input()
        
        # Solution display section
        if st.session_state.current_solution:
            self.render_solution_display()
    
    def render_problem_input(self):
        """Render the problem input interface."""
        st.header("📝 Enter Your Math Problem")
        
        # Problem input
        problem_text = st.text_area(
            "Type your mathematical problem here:",
            height=100,
            placeholder="e.g., Solve for x: 2x + 5 = 13",
            key="problem_input"
        )
        
        # Input validation and solve buttons
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("🔍 Validate", type="secondary"):
                if problem_text.strip():
                    self.validate_problem(problem_text)
                else:
                    st.error("Please enter a problem first!")
        
        with col2:
            if st.button("🚀 Solve", type="primary"):
                if problem_text.strip():
                    self.solve_problem(problem_text)
                else:
                    st.error("Please enter a problem first!")
        
        # Quick examples
        st.write("**Quick examples:** Click to insert")
        example_cols = st.columns(3)
        quick_examples = [
            "x + 5 = 10",
            "d/dx(x^2)",
            "∫(2x + 1)dx"
        ]
        
        for i, example in enumerate(quick_examples):
            with example_cols[i]:
                if st.button(f"📌 {example}", key=f"quick_{i}"):
                    st.session_state.problem_input = example
                    st.experimental_rerun()
    
    def validate_problem(self, problem_text: str):
        """Validate a mathematical problem."""
        try:
            with st.spinner("Validating problem..."):
                # Parse the problem first
                parsed_problem = self.pipeline.parser.parse(problem_text)
                
                # Validate
                is_valid = self.pipeline.validator.validate(parsed_problem)
                
                if is_valid:
                    st.success("✅ Problem is valid and ready to solve!")
                    
                    # Show parsed problem details
                    with st.expander("📋 Parsed Problem Details"):
                        st.write(f"**Type:** {parsed_problem.problem_type}")
                        st.write(f"**Goal:** {parsed_problem.goal}")
                        
                        if parsed_problem.variables:
                            st.write("**Variables:**")
                            for var in parsed_problem.variables:
                                st.write(f"  - {var.name} (domain: {var.domain})")
                        
                        if parsed_problem.equations:
                            st.write("**Equations:**")
                            for i, eq in enumerate(parsed_problem.equations):
                                st.write(f"  {i+1}. {eq.left_side} = {eq.right_side}")
                
        except ValidationError as e:
            st.error(f"❌ Validation failed: {str(e)}")
            
            # Show validation errors
            errors = self.pipeline.validator.get_validation_errors()
            if errors:
                with st.expander("🔍 Validation Errors"):
                    for error in errors:
                        st.write(f"• {error}")
        
        except Exception as e:
            st.error(f"❌ Unexpected error during validation: {str(e)}")
    
    def solve_problem(self, problem_text: str):
        """Solve a mathematical problem."""
        try:
            with st.spinner("🧠 Solving problem..."):
                # Solve the problem
                solution = self.pipeline.process(problem_text)
                
                # Store in session state
                st.session_state.current_solution = solution
                
                # Add to history
                st.session_state.solution_history.append({
                    'problem': problem_text,
                    'solution': solution,
                    'timestamp': str(st.session_state.get('timestamp', 'now'))
                })
                
                st.success("✅ Problem solved successfully!")
                
        except Exception as e:
            st.error(f"❌ Error solving problem: {str(e)}")
            
            # Show detailed error in expander
            with st.expander("🔍 Error Details"):
                st.code(traceback.format_exc())
    
    def render_solution_display(self):
        """Render the solution display section."""
        solution = st.session_state.current_solution
        viz_options = st.session_state.get('viz_options', {})
        
        st.header("🎯 Solution")
        
        # Solution overview
        st.markdown('<div class="solution-section">', unsafe_allow_html=True)
        
        # Final answer prominently displayed
        if solution.final_answer:
            st.subheader("📋 Final Answer")
            
            if isinstance(solution.final_answer, dict):
                if 'solutions' in solution.final_answer:
                    st.success(f"**Solution:** {solution.final_answer['solutions']}")
                elif 'derivative' in solution.final_answer:
                    st.success(f"**Derivative:** {solution.final_answer['derivative']}")
                elif 'integral' in solution.final_answer:
                    st.success(f"**Integral:** {solution.final_answer['integral']} + C")
                else:
                    st.success(f"**Result:** {solution.final_answer}")
            else:
                st.success(f"**Result:** {solution.final_answer}")
        
        # Step-by-step breakdown
        if viz_options.get('show_steps', True) and solution.solution_steps:
            self.render_solution_steps(solution)
        
        # Detailed reasoning
        if viz_options.get('show_reasoning', True) and solution.reasoning:
            self.render_reasoning(solution)
        
        # Interactive visualization
        if viz_options.get('show_plot', True):
            self.render_interactive_plot(solution)
        
        # LaTeX output
        if viz_options.get('show_latex', False) and solution.visualization:
            self.render_latex_output(solution)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_solution_steps(self, solution: MathSolution):
        """Render the step-by-step solution breakdown."""
        st.subheader("🔄 Solution Steps")
        
        for i, step in enumerate(solution.solution_steps):
            with st.expander(f"Step {i+1}: {step.get('operation', 'Unknown').replace('_', ' ').title()}", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Before:**")
                    st.code(step.get('expression_before', 'N/A'))
                
                with col2:
                    st.write("**After:**")
                    st.code(step.get('expression_after', 'N/A'))
                
                if step.get('justification'):
                    st.write("**Explanation:**")
                    st.info(step['justification'])
    
    def render_reasoning(self, solution: MathSolution):
        """Render the detailed reasoning."""
        st.subheader("🧠 Detailed Reasoning")
        
        # Split reasoning into paragraphs for better readability
        reasoning_parts = solution.reasoning.split('\n\n')
        
        for part in reasoning_parts:
            if part.strip():
                if part.startswith('**'):
                    st.markdown(part)
                else:
                    st.write(part)
    
    def render_interactive_plot(self, solution: MathSolution):
        """Render interactive Plotly visualization."""
        st.subheader("📊 Interactive Visualization")
        
        try:
            # Create a trace from solution steps for visualization
            from .trace import StepTrace, Step
            
            # Reconstruct trace from solution steps
            trace = StepTrace(problem_id=f"streamlit_{hash(solution.problem.problem_text)}")
            
            for step_data in solution.solution_steps:
                step = Step(
                    step_id=step_data.get('step_id', 0),
                    operation=step_data.get('operation', 'unknown'),
                    expression_before=step_data.get('expression_before', ''),
                    expression_after=step_data.get('expression_after', ''),
                    justification=step_data.get('justification', '')
                )
                trace.add_step(step)
            
            # Generate plot
            fig = self.pipeline.visualizer.generate_plotly_figure(trace, solution)
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("📈 Interactive plot not available for this problem type.")
                
        except Exception as e:
            st.warning(f"Could not generate interactive plot: {str(e)}")
    
    def render_latex_output(self, solution: MathSolution):
        """Render LaTeX mathematical output."""
        st.subheader("📐 LaTeX Output")
        
        if solution.visualization:
            st.latex(solution.visualization)
            
            # Provide raw LaTeX for copying
            with st.expander("📋 Raw LaTeX Code"):
                st.code(solution.visualization, language="latex")
        else:
            st.info("LaTeX output not available for this solution.")


def main():
    """Main function to run the Streamlit app."""
    app = MathVizApp()
    app.run()


if __name__ == "__main__":
    main()