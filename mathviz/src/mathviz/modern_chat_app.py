"""
MathViz Modern Chat Interface - Beautiful mathematical problem solving with stunning UI.

A modern, polished chat interface inspired by the best messaging apps with:
- Beautiful gradients and modern design
- Smooth animations and transitions
- Professional LaTeX rendering
- Interactive Desmos graph integration
- Step-by-step solution display with visual hierarchy
"""

import streamlit as st
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import base64

# Import MathViz components
import sys
import os
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from mathviz.pipeline import MathVizPipeline
from mathviz.schemas import MathSolution

# Try to import educational visualization
try:
    from mathviz.educational_viz import EducationalVisualizer, VisualizationConfig
    EDUCATIONAL_VIZ_AVAILABLE = True
except ImportError:
    print("Warning: Educational visualization not available")
    EDUCATIONAL_VIZ_AVAILABLE = False
    EducationalVisualizer = None
    VisualizationConfig = None

# Configure page with custom styling
st.set_page_config(
    page_title="MathViz | AI Math Tutor",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS styling
def inject_custom_css():
    """Inject modern, beautiful CSS styling."""
    st.markdown("""
    <style>
        /* Import Google Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
        
        /* Global styling */
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stDeployButton {display: none;}
        .stActionButton {display: none;}
        
        /* Main container styling */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 1rem;
            max-width: 900px;
        }
        
        /* Header styling */
        .main-header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        
        .main-title {
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }
        
        .main-subtitle {
            font-size: 1.2rem;
            color: #6b7280;
            font-weight: 400;
        }
        
        /* Chat container styling */
        .chat-container {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 2rem;
            min-height: 500px;
            max-height: 600px;
            overflow-y: auto;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        
        /* Custom scrollbar */
        .chat-container::-webkit-scrollbar {
            width: 8px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: rgba(0, 0, 0, 0.05);
            border-radius: 4px;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 4px;
        }
        
        .chat-container::-webkit-scrollbar-thumb:hover {
            background: rgba(0, 0, 0, 0.3);
        }
        
        /* Message styling */
        .message {
            margin-bottom: 1.5rem;
            animation: fadeInUp 0.5s ease-out;
        }
        
        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(20px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }
        
        /* User message styling */
        .user-message {
            display: flex;
            justify-content: flex-end;
        }
        
        .user-bubble {
            max-width: 75%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 1.25rem;
            border-radius: 20px 20px 8px 20px;
            font-size: 0.95rem;
            line-height: 1.5;
            box-shadow: 0 8px 24px rgba(102, 126, 234, 0.3);
            word-wrap: break-word;
        }
        
        /* Bot message styling */
        .bot-message {
            display: flex;
            justify-content: flex-start;
            align-items: flex-start;
        }
        
        .bot-avatar {
            width: 40px;
            height: 40px;
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            margin-right: 1rem;
            font-size: 1.2rem;
            flex-shrink: 0;
            box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
        }
        
        .bot-bubble {
            max-width: 80%;
            background: #ffffff;
            border: 1px solid rgba(0, 0, 0, 0.08);
            color: #374151;
            padding: 1rem 1.25rem;
            border-radius: 20px 20px 20px 8px;
            font-size: 0.95rem;
            line-height: 1.6;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
            word-wrap: break-word;
        }
        
        /* Solution answer styling */
        .solution-answer {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 1rem 1.25rem;
            border-radius: 12px;
            margin: 1rem 0;
            font-weight: 600;
            font-size: 1.1rem;
            box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3);
            display: flex;
            align-items: center;
        }
        
        .solution-answer::before {
            content: "✨";
            margin-right: 0.75rem;
            font-size: 1.2rem;
        }
        
        /* Solution steps styling */
        .solution-steps {
            margin: 1.5rem 0;
            background: #f9fafb;
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(0, 0, 0, 0.05);
        }
        
        .steps-header {
            font-weight: 600;
            font-size: 1rem;
            color: #374151;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
        }
        
        .steps-header::before {
            content: "📝";
            margin-right: 0.5rem;
        }
        
        .step-item {
            background: white;
            border: 1px solid rgba(0, 0, 0, 0.08);
            border-radius: 10px;
            padding: 1rem;
            margin: 0.75rem 0;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
            transition: all 0.2s ease;
        }
        
        .step-item:hover {
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
            transform: translateY(-1px);
        }
        
        .step-title {
            font-weight: 600;
            color: #1f2937;
            font-size: 0.95rem;
            margin-bottom: 0.5rem;
        }
        
        .step-rule {
            background: #eff6ff;
            color: #1e40af;
            padding: 0.375rem 0.75rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-family: 'JetBrains Mono', monospace;
            margin: 0.5rem 0;
            border-left: 3px solid #3b82f6;
        }
        
        .step-transformation {
            background: #f3f4f6;
            padding: 0.75rem;
            border-radius: 8px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            margin: 0.5rem 0;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .step-arrow {
            color: #6366f1;
            font-weight: bold;
            margin: 0 1rem;
        }
        
        .step-explanation {
            color: #6b7280;
            font-size: 0.85rem;
            line-height: 1.4;
            font-style: italic;
        }
        
        /* Desmos graph styling */
        .desmos-container {
            margin: 1.5rem 0;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.1);
            border: 1px solid rgba(0, 0, 0, 0.05);
        }
        
        .desmos-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem 1.25rem;
            font-weight: 600;
            display: flex;
            align-items: center;
        }
        
        .desmos-header::before {
            content: "📊";
            margin-right: 0.75rem;
        }
        
        .desmos-frame {
            width: 100%;
            height: 400px;
            border: none;
        }
        
        /* Input area styling */
        .input-area {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            padding: 2rem;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        }
        
        /* Suggestions styling */
        .suggestions-container {
            margin-bottom: 1.5rem;
        }
        
        .suggestions-title {
            font-weight: 600;
            color: #374151;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
        }
        
        .suggestions-title::before {
            content: "💡";
            margin-right: 0.5rem;
        }
        
        .suggestion-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 0.75rem;
        }
        
        .suggestion-chip {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
            border: 1px solid rgba(102, 126, 234, 0.2);
            color: #4f46e5;
            padding: 0.75rem 1rem;
            border-radius: 12px;
            font-size: 0.85rem;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
        }
        
        .suggestion-chip:hover {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.15), rgba(118, 75, 162, 0.15));
            border-color: rgba(102, 126, 234, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.2);
        }
        
        /* Typing indicator styling */
        .typing-indicator {
            display: flex;
            align-items: center;
            padding: 1rem 1.25rem;
            background: #f3f4f6;
            border-radius: 20px 20px 20px 8px;
            font-size: 0.9rem;
            color: #6b7280;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        .typing-dots {
            display: inline-block;
            width: 6px;
            height: 6px;
            border-radius: 50%;
            background: #6b7280;
            margin: 0 2px;
            animation: typing 1.4s infinite ease-in-out both;
        }
        
        .typing-dots:nth-child(1) { animation-delay: -0.32s; }
        .typing-dots:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes typing {
            0%, 80%, 100% {
                transform: scale(0.8);
                opacity: 0.5;
            }
            40% {
                transform: scale(1.2);
                opacity: 1;
            }
        }
        
        /* Footer styling */
        .app-footer {
            text-align: center;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.85rem;
            margin-top: 2rem;
            padding: 1rem;
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .main .block-container {
                padding: 1rem;
            }
            
            .main-title {
                font-size: 2rem;
            }
            
            .user-bubble, .bot-bubble {
                max-width: 90%;
            }
            
            .suggestion-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
    """, unsafe_allow_html=True)

class ModernMathChatApp:
    """Modern, beautiful chat interface for MathViz."""
    
    def __init__(self):
        """Initialize the modern chat application."""
        self.pipeline = MathVizPipeline()
        
        # Initialize visualizer if available
        if EDUCATIONAL_VIZ_AVAILABLE:
            self.visualizer = EducationalVisualizer(VisualizationConfig())
        else:
            self.visualizer = None
            
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize session state for chat."""
        if 'messages' not in st.session_state:
            st.session_state.messages = [
                {
                    'id': str(uuid.uuid4()),
                    'type': 'bot',
                    'content': "👋 Hello! I'm your AI-powered math tutor. I can solve equations, find derivatives, compute integrals, and create interactive graphs. What would you like to learn today?",
                    'timestamp': datetime.now(),
                    'has_solution': False
                }
            ]
        
        if 'is_solving' not in st.session_state:
            st.session_state.is_solving = False
            
        if 'suggestions' not in st.session_state:
            st.session_state.suggestions = [
                "Solve: 2x² - 8x + 6 = 0",
                "Find derivative of x³ + 4x² - 2x + 1",
                "Integrate: ∫(3x² + 2x + 1) dx",
                "Factor: x² - 5x + 6",
                "Graph: y = x² - 4x + 3",
                "Find critical points of x³ - 3x² + 2"
            ]
    
    def run(self):
        """Run the modern chat application."""
        # Inject custom CSS
        inject_custom_css()
        
        # Header
        self.render_header()
        
        # Chat container
        self.render_chat_messages()
        
        # Input area
        self.render_input_area()
        
        # Footer
        self.render_footer()
    
    def render_header(self):
        """Render the beautiful header."""
        st.markdown("""
        <div class="main-header">
            <h1 class="main-title">🧮 MathViz</h1>
            <p class="main-subtitle">Your AI-powered mathematical companion with step-by-step solutions</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_chat_messages(self):
        """Render all chat messages in a modern container."""
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        for message in st.session_state.messages:
            if message['type'] == 'user':
                self.render_user_message(message)
            else:
                self.render_bot_message(message)
        
        # Show typing indicator if solving
        if st.session_state.is_solving:
            self.render_typing_indicator()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_user_message(self, message: Dict):
        """Render a user message with modern styling."""
        st.markdown(f"""
        <div class="message user-message">
            <div class="user-bubble">
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_bot_message(self, message: Dict):
        """Render a bot message with modern styling."""
        avatar = "🧮" if message.get('has_solution') else "🤖"
        
        content_html = f'<div class="bot-bubble">{message["content"]}'
        
        # Add solution if present
        if message.get('solution'):
            solution = message['solution']
            content_html += self.format_modern_solution(solution)
        
        # Add Desmos graph if available
        if message.get('desmos_url'):
            content_html += f"""
            <div class="desmos-container">
                <div class="desmos-header">Interactive Graph</div>
                <iframe src="{message['desmos_url']}" class="desmos-frame" title="Interactive Math Graph"></iframe>
            </div>
            """
        
        content_html += '</div>'
        
        st.markdown(f"""
        <div class="message bot-message">
            <div class="bot-avatar">{avatar}</div>
            {content_html}
        </div>
        """, unsafe_allow_html=True)
    
    def format_modern_solution(self, solution: MathSolution) -> str:
        """Format solution with modern, beautiful styling."""
        html = ""
        
        # Final answer with beautiful styling
        if solution.final_answer:
            answer_text = self.format_final_answer(solution.final_answer)
            html += f'<div class="solution-answer">{answer_text}</div>'
        
        # Solution steps with modern styling
        if solution.solution_steps and len(solution.solution_steps) > 0:
            html += '<div class="solution-steps">'
            html += '<div class="steps-header">Step-by-Step Solution</div>'
            
            # Show detailed steps with enhanced formatting
            steps_to_show = min(6, len(solution.solution_steps))
            for i in range(steps_to_show):
                step = solution.solution_steps[i]
                
                # Handle both old and new step formats
                if 'description' in step:
                    # New detailed format
                    operation = step.get('operation', '').replace('_', ' ').title()
                    description = step.get('reasoning', step.get('description', ''))
                    rule_formula = step.get('rule_formula')
                    
                    # Get mathematical expressions
                    input_state = step.get('input_state', {})
                    output_state = step.get('output_state', {})
                    
                    before_expr = input_state.get('expression', '')
                    after_expr = output_state.get('expression', '')
                    
                    step_html = f'<div class="step-item">'
                    step_html += f'<div class="step-title">{i+1}. {operation}</div>'
                    
                    if rule_formula:
                        step_html += f'<div class="step-rule">{rule_formula}</div>'
                    
                    if before_expr and after_expr:
                        step_html += f'''
                        <div class="step-transformation">
                            <span>{before_expr}</span>
                            <span class="step-arrow">→</span>
                            <span>{after_expr}</span>
                        </div>
                        '''
                    
                    step_html += f'<div class="step-explanation">{description}</div>'
                    step_html += '</div>'
                    
                    html += step_html
                else:
                    # Legacy format fallback
                    operation = step.get('operation', '').replace('_', ' ').title()
                    justification = step.get('justification', '')
                    
                    step_html = f'<div class="step-item">'
                    step_html += f'<div class="step-title">{i+1}. {operation}</div>'
                    step_html += f'<div class="step-explanation">{justification}</div>'
                    step_html += '</div>'
                    
                    html += step_html
            
            if len(solution.solution_steps) > steps_to_show:
                remaining = len(solution.solution_steps) - steps_to_show
                html += f'<div style="text-align: center; color: #6b7280; font-size: 0.85rem; margin-top: 1rem; font-style: italic;">+ {remaining} more steps available</div>'
            
            html += '</div>'
        
        return html
    
    def format_final_answer(self, answer) -> str:
        """Format the final answer for modern display."""
        if isinstance(answer, dict):
            if 'solutions' in answer:
                return f"Solution: x = {answer['solutions']}"
            elif 'derivative' in answer:
                return f"f'(x) = {answer['derivative']}"
            elif 'integral' in answer:
                return f"∫f(x)dx = {answer['integral']} + C"
            elif 'type' in answer:
                if answer['type'] == 'algebraic_solution':
                    solutions = answer.get('solutions', {})
                    if solutions:
                        return 'Solutions: ' + ', '.join([f"{k} = {v}" for k, v in solutions.items()])
                elif answer['type'] == 'derivative':
                    return f"f'(x) = {answer.get('derivative', 'N/A')}"
                elif answer['type'] == 'integral':
                    return f"∫f(x)dx = {answer.get('integral', 'N/A')} + C"
            
            return str(answer)
        else:
            return str(answer)
    
    def render_typing_indicator(self):
        """Show a modern typing indicator."""
        st.markdown("""
        <div class="message bot-message">
            <div class="bot-avatar">🤖</div>
            <div class="typing-indicator">
                MathViz is solving your problem
                <span class="typing-dots"></span>
                <span class="typing-dots"></span>
                <span class="typing-dots"></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_input_area(self):
        """Render the modern input area."""
        st.markdown('<div class="input-area">', unsafe_allow_html=True)
        
        # Quick suggestions (show only if no messages or just welcome message)
        if len(st.session_state.messages) <= 1:
            st.markdown("""
            <div class="suggestions-container">
                <div class="suggestions-title">Try these examples:</div>
                <div class="suggestion-grid">
            """, unsafe_allow_html=True)
            
            cols = st.columns(3)
            for i, suggestion in enumerate(st.session_state.suggestions[:6]):
                col_idx = i % 3
                with cols[col_idx]:
                    if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                        self.handle_user_message(suggestion)
                        st.rerun()
            
            st.markdown('</div></div>', unsafe_allow_html=True)
        
        # Chat input
        col1, col2 = st.columns([5, 1])
        
        with col1:
            user_input = st.text_input(
                "",
                placeholder="Type your math problem here... (e.g., 'Find the derivative of x² + 3x + 1')",
                key="chat_input",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button(
                "Send ✨", 
                use_container_width=True, 
                disabled=st.session_state.is_solving,
                type="primary"
            )
        
        # Handle input submission
        if (send_button or user_input) and user_input.strip() and not st.session_state.is_solving:
            self.handle_user_message(user_input.strip())
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_footer(self):
        """Render the app footer."""
        st.markdown("""
        <div class="app-footer">
            💬 Chat with MathViz • 🧮 Get step-by-step solutions • 📊 View interactive graphs
            <br>
            <small>Powered by SymPy, Streamlit, and AI • Made with ❤️ for mathematics</small>
        </div>
        """, unsafe_allow_html=True)
    
    def handle_user_message(self, message: str):
        """Handle a new user message."""
        # Add user message
        user_msg = {
            'id': str(uuid.uuid4()),
            'type': 'user',
            'content': message,
            'timestamp': datetime.now(),
            'has_solution': False
        }
        st.session_state.messages.append(user_msg)
        
        # Clear input
        st.session_state.chat_input = ""
        
        # Start solving
        st.session_state.is_solving = True
        
        # Solve the problem
        self.solve_and_respond(message)
    
    def solve_and_respond(self, problem_text: str):
        """Solve the math problem and create a modern response."""
        try:
            # Solve the problem
            solution = self.pipeline.process(problem_text)
            
            # Create educational visualization if available
            viz_result = None
            if self.visualizer:
                try:
                    viz_result = self.visualizer.create_solution_visualization(solution)
                except Exception as e:
                    print(f"Visualization failed: {e}")
            
            # Create bot response with modern styling
            response_content = f"Perfect! I've solved your problem: **{problem_text}**"
            
            # Get Desmos URL if available
            desmos_url = None
            if viz_result and viz_result.get('success') and viz_result.get('visualizations'):
                desmos_viz = viz_result['visualizations'].get('desmos')
                if desmos_viz:
                    desmos_url = desmos_viz.get('url')
            
            bot_msg = {
                'id': str(uuid.uuid4()),
                'type': 'bot',
                'content': response_content,
                'timestamp': datetime.now(),
                'has_solution': True,
                'solution': solution,
                'desmos_url': desmos_url
            }
            
            st.session_state.messages.append(bot_msg)
            
        except Exception as e:
            # Error response with modern styling
            error_msg = {
                'id': str(uuid.uuid4()),
                'type': 'bot',
                'content': f"🤔 I encountered an issue solving that problem: **{str(e)}**\n\nCould you try rephrasing it or asking a different question? I'm here to help!",
                'timestamp': datetime.now(),
                'has_solution': False
            }
            st.session_state.messages.append(error_msg)
        
        finally:
            # Stop solving indicator
            st.session_state.is_solving = False


def main():
    """Main function to run the modern chat app."""
    app = ModernMathChatApp()
    app.run()


if __name__ == "__main__":
    main()