"""
Simplified MathViz Chat Interface - Works without external visualization dependencies

This provides a basic chat-style interface that focuses on problem solving
with minimal dependencies.
"""

import streamlit as st
import time
import uuid
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
import os
from pathlib import Path

# Add src directory to path for imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from mathviz.pipeline import MathVizPipeline
from mathviz.schemas import MathSolution

# Configure page
st.set_page_config(
    page_title="MathViz Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    /* Hide Streamlit default elements */
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Chat container */
    .chat-container {
        height: 70vh;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 15px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin-bottom: 1rem;
    }
    
    /* User message */
    .user-message {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 1rem;
    }
    
    .user-bubble {
        max-width: 70%;
        background: #007bff;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        font-size: 14px;
        line-height: 1.4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    /* Bot message */
    .bot-message {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 1rem;
        align-items: flex-start;
    }
    
    .bot-avatar {
        width: 35px;
        height: 35px;
        background: #28a745;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-size: 18px;
        flex-shrink: 0;
    }
    
    .bot-bubble {
        max-width: 75%;
        background: white;
        color: #333;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        font-size: 14px;
        line-height: 1.4;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    
    /* Solution display */
    .solution-answer {
        font-size: 16px;
        font-weight: bold;
        color: #28a745;
        margin: 10px 0;
        padding: 8px 12px;
        background: #e8f5e8;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    
    .solution-steps {
        margin: 10px 0;
        font-size: 13px;
    }
    
    .step-item {
        margin: 5px 0;
        padding: 6px 10px;
        background: #f8f9fa;
        border-radius: 6px;
        border-left: 3px solid #007bff;
    }
    
    /* Desmos link */
    .desmos-link {
        margin: 10px 0;
        padding: 8px 12px;
        background: #e3f2fd;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
    }
    
    .desmos-link a {
        color: #1976d2;
        text-decoration: none;
        font-weight: bold;
    }
    
    .desmos-link a:hover {
        text-decoration: underline;
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 8px 16px;
        background: #f0f0f0;
        border-radius: 18px;
        margin: 10px 0;
        font-size: 14px;
        color: #666;
    }
    
    .typing-dots {
        display: inline-block;
        width: 4px;
        height: 4px;
        border-radius: 50%;
        background: #666;
        margin: 0 1px;
        animation: typing 1.4s infinite ease-in-out;
    }
    
    .typing-dots:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing {
        0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
        40% { transform: scale(1.2); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

class SimpleMathChatApp:
    """Simplified chat-style interface for MathViz."""
    
    def __init__(self):
        """Initialize the chat application."""
        self.pipeline = MathVizPipeline()
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize session state for chat."""
        if 'messages' not in st.session_state:
            st.session_state.messages = [
                {
                    'id': str(uuid.uuid4()),
                    'type': 'bot',
                    'content': "👋 Hi! I'm MathViz, your AI math tutor. Ask me any math problem!",
                    'timestamp': datetime.now(),
                    'has_solution': False
                }
            ]
        
        if 'is_solving' not in st.session_state:
            st.session_state.is_solving = False
            
        if 'suggestions' not in st.session_state:
            st.session_state.suggestions = [
                "Solve for x: 2x + 5 = 13",
                "Find derivative of x² + 3x",
                "Integrate 2x + 1",
                "Factor x² - 5x + 6"
            ]
    
    def run(self):
        """Run the chat application."""
        # Header
        st.markdown("# 💬 MathViz Chat")
        st.markdown("*Your AI-powered math tutor*")
        
        # Chat container
        self.render_chat_messages()
        
        # Input area
        self.render_chat_input()
    
    def render_chat_messages(self):
        """Render all chat messages."""
        chat_container = st.container()
        
        with chat_container:
            # Create scrollable chat area
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
        """Render a user message."""
        st.markdown(f"""
        <div class="user-message">
            <div class="user-bubble">
                {message['content']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_bot_message(self, message: Dict):
        """Render a bot message."""
        avatar = "🧮" if message.get('has_solution') else "🤖"
        
        content_html = f'<div class="bot-bubble">{message["content"]}'
        
        # Add solution if present
        if message.get('solution'):
            solution = message['solution']
            content_html += self.format_solution(solution)
        
        # Add simple Desmos link if available
        if message.get('desmos_url'):
            content_html += f"""
            <div class="desmos-link">
                📊 <a href="{message['desmos_url']}" target="_blank">View Interactive Graph on Desmos</a>
            </div>
            """
        
        content_html += '</div>'
        
        st.markdown(f"""
        <div class="bot-message">
            <div class="bot-avatar">{avatar}</div>
            {content_html}
        </div>
        """, unsafe_allow_html=True)
    
    def format_solution(self, solution: MathSolution) -> str:
        """Format solution for display in chat."""
        html = ""
        
        # Final answer
        if solution.final_answer:
            answer_text = self.format_final_answer(solution.final_answer)
            html += f'<div class="solution-answer">📍 <strong>Answer:</strong> {answer_text}</div>'
        
        # Solution steps (condensed for chat)
        if solution.solution_steps and len(solution.solution_steps) > 0:
            html += '<div class="solution-steps"><strong>🔄 Solution Steps:</strong><br>'
            
            # Show first few steps
            steps_to_show = min(3, len(solution.solution_steps))
            for i in range(steps_to_show):
                step = solution.solution_steps[i]
                operation = step.get('operation', '').replace('_', ' ').title()
                justification = step.get('justification', '')[:50] + "..." if len(step.get('justification', '')) > 50 else step.get('justification', '')
                
                html += f'<div class="step-item">{i+1}. {operation}: {justification}</div>'
            
            if len(solution.solution_steps) > steps_to_show:
                remaining = len(solution.solution_steps) - steps_to_show
                html += f'<div style="font-size: 12px; color: #666; margin-top: 5px;">... and {remaining} more steps</div>'
            
            html += '</div>'
        
        return html
    
    def format_final_answer(self, answer) -> str:
        """Format the final answer for display."""
        if isinstance(answer, dict):
            if 'solutions' in answer:
                return f"x = {answer['solutions']}"
            elif 'derivative' in answer:
                return f"f'(x) = {answer['derivative']}"
            elif 'integral' in answer:
                return f"∫f(x)dx = {answer['integral']} + C"
            elif 'type' in answer:
                # Handle different solution types
                if answer['type'] == 'algebraic_solution':
                    solutions = answer.get('solutions', {})
                    if solutions:
                        return ', '.join([f"{k} = {v}" for k, v in solutions.items()])
                elif answer['type'] == 'derivative':
                    return f"f'(x) = {answer.get('derivative', 'N/A')}"
                elif answer['type'] == 'integral':
                    return f"∫f(x)dx = {answer.get('integral', 'N/A')} + C"
            
            # Fallback for dict
            return str(answer)
        else:
            return str(answer)
    
    def render_typing_indicator(self):
        """Show typing indicator while solving."""
        st.markdown("""
        <div class="bot-message">
            <div class="bot-avatar">🤖</div>
            <div class="typing-indicator">
                MathViz is solving
                <span class="typing-dots"></span>
                <span class="typing-dots"></span>
                <span class="typing-dots"></span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def render_chat_input(self):
        """Render the chat input area."""
        
        # Quick suggestions (show only if no messages or just welcome message)
        if len(st.session_state.messages) <= 1:
            st.markdown("**💡 Try these examples:**")
            cols = st.columns(len(st.session_state.suggestions))
            
            for i, suggestion in enumerate(st.session_state.suggestions):
                with cols[i]:
                    if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                        self.handle_user_message(suggestion)
                        st.rerun()
        
        # Chat input
        col1, col2 = st.columns([6, 1])
        
        with col1:
            user_input = st.text_input(
                "Type your math problem here...",
                placeholder="e.g., Solve for x: 2x + 5 = 13",
                key="chat_input",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.button("Send 📤", use_container_width=True, disabled=st.session_state.is_solving)
        
        # Handle input submission
        if (send_button or user_input) and user_input.strip() and not st.session_state.is_solving:
            self.handle_user_message(user_input.strip())
            st.rerun()
        
        # Instructions
        st.markdown("""
        <div style="font-size: 12px; color: #666; text-align: center; margin-top: 10px;">
            💬 Type any math problem • 📊 Get step-by-step solutions
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
        """Solve the math problem and create response."""
        try:
            # Solve the problem
            solution = self.pipeline.process(problem_text)
            
            # Create bot response
            response_content = f"Great! I solved your problem: **{problem_text}**"
            
            # Try to create a simple Desmos URL for plottable expressions
            desmos_url = self.create_simple_desmos_url(problem_text, solution)
            
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
            # Error response
            error_msg = {
                'id': str(uuid.uuid4()),
                'type': 'bot',
                'content': f"😅 I had trouble solving that problem: {str(e)}\n\nTry rephrasing it or ask for help!",
                'timestamp': datetime.now(),
                'has_solution': False
            }
            st.session_state.messages.append(error_msg)
        
        finally:
            # Stop solving indicator
            st.session_state.is_solving = False
    
    def create_simple_desmos_url(self, problem_text: str, solution: MathSolution) -> Optional[str]:
        """Create a simple Desmos URL for basic expressions."""
        try:
            import urllib.parse
            
            # Look for expressions that might be graphable
            expressions = []
            
            # Check if it's a derivative problem
            if solution.final_answer and isinstance(solution.final_answer, dict):
                if solution.final_answer.get('type') == 'derivative':
                    original = solution.final_answer.get('original_expression', '')
                    derivative = solution.final_answer.get('derivative', '')
                    
                    if original and derivative:
                        expressions = [original, derivative]
                elif solution.final_answer.get('type') == 'algebraic_solution':
                    # For algebraic solutions, try to extract the equation
                    problem_lower = problem_text.lower()
                    if 'x^2' in problem_lower or 'x²' in problem_lower:
                        # Try to extract quadratic
                        import re
                        match = re.search(r'([x^2²\+\-\d\s]+)\s*=\s*(\d+)', problem_text)
                        if match:
                            expressions = [f"{match.group(1)}-{match.group(2)}"]
            
            # Create basic Desmos URL if we have expressions
            if expressions and len(expressions) > 0:
                # Simple Desmos URL with basic expressions
                expr_str = urllib.parse.quote(expressions[0].replace('**', '^'))
                return f"https://www.desmos.com/calculator/graph?expressions=%5B%7B%22latex%22%3A%22y%3D{expr_str}%22%7D%5D"
            
            return None
            
        except Exception:
            return None


def main():
    """Main function to run the simple chat app."""
    app = SimpleMathChatApp()
    app.run()


if __name__ == "__main__":
    main()