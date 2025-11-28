#!/usr/bin/env python3
"""
MathViz Chat Interface Launcher

Quick launcher for the chat-style interface with Desmos integration.
"""

import sys
import os
from pathlib import Path

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Launch the MathViz chat interface."""
    try:
        import streamlit.web.cli as stcli
        
        # Set up the Streamlit app path
        chat_app_path = Path(__file__).parent / "src" / "mathviz" / "chat_app.py"
        
        print("🚀 Starting MathViz Chat Interface...")
        print("💬 Chat-style math solving with Desmos graphs")
        print("📱 Open your browser to: http://localhost:8501")
        print("🛑 Press Ctrl+C to stop")
        print("-" * 50)
        
        # Launch Streamlit
        sys.argv = ["streamlit", "run", str(chat_app_path)]
        stcli.main()
        
    except ImportError:
        print("❌ Streamlit not installed!")
        print("Install with: pip install streamlit")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n👋 Chat interface stopped. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting chat interface: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()