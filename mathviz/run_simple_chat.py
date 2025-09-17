#!/usr/bin/env python3
"""
Simple MathViz Chat Interface Launcher

Launches the simplified chat interface that works without complex dependencies.
"""

import sys
import os
from pathlib import Path

def main():
    """Launch the simple MathViz chat interface."""
    try:
        import streamlit.web.cli as stcli
        
        # Set up the Streamlit app path
        chat_app_path = Path(__file__).parent / "src" / "mathviz" / "simple_chat_app.py"
        
        print("🚀 Starting MathViz Simple Chat Interface...")
        print("💬 Chat-style math solving with basic Desmos links")
        print("📱 Open your browser to: http://localhost:8501")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 50)
        print("✅ This version works with minimal dependencies")
        print("✅ Chat-style interface with solutions")
        print("✅ Basic Desmos graph links")
        print("✅ Step-by-step explanations")
        print("=" * 50)
        
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