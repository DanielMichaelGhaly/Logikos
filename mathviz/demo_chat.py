#!/usr/bin/env python3
"""
MathViz Chat Interface Demo

Quick demonstration and launch script for the chat-style interface.
"""

import sys
import os
from pathlib import Path

def show_demo_info():
    """Show information about the chat interface."""
    print("🎉 MathViz Chat Interface Ready!")
    print("=" * 50)
    
    print("\n💬 **Chat-Style Interface Features:**")
    print("✅ WhatsApp/ChatGPT-like chat bubbles")
    print("✅ Type math problems naturally")
    print("✅ Get step-by-step solutions")
    print("✅ Interactive Desmos graphs embedded")
    print("✅ Typing indicators while solving")
    print("✅ Quick example suggestions")
    
    print("\n📊 **Integrated Visualizations:**")
    print("🎯 Desmos Graphing Calculator (primary)")
    print("📐 GeoGebra integration")
    print("📈 Function Plot.js charts")
    print("🧮 LaTeX mathematical formatting")
    
    print("\n🧮 **Example Problems to Try:**")
    examples = [
        "Solve for x: 2x + 5 = 13",
        "Find derivative of x² + 3x + 1", 
        "Integrate 3x² + 2x",
        "Factor x² - 5x + 6",
        "Graph y = x² - 4x + 3",
        "Find roots of 2x² + 7x - 4",
        "Differentiate sin(x) + cos(x)",
        "Solve system: x + y = 5, 2x - y = 1"
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"   {i}. {example}")
    
    print("\n🚀 **How to Start:**")
    print("1. Run: python mathviz/run_chat.py")
    print("2. Open browser to: http://localhost:8501") 
    print("3. Type math problems in the chat!")
    print("4. Get solutions with Desmos graphs!")
    
    print("\n⚡ **Chat Interface vs Original:**")
    print("📱 Chat Interface:        Conversational, embedded graphs")
    print("📊 Original Streamlit:    Traditional form-based interface")
    print("💻 CLI:                  Command-line solving")
    
    print("\n" + "=" * 50)

def quick_test():
    """Run a quick test of the chat interface components."""
    print("🧪 Testing Chat Interface Components...")
    
    try:
        # Add path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Test imports
        from mathviz.pipeline import MathVizPipeline
        from mathviz.educational_viz import EducationalVisualizer
        print("✅ Core components imported")
        
        # Test basic solving
        pipeline = MathVizPipeline()
        solution = pipeline.process("2x + 4 = 10")
        print("✅ Problem solving works")
        
        # Test visualization
        visualizer = EducationalVisualizer()
        viz_result = visualizer.create_solution_visualization(solution)
        if viz_result.get('success'):
            print("✅ Educational visualization works")
            if viz_result.get('visualizations', {}).get('desmos'):
                print("✅ Desmos integration ready")
        else:
            print("⚠️  Visualization available but no plottable expressions")
        
        print("🎉 All systems ready for chat interface!")
        return True
        
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def launch_chat():
    """Launch the chat interface."""
    try:
        import streamlit.web.cli as stcli
        
        chat_app_path = Path(__file__).parent / "src" / "mathviz" / "chat_app.py"
        
        print("\n🚀 Launching MathViz Chat Interface...")
        print("💬 Open browser to: http://localhost:8501")
        print("🛑 Press Ctrl+C to stop")
        print("-" * 30)
        
        # Launch Streamlit
        sys.argv = ["streamlit", "run", str(chat_app_path)]
        stcli.main()
        
    except ImportError:
        print("❌ Streamlit not installed. Install with: pip install streamlit")
        return False
    except KeyboardInterrupt:
        print("\n👋 Chat stopped!")
        return True
    except Exception as e:
        print(f"❌ Launch error: {e}")
        return False

def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MathViz Chat Interface Demo")
    parser.add_argument("--info", action="store_true", help="Show demo information")
    parser.add_argument("--test", action="store_true", help="Test components")
    parser.add_argument("--launch", action="store_true", help="Launch chat interface")
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        # Default: show info and launch
        show_demo_info()
        
        response = input("\n🚀 Launch chat interface now? (y/N): ").lower()
        if response == 'y':
            if quick_test():
                launch_chat()
        else:
            print("💡 Run 'python demo_chat.py --launch' when ready!")
    
    elif args.info:
        show_demo_info()
    elif args.test:
        quick_test()
    elif args.launch:
        if quick_test():
            launch_chat()
        else:
            print("❌ Component test failed. Please check dependencies.")

if __name__ == "__main__":
    main()