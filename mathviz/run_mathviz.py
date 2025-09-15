#!/usr/bin/env python3
"""
MathViz CLI Runner - Easy access to all MathViz functionality.
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def run_streamlit():
    """Run the Streamlit web application."""
    try:
        import streamlit.web.cli as stcli
        import sys
        
        # Set up Streamlit arguments
        streamlit_script = Path(__file__).parent / "src" / "mathviz" / "streamlit_app.py"
        sys.argv = ["streamlit", "run", str(streamlit_script)]
        
        print("🚀 Starting MathViz Streamlit App...")
        print("📱 Open your browser to: http://localhost:8501")
        print("🛑 Press Ctrl+C to stop the server")
        
        stcli.main()
    except ImportError:
        print("❌ Streamlit not installed. Install with: pip install streamlit")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        sys.exit(1)


def run_api():
    """Run the FastAPI backend server."""
    try:
        from mathviz.api import run_server
        
        print("🚀 Starting MathViz API Server...")
        print("📡 API available at: http://localhost:8000")
        print("📚 Documentation at: http://localhost:8000/docs")
        print("🛑 Press Ctrl+C to stop the server")
        
        run_server(host="0.0.0.0", port=8000, reload=True)
    except ImportError as e:
        print(f"❌ Required packages not installed: {e}")
        print("Install with: pip install fastapi uvicorn")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting API server: {e}")
        sys.exit(1)


def run_examples(mode="full"):
    """Run the example demonstrations."""
    try:
        import examples
        
        if mode == "interactive":
            demo = examples.MathVizDemo()
            demo.run_interactive_demo()
        elif mode == "quick":
            sys.argv = ["examples.py", "quick"]
            examples.main()
        else:
            demo = examples.MathVizDemo()
            demo.run_full_demo()
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        sys.exit(1)


def run_tests():
    """Run the test suite."""
    try:
        import subprocess
        import sys
        
        test_dir = Path(__file__).parent / "tests"
        
        if not test_dir.exists():
            print("❌ Tests directory not found")
            sys.exit(1)
        
        print("🧪 Running MathViz Test Suite...")
        
        # Try to run pytest
        try:
            result = subprocess.run([
                sys.executable, "-m", "pytest", 
                str(test_dir), 
                "-v", "--tb=short"
            ], capture_output=False, text=True)
            
            if result.returncode == 0:
                print("✅ All tests passed!")
            else:
                print("❌ Some tests failed")
                sys.exit(result.returncode)
                
        except FileNotFoundError:
            print("❌ pytest not installed. Install with: pip install pytest")
            print("🔧 Running basic test instead...")
            
            # Fallback: run basic pipeline test
            sys.path.insert(0, str(Path(__file__).parent / "tests"))
            from test_pipeline import TestMathVizPipeline
            
            # Create and run basic test
            test_instance = TestMathVizPipeline()
            pipeline = test_instance.pipeline()
            
            try:
                test_instance.test_pipeline_initialization(pipeline)
                test_instance.test_simple_algebra_solve(pipeline)
                print("✅ Basic tests passed!")
            except Exception as e:
                print(f"❌ Basic test failed: {e}")
                sys.exit(1)
                
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        sys.exit(1)


def solve_problem(problem_text):
    """Solve a single problem via CLI."""
    try:
        from mathviz.pipeline import MathVizPipeline
        
        print(f"🧮 Solving: {problem_text}")
        print("-" * 50)
        
        pipeline = MathVizPipeline()
        solution = pipeline.process(problem_text)
        
        # Display result
        print("🎯 Final Answer:")
        if isinstance(solution.final_answer, dict):
            for key, value in solution.final_answer.items():
                print(f"   {key}: {value}")
        else:
            print(f"   {solution.final_answer}")
        
        print(f"\n📖 Solution completed in {len(solution.solution_steps)} steps")
        
        # Optionally show reasoning
        if solution.reasoning:
            print("\n🧠 Reasoning:")
            # Show first 300 characters of reasoning
            reasoning_preview = solution.reasoning[:300]
            if len(solution.reasoning) > 300:
                reasoning_preview += "..."
            print(f"   {reasoning_preview}")
        
        print("\n✨ Done!")
        
    except Exception as e:
        print(f"❌ Error solving problem: {e}")
        sys.exit(1)


def install_dependencies():
    """Install MathViz dependencies."""
    try:
        import subprocess
        import sys
        
        print("📦 Installing MathViz dependencies...")
        
        # Install the package in development mode
        mathviz_dir = Path(__file__).parent / "mathviz"
        
        if mathviz_dir.exists():
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-e", str(mathviz_dir)
            ], capture_output=False)
            
            if result.returncode == 0:
                print("✅ MathViz installed successfully!")
                print("🎉 You can now use: python run_mathviz.py --help")
            else:
                print("❌ Installation failed")
                sys.exit(1)
        else:
            print("❌ MathViz directory not found")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)


def show_status():
    """Show MathViz system status."""
    print("🧮 MathViz System Status")
    print("=" * 30)
    
    # Check Python version
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    # Check dependencies
    dependencies = [
        ("sympy", "Symbolic mathematics"),
        ("numpy", "Numerical computing"),
        ("pydantic", "Data validation"),
        ("streamlit", "Web interface"),
        ("fastapi", "API backend"), 
        ("plotly", "Interactive visualization"),
        ("pint", "Unit handling"),
    ]
    
    print("\n📦 Dependencies:")
    for package, description in dependencies:
        try:
            __import__(package)
            print(f"   ✅ {package} - {description}")
        except ImportError:
            print(f"   ❌ {package} - {description} (not installed)")
    
    # Check MathViz components
    print("\n🔧 MathViz Components:")
    try:
        from mathviz.pipeline import MathVizPipeline
        pipeline = MathVizPipeline()
        print("   ✅ Pipeline initialization")
        
        # Test basic functionality
        test_solution = pipeline.process("2 + 2")
        print("   ✅ Basic problem solving")
        
    except Exception as e:
        print(f"   ❌ Component error: {e}")
    
    print("\n🚀 Available Commands:")
    commands = [
        ("--streamlit", "Launch web interface"),
        ("--api", "Start API server"),
        ("--examples", "Run demonstrations"),
        ("--tests", "Run test suite"),
        ("--solve 'problem'", "Solve a problem"),
        ("--install", "Install dependencies"),
    ]
    
    for cmd, desc in commands:
        print(f"   {cmd:<20} {desc}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MathViz - AI-powered mathematical problem solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_mathviz.py --streamlit              # Launch web interface
  python run_mathviz.py --api                    # Start API server
  python run_mathviz.py --examples               # Run full demo
  python run_mathviz.py --examples interactive   # Interactive demo
  python run_mathviz.py --solve "2x + 3 = 7"     # Solve specific problem
  python run_mathviz.py --tests                  # Run test suite
  python run_mathviz.py --status                 # Show system status
        """
    )
    
    parser.add_argument("--streamlit", action="store_true",
                       help="Launch Streamlit web interface")
    parser.add_argument("--api", action="store_true",
                       help="Start FastAPI backend server")
    parser.add_argument("--examples", nargs="?", const="full", 
                       choices=["full", "quick", "interactive"],
                       help="Run example demonstrations")
    parser.add_argument("--tests", action="store_true",
                       help="Run the test suite")
    parser.add_argument("--solve", type=str, metavar="PROBLEM",
                       help="Solve a specific mathematical problem")
    parser.add_argument("--install", action="store_true",
                       help="Install MathViz dependencies")
    parser.add_argument("--status", action="store_true",
                       help="Show system status and available commands")
    
    args = parser.parse_args()
    
    # If no arguments provided, show status
    if not any(vars(args).values()):
        show_status()
        return
    
    # Execute the requested action
    if args.streamlit:
        run_streamlit()
    elif args.api:
        run_api()
    elif args.examples:
        run_examples(args.examples)
    elif args.tests:
        run_tests()
    elif args.solve:
        solve_problem(args.solve)
    elif args.install:
        install_dependencies()
    elif args.status:
        show_status()


if __name__ == "__main__":
    main()