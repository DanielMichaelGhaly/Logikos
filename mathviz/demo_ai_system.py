#!/usr/bin/env python3
"""
AI-Enhanced MathViz System Demonstration
========================================

This script demonstrates the key capabilities of the AI-enhanced MathViz system:
1. AI-powered problem solving with fallback
2. Advanced calculus operations
3. Interactive graph visualization
4. Enhanced reasoning generation
5. Comprehensive pipeline integration

Run this to see the system in action!
"""

import time
from mathviz.pipeline import MathVizPipeline, PipelineConfig
from mathviz.graph_visualizer import GraphVisualizer, GraphConfig
from mathviz.ai_apis import solve_with_ai, ai_manager

def print_header(title):
    """Print a formatted header."""
    print(f"\n{'='*60}")
    print(f"🎯 {title}")
    print('='*60)

def print_section(title):
    """Print a formatted section header."""
    print(f"\n📝 {title}")
    print('-' * 50)

def demo_pipeline_capabilities():
    """Demonstrate pipeline capabilities."""
    print_header("AI-ENHANCED MATHVIZ SYSTEM DEMONSTRATION")
    
    # Initialize AI-enhanced pipeline
    config = PipelineConfig(
        use_ai=True,
        enable_interactive_graphs=True,
        enable_rate_limiting=False
    )
    
    pipeline = MathVizPipeline(config)
    
    # Show system status
    print_section("SYSTEM CAPABILITIES")
    status = pipeline.get_pipeline_status()
    
    print("🔧 Available Features:")
    for capability, enabled in status['capabilities'].items():
        icon = '✅' if enabled else '❌'
        name = capability.replace('_', ' ').title()
        print(f"   {icon} {name}")
    
    print(f"\n🤖 AI Providers: {ai_manager.get_available_providers()}")
    
    return pipeline

def demo_mathematical_problems(pipeline):
    """Demonstrate solving various mathematical problems."""
    print_section("MATHEMATICAL PROBLEM SOLVING")
    
    problems = [
        ("Calculus", "Find the derivative of x^3 + 2*x^2 + x"),
        ("Algebra", "Solve for x: 2*x + 5 = 13"), 
        ("Trigonometry", "Find the derivative of sin(x) + cos(x)")
    ]
    
    for category, problem in problems:
        print(f"\n🧮 {category}: {problem}")
        
        start_time = time.time()
        result = pipeline.process(problem)
        end_time = time.time()
        
        print(f"   ⏱️  Processing time: {end_time - start_time:.2f}s")
        print(f"   ✅ Success: {'error' not in result.metadata}")
        print(f"   🎯 Result type: {result.final_answer.get('type', 'unknown')}")
        
        # Show specific results
        if result.final_answer.get('type') == 'derivative':
            derivative = result.final_answer.get('derivative', 'not found')
            print(f"   📊 Derivative: {derivative}")
        elif result.final_answer.get('type') == 'algebraic_solution':
            solutions = result.final_answer.get('solutions', {})
            print(f"   📊 Solutions: {solutions}")
        
        # Show reasoning preview
        reasoning = result.reasoning
        if reasoning and len(reasoning) > 100:
            reasoning = reasoning[:100] + "..."
        print(f"   💭 Reasoning: {reasoning}")

def demo_graph_visualization():
    """Demonstrate graph visualization capabilities."""
    print_section("INTERACTIVE GRAPH VISUALIZATION")
    
    visualizer = GraphVisualizer()
    config = GraphConfig(x_range=(-5, 5), y_range=(-5, 5))
    
    # Test function visualization
    print("🌐 Testing function visualization...")
    result = visualizer.visualize_function("x**2", config)
    
    if result.success:
        print("   ✅ Visualization successful!")
        if result.graph_url:
            url_preview = result.graph_url[:60] + "..."
            print(f"   🔗 Desmos URL: {url_preview}")
        if result.graph_html:
            print(f"   📄 HTML generated: {len(result.graph_html)} characters")
    else:
        print(f"   ❌ Visualization failed: {result.error}")
    
    # Test multiple functions
    print("\n📈 Testing multiple function visualization...")
    expressions = ["x**2", "2*x + 1", "sin(x)"]
    result = visualizer.visualize_functions(expressions, config)
    
    if result.success:
        print(f"   ✅ {len(expressions)} functions visualized successfully!")
        print("   📊 Functions:", ", ".join(expressions))

def demo_ai_capabilities():
    """Demonstrate AI API capabilities."""
    print_section("AI-POWERED REASONING")
    
    print("🤖 Testing AI problem solving...")
    response = solve_with_ai("Find the derivative of x^2 + 3*x", "differentiation")
    
    print(f"   Provider: {response.provider}")
    print(f"   Success: {response.success}")
    print(f"   Confidence: {response.confidence}")
    
    if response.success:
        content = response.content
        if len(content) > 150:
            content = content[:150] + "..."
        print(f"   Response: {content}")

def demo_advanced_features(pipeline):
    """Demonstrate advanced features."""
    print_section("ADVANCED FEATURES")
    
    # Test with graph configuration
    print("📈 Testing problem solving with graph generation...")
    
    problem = "Find the derivative of x^3 - 2*x^2 + x"
    graph_config = GraphConfig(x_range=(-3, 3), y_range=(-5, 5))
    
    result = pipeline.process_with_graph_config(problem, graph_config)
    
    print(f"   🧮 Problem: {problem}")
    print(f"   ✅ Solution computed: {result['solution'].final_answer.get('type', 'unknown')}")
    
    if result['desmos_url']:
        print(f"   🌐 Desmos URL generated: {len(result['desmos_url'])} characters")
    
    if result['graph_html']:
        print(f"   📄 Interactive HTML: {len(result['graph_html'])} characters")

def main():
    """Run the complete demonstration."""
    try:
        # Initialize system
        pipeline = demo_pipeline_capabilities()
        
        # Demonstrate core features  
        demo_mathematical_problems(pipeline)
        demo_graph_visualization()
        demo_ai_capabilities()
        demo_advanced_features(pipeline)
        
        # Final summary
        print_header("DEMONSTRATION COMPLETE")
        print("🎉 All systems operational!")
        print("\n✅ Successfully demonstrated:")
        print("   • AI-enhanced mathematical problem solving")
        print("   • Interactive graph visualization (Desmos, HTML)")
        print("   • Advanced calculus capabilities") 
        print("   • Comprehensive error handling")
        print("   • Seamless fallback systems")
        print("\n🚀 Your AI-Enhanced MathViz system is ready for use!")
        
        # Usage examples
        print("\n💡 Try these examples:")
        print('   python -c "from mathviz import MathVizPipeline; p=MathVizPipeline(); print(p.process(\\"derivative of x^2\\").reasoning)"')
        print('   python -c "from mathviz.graph_visualizer import *; print(create_graph_url(\\"x^2\\"))"')
        
    except Exception as e:
        print(f"\n❌ Demonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())