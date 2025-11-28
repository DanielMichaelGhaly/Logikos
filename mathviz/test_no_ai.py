"""
Test pipeline with AI completely disabled
"""
from mathviz import MathVizPipeline
from mathviz.pipeline import PipelineConfig

# Create config with AI disabled
config = PipelineConfig(
    use_ai=False,
    ai_first=False,
    use_ai_parser=False,
    enable_interactive_graphs=False,
    enable_rate_limiting=False
)

# Initialize the pipeline
pipeline = MathVizPipeline(config=config)

# Solve a simple equation
problem_text = "Solve for x: 2x + 5 = 13"
print(f"Processing problem: '{problem_text}'")
print("=" * 60)

try:
    result = pipeline.process(problem_text)
    
    print("\n--- Results ---")
    print(f"Problem Type: {result.problem.problem_type}")
    print(f"Variables: {[v.name for v in result.problem.variables]}")
    print(f"Equations: {[(e.left_side, e.right_side) for e in result.problem.equations]}")
    print(f"Solution Steps: {len(result.solution_steps)} steps")
    
    if result.solution_steps:
        print("\nSteps:")
        for i, step in enumerate(result.solution_steps[:5], 1):  # Show first 5
            print(f"  {i}. {step}")
    
    print(f"\nFinal Answer: {result.final_answer}")
    print(f"\nReasoning length: {len(result.reasoning)} chars")
    
    if result.metadata and result.metadata.get('error'):
        print(f"\n[ERROR IN METADATA]")
        print(f"Metadata: {result.metadata}")
        
except Exception as e:
    print(f"\n[EXCEPTION CAUGHT]")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
