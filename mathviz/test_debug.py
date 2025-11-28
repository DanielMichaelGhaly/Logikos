import traceback
from mathviz import MathVizPipeline

# Initialize the pipeline
pipeline = MathVizPipeline()

# Solve a simple equation
problem_text = "Solve for x: 2x + 5 = 13"
print(f"Processing problem: '{problem_text}'")
print("=" * 60)

try:
    result = pipeline.process(problem_text)
    
    # Access results
    print("\n--- Results ---")
    print(f"Problem Type: {result.problem.problem_type}")
    print(f"Parsed Equations: {result.problem.equations}")
    print(f"Variables: {result.problem.variables}")
    print(f"Goal: {result.problem.goal}")
    print(f"\nSolution Steps ({len(result.solution_steps)} steps):")
    for i, step in enumerate(result.solution_steps, 1):
        print(f"  Step {i}: {step}")
    print(f"\nFinal Answer: {result.final_answer}")
    print(f"\nReasoning: {result.reasoning}")
    print(f"\nVisualization: {result.visualization}")
    
    if result.metadata:
        print(f"\nMetadata: {result.metadata}")

except Exception as e:
    print(f"\n!!! ERROR !!!")
    print(f"Error type: {type(e).__name__}")
    print(f"Error message: {e}")
    print(f"\nFull traceback:")
    traceback.print_exc()
