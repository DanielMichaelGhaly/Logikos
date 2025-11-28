from mathviz import MathVizPipeline

# Initialize the pipeline
pipeline = MathVizPipeline()

# Solve a simple equation
problem_text = "Solve for x: 2x + 5 = 13"
print(f"Processing problem: '{problem_text}'")

try:
    result = pipeline.process(problem_text)
    
    # Access results
    print("\n--- Results ---")
    print("Problem Type:", result.problem.problem_type)
    print("Parsed Equations:", result.problem.equations)
    print("Solution Steps:", result.solution_steps)
    print("Final Answer:", result.final_answer)
    print("Reasoning:", result.reasoning)
    print("Visualization:", result.visualization)

except Exception as e:
    print(f"\nError: {e}")
