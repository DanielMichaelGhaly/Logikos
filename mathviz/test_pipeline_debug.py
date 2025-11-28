"""
Test the pipeline with detailed error logging
"""
import logging

# Enable detailed logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')

from mathviz import MathVizPipeline

# Initialize the pipeline
pipeline = MathVizPipeline()

# Solve a simple equation
problem_text = "Solve for x: 2x + 5 = 13"
print(f"Processing problem: '{problem_text}'")
print("=" * 60)

result = pipeline.process(problem_text)

print("\n--- Results ---")
print(f"Problem Type: {result.problem.problem_type}")
print(f"Solution Steps: {len(result.solution_steps)} steps")
print(f"Final Answer: {result.final_answer}")
print(f"Reasoning: {result.reasoning[:200] if result.reasoning else 'None'}")

if result.metadata and result.metadata.get('error'):
    print(f"\n[ERROR DETECTED]")
    print(f"Error in metadata: {result.metadata}")
