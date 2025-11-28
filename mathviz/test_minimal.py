"""
Minimal test to isolate the solver issue
"""
import sys
import traceback

print("=" * 60)
print("TEST 1: Direct Solver Test")
print("=" * 60)

try:
    from mathviz.solver import MathSolver
    from mathviz.schemas import MathProblem, Variable, Equation
    
    # Create a simple problem manually
    problem = MathProblem(
        problem_text="Solve for x: 2x + 5 = 13",
        problem_type="algebraic",
        variables=[Variable(name="x", domain="real")],
        equations=[Equation(left_side="2*x + 5", right_side="13")],
        constraints=[],
        goal="solve for x"
    )
    
    print(f"Problem created: {problem.problem_text}")
    print(f"Variables: {[v.name for v in problem.variables]}")
    print(f"Equations: {[(eq.left_side, eq.right_side) for eq in problem.equations]}")
    
    # Create solver without AI
    solver = MathSolver(use_ai=False, ai_first=False)
    print("\nSolver created (AI disabled)")
    
    # Solve
    print("\nCalling solver.solve()...")
    solution = solver.solve(problem)
    
    print(f"\n[OK] Solver completed!")
    print(f"Solution steps count: {len(solution.solution_steps)}")
    print(f"Final answer: {solution.final_answer}")
    print(f"Reasoning: {solution.reasoning[:100]}..." if solution.reasoning else "No reasoning")
    
    if solution.solution_steps:
        print("\nSteps:")
        for i, step in enumerate(solution.solution_steps, 1):
            print(f"  {i}. {step}")
    else:
        print("\n[WARNING] No solution steps generated!")
        
    if hasattr(solution, 'trace') and solution.trace:
        print(f"\nTrace object exists: {type(solution.trace)}")
        print(f"Trace steps count: {solution.trace.get_step_count()}")
        if solution.trace.steps:
            print("Trace steps:")
            for step in solution.trace.steps:
                print(f"  - {step.operation}: {step.reasoning}")
    
except Exception as e:
    print(f"\n[ERROR] in Test 1!")
    print(f"Error: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 2: Parser Test")
print("=" * 60)

try:
    from mathviz.parser import MathParser
    
    parser = MathParser()
    problem_text = "Solve for x: 2x + 5 = 13"
    
    print(f"Parsing: '{problem_text}'")
    parsed = parser.parse(problem_text)
    
    print(f"\n[OK] Parser completed!")
    print(f"Problem type: {parsed.problem_type}")
    print(f"Variables: {[v.name for v in parsed.variables]}")
    print(f"Equations: {[(eq.left_side, eq.right_side) for eq in parsed.equations]}")
    print(f"Goal: {parsed.goal}")
    
except Exception as e:
    print(f"\n[ERROR] in Test 2!")
    print(f"Error: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 3: Validator Test")
print("=" * 60)

try:
    from mathviz.validator import MathValidator
    from mathviz.schemas import MathProblem, Variable, Equation
    
    validator = MathValidator()
    
    problem = MathProblem(
        problem_text="Solve for x: 2x + 5 = 13",
        problem_type="algebraic",
        variables=[Variable(name="x")],
        equations=[Equation(left_side="2*x + 5", right_side="13")],
        constraints=[],
        goal="solve for x"
    )
    
    print(f"Validating problem...")
    is_valid = validator.validate(problem)
    
    print(f"\n[OK] Validation result: {is_valid}")
    
    if not is_valid:
        errors = validator.get_validation_errors()
        print(f"Validation errors: {errors}")
    
except Exception as e:
    print(f"\n[ERROR] in Test 3!")
    print(f"Error: {e}")
    traceback.print_exc()
