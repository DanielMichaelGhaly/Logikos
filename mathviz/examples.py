#!/usr/bin/env python3
"""
Comprehensive examples and demonstrations for MathViz.
This script showcases the full capabilities of the MathViz framework.
"""

import sys
import time
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mathviz.pipeline import MathVizPipeline
from mathviz.validator import ValidationError


class MathVizDemo:
    """Demonstration class for MathViz capabilities."""

    def __init__(self):
        """Initialize the demo with a MathViz pipeline."""
        self.pipeline = MathVizPipeline()
        self.examples = self.load_example_problems()

    def load_example_problems(self):
        """Load comprehensive example problems organized by category."""
        return {
            "Basic Algebra": [
                {
                    "problem": "Solve for x: 2x + 5 = 13",
                    "expected_answer": "x = 4",
                    "difficulty": "Easy",
                    "description": "Simple linear equation"
                },
                {
                    "problem": "Solve for x: x + 7 = 12",
                    "expected_answer": "x = 5",
                    "difficulty": "Easy",
                    "description": "Basic linear equation"
                },
                {
                    "problem": "Solve for y: 3y - 4 = 8",
                    "expected_answer": "y = 4",
                    "difficulty": "Easy",
                    "description": "Linear equation with coefficient"
                }
            ],
            "Quadratic Equations": [
                {
                    "problem": "Find the roots of x^2 - 5x + 6",
                    "expected_answer": "x = 2, x = 3",
                    "difficulty": "Medium",
                    "description": "Factorable quadratic equation"
                },
                {
                    "problem": "Solve x^2 - 4x + 4 = 0",
                    "expected_answer": "x = 2 (double root)",
                    "difficulty": "Medium",
                    "description": "Perfect square quadratic"
                }
            ],
            "Basic Calculus - Derivatives": [
                {
                    "problem": "Find the derivative of x^2 + 3x",
                    "expected_answer": "2x + 3",
                    "difficulty": "Easy",
                    "description": "Polynomial differentiation"
                },
                {
                    "problem": "Differentiate x^3 + 2x^2 - 5x + 1",
                    "expected_answer": "3x^2 + 4x - 5",
                    "difficulty": "Easy",
                    "description": "Higher degree polynomial"
                },
                {
                    "problem": "Find the derivative of sin(x) + cos(x)",
                    "expected_answer": "cos(x) - sin(x)",
                    "difficulty": "Medium",
                    "description": "Trigonometric functions"
                }
            ],
            "Basic Calculus - Integrals": [
                {
                    "problem": "Integrate 2x + 1",
                    "expected_answer": "x^2 + x + C",
                    "difficulty": "Easy",
                    "description": "Polynomial integration"
                },
                {
                    "problem": "Find the integral of x^3",
                    "expected_answer": "x^4/4 + C",
                    "difficulty": "Easy",
                    "description": "Power rule integration"
                }
            ],
            "Advanced Problems": [
                {
                    "problem": "Find the derivative of x^2 * sin(x)",
                    "expected_answer": "2x*sin(x) + x^2*cos(x)",
                    "difficulty": "Hard",
                    "description": "Product rule application"
                },
                {
                    "problem": "Differentiate e^(x^2)",
                    "expected_answer": "2x * e^(x^2)",
                    "difficulty": "Hard",
                    "description": "Chain rule with exponential"
                }
            ]
        }

    def run_full_demo(self):
        """Run the complete demonstration."""
        print("🧮 MathViz Framework Demonstration")
        print("=" * 50)
        print()

        # Demo overview
        total_problems = sum(len(problems) for problems in self.examples.values())
        print(f"Running demonstrations with {total_problems} example problems")
        print(f"Categories: {', '.join(self.examples.keys())}")
        print()

        # Run demos by category
        overall_results = []
        
        for category, problems in self.examples.items():
            print(f"📚 Category: {category}")
            print("-" * 40)
            
            category_results = self.run_category_demo(problems)
            overall_results.extend(category_results)
            
            self.print_category_summary(category_results)
            print()

        # Print overall summary
        self.print_overall_summary(overall_results)
        
        # Run interactive demo if requested
        self.offer_interactive_demo()

    def run_category_demo(self, problems):
        """Run demonstration for a specific category."""
        results = []
        
        for i, problem_data in enumerate(problems, 1):
            print(f"  Problem {i}: {problem_data['description']}")
            print(f"  📝 {problem_data['problem']}")
            print(f"  🎯 Expected: {problem_data['expected_answer']}")
            print(f"  📊 Difficulty: {problem_data['difficulty']}")
            
            # Solve the problem
            start_time = time.time()
            try:
                solution = self.pipeline.process(problem_data['problem'])
                solve_time = time.time() - start_time
                
                result = {
                    'problem': problem_data['problem'],
                    'expected': problem_data['expected_answer'],
                    'solution': solution,
                    'success': True,
                    'time': solve_time,
                    'error': None
                }
                
                print(f"  ✅ Solved in {solve_time:.3f}s")
                self.display_solution_summary(solution)
                
            except Exception as e:
                solve_time = time.time() - start_time
                result = {
                    'problem': problem_data['problem'],
                    'expected': problem_data['expected_answer'],
                    'solution': None,
                    'success': False,
                    'time': solve_time,
                    'error': str(e)
                }
                
                print(f"  ❌ Failed: {str(e)}")
            
            results.append(result)
            print()
        
        return results

    def display_solution_summary(self, solution):
        """Display a concise summary of the solution."""
        if solution.final_answer:
            if isinstance(solution.final_answer, dict):
                if 'solutions' in solution.final_answer:
                    print(f"  🎯 Result: {solution.final_answer['solutions']}")
                elif 'derivative' in solution.final_answer:
                    print(f"  🎯 Derivative: {solution.final_answer['derivative']}")
                elif 'integral' in solution.final_answer:
                    print(f"  🎯 Integral: {solution.final_answer['integral']} + C")
                else:
                    print(f"  🎯 Result: {solution.final_answer}")
            else:
                print(f"  🎯 Result: {solution.final_answer}")
        
        print(f"  📖 Steps: {len(solution.solution_steps)}")

    def print_category_summary(self, results):
        """Print summary statistics for a category."""
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total - successful
        avg_time = sum(r['time'] for r in results) / total if results else 0
        
        print(f"  📊 Category Summary:")
        print(f"    Total problems: {total}")
        print(f"    Successful: {successful} ({100*successful/total:.1f}%)")
        if failed > 0:
            print(f"    Failed: {failed} ({100*failed/total:.1f}%)")
        print(f"    Average time: {avg_time:.3f}s")

    def print_overall_summary(self, results):
        """Print overall demonstration summary."""
        print("🎯 Overall Demonstration Summary")
        print("=" * 40)
        
        total = len(results)
        successful = sum(1 for r in results if r['success'])
        failed = total - successful
        total_time = sum(r['time'] for r in results)
        avg_time = total_time / total if results else 0
        
        print(f"Total problems processed: {total}")
        print(f"Successful solutions: {successful} ({100*successful/total:.1f}%)")
        if failed > 0:
            print(f"Failed solutions: {failed} ({100*failed/total:.1f}%)")
            print("\nFailed problems:")
            for r in results:
                if not r['success']:
                    print(f"  ❌ {r['problem']}: {r['error']}")
        
        print(f"Total execution time: {total_time:.3f}s")
        print(f"Average time per problem: {avg_time:.3f}s")
        
        # Performance categorization
        if avg_time < 0.1:
            perf_rating = "🚀 Excellent"
        elif avg_time < 0.5:
            perf_rating = "⚡ Very Good"
        elif avg_time < 1.0:
            perf_rating = "✅ Good"
        elif avg_time < 2.0:
            perf_rating = "⏳ Fair"
        else:
            perf_rating = "🐌 Needs Optimization"
        
        print(f"Performance rating: {perf_rating}")
        print()

    def offer_interactive_demo(self):
        """Offer an interactive demonstration session."""
        print("🎮 Interactive Demo Available!")
        print("Would you like to try solving your own problems?")
        
        try:
            response = input("Enter 'y' for interactive demo, or any other key to exit: ").lower()
            if response == 'y':
                self.run_interactive_demo()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")

    def run_interactive_demo(self):
        """Run an interactive problem-solving session."""
        print("\n🎮 Interactive MathViz Demo")
        print("=" * 30)
        print("Enter mathematical problems to solve. Type 'quit' to exit.")
        print("Examples:")
        print("  - Solve for x: 3x - 7 = 14")
        print("  - Find the derivative of x^3 + 2x")
        print("  - Integrate sin(x)")
        print()

        while True:
            try:
                problem = input("🧮 Enter problem: ").strip()
                
                if problem.lower() in ['quit', 'exit', 'q']:
                    print("👋 Thanks for trying MathViz!")
                    break
                
                if not problem:
                    continue
                
                print(f"\n🔍 Solving: {problem}")
                print("-" * 40)
                
                # Validate first
                try:
                    parsed = self.pipeline.parser.parse(problem)
                    is_valid = self.pipeline.validator.validate(parsed)
                    
                    print(f"✅ Problem parsed as: {parsed.problem_type}")
                    print(f"✅ Goal: {parsed.goal}")
                except ValidationError as e:
                    print(f"⚠️  Validation warning: {str(e)}")
                except Exception as e:
                    print(f"❌ Parsing error: {str(e)}")
                    continue
                
                # Solve
                try:
                    start_time = time.time()
                    solution = self.pipeline.process(problem)
                    solve_time = time.time() - start_time
                    
                    print(f"✅ Solved in {solve_time:.3f}s")
                    print()
                    
                    # Display detailed results
                    self.display_detailed_solution(solution)
                    
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
                
                print("\n" + "="*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break

    def display_detailed_solution(self, solution):
        """Display a detailed solution breakdown."""
        # Final answer
        print("🎯 Final Answer:")
        if solution.final_answer:
            if isinstance(solution.final_answer, dict):
                for key, value in solution.final_answer.items():
                    print(f"   {key}: {value}")
            else:
                print(f"   {solution.final_answer}")
        
        # Step-by-step
        if solution.solution_steps:
            print(f"\n📝 Solution Steps ({len(solution.solution_steps)}):")
            for i, step in enumerate(solution.solution_steps, 1):
                print(f"   Step {i}: {step.get('operation', 'Unknown').replace('_', ' ').title()}")
                print(f"     From: {step.get('expression_before', 'N/A')}")
                print(f"     To: {step.get('expression_after', 'N/A')}")
                if step.get('justification'):
                    print(f"     Why: {step['justification']}")
                print()
        
        # Reasoning (abbreviated)
        if solution.reasoning:
            print("🧠 Reasoning (first 200 chars):")
            reasoning_preview = solution.reasoning[:200]
            if len(solution.reasoning) > 200:
                reasoning_preview += "..."
            print(f"   {reasoning_preview}")
        
        # Metadata
        if solution.metadata:
            print(f"\n📊 Metadata:")
            for key, value in solution.metadata.items():
                print(f"   {key}: {value}")

    def run_specific_demo(self, category=None, problem_index=None):
        """Run a specific demo by category and problem index."""
        if category and category in self.examples:
            problems = self.examples[category]
            if problem_index is not None and 0 <= problem_index < len(problems):
                problem_data = problems[problem_index]
                print(f"🔍 Demo: {category} - Problem {problem_index + 1}")
                print(f"Problem: {problem_data['problem']}")
                
                try:
                    solution = self.pipeline.process(problem_data['problem'])
                    self.display_detailed_solution(solution)
                except Exception as e:
                    print(f"Error: {str(e)}")
            else:
                print(f"Available problems in {category}: {len(problems)}")
                for i, p in enumerate(problems):
                    print(f"  {i}: {p['problem']}")
        else:
            print("Available categories:")
            for cat in self.examples.keys():
                print(f"  - {cat}")


def main():
    """Main function to run the demonstration."""
    demo = MathVizDemo()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "interactive":
            demo.run_interactive_demo()
        elif sys.argv[1] == "quick":
            # Quick demo with just a few problems
            print("🚀 Quick MathViz Demo")
            print("=" * 25)
            quick_problems = [
                "Solve for x: 2x + 3 = 11",
                "Find the derivative of x^2 + 5x",
                "Integrate 3x^2"
            ]
            
            for problem in quick_problems:
                print(f"\n🧮 Problem: {problem}")
                try:
                    solution = demo.pipeline.process(problem)
                    demo.display_solution_summary(solution)
                except Exception as e:
                    print(f"❌ Error: {str(e)}")
            print("\n✨ Quick demo complete!")
        else:
            print("Usage: python examples.py [interactive|quick]")
            return
    else:
        # Run full comprehensive demo
        demo.run_full_demo()


if __name__ == "__main__":
    main()