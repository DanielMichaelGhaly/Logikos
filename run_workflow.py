#!/usr/bin/env python3
"""
Logikos Mathematical Problem Solving Workflow

Integrates AI explanation with SymPy verification for mathematical problem solving.
This is the main entry point for the restructured Logikos project.
"""

import sys
import argparse
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Import our custom modules
from ai.ai_solver import AISolver, AIResponse
from ai.response_parser import ResponseParser
from sympy_backend.expression_parser import EnhancedMathParser
from sympy_backend.solver import SymPySolver 
from sympy_backend.verifier import SolutionVerifier
from visualization.latex_formatter import LaTeXFormatter
from visualization.step_visualizer import StepVisualizer


@dataclass
class WorkflowResult:
    """Container for complete workflow result"""
    problem: str
    ai_response: AIResponse
    ai_parsed: Any  # ParsedResponse
    sympy_result: Any  # SolutionResult  
    verification: Any  # VerificationResult
    formatted_solution: Any  # FormattedSolution
    visual_solution: Any  # VisualSolution
    success: bool
    error_message: str = ""


class MathWorkflow:
    """Main mathematical problem solving workflow"""
    
    def __init__(self, enable_ai: bool = True, verbose: bool = False):
        self.enable_ai = enable_ai
        self.verbose = verbose
        
        # Initialize components
        self.math_parser = EnhancedMathParser()
        self.sympy_solver = SymPySolver()
        self.verifier = SolutionVerifier()
        self.latex_formatter = LaTeXFormatter()
        self.step_visualizer = StepVisualizer()
        
        # Initialize AI components if enabled
        if self.enable_ai:
            self.ai_solver = AISolver()
            self.response_parser = ResponseParser()
        else:
            self.ai_solver = None
            self.response_parser = None
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def solve_problem(self, problem: str) -> WorkflowResult:
        """Solve a mathematical problem using the complete workflow"""
        
        self.logger.info(f"Starting workflow for problem: {problem}")
        
        try:
            # Step 1: Parse the problem with SymPy
            self.logger.info("Step 1: Parsing problem with SymPy")
            parsed_problem = self.math_parser.parse(problem)
            
            if parsed_problem.problem_type == 'error':
                return WorkflowResult(
                    problem=problem,
                    ai_response=None,
                    ai_parsed=None,
                    sympy_result=None,
                    verification=None,
                    formatted_solution=None,
                    visual_solution=None,
                    success=False,
                    error_message=f"Failed to parse problem: {parsed_problem.metadata.get('error')}"
                )
            
            # Step 2: Solve with SymPy
            self.logger.info("Step 2: Solving with SymPy")
            sympy_result = self.sympy_solver.solve(parsed_problem)
            
            # Step 3: Get AI explanation (if enabled)
            ai_response = None
            ai_parsed = None
            if self.enable_ai and self.ai_solver:
                self.logger.info("Step 3: Getting AI explanation")
                
                # Check if AI service is available
                if self.ai_solver.check_availability():
                    ai_response = self.ai_solver.solve_problem(
                        problem, 
                        parsed_problem.problem_type
                    )
                    
                    if ai_response.success:
                        ai_parsed = self.response_parser.parse_response(
                            ai_response.content, 
                            parsed_problem.problem_type
                        )
                    else:
                        self.logger.warning(f"AI failed: {ai_response.error}")
                else:
                    self.logger.warning("AI service (Ollama) not available")
                    ai_response = AIResponse(
                        success=False,
                        content="",
                        error="Ollama service not available"
                    )
            
            # Step 4: Verify AI solution against SymPy (if AI was used)
            verification = None
            if ai_response and ai_response.success and sympy_result.success:
                self.logger.info("Step 4: Verifying AI solution against SymPy")
                verification = self.verifier.verify(ai_response.content, sympy_result)
            
            # Step 5: Format the solution
            self.logger.info("Step 5: Formatting solution")
            ai_explanation = ai_response.content if ai_response and ai_response.success else "AI explanation not available"
            verification_status = verification.status.value if verification else "no_verification"
            
            formatted_solution = self.latex_formatter.format_problem_solution(
                problem,
                ai_explanation,
                sympy_result,
                verification_status
            )
            
            # Step 6: Create visual representation
            self.logger.info("Step 6: Creating visual representation")
            ai_steps = ai_parsed.steps if ai_parsed else []
            sympy_steps = sympy_result.steps if sympy_result.success else []
            verification_info = {
                'status': verification_status,
                'explanation': verification.explanation if verification else 'No verification performed',
                'confidence': verification.confidence if verification else 0.0
            }
            
            visual_solution = self.step_visualizer.create_visual_solution(
                problem,
                ai_steps,
                sympy_steps,
                sympy_result.result if sympy_result.success else "No result",
                verification_info
            )
            
            self.logger.info("Workflow completed successfully")
            
            return WorkflowResult(
                problem=problem,
                ai_response=ai_response,
                ai_parsed=ai_parsed,
                sympy_result=sympy_result,
                verification=verification,
                formatted_solution=formatted_solution,
                visual_solution=visual_solution,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Workflow failed: {e}")
            return WorkflowResult(
                problem=problem,
                ai_response=None,
                ai_parsed=None,
                sympy_result=None,
                verification=None,
                formatted_solution=None,
                visual_solution=None,
                success=False,
                error_message=str(e)
            )
    
    def print_result(self, result: WorkflowResult) -> None:
        """Print a formatted result to console"""
        
        if not result.success:
            print(f"❌ Error: {result.error_message}")
            return
        
        print(f"\n🧮 Problem: {result.problem}")
        print("=" * 60)
        
        # SymPy Result
        if result.sympy_result and result.sympy_result.success:
            print(f"🔢 SymPy Solution: {result.sympy_result.result}")
            print(f"📝 SymPy Steps: {len(result.sympy_result.steps)}")
            if self.verbose:
                for i, step in enumerate(result.sympy_result.steps, 1):
                    print(f"   {i}. {step}")
        else:
            print("❌ SymPy failed to solve the problem")
        
        # AI Result  
        if result.ai_response and result.ai_response.success:
            print(f"\n🤖 AI Explanation Available: {len(result.ai_response.content)} chars")
            if result.ai_parsed:
                print(f"📊 AI Final Answer: {result.ai_parsed.final_answer}")
                print(f"🎯 AI Confidence: {result.ai_parsed.confidence:.2f}")
        else:
            if result.ai_response:
                print(f"❌ AI failed: {result.ai_response.error}")
            else:
                print("⚠️ AI not enabled or not available")
        
        # Verification
        if result.verification:
            status_emoji = {
                'match': '✅',
                'mismatch': '❌', 
                'partial_match': '⚠️',
                'inconclusive': '❓'
            }
            emoji = status_emoji.get(result.verification.status.value, '❓')
            print(f"\n{emoji} Verification: {result.verification.status.value}")
            print(f"📋 Explanation: {result.verification.explanation}")
            print(f"🎯 Confidence: {result.verification.confidence:.2f}")
        else:
            print("\n❓ No verification performed")
        
        # Summary
        if result.visual_solution:
            print(f"\n📊 Summary: {result.visual_solution.summary}")
        
        print("\n✨ Workflow completed!")


def main():
    """Main CLI interface"""
    
    parser = argparse.ArgumentParser(
        description="Logikos - AI-Enhanced Mathematical Problem Solver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_workflow.py "solve 2x+5=0"
  python run_workflow.py "derivative of x^2" --verbose
  python run_workflow.py "find roots of x^2-4" --no-ai
  python run_workflow.py "integrate sin(x)" --save-html output.html
        """
    )
    
    parser.add_argument('problem', help='Mathematical problem to solve')
    parser.add_argument('--no-ai', action='store_true', help='Disable AI explanation')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--save-html', help='Save HTML visualization to file')
    parser.add_argument('--save-json', help='Save JSON result to file')
    
    args = parser.parse_args()
    
    # Initialize workflow
    workflow = MathWorkflow(
        enable_ai=not args.no_ai,
        verbose=args.verbose
    )
    
    print("🚀 Starting Logikos Mathematical Problem Solver")
    print("=" * 60)
    
    # Solve the problem
    result = workflow.solve_problem(args.problem)
    
    # Print results
    workflow.print_result(result)
    
    # Save outputs if requested
    if result.success:
        if args.save_html and result.visual_solution:
            html_content = workflow.step_visualizer.create_html_visualization(result.visual_solution)
            with open(args.save_html, 'w') as f:
                f.write(html_content)
            print(f"\n💾 HTML saved to {args.save_html}")
        
        if args.save_json and result.visual_solution:
            json_content = workflow.step_visualizer.export_to_json(result.visual_solution)
            with open(args.save_json, 'w') as f:
                f.write(json_content)
            print(f"💾 JSON saved to {args.save_json}")


if __name__ == "__main__":
    main()