#!/usr/bin/env python3
"""
Step Visualizer

Creates visual representations of mathematical problem-solving steps.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json


@dataclass
class VisualStep:
    """Container for a visual step"""
    step_number: int
    title: str
    description: str
    input_expression: str
    output_expression: str
    rule_applied: str = ""
    latex_representation: str = ""
    metadata: Dict[str, Any] = None


@dataclass
class VisualSolution:
    """Container for complete visual solution"""
    problem: str
    steps: List[VisualStep]
    final_result: str
    verification_info: Dict[str, Any]
    summary: str = ""


class StepVisualizer:
    """Creates visual representations of solution steps"""
    
    def __init__(self):
        self.step_templates = {
            'algebraic': {
                'isolation': "Isolate the variable term",
                'substitution': "Substitute values",
                'simplification': "Simplify the expression", 
                'solving': "Solve for the variable",
                'verification': "Verify the solution"
            },
            'calculus': {
                'identification': "Identify the function and operation",
                'rule_application': "Apply calculus rules",
                'differentiation': "Differentiate the function",
                'integration': "Integrate the function",
                'simplification': "Simplify the result"
            }
        }
    
    def create_visual_solution(self, 
                             problem: str,
                             ai_steps: List[str],
                             sympy_steps: List[str],
                             final_result: Any,
                             verification: Dict[str, Any]) -> VisualSolution:
        """Create a complete visual solution"""
        
        # Combine AI and SymPy steps
        visual_steps = self._merge_steps(ai_steps, sympy_steps)
        
        # Generate summary
        summary = self._generate_summary(problem, len(visual_steps), verification)
        
        return VisualSolution(
            problem=problem,
            steps=visual_steps,
            final_result=str(final_result),
            verification_info=verification,
            summary=summary
        )
    
    def _merge_steps(self, ai_steps: List[str], sympy_steps: List[str]) -> List[VisualStep]:
        """Merge AI explanation steps with SymPy computational steps"""
        visual_steps = []
        
        # Use AI steps as the main narrative
        for i, ai_step in enumerate(ai_steps, 1):
            # Try to find corresponding SymPy step
            sympy_step = sympy_steps[i-1] if i-1 < len(sympy_steps) else ""
            
            # Extract key information from the AI step
            title, description, expressions = self._parse_ai_step(ai_step)
            
            visual_step = VisualStep(
                step_number=i,
                title=title or f"Step {i}",
                description=description,
                input_expression=expressions.get('input', ''),
                output_expression=expressions.get('output', ''),
                rule_applied=self._extract_rule_from_sympy_step(sympy_step),
                latex_representation=self._create_step_latex(expressions),
                metadata={
                    'ai_step': ai_step,
                    'sympy_step': sympy_step,
                    'step_type': self._classify_step(ai_step)
                }
            )
            
            visual_steps.append(visual_step)
        
        return visual_steps
    
    def _parse_ai_step(self, step: str) -> tuple:
        """Parse an AI step to extract title, description, and expressions"""
        
        # Try to extract title (usually at the beginning)
        title = ""
        if step.startswith('Step'):
            # Extract title from "Step X: Title"
            parts = step.split(':', 1)
            if len(parts) == 2:
                title = parts[0].strip()
                step = parts[1].strip()
        
        # Extract mathematical expressions
        expressions = self._extract_expressions_from_text(step)
        
        return title, step, expressions
    
    def _extract_expressions_from_text(self, text: str) -> Dict[str, str]:
        """Extract input and output mathematical expressions from text"""
        import re
        
        expressions = {'input': '', 'output': ''}
        
        # Look for equations or expressions
        eq_patterns = [
            r'(\S+\s*=\s*\S+)',  # Basic equations
            r'([a-zA-Z0-9+\-*/^().\s]+\s*=\s*[a-zA-Z0-9+\-*/^().\s]+)',  # Complex equations
        ]
        
        equations = []
        for pattern in eq_patterns:
            equations.extend(re.findall(pattern, text))
        
        if equations:
            expressions['input'] = equations[0] if equations else ''
            expressions['output'] = equations[-1] if len(equations) > 1 else ''
        
        return expressions
    
    def _extract_rule_from_sympy_step(self, sympy_step: str) -> str:
        """Extract the mathematical rule applied from SymPy step"""
        
        if not sympy_step:
            return ""
        
        # Common rule patterns
        rule_keywords = {
            'solve': 'Equation solving',
            'simplify': 'Algebraic simplification',
            'diff': 'Differentiation',
            'integrate': 'Integration',
            'substitute': 'Substitution',
            'factor': 'Factorization',
            'expand': 'Expansion'
        }
        
        for keyword, rule in rule_keywords.items():
            if keyword in sympy_step.lower():
                return rule
        
        return "Mathematical transformation"
    
    def _create_step_latex(self, expressions: Dict[str, str]) -> str:
        """Create LaTeX representation for a step"""
        
        input_expr = expressions.get('input', '')
        output_expr = expressions.get('output', '')
        
        if input_expr and output_expr:
            return f"$${input_expr} \\Rightarrow {output_expr}$$"
        elif output_expr:
            return f"$${output_expr}$$"
        else:
            return ""
    
    def _classify_step(self, step_text: str) -> str:
        """Classify the type of step based on its content"""
        
        step_lower = step_text.lower()
        
        if any(word in step_lower for word in ['isolate', 'move', 'subtract', 'add']):
            return 'isolation'
        elif any(word in step_lower for word in ['divide', 'multiply', 'solve']):
            return 'solving'
        elif any(word in step_lower for word in ['simplify', 'combine']):
            return 'simplification'
        elif any(word in step_lower for word in ['substitute', 'replace']):
            return 'substitution'
        elif any(word in step_lower for word in ['verify', 'check']):
            return 'verification'
        else:
            return 'general'
    
    def _generate_summary(self, problem: str, num_steps: int, verification: Dict[str, Any]) -> str:
        """Generate a summary of the solution process"""
        
        problem_type = "algebraic" if "solve" in problem.lower() else "mathematical"
        verification_status = verification.get('status', 'unknown')
        
        summary = f"Solved {problem_type} problem in {num_steps} steps. "
        
        if verification_status == 'match':
            summary += "✅ Solution verified by SymPy."
        elif verification_status == 'mismatch':
            summary += "⚠️ Solution differs from SymPy result."
        else:
            summary += "Verification pending."
        
        return summary
    
    def export_to_json(self, visual_solution: VisualSolution) -> str:
        """Export visual solution to JSON format"""
        
        def step_to_dict(step: VisualStep) -> dict:
            return {
                'step_number': step.step_number,
                'title': step.title,
                'description': step.description,
                'input_expression': step.input_expression,
                'output_expression': step.output_expression,
                'rule_applied': step.rule_applied,
                'latex_representation': step.latex_representation,
                'metadata': step.metadata or {}
            }
        
        solution_dict = {
            'problem': visual_solution.problem,
            'steps': [step_to_dict(step) for step in visual_solution.steps],
            'final_result': visual_solution.final_result,
            'verification_info': visual_solution.verification_info,
            'summary': visual_solution.summary
        }
        
        return json.dumps(solution_dict, indent=2)
    
    def create_html_visualization(self, visual_solution: VisualSolution) -> str:
        """Create HTML visualization of the solution"""
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Mathematical Solution</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }}
        .problem {{ background: #f0f8ff; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
        .step {{ margin: 15px 0; padding: 15px; border-left: 3px solid #4CAF50; background: #f9f9f9; }}
        .step-title {{ font-weight: bold; color: #2E7D32; margin-bottom: 8px; }}
        .expression {{ font-family: 'Courier New', monospace; margin: 10px 0; }}
        .verification {{ padding: 10px; border-radius: 5px; margin-top: 20px; }}
        .verification.match {{ background: #d4edda; color: #155724; }}
        .verification.mismatch {{ background: #f8d7da; color: #721c24; }}
        .summary {{ font-style: italic; margin-top: 20px; color: #666; }}
    </style>
</head>
<body>
    <div class="problem">
        <h2>Problem: {visual_solution.problem}</h2>
    </div>
"""
        
        for step in visual_solution.steps:
            html += f"""
    <div class="step">
        <div class="step-title">{step.title}</div>
        <p>{step.description}</p>
        {f'<div class="expression">{step.latex_representation}</div>' if step.latex_representation else ''}
        {f'<small><em>Rule: {step.rule_applied}</em></small>' if step.rule_applied else ''}
    </div>"""
        
        verification_class = visual_solution.verification_info.get('status', 'unknown')
        html += f"""
    <div class="verification {verification_class}">
        <strong>Final Result:</strong> {visual_solution.final_result}<br>
        <strong>Verification:</strong> {visual_solution.verification_info.get('explanation', 'No verification info')}
    </div>
    
    <div class="summary">
        {visual_solution.summary}
    </div>
</body>
</html>"""
        
        return html


def test_step_visualizer():
    """Test the step visualizer"""
    visualizer = StepVisualizer()
    
    print("🧪 Testing Step Visualizer")
    print("=" * 50)
    
    # Test data
    problem = "solve 2x+5=0"
    ai_steps = [
        "Step 1: Isolate the variable term by subtracting 5 from both sides: 2x + 5 - 5 = 0 - 5",
        "Step 2: Simplify to get 2x = -5",
        "Step 3: Divide both sides by 2 to solve for x: x = -5/2"
    ]
    sympy_steps = [
        "Parsing equation: solve 2x+5=0",
        "Equation in symbolic form: Eq(2*x + 5, 0)",
        "Solving for: x",
        "Final result: Solutions: [-5/2]"
    ]
    final_result = "-5/2"
    verification = {
        'status': 'match',
        'explanation': 'AI solution matches SymPy result',
        'confidence': 0.95
    }
    
    # Create visual solution
    visual_solution = visualizer.create_visual_solution(
        problem, ai_steps, sympy_steps, final_result, verification
    )
    
    print(f"Problem: {visual_solution.problem}")
    print(f"Steps: {len(visual_solution.steps)}")
    print(f"Summary: {visual_solution.summary}")
    
    for step in visual_solution.steps:
        print(f"  {step.step_number}. {step.title}")
        print(f"     {step.description[:60]}...")
        if step.rule_applied:
            print(f"     Rule: {step.rule_applied}")


if __name__ == "__main__":
    test_step_visualizer()