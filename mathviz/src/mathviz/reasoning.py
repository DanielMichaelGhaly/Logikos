"""
AI-Enhanced reasoning generation from solution traces with natural language explanations.
Supports both AI-powered and template-based reasoning generation.
"""

from typing import Dict, List, Optional
import logging
from .trace import StepTrace, Step

try:
    from .ai_apis import generate_reasoning_with_ai, AIResponse
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("AI APIs not available for reasoning - using template-based generation only")

logger = logging.getLogger(__name__)


class ReasoningGenerator:
    """Generate human-readable reasoning from solution traces using AI and template-based methods."""

    def __init__(self, use_ai: bool = True, ai_first: bool = True):
        """Initialize with AI capabilities and operation templates for fallback."""
        self.use_ai = use_ai and AI_AVAILABLE
        self.ai_first = ai_first  # Try AI first, then fallback to templates
        
        logger.info(f"ReasoningGenerator initialized - AI: {self.use_ai}")
        self.operation_templates = {
            # Parsing operations
            'parse_equation': "We start by parsing the equation: {before}",
            'parse_expression': "First, let's identify the expression: {after}",
            'symbolic_conversion': "Converting to symbolic form: {after}",
            
            # Algebraic operations
            'identify_target': "Our goal is to {after}",
            'solve': "Solving the equation(s): {before} \n→ Solutions: {after}",
            'substitute': "Substituting {before} into the equation gives us {after}",
            'simplify': "Simplifying the expression: {before} = {after}",
            'isolate': "Isolating the variable: {before} becomes {after}",
            
            # Calculus operations
            'differentiate': "Taking the derivative: d/dx({before}) = {after}",
            'integrate': "Integrating: ∫{before} dx = {after}",
            'apply_chain_rule': "Using the chain rule: {before} → {after}",
            'apply_product_rule': "Using the product rule: {before} → {after}",
            'apply_power_rule': "Using the power rule: {before} → {after}",
            
            # General operations
            'expand': "Expanding: {before} = {after}",
            'factor': "Factoring: {before} = {after}",
            'combine_terms': "Combining like terms: {before} = {after}",
            'apply_identity': "Applying the identity {justification}: {before} = {after}",
            
            # Default template
            'default': "Step {step_id}: {justification}. {before} → {after}"
        }
        
        self.mathematical_language = {
            'equation_solving': [
                "To solve this equation, we need to isolate the variable.",
                "Our approach will be to perform the same operation on both sides.",
                "Let's work step by step to find the solution."
            ],
            'calculus': [
                "For this calculus problem, we'll apply the fundamental rules of differentiation/integration.",
                "Remember that differentiation and integration are inverse operations.",
                "Let's carefully apply the appropriate calculus rules."
            ],
            'simplification': [
                "We can simplify this expression by combining like terms.",
                "Let's reduce this to its simplest form.",
                "Simplification will make the solution clearer."
            ]
        }

    def generate_reasoning(self, trace: StepTrace) -> str:
        """Generate comprehensive reasoning text from a step trace using AI or templates."""
        if not trace.steps:
            return "No solution steps were recorded."
        
        if not trace.success:
            return f"Solution failed: {trace.error_message or 'Unknown error occurred'}"
        
        # Try AI-powered reasoning first if enabled
        if self.use_ai and self.ai_first:
            ai_reasoning = self._generate_ai_reasoning(trace)
            if ai_reasoning:
                return ai_reasoning
        
        # Fallback to template-based reasoning
        template_reasoning = self._generate_template_reasoning(trace)
        
        # Try AI as fallback if template reasoning is minimal and AI not tried yet
        if (len(template_reasoning) < 100 and  # Arbitrary threshold for "minimal"
            self.use_ai and not self.ai_first):
            ai_reasoning = self._generate_ai_reasoning(trace)
            if ai_reasoning:
                return ai_reasoning
        
        return template_reasoning
    
    def _generate_ai_reasoning(self, trace: StepTrace) -> Optional[str]:
        """Generate reasoning using AI APIs."""
        if not self.use_ai:
            return None
        
        try:
            # Create a problem summary from the trace
            problem_summary = self._create_problem_summary(trace)
            solution_summary = self._create_solution_summary(trace)
            
            logger.info("Attempting AI reasoning generation")
            
            ai_response = generate_reasoning_with_ai(problem_summary, solution_summary)
            
            if ai_response.success:
                logger.info(f"AI reasoning generated successfully by {ai_response.provider}")
                # Add AI attribution
                ai_reasoning = ai_response.content
                if ai_response.confidence < 0.8:  # Add disclaimer for low confidence
                    ai_reasoning += "\n\n*Note: This explanation was generated by AI and may require verification.*"
                return ai_reasoning
            else:
                logger.warning(f"AI reasoning failed: {ai_response.error}")
                return None
        
        except Exception as e:
            logger.warning(f"AI reasoning generation error: {e}")
            return None
    
    def _create_problem_summary(self, trace: StepTrace) -> str:
        """Create a summary of the problem from the trace."""
        if not trace.steps:
            return "Mathematical problem solving"
        
        # Extract problem context from first few steps
        initial_steps = trace.steps[:3]
        problem_indicators = []
        
        for step in initial_steps:
            if any(keyword in step.operation for keyword in ['parse', 'equation', 'expression']):
                problem_indicators.append(step.expression_before)
            if step.justification and 'problem' in step.justification.lower():
                problem_indicators.append(step.justification)
        
        if problem_indicators:
            return f"Problem: {' | '.join(problem_indicators[:2])}"
        else:
            return f"Mathematical problem with {len(trace.steps)} solution steps"
    
    def _create_solution_summary(self, trace: StepTrace) -> str:
        """Create a summary of the solution from the trace."""
        if trace.final_state:
            return f"Final answer: {trace.final_state}"
        elif trace.steps:
            last_step = trace.steps[-1]
            return f"Result: {last_step.expression_after}"
        else:
            return "Solution completed"
    
    def _generate_template_reasoning(self, trace: StepTrace) -> str:
        """Generate reasoning using template-based method (original implementation)."""
        # Build the reasoning narrative
        reasoning_parts = []
        
        # Add introduction based on problem context
        intro = self._generate_introduction(trace)
        if intro:
            reasoning_parts.append(intro)
        
        # Process each step
        for i, step in enumerate(trace.steps):
            step_reasoning = self._generate_step_reasoning(step, i, trace)
            if step_reasoning:
                reasoning_parts.append(step_reasoning)
        
        # Add conclusion
        conclusion = self._generate_conclusion(trace)
        if conclusion:
            reasoning_parts.append(conclusion)
        
        return "\n\n".join(reasoning_parts)

    def _generate_introduction(self, trace: StepTrace) -> str:
        """Generate an introduction to the solution based on the problem type."""
        if not trace.steps:
            return ""
        
        first_step = trace.steps[0]
        problem_context = self._identify_problem_context(trace)
        
        intros = {
            'algebraic': "Let's solve this algebraic equation step by step.",
            'calculus_derivative': "We need to find the derivative of the given function.",
            'calculus_integral': "Let's compute the integral of this expression.",
            'optimization': "This is an optimization problem that requires finding extrema.",
            'general': "Let's work through this mathematical problem systematically."
        }
        
        return intros.get(problem_context, intros['general'])

    def _identify_problem_context(self, trace: StepTrace) -> str:
        """Identify the type of mathematical problem from the trace."""
        operations = [step.operation for step in trace.steps]
        
        if any(op in ['differentiate', 'apply_chain_rule', 'apply_product_rule'] for op in operations):
            return 'calculus_derivative'
        elif any(op == 'integrate' for op in operations):
            return 'calculus_integral'
        elif any(op in ['solve', 'isolate', 'substitute'] for op in operations):
            return 'algebraic'
        elif any(op.startswith('optimization') for op in operations):
            return 'optimization'
        else:
            return 'general'

    def _generate_step_reasoning(self, step: Step, step_index: int, trace: StepTrace) -> str:
        """Generate reasoning text for a single step."""
        # Get the template for this operation
        template = self.operation_templates.get(step.operation, self.operation_templates['default'])
        
        # Format the template with step data
        try:
            formatted_step = template.format(
                step_id=step.step_id + 1,  # Human-friendly numbering
                before=step.expression_before,
                after=step.expression_after,
                justification=step.justification,
                operation=step.operation
            )
        except KeyError:
            # Fallback if template formatting fails
            formatted_step = self.operation_templates['default'].format(
                step_id=step.step_id + 1,
                before=step.expression_before,
                after=step.expression_after,
                justification=step.justification,
                operation=step.operation
            )
        
        # Add contextual explanation based on operation type
        context_explanation = self._get_contextual_explanation(step, step_index, trace)
        
        if context_explanation:
            return f"**Step {step.step_id + 1}:** {formatted_step}\n\n{context_explanation}"
        else:
            return f"**Step {step.step_id + 1}:** {formatted_step}"

    def _get_contextual_explanation(self, step: Step, step_index: int, trace: StepTrace) -> str:
        """Provide additional context and explanation for specific operations."""
        operation = step.operation
        
        explanations = {
            'solve': self._explain_solving_step(step),
            'differentiate': self._explain_differentiation_step(step),
            'integrate': self._explain_integration_step(step),
            'simplify': self._explain_simplification_step(step),
            'symbolic_conversion': "This converts the natural language equation into a form that can be mathematically manipulated.",
            'identify_target': "Clearly defining what we're solving for helps guide our solution strategy."
        }
        
        return explanations.get(operation, "")

    def _explain_solving_step(self, step: Step) -> str:
        """Provide explanation for solving operations."""
        if "solutions" in step.expression_after.lower():
            return "The equation has been solved! We can verify our answer by substituting back into the original equation."
        return "This step brings us closer to isolating the variable and finding the solution."

    def _explain_differentiation_step(self, step: Step) -> str:
        """Provide explanation for differentiation operations."""
        explanations = [
            "Differentiation gives us the rate of change of the function.",
            "This tells us how the function's output changes with respect to its input.",
            "The derivative represents the slope of the tangent line at any point on the curve."
        ]
        
        # Simple heuristic to choose explanation
        if "x^" in step.expression_before:
            return "Using the power rule: the derivative of x^n is n·x^(n-1)."
        elif "sin" in step.expression_before or "cos" in step.expression_before:
            return "Using trigonometric differentiation rules: d/dx(sin(x)) = cos(x), d/dx(cos(x)) = -sin(x)."
        else:
            return explanations[0]

    def _explain_integration_step(self, step: Step) -> str:
        """Provide explanation for integration operations."""
        base_explanation = "Integration is the reverse of differentiation - it finds the original function given its rate of change."
        
        if "x^" in step.expression_before:
            return f"{base_explanation} Using the power rule for integration: ∫x^n dx = x^(n+1)/(n+1) + C."
        elif any(trig in step.expression_before for trig in ["sin", "cos"]):
            return f"{base_explanation} Using trigonometric integration rules."
        else:
            return base_explanation

    def _explain_simplification_step(self, step: Step) -> str:
        """Provide explanation for simplification operations."""
        if step.expression_before != step.expression_after:
            return "Simplifying makes the expression easier to work with and understand."
        return "The expression is already in its simplest form."

    def _generate_conclusion(self, trace: StepTrace) -> str:
        """Generate a conclusion summarizing the solution."""
        if not trace.steps:
            return "No solution steps were completed."
        
        if trace.final_state:
            conclusion_parts = ["**Final Answer:**", trace.final_state]
            
            # Add verification suggestion based on problem type
            problem_context = self._identify_problem_context(trace)
            
            if problem_context == 'algebraic':
                conclusion_parts.append(
                    "\n*Verification tip: You can check this answer by substituting it back into the original equation.*"
                )
            elif problem_context in ['calculus_derivative', 'calculus_integral']:
                conclusion_parts.append(
                    "\n*Note: For derivatives, you can verify by differentiating your result. For integrals, you can verify by differentiating your result.*"
                )
            
            return "\n".join(conclusion_parts)
        
        return f"Solution completed in {len(trace.steps)} steps."

    def generate_step_by_step_html(self, trace: StepTrace) -> str:
        """Generate HTML formatted step-by-step reasoning."""
        if not trace.steps:
            return "<p>No solution steps were recorded.</p>"
        
        html_parts = ["<div class='math-reasoning'>"]
        
        # Introduction
        intro = self._generate_introduction(trace)
        if intro:
            html_parts.append(f"<div class='intro'><p>{intro}</p></div>")
        
        # Steps
        html_parts.append("<div class='solution-steps'>")
        for i, step in enumerate(trace.steps):
            step_html = self._generate_step_html(step, i)
            html_parts.append(step_html)
        html_parts.append("</div>")
        
        # Conclusion
        conclusion = self._generate_conclusion(trace)
        if conclusion:
            html_parts.append(f"<div class='conclusion'>{conclusion}</div>")
        
        html_parts.append("</div>")
        
        return "\n".join(html_parts)

    def _generate_step_html(self, step: Step, step_index: int) -> str:
        """Generate HTML for a single step."""
        return f"""
        <div class='step' data-step='{step.step_id}'>
            <h4>Step {step.step_id + 1}</h4>
            <div class='step-content'>
                <div class='before'>From: <code>{step.expression_before}</code></div>
                <div class='after'>To: <code>{step.expression_after}</code></div>
                <div class='justification'>{step.justification}</div>
            </div>
        </div>
        """
