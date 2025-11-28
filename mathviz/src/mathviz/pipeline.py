"""
AI-Enhanced main orchestration pipeline with comprehensive error handling and rate limiting.
"""

import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from .parser import MathParser
from .validator import MathValidator
from .solver import MathSolver
from .reasoning import ReasoningGenerator
from .viz import Visualizer
from .schemas import MathSolution
try:
    from .ai_parser import AIMathParser, ParseResult
    AI_PARSER_AVAILABLE = True
except ImportError:
    AI_PARSER_AVAILABLE = False
    print("AI parser not available - using regex parser")
try:
    from .graph_visualizer import GraphConfig, VisualizationResult
    GRAPH_VISUALIZATION_AVAILABLE = True
except ImportError:
    GRAPH_VISUALIZATION_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class PipelineConfig:
    """Configuration for the MathViz pipeline"""
    use_ai: bool = True
    ai_first: bool = True
    use_ai_parser: bool = True
    enable_interactive_graphs: bool = True
    enable_rate_limiting: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0


class MathVizPipeline:
    """AI-Enhanced main pipeline orchestrating the comprehensive math problem solving process."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        if config is None:
            config = PipelineConfig()
        
        self.config = config
        self.last_request_time = 0.0
        
        # Initialize components with AI configuration
        if config.use_ai_parser and AI_PARSER_AVAILABLE:
            self.parser = AIMathParser()
            self.use_ai_parser = True
        else:
            self.parser = MathParser()
            self.use_ai_parser = False
        
        self.validator = MathValidator()
        self.solver = MathSolver(
            use_ai=config.use_ai, 
            ai_first=config.ai_first
        )
        self.reasoner = ReasoningGenerator(
            use_ai=config.use_ai, 
            ai_first=config.ai_first
        )
        self.visualizer = Visualizer(
            enable_interactive_graphs=config.enable_interactive_graphs
        )
        
        logger.info(f"MathVizPipeline initialized - AI: {config.use_ai}, AI Parser: {self.use_ai_parser}, Interactive Graphs: {config.enable_interactive_graphs}")

    def process(self, problem_text: str) -> MathSolution:
        """Process a math problem through the complete AI-enhanced pipeline."""
        start_time = time.time()
        
        try:
            # Apply rate limiting if enabled
            if self.config.enable_rate_limiting:
                self._enforce_rate_limit()
            
            logger.info(f"Processing problem: {problem_text[:100]}...")
            
            # Parse natural language to structured problem with retries
            problem = self._parse_with_retries(problem_text)
            
            # Validate the problem
            validation_result = self.validator.validate(problem)
            if not validation_result:
                raise ValueError("Problem validation failed")
            
            # Solve the problem with AI enhancement
            solution = self._solve_with_retries(problem)
            
            # Enhance solution with interactive graphs if available
            solution = self._enhance_with_interactive_visualization(solution)
            
            processing_time = time.time() - start_time
            logger.info(f"Problem processed successfully in {processing_time:.2f}s")
            
            # Add processing metadata
            if solution.metadata is None:
                solution.metadata = {}
            solution.metadata.update({
                "processing_time": processing_time,
                "ai_enabled": self.config.use_ai,
                "interactive_graphs_enabled": self.config.enable_interactive_graphs,
                "pipeline_version": "ai-enhanced"
            })
            
            return solution
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Pipeline processing failed after {processing_time:.2f}s: {str(e)}")
            
            # Return error solution
            from .schemas import MathProblem, MathSolution
            fallback_problem = MathProblem(
                problem_text=problem_text,
                problem_type="error",
                variables=[],
                equations=[],
                constraints=[],
                goal="Error handling"
            )
            
            return MathSolution(
                problem=fallback_problem,
                solution_steps=[],
                final_answer={"error": str(e)},
                reasoning=f"Processing failed: {str(e)}",
                visualization="",
                metadata={
                    "error": True,
                    "processing_time": processing_time,
                    "pipeline_version": "ai-enhanced"
                }
            )
    
    def process_with_graph_config(self, problem_text: str, graph_config: Optional[GraphConfig] = None) -> Dict[str, Any]:
        """Process a problem and return solution with interactive graph data."""
        solution = self.process(problem_text)
        
        result = {
            "solution": solution,
            "interactive_graph": None,
            "desmos_url": None,
            "graph_html": None
        }
        
        # Generate interactive visualizations if available
        if self.config.enable_interactive_graphs:
            try:
                # Generate interactive graph
                interactive_result = self.visualizer.generate_interactive_graph(
                    solution.solution_steps if hasattr(solution, 'solution_steps') else [],
                    solution,
                    graph_config
                )
                
                if interactive_result:
                    result["interactive_graph"] = interactive_result
                    result["desmos_url"] = interactive_result.graph_url
                    result["graph_html"] = interactive_result.graph_html

                    # CRITICAL: Set the visualization field for frontend display
                    if interactive_result.graph_html:
                        # Update the solution object's visualization field
                        if hasattr(solution, '__dict__'):
                            solution.visualization = interactive_result.graph_html
                        logger.info("Contour visualization set for frontend display")

                    logger.info("Interactive graph generated successfully")
                else:
                    logger.warning("Interactive graph generation failed")
                    
            except Exception as e:
                logger.error(f"Error generating interactive visualization: {e}")
        
        return result
    
    def _enforce_rate_limit(self) -> None:
        """Enforce rate limiting between pipeline requests."""
        if self.config.retry_delay > 0:
            elapsed = time.time() - self.last_request_time
            if elapsed < self.config.retry_delay:
                wait_time = self.config.retry_delay - elapsed
                logger.debug(f"Rate limiting: waiting {wait_time:.2f}s")
                time.sleep(wait_time)
            self.last_request_time = time.time()
    
    def _parse_with_retries(self, problem_text: str):
        """Parse problem with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Parsing attempt {attempt + 1}/{self.config.max_retries}")
                
                if self.use_ai_parser:
                    # Use AI parser
                    parse_result = self.parser.parse(problem_text)
                    if not parse_result.success:
                        raise ValueError(f"AI parsing failed: {parse_result.error}")
                    logger.info(f"AI parsing successful (confidence: {parse_result.confidence:.2f})")
                    return parse_result.problem
                else:
                    # Use regex parser
                    return self.parser.parse(problem_text)
                    
            except Exception as e:
                last_exception = e
                logger.warning(f"Parsing attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
        
        raise last_exception or Exception("Parsing failed after all retries")
    
    def _solve_with_retries(self, problem):
        """Solve problem with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                logger.debug(f"Solving attempt {attempt + 1}/{self.config.max_retries}")
                return self.solver.solve(problem)
            except Exception as e:
                last_exception = e
                logger.warning(f"Solving attempt {attempt + 1} failed: {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
        
        raise last_exception or Exception("Solving failed after all retries")
    
    def _enhance_with_interactive_visualization(self, solution: MathSolution) -> MathSolution:
        """Enhance solution with interactive visualization data."""
        if not self.config.enable_interactive_graphs or not GRAPH_VISUALIZATION_AVAILABLE:
            return solution
        
        try:
            # Extract expressions for visualization
            trace_steps = []
            if hasattr(solution, 'solution_steps'):
                # Convert solution steps back to trace format if needed
                from .trace import Step, StepTrace
                trace = StepTrace(problem_id="enhanced_viz")
                
                for step_dict in solution.solution_steps:
                    if isinstance(step_dict, dict):
                        step = Step(
                            step_id=str(step_dict.get('step_id', 0)),
                            description=step_dict.get('description', step_dict.get('operation', 'Mathematical operation')),
                            operation=step_dict.get('operation', 'unknown'),
                            input_state={"expression": step_dict.get('expression_before', '')},
                            output_state={"expression": step_dict.get('expression_after', '')},
                            reasoning=step_dict.get('justification', step_dict.get('reasoning', ''))
                        )
                        trace.add_step(step)
                
                # Generate interactive graph
                interactive_result = self.visualizer.generate_interactive_graph(trace, solution)
                
                if interactive_result and interactive_result.success:
                    # Add visualization data to solution metadata
                    if solution.metadata is None:
                        solution.metadata = {}
                    
                    solution.metadata['interactive_visualization'] = {
                        'desmos_url': interactive_result.graph_url,
                        'interactive_url': interactive_result.interactive_url,
                        'has_html': interactive_result.graph_html is not None,
                        'visualization_metadata': interactive_result.metadata
                    }
                    
                    logger.info("Enhanced solution with interactive visualization")
            
        except Exception as e:
            logger.warning(f"Failed to enhance with interactive visualization: {e}")
        
        return solution
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status and configuration."""
        return {
            "config": self.config.__dict__,
            "components": {
                "parser": "initialized",
                "validator": "initialized",
                "solver": f"AI: {getattr(self.solver, 'use_ai', False)}",
                "reasoner": f"AI: {getattr(self.reasoner, 'use_ai', False)}",
                "visualizer": f"Interactive: {getattr(self.visualizer, 'enable_interactive_graphs', False)}"
            },
            "capabilities": {
                "ai_solving": self.config.use_ai,
                "interactive_graphs": self.config.enable_interactive_graphs and GRAPH_VISUALIZATION_AVAILABLE,
                "advanced_calculus": hasattr(self.solver, 'advanced_calc') and self.solver.advanced_calc is not None,
                "graph_visualization_available": GRAPH_VISUALIZATION_AVAILABLE
            }
        }
