import React, { useState } from "react";
import styled from "styled-components";
import { motion, AnimatePresence } from "framer-motion";
import "katex/dist/katex.min.css";
import { InlineMath, BlockMath } from "react-katex";
import { MathSolution, MathStep, LegacyStep } from "../types/mathviz";

const SolutionContainer = styled(motion.div)`
  background: #f9fafb;
  border-radius: 16px;
  padding: 1.5rem;
  margin: 1rem 0;
  border: 1px solid rgba(0, 0, 0, 0.05);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
`;

const FinalAnswer = styled(motion.div)`
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  color: white;
  padding: 1rem 1.5rem;
  border-radius: 12px;
  margin-bottom: 1.5rem;
  font-weight: 600;
  font-size: 1.1rem;
  box-shadow: 0 8px 24px rgba(16, 185, 129, 0.3);
  display: flex;
  align-items: center;

  &::before {
    content: "";
  }
`;

const StepsHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 1rem;
  cursor: pointer;
  padding: 0.5rem;
  border-radius: 8px;
  transition: background-color 0.2s ease;

  &:hover {
    background: rgba(0, 0, 0, 0.02);
  }
`;

const StepsTitle = styled.h3`
  font-weight: 700;
  font-size: 1rem;
  color: #1f2937;
  margin: 0;
  display: flex;
  align-items: center;
`;

const ExpandButton = styled(motion.button)<{ isExpanded: boolean }>`
  background: none;
  border: none;
  color: #6366f1;
  font-weight: 600;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;

  &::after {
    content: "${(props) => (props.isExpanded ? "▲" : "▼")}";
    transition: transform 0.2s ease;
  }
`;

const StepsContainer = styled(motion.div)`
  overflow: hidden;
`;

const StepCard = styled(motion.div)`
  background: white;
  border: 1px solid rgba(0, 0, 0, 0.08);
  border-radius: 12px;
  padding: 1rem;
  margin-bottom: 0.75rem;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
  transition: all 0.2s ease;

  &:hover {
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08);
    transform: translateY(-1px);
  }

  &:last-child {
    margin-bottom: 0;
  }
`;

const StepTitle = styled.div`
  font-weight: 600;
  color: #1f2937;
  font-size: 0.95rem;
  margin-bottom: 0.5rem;
  display: flex;
  align-items: center;

  &::before {
    content: counter(step-counter);
    counter-increment: step-counter;
    background: #6366f1;
    color: white;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 600;
    margin-right: 0.75rem;
    flex-shrink: 0;
  }
`;

const RuleFormula = styled.div`
  background: #eff6ff;
  color: #1e40af;
  padding: 0.5rem 0.75rem;
  border-radius: 8px;
  font-size: 0.85rem;
  font-family: "JetBrains Mono", monospace;
  margin: 0.5rem 0;
  border-left: 3px solid #3b82f6;
`;

const MathTransformation = styled.div`
  background: #f3f4f6;
  padding: 0.75rem;
  border-radius: 10px;
  font-family: "JetBrains Mono", monospace;
  font-size: 0.9rem;
  margin: 0.5rem 0;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  flex-wrap: wrap;
`;

const MathExpression = styled.div`
  display: flex;
  align-items: center;
  padding: 0.5rem;
  background: white;
  border-radius: 6px;
  border: 1px solid rgba(0, 0, 0, 0.05);
`;

const Arrow = styled.div`
  color: #6366f1;
  font-weight: bold;
  font-size: 1.2rem;
`;

const StepExplanation = styled.div`
  color: #6b7280;
  font-size: 0.85rem;
  line-height: 1.4;
  font-style: italic;
  margin-top: 0.5rem;
  padding: 0.5rem;
  background: rgba(107, 114, 128, 0.05);
  border-radius: 6px;
`;

const StepCounter = styled.div`
  counter-reset: step-counter;
`;

interface SolutionDisplayProps {
  solution: MathSolution;
}

const isLegacyStep = (step: MathStep | LegacyStep): step is LegacyStep => {
  return "step_id" in step && "description" in step;
};

const formatMathExpression = (expr: string): string => {
  if (!expr) return "";

  // Convert common math notation to LaTeX
  return expr
    .replace(/\^(\d+)/g, "^{$1}")
    .replace(/\^([a-zA-Z]+)/g, "^{$1}")
    .replace(/sqrt\(([^)]+)\)/g, "\\sqrt{$1}")
    .replace(/\bpi\b/g, "\\pi")
    .replace(/\binfinity\b/g, "\\infty")
    .replace(/\*/g, "\\cdot")
    .replace(/\+-/g, "\\pm");
};

const SafeMath: React.FC<{ children: string; block?: boolean }> = ({
  children,
  block = false,
}) => {
  try {
    const formatted = formatMathExpression(children);
    return block ? (
      <BlockMath>{formatted}</BlockMath>
    ) : (
      <InlineMath>{formatted}</InlineMath>
    );
  } catch (error) {
    // Fallback to plain text if LaTeX parsing fails
    return (
      <code style={{ fontFamily: "JetBrains Mono, monospace" }}>
        {children}
      </code>
    );
  }
};

export const SolutionDisplay: React.FC<SolutionDisplayProps> = ({
  solution,
}) => {
  const [isExpanded, setIsExpanded] = useState(true);

  const formatFinalAnswer = (answer: any): string => {
    if (typeof answer === "object" && answer !== null) {
      if (answer.solutions) {
        return `x = ${JSON.stringify(answer.solutions)}`;
      } else if (answer.derivative) {
        return `f'(x) = ${answer.derivative}`;
      } else if (answer.integral) {
        return `∫f(x)dx = ${answer.integral} + C`;
      } else if (answer.type === "derivative") {
        return `f'(x) = ${answer.derivative || "N/A"}`;
      } else if (answer.type === "algebraic_solution" && answer.solutions) {
        const solutions = answer.solutions;
        return Object.entries(solutions)
          .map(([k, v]) => `${k} = ${v}`)
          .join(", ");
      }
      return JSON.stringify(answer);
    }
    return String(answer);
  };

  // Render HTML visualization if present (e.g., contour map, plotly, gnuplot)
  const renderVisualization = () => {
    if (
      solution.visualization &&
      typeof solution.visualization === "string" &&
      solution.visualization.includes("<")
    ) {
      return (
        <div style={{ margin: "2rem 0", textAlign: "center" }}>
          <div dangerouslySetInnerHTML={{ __html: solution.visualization }} />
        </div>
      );
    }
    // Optionally support graph_html in metadata (type safe)
    const meta = solution.metadata as any;
    if (
      meta &&
      meta.interactive_visualization &&
      meta.interactive_visualization.has_html
    ) {
      const html = meta.interactive_visualization.graph_html;
      if (html) {
        return (
          <div style={{ margin: "2rem 0", textAlign: "center" }}>
            <div dangerouslySetInnerHTML={{ __html: html }} />
          </div>
        );
      }
    }
    return null;
  };

  const renderStep = (step: MathStep | LegacyStep, index: number) => {
    if (isLegacyStep(step)) {
      // Handle new detailed format
      const operation = step.operation
        .replace(/_/g, " ")
        .replace(/\b\w/g, (l) => l.toUpperCase());
      const description = step.reasoning || step.description;
      const ruleFormula = step.rule_formula;

      const beforeExpr = step.input_state?.expression || "";
      const afterExpr = step.output_state?.expression || "";

      return (
        <StepCard
          key={step.step_id}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          <StepTitle>{operation}</StepTitle>

          {ruleFormula && (
            <RuleFormula>
              <SafeMath>{ruleFormula}</SafeMath>
            </RuleFormula>
          )}

          {beforeExpr && afterExpr && beforeExpr !== afterExpr && (
            <MathTransformation>
              <MathExpression>
                <SafeMath>{beforeExpr}</SafeMath>
              </MathExpression>
              <Arrow>→</Arrow>
              <MathExpression>
                <SafeMath>{afterExpr}</SafeMath>
              </MathExpression>
            </MathTransformation>
          )}

          <StepExplanation>{description}</StepExplanation>
        </StepCard>
      );
    } else {
      // Handle legacy format
      return (
        <StepCard
          key={index}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
        >
          <StepTitle>{step.rule_name}</StepTitle>

          {step.rule_formula && (
            <RuleFormula>
              <SafeMath>{step.rule_formula}</SafeMath>
            </RuleFormula>
          )}

          {step.before_expression && step.after_expression && (
            <MathTransformation>
              <MathExpression>
                {step.latex_before ? (
                  <SafeMath>{step.latex_before}</SafeMath>
                ) : (
                  <SafeMath>{step.before_expression}</SafeMath>
                )}
              </MathExpression>
              <Arrow>→</Arrow>
              <MathExpression>
                {step.latex_after ? (
                  <SafeMath>{step.latex_after}</SafeMath>
                ) : (
                  <SafeMath>{step.after_expression}</SafeMath>
                )}
              </MathExpression>
            </MathTransformation>
          )}

          <StepExplanation>{step.explanation}</StepExplanation>
        </StepCard>
      );
    }
  };

  return (
    <SolutionContainer
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{
        type: "spring",
        stiffness: 300,
        damping: 30,
      }}
    >
      {/* Final Answer */}
      {solution.final_answer && (
        <FinalAnswer
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <SafeMath>{formatFinalAnswer(solution.final_answer)}</SafeMath>
        </FinalAnswer>
      )}

      {/* Visualization (contour map, plot, etc.) */}
      {renderVisualization()}

      {/* Solution Steps */}
      {(solution.solution_steps || (solution as any).steps) &&
        (solution.solution_steps || (solution as any).steps).length > 0 && (
          <div>
            <StepsHeader onClick={() => setIsExpanded(!isExpanded)}>
              <StepsTitle>
                Step-by-Step Solution (
                {(solution.solution_steps || (solution as any).steps).length}{" "}
                steps)
              </StepsTitle>
              <ExpandButton
                isExpanded={isExpanded}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                {isExpanded ? "Hide steps" : "Show steps"}
              </ExpandButton>
            </StepsHeader>

            <AnimatePresence>
              {isExpanded && (
                <StepsContainer
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3 }}
                >
                  <StepCounter>
                    {(solution.solution_steps || (solution as any).steps)
                      .slice(0, 10)
                      .map((step: any, index: number) =>
                        renderStep(step, index)
                      )}

                    {(solution.solution_steps || (solution as any).steps)
                      .length > 10 && (
                      <div
                        style={{
                          textAlign: "center",
                          color: "#6b7280",
                          fontSize: "0.85rem",
                          marginTop: "1rem",
                          fontStyle: "italic",
                        }}
                      >
                        +{" "}
                        {(solution.solution_steps || (solution as any).steps)
                          .length - 10}{" "}
                        more steps available
                      </div>
                    )}
                  </StepCounter>
                </StepsContainer>
              )}
            </AnimatePresence>
          </div>
        )}
    </SolutionContainer>
  );
};
