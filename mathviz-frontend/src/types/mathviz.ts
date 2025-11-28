export interface MathStep {
  step_number: number;
  rule_name: string;
  before_expression: string;
  after_expression: string;
  explanation: string;
  rule_formula?: string;
  latex_before?: string;
  latex_after?: string;
}

export interface LegacyStep {
  step_id: string;
  description: string;
  operation: string;
  input_state: {
    expression: string;
    rule?: string;
    latex?: string;
  };
  output_state: {
    expression: string;
    latex?: string;
  };
  reasoning: string;
  rule_formula?: string;
}

export interface MathSolution {
  problem: {
    problem_text: string;
    problem_type: string;
    variables: Array<{
      name: string;
      domain: string;
    }>;
    equations: Array<{
      left_side: string;
      right_side: string;
    }>;
    goal: string;
  };
  solution_steps: Array<MathStep | LegacyStep>;
  final_answer: any;
  reasoning: string;
  visualization: string;
  metadata?: {
    processing_time?: number;
    ai_enabled?: boolean;
    step_count?: number;
    error?: boolean;
  };
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  solution?: MathSolution;
  desmos_url?: string;
  desmos_config?: {
    expressions: string[];
    xmin?: number; xmax?: number; ymin?: number; ymax?: number;
  };
  has_solution: boolean;
}

export interface APIResponse {
  success: boolean;
  solution?: MathSolution;
  error?: string;
  desmos_url?: string;
}

export interface DesmosGraphConfig {
  expressions: string[];
  viewport?: {
    xmin: number;
    xmax: number;
    ymin: number;
    ymax: number;
  };
  colors?: string[];
}