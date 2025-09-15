#!/usr/bin/env python3
"""
Custom dataset creation for MathViz model training.
Generates synthetic mathematical problems with step-by-step solutions.
"""

import json
import random
import sympy as sp
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
import os
from pathlib import Path

@dataclass
class MathProblem:
    """Mathematical problem with solution steps."""
    problem_text: str
    problem_type: str
    solution_steps: List[Dict[str, str]]
    final_answer: str
    difficulty: str

class MathDatasetGenerator:
    """Generate synthetic mathematical problems for training."""
    
    def __init__(self):
        self.algebra_templates = [
            "Solve for x: {eq}",
            "Find the value of x when {eq}",
            "What is x if {eq}?",
            "Determine x such that {eq}",
        ]
        
        self.calculus_templates = [
            "Find the derivative of {expr}",
            "What is the derivative of {expr}?",
            "Differentiate {expr}",
            "Compute d/dx({expr})",
        ]
        
        self.integration_templates = [
            "Integrate {expr}",
            "Find the integral of {expr}",
            "What is ∫{expr} dx?",
            "Compute the antiderivative of {expr}",
        ]
    
    def generate_linear_problems(self, count: int = 1000) -> List[MathProblem]:
        """Generate linear equation problems."""
        problems = []
        
        for _ in range(count):
            # Generate coefficients
            a = random.randint(2, 10)
            b = random.randint(1, 20)
            c = random.randint(1, 50)
            
            # Create equation: ax + b = c
            x = sp.Symbol('x')
            equation = sp.Eq(a*x + b, c)
            solution = sp.solve(equation, x)[0]
            
            # Generate problem text
            problem_text = random.choice(self.algebra_templates).format(
                eq=f"{a}x + {b} = {c}"
            )
            
            # Create solution steps
            steps = [
                {
                    "step": 1,
                    "operation": "isolate_variable",
                    "description": f"Subtract {b} from both sides",
                    "equation": f"{a}x = {c - b}"
                },
                {
                    "step": 2,
                    "operation": "solve",
                    "description": f"Divide both sides by {a}",
                    "equation": f"x = {solution}"
                }
            ]
            
            problems.append(MathProblem(
                problem_text=problem_text,
                problem_type="linear_equation",
                solution_steps=steps,
                final_answer=str(solution),
                difficulty="easy"
            ))
        
        return problems
    
    def generate_quadratic_problems(self, count: int = 500) -> List[MathProblem]:
        """Generate quadratic equation problems."""
        problems = []
        
        for _ in range(count):
            # Generate coefficients for factorable quadratics
            r1, r2 = random.randint(-5, 5), random.randint(-5, 5)
            while r1 == r2:  # Ensure distinct roots
                r2 = random.randint(-5, 5)
            
            # Create quadratic from roots: (x - r1)(x - r2) = 0
            x = sp.Symbol('x')
            expanded = sp.expand((x - r1) * (x - r2))
            
            problem_text = f"Find the roots of {expanded} = 0"
            
            steps = [
                {
                    "step": 1,
                    "operation": "factor",
                    "description": "Factor the quadratic expression",
                    "equation": f"(x - {r1})(x - {r2}) = 0"
                },
                {
                    "step": 2,
                    "operation": "zero_product_property",
                    "description": "Apply zero product property",
                    "equation": f"x - {r1} = 0 or x - {r2} = 0"
                },
                {
                    "step": 3,
                    "operation": "solve",
                    "description": "Solve each equation",
                    "equation": f"x = {r1} or x = {r2}"
                }
            ]
            
            problems.append(MathProblem(
                problem_text=problem_text,
                problem_type="quadratic_equation",
                solution_steps=steps,
                final_answer=f"x = {r1}, x = {r2}",
                difficulty="medium"
            ))
        
        return problems
    
    def generate_derivative_problems(self, count: int = 800) -> List[MathProblem]:
        """Generate derivative problems."""
        problems = []
        
        function_types = [
            lambda: (f"x^{random.randint(2, 5)}", f"{random.randint(2, 5)}*x^{random.randint(2, 5)-1}"),
            lambda: (f"{random.randint(2, 8)}x^2", f"{2*random.randint(2, 8)}*x"),
            lambda: ("sin(x)", "cos(x)"),
            lambda: ("cos(x)", "-sin(x)"),
            lambda: (f"x^2 + {random.randint(1, 10)}x", f"2*x + {random.randint(1, 10)}"),
        ]
        
        for _ in range(count):
            func_expr, derivative_expr = random.choice(function_types)()
            
            problem_text = random.choice(self.calculus_templates).format(expr=func_expr)
            
            steps = [
                {
                    "step": 1,
                    "operation": "identify_function",
                    "description": f"Identify the function to differentiate: f(x) = {func_expr}",
                    "equation": f"f(x) = {func_expr}"
                },
                {
                    "step": 2,
                    "operation": "apply_derivative_rules",
                    "description": "Apply appropriate differentiation rules",
                    "equation": f"f'(x) = {derivative_expr}"
                }
            ]
            
            problems.append(MathProblem(
                problem_text=problem_text,
                problem_type="derivative",
                solution_steps=steps,
                final_answer=derivative_expr,
                difficulty="medium"
            ))
        
        return problems
    
    def generate_word_problems(self, count: int = 300) -> List[MathProblem]:
        """Generate word problems."""
        problems = []
        
        templates = [
            {
                "template": "Sarah has {initial} apples. She gives away {given} apples. How many apples does she have left?",
                "type": "subtraction_word_problem",
                "solution_func": lambda i, g: i - g
            },
            {
                "template": "A rectangle has length {length} and width {width}. What is its area?",
                "type": "geometry_word_problem", 
                "solution_func": lambda l, w: l * w
            },
            {
                "template": "John runs at {speed} mph for {time} hours. How far does he travel?",
                "type": "rate_time_distance",
                "solution_func": lambda s, t: s * t
            }
        ]
        
        for _ in range(count):
            template_info = random.choice(templates)
            
            if template_info["type"] == "subtraction_word_problem":
                initial = random.randint(10, 50)
                given = random.randint(1, initial-1)
                problem_text = template_info["template"].format(initial=initial, given=given)
                answer = template_info["solution_func"](initial, given)
                
                steps = [
                    {
                        "step": 1,
                        "operation": "identify_operation",
                        "description": "This is a subtraction problem",
                        "equation": f"{initial} - {given}"
                    },
                    {
                        "step": 2,
                        "operation": "calculate",
                        "description": f"Subtract {given} from {initial}",
                        "equation": f"{answer}"
                    }
                ]
            
            elif template_info["type"] == "geometry_word_problem":
                length = random.randint(3, 15)
                width = random.randint(2, 12)
                problem_text = template_info["template"].format(length=length, width=width)
                answer = template_info["solution_func"](length, width)
                
                steps = [
                    {
                        "step": 1,
                        "operation": "identify_formula",
                        "description": "Area of rectangle = length × width",
                        "equation": f"Area = {length} × {width}"
                    },
                    {
                        "step": 2,
                        "operation": "calculate",
                        "description": f"Multiply {length} by {width}",
                        "equation": f"Area = {answer}"
                    }
                ]
            
            else:  # rate_time_distance
                speed = random.randint(2, 10)
                time = random.randint(1, 8)
                problem_text = template_info["template"].format(speed=speed, time=time)
                answer = template_info["solution_func"](speed, time)
                
                steps = [
                    {
                        "step": 1,
                        "operation": "identify_formula",
                        "description": "Distance = speed × time",
                        "equation": f"Distance = {speed} × {time}"
                    },
                    {
                        "step": 2,
                        "operation": "calculate",
                        "description": f"Multiply {speed} by {time}",
                        "equation": f"Distance = {answer} miles"
                    }
                ]
            
            problems.append(MathProblem(
                problem_text=problem_text,
                problem_type=template_info["type"],
                solution_steps=steps,
                final_answer=str(answer),
                difficulty="easy"
            ))
        
        return problems
    
    def generate_training_dataset(self, output_dir: str = "training_data"):
        """Generate complete training dataset."""
        os.makedirs(output_dir, exist_ok=True)
        
        print("🔄 Generating mathematical problems...")
        
        all_problems = []
        all_problems.extend(self.generate_linear_problems(1000))
        all_problems.extend(self.generate_quadratic_problems(500))
        all_problems.extend(self.generate_derivative_problems(800))
        all_problems.extend(self.generate_word_problems(300))
        
        # Shuffle the dataset
        random.shuffle(all_problems)
        
        # Convert to training format
        training_data = []
        for problem in all_problems:
            # Create instruction-following format
            instruction = "Solve the following mathematical problem step by step:"
            input_text = problem.problem_text
            
            # Format output with steps
            output_parts = ["Let me solve this step by step.\n"]
            for step in problem.solution_steps:
                output_parts.append(f"Step {step['step']}: {step['description']}")
                output_parts.append(f"  {step['equation']}")
            
            output_parts.append(f"\nFinal Answer: {problem.final_answer}")
            output_text = "\n".join(output_parts)
            
            training_data.append({
                "instruction": instruction,
                "input": input_text,
                "output": output_text,
                "problem_type": problem.problem_type,
                "difficulty": problem.difficulty
            })
        
        # Split into train/validation
        split_idx = int(0.8 * len(training_data))
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        # Save datasets
        with open(f"{output_dir}/train.json", "w") as f:
            json.dump(train_data, f, indent=2)
        
        with open(f"{output_dir}/validation.json", "w") as f:
            json.dump(val_data, f, indent=2)
        
        print(f"✅ Generated {len(train_data)} training examples")
        print(f"✅ Generated {len(val_data)} validation examples")
        print(f"💾 Saved to {output_dir}/")
        
        return train_data, val_data

def main():
    """Generate the mathematical training dataset."""
    generator = MathDatasetGenerator()
    train_data, val_data = generator.generate_training_dataset()
    
    print("\n📊 Dataset Statistics:")
    print(f"Total problems: {len(train_data) + len(val_data)}")
    
    # Show problem type distribution
    problem_types = {}
    for item in train_data + val_data:
        ptype = item["problem_type"]
        problem_types[ptype] = problem_types.get(ptype, 0) + 1
    
    for ptype, count in problem_types.items():
        print(f"  {ptype}: {count}")

if __name__ == "__main__":
    main()