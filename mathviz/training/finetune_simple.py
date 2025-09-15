#!/usr/bin/env python3
"""
Simple fine-tuning script for mathematical reasoning using Hugging Face transformers.
This is a practical implementation that works on most hardware.
"""

import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
import os
import sys
sys.path.append('..')

def load_training_data(file_path: str = "training_data/train.json"):
    """Load the mathematical training dataset."""
    if not os.path.exists(file_path):
        print(f"❌ Training data not found at {file_path}")
        print("Run 'python create_dataset.py' first to generate training data.")
        return None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} training examples")
    return data

def format_training_example(example):
    """Format a single training example for the model."""
    prompt = f"""<s>### Instruction:
{example['instruction']}

### Input:
{example['input']}

### Response:
{example['output']}</s>"""
    
    return {"text": prompt}

def setup_model_and_tokenizer(model_name: str = "microsoft/DialoGPT-small"):
    """Set up the model and tokenizer for training."""
    print(f"🔄 Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    return model, tokenizer

def tokenize_function(examples, tokenizer, max_length=512):
    """Tokenize the training examples."""
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

def finetune_math_model(
    model_name: str = "microsoft/DialoGPT-small",
    output_dir: str = "mathviz_finetuned_model",
    num_epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5
):
    """Fine-tune a model for mathematical reasoning."""
    
    print("🚀 Starting MathViz Model Fine-tuning")
    print("=" * 50)
    
    # Load training data
    train_data = load_training_data()
    if train_data is None:
        return
    
    # Set up model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(model_name)
    
    # Format training data
    print("🔄 Formatting training data...")
    formatted_data = [format_training_example(example) for example in train_data]
    
    # Create dataset
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize the dataset
    print("🔄 Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Split into train/validation
    split_dataset = tokenized_dataset.train_test_split(test_size=0.1)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']
    
    print(f"📊 Train samples: {len(train_dataset)}")
    print(f"📊 Eval samples: {len(eval_dataset)}")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=learning_rate,
        logging_steps=50,
        eval_steps=200,
        save_steps=500,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_pin_memory=False,
        fp16=torch.cuda.is_available(),
        report_to=None,  # Disable wandb/tensorboard
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    
    print("🎯 Starting training...")
    
    # Train the model
    trainer.train()
    
    # Save the final model
    print(f"💾 Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    print("✅ Fine-tuning completed!")
    print(f"📁 Model saved to: {output_dir}")
    
    return output_dir

def test_finetuned_model(model_path: str):
    """Test the fine-tuned model with some example problems."""
    print(f"\n🧪 Testing fine-tuned model from {model_path}")
    
    try:
        # Load the fine-tuned model
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        test_problems = [
            "Solve for x: 2x + 5 = 13",
            "Find the derivative of x^2 + 3x",
            "What is 25% of 80?",
        ]
        
        for problem in test_problems:
            prompt = f"""<s>### Instruction:
Solve the following mathematical problem step by step:

### Input:
{problem}

### Response:
"""
            
            print(f"\n🧮 Problem: {problem}")
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            solution = response.split("### Response:")[-1].strip()
            
            print(f"🤖 AI Solution:")
            print(solution[:300] + "..." if len(solution) > 300 else solution)
            print("-" * 50)
    
    except Exception as e:
        print(f"❌ Error testing model: {e}")

def main():
    """Main function to run fine-tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune a model for mathematical reasoning")
    parser.add_argument("--model", default="microsoft/DialoGPT-small", 
                       help="Base model to fine-tune")
    parser.add_argument("--epochs", type=int, default=3, 
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, 
                       help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, 
                       help="Learning rate")
    parser.add_argument("--output-dir", default="mathviz_finetuned_model", 
                       help="Output directory for the fine-tuned model")
    parser.add_argument("--test-only", action="store_true", 
                       help="Only test an existing model")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_finetuned_model(args.output_dir)
    else:
        # Generate training data if it doesn't exist
        if not os.path.exists("training_data/train.json"):
            print("🔄 Generating training data first...")
            os.system("python create_dataset.py")
        
        # Fine-tune the model
        model_path = finetune_math_model(
            model_name=args.model,
            output_dir=args.output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        # Test the fine-tuned model
        test_finetuned_model(model_path)

if __name__ == "__main__":
    main()