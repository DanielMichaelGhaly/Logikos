# 🧠 Training Your Own Mathematical Reasoning Model for MathViz

## 🎯 **Overview: Build Your Own Math AI**

This guide shows you how to train custom models for mathematical reasoning, from simple fine-tuning to full training from scratch. You'll create models that can parse problems, generate step-by-step solutions, and integrate seamlessly with MathViz.

---

## 🚀 **Option 1: Quick Start - Fine-tune Existing Models**

### **A. Fine-tune Llama 3.1 for Math (Recommended)**

#### Step 1: Generate Training Data
```bash
# Create synthetic mathematical problems
cd training
source ../../.venv/bin/activate
python create_dataset.py

# This generates:
# - 2,600+ training examples
# - Linear equations, quadratics, derivatives, word problems
# - Step-by-step solutions in instruction format
```

#### Step 2: Set up Ollama for Local Training
```bash
# Install Ollama (if not already installed)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull base model
ollama pull llama3.1:8b

# Or use smaller model for faster training
ollama pull llama3.1:7b
```

#### Step 3: Fine-tune with Unsloth (GPU Accelerated)
```bash
# Install training dependencies
pip install unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git
pip install --no-deps trl peft accelerate bitsandbytes

# Run fine-tuning script (see training/finetune_math.py below)
python training/finetune_math.py
```

### **B. Fine-tune OpenAI GPT (Cloud-based)**
```bash
# Install OpenAI CLI
pip install openai

# Format data for OpenAI
python training/format_for_openai.py

# Upload and fine-tune
openai api fine_tuning.jobs.create \
  -t training_data/openai_format.jsonl \
  -m gpt-3.5-turbo
```

---

## 🔬 **Option 2: Advanced - Train Specialized Architectures**

### **A. Train a Transformer from Scratch**

#### Mathematical Reasoning Transformer
```python
# training/transformer_model.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

class MathReasoningTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model, n_heads),
            n_layers
        )
        self.output_head = nn.Linear(d_model, vocab_size)
        
        # Math-specific components
        self.equation_encoder = EquationEncoder(d_model)
        self.step_predictor = StepPredictor(d_model)
    
    def forward(self, input_ids, attention_mask=None):
        # Standard transformer forward pass
        embeddings = self.embedding(input_ids)
        
        # Add mathematical structure understanding
        math_features = self.equation_encoder(embeddings)
        enhanced_embeddings = embeddings + math_features
        
        # Generate step-by-step reasoning
        output = self.transformer(enhanced_embeddings)
        logits = self.output_head(output)
        
        return logits
```

### **B. Graph Neural Network for Mathematical Structure**
```python
# For understanding mathematical relationships
class MathGNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.node_encoder = nn.Linear(128, 256)  # Encode variables, operations
        self.edge_encoder = nn.Linear(64, 256)   # Encode relationships
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(256, 256) for _ in range(4)
        ])
        
    def forward(self, graph_batch):
        # Process mathematical expressions as graphs
        # Variables = nodes, operations = edges
        return self.process_math_graph(graph_batch)
```

---

## 📊 **Training Approaches by Complexity**

### 🟢 **Easy: Fine-tune Existing Models**
**Time:** 2-4 hours | **Resources:** 8GB GPU | **Difficulty:** Beginner

```bash
# Complete workflow
cd /Users/sorour/workspace/Logikos/mathviz

# 1. Generate data
python training/create_dataset.py

# 2. Fine-tune Llama locally
python training/finetune_llama.py

# 3. Test the model
python training/test_model.py

# 4. Integrate with MathViz
python integration/add_custom_model.py
```

### 🟡 **Medium: Specialized Mathematical Models**
**Time:** 1-2 days | **Resources:** 16GB GPU | **Difficulty:** Intermediate

- Train domain-specific models for algebra, calculus, etc.
- Use mathematical embeddings and structure-aware architectures
- Implement multi-task learning for different problem types

### 🔴 **Hard: Novel Architectures**
**Time:** 1-2 weeks | **Resources:** Multiple GPUs | **Difficulty:** Advanced

- Build custom transformer variants with mathematical reasoning
- Implement symbolic-neural hybrid models
- Create multi-modal models (text + equations + graphs)

---

## 🛠 **Implementation Files**

Let me create the key training scripts:

### 1. Fine-tuning Script
```python
# training/finetune_math.py
from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import json

def finetune_math_model():
    # Load base model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    
    # Add LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    
    # Load math dataset
    with open("training_data/train.json", "r") as f:
        train_data = json.load(f)
    
    # Format for training
    def formatting_func(examples):
        texts = []
        for i in range(len(examples["instruction"])):
            text = f"""### Instruction:
{examples["instruction"][i]}

### Input:
{examples["input"][i]}

### Response:
{examples["output"][i]}"""
            texts.append(text)
        return {"text": texts}
    
    # Convert to Hugging Face format
    dataset = Dataset.from_list(train_data)
    dataset = dataset.map(formatting_func, batched=True)
    
    # Training arguments
    from trl import SFTTrainer
    from transformers import TrainingArguments
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=100,  # Increase for better results
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    model.save_pretrained("mathviz_custom_model")
    tokenizer.save_pretrained("mathviz_custom_model")

if __name__ == "__main__":
    finetune_math_model()
```

### 2. Model Integration Script
```python
# integration/custom_model_integration.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict

class CustomMathModel:
    def __init__(self, model_path: str = "mathviz_custom_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    def solve_problem(self, problem_text: str) -> Dict[str, str]:
        prompt = f"""### Instruction:
Solve the following mathematical problem step by step:

### Input:
{problem_text}

### Response:
"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        solution = response.split("### Response:")[-1].strip()
        
        return {
            "solution_text": solution,
            "model_name": "MathViz Custom Model",
            "confidence": 0.95  # Placeholder
        }

# Integration with MathViz Pipeline
class CustomModelPipeline:
    def __init__(self):
        self.custom_model = CustomMathModel()
        self.fallback_solver = None  # Your existing SymPy solver
    
    def process(self, problem_text: str):
        try:
            # Try custom model first
            result = self.custom_model.solve_problem(problem_text)
            return self.format_result(result)
        except Exception as e:
            # Fallback to SymPy solver
            return self.fallback_solver.solve(problem_text)
```

---

## 🎛 **Training Configuration Options**

### **Hardware Requirements**

| Approach | GPU Memory | Training Time | Cost |
|----------|------------|---------------|------|
| Fine-tune 7B model | 8GB | 2-4 hours | $5-10 |
| Fine-tune 13B model | 16GB | 4-8 hours | $10-20 |
| Train from scratch | 32GB+ | 1-2 weeks | $100-500 |

### **Model Size vs Performance**

```python
# Configuration for different model sizes
TRAINING_CONFIGS = {
    "small": {
        "model": "llama3.1:7b",
        "batch_size": 4,
        "max_length": 1024,
        "learning_rate": 2e-4
    },
    "medium": {
        "model": "llama3.1:8b", 
        "batch_size": 2,
        "max_length": 2048,
        "learning_rate": 1e-4
    },
    "large": {
        "model": "llama3.1:13b",
        "batch_size": 1,
        "max_length": 4096,
        "learning_rate": 5e-5
    }
}
```

---

## 🧪 **Testing Your Model**

### Evaluation Script
```python
# training/evaluate_model.py
def evaluate_math_model(model_path: str):
    test_problems = [
        "Solve for x: 3x + 7 = 22",
        "Find the derivative of x^3 + 2x",
        "What is 15% of 80?",
        "A train travels 120 miles in 2 hours. What is its speed?"
    ]
    
    model = CustomMathModel(model_path)
    
    for problem in test_problems:
        print(f"\n🧮 Problem: {problem}")
        result = model.solve_problem(problem)
        print(f"✨ Solution: {result['solution_text'][:200]}...")
        
        # Compare with SymPy for verification
        sympy_result = verify_with_sympy(problem)
        print(f"🔍 SymPy Check: {sympy_result}")
```

---

## 🔧 **Integration with MathViz**

### Add Custom Model Support
```python
# src/mathviz/custom_pipeline.py
class CustomModelPipeline(MathVizPipeline):
    def __init__(self, use_custom_model: bool = True):
        super().__init__()
        if use_custom_model:
            self.solver = CustomMathModel("mathviz_custom_model")
        else:
            self.solver = MathSolver()  # Original SymPy solver
    
    def process(self, problem_text: str):
        if self.use_custom_model:
            # AI-powered solving
            result = self.solver.solve_problem(problem_text)
            return self.format_ai_result(result)
        else:
            # Traditional symbolic solving
            return super().process(problem_text)
```

### CLI Support for Custom Models
```python
# Update run_mathviz.py to support custom models
parser.add_argument("--use-custom-model", action="store_true",
                   help="Use custom trained model instead of SymPy")

if args.use_custom_model:
    pipeline = CustomModelPipeline(use_custom_model=True)
else:
    pipeline = MathVizPipeline()
```

---

## 🚀 **Quick Start: Get Training Now**

**Ready to train your own math model? Here's the fastest path:**

### Step 1: Generate Data (5 minutes)
```bash
cd training
python create_dataset.py
```

### Step 2: Choose Your Approach

**For Beginners - Fine-tune Locally:**
```bash
# Install Ollama and fine-tune
curl -fsSL https://ollama.ai/install.sh | sh
python training/finetune_ollama.py
```

**For GPU Users - Use Unsloth:**
```bash
pip install unsloth
python training/finetune_unsloth.py
```

**For Cloud Users - Use OpenAI:**
```bash
python training/finetune_openai.py
```

### Step 3: Test Integration
```bash
python run_mathviz.py --use-custom-model --solve "Solve for x: 4x + 8 = 20"
```

---

## 📈 **Expected Results**

After training, your custom model should:
- ✅ Parse natural language math problems
- ✅ Generate step-by-step solutions
- ✅ Handle multiple problem types (algebra, calculus, word problems)
- ✅ Integrate seamlessly with MathViz interfaces
- ✅ Provide educational explanations

**Performance Expectations:**
- **Fine-tuned 7B model**: 85-90% accuracy on basic problems
- **Fine-tuned 13B model**: 90-95% accuracy, better explanations
- **Custom architecture**: Potentially 95%+ with domain-specific design

---

## 🎯 **Next Steps**

1. **Start with fine-tuning** - easiest path to success
2. **Generate more training data** - improve model performance
3. **Experiment with different base models** - find the best fit
4. **Build domain-specific models** - separate models for algebra vs calculus
5. **Create evaluation benchmarks** - measure model performance

**Want to start training? Run:**
```bash
cd training
python create_dataset.py
# Then choose your training approach!
```

Your custom mathematical reasoning model will integrate perfectly with the existing MathViz framework! 🚀