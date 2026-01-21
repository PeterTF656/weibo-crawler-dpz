# DSPy Optimized vs Unoptimized Modules - Complete Reference

> Researched from DSPy Official Documentation

## Table of Contents

1. [Overview](#overview)
2. [Unoptimized (Baseline) DSPy Modules](#unoptimized-baseline-dspy-modules)
3. [Optimized DSPy Modules](#optimized-dspy-modules)
4. [What Gets Optimized](#what-gets-optimized)
5. [How Optimization Works](#how-optimization-works)
6. [Saving and Loading Optimized Modules](#saving-and-loading-optimized-modules)
7. [Practical Differences in Behavior](#practical-differences-in-behavior)
8. [Inspecting Optimized Prompts](#inspecting-optimized-prompts)
9. [Best Practices](#best-practices)

---

## Overview

DSPy modules can exist in two states: **unoptimized (baseline)** and **optimized (compiled)**. The difference between these states is fundamental to understanding DSPy's value proposition:

- **Unoptimized modules** use default prompts and zero-shot behavior
- **Optimized modules** have been processed by a DSPy optimizer (teleprompter) to improve performance through learned prompts, few-shot examples, or fine-tuned weights

The optimization process is called **compilation**, and it transforms a baseline module into a more effective version without changing the module's architecture or code.

---

## Unoptimized (Baseline) DSPy Modules

### Definition

An **unoptimized DSPy module** is the initial state of any DSPy program before optimization. It represents your program's architecture and logic, but uses:

- **Default instructions**: Generic prompt instructions based on the signature
- **No demonstrations**: Zero-shot behavior (no few-shot examples)
- **Base model weights**: No fine-tuning

### Characteristics

**1. Zero-Shot Behavior**

```python
import dspy

# Configure LM
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

# Create an unoptimized module
qa = dspy.ChainOfThought("question -> answer")

# This runs zero-shot (no examples provided to the LM)
result = qa(question="What is the capital of France?")
```

**Source:** https://dspy.ai/api/optimizers/COPRO

The optimizer documentation states: "optimizes signature of student program - note that it may be zero-shot or already pre-optimized (demos already chosen - `demos != []`)". This confirms that unoptimized modules start in zero-shot mode with `demos = []`.

**2. Generic Instructions**

Unoptimized modules use basic signature-derived instructions:

```python
class QASignature(dspy.Signature):
    """Answer questions with short factual answers."""
    question = dspy.InputField()
    answer = dspy.OutputField()

# Unoptimized module uses this basic docstring as instructions
qa = dspy.ChainOfThought(QASignature)
```

**3. Module State**

An unoptimized module's state includes:

```python
# Empty demonstrations
predictor.demos = []

# No training history
predictor.train = []

# No optimization traces
predictor.traces = []

# Default signature with basic instructions
predictor.signature.instructions  # Basic task description
```

**Source:** https://dspy.ai/api/modules/Predict (dump_state method)

**4. Performance Baseline**

Unoptimized modules establish the baseline performance:

```python
# Evaluate base program
base_score = evaluate(base_program, devset=testset)
print(f"Base score: {base_score}")  # e.g., 24%
```

**Source:** https://dspy.ai/learn/optimization/optimizers

The documentation shows: "The optimization is performed using optimizers like `dspy.MIPROv2`... This process can significantly improve the agent's performance, as demonstrated by an increase in its score from 24% to 51%."

---

## Optimized DSPy Modules

### Definition

An **optimized DSPy module** is the result of running a DSPy optimizer on an unoptimized module. The optimization process is called **compilation** and produces a module with:

- **Refined instructions**: Task-specific, optimized prompt instructions
- **Few-shot demonstrations**: Selected or synthesized examples
- **Fine-tuned weights**: (Optional) Model weights adapted to the task

### Compilation Process

```python
from dspy.teleprompt import BootstrapFewShot

# 1. Create unoptimized module
program = dspy.ChainOfThought("question -> answer")

# 2. Define metric
def qa_metric(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# 3. Create optimizer
optimizer = BootstrapFewShot(
    metric=qa_metric,
    max_bootstrapped_demos=4
)

# 4. Compile (optimize) the module
optimized_program = optimizer.compile(
    student=program,
    trainset=trainset
)
```

**Source:** https://dspy.ai/api/optimizers/BootstrapFewShot

### Module Frozen State

After optimization, modules are marked as compiled:

```python
# After compilation
optimized_program._compiled = True
```

**Source:** https://dspy.ai/faqs

The documentation states: "Modules in DSPy can be frozen by setting their `._compiled` attribute to `True`. This indicates that the module has been optimized and its parameters should not be altered. Optimizers like `dspy.BootstrapFewShot` automatically handle this by freezing the student program before propagating demonstrations from the teacher."

---

## What Gets Optimized

DSPy optimizers can modify three main aspects of your modules:

### 1. Prompts and Instructions

**What changes:** The natural language instructions in your signature

**Optimizers that modify prompts:**
- `MIPROv2` - Multi-stage Instruction and Prompt Optimization
- `COPRO` - Coordinate Prompt Optimization
- `GEPA` - Generate, Explore, and Propose Actions

**Example:**

```python
# Before optimization (default instruction)
print(program.predict.signature.instructions)
# Output: "Answer questions with short factual answers."

# After MIPROv2 optimization
print(optimized_program.predict.signature.instructions)
# Output: "Given a question, provide a concise and accurate answer
#          based on factual information. Focus on clarity and precision."
```

**Source:** https://dspy.ai/tutorials/gepa_papillon

The documentation shows: "Prints the optimized signature instructions from the GEPA-generated prompt for the craft_redacted_request predictor."

**How it works:**

```python
# COPRO generates candidate instructions
instruct = dspy.Predict(
    BasicGenerateInstruction,
    n=self.breadth - 1,
    temperature=self.init_temperature,
)(basic_instruction=basic_instruction)

# Add original instruction as candidate
instruct.completions.proposed_instruction.append(basic_instruction)
```

**Source:** https://dspy.ai/api/optimizers/COPRO

### 2. Few-Shot Demonstrations

**What changes:** Example demonstrations added to prompts

**Optimizers that modify demonstrations:**
- `LabeledFewShot` - Uses labeled training examples directly
- `BootstrapFewShot` - Synthesizes examples from successful traces
- `BootstrapFewShotWithRandomSearch` - Explores different demo configurations
- `KNNFewShot` - Dynamically selects similar examples at inference

**Example:**

```python
# Before optimization
print(len(predictor.demos))  # 0 (zero-shot)

# After BootstrapFewShot optimization
print(len(predictor.demos))  # 4 (few-shot with 4 examples)

# Demos are stored as dictionaries
for demo in predictor.demos:
    print(demo)
    # {'question': 'What is Python?', 'answer': 'A programming language'}
```

**Source:** https://dspy.ai/api/modules/Predict (dump_state method)

**How it works:**

```python
# LabeledFewShot: Direct selection from training set
for predictor in self.student.predictors():
    predictor.demos = rng.sample(self.trainset, min(self.k, len(self.trainset)))
```

**Source:** https://dspy.ai/api/optimizers/LabeledFewShot

```python
# BootstrapFewShot: Synthesize from successful traces
# 1. Run program on training examples
# 2. Keep traces where metric returns True
# 3. Use successful traces as demonstrations
```

**Source:** https://dspy.ai/api/optimizers/BootstrapFewShot

### 3. Model Weights (Fine-tuning)

**What changes:** The underlying language model's weights

**Optimizers that modify weights:**
- `BootstrapFinetune` - Fine-tunes model using bootstrapped dataset
- `GEPA` - Can include fine-tuning component

**Example:**

```python
from dspy.teleprompt import BootstrapFinetune

optimizer = BootstrapFinetune(
    metric=validation_metric,
    num_threads=24
)

# Compile (this will finetune the model)
finetuned_program = optimizer.compile(
    student=program,
    trainset=trainset,
    epochs=2,
    bf16=True,
    bsize=6,
    lr=5e-5
)
```

**Source:** https://dspy.ai/tutorials/classification_finetuning

The documentation states: "During bootstrapped finetuning, the teacher program generates reasoning and class selections for each training input. These outputs are traced and form a training dataset for the student program's modules."

### Summary of What Gets Optimized

| Optimizer Type | Instructions | Demonstrations | Weights |
|---------------|--------------|----------------|---------|
| `LabeledFewShot` | ❌ | ✅ (from trainset) | ❌ |
| `BootstrapFewShot` | ❌ | ✅ (synthesized) | ❌ |
| `BootstrapFewShotWithRandomSearch` | ❌ | ✅ (explored) | ❌ |
| `KNNFewShot` | ❌ | ✅ (dynamic) | ❌ |
| `MIPROv2` | ✅ | ✅ | ❌ |
| `COPRO` | ✅ | ✅ | ❌ |
| `GEPA` | ✅ | ✅ | Optional |
| `BootstrapFinetune` | ❌ | ✅ (for training) | ✅ |

**Source:** https://dspy.ai/learn/optimization/optimizers

---

## How Optimization Works

### The Compilation Process

Optimization in DSPy is called **compilation** and follows this general pattern:

```python
optimizer = OptimizerClass(metric=metric, **config)
optimized_program = optimizer.compile(
    student=program,      # Your unoptimized program
    trainset=trainset,    # Training examples
    valset=valset         # Optional validation set
)
```

### Core Optimization Mechanisms

**1. Few-Shot Example Selection (BootstrapFewShot)**

```python
# Step 1: Run program on training examples
for example in trainset:
    with dspy.context(lm=teacher_lm):
        prediction = program(**example.inputs())

    # Step 2: Evaluate with metric
    if metric(example, prediction, trace=True):
        # Step 3: Keep successful traces
        successful_demos.append(prediction.trace)

# Step 4: Assign demos to predictors
for predictor in program.predictors():
    predictor.demos = successful_demos[:max_bootstrapped_demos]
```

**Source:** https://dspy.ai/api/optimizers/BootstrapFewShot

**2. Instruction Optimization (MIPROv2)**

```python
# MIPROv2 works in stages:

# Stage 1: Generate instruction candidates
with dspy.context(lm=prompt_model):
    instruction_candidates = generate_instructions(
        basic_instruction=signature.instructions,
        num_candidates=num_candidates
    )

# Stage 2: Evaluate each candidate on minibatches
for instruction in instruction_candidates:
    signature.instructions = instruction
    score = evaluate(program, devset=minibatch)
    scores[instruction] = score

# Stage 3: Bayesian optimization to find best combination
best_instruction = bayesian_search(
    candidates=instruction_candidates,
    scores=scores,
    num_trials=num_trials
)
```

**Source:** https://dspy.ai/api/optimizers/MIPROv2

The documentation states: "Finally, we use Bayesian Optimization to choose which combinations of instructions and demonstrations work best for each predictor in our program."

**3. Dynamic Example Selection (KNNFewShot)**

```python
# At optimization time:
# 1. Embed all training examples
embeddings = [vectorizer(ex.question) for ex in trainset]

# At inference time:
# 2. Embed the input query
query_embedding = vectorizer(input_question)

# 3. Find k nearest neighbors
nearest_demos = knn_search(query_embedding, embeddings, k=3)

# 4. Use as demonstrations
predictor.demos = nearest_demos
```

**Source:** https://dspy.ai/api/optimizers/KNNFewShot

### Optimizer Categories

**Category 1: Few-Shot Learning**

Adds demonstrations without changing instructions:

- `LabeledFewShot` - Direct use of labeled examples
- `BootstrapFewShot` - Synthesize from successful traces
- `BootstrapFewShotWithRandomSearch` - Explore demo configurations
- `KNNFewShot` - Semantic similarity-based selection

**Category 2: Prompt Optimization**

Refines both instructions and demonstrations:

- `MIPROv2` - Multi-stage instruction and prompt optimization
- `COPRO` - Coordinate prompt optimization
- `GEPA` - Generate, explore, propose actions

**Category 3: Weight Optimization**

Fine-tunes the model:

- `BootstrapFinetune` - Model fine-tuning with bootstrapped data

**Source:** https://dspy.ai/index

The documentation states: "The **BootstrapRS** optimizer synthesizes good few-shot examples for every module... The **GEPA** and **MIPROv2** optimizers propose and intelligently explore better natural-language instructions... The **BootstrapFinetune** optimizer builds datasets for your modules and uses them to finetune the language model weights."

---

## Saving and Loading Optimized Modules

### Two Save Modes

DSPy provides two modes for saving optimized modules:

**Mode 1: State-Only Saving (Recommended)**

Saves only the optimized parameters (demos, instructions) to a JSON or pickle file.

```python
# Save state only (requires .json or .pkl extension)
optimized_program.save("optimized_program.json", save_program=False)
# or
optimized_program.save("optimized_program.pkl", save_program=False)
```

**Saves:**
- Demonstrations (`predictor.demos`)
- Optimized signature instructions
- Field prefixes and descriptions
- Traces (if any)
- Training history

**Requires:** You must have the same program architecture to load

**Mode 2: Full Program Saving**

Saves the entire program including architecture using cloudpickle.

```python
# Save full program (directory path, no extension)
optimized_program.save(
    "optimized_program_dir/",
    save_program=True
)
```

**Saves:**
- Everything from state-only mode
- Program architecture (class definitions)
- Custom modules (if specified)

**Source:** https://dspy.ai/api/modules/Predict

### Save Method Signature

```python
def save(self, path, save_program=False, modules_to_serialize=None):
    """Save the module.

    Save the module to a directory or a file. There are two modes:
    - `save_program=False`: Save only the state of the module to a json or
      pickle file, based on the value of the file extension.
    - `save_program=True`: Save the whole module to a directory via cloudpickle,
      which contains both the state and architecture of the model.

    Args:
        path (str): Path to the saved state file (.json or .pkl) when
            `save_program=False`, and a directory when `save_program=True`.
        save_program (bool): If True, save the whole module to a directory
            via cloudpickle, otherwise only save the state.
        modules_to_serialize (list): A list of modules to serialize with
            cloudpickle's `register_pickle_by_value`.
    """
```

**Source:** https://dspy.ai/api/modules/Module

### Loading Optimized Modules

**Method 1: Load State into Existing Architecture**

```python
import dspy

# 1. Configure the same LM used during optimization
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

# 2. Create the same program architecture
program = dspy.ChainOfThought("question -> answer")

# 3. Load the optimized state
program.load("optimized_program.json")

# 4. Use the loaded program
result = program(question="What is Rust?")
```

**Method 2: Load Full Program**

```python
# Load full program (if saved with save_program=True)
program = dspy.load("optimized_program_dir/")

# Use immediately
result = program(question="What is Rust?")
```

**Source:** https://dspy.ai/tutorials/agents

Example from documentation:

```python
optimized_react.save("optimized_react.json")

loaded_react = dspy.ReAct(
    "claim -> titles: list[str]",
    tools=[search_wikipedia, lookup_wikipedia],
    max_iters=20
)
loaded_react.load("optimized_react.json")

# Use loaded module
loaded_react(claim="...").titles
```

### Saving with Custom Modules

If your program uses custom modules, specify them for serialization:

```python
import my_custom_module

compiled_program.save(
    "./dspy_program/",
    save_program=True,
    modules_to_serialize=[my_custom_module]
)
```

**Source:** https://dspy.ai/tutorials/saving

### What Gets Saved

**State Dictionary Structure:**

```python
state = {
    "demos": [
        {"question": "What is Python?", "answer": "A programming language"},
        {"question": "What is Java?", "answer": "An OOP language"}
    ],
    "signature": {
        "instructions": "Optimized instruction text...",
        "fields": [
            {"prefix": "Question:", "description": "The input question"},
            {"prefix": "Answer:", "description": "The factual answer"}
        ]
    },
    "traces": [],
    "train": [],
    "lm": {...}  # LM configuration
}
```

**Source:** https://dspy.ai/api/modules/Predict (dump_state method)

---

## Practical Differences in Behavior

### Performance Differences

**Typical improvements from optimization:**

```python
# Before optimization
base_score = evaluate(base_program, devset=testset)
print(f"Base score: {base_score}")  # 24%

# After optimization (MIPROv2)
optimized_score = evaluate(optimized_program, devset=testset)
print(f"Optimized score: {optimized_score}")  # 51%
print(f"Improvement: {optimized_score - base_score}")  # +27%
```

**Source:** https://dspy.ai/learn/optimization/optimizers

The documentation shows: "This process can significantly improve the agent's performance, as demonstrated by an increase in its score from 24% to 51%."

**RAG example:**

```python
# Unoptimized RAG: 53%
# Optimized with MIPROv2: 61%
# Improvement: +8%
```

**Source:** https://dspy.ai/tutorials/rag

### Prompt Size Differences

**Unoptimized (zero-shot):**

```
System: Answer questions with short factual answers.

User: What is the capital of France?

[No few-shot examples]
[~50 tokens total]
```

**Optimized (few-shot with 3 demos):**

```
System: Given a question, provide a concise and accurate answer based on
factual information. Focus on clarity and precision.

Example 1:
Question: What is Python?
Answer: Python is a high-level programming language.

Example 2:
Question: What is JavaScript?
Answer: JavaScript is a scripting language for web development.

Example 3:
Question: What is Java?
Answer: Java is an object-oriented programming language.

User: What is the capital of France?

[~250-300 tokens total]
```

### Inference Behavior Differences

**1. Static vs Dynamic Demonstrations**

**Unoptimized or Static Optimizers (BootstrapFewShot):**

```python
# Same demos used for all inputs
optimized_program(question="What is Python?")    # Uses demos A, B, C
optimized_program(question="Who is Einstein?")   # Uses same demos A, B, C
```

**KNNFewShot Optimizer:**

```python
# Different demos selected based on input similarity
optimized_program(question="What is Python?")    # Uses programming demos
optimized_program(question="Who is Einstein?")   # Uses biography demos
```

**Source:** https://dspy.ai/api/optimizers/KNNFewShot

**2. Consistency and Reasoning Quality**

**Unoptimized:**
- More variable outputs
- Generic reasoning patterns
- May miss task-specific nuances

**Optimized:**
- More consistent outputs
- Task-specific reasoning from demonstrations
- Better alignment with training data patterns

### Token Usage Differences

**Cost implications:**

```python
# Unoptimized (zero-shot)
# Input: ~50 tokens
# Output: ~30 tokens
# Total: ~80 tokens per call

# Optimized (few-shot with 4 demos)
# Input: ~300 tokens (includes demos)
# Output: ~30 tokens
# Total: ~330 tokens per call

# Cost increase: ~4x per call
# Quality increase: Often 20-50% improvement
```

### Cold Start Differences

**Unoptimized:**

```python
# Fast cold start - no loading required
program = dspy.ChainOfThought("question -> answer")
# Ready to use immediately
```

**Optimized:**

```python
# Requires loading state
program = dspy.ChainOfThought("question -> answer")
program.load("optimized.json")  # Loading time: ~100-500ms
# Then ready to use
```

### Memory Footprint

**Unoptimized:**
- Minimal memory: ~1-5 KB
- Only signature and architecture

**Optimized:**
- Larger memory: ~50-500 KB
- Stores demos, optimized instructions, traces
- Depends on number and size of demonstrations

---

## Inspecting Optimized Prompts

### Using `dspy.inspect_history()`

After running an optimized program, inspect the actual prompts sent to the LM:

```python
# Run optimized program
result = optimized_program(question="What is the capital of France?")

# Inspect the last prompt
dspy.inspect_history(n=1)
```

**Output shows:**

```
System message:
Given a question, provide a concise and accurate answer based on factual
information. Focus on clarity and precision.

---

Example 1:
Question: What is Python?
Reasoning: Python is a widely-used programming language known for its
simplicity and versatility.
Answer: Python is a high-level programming language.

Example 2:
Question: What is JavaScript?
Reasoning: JavaScript is primarily used for web development to create
interactive web pages.
Answer: JavaScript is a scripting language for web development.

---

User message:
Question: What is the capital of France?

Assistant response:
Reasoning: France is a country in Europe, and its capital city is Paris.
Answer: Paris
```

**Source:** https://dspy.ai/tutorials/multihop_search

The documentation states: "Inspecting the optimized prompts is a key step in understanding what the program has learned. After running an optimized program on a query, `dspy.inspect_history(n=2)` can be used to view the prompts used by the sub-modules within the program."

### Comparing Before and After Optimization

```python
# Before optimization
base_program(question="What is the capital of France?")
dspy.inspect_history(n=1)
# Shows: Zero-shot prompt with basic instructions

# After optimization
optimized_program(question="What is the capital of France?")
dspy.inspect_history(n=1)
# Shows: Few-shot prompt with optimized instructions and examples
```

**Source:** https://dspy.ai/tutorials/math

The documentation states: "After optimizing a DSPy module, you can inspect the changes in the prompt history. If MLflow tracing was enabled, you can compare prompts before and after optimization within the MLflow UI."

### Inspecting Signature State

```python
# Inspect optimized instructions
print(optimized_program.predict.signature.instructions)
# Output: "Given a question, provide a concise and accurate..."

# Inspect field prefixes
for field_name, field in optimized_program.predict.signature.fields.items():
    prefix = field.json_schema_extra.get("prefix", "")
    description = field.json_schema_extra.get("desc", "")
    print(f"{field_name}: {prefix} - {description}")
```

**Source:** https://dspy.ai/tutorials/gepa_papillon

### Accessing Demonstration State

```python
# Access the demonstrations
for predictor in optimized_program.predictors():
    print(f"Number of demos: {len(predictor.demos)}")

    for i, demo in enumerate(predictor.demos):
        print(f"\nDemo {i+1}:")
        print(f"  Question: {demo['question']}")
        print(f"  Answer: {demo['answer']}")
```

---

## Best Practices

### 1. Always Compare Unoptimized vs Optimized

```python
# Establish baseline
base_score = evaluate(base_program, devset=testset)
print(f"Base score: {base_score}")

# Optimize
optimized_program = optimizer.compile(base_program, trainset=trainset)

# Evaluate improvement
optimized_score = evaluate(optimized_program, devset=testset)
print(f"Optimized score: {optimized_score}")
print(f"Improvement: {optimized_score - base_score}")
```

### 2. Version Your Optimized Programs

```python
# Save with descriptive names
optimized.save(f"program_v1_mipro_{len(trainset)}examples.json")

# Include metadata
metadata = {
    "optimizer": "MIPROv2",
    "base_score": 0.45,
    "optimized_score": 0.68,
    "train_size": len(trainset),
    "date": "2024-01-15"
}

with open("program_v1_metadata.json", "w") as f:
    json.dump(metadata, f)
```

### 3. Choose Optimizer Based on Dataset Size

| Dataset Size | Recommended Optimizer |
|-------------|----------------------|
| 5-10 examples | `LabeledFewShot(k=5)` |
| 10-50 examples | `BootstrapFewShot` |
| 50-200 examples | `BootstrapFewShotWithRandomSearch` |
| 200+ examples | `MIPROv2(auto="light")` |

**Source:** https://dspy.ai/learn/optimization/optimizers

### 4. Use State-Only Saving for Production

```python
# Preferred: State-only saving (JSON)
optimized.save("program.json", save_program=False)

# Avoid: Full program saving (security risk)
# optimized.save("program_dir/", save_program=True)  # Uses pickle
```

**Source:** https://dspy.ai/api/modules/Module

The documentation warns: "Loading untrusted .pkl files can run arbitrary code, which may be dangerous. To avoid this, prefer saving using json format."

### 5. Match LM Configuration on Load

```python
# IMPORTANT: Use same LM as during optimization
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))  # Must match!

program = dspy.ChainOfThought("question -> answer")
program.load("optimized.json")
```

### 6. Inspect Prompts to Understand Behavior

```python
# After optimization, always inspect
result = optimized_program(question="test question")
dspy.inspect_history(n=1)

# Check what was learned
print(optimized_program.predict.signature.instructions)
print(f"Number of demos: {len(optimized_program.predict.demos)}")
```

### 7. Consider Token Cost Trade-offs

```python
# For cost-sensitive applications:
# Option 1: Use fewer demos
optimizer = BootstrapFewShot(max_bootstrapped_demos=2)  # Instead of 4

# Option 2: Optimize then fine-tune
# 1. Get good demos with BootstrapFewShot
# 2. Fine-tune a smaller model with BootstrapFinetune
# 3. Deploy fine-tuned model (zero-shot, lower cost)
```

### 8. Handle Module Freezing

```python
# After optimization, modules are frozen
assert optimized_program._compiled == True

# Don't modify optimized modules manually
# optimized_program.predict.demos = []  # ❌ Don't do this

# Instead, create a new optimization run if needed
```

**Source:** https://dspy.ai/faqs

---

## Summary Table

| Aspect | Unoptimized | Optimized |
|--------|-------------|-----------|
| **Demonstrations** | None (zero-shot) | 2-16 examples typically |
| **Instructions** | Generic/default | Task-specific, refined |
| **Performance** | Baseline (e.g., 45%) | Improved (e.g., 68%) |
| **Token usage** | Low (~80 tokens) | Higher (~330 tokens) |
| **Cold start** | Instant | Requires loading (~100-500ms) |
| **Memory** | ~1-5 KB | ~50-500 KB |
| **Consistency** | Variable | More consistent |
| **Customization** | Generic | Task-adapted |
| **Cost per call** | Lower | Higher (but better quality) |
| **Setup required** | None | Training data + optimization run |

---

## Key Takeaways

1. **Unoptimized modules** are zero-shot with generic prompts - good for starting, testing architecture
2. **Optimized modules** have learned prompts, demos, or weights - better for production
3. **Optimization changes** instructions, adds few-shot examples, or fine-tunes weights
4. **Compilation process** uses optimizers (teleprompters) to improve module performance
5. **Saving options**: State-only (JSON, recommended) or full program (pickle, less secure)
6. **Loading requires** matching LM configuration and program architecture
7. **Inspect with** `dspy.inspect_history()` to see actual prompts sent to LM
8. **Trade-off**: Optimized modules cost more tokens but deliver better quality
9. **Choose optimizer** based on dataset size and optimization goals
10. **Always measure** improvement: baseline score → optimized score

---

## Additional Resources

- **Optimizers Guide**: https://dspy.ai/learn/optimization/optimizers
- **Saving/Loading Tutorial**: https://dspy.ai/tutorials/saving
- **MIPROv2 API**: https://dspy.ai/api/optimizers/MIPROv2
- **BootstrapFewShot API**: https://dspy.ai/api/optimizers/BootstrapFewShot
- **Module API**: https://dspy.ai/api/modules/Module

---

**Document Version:** 1.0
**Last Updated:** 2024-12-19
**Sources:** DSPy Official Documentation (dspy.ai) via Context7
