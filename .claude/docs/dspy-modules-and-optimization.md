# DSPy Modules & Optimization - Comprehensive Guide

> A practical reference for understanding DSPy modules, optimization, and production usage.

## Table of Contents

1. [TL;DR (30 seconds)](#tldr-30-seconds)
2. [Part 1: Module Fundamentals](#part-1-module-fundamentals)
3. [Part 2: Built-in Module Types](#part-2-built-in-module-types)
4. [Part 3: Module State & Parameters](#part-3-module-state--parameters)
5. [Part 4: Optimization Deep Dive](#part-4-optimization-deep-dive)
6. [Part 5: Before vs After Optimization](#part-5-before-vs-after-optimization)
7. [Part 6: Production Usage](#part-6-production-usage)
8. [Our Codebase Patterns](#our-codebase-patterns)
9. [Quick Reference](#quick-reference)

---

## TL;DR (30 seconds)

| Concept | What It Is |
|---------|-----------|
| **dspy.Module** | A stateful container for LLM calls that can be optimized |
| **Signature** | Declares *what* you want (inputs → outputs) |
| **Predictor** | A module that makes LLM calls (Predict, ChainOfThought, etc.) |
| **Unoptimized** | Zero-shot, generic prompts, baseline performance |
| **Optimized** | Has learned demos/instructions, 20-50% better performance |
| **Optimizer** | Compiles modules to find best prompts/demos (BootstrapFewShot, MIPROv2) |

```python
# The core pattern
import dspy

# 1. Define what you want (signature)
class QA(dspy.Module):
    def __init__(self):
        self.predict = dspy.ChainOfThought("question -> answer")

    def forward(self, question):
        return self.predict(question=question)

# 2. Use unoptimized (baseline)
qa = QA()
result = qa(question="What is Python?")  # Zero-shot

# 3. Optimize it
optimizer = dspy.MIPROv2(metric=my_metric, auto="light")
optimized_qa = optimizer.compile(qa, trainset=examples)

# 4. Use optimized (better performance)
result = optimized_qa(question="What is Python?")  # Few-shot with learned demos

# 5. Save for production
optimized_qa.save("qa_optimized.json")
```

---

## Part 1: Module Fundamentals

### What is dspy.Module?

`dspy.Module` is the base class for all DSPy programs. It differs from regular Python classes in that:

1. **Stateful**: Stores demos, instructions, and optimization state
2. **Optimizable**: Can be compiled by optimizers to improve performance
3. **Composable**: Modules can contain other modules
4. **Traceable**: All LLM calls are tracked for optimization

### Module Lifecycle

```python
import dspy

class MyModule(dspy.Module):
    def __init__(self):
        """
        1. INIT: Define sub-modules and attributes
        - Declare predictors (Predict, ChainOfThought, etc.)
        - Set up any configuration
        """
        self.step1 = dspy.ChainOfThought("question -> reasoning")
        self.step2 = dspy.Predict("reasoning -> answer")

    def forward(self, question):
        """
        2. FORWARD: Define the computation flow
        - Call sub-modules
        - Use any Python logic (loops, conditionals, etc.)
        - Can call external functions (APIs, databases, etc.)
        """
        reasoning = self.step1(question=question)
        return self.step2(reasoning=reasoning.reasoning)

# 3. INVOCATION: Always use __call__, not forward() directly
module = MyModule()
result = module(question="What is Python?")  # Correct
# result = module.forward(question="...")    # Wrong - bypasses internal processing
```

**Why use `__call__` instead of `forward()`?**

The `__call__` method handles:
- Callback tracking
- Usage tracking (tokens, costs)
- Context management
- Internal state updates

### Module Composition

Modules can nest arbitrarily deep:

```python
class InnerModule(dspy.Module):
    def __init__(self):
        self.predict = dspy.ChainOfThought("context, question -> answer")

    def forward(self, context, question):
        return self.predict(context=context, question=question)

class OuterModule(dspy.Module):
    def __init__(self):
        self.retriever = dspy.Predict("question -> query")
        self.answerer = InnerModule()  # Nested module

    def forward(self, question):
        query = self.retriever(question=question).query
        context = search(query)  # External function
        return self.answerer(context=context, question=question)
```

### Async Modules

For async operations, implement `aforward()` instead of `forward()`:

```python
class AsyncModule(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("question -> answer")
        self.predict2 = dspy.ChainOfThought("answer -> simplified")

    async def aforward(self, question):
        answer = await self.predict1.acall(question=question)
        return await self.predict2.acall(answer=answer.answer)

# Usage
result = await module.acall(question="...")
```

---

## Part 2: Built-in Module Types

### dspy.Predict

The fundamental predictor - makes a single LLM call.

```python
# Basic usage
predict = dspy.Predict("question -> answer")
result = predict(question="What is 2+2?")
print(result.answer)

# With signature class
class QASignature(dspy.Signature):
    """Answer questions accurately."""
    question = dspy.InputField()
    answer = dspy.OutputField()

predict = dspy.Predict(QASignature)
```

### dspy.ChainOfThought

Extends Predict with explicit reasoning. Adds a `reasoning` field automatically.

```python
cot = dspy.ChainOfThought("question -> answer")
result = cot(question="What is 15% of 80?")
print(result.reasoning)  # "To find 15% of 80, I multiply 80 by 0.15..."
print(result.answer)     # "12"
```

**When to use**: Complex reasoning tasks, math, multi-step problems.

### dspy.ReAct

Reasoning + Acting with tools. Iteratively reasons and calls tools.

```python
def search(query: str) -> str:
    """Search the web for information."""
    return f"Results for: {query}"

def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    return str(eval(expression))

react = dspy.ReAct(
    signature="question -> answer",
    tools=[search, calculator],
    max_iters=5
)

result = react(question="What is the population of France times 2?")
print(result.trajectory)  # Shows thought/action/observation steps
print(result.answer)
```

**When to use**: Tasks requiring external tool use, information gathering.

### dspy.ProgramOfThought

Generates and executes code to solve problems.

```python
pot = dspy.ProgramOfThought("question -> answer", max_iters=3)
result = pot(question="What is the sum of the first 100 prime numbers?")
```

**When to use**: Math problems, data processing, anything that benefits from code execution.

### Module Comparison

| Module | Reasoning | Tools | Code Exec | Best For |
|--------|-----------|-------|-----------|----------|
| `Predict` | No | No | No | Simple Q&A, classification |
| `ChainOfThought` | Yes | No | No | Complex reasoning, multi-step |
| `ReAct` | Yes | Yes | No | Tool use, information gathering |
| `ProgramOfThought` | Yes | No | Yes | Math, computation, data processing |

---

## Part 3: Module State & Parameters

### What's Stored in Module State

Each predictor stores:

```python
predictor = dspy.ChainOfThought("question -> answer")

# State components
predictor.demos      # List of demonstration examples
predictor.traces     # Execution traces (for debugging)
predictor.train      # Training history
predictor.lm         # Language model reference
predictor.signature  # The signature with instructions
```

### Inspecting Module State

```python
# Get all predictors in a module
for name, pred in module.named_predictors():
    print(f"Predictor: {name}")
    print(f"  Demos: {len(pred.demos)}")
    print(f"  Instructions: {pred.signature.instructions}")

# Dump full state (for debugging)
state = module.dump_state(json_mode=True)
print(state)
```

### named_predictors() vs predictors()

```python
# named_predictors() - returns (name, predictor) tuples
for name, pred in module.named_predictors():
    print(f"{name}: {type(pred)}")
# Output: "predict: <class 'dspy.predict.predict.Predict'>"

# predictors() - returns just predictor objects
for pred in module.predictors():
    print(type(pred))
```

### Module Parameters vs Regular Attributes

```python
class MyModule(dspy.Module):
    def __init__(self):
        # PARAMETERS - tracked by DSPy, can be optimized
        self.qa = dspy.ChainOfThought("question -> answer")  # Predictor

        # REGULAR ATTRIBUTES - not tracked
        self.max_retries = 3  # Regular Python attribute
        self.cache = {}       # Regular Python attribute
```

**Rule**: Only `dspy.Module` subclasses (predictors) are tracked as parameters.

### The _compiled Flag

After optimization, modules are marked as compiled:

```python
# Before optimization
print(module._compiled)  # False (or not set)

# After optimization
optimized = optimizer.compile(module, trainset=data)
print(optimized._compiled)  # True

# Compiled modules are "frozen" - don't modify manually
```

---

## Part 4: Optimization Deep Dive

### The Compilation Process

```python
# 1. Create unoptimized module
module = dspy.ChainOfThought("question -> answer")

# 2. Define a metric
def exact_match(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# 3. Create optimizer
optimizer = dspy.MIPROv2(
    metric=exact_match,
    auto="light"  # "light", "medium", or "heavy"
)

# 4. Compile
optimized = optimizer.compile(
    module,
    trainset=train_examples,  # Required
    valset=val_examples       # Optional
)
```

### Optimizer Categories

#### Category 1: Few-Shot Learning

Add demonstrations without changing instructions.

| Optimizer | How It Works | When to Use |
|-----------|--------------|-------------|
| `LabeledFewShot(k=N)` | Uses N labeled examples directly | Small datasets (5-20 examples) |
| `BootstrapFewShot` | Synthesizes demos from successful traces | Medium datasets (20-100) |
| `BootstrapFewShotWithRandomSearch` | Explores different demo configurations | Larger datasets (100+) |
| `KNNFewShot` | Dynamically selects similar demos at inference | When examples vary widely |

```python
# LabeledFewShot - simplest
optimizer = dspy.LabeledFewShot(k=5)

# BootstrapFewShot - most common
optimizer = dspy.BootstrapFewShot(
    metric=my_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)
```

#### Category 2: Prompt Optimization

Refines both instructions AND demonstrations.

| Optimizer | How It Works | When to Use |
|-----------|--------------|-------------|
| `MIPROv2` | Multi-stage instruction + prompt optimization | Best overall performance |
| `COPRO` | Coordinate descent on prompts | Alternative to MIPROv2 |

```python
# MIPROv2 - recommended for most cases
optimizer = dspy.MIPROv2(
    metric=my_metric,
    auto="light",      # Optimization intensity
    num_threads=4      # Parallelism
)
```

#### Category 3: Weight Optimization

Fine-tunes the actual model weights.

```python
optimizer = dspy.BootstrapFinetune(
    metric=my_metric,
    num_threads=24
)

finetuned = optimizer.compile(
    module,
    trainset=train_data,
    epochs=2,
    lr=5e-5
)
```

### Writing Good Metrics

```python
def my_metric(example, pred, trace=None):
    """
    Args:
        example: The ground truth example (has .answer, .question, etc.)
        pred: The prediction (has .answer, .reasoning, etc.)
        trace: Optional trace for debugging

    Returns:
        float or bool: Score (True/False or 0.0-1.0)
    """
    # Exact match
    if example.answer.lower() == pred.answer.lower():
        return 1.0

    # Partial credit
    if example.answer.lower() in pred.answer.lower():
        return 0.5

    return 0.0

# Using LLM-based evaluation
def semantic_match(example, pred, trace=None):
    judge = dspy.ChainOfThought("reference, prediction -> is_correct: bool")
    result = judge(reference=example.answer, prediction=pred.answer)
    return result.is_correct
```

### Choosing the Right Optimizer

| Dataset Size | Recommended Approach |
|--------------|---------------------|
| 5-20 examples | `LabeledFewShot(k=5)` |
| 20-100 examples | `BootstrapFewShot` |
| 100-500 examples | `BootstrapFewShotWithRandomSearch` |
| 500+ examples | `MIPROv2(auto="light")` |
| Need best performance | `MIPROv2(auto="medium")` |

---

## Part 5: Before vs After Optimization

### Prompt Comparison

**Before (Unoptimized - Zero-shot)**:
```
System: Answer questions with short factual answers.

User: What is the capital of France?
```
~50 tokens

**After (Optimized - Few-shot)**:
```
System: Given a question, provide a concise, accurate answer
based on factual information. Focus on clarity and precision.

---
Example 1:
Question: What is Python?
Reasoning: Python is a widely-used programming language.
Answer: A high-level programming language.

Example 2:
Question: What is JavaScript?
Reasoning: JavaScript is used for web development.
Answer: A scripting language for web browsers.
---

User: What is the capital of France?
```
~300 tokens

### Inspecting Actual Prompts

```python
# After running the module
result = optimized_module(question="What is Python?")

# Inspect the prompt that was sent
dspy.inspect_history(n=1)

# Inspect signature changes
print(optimized_module.predict.signature.instructions)

# Count demos
print(f"Demos: {len(optimized_module.predict.demos)}")
```

### Performance Impact

| Aspect | Unoptimized | Optimized |
|--------|-------------|-----------|
| **Accuracy** | Baseline (e.g., 45%) | +20-50% (e.g., 65-70%) |
| **Token usage** | ~80 tokens/call | ~330 tokens/call |
| **Latency** | Lower | Slightly higher |
| **Consistency** | More variable | More consistent |
| **Cost** | Lower | ~4x higher per call |

---

## Part 6: Production Usage

### Saving Optimized Modules

```python
# Option 1: State-only (RECOMMENDED)
# Saves demos and instructions to JSON
optimized_module.save("module_v1.json")

# Option 2: Full program with architecture
# Uses pickle - less secure
optimized_module.save("module_dir/", save_program=True)
```

### Loading Optimized Modules

```python
# Must match the original architecture
import dspy

# 1. Configure same LM
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# 2. Create same module structure
module = MyModule()

# 3. Load optimized state
module.load("module_v1.json")

# 4. Use it
result = module(question="...")
```

### Versioning Best Practices

```python
import json
from datetime import datetime

# Save with metadata
optimized.save("models/qa_v1.2.json")

metadata = {
    "version": "1.2",
    "optimizer": "MIPROv2",
    "base_score": 0.45,
    "optimized_score": 0.68,
    "train_size": len(trainset),
    "created": datetime.now().isoformat()
}

with open("models/qa_v1.2_meta.json", "w") as f:
    json.dump(metadata, f)
```

### Security Considerations

```python
# SAFE: JSON format (recommended)
module.save("module.json", save_program=False)

# RISKY: Pickle format (can execute arbitrary code)
# Only use if you trust the source
module.save("module_dir/", save_program=True)
```

---

## Our Codebase Patterns

### DSPy Tracking Utilities

We have custom tracking in [src/utils/dspy_tracking.py](src/utils/dspy_tracking.py):

```python
from src.utils.dspy_tracking import (
    extract_lm_history,      # Get structured prompt history
    extract_token_usage,     # Get token counts
    set_mlflow_span_tracking # Track in MLflow
)

# After DSPy prediction
history = extract_lm_history(lm, logger)
tokens = extract_token_usage(prediction, logger)
set_mlflow_span_tracking(span, history, tokens, output, fallback_inputs)
```

### Where DSPy is Used

- **Composition Analysis**: [src/agent/composition_analysis/](src/agent/composition_analysis/)
- **Search Tools**: [src/tools/graphdb/search/](src/tools/graphdb/search/)

### MLflow Integration

DSPy autolog is enabled at startup:

```python
# In main.py:lifespan()
import dspy
dspy.configure(experimental_track_usage=True)

# All predictions are traced to MLflow
# View at: ./mlruns (or MLFLOW_TRACKING_URI)
```

---

## Quick Reference

### Module Creation Checklist

```python
class MyModule(dspy.Module):
    def __init__(self):
        # 1. Always call super().__init__() if overriding
        super().__init__()

        # 2. Define predictors as attributes
        self.step1 = dspy.ChainOfThought("input -> intermediate")
        self.step2 = dspy.Predict("intermediate -> output")

    def forward(self, **kwargs):
        # 3. Implement forward logic
        # 4. Return dspy.Prediction or predictor output
        pass
```

### Optimization Checklist

1. [ ] Create module with clear signatures
2. [ ] Prepare trainset as list of `dspy.Example`
3. [ ] Define metric function `(example, pred, trace) -> score`
4. [ ] Choose optimizer based on dataset size
5. [ ] Run `optimizer.compile(module, trainset=...)`
6. [ ] Evaluate: `evaluate(optimized, devset=testset)`
7. [ ] Inspect prompts: `dspy.inspect_history(n=1)`
8. [ ] Save: `optimized.save("module.json")`

### Common Gotchas

1. **Call `module()` not `module.forward()`** - Forward bypasses tracking
2. **Match LM on load** - Must use same LM as during optimization
3. **Match architecture on load** - Module structure must be identical
4. **Don't modify compiled modules** - They're frozen after optimization
5. **Token costs increase ~4x** - Few-shot examples add tokens

---

## Further Reading

- [DSPy Official Docs](https://dspy.ai)
- [Optimizers Guide](https://dspy.ai/learn/optimization/optimizers)
- [Module API](https://dspy.ai/api/modules/Module)
- [Tutorials](https://dspy.ai/tutorials)

---

**Document Version**: 2.0
**Last Updated**: 2024-12-20
**Sources**: DSPy Official Documentation (dspy.ai)
