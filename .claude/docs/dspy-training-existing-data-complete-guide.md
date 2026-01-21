# DSPy Training on Existing Data - Complete Guide

> Comprehensive guide on loading, preparing, and using existing datasets for training and optimization in DSPy
>
> Sources: DSPy Official Documentation (dspy.ai), Context7 Documentation

## Table of Contents

1. [Overview](#overview)
2. [Creating DSPy Examples from Existing Data](#creating-dspy-examples-from-existing-data)
3. [Loading Datasets](#loading-datasets)
4. [Structuring Training Data](#structuring-training-data)
5. [Understanding Optimizers](#understanding-optimizers)
6. [Working with Metrics](#working-with-metrics)
7. [Data Splitting Best Practices](#data-splitting-best-practices)
8. [Optimizer Selection Guide](#optimizer-selection-guide)
9. [Complete End-to-End Examples](#complete-end-to-end-examples)
10. [Best Practices and Tips](#best-practices-and-tips)

---

## Overview

DSPy optimizers (also called "teleprompters") learn from existing datasets to improve your programs by:
- Selecting and ordering few-shot demonstrations
- Optimizing instruction prompts
- Fine-tuning model weights
- Generating synthetic demonstrations from successful traces

The key to successful optimization is properly preparing your existing data as DSPy `Example` objects with clearly defined inputs and outputs.

---

## Creating DSPy Examples from Existing Data

### The `dspy.Example` Class

DSPy uses the `Example` class to represent individual data points. Each example is a structured object with named fields.

**Source:** https://dspy.ai/api/primitives/Example

#### Basic Example Creation

```python
import dspy

# Create a simple QA example
example = dspy.Example(
    question="What is the capital of France?",
    answer="Paris"
)

# Access fields using dot notation
print(example.question)  # "What is the capital of France?"
print(example.answer)    # "Paris"
```

#### Marking Input Fields with `with_inputs()`

**Critical:** You must specify which fields are inputs vs outputs using `with_inputs()`:

```python
# Single input field
example = dspy.Example(
    question="What is machine learning?",
    answer="A subset of AI that enables systems to learn from data"
).with_inputs("question")

# Multiple input fields (e.g., for RAG with context)
example = dspy.Example(
    context="The sky is blue and clear.",
    question="What is the weather?",
    answer="It's sunny"
).with_inputs("context", "question")
```

**Why this matters:** The `with_inputs()` method creates a copy of the example and sets `_input_keys` to distinguish what should be provided to the model vs what should be predicted.

**Source:** https://dspy.ai/learn/evaluation/data

#### Separating Inputs and Labels

```python
example = dspy.Example(
    question="What is the weather?",
    answer="It's sunny",
).with_inputs("question")

# Get only input fields
inputs = example.inputs()
print(inputs.question)  # "What is the weather?"
# print(inputs.answer)  # This would raise an error

# Get only output/label fields
labels = example.labels()
print(labels.answer)  # "It's sunny"
# print(labels.question)  # This would raise an error
```

**Source:** https://dspy.api/primitives/Example

#### Converting to/from Dictionaries

```python
# Convert Example to dictionary (for JSON serialization)
example_dict = example.toDict()
# Returns: {'question': '...', 'answer': '...'}

# Create Example from dictionary
example = dspy.Example(**example_dict).with_inputs("question")
```

**Source:** https://dspy.ai/api/primitives/Example

---

## Loading Datasets

DSPy provides multiple ways to load existing datasets.

### Method 1: Using DSPy's DataLoader (Hugging Face)

The `DataLoader` class can load datasets directly from Hugging Face Hub:

```python
import dspy
from dspy.datasets import DataLoader

# Load from Hugging Face with field mapping
kwargs = dict(
    fields=("text", "label"),      # Which fields to extract
    input_keys=("text",),           # Which fields are inputs
    split="train",                  # Which split to load
    trust_remote_code=True
)

# Load dataset
dataset = DataLoader().from_huggingface(
    dataset_name="PolyAI/banking77",
    **kwargs
)

# dataset is now a list of dspy.Example objects
print(len(dataset))
print(dataset[0])
```

**Source:** https://dspy.ai/tutorials/classification_finetuning

#### Example: Loading Banking77 Dataset

```python
import dspy
import random
from dspy.datasets import DataLoader
from datasets import load_dataset

# Get class labels
CLASSES = load_dataset(
    "PolyAI/banking77",
    split="train",
    trust_remote_code=True
).features['label'].names

# Load and prepare examples
kwargs = dict(
    fields=("text", "label"),
    input_keys=("text",),
    split="train",
    trust_remote_code=True
)

raw_data = [
    dspy.Example(x, label=CLASSES[x.label]).with_inputs("text")
    for x in DataLoader().from_huggingface(
        dataset_name="PolyAI/banking77",
        **kwargs
    )[:1000]
]

random.Random(0).shuffle(raw_data)
```

**Source:** https://dspy.ai/tutorials/classification_finetuning

### Method 2: Using HuggingFace datasets Library

```python
from datasets import load_dataset
import dspy

# Load dataset
dataset = load_dataset("conll2003", trust_remote_code=True)

# Convert to DSPy Examples
def prepare_dataset(data_split, start, end):
    return [
        dspy.Example(
            tokens=row["tokens"],
            expected_entities=extract_entities(row)
        ).with_inputs("tokens")
        for row in data_split.select(range(start, end))
    ]

train_set = prepare_dataset(dataset["train"], 0, 50)
test_set = prepare_dataset(dataset["test"], 0, 200)
```

**Source:** https://dspy.ai/tutorials/entity_extraction

### Method 3: Loading from JSON/JSONL

```python
import ujson
import dspy
from dspy.utils import download

# Download and load JSONL file
download("https://example.com/data.jsonl")

with open("data.jsonl") as f:
    data = [ujson.loads(line) for line in f]

# Convert to DSPy Examples
trainset = [
    dspy.Example(**d).with_inputs('question')
    for d in data
]
```

**Source:** https://dspy.ai/tutorials/rag

### Method 4: Loading from JSON (Remote URL)

```python
import requests
import dspy
import json
import random

def init_dataset():
    # Load from URL
    url = "https://example.com/dataset.json"
    dataset = json.loads(requests.get(url).text)

    # Convert to DSPy Examples
    dspy_dataset = [
        dspy.Example({
            "message": d['fields']['input'],
            "answer": d['answer'],
        }).with_inputs("message")
        for d in dataset
    ]

    random.Random(0).shuffle(dspy_dataset)
    return dspy_dataset
```

**Source:** https://dspy.ai/tutorials/gepa_facilitysupportanalyzer

### Method 5: Using Built-in DSPy Datasets

DSPy provides several built-in datasets:

```python
from dspy.datasets import HotPotQA, GSM8K, MATH

# HotPotQA
hotpot = HotPotQA(train_seed=2024, train_size=500)
trainset = [x.with_inputs('question') for x in hotpot.train]

# GSM8K
gsm8k = GSM8K()
trainset = gsm8k.train[:10]

# MATH dataset
math_dataset = MATH(subset='algebra')
print(len(math_dataset.train), len(math_dataset.dev))
```

**Source:** https://dspy.ai/tutorials/math, https://dspy.ai/learn/optimization/optimizers

### Method 6: Loading from CSV/Pandas

```python
import pandas as pd
import dspy

# Load CSV
df = pd.read_csv("data.csv")

# Convert to DSPy Examples
trainset = [
    dspy.Example(
        question=row['question'],
        answer=row['answer']
    ).with_inputs("question")
    for _, row in df.iterrows()
]
```

---

## Structuring Training Data

### Understanding Input vs Output Fields

Your training data must match your program's signature. The signature defines the expected input and output fields.

#### Matching Examples to Signatures

```python
# Signature: "question -> answer"
# Your examples must have 'question' input and 'answer' output

trainset = [
    dspy.Example(
        question="What is Python?",
        answer="A programming language"
    ).with_inputs("question")
]

# Signature: "context, question -> response"
# Your examples must have both 'context' and 'question' as inputs

trainset = [
    dspy.Example(
        context="Python is a high-level programming language.",
        question="What is Python?",
        response="Python is a high-level programming language"
    ).with_inputs("context", "question")
]
```

**Key Rule:** Field names in your examples should match the field names in your signature.

**Source:** https://dspy.ai/learn/programming/signatures

### Common Data Structures

#### 1. Question Answering

```python
trainset = [
    dspy.Example(
        question="What is the capital of France?",
        answer="Paris"
    ).with_inputs("question"),
    dspy.Example(
        question="What is 2+2?",
        answer="4"
    ).with_inputs("question"),
]
```

#### 2. Classification

```python
trainset = [
    dspy.Example(
        text="I love this product!",
        label="positive"
    ).with_inputs("text"),
    dspy.Example(
        text="This is terrible",
        label="negative"
    ).with_inputs("text"),
]
```

#### 3. Retrieval-Augmented Generation (RAG)

```python
trainset = [
    dspy.Example(
        context=["Paris is the capital of France.", "France is in Europe."],
        question="What is the capital of France?",
        answer="Paris"
    ).with_inputs("context", "question"),
]
```

#### 4. Multi-Field Classification

```python
trainset = [
    dspy.Example(
        question="What is the sentiment?",
        choices=["positive", "negative", "neutral"],
        reasoning="The text expresses happiness",
        selection=0
    ).with_inputs("question", "choices"),
]
```

**Source:** https://dspy.ai/learn/programming/signatures

### Working with Complex Types

DSPy supports complex types in examples:

```python
from typing import List, Literal

# List inputs/outputs
example = dspy.Example(
    tokens=["The", "cat", "sat"],
    extracted_entities=["cat"]
).with_inputs("tokens")

# Literal types for constrained outputs
example = dspy.Example(
    text="Great product!",
    sentiment=Literal["positive", "negative", "neutral"]("positive")
).with_inputs("text")

# Images (multimodal)
example = dspy.Example(
    image=dspy.Image(url="https://..."),
    description="A cat"
).with_inputs("image")
```

**Source:** https://dspy.ai/learn/programming/signatures

### Unlabeled Data

For bootstrapping without labels, create examples with only inputs:

```python
# Unlabeled examples (for BootstrapFewShot to generate labels)
unlabeled_trainset = [
    dspy.Example(text=x.text).with_inputs("text")
    for x in raw_data[:500]
]
```

**Source:** https://dspy.ai/tutorials/classification_finetuning

---

## Understanding Optimizers

DSPy optimizers take your base program and training data, then return an optimized version. There are several types of optimizers, each suited for different scenarios.

### Optimizer Categories

#### 1. Few-Shot Learning Optimizers

These add example demonstrations to your prompts.

##### `LabeledFewShot` - Simplest Optimizer

Uses your labeled training examples directly as demonstrations.

```python
from dspy.teleprompt import LabeledFewShot
import dspy

# Prepare training data
trainset = [
    dspy.Example(
        text="Cancel my subscription",
        label="account_management"
    ).with_inputs("text"),
    dspy.Example(
        text="When will my order arrive?",
        label="shipping_inquiry"
    ).with_inputs("text"),
    # ... more examples
]

# Create base program
program = dspy.ChainOfThought("text -> label")

# Compile with LabeledFewShot (k=3 means use up to 3 demos)
optimizer = LabeledFewShot(k=3)
compiled_program = optimizer.compile(
    student=program,
    trainset=trainset
)
```

**Parameters:**
- `k`: Number of examples to use as demonstrations
- `trainset`: Your labeled training examples

**Use when:** You have 5-20 labeled examples and want the simplest approach.

**Source:** https://dspy.ai/api/optimizers/LabeledFewShot

##### `BootstrapFewShot` - Intelligent Demo Selection

Runs your program on training examples, keeps successful traces, and uses them as demonstrations.

```python
from dspy.teleprompt import BootstrapFewShot
import dspy

def validation_metric(example, pred, trace=None):
    """Returns True/False for bootstrapping, float for evaluation."""
    answer_match = example.answer.lower() == pred.answer.lower()

    if trace is None:  # Evaluation mode
        return float(answer_match)
    else:  # Bootstrapping mode (only keep if True)
        return answer_match

# Configure optimizer
optimizer = BootstrapFewShot(
    metric=validation_metric,
    max_bootstrapped_demos=4,  # Generate up to 4 examples
    max_labeled_demos=16,      # Include up to 16 labeled examples
    max_rounds=1,              # Number of bootstrap rounds
    max_errors=10              # Error tolerance
)

# Compile
optimized_program = optimizer.compile(
    student=program,
    trainset=trainset
)
```

**How it works:**
1. Runs the program on your training examples
2. For each example where `metric(example, prediction, trace)` returns `True`, saves the trace
3. Uses successful traces as few-shot demonstrations
4. Combines with labeled examples from trainset

**Parameters:**
- `metric`: Function that returns True/False (for bootstrapping) or float (for evaluation)
- `max_bootstrapped_demos`: How many generated demos to keep
- `max_labeled_demos`: How many labeled examples to include
- `max_rounds`: Number of times to run bootstrapping
- `max_errors`: How many failures to tolerate before stopping

**Use when:** You have 10-50 labeled examples and want better quality demonstrations.

**Source:** https://dspy.ai/api/optimizers/BootstrapFewShot

##### `BootstrapFewShotWithRandomSearch` - Demo Selection with Search

Runs BootstrapFewShot multiple times with different random seeds and selects the best combination.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

config = dict(
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=10,  # Try 10 different configurations
    num_threads=4
)

optimizer = BootstrapFewShotWithRandomSearch(
    metric=validation_metric,
    **config
)

optimized_program = optimizer.compile(
    student=program,
    trainset=trainset,
    valset=devset  # Validation set for selecting best program
)
```

**Parameters:**
- All `BootstrapFewShot` parameters
- `num_candidate_programs`: How many random configurations to try
- `valset`: Validation set for selecting the best candidate

**Use when:** You have 50+ examples and want to explore different demo configurations.

**Source:** https://dspy.ai/api/optimizers/BootstrapFewShotWithRandomSearch

##### `KNNFewShot` - Semantic Similarity-Based Selection

Selects demonstrations based on semantic similarity to the input.

```python
from dspy.teleprompt import KNNFewShot
from sentence_transformers import SentenceTransformer
import dspy

trainset = [
    dspy.Example(
        question="What is Python?",
        answer="A programming language"
    ).with_inputs("question"),
    # ... more examples
]

# Initialize with embedder
knn_optimizer = KNNFewShot(
    k=3,  # Select 3 most similar examples
    trainset=trainset,
    vectorizer=dspy.Embedder(
        SentenceTransformer("all-MiniLM-L6-v2").encode
    )
)

# Compile
compiled_qa = knn_optimizer.compile(program)

# At inference, automatically selects 3 most relevant examples
result = compiled_qa(question="What is JavaScript?")
```

**How it works:**
1. Embeds all training examples
2. At inference time, embeds the input
3. Finds k nearest neighbors from training set
4. Uses them as demonstrations

**Use when:** You want dynamic example selection based on input similarity.

**Source:** https://dspy.ai/api/optimizers/KNNFewShot

#### 2. Prompt Optimization

##### `MIPROv2` - Multi-Stage Instruction and Prompt Optimization

The most sophisticated optimizer, optimizing both instructions and demonstrations.

```python
from dspy.teleprompt import MIPROv2
import dspy

# Initialize optimizer
optimizer = MIPROv2(
    metric=validation_metric,
    auto="light",  # Options: "light", "medium", "heavy"
    num_threads=24
)

# Compile
optimized_program = optimizer.compile(
    program.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=4,
    max_labeled_demos=4
)

# Save
optimized_program.save("optimized_program.json")
```

**Auto modes:**
- `"light"`: ~10-20 minutes, good for quick iteration (~$1-2 with GPT-3.5)
- `"medium"`: ~30-60 minutes, balanced quality (~$3-5)
- `"heavy"`: 1+ hours, maximum quality (~$10-15)

**Advanced configuration:**

```python
# With custom teacher model
kwargs = dict(
    num_threads=24,
    teacher_settings=dict(lm=dspy.LM('openai/gpt-4o')),
    prompt_model=dspy.LM('openai/gpt-4o-mini')
)

optimizer = MIPROv2(
    metric=validation_metric,
    auto="medium",
    **kwargs
)
```

**Use when:**
- You have 200+ training examples
- You want the best possible performance
- You can afford 20-60 minutes of optimization time

**Source:** https://dspy.ai/api/optimizers/MIPROv2

#### 3. Weight Optimization (Fine-tuning)

##### `BootstrapFinetune` - Model Fine-tuning

Fine-tunes a smaller model using traces from a larger teacher model.

```python
from dspy.teleprompt import BootstrapFinetune
import dspy

# Configure fine-tuning
config = dict(
    epochs=2,
    bf16=True,
    bsize=6,
    accumsteps=2,
    lr=5e-5
)

# Initialize optimizer
optimizer = BootstrapFinetune(
    metric=validation_metric,
    num_threads=24
)

# Compile (this will finetune the model)
finetuned_program = optimizer.compile(
    student=program,
    trainset=trainset,
    **config
)
```

**Use when:**
- You need a cost-efficient production model
- You have a large LM (7B+ parameters) available for fine-tuning
- You have substantial training data (100+ examples)

**Source:** https://dspy.ai/tutorials/classification_finetuning

---

## Working with Metrics

Metrics evaluate your program's performance and guide optimization. They have two modes:

1. **Evaluation mode** (`trace=None`): Returns a float score (0.0 to 1.0)
2. **Bootstrapping mode** (`trace!=None`): Returns True/False to keep/discard traces

### Simple Metric: Exact Match

```python
def validate_answer(example, pred, trace=None):
    """Simple exact match metric."""
    return example.answer.lower() == pred.answer.lower()

# Returns True/False in both modes
```

**Source:** https://dspy.ai/learn/evaluation/metrics

### Metric with Different Evaluation/Bootstrapping Logic

```python
def validate_context_and_answer(example, pred, trace=None):
    """More sophisticated metric with different behavior."""
    # Check answer matches
    answer_match = example.answer.lower() == pred.answer.lower()

    # Check answer comes from context
    context_match = any(
        (pred.answer.lower() in c)
        for c in pred.context
    )

    if trace is None:  # Evaluation mode
        # Return average score
        return (answer_match + context_match) / 2.0
    else:  # Bootstrapping mode
        # Only keep if both conditions are true
        return answer_match and context_match
```

**Source:** https://dspy.ai/learn/evaluation/metrics

### Trace-Based Metric (Validating Intermediate Steps)

```python
def validate_hops(example, pred, trace=None):
    """Validates intermediate reasoning steps."""
    # Extract queries from trace
    hops = [example.question] + [
        outputs.query
        for *_, outputs in trace
        if 'query' in outputs
    ]

    # Check hop length
    if max([len(h) for h in hops]) > 100:
        return False

    # Check for redundant hops
    if any(
        dspy.evaluate.answer_exact_match_str(hops[idx], hops[:idx], frac=0.8)
        for idx in range(2, len(hops))
    ):
        return False

    return True
```

**Source:** https://dspy.ai/learn/evaluation/metrics

### Built-in Metrics

DSPy provides several built-in metrics:

```python
# Exact answer matching
from dspy.evaluate import answer_exact_match
metric = answer_exact_match

# Semantic F1 score
from dspy.evaluate import SemanticF1
metric = SemanticF1(decompositional=True)
```

**Source:** https://dspy.ai/tutorials/rag

### Using Metrics with Optimizers

```python
# With BootstrapFewShot
optimizer = BootstrapFewShot(metric=your_metric)

# With MIPROv2
optimizer = MIPROv2(metric=your_metric, auto="light")

# With evaluation
evaluate = dspy.Evaluate(
    devset=test_set,
    metric=your_metric,
    num_threads=16,
    display_progress=True,
    display_table=5
)

score = evaluate(optimized_program)
```

---

## Data Splitting Best Practices

### Recommended Splits for Prompt Optimization

For prompt-based optimizers (LabeledFewShot, BootstrapFewShot, MIPROv2), DSPy recommends an **unusual split** compared to deep learning:

**20% Training / 80% Validation**

```python
import random

# Load all data
random.Random(0).shuffle(data)

# Split: 20% train, 80% validation
split_point = int(len(data) * 0.2)
trainset = data[:split_point]
valset = data[split_point:]

# Use with optimizer
optimizer = MIPROv2(metric=metric, auto="light")
optimized = optimizer.compile(
    program,
    trainset=trainset,
    # MIPROv2 will use valset internally for selection
)
```

**Why?** Prompt optimizers can overfit to small training sets, so more validation data provides stable selection.

**Source:** https://dspy.ai/learn/optimization/overview

### Recommended Splits for Fine-tuning (GEPA)

For weight optimization (GEPA, BootstrapFinetune), use standard ML splits:

**Maximize training set, keep validation large enough**

```python
# Standard split: 60% train, 20% val, 20% test
train_end = int(len(data) * 0.6)
val_end = int(len(data) * 0.8)

trainset = data[:train_end]
valset = data[train_end:val_end]
testset = data[val_end:]
```

**Source:** https://dspy.ai/learn/optimization/overview

### Three-Way Split: Train, Dev, Test

For comprehensive evaluation:

```python
import random

random.Random(0).shuffle(data)

# Example: 200 train, 300 dev, 500 test
trainset = data[:200]
devset = data[200:500]
testset = data[500:1000]

# Use train+dev for optimization
optimizer.compile(program, trainset=trainset, valset=devset)

# Use testset for final held-out evaluation
final_score = evaluate(optimized_program, devset=testset)
```

**Purposes:**
- **Training**: Direct learning by optimizer
- **Dev/Validation**: Monitoring during optimization, hyperparameter selection
- **Test**: Final held-out evaluation (don't look at until the end!)

**Source:** https://dspy.ai/tutorials/rag

### Dataset Size Recommendations

| Training Examples | Recommended Approach |
|------------------|---------------------|
| 5-10 examples | `LabeledFewShot(k=5)` |
| 10-50 examples | `BootstrapFewShot` |
| 50-200 examples | `BootstrapFewShotWithRandomSearch` |
| 200+ examples | `MIPROv2` (auto="light" or "medium") |
| 100+ examples + 7B model | `BootstrapFinetune` |

**Source:** https://dspy.ai/learn/optimization/optimizers

---

## Optimizer Selection Guide

### Decision Tree

```
How many labeled examples do you have?
│
├─ 5-10 examples
│  └─ Use: LabeledFewShot(k=5)
│
├─ 10-50 examples
│  └─ Use: BootstrapFewShot(max_bootstrapped_demos=4, max_labeled_demos=8)
│
├─ 50-200 examples
│  └─ Use: BootstrapFewShotWithRandomSearch(num_candidate_programs=10)
│
└─ 200+ examples
   ├─ Want instruction optimization only (0-shot)?
   │  └─ Use: MIPROv2(auto="light", max_bootstrapped_demos=0, max_labeled_demos=0)
   │
   ├─ Want both instructions + demos?
   │  └─ Use: MIPROv2(auto="medium", max_bootstrapped_demos=4, max_labeled_demos=4)
   │
   └─ Want to finetune a small model?
      └─ Use: BootstrapFinetune (requires 7B+ model)
```

**Source:** https://dspy.ai/learn/optimization/optimizers

### Quick Reference Table

| Optimizer | Min Examples | Optimization Time | Cost (GPT-3.5) | Best For |
|-----------|-------------|-------------------|----------------|----------|
| `LabeledFewShot` | 5-10 | Seconds | Free | Quick start, simple use cases |
| `BootstrapFewShot` | 10-50 | 2-5 minutes | $0.10-0.50 | Quality demos from traces |
| `BootstrapFewShotWithRandomSearch` | 50-200 | 10-20 minutes | $0.50-2.00 | Finding best demo config |
| `KNNFewShot` | 20-100 | 1-2 minutes | $0.20-0.80 | Dynamic example selection |
| `MIPROv2` (light) | 200+ | 10-20 minutes | $1-2 | Fast prompt optimization |
| `MIPROv2` (medium) | 200+ | 30-60 minutes | $3-5 | Balanced optimization |
| `MIPROv2` (heavy) | 200+ | 1-3 hours | $10-15 | Maximum quality |
| `BootstrapFinetune` | 100+ | 30-120 minutes | Model-dependent | Production deployment |

---

## Complete End-to-End Examples

### Example 1: Simple QA with BootstrapFewShot

```python
import dspy
from dspy.teleprompt import BootstrapFewShot
import random

# 1. Configure DSPy
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

# 2. Load and prepare data
data = [
    {"question": "What is Python?", "answer": "A programming language"},
    {"question": "What is JavaScript?", "answer": "A scripting language for web"},
    {"question": "What is Java?", "answer": "An object-oriented language"},
    # ... add more examples (aim for 30-50)
]

# Convert to DSPy Examples
dataset = [
    dspy.Example(**d).with_inputs('question')
    for d in data
]

random.Random(0).shuffle(dataset)

# Split: 20% train, 80% validation
split_point = int(len(dataset) * 0.2)
trainset = dataset[:split_point]
devset = dataset[split_point:]

# 3. Define program
program = dspy.ChainOfThought("question -> answer")

# 4. Define metric
def qa_metric(example, pred, trace=None):
    answer_match = example.answer.lower() in pred.answer.lower()
    if trace is None:
        return float(answer_match)
    return answer_match

# 5. Optimize
optimizer = BootstrapFewShot(
    metric=qa_metric,
    max_bootstrapped_demos=3,
    max_labeled_demos=5
)

optimized_program = optimizer.compile(
    student=program,
    trainset=trainset
)

# 6. Evaluate
evaluate = dspy.Evaluate(
    devset=devset,
    metric=qa_metric,
    num_threads=4,
    display_progress=True
)

score = evaluate(optimized_program)
print(f"Score: {score}")

# 7. Save
optimized_program.save("qa_optimized.json")
```

### Example 2: Classification with MIPROv2

```python
import dspy
from dspy.teleprompt import MIPROv2
from dspy.datasets import DataLoader
import random

# 1. Configure
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

# 2. Load Banking77 dataset
from datasets import load_dataset

CLASSES = load_dataset(
    "PolyAI/banking77",
    split="train",
    trust_remote_code=True
).features['label'].names

kwargs = dict(
    fields=("text", "label"),
    input_keys=("text",),
    split="train",
    trust_remote_code=True
)

raw_data = [
    dspy.Example(x, label=CLASSES[x.label]).with_inputs("text")
    for x in DataLoader().from_huggingface(
        dataset_name="PolyAI/banking77",
        **kwargs
    )[:1000]
]

random.Random(0).shuffle(raw_data)

# Split: use first 300 for training/validation
trainset = raw_data[:300]
testset = raw_data[300:500]

# 3. Define signature with constrained output
from typing import Literal

signature = dspy.Signature("text -> label").with_updated_fields(
    'label',
    type_=Literal[tuple(CLASSES)]
)

# 4. Create program
classifier = dspy.ChainOfThought(signature)

# 5. Define metric
def classification_metric(example, pred, trace=None):
    return example.label == pred.label

# 6. Optimize with MIPROv2
optimizer = MIPROv2(
    metric=classification_metric,
    auto="light",  # 10-20 min optimization
    num_threads=16
)

optimized = optimizer.compile(
    classifier.deepcopy(),
    trainset=trainset,
    max_bootstrapped_demos=2,
    max_labeled_demos=3
)

# 7. Evaluate
evaluate = dspy.Evaluate(
    devset=testset,
    metric=classification_metric,
    display_progress=True,
    display_table=5
)

score = evaluate(optimized)
print(f"Accuracy: {score}")

# 8. Test
result = optimized(text="I want to cancel my card")
print(f"Predicted label: {result.label}")

# 9. Save
optimized.save("banking77_classifier.json")
```

### Example 3: RAG with KNNFewShot

```python
import dspy
from dspy.teleprompt import KNNFewShot
from sentence_transformers import SentenceTransformer
import random

# 1. Configure
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

# 2. Prepare training data with context
trainset = [
    dspy.Example(
        context=["Paris is the capital of France.", "France is in Europe."],
        question="What is the capital of France?",
        answer="Paris"
    ).with_inputs("context", "question"),
    dspy.Example(
        context=["London is the capital of England.", "England is in the UK."],
        question="What is the capital of England?",
        answer="London"
    ).with_inputs("context", "question"),
    # ... add 20-50 more examples
]

random.Random(0).shuffle(trainset)

# Split
train = trainset[:30]
dev = trainset[30:]

# 3. Define RAG program
class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> answer')

    def forward(self, context, question):
        # In production, you'd retrieve context here
        # For this example, context is provided in the input
        return self.respond(context=context, question=question)

program = RAG()

# 4. Optimize with KNNFewShot
knn_optimizer = KNNFewShot(
    k=3,  # Use 3 most similar examples
    trainset=train,
    vectorizer=dspy.Embedder(
        SentenceTransformer("all-MiniLM-L6-v2").encode
    )
)

optimized_rag = knn_optimizer.compile(program)

# 5. Test
result = optimized_rag(
    context=["Berlin is the capital of Germany."],
    question="What is the capital of Germany?"
)
print(f"Answer: {result.answer}")

# 6. Save
optimized_rag.save("rag_knn.json")
```

### Example 4: Loading Previously Saved Program

```python
import dspy

# Configure (must match the LM used during optimization)
dspy.configure(lm=dspy.LM('openai/gpt-4o-mini'))

# Method 1: Load state into existing program architecture
program = dspy.ChainOfThought("question -> answer")
program.load("qa_optimized.json")

# Method 2: Load full program (if saved with save_program=True)
program = dspy.load("full_program_directory/")

# Use loaded program
result = program(question="What is Rust?")
print(result.answer)
```

---

## Best Practices and Tips

### 1. Data Preparation

**Always mark inputs explicitly:**
```python
# ✅ Good
example = dspy.Example(
    question="...",
    answer="..."
).with_inputs("question")

# ❌ Bad - inputs not marked
example = dspy.Example(question="...", answer="...")
```

**Match your signature:**
```python
# If your signature is "context, question -> answer"
# Your examples must have those exact field names

signature = "context, question -> answer"
example = dspy.Example(
    context="...",
    question="...",
    answer="..."
).with_inputs("context", "question")  # ✅
```

**Source:** https://dspy.ai/learn/evaluation/data

### 2. Start Simple, Then Scale

```python
# 1. Start with LabeledFewShot on 10 examples
optimizer = LabeledFewShot(k=5)

# 2. Move to BootstrapFewShot with 30 examples
optimizer = BootstrapFewShot(max_bootstrapped_demos=3)

# 3. Scale to MIPROv2 with 200+ examples
optimizer = MIPROv2(auto="light")
```

### 3. Use Appropriate Validation Set Size

For prompt optimizers, **validation should be larger than training**:

```python
# ✅ Good for MIPROv2
trainset = data[:100]   # 20%
valset = data[100:]     # 80%

# ❌ Bad for MIPROv2
trainset = data[:400]   # 80%
valset = data[400:]     # 20%
```

**Source:** https://dspy.ai/learn/optimization/overview

### 4. Monitor Memory with Large Datasets

For large datasets, disable trace logging:

```python
optimizer = MIPROv2(
    metric=metric,
    auto="light",
    log_traces_from_compile=False  # ✅ Prevent memory issues
)
```

**Source:** https://dspy.ai/learn/optimization/optimizers

### 5. Iterate on Your Metric

Your metric is crucial. Start simple and refine:

```python
# V1: Simple exact match
def metric_v1(example, pred, trace=None):
    return example.answer == pred.answer

# V2: Case-insensitive
def metric_v2(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# V3: Partial match
def metric_v3(example, pred, trace=None):
    return example.answer.lower() in pred.answer.lower()

# V4: Different logic for bootstrapping vs evaluation
def metric_v4(example, pred, trace=None):
    exact = example.answer.lower() == pred.answer.lower()
    partial = example.answer.lower() in pred.answer.lower()

    if trace is None:  # Evaluation
        return 1.0 if exact else (0.5 if partial else 0.0)
    else:  # Bootstrapping
        return exact  # Only keep exact matches for demos
```

### 6. Save and Version Your Optimized Programs

```python
# Save with descriptive names
optimized.save(f"program_v1_mipro_light_{len(trainset)}examples.json")

# Include metadata
import json
metadata = {
    "optimizer": "MIPROv2",
    "auto_mode": "light",
    "train_size": len(trainset),
    "val_size": len(valset),
    "score": score,
    "date": "2024-01-15"
}

with open("program_v1_metadata.json", "w") as f:
    json.dump(metadata, f)
```

### 7. Test Before and After Optimization

```python
# Evaluate base program
base_score = evaluate(base_program, devset=testset)
print(f"Base score: {base_score}")

# Optimize
optimized = optimizer.compile(base_program, trainset=trainset)

# Evaluate optimized
optimized_score = evaluate(optimized, devset=testset)
print(f"Optimized score: {optimized_score}")
print(f"Improvement: {optimized_score - base_score}")
```

### 8. Use Multi-Threading for Speed

```python
# Enable threading for faster optimization and evaluation
optimizer = MIPROv2(
    metric=metric,
    auto="light",
    num_threads=24  # ✅ Use multiple threads
)

evaluate = dspy.Evaluate(
    devset=testset,
    metric=metric,
    num_threads=16  # ✅ Parallel evaluation
)
```

### 9. Handle Class Imbalance

For classification tasks, ensure balanced training data:

```python
from collections import Counter

# Check label distribution
labels = [ex.label for ex in trainset]
print(Counter(labels))

# Balance if needed
from sklearn.utils import resample

# Oversample minority classes or undersample majority
balanced_trainset = balance_dataset(trainset)
```

### 10. Serialize and Deserialize Properly

```python
# For JSON-serializable data (recommended)
trainset_dicts = [ex.toDict() for ex in trainset]
with open("trainset.json", "w") as f:
    json.dump(trainset_dicts, f)

# Load back
with open("trainset.json", "r") as f:
    trainset_dicts = json.load(f)

trainset = [
    dspy.Example(**d).with_inputs("question")  # ✅ Remember to mark inputs!
    for d in trainset_dicts
]
```

---

## Summary Cheatsheet

### Quick Start Steps

1. **Load your data** → List of dicts
2. **Convert to Examples** → `dspy.Example(**d).with_inputs(...)`
3. **Split data** → 20% train, 80% val (for prompt optimizers)
4. **Define program** → `dspy.ChainOfThought("inputs -> outputs")`
5. **Define metric** → Function returning True/False or float
6. **Choose optimizer** → Based on dataset size (see table)
7. **Compile** → `optimizer.compile(program, trainset=trainset)`
8. **Evaluate** → `dspy.Evaluate(devset=testset, metric=metric)`
9. **Save** → `optimized.save("program.json")`

### Common Patterns

```python
# Loading from HuggingFace
from dspy.datasets import DataLoader
data = DataLoader().from_huggingface(
    dataset_name="...",
    fields=("text", "label"),
    input_keys=("text",)
)

# Loading from JSON/JSONL
import ujson
with open("data.jsonl") as f:
    data = [ujson.loads(line) for line in f]
trainset = [dspy.Example(**d).with_inputs('question') for d in data]

# Basic optimization
from dspy.teleprompt import BootstrapFewShot
optimizer = BootstrapFewShot(metric=my_metric)
optimized = optimizer.compile(program, trainset=trainset)

# Advanced optimization
from dspy.teleprompt import MIPROv2
optimizer = MIPROv2(metric=my_metric, auto="light", num_threads=24)
optimized = optimizer.compile(program, trainset=trainset)
```

---

## Additional Resources

- **Official Documentation**: https://dspy.ai
- **Tutorials**: https://dspy.ai/tutorials
- **API Reference**: https://dspy.ai/api
- **Optimizers Guide**: https://dspy.ai/learn/optimization/optimizers
- **Examples Repository**: https://github.com/stanfordnlp/dspy/tree/main/examples

---

**Document Version:** 1.0
**Last Updated:** 2024-12-08
**Sources:** DSPy Official Documentation (dspy.ai), Context7 Documentation Database
