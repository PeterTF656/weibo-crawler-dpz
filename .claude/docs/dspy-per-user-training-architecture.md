# DSPy Per-User Training Architecture - Official Documentation

> Researched from DSPy Official Documentation (dspy.ai)

## Overview

This document provides comprehensive architectural patterns and best practices for building personalized DSPy agents where each user has their own trained/optimized program. While DSPy doesn't provide explicit "per-user" APIs, its modular design enables flexible architectures for user-specific optimization, storage, and retrieval.

## Table of Contents

1. [Per-User Training Architecture Patterns](#per-user-training-architecture-patterns)
2. [Optimization Storage and Persistence](#optimization-storage-and-persistence)
3. [Training Data Management](#training-data-management)
4. [Few-Shot Learning with User-Specific Examples](#few-shot-learning-with-user-specific-examples)
5. [Teleprompter/Optimizer Usage for Individuals](#teleprompteroptimizer-usage-for-individuals)
6. [Model Persistence and Loading](#model-persistence-and-loading)
7. [Scaling Considerations](#scaling-considerations)
8. [Memory and Context Management](#memory-and-context-management)

---

## Per-User Training Architecture Patterns

### Architecture Pattern 1: Shared Base Program + Per-User Compiled State

The most efficient approach is to maintain a single base program architecture with per-user optimized state (demos, signatures, prompts).

```python
import dspy
from pathlib import Path

class UserSpecificProgramManager:
    """Manages per-user DSPy program instances with shared architecture."""

    def __init__(self, base_program_class, storage_dir="./user_programs"):
        self.base_program_class = base_program_class
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def get_user_program_path(self, user_id: str) -> Path:
        """Get the storage path for a user's optimized program."""
        return self.storage_dir / f"user_{user_id}_program.json"

    def load_user_program(self, user_id: str):
        """Load a user's optimized program or return a fresh instance."""
        program = self.base_program_class()  # Create base architecture
        program_path = self.get_user_program_path(user_id)

        if program_path.exists():
            program.load(str(program_path))  # Load user-specific state
            print(f"Loaded optimized program for user {user_id}")
        else:
            print(f"No optimized program found for user {user_id}, using base program")

        return program

    def save_user_program(self, user_id: str, program):
        """Save a user's optimized program state."""
        program_path = self.get_user_program_path(user_id)
        program.save(str(program_path), save_program=False)  # Save state only
        print(f"Saved optimized program for user {user_id}")

# Usage Example
class MyRAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)

# Initialize manager
manager = UserSpecificProgramManager(MyRAG)

# Load user-specific program
user_program = manager.load_user_program("user_123")

# After optimization
optimized_program = optimizer.compile(user_program, trainset=user_trainset)
manager.save_user_program("user_123", optimized_program)
```

**Source:** Derived from https://dspy.ai/tutorials/saving

### Architecture Pattern 2: Per-User Finetuned Models

For users requiring model-level customization, maintain separate finetuned model checkpoints per user.

```python
import dspy
from pathlib import Path

class UserModelManager:
    """Manages per-user finetuned language models."""

    def __init__(self, base_model: str, checkpoint_dir="./user_checkpoints"):
        self.base_model = base_model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def get_user_checkpoint_path(self, user_id: str) -> Path:
        """Get checkpoint path for user's finetuned model."""
        return self.checkpoint_dir / f"user_{user_id}_checkpoint"

    def load_user_lm(self, user_id: str):
        """Load user-specific finetuned LM or return base model."""
        checkpoint_path = self.get_user_checkpoint_path(user_id)

        if checkpoint_path.exists():
            # Load finetuned checkpoint
            lm = dspy.HFModel(checkpoint=str(checkpoint_path), model=self.base_model)
            print(f"Loaded finetuned model for user {user_id}")
        else:
            # Use base model
            lm = dspy.LM(self.base_model)
            print(f"Using base model for user {user_id}")

        return lm

    def finetune_for_user(self, user_id: str, program, trainset, metric):
        """Finetune a model specifically for a user."""
        from dspy.teleprompt import BootstrapFinetune

        # Configure finetuning
        config = dict(
            epochs=2,
            bf16=True,
            bsize=6,
            accumsteps=2,
            lr=5e-5
        )

        # Run finetuning
        finetune_optimizer = BootstrapFinetune(metric=metric)
        finetuned_program = finetune_optimizer.compile(
            program,
            trainset=trainset,
            **config
        )

        # Checkpoint is saved automatically during finetuning
        # Move it to user-specific location
        checkpoint_path = self.get_user_checkpoint_path(user_id)
        # Implementation depends on your finetuning setup

        return finetuned_program

# Usage
model_manager = UserModelManager("meta-llama/Llama-3.2-1B-Instruct")
user_lm = model_manager.load_user_lm("user_123")

# Set user's LM in program
program = MyRAG()
program.set_lm(user_lm)
```

**Source:** Derived from https://dspy.ai/tutorials/classification_finetuning and https://dspy.ai/cheatsheet_h=basicqa

### Architecture Pattern 3: Hybrid Approach (Shared + User-Specific Components)

Combine shared base components with user-specific modules for maximum flexibility.

```python
import dspy

class HybridUserProgram(dspy.Module):
    """Program with both shared and user-specific components."""

    def __init__(self, user_id: str):
        super().__init__()
        self.user_id = user_id

        # Shared component (same for all users)
        self.query_generator = dspy.Predict("question -> query")

        # User-specific component (optimized per user)
        self.answer_generator = dspy.ChainOfThought("context, question -> answer")

        # Load user-specific state if available
        self._load_user_state()

    def _load_user_state(self):
        """Load user-specific optimizations."""
        user_state_path = f"./user_states/{self.user_id}_state.json"
        if Path(user_state_path).exists():
            # Only load state for user-specific components
            state = self._load_state_from_file(user_state_path)
            self.answer_generator.load_state(state['answer_generator'])

    def save_user_state(self):
        """Save only user-specific component state."""
        user_state = {
            'answer_generator': self.answer_generator.dump_state()
        }
        user_state_path = f"./user_states/{self.user_id}_state.json"
        self._save_state_to_file(user_state, user_state_path)

    def forward(self, question):
        query = self.query_generator(question=question).query
        context = search(query)
        return self.answer_generator(context=context, question=question)
```

**Source:** Derived from DSPy Module patterns

---

## Optimization Storage and Persistence

### Saving Optimized Programs

DSPy provides two modes for saving programs: **state-only** and **full program** serialization.

#### Mode 1: State-Only Saving (Recommended for Per-User Systems)

State-only saving stores signatures, demos, and configurations without the program architecture. This is ideal for per-user scenarios where the base architecture is shared.

```python
# Save state to JSON (recommended - safe and readable)
optimized_program.save("./user_programs/user_123.json", save_program=False)

# Save state to pickle (for non-serializable objects like dspy.Image)
optimized_program.save("./user_programs/user_123.pkl", save_program=False)
```

**What gets saved in state:**
- `demos`: Few-shot demonstration examples
- `signature`: Input/output field definitions and instructions
- `lm`: Language model configuration
- `metadata`: Dependency versions for compatibility checking

**Source:** https://dspy.ai/tutorials/saving

#### Mode 2: Full Program Saving (Save Architecture + State)

Full program saving uses cloudpickle to serialize the entire module including architecture. Use this when each user has a unique program structure.

```python
# Save entire program to directory
optimized_program.save(
    "./user_programs/user_123/",
    save_program=True,
    modules_to_serialize=[custom_module]  # Optional: custom modules to serialize
)
```

**Warning:** Pickle files can execute arbitrary code. Only load from trusted sources.

**Source:** https://dspy.ai/api/modules/ProgramOfThought

### Loading Optimized Programs

```python
# Create base program with same architecture
loaded_program = dspy.ChainOfThought("question -> answer")

# Load state from JSON
loaded_program.load("./user_programs/user_123.json")

# Load state from pickle (requires allow_pickle=True)
loaded_program.load("./user_programs/user_123.pkl", allow_pickle=True)

# Load full program from directory
loaded_program = dspy.load("./user_programs/user_123/")
```

**Important:** When loading state-only saves, you must recreate the same program architecture before loading.

**Source:** https://dspy.ai/tutorials/saving

### State Structure Details

The `dump_state()` method serializes the following for each predictor:

```python
def dump_state(self, json_mode=True):
    state_keys = ["traces", "train"]
    state = {k: getattr(self, k) for k in state_keys}

    state["demos"] = []
    for demo in self.demos:
        demo = demo.copy()
        for field in demo:
            demo[field] = serialize_object(demo[field])

        if isinstance(demo, dict) or not json_mode:
            state["demos"].append(demo)
        else:
            state["demos"].append(demo.toDict())

    state["signature"] = self.signature.dump_state()
    state["lm"] = self.lm.dump_state() if self.lm else None
    return state
```

**Source:** https://dspy.ai/api/modules/Predict

---

## Training Data Management

### Organizing Per-User Training Data

Each user's training data should be structured as DSPy `Example` objects with clearly defined inputs.

```python
import dspy
from pathlib import Path
import json

class UserDataManager:
    """Manages per-user training datasets."""

    def __init__(self, data_dir="./user_training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_user_trainset(self, user_id: str, examples: list):
        """Save user's training examples."""
        data_path = self.data_dir / f"user_{user_id}_trainset.json"

        # Convert Examples to dicts for JSON serialization
        serialized = [example.toDict() for example in examples]

        with open(data_path, 'w') as f:
            json.dump(serialized, f, indent=2)

    def load_user_trainset(self, user_id: str) -> list:
        """Load user's training examples."""
        data_path = self.data_dir / f"user_{user_id}_trainset.json"

        if not data_path.exists():
            return []

        with open(data_path, 'r') as f:
            serialized = json.load(f)

        # Convert back to DSPy Examples with proper input marking
        examples = [
            dspy.Example(**ex).with_inputs(*self._get_input_keys(ex))
            for ex in serialized
        ]

        return examples

    def _get_input_keys(self, example_dict: dict) -> list:
        """Determine which keys are inputs (customizable logic)."""
        # Example: all keys except 'answer', 'label', etc. are inputs
        output_keys = {'answer', 'label', 'output', 'response'}
        return [k for k in example_dict.keys() if k not in output_keys]

    def add_user_example(self, user_id: str, example: dspy.Example):
        """Add a new training example for a user."""
        trainset = self.load_user_trainset(user_id)
        trainset.append(example)
        self.save_user_trainset(user_id, trainset)

    def get_user_stats(self, user_id: str) -> dict:
        """Get statistics about user's training data."""
        trainset = self.load_user_trainset(user_id)
        return {
            'num_examples': len(trainset),
            'has_data': len(trainset) > 0
        }

# Usage
data_manager = UserDataManager()

# Add examples for a user
user_examples = [
    dspy.Example(
        question="What is machine learning?",
        answer="A subset of AI that enables systems to learn from data"
    ).with_inputs("question"),
    dspy.Example(
        question="What is deep learning?",
        answer="A subset of ML using neural networks with multiple layers"
    ).with_inputs("question")
]

data_manager.save_user_trainset("user_123", user_examples)

# Load for optimization
trainset = data_manager.load_user_trainset("user_123")
```

**Source:** Derived from https://dspy.ai/tutorials/rag and https://dspy.ai/api/primitives/Example

### Creating User-Specific Examples

```python
# Method 1: Explicit field specification
example = dspy.Example(
    question="What is the weather?",
    answer="It's sunny",
).with_inputs("question")

# Method 2: Multiple inputs
example = dspy.Example(
    context="The sky is blue and clear.",
    question="What is the weather?",
    answer="It's sunny"
).with_inputs("context", "question")

# Accessing only inputs or labels
inputs = example.inputs()  # Returns Example with only input fields
labels = example.labels()  # Returns Example with only non-input fields
```

**Source:** https://dspy.ai/api/primitives/Example

### Incremental Learning from User Feedback

```python
class IncrementalUserLearning:
    """Incrementally update user's training data from interactions."""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.data_manager = UserDataManager()
        self.program_manager = UserSpecificProgramManager(MyRAG)

    def record_interaction(self, question: str, answer: str, feedback: str):
        """Record user interaction and feedback."""
        if feedback == "positive":
            # Add to training set
            example = dspy.Example(
                question=question,
                answer=answer
            ).with_inputs("question")

            self.data_manager.add_user_example(self.user_id, example)
            print(f"Added example to user {self.user_id}'s training set")

    def should_reoptimize(self) -> bool:
        """Check if we have enough new data to trigger reoptimization."""
        stats = self.data_manager.get_user_stats(self.user_id)

        # Example threshold: reoptimize every 10 examples
        return stats['num_examples'] > 0 and stats['num_examples'] % 10 == 0

    def reoptimize_user_program(self, optimizer, metric):
        """Reoptimize user's program with accumulated data."""
        trainset = self.data_manager.load_user_trainset(self.user_id)

        if len(trainset) < 5:  # Minimum examples needed
            print(f"Not enough examples for user {self.user_id} (need at least 5)")
            return None

        # Load current program
        program = self.program_manager.load_user_program(self.user_id)

        # Compile with user's data
        optimized = optimizer.compile(program, trainset=trainset)

        # Save optimized version
        self.program_manager.save_user_program(self.user_id, optimized)

        return optimized
```

---

## Few-Shot Learning with User-Specific Examples

### LabeledFewShot: Using User Examples Directly

The simplest approach is using `LabeledFewShot` to directly inject user examples as demonstrations.

```python
from dspy.teleprompt import LabeledFewShot
import dspy

# User's labeled examples
user_trainset = [
    dspy.Example(text="Cancel my subscription", label="account_management").with_inputs("text"),
    dspy.Example(text="When will my order arrive?", label="shipping_inquiry").with_inputs("text"),
    dspy.Example(text="I forgot my password", label="authentication").with_inputs("text"),
]

# Create base program
program = dspy.ChainOfThought("text -> label")

# Compile with user's examples (k=3 means use up to 3 demos)
optimizer = LabeledFewShot(k=3)
compiled_program = optimizer.compile(program, trainset=user_trainset)

# Now compiled_program has user's examples as demonstrations
```

**How it works:**
```python
def compile(self, student, *, trainset, sample=True):
    self.student = student.reset_copy()
    self.trainset = trainset

    if len(self.trainset) == 0:
        return self.student

    rng = random.Random(0)

    for predictor in self.student.predictors():
        if sample:
            predictor.demos = rng.sample(self.trainset, min(self.k, len(self.trainset)))
        else:
            predictor.demos = self.trainset[: min(self.k, len(self.trainset))]

    return self.student
```

**Source:** https://dspy.ai/api/optimizers/LabeledFewShot

### KNNFewShot: Semantic Similarity-Based Example Selection

For intelligent example selection based on input similarity, use `KNNFewShot` with user-specific examples.

```python
from dspy.teleprompt import KNNFewShot
from sentence_transformers import SentenceTransformer
import dspy

# User's training examples
user_trainset = [
    dspy.Example(question="What is Python?", answer="A programming language").with_inputs("question"),
    dspy.Example(question="What is Java?", answer="An OOP language").with_inputs("question"),
    dspy.Example(question="What is ML?", answer="Machine Learning").with_inputs("question"),
    # ... more user-specific examples
]

# Initialize KNN with user's examples
knn_optimizer = KNNFewShot(
    k=3,  # Select 3 most similar examples
    trainset=user_trainset,
    vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
)

# Compile program
qa_program = dspy.ChainOfThought("question -> answer")
compiled_qa = knn_optimizer.compile(qa_program)

# At inference, KNN will automatically select the 3 most relevant examples
# from user's trainset based on semantic similarity
result = compiled_qa(question="What is JavaScript?")
```

**Source:** https://dspy.ai/api/optimizers/KNNFewShot and https://dspy.ai/cheatsheet_h=basicqa

### Per-User Dynamic Example Selection

Create a custom wrapper that dynamically loads user examples at runtime.

```python
class UserAwareFewShot:
    """Dynamically load and apply user-specific examples."""

    def __init__(self, base_program_class, k=3):
        self.base_program_class = base_program_class
        self.k = k
        self.data_manager = UserDataManager()

    def get_program_for_user(self, user_id: str):
        """Get program with user's examples as demos."""
        # Load user's examples
        user_trainset = self.data_manager.load_user_trainset(user_id)

        # Create program
        program = self.base_program_class()

        if len(user_trainset) > 0:
            # Apply examples as demos to each predictor
            import random
            rng = random.Random(0)

            for predictor in program.predictors():
                predictor.demos = rng.sample(
                    user_trainset,
                    min(self.k, len(user_trainset))
                )

            print(f"Applied {min(self.k, len(user_trainset))} user examples as demos")

        return program

# Usage
user_few_shot = UserAwareFewShot(MyRAG, k=5)
user_program = user_few_shot.get_program_for_user("user_123")

# User's examples are now in the program's demos
response = user_program(question="User's question")
```

---

## Teleprompter/Optimizer Usage for Individuals

### BootstrapFewShot: Self-Generated Examples from User Data

`BootstrapFewShot` creates high-quality demonstrations by running the program and keeping successful traces.

```python
from dspy.teleprompt import BootstrapFewShot
import dspy

def user_metric(example, pred, trace=None):
    """Metric for evaluating examples (customize per use case)."""
    answer_match = example.answer.lower() == pred.answer.lower()

    if trace is None:  # Evaluation mode
        return float(answer_match)
    else:  # Bootstrapping mode
        return answer_match

# Load user's training data
user_trainset = data_manager.load_user_trainset("user_123")

# Configure optimizer
optimizer = BootstrapFewShot(
    metric=user_metric,
    max_bootstrapped_demos=4,  # Generate up to 4 examples
    max_labeled_demos=16,      # Include up to 16 labeled examples
    max_rounds=1,              # Number of bootstrap rounds
    max_errors=10              # Error tolerance
)

# Compile with user's data
base_program = MyRAG()
optimized_program = optimizer.compile(
    student=base_program,
    trainset=user_trainset
)

# Save for this user
program_manager.save_user_program("user_123", optimized_program)
```

**How it works:**
1. Runs the program on user's training examples
2. Keeps traces where `metric` returns `True`
3. Uses successful traces as few-shot demonstrations
4. Combines with labeled examples from trainset

**Source:** https://dspy.ai/api/optimizers/BootstrapFewShot

### BootstrapFewShotWithRandomSearch: Finding Best Configuration

For users with sufficient data (50+ examples), use random search to find optimal demo configurations.

```python
from dspy.teleprompt import BootstrapFewShotWithRandomSearch

# Load user's data (need 50+ examples for best results)
user_trainset = data_manager.load_user_trainset("user_123")
user_devset = user_trainset[:20]   # Validation set
user_train = user_trainset[20:]    # Training set

# Configure optimizer
config = dict(
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    num_candidate_programs=10,  # Try 10 different configurations
    num_threads=4
)

optimizer = BootstrapFewShotWithRandomSearch(
    metric=user_metric,
    **config
)

# Compile and find best program
optimized_program = optimizer.compile(
    student=base_program,
    trainset=user_train,
    valset=user_devset
)

# Access candidate programs (ranked by performance)
print(f"Found {len(optimized_program.candidate_programs)} candidates")
for i, candidate in enumerate(optimized_program.candidate_programs[:3]):
    print(f"Rank {i+1}: Score={candidate['score']}")
```

**Source:** https://dspy.ai/api/optimizers/BootstrapFewShotWithRandomSearch and https://dspy.ai/learn/optimization/optimizers

### MIPROv2: Advanced Multi-Stage Optimization

For users with 200+ examples, MIPROv2 provides the most sophisticated optimization.

```python
from dspy.teleprompt import MIPROv2
import dspy

# Load substantial user dataset
user_trainset = data_manager.load_user_trainset("user_123")  # 200+ examples

# Initialize optimizer
teleprompter = MIPROv2(
    metric=user_metric,
    auto="light",  # Options: light, medium, heavy
    # light: faster, fewer trials
    # medium: balanced
    # heavy: most thorough, expensive
)

# Optimize
print(f"Optimizing program for user_123 with {len(user_trainset)} examples...")
optimized_program = teleprompter.compile(
    program.deepcopy(),
    trainset=user_trainset,
    max_bootstrapped_demos=0,  # Or specify number
    max_labeled_demos=0,       # Or specify number
)

# Save
optimized_program.save(f"./user_programs/user_123_mipro.json")
```

**Optimization levels:**
- `auto="light"`: ~10-20 minutes, good for quick iteration
- `auto="medium"`: ~30-60 minutes, balanced quality
- `auto="heavy"`: 1+ hours, maximum quality

**Cost estimates (GPT-3.5):**
- Light: ~$1-2 USD
- Medium: ~$3-5 USD
- Heavy: ~$10-15 USD

**Source:** https://dspy.ai/api/optimizers/MIPROv2

### Choosing the Right Optimizer for User Data Size

**Official DSPy recommendations:**

| Training Examples | Recommended Optimizer |
|------------------|----------------------|
| 10 examples | `BootstrapFewShot` |
| 50+ examples | `BootstrapFewShotWithRandomSearch` |
| 200+ examples, 40+ trials | `MIPROv2` |
| Large LM (7B+ params) | `BootstrapFinetune` for efficiency |

**Source:** https://dspy.ai/learn/optimization/optimizers

---

## Model Persistence and Loading

### Complete Save/Load Workflow

```python
import dspy
from pathlib import Path

class UserProgramPersistence:
    """Complete persistence layer for user programs."""

    def __init__(self, storage_root="./user_storage"):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)

    def save_user_program(
        self,
        user_id: str,
        program,
        save_mode="state",  # "state" or "full"
        format="json"       # "json" or "pickle"
    ):
        """
        Save user's program with specified mode and format.

        Args:
            user_id: User identifier
            program: DSPy program to save
            save_mode: "state" (recommended) or "full"
            format: "json" (safe) or "pickle" (for complex objects)
        """
        user_dir = self.storage_root / user_id
        user_dir.mkdir(exist_ok=True)

        if save_mode == "state":
            # Save state only (architecture separate)
            ext = ".json" if format == "json" else ".pkl"
            path = user_dir / f"program_state{ext}"
            program.save(str(path), save_program=False)

            # Also save metadata
            metadata = {
                'user_id': user_id,
                'save_mode': save_mode,
                'format': format,
                'num_demos': len(program.demos) if hasattr(program, 'demos') else 0,
            }
            self._save_metadata(user_dir / "metadata.json", metadata)

        else:  # full program
            path = user_dir / "full_program"
            program.save(str(path), save_program=True)

        print(f"Saved {save_mode} for user {user_id} in {format} format")

    def load_user_program(
        self,
        user_id: str,
        base_program_class=None,  # Required for state-only loads
        format="json"
    ):
        """
        Load user's program.

        Args:
            user_id: User identifier
            base_program_class: Class to instantiate (for state loads)
            format: "json" or "pickle"
        """
        user_dir = self.storage_root / user_id

        # Try loading metadata first
        metadata_path = user_dir / "metadata.json"
        if metadata_path.exists():
            metadata = self._load_metadata(metadata_path)
            save_mode = metadata.get('save_mode', 'state')
        else:
            # Detect from file existence
            if (user_dir / "full_program").exists():
                save_mode = "full"
            else:
                save_mode = "state"

        if save_mode == "state":
            if base_program_class is None:
                raise ValueError(
                    "base_program_class required for loading state-only saves"
                )

            # Create base architecture
            program = base_program_class()

            # Load state
            ext = ".json" if format == "json" else ".pkl"
            path = user_dir / f"program_state{ext}"

            if path.exists():
                allow_pickle = (format == "pickle")
                program.load(str(path), allow_pickle=allow_pickle)
                print(f"Loaded state for user {user_id}")
            else:
                print(f"No saved state for user {user_id}, using base program")

        else:  # full program
            path = user_dir / "full_program"
            program = dspy.load(str(path))
            print(f"Loaded full program for user {user_id}")

        return program

    def _save_metadata(self, path, metadata):
        import json
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self, path):
        import json
        with open(path, 'r') as f:
            return json.load(f)

# Usage
persistence = UserProgramPersistence()

# Save user program
optimized_program = optimizer.compile(program, trainset=user_trainset)
persistence.save_user_program(
    "user_123",
    optimized_program,
    save_mode="state",
    format="json"
)

# Load user program
loaded_program = persistence.load_user_program(
    "user_123",
    base_program_class=MyRAG,
    format="json"
)
```

**Source:** https://dspy.ai/tutorials/saving

### Validating Loaded Programs

```python
def validate_loaded_program(original, loaded):
    """Validate that loaded program matches original."""

    # Check demo counts
    assert len(original.demos) == len(loaded.demos), \
        "Demo count mismatch"

    # Check demo content
    for orig_demo, loaded_demo in zip(original.demos, loaded.demos):
        # Note: loaded_demo is a dict, original is dspy.Example
        assert orig_demo.toDict() == loaded_demo, \
            "Demo content mismatch"

    # Check signatures
    assert str(original.signature) == str(loaded.signature), \
        "Signature mismatch"

    print("Validation successful!")

# Example
validate_loaded_program(compiled_program, loaded_program)
```

**Source:** https://dspy.ai/tutorials/saving

---

## Scaling Considerations

### Strategy 1: Lazy Loading

Don't load all user programs into memory. Load on-demand per request.

```python
from functools import lru_cache
import dspy

class LazyUserProgramLoader:
    """Lazy-load user programs with LRU cache."""

    def __init__(self, base_program_class, cache_size=100):
        self.base_program_class = base_program_class
        self.persistence = UserProgramPersistence()
        # Use LRU cache to keep recently used programs in memory
        self._load_cached = lru_cache(maxsize=cache_size)(self._load_program)

    def _load_program(self, user_id: str):
        """Load program (cached by LRU)."""
        return self.persistence.load_user_program(
            user_id,
            base_program_class=self.base_program_class
        )

    def get_program(self, user_id: str):
        """Get program for user (uses cache)."""
        return self._load_cached(user_id)

    def clear_cache(self):
        """Clear LRU cache."""
        self._load_cached.cache_clear()

    def get_cache_info(self):
        """Get cache statistics."""
        return self._load_cached.cache_info()

# Usage
loader = LazyUserProgramLoader(MyRAG, cache_size=100)

# First call loads from disk
program1 = loader.get_program("user_123")

# Second call uses cache
program2 = loader.get_program("user_123")  # Fast!

# Check cache performance
print(loader.get_cache_info())
# CacheInfo(hits=1, misses=1, maxsize=100, currsize=1)
```

### Strategy 2: Batch Optimization

Instead of optimizing per user immediately, batch optimization during off-peak hours.

```python
import dspy
from datetime import datetime
import asyncio

class BatchOptimizationScheduler:
    """Schedule and run batch optimization for multiple users."""

    def __init__(self):
        self.pending_optimizations = {}  # user_id -> trainset
        self.data_manager = UserDataManager()
        self.program_manager = UserSpecificProgramManager(MyRAG)

    def queue_user_for_optimization(self, user_id: str):
        """Queue a user for next optimization batch."""
        trainset = self.data_manager.load_user_trainset(user_id)

        if len(trainset) >= 10:  # Minimum threshold
            self.pending_optimizations[user_id] = trainset
            print(f"Queued user {user_id} for optimization ({len(trainset)} examples)")
        else:
            print(f"User {user_id} has insufficient data ({len(trainset)} examples)")

    def run_batch_optimization(self, optimizer, metric, max_concurrent=5):
        """Run optimization for all queued users."""
        print(f"Starting batch optimization for {len(self.pending_optimizations)} users")
        start_time = datetime.now()

        # Process in batches to limit concurrent operations
        user_ids = list(self.pending_optimizations.keys())

        for i in range(0, len(user_ids), max_concurrent):
            batch = user_ids[i:i+max_concurrent]

            for user_id in batch:
                try:
                    trainset = self.pending_optimizations[user_id]
                    program = self.program_manager.load_user_program(user_id)

                    # Optimize
                    optimized = optimizer.compile(program, trainset=trainset)

                    # Save
                    self.program_manager.save_user_program(user_id, optimized)

                    print(f"✓ Optimized program for user {user_id}")

                except Exception as e:
                    print(f"✗ Failed to optimize for user {user_id}: {e}")

        # Clear queue
        self.pending_optimizations.clear()

        duration = (datetime.now() - start_time).total_seconds()
        print(f"Batch optimization completed in {duration:.2f} seconds")

# Usage
scheduler = BatchOptimizationScheduler()

# Throughout the day, queue users
scheduler.queue_user_for_optimization("user_123")
scheduler.queue_user_for_optimization("user_456")
scheduler.queue_user_for_optimization("user_789")

# Run batch optimization (e.g., nightly)
from dspy.teleprompt import BootstrapFewShot
optimizer = BootstrapFewShot(metric=user_metric, max_bootstrapped_demos=4)
scheduler.run_batch_optimization(optimizer, user_metric, max_concurrent=5)
```

### Strategy 3: Tiered Optimization

Different optimization strategies based on user tier or data availability.

```python
class TieredOptimizationStrategy:
    """Apply different optimization strategies based on user tier."""

    def __init__(self):
        self.data_manager = UserDataManager()
        self.program_manager = UserSpecificProgramManager(MyRAG)

    def get_optimization_tier(self, user_id: str) -> str:
        """Determine optimization tier based on data availability."""
        stats = self.data_manager.get_user_stats(user_id)
        num_examples = stats['num_examples']

        if num_examples < 10:
            return "none"
        elif num_examples < 50:
            return "basic"
        elif num_examples < 200:
            return "advanced"
        else:
            return "premium"

    def optimize_for_user(self, user_id: str, metric):
        """Optimize using tier-appropriate strategy."""
        tier = self.get_optimization_tier(user_id)
        trainset = self.data_manager.load_user_trainset(user_id)
        program = self.program_manager.load_user_program(user_id)

        print(f"User {user_id}: tier={tier}, examples={len(trainset)}")

        if tier == "none":
            # No optimization, use base program
            print("Using base program (insufficient data)")
            return program

        elif tier == "basic":
            # Simple few-shot with labeled examples
            from dspy.teleprompt import LabeledFewShot
            optimizer = LabeledFewShot(k=min(5, len(trainset)))
            optimized = optimizer.compile(program, trainset=trainset)

        elif tier == "advanced":
            # Bootstrap with random search
            from dspy.teleprompt import BootstrapFewShot
            optimizer = BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=8
            )
            optimized = optimizer.compile(program, trainset=trainset)

        else:  # premium
            # Full MIPRO optimization
            from dspy.teleprompt import MIPROv2
            optimizer = MIPROv2(metric=metric, auto="light")
            optimized = optimizer.compile(program, trainset=trainset)

        # Save
        self.program_manager.save_user_program(user_id, optimized)
        return optimized

# Usage
tiered = TieredOptimizationStrategy()
optimized_program = tiered.optimize_for_user("user_123", user_metric)
```

### Strategy 4: Shared Embeddings for KNN

When using KNN-based few-shot, compute embeddings once and reuse.

```python
from sentence_transformers import SentenceTransformer
import dspy
import pickle
from pathlib import Path

class SharedEmbeddingStore:
    """Shared embedding computation for efficient KNN across users."""

    def __init__(self, model_name="all-MiniLM-L6-v2", cache_dir="./embeddings_cache"):
        self.model = SentenceTransformer(model_name)
        self.embedder = dspy.Embedder(self.model.encode)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_user_embeddings(self, user_id: str, trainset: list):
        """Get or compute embeddings for user's trainset."""
        cache_file = self.cache_dir / f"{user_id}_embeddings.pkl"

        # Check cache
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
                if len(cached['trainset']) == len(trainset):
                    print(f"Using cached embeddings for user {user_id}")
                    return cached['embeddings']

        # Compute embeddings
        print(f"Computing embeddings for user {user_id}")
        texts = [self._example_to_text(ex) for ex in trainset]
        embeddings = self.embedder(texts)

        # Cache
        with open(cache_file, 'wb') as f:
            pickle.dump({'trainset': trainset, 'embeddings': embeddings}, f)

        return embeddings

    def _example_to_text(self, example):
        """Convert example to text for embedding."""
        # Customize based on your example structure
        return example.get('question', '') or example.get('text', '')

# Usage
embedding_store = SharedEmbeddingStore()

# Get embeddings for user
user_trainset = data_manager.load_user_trainset("user_123")
embeddings = embedding_store.get_user_embeddings("user_123", user_trainset)

# Use with KNN
from dspy.teleprompt import KNNFewShot
optimizer = KNNFewShot(
    k=3,
    trainset=user_trainset,
    vectorizer=embedding_store.embedder
)
```

---

## Memory and Context Management

### Pattern 1: User Conversation History

Maintain conversation history per user and use it as context.

```python
import dspy
from collections import deque
from datetime import datetime

class UserConversationMemory:
    """Manage per-user conversation history."""

    def __init__(self, max_history=10):
        self.max_history = max_history
        self.conversations = {}  # user_id -> deque of exchanges

    def add_exchange(self, user_id: str, question: str, answer: str):
        """Add a Q&A exchange to user's history."""
        if user_id not in self.conversations:
            self.conversations[user_id] = deque(maxlen=self.max_history)

        exchange = {
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }

        self.conversations[user_id].append(exchange)

    def get_history(self, user_id: str, n: int = None) -> list:
        """Get user's conversation history."""
        if user_id not in self.conversations:
            return []

        history = list(self.conversations[user_id])

        if n is not None:
            history = history[-n:]  # Get last n exchanges

        return history

    def format_history_as_context(self, user_id: str, n: int = 5) -> str:
        """Format history as context string."""
        history = self.get_history(user_id, n)

        if not history:
            return ""

        context_parts = ["Previous conversation:"]
        for i, exchange in enumerate(history, 1):
            context_parts.append(f"Q{i}: {exchange['question']}")
            context_parts.append(f"A{i}: {exchange['answer']}")

        return "\n".join(context_parts)

# Integration with DSPy program
class MemoryAwareRAG(dspy.Module):
    """RAG with conversation memory."""

    def __init__(self, user_id: str, memory: UserConversationMemory):
        super().__init__()
        self.user_id = user_id
        self.memory = memory
        self.respond = dspy.ChainOfThought('context, conversation_history, question -> response')

    def forward(self, question):
        # Get context from retrieval
        context = search(question).passages

        # Get conversation history
        history = self.memory.format_history_as_context(self.user_id, n=5)

        # Generate response with history
        response = self.respond(
            context=context,
            conversation_history=history,
            question=question
        )

        # Store exchange
        self.memory.add_exchange(self.user_id, question, response.response)

        return response

# Usage
memory = UserConversationMemory(max_history=20)
user_rag = MemoryAwareRAG("user_123", memory)

# First question
response1 = user_rag(question="What is Python?")

# Second question (has access to previous exchange)
response2 = user_rag(question="Can you give me an example?")
```

### Pattern 2: User Preferences and Profile

Store and utilize user preferences as part of context.

```python
import dspy
import json
from pathlib import Path

class UserProfileManager:
    """Manage user profiles and preferences."""

    def __init__(self, profile_dir="./user_profiles"):
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(exist_ok=True)

    def get_profile_path(self, user_id: str) -> Path:
        return self.profile_dir / f"{user_id}_profile.json"

    def load_profile(self, user_id: str) -> dict:
        """Load user profile."""
        path = self.get_profile_path(user_id)

        if path.exists():
            with open(path, 'r') as f:
                return json.load(f)

        # Return default profile
        return {
            'user_id': user_id,
            'preferences': {},
            'context': {}
        }

    def save_profile(self, user_id: str, profile: dict):
        """Save user profile."""
        path = self.get_profile_path(user_id)
        with open(path, 'w') as f:
            json.dump(profile, f, indent=2)

    def update_preference(self, user_id: str, key: str, value):
        """Update a user preference."""
        profile = self.load_profile(user_id)
        profile['preferences'][key] = value
        self.save_profile(user_id, profile)

    def format_profile_as_context(self, user_id: str) -> str:
        """Format profile as context string."""
        profile = self.load_profile(user_id)

        parts = ["User Profile:"]

        if profile['preferences']:
            parts.append("Preferences:")
            for key, value in profile['preferences'].items():
                parts.append(f"  - {key}: {value}")

        if profile['context']:
            parts.append("Context:")
            for key, value in profile['context'].items():
                parts.append(f"  - {key}: {value}")

        return "\n".join(parts)

# Integration with DSPy
class ProfileAwareProgram(dspy.Module):
    """Program that uses user profile as context."""

    def __init__(self, user_id: str, profile_manager: UserProfileManager):
        super().__init__()
        self.user_id = user_id
        self.profile_manager = profile_manager
        self.respond = dspy.ChainOfThought('user_profile, question -> response')

    def forward(self, question):
        profile_context = self.profile_manager.format_profile_as_context(self.user_id)

        return self.respond(
            user_profile=profile_context,
            question=question
        )

# Usage
profile_mgr = UserProfileManager()

# Set some preferences
profile_mgr.update_preference("user_123", "language", "Python")
profile_mgr.update_preference("user_123", "expertise_level", "intermediate")
profile_mgr.update_preference("user_123", "preferred_style", "concise")

# Use in program
program = ProfileAwareProgram("user_123", profile_mgr)
response = program(question="How do I write a loop?")
# Response will be influenced by user's preferences
```

### Pattern 3: RAG with User-Specific Knowledge Base

Maintain a per-user document corpus for retrieval.

```python
import dspy
from pathlib import Path

class UserKnowledgeBase:
    """Manage per-user document corpora for RAG."""

    def __init__(self, kb_dir="./user_kb"):
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(exist_ok=True)
        self.user_retrievers = {}  # Cache retrievers

    def get_user_corpus_path(self, user_id: str) -> Path:
        return self.kb_dir / f"{user_id}_corpus.json"

    def load_user_corpus(self, user_id: str) -> list:
        """Load user's document corpus."""
        path = self.get_user_corpus_path(user_id)

        if path.exists():
            import json
            with open(path, 'r') as f:
                return json.load(f)

        return []

    def save_user_corpus(self, user_id: str, corpus: list):
        """Save user's document corpus."""
        path = self.get_user_corpus_path(user_id)
        import json
        with open(path, 'w') as f:
            json.dump(corpus, f, indent=2)

    def add_document(self, user_id: str, document: str):
        """Add document to user's corpus."""
        corpus = self.load_user_corpus(user_id)
        corpus.append(document)
        self.save_user_corpus(user_id, corpus)

        # Invalidate cached retriever
        if user_id in self.user_retrievers:
            del self.user_retrievers[user_id]

    def get_retriever(self, user_id: str, k=5):
        """Get or create retriever for user's corpus."""
        if user_id in self.user_retrievers:
            return self.user_retrievers[user_id]

        corpus = self.load_user_corpus(user_id)

        if not corpus:
            # Return empty retriever
            return lambda query: dspy.Prediction(passages=[], indices=[])

        # Create embedder and retriever
        embedder = dspy.Embedder('openai/text-embedding-3-small', dimensions=512)
        retriever = dspy.retrievers.Embeddings(
            embedder=embedder,
            corpus=corpus,
            k=k
        )

        # Cache
        self.user_retrievers[user_id] = retriever

        return retriever

# Integration
class UserKnowledgeRAG(dspy.Module):
    """RAG using user's personal knowledge base."""

    def __init__(self, user_id: str, kb: UserKnowledgeBase):
        super().__init__()
        self.user_id = user_id
        self.kb = kb
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        # Retrieve from user's personal corpus
        retriever = self.kb.get_retriever(self.user_id)
        context_result = retriever(question)

        # Generate response
        return self.respond(
            context=context_result.passages,
            question=question
        )

# Usage
kb = UserKnowledgeBase()

# Add documents to user's knowledge base
kb.add_document("user_123", "The user prefers Python for scripting tasks.")
kb.add_document("user_123", "The user's main project is a web application using FastAPI.")
kb.add_document("user_123", "The user has experience with Docker and Kubernetes.")

# Use in RAG
user_rag = UserKnowledgeRAG("user_123", kb)
response = user_rag(question="What should I use for my API?")
# Will retrieve from user's documents and personalize response
```

**Source:** Derived from https://dspy.ai/tutorials/rag and https://dspy.ai/api/tools/Embeddings

---

## Complete End-to-End Example

Here's a complete example integrating all concepts:

```python
import dspy
from pathlib import Path
from dspy.teleprompt import BootstrapFewShot, MIPROv2

class PersonalizedAgentSystem:
    """Complete per-user agent system."""

    def __init__(self):
        # Initialize all managers
        self.data_manager = UserDataManager()
        self.program_manager = UserSpecificProgramManager(MyRAG)
        self.memory = UserConversationMemory()
        self.profile_manager = UserProfileManager()
        self.kb = UserKnowledgeBase()

    def get_agent_for_user(self, user_id: str):
        """Get personalized agent for user."""
        # Load optimized program
        program = self.program_manager.load_user_program(user_id)

        # Enhance with memory
        class EnhancedUserAgent(dspy.Module):
            def __init__(self, base_program, user_id, memory, profile_mgr, kb):
                super().__init__()
                self.base_program = base_program
                self.user_id = user_id
                self.memory = memory
                self.profile_mgr = profile_mgr
                self.kb = kb

            def forward(self, question):
                # Get user context
                history = self.memory.format_history_as_context(self.user_id, n=3)
                profile = self.profile_mgr.format_profile_as_context(self.user_id)

                # Get from knowledge base
                retriever = self.kb.get_retriever(self.user_id)
                kb_context = retriever(question).passages

                # Combine contexts
                full_context = f"{profile}\n\n{history}\n\nKnowledge Base:\n{kb_context}"

                # Use base program with enhanced context
                response = self.base_program(question=question)

                # Store in memory
                self.memory.add_exchange(self.user_id, question, str(response))

                return response

        return EnhancedUserAgent(
            program, user_id, self.memory,
            self.profile_manager, self.kb
        )

    def optimize_for_user(self, user_id: str, metric):
        """Optimize agent for specific user."""
        # Load training data
        trainset = self.data_manager.load_user_trainset(user_id)

        if len(trainset) < 10:
            print(f"Insufficient data for user {user_id}")
            return None

        # Choose optimizer based on data size
        if len(trainset) < 50:
            optimizer = BootstrapFewShot(
                metric=metric,
                max_bootstrapped_demos=4,
                max_labeled_demos=8
            )
        else:
            optimizer = MIPROv2(metric=metric, auto="light")

        # Load base program
        program = self.program_manager.load_user_program(user_id)

        # Compile
        optimized = optimizer.compile(program, trainset=trainset)

        # Save
        self.program_manager.save_user_program(user_id, optimized)

        return optimized

    def add_user_feedback(self, user_id: str, question: str, answer: str,
                         feedback: str, correct_answer: str = None):
        """Process user feedback and update training data."""
        if feedback == "positive":
            # Add to training set
            example = dspy.Example(
                question=question,
                answer=answer
            ).with_inputs("question")
            self.data_manager.add_user_example(user_id, example)

        elif feedback == "negative" and correct_answer:
            # Add corrected example
            example = dspy.Example(
                question=question,
                answer=correct_answer
            ).with_inputs("question")
            self.data_manager.add_user_example(user_id, example)

        # Check if we should trigger optimization
        stats = self.data_manager.get_user_stats(user_id)
        if stats['num_examples'] > 0 and stats['num_examples'] % 20 == 0:
            print(f"Triggering optimization for user {user_id}")
            # Queue for batch optimization
            return True

        return False

# Usage
system = PersonalizedAgentSystem()

# Get agent for user
agent = system.get_agent_for_user("user_123")

# Use agent
response = agent(question="How do I deploy my app?")

# Process feedback
should_optimize = system.add_user_feedback(
    "user_123",
    question="How do I deploy my app?",
    answer=str(response),
    feedback="positive"
)

# Optimize when threshold reached
if should_optimize:
    def simple_metric(example, pred, trace=None):
        return example.answer.lower() in str(pred).lower()

    optimized = system.optimize_for_user("user_123", simple_metric)
```

---

## Summary and Best Practices

### Architecture Recommendations

1. **Start Simple**: Begin with shared architecture + per-user state (Pattern 1)
2. **Scale Gradually**: Add per-user finetuning only when needed (Pattern 2)
3. **Hybrid When Needed**: Use hybrid approach for flexibility (Pattern 3)

### Optimization Strategy

| User Data Size | Recommended Approach |
|----------------|---------------------|
| 0-10 examples | Use base program with `LabeledFewShot` |
| 10-50 examples | `BootstrapFewShot` optimization |
| 50-200 examples | `BootstrapFewShotWithRandomSearch` |
| 200+ examples | `MIPROv2` with auto="light" or "medium" |
| Very large datasets | Consider `BootstrapFinetune` |

### Storage Best Practices

1. **Use JSON for state saves** (safe, readable, version-controllable)
2. **Use pickle only for complex objects** (dspy.Image, datetime, etc.)
3. **Save metadata** (user_id, timestamp, num_examples, etc.)
4. **Implement versioning** to track program evolution

### Scaling Best Practices

1. **Lazy loading** with LRU cache for frequently accessed programs
2. **Batch optimization** during off-peak hours
3. **Tiered strategies** based on user value or data availability
4. **Shared embeddings** for KNN-based approaches
5. **Async operations** where possible

### Memory and Context

1. **Conversation history**: Keep last 10-20 exchanges
2. **User profiles**: Store preferences, expertise level, style
3. **Knowledge bases**: Maintain per-user document corpora
4. **Combine contexts** effectively in prompts

### Cost Management

- **Monitor optimization costs** (MIPROv2 can be $3-15 per user)
- **Use lighter optimizers** for most users
- **Reserve heavy optimization** for premium/high-value users
- **Batch operations** to amortize costs

### Production Considerations

1. **Implement proper error handling** for load/save operations
2. **Version control** saved programs and training data
3. **Monitor program performance** per user
4. **Implement rollback** mechanisms
5. **Log all optimizations** for debugging
6. **Use MLflow** for tracking user-specific experiments

---

## Additional Resources

- **Saving Programs**: https://dspy.ai/tutorials/saving
- **Optimizers Overview**: https://dspy.ai/learn/optimization/optimizers
- **BootstrapFewShot**: https://dspy.ai/api/optimizers/BootstrapFewShot
- **MIPROv2**: https://dspy.ai/api/optimizers/MIPROv2
- **Examples and Training Data**: https://dspy.ai/api/primitives/Example
- **RAG Patterns**: https://dspy.ai/tutorials/rag
- **Evaluation**: https://dspy.ai/learn/evaluation/metrics
