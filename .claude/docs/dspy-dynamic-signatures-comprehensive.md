# DSPy Dynamic Signatures - Comprehensive Reference

> Researched from DSPy Official Documentation

## Overview

DSPy signatures define the input/output specifications for LLM tasks. Unlike traditional function signatures that just describe behavior, **DSPy Signatures declare and initialize the behavior of modules**. This document focuses on the dynamic creation and manipulation of signatures at runtime.

## Core Concepts

### What are DSPy Signatures?

Signatures specify the inputs and outputs expected for a particular LLM operation. The field names matter semantically - a `question` is different from an `answer`, a `sql_query` is different from `python_code`. This semantic naming guides the LLM's understanding of the task.

### Static vs Dynamic Signatures

**Static Signatures**: Defined as classes at module level
```python
class BasicQA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="often between 1 and 5 words")
```

**Dynamic Signatures**: Created or modified at runtime based on conditions

## Dynamic Signature Creation Methods

### 1. String-Based Signature Creation (Inline Signatures)

The simplest form of dynamic signature creation uses string notation at runtime.

#### Basic Syntax

```python
import dspy

# Simple input -> output
signature = dspy.Signature("question -> answer")

# Multiple inputs
signature = dspy.Signature("question, context -> answer")

# Multiple outputs
signature = dspy.Signature("inputs -> output1, output2")

# With type annotations
signature = dspy.Signature("sentence -> sentiment: bool")
classify = dspy.Predict(signature)
```

#### With Instructions

```python
# Add custom instructions to guide the LM
toxicity = dspy.Predict(
    dspy.Signature(
        "comment -> toxic: bool",
        instructions="Mark as 'toxic' if the comment includes insults, harassment, or sarcastic derogatory remarks."
    )
)

# Instructions with variables (can reference runtime values)
qa_sig = dspy.Signature(
    "question, context -> answer",
    "Answer based on context only."
)
```

#### With Custom Types

```python
import pydantic

class QueryResult(pydantic.BaseModel):
    text: str
    score: float

# Use custom Pydantic types in signatures
signature = dspy.Signature("query: str -> result: QueryResult")

# Nested custom types
class MyContainer:
    class Query(pydantic.BaseModel):
        text: str
    class Score(pydantic.BaseModel):
        score: float

signature = dspy.Signature("query: MyContainer.Query -> score: MyContainer.Score")
```

### 2. Using the Signature Constructor Directly

The `Signature` class constructor accepts a fields dictionary and instructions string.

#### Constructor Signature

```python
def __init__(self, fields: dict[str, Any], instructions: str = ""):
    self.fields = fields
    self.instructions = instructions
```

#### Understanding Field Structure

From the `insert()` method implementation, we can see that fields are stored as tuples:

```python
# Fields are stored as: name -> (type, field_info)
new_fields = dict(input_fields + output_fields)
return Signature(new_fields, cls.instructions)
```

#### Programmatic Field Creation Example

```python
import dspy

# Create fields programmatically
fields = {}

# Method 1: Build from scratch (shown in insert() implementation)
input_fields = [
    ("question", (str, dspy.InputField(desc="User's question"))),
    ("context", (str, dspy.InputField(desc="Relevant context")))
]

output_fields = [
    ("answer", (str, dspy.OutputField(desc="Concise answer"))),
    ("confidence", (float, dspy.OutputField(desc="Confidence score")))
]

# Combine into fields dictionary
fields = dict(input_fields + output_fields)

# Create signature with constructor
dynamic_signature = dspy.Signature(fields, "Answer questions based on context.")
```

### 3. Dynamic Field Manipulation Methods

Start with a base signature and modify it at runtime using built-in methods.

#### append() - Add Field at End

```python
import dspy

class MySig(dspy.Signature):
    input_text: str = dspy.InputField(desc="Input sentence")
    output_text: str = dspy.OutputField(desc="Translated sentence")

# Add confidence field to outputs
NewSig = MySig.append("confidence", dspy.OutputField(desc="Translation confidence"))
print(list(NewSig.fields.keys()))
# Output: ['input_text', 'output_text', 'confidence']
```

#### prepend() - Add Field at Beginning

```python
# Add context field to inputs
NewSig = MySig.prepend("context", dspy.InputField(desc="Context for translation"))
print(list(NewSig.fields.keys()))
# Output: ['context', 'input_text', 'output_text']
```

#### insert() - Add Field at Specific Position

```python
# Insert field at position 0
NewSig = MySig.insert(0, "context", dspy.InputField(desc="Context for translation"))
print(list(NewSig.fields.keys()))
# Output: ['context', 'input_text', 'output_text']

# Insert at end using negative index (-1 appends)
NewSig2 = NewSig.insert(-1, "confidence", dspy.OutputField(desc="Translation confidence"))
print(list(NewSig2.fields.keys()))
# Output: ['context', 'input_text', 'output_text', 'confidence']
```

**insert() Method Signature**:

```python
@classmethod
def insert(cls, index: int, name: str, field, type_: type | None = None) -> type["Signature"]:
    """Insert a field at a specific position among inputs or outputs.

    Negative indices are supported (e.g., `-1` appends). If `type_` is omitted, the field's
    existing `annotation` is used; if that is missing, `str` is used.

    Args:
        index (int): Insertion position within the chosen section; negatives append.
        name (str): Field name to add.
        field: InputField or OutputField instance to insert.
        type_ (type | None): Optional explicit type annotation.

    Returns:
        A new Signature class with the field inserted.

    Raises:
        ValueError: If `index` falls outside the valid range for the chosen section.
    """
```

#### delete() - Remove Field

```python
class MySig(dspy.Signature):
    input_text: str = dspy.InputField(desc="Input sentence")
    temp_field: str = dspy.InputField(desc="Temporary debug field")
    output_text: str = dspy.OutputField(desc="Translated sentence")

# Remove temp_field
NewSig = MySig.delete("temp_field")
print(list(NewSig.fields.keys()))
# Output: ['input_text', 'output_text']

# No error if field doesn't exist
Unchanged = NewSig.delete("nonexistent")
print(list(Unchanged.fields.keys()))
# Output: ['input_text', 'output_text']
```

#### with_updated_fields() - Update Field Properties

```python
import dspy
from copy import deepcopy
from typing import Any, Type, Union

# Update field type and properties
UpdatedSig = MySig.with_updated_fields(
    "answer",
    type_=int,
    desc="detailed answer with sources",
    another_option=True
)
```

**with_updated_fields() Method**:

```python
@classmethod
def with_updated_fields(cls, name: str, type_: Union[Type, None] = None, **kwargs: dict[str, Any]) -> Type["Signature"]:
    """Create a new Signature class with the updated field information.

    Returns a new Signature class with the field, name, updated
    with fields[name].json_schema_extra[key] = value.

    Args:
        name: The name of the field to update.
        type_: The new type of the field.
        kwargs: The new values for the field.

    Returns:
        A new Signature class (not an instance) with the updated field information.
    """
    fields_copy = deepcopy(cls.fields)
    fields_copy[name].json_schema_extra = {
        **fields_copy[name].json_schema_extra,
        **kwargs,
    }
    if type_ is not None:
        fields_copy[name].annotation = type_
    return Signature(fields_copy, cls.instructions)
```

#### with_instructions() - Update Instructions

```python
import dspy

class MySig(dspy.Signature):
    input_text: str = dspy.InputField(desc="Input text")
    output_text: str = dspy.OutputField(desc="Output text")

# Create new signature with different instructions
NewSig = MySig.with_instructions("Translate to French.")
assert NewSig is not MySig
assert NewSig.instructions == "Translate to French."
```

### 4. Loading/Saving Signature State

Signatures can be serialized and reconstructed from state dictionaries.

#### load_state() - Reconstruct from State

```python
@classmethod
def load_state(cls, state):
    """Reconstruct a Signature object from a saved state dictionary."""
    signature_copy = Signature(deepcopy(cls.fields), cls.instructions)

    signature_copy.instructions = state["instructions"]
    for field, saved_field in zip(signature_copy.fields.values(), state["fields"], strict=False):
        field.json_schema_extra["prefix"] = saved_field["prefix"]
        field.json_schema_extra["desc"] = saved_field["description"]

    return signature_copy
```

## Field Types and Properties

### InputField

```python
dspy.InputField(
    desc="Description of the input",      # Field description
    prefix="Custom prefix:",               # Optional prefix for prompts
)
```

### OutputField

```python
dspy.OutputField(
    desc="Description of the output",     # Field description
    prefix="Output:",                      # Optional prefix for prompts
)
```

### Field Annotations

Fields support various type annotations:

```python
# Basic types
field: str
field: int
field: float
field: bool

# Collections
field: list[str]
field: dict[str, int]

# Pydantic models
field: CustomPydanticModel

# Literals (enums)
from typing import Literal
field: Literal['option1', 'option2', 'option3']

# Optional fields
from typing import Optional
field: Optional[str]
```

## Advanced Dynamic Patterns

### 1. Conditional Field Addition

```python
import dspy

def create_signature_for_task(task_type: str, include_reasoning: bool = False):
    """Create a signature based on task requirements."""

    # Start with base signature
    if task_type == "classification":
        sig = dspy.Signature("text -> category: str")
    elif task_type == "extraction":
        sig = dspy.Signature("document -> entities: list[str]")
    else:
        sig = dspy.Signature("input -> output")

    # Conditionally add reasoning field
    if include_reasoning:
        sig = sig.prepend("reasoning", dspy.OutputField(desc="Step-by-step reasoning"))

    return sig

# Usage
sig = create_signature_for_task("classification", include_reasoning=True)
```

### 2. Iterative Field Building

```python
import dspy

def build_multi_input_signature(input_fields: list[str], output_field: str, instructions: str = ""):
    """Build a signature with dynamic number of inputs."""

    # Start with minimal signature
    if not input_fields:
        raise ValueError("Must have at least one input field")

    # Create base signature with first input
    sig_str = f"{input_fields[0]} -> {output_field}"
    sig = dspy.Signature(sig_str, instructions)

    # Add remaining inputs
    for field_name in input_fields[1:]:
        sig = sig.prepend(field_name, dspy.InputField(desc=f"{field_name} input"))

    return sig

# Usage
sig = build_multi_input_signature(
    ["question", "context", "history"],
    "answer",
    "Answer based on all provided information"
)
```

### 3. Schema-Driven Signature Creation

```python
import dspy
from typing import Dict, Any

def signature_from_schema(schema: Dict[str, Any]) -> type:
    """Create a signature from a schema definition."""

    fields = []

    # Build input fields
    for name, spec in schema.get("inputs", {}).items():
        field_type = spec.get("type", str)
        description = spec.get("description", "")
        fields.append((name, (field_type, dspy.InputField(desc=description))))

    # Build output fields
    for name, spec in schema.get("outputs", {}).items():
        field_type = spec.get("type", str)
        description = spec.get("description", "")
        fields.append((name, (field_type, dspy.OutputField(desc=description))))

    # Create signature
    fields_dict = dict(fields)
    instructions = schema.get("instructions", "")

    return dspy.Signature(fields_dict, instructions)

# Usage
schema = {
    "inputs": {
        "question": {"type": str, "description": "User's question"},
        "context": {"type": list[str], "description": "Relevant passages"}
    },
    "outputs": {
        "answer": {"type": str, "description": "Concise answer"},
        "confidence": {"type": float, "description": "Confidence score"}
    },
    "instructions": "Answer based on provided context only."
}

sig = signature_from_schema(schema)
```

### 4. Runtime Field Modification (Real-world Example)

From DSPy's MultiChainComparison module:

```python
def __init__(self, signature, M=3, temperature=0.7, **config):
    super().__init__()

    self.M = M
    signature = ensure_signature(signature)

    *_, self.last_key = signature.output_fields.keys()

    # Dynamically append M reasoning attempt fields
    for idx in range(M):
        signature = signature.append(
            f"reasoning_attempt_{idx+1}",
            InputField(
                prefix=f"Student Attempt #{idx+1}:",
                desc="${reasoning attempt}",
            ),
        )

    # Prepend rationale output field
    signature = signature.prepend(
        "rationale",
        OutputField(
            prefix="Accurate Reasoning: Thank you everyone. Let's now holistically",
            desc="${corrected reasoning}",
        ),
    )

    self.predict = Predict(signature, temperature=temperature, **config)
```

### 5. Dynamic Signature in Loop

```python
import dspy

def iterative_refinement(initial_prompt: str, max_iterations: int = 5):
    """Iteratively refine output by dynamically modifying signature."""

    current_prompt = initial_prompt

    for i in range(max_iterations):
        print(f"Iteration {i+1} of {max_iterations}")

        # Create signature for this iteration
        if i == 0:
            # First iteration: simple signature
            sig = dspy.Signature("prompt: str -> output: str")
        else:
            # Later iterations: add feedback and refinement
            sig = dspy.Signature(
                "prompt: str, previous_output: str, feedback: str -> improved_output: str",
                "Improve the output based on feedback."
            )

        predictor = dspy.Predict(sig)

        # Use the signature...
        # (rest of iteration logic)
```

## Complete Examples

### Example 1: Dynamic QA System

```python
import dspy
from typing import List

class QASignature(dspy.Signature):
    """Answer questions with factual information."""
    question: str = dspy.InputField(desc="user's question")
    context: list[str] = dspy.InputField(desc="relevant passages from documents")
    answer: str = dspy.OutputField(desc="concise factual answer")
    confidence: float = dspy.OutputField(desc="confidence score between 0-1")

# Dynamic field manipulation
ExtendedQA = QASignature.append("reasoning", dspy.OutputField(desc="explanation"), type_=str)
ModifiedQA = QASignature.with_updated_fields("answer", desc="detailed answer with sources")

# Use the modified signature
predictor = dspy.ChainOfThought(ExtendedQA)
```

### Example 2: Multi-Modal Signature

```python
import dspy

class DogPictureSignature(dspy.Signature):
    """Output the dog breed of the dog in the image."""
    image_1: dspy.Image = dspy.InputField(desc="An image of a dog")
    answer: str = dspy.OutputField(desc="The dog breed of the dog in the image")

image_url = "https://picsum.photos/id/237/200/300"
classify = dspy.Predict(DogPictureSignature)
classify(image_1=dspy.Image.from_url(image_url))
```

### Example 3: Complex Pydantic Types

```python
import dspy
import pydantic

class ScienceNews(pydantic.BaseModel):
    text: str
    scientists_involved: list[str]

class NewsQA(dspy.Signature):
    """Get news about the given science field"""
    science_field: str = dspy.InputField()
    year: int = dspy.InputField()
    num_of_outputs: int = dspy.InputField()
    news: list[ScienceNews] = dspy.OutputField(desc="science news")

predict = dspy.Predict(NewsQA)
result = predict(science_field="Computer Theory", year=2022, num_of_outputs=1)
```

### Example 4: Email Processing with Dynamic Signatures

```python
import dspy
from typing import List, Optional
from pydantic import BaseModel
from enum import Enum

class EmailType(str, Enum):
    ORDER_CONFIRMATION = "order_confirmation"
    SUPPORT_REQUEST = "support_request"
    MEETING_INVITATION = "meeting_invitation"
    NEWSLETTER = "newsletter"
    PROMOTIONAL = "promotional"
    INVOICE = "invoice"
    SHIPPING_NOTIFICATION = "shipping_notification"
    OTHER = "other"

class UrgencyLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ExtractedEntity(BaseModel):
    entity_type: str
    value: str
    confidence: float

class ClassifyEmail(dspy.Signature):
    """Classify the type and urgency of an email based on its content."""
    email_subject: str = dspy.InputField(desc="The subject line of the email")
    email_body: str = dspy.InputField(desc="The main content of the email")
    sender: str = dspy.InputField(desc="Email sender information")

    email_type: EmailType = dspy.OutputField(desc="The classified type of email")
    urgency: UrgencyLevel = dspy.OutputField(desc="The urgency level of the email")
    reasoning: str = dspy.OutputField(desc="Brief explanation of the classification")

class ExtractEntities(dspy.Signature):
    """Extract key entities and information from email content."""
    email_content: str = dspy.InputField(desc="The full email content including subject and body")
    email_type: EmailType = dspy.InputField(desc="The classified type of email")

    key_entities: list[ExtractedEntity] = dspy.OutputField(desc="List of extracted entities with type, value, and confidence")
    financial_amount: Optional[float] = dspy.OutputField(desc="Any monetary amounts found (e.g., '$99.99')")
    important_dates: list[str] = dspy.OutputField(desc="List of important dates found in the email")
    contact_info: list[str] = dspy.OutputField(desc="Relevant contact information extracted")
```

## Best Practices

### 1. Field Naming

- Use semantically meaningful names: `question` vs `input`, `answer` vs `output`
- Be consistent with naming conventions across signatures
- Start simple - don't prematurely optimize field names
- The DSPy optimizer can handle refinement later

### 2. Dynamic Signature Design

- **Start simple**: Begin with string-based signatures and add complexity as needed
- **Immutability**: All signature modification methods return new signature classes, never modify in place
- **Type safety**: Always specify type annotations for better structure and validation
- **Documentation**: Use `desc` parameter extensively to guide the LM

### 3. When to Use Each Method

| Method | Use Case |
|--------|----------|
| String-based | Quick prototyping, simple tasks, runtime generation |
| Class-based | Complex signatures, reusable components, clear structure |
| Constructor | Programmatic generation from schemas, configurations |
| append/prepend | Adding fields to existing signatures |
| insert | Precise field positioning |
| with_updated_fields | Modifying existing field properties |
| with_instructions | Changing behavior without altering fields |

### 4. Common Patterns

**Pattern 1: Progressive Enhancement**
```python
# Start minimal
sig = dspy.Signature("query -> answer")

# Add context when available
if has_context:
    sig = sig.prepend("context", dspy.InputField(desc="Background information"))

# Add confidence tracking
if track_confidence:
    sig = sig.append("confidence", dspy.OutputField(desc="Confidence score"))
```

**Pattern 2: Factory Functions**
```python
def create_task_signature(task_type: str, **options):
    """Factory for creating task-specific signatures."""
    base_sigs = {
        "qa": "question, context -> answer",
        "classification": "text -> category",
        "extraction": "document -> entities: list[str]",
    }

    sig = dspy.Signature(base_sigs[task_type])

    if options.get("with_reasoning"):
        sig = sig.prepend("reasoning", dspy.OutputField())

    if options.get("with_confidence"):
        sig = sig.append("confidence", dspy.OutputField(desc="confidence score"))

    return sig
```

**Pattern 3: Configuration-Driven**
```python
def signature_from_config(config: dict):
    """Create signature from configuration dictionary."""
    input_spec = " ".join(config["inputs"])
    output_spec = " ".join(config["outputs"])
    sig_string = f"{input_spec} -> {output_spec}"

    return dspy.Signature(sig_string, config.get("instructions", ""))
```

## Integration with DSPy Modules

All dynamic signatures work seamlessly with DSPy modules:

```python
# With Predict
predictor = dspy.Predict(dynamic_signature)

# With ChainOfThought
cot = dspy.ChainOfThought(dynamic_signature)

# With ProgramOfThought
pot = dspy.ProgramOfThought(dynamic_signature)

# With custom modules
class CustomModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = dspy.Predict(signature)

    def forward(self, **kwargs):
        return self.predictor(**kwargs)
```

## Limitations and Considerations

### Immutability

All signature modification methods return **new** signature classes. Original signatures are never modified:

```python
original = MySig
modified = original.append("new_field", dspy.OutputField())

assert original is not modified  # True - they are different classes
```

### Type Inference

When type is not specified, it defaults to `str`:

```python
# These are equivalent
sig1 = MySig.append("field", dspy.OutputField())
sig2 = MySig.append("field", dspy.OutputField(), type_=str)
```

### Field Order Matters

For input/output sections, field order can affect prompt construction:

```python
# Order 1: context first
sig1 = dspy.Signature("context, question -> answer")

# Order 2: question first
sig2 = dspy.Signature("question, context -> answer")

# These create different prompts to the LM
```

## Summary

**Yes, DSPy signatures can be created and modified dynamically at runtime** through multiple approaches:

1. ✅ **String-based creation**: `dspy.Signature("input -> output")` - fully runtime
2. ✅ **Constructor-based**: `dspy.Signature(fields_dict, instructions)` - programmatic
3. ✅ **Method chaining**: `append()`, `prepend()`, `insert()`, `delete()` - modify existing
4. ✅ **Field updates**: `with_updated_fields()`, `with_instructions()` - refine behavior
5. ✅ **Schema-driven**: Build from configuration, JSON, or external specs

The framework is designed with flexibility in mind, allowing signatures to adapt to runtime conditions, user input, configuration files, or iterative refinement processes.

## Additional Resources

- Official DSPy Documentation: https://dspy.ai
- DSPy GitHub Repository: https://github.com/stanfordnlp/dspy
- DSPy Signatures API Reference: https://dspy.ai/api/signatures/Signature
- DSPy Modules Documentation: https://dspy.ai/learn/programming/modules
