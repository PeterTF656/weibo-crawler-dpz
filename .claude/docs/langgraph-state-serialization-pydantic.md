# LangGraph State Serialization with Pydantic BaseModel

**Source**: Official LangGraph Documentation (langchain-ai/langgraph)
**Date Retrieved**: 2025-12-15
**Version Coverage**: LangGraph 0.2.74+

## Summary

LangGraph supports using Pydantic BaseModel classes as state schemas, enabling runtime validation on inputs to graph nodes. However, the serialization behavior is critical to understand: **graph outputs are always dictionaries, not Pydantic instances**, even when using BaseModel as the state schema. This document covers state serialization mechanics, TypedDict vs Pydantic tradeoffs, handling complex nested types, and best practices for managing state with Pydantic models.

---

## Key Concepts

### 1. Pydantic BaseModel as State Schema

LangGraph accepts three main types for `state_schema`:
- **TypedDict** (most common, best performance)
- **Dataclass** (allows default values)
- **Pydantic BaseModel** (enables runtime validation)

**Example**:
```python
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

class MyState(BaseModel):
    text: str
    count: int

builder = StateGraph(MyState)
# ... add nodes ...
graph = builder.compile()
```

**Key Characteristics**:
- Runtime validation occurs on **inputs to nodes**
- Validation does NOT occur on node outputs
- Pydantic is slower than TypedDict/dataclass
- Enables Pydantic features: validators, computed fields, type coercion

### 2. Critical Serialization Behavior

**THE MOST IMPORTANT RULE**: Graph outputs are always dictionaries, never Pydantic instances.

```python
from pydantic import BaseModel

class MyState(BaseModel):
    text: str
    count: int

def my_node(state: MyState):
    # Node RECEIVES a Pydantic instance
    print(type(state))  # <class '__main__.MyState'>
    print(state.text)   # Can access as attributes

    # Node RETURNS a dict for state updates
    return {"text": state.text + " processed", "count": state.count + 1}

graph = StateGraph(MyState)
graph.add_node("my_node", my_node)
graph.add_edge(START, "my_node")
graph.add_edge("my_node", END)
compiled = graph.compile()

# Input: Can be Pydantic instance OR dict
input_state = MyState(text="hello", count=0)
result = compiled.invoke(input_state)

# Output: ALWAYS a dict, never a Pydantic instance
print(type(result))  # <class 'dict'>
print(result)        # {'text': 'hello processed', 'count': 1}

# Convert back to Pydantic if needed
output_model = MyState(**result)
print(type(output_model))  # <class '__main__.MyState'>
```

**Key Points**:
- **Node Input**: Pydantic instance (validated)
- **Node Output**: Dictionary (no validation)
- **Graph Output**: Dictionary (not a Pydantic instance)
- **Manual Conversion Required**: `MyState(**result)` to get Pydantic instance back

### 3. Why Outputs Are Dictionaries

LangGraph's state management is based on **channels** and **reducers**:
- State updates are merged using reducer functions
- Nodes return partial updates (dicts), not full state replacements
- Checkpointers serialize state to JSON-compatible format
- Pydantic instances would break this reduction/merging mechanism

**State Update Flow**:
```
1. Previous state (dict) → Deserialized to Pydantic → Passed to node
2. Node processes → Returns dict update
3. Dict update → Merged with previous state via reducers
4. Merged state (dict) → Serialized to checkpoint
5. Next node → Cycle repeats
```

### 4. Runtime Validation Behavior

**Validation occurs ONLY on inputs to nodes**:

```python
from pydantic import BaseModel, Field

class MyState(BaseModel):
    count: int = Field(ge=0)  # Must be >= 0

def valid_node(state: MyState):
    # This works - input is validated
    return {"count": state.count + 1}

def invalid_node(state: MyState):
    # This WILL NOT raise validation error!
    # Output validation doesn't happen
    return {"count": -10}  # Violates ge=0 constraint

graph = StateGraph(MyState)
graph.add_node("valid", valid_node)
graph.add_node("invalid", invalid_node)
graph.add_edge(START, "valid")
graph.add_edge("valid", "invalid")
graph.add_edge("invalid", END)
compiled = graph.compile()

# Initial input: VALIDATED
compiled.invoke({"count": -5})  # ❌ Raises ValidationError

# Input to second node: VALIDATED (from previous state)
compiled.invoke({"count": 5})
# First node: count=6 (valid input to invalid_node)
# Second node: Returns {"count": -10} (NO ERROR!)
# Final result: {"count": -10}  # Invalid state persists!
```

**Limitation**: The error trace from Pydantic doesn't show which node caused the error.

---

## TypedDict vs Pydantic BaseModel

### Performance Comparison

| Feature | TypedDict | Pydantic BaseModel |
|---------|-----------|-------------------|
| Validation | Type hints only (static) | Runtime validation |
| Performance | Fast (no overhead) | Slower (validation + conversion) |
| Default values | ❌ (use dataclass) | ✅ |
| Type coercion | ❌ | ✅ (can be unexpected) |
| Computed fields | ❌ | ✅ |
| Validators | ❌ | ✅ |
| Reducers | ✅ Via Annotated | ❌ (use Annotated on fields) |

### When to Use TypedDict

```python
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages, AnyMessage

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    count: int
    data: dict
```

**Use TypedDict when**:
- Performance is critical
- State schema is simple
- No runtime validation needed
- You want the lightest-weight solution

### When to Use Pydantic BaseModel

```python
from pydantic import BaseModel, Field, validator
from typing import List
from langchain_core.messages import AnyMessage

class State(BaseModel):
    messages: List[AnyMessage]
    count: int = Field(ge=0, description="Must be non-negative")
    temperature: float = Field(ge=0.0, le=2.0)

    @validator('messages')
    def validate_messages(cls, v):
        if not v:
            raise ValueError("Messages cannot be empty")
        return v
```

**Use Pydantic when**:
- You need input validation
- State has complex constraints
- Type coercion is helpful (e.g., "123" → 123)
- You want self-documenting schemas
- Recursive validation is needed

---

## TypedDict with Pydantic Models as Field Types

### The Problem

When you use TypedDict with Pydantic models as field values:

```python
from typing_extensions import TypedDict
from pydantic import BaseModel

class EntityInfo(BaseModel):
    name: str
    type: str
    confidence: float

class State(TypedDict):
    entities: list[EntityInfo]  # Pydantic model as field type
    count: int
```

**What happens**:
1. LangGraph doesn't automatically validate nested Pydantic models in TypedDict
2. Checkpointer serializes to dict (via JsonPlusSerializer)
3. When deserialized, you get **plain dicts**, not Pydantic instances

```python
def process_entities(state: State):
    # state["entities"] is a list of DICTS, not EntityInfo instances!
    for entity in state["entities"]:
        print(type(entity))  # <class 'dict'>
        print(entity["name"])  # Must use dict access, not attributes
        # entity.name  # ❌ AttributeError!
```

### Solution 1: Use All Pydantic

Make the entire state a Pydantic model:

```python
from pydantic import BaseModel
from typing import List

class EntityInfo(BaseModel):
    name: str
    type: str
    confidence: float

class State(BaseModel):  # ← Top-level is Pydantic
    entities: List[EntityInfo]  # Nested Pydantic models
    count: int

def process_entities(state: State):
    # state.entities[0] is NOW an EntityInfo instance (on input)
    print(type(state.entities[0]))  # <class '__main__.EntityInfo'>
    print(state.entities[0].name)   # ✅ Attribute access works
```

**Caveat**: Graph output is still a dict, nested entities will be dicts:

```python
result = graph.invoke({"entities": [{"name": "Bob", "type": "Person", "confidence": 0.9}], "count": 0})
print(type(result))  # dict
print(type(result["entities"][0]))  # dict (not EntityInfo!)

# Convert back if needed
output_state = State(**result)
print(type(output_state.entities[0]))  # EntityInfo
```

### Solution 2: Manual Conversion in Nodes

Keep TypedDict and convert manually:

```python
from typing_extensions import TypedDict

class EntityInfo(BaseModel):
    name: str
    type: str

class State(TypedDict):
    entities: list[dict]  # Be explicit: it's dicts
    count: int

def process_entities(state: State):
    # Convert dicts to Pydantic instances when needed
    entity_objects = [EntityInfo(**e) for e in state["entities"]]

    # Process with validation
    for entity in entity_objects:
        print(entity.name)  # Attribute access

    # Return as dicts
    return {"entities": [e.dict() for e in entity_objects]}
```

### Solution 3: Use Plain Dicts

Simplest approach - embrace the dict nature:

```python
class State(TypedDict):
    entities: list[dict]  # Just use dicts
    count: int

def process_entities(state: State):
    for entity in state["entities"]:
        name = entity.get("name", "Unknown")
        entity_type = entity.get("type", "Unknown")
        # Work with dicts directly
```

---

## Handling Complex Nested Types

### Messages and LangChain Types

**Critical**: Use `AnyMessage`, not `BaseMessage`:

```python
from pydantic import BaseModel
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from typing import List

class ChatState(BaseModel):
    messages: List[AnyMessage]  # ✅ Use AnyMessage for proper serialization
    # messages: List[BaseMessage]  # ❌ Will cause serialization issues

def add_message(state: ChatState):
    # Input: messages are proper LangChain message instances
    print(type(state.messages[0]))  # HumanMessage or AIMessage

    new_messages = state.messages + [AIMessage(content="Hello!")]
    return {"messages": new_messages}
```

**Why**: `AnyMessage` is a Union type that includes all message variants, enabling proper serialization/deserialization.

### Custom Objects with Serialization

For custom objects in state, use JsonPlusSerializer with pickle fallback:

```python
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
import pandas as pd

class State(TypedDict):
    dataframe: pd.DataFrame  # Not JSON-serializable
    count: int

# Use pickle fallback for non-standard types
serializer = JsonPlusSerializer(pickle_fallback=True)
checkpointer = SqliteSaver.from_conn_string(
    ":memory:",
    serde=serializer
)

graph = builder.compile(checkpointer=checkpointer)
```

**Supported by JsonPlusSerializer (without pickle)**:
- LangChain primitives (messages, documents)
- LangGraph primitives
- Datetimes
- Enums
- Standard Python types (str, int, list, dict, etc.)

**Requires pickle fallback**:
- Pandas DataFrames
- NumPy arrays
- Custom class instances (without custom serialization)
- Complex third-party objects

---

## Best Practices

### 1. Choose the Right State Schema Type

**Decision Tree**:
```
Need runtime validation?
  ├─ Yes → Use Pydantic BaseModel
  │         • Accept performance overhead
  │         • Validate on node inputs
  │         • Convert outputs back when needed
  │
  └─ No → Need default values?
            ├─ Yes → Use dataclass
            └─ No → Use TypedDict (fastest)
```

### 2. Always Convert Graph Outputs Back to Pydantic

```python
class State(BaseModel):
    text: str
    count: int

# After graph execution
result = graph.invoke(input_state)

# ✅ Convert back to get Pydantic features
validated_result = State(**result)

# Now you can:
validated_result.json()  # Serialize to JSON
validated_result.dict()  # Get dict
validated_result.count   # Attribute access
```

### 3. Use Type Hints in Node Signatures

```python
def my_node(state: MyPydanticState) -> dict:
    """
    Explicitly type the input and output.
    Input: Pydantic instance
    Output: dict (for state update)
    """
    return {"field": state.field + 1}
```

### 4. Handle Pydantic Type Coercion Carefully

Pydantic automatically coerces types:

```python
from pydantic import BaseModel

class State(BaseModel):
    count: int
    flag: bool

# String to int coercion
result = graph.invoke({"count": "123", "flag": "true"})
print(result["count"])  # 123 (int, not string!)
print(result["flag"])   # True (bool, not string!)

# Invalid coercion raises error
graph.invoke({"count": "invalid"})  # ❌ ValidationError
```

**Watch out for**:
- Unexpected type conversions
- String booleans: `"true"` → `True`, `"false"` → `False`
- Numeric strings: `"123"` → `123`

### 5. Validate Node Outputs Manually (If Critical)

Since LangGraph doesn't validate node outputs:

```python
class State(BaseModel):
    count: int = Field(ge=0)

def careful_node(state: State) -> dict:
    new_count = state.count - 10

    # Manual validation before returning
    update = {"count": new_count}
    State(**{**state.dict(), **update})  # Validate the full state

    return update
```

### 6. Use Reducers with Pydantic (Via Annotated)

You can still use reducers with Pydantic fields:

```python
from pydantic import BaseModel
from typing import Annotated, List
import operator

class State(BaseModel):
    # Reducer on Pydantic field
    results: Annotated[List[str], operator.add] = []
    count: int = 0

# When multiple nodes update "results", they're concatenated
def node_a(state: State):
    return {"results": ["A"]}

def node_b(state: State):
    return {"results": ["B"]}

# After parallel execution: state["results"] = ["A", "B"]
```

### 7. Message Handling Pattern

```python
from pydantic import BaseModel
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from typing import List, Annotated
from langgraph.graph.message import add_messages

class ChatState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages] = []

def chat_node(state: ChatState):
    # Messages are proper LangChain instances on input
    last_message = state.messages[-1]

    # Return list to be appended via add_messages reducer
    return {"messages": [AIMessage(content="Response")]}
```

### 8. Dealing with Checkpointer Serialization

State must be serializable for checkpointing:

```python
from pydantic import BaseModel
from datetime import datetime

class State(BaseModel):
    timestamp: datetime  # ✅ Supported by JsonPlusSerializer
    text: str
    # complex_object: MyCustomClass  # ❌ Needs pickle fallback

# If you need custom objects, enable pickle
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

serde = JsonPlusSerializer(pickle_fallback=True)
checkpointer = SqliteSaver.from_conn_string(":memory:", serde=serde)
```

### 9. Debugging Validation Errors

Pydantic validation errors don't show which node failed:

```python
import logging

logger = logging.getLogger(__name__)

def my_node(state: State):
    try:
        # Your logic
        return {"field": value}
    except Exception as e:
        logger.error(f"Error in my_node: {e}")
        raise
```

### 10. Performance Optimization

If using Pydantic for state is too slow:

**Option A**: Use TypedDict for state, Pydantic for validation:
```python
class State(TypedDict):
    text: str
    count: int

class StateValidator(BaseModel):
    text: str
    count: int = Field(ge=0)

def validated_node(state: State):
    # Validate only when needed
    StateValidator(**state)  # Raises if invalid
    return {"count": state["count"] + 1}
```

**Option B**: Use dataclass for state:
```python
from dataclasses import dataclass

@dataclass
class State:
    text: str
    count: int = 0  # Default values supported
```

---

## Common Patterns

### Pattern 1: Input Validation, Dict Processing

```python
from pydantic import BaseModel, Field

class State(BaseModel):
    count: int = Field(ge=0)
    text: str

def my_node(state: State):
    # Input is validated Pydantic instance
    # Work with it as Pydantic
    validated_count = state.count

    # Return dict update
    return {"count": validated_count + 1, "text": state.text.upper()}

graph = StateGraph(State)
graph.add_node("my_node", my_node)
# ... build graph ...
compiled = graph.compile()

# Output is dict
result = compiled.invoke({"count": 5, "text": "hello"})
print(type(result))  # dict

# Convert back for Pydantic features
final_state = State(**result)
print(final_state.json())
```

### Pattern 2: Nested Pydantic Models

```python
from pydantic import BaseModel
from typing import List, Optional

class Entity(BaseModel):
    name: str
    type: str
    confidence: float

class State(BaseModel):
    entities: List[Entity] = []
    text: str

def extract_entities(state: State):
    # Input: state.entities is List[Entity] (Pydantic instances)
    print(type(state.entities[0]))  # Entity (if exists)

    # Create new entities
    new_entities = [
        Entity(name="Bob", type="Person", confidence=0.9)
    ]

    # Return as list (will be serialized to dicts)
    return {"entities": new_entities}

# After invocation
result = graph.invoke({"entities": [], "text": "Bob is here"})
print(type(result["entities"][0]))  # dict, not Entity!

# Convert back
final_state = State(**result)
print(type(final_state.entities[0]))  # Entity
```

### Pattern 3: Hybrid State (TypedDict + Selective Validation)

```python
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# TypedDict for performance
class State(TypedDict):
    user_input: str
    count: int
    validated_output: dict

# Pydantic for critical validation
class OutputValidator(BaseModel):
    result: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

def process_node(state: State):
    # Process with TypedDict (fast)
    output_data = {
        "result": state["user_input"].upper(),
        "confidence": 0.95
    }

    # Validate only the output
    validated = OutputValidator(**output_data)

    return {"validated_output": validated.dict()}
```

### Pattern 4: Message State with Pydantic

```python
from pydantic import BaseModel
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from typing import List, Annotated
from langgraph.graph.message import add_messages

class ConversationState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages] = []
    user_id: str
    context: dict = {}

def chat_node(state: ConversationState):
    # Access last message
    last_msg = state.messages[-1] if state.messages else None

    # Generate response
    response = AIMessage(content=f"Hello {state.user_id}!")

    return {"messages": [response]}

# Output still a dict
result = graph.invoke({
    "messages": [HumanMessage(content="Hi")],
    "user_id": "alice",
    "context": {}
})

# Messages in output are dicts
print(type(result["messages"][0]))  # dict

# Convert back
final = ConversationState(**result)
print(type(final.messages[0]))  # HumanMessage or AIMessage
```

---

## Limitations and Gotchas

### 1. Output is NOT Pydantic

```python
# ❌ Common mistake
result = graph.invoke(input_state)
result.field  # AttributeError! result is a dict

# ✅ Correct
result = graph.invoke(input_state)
value = result["field"]  # Dict access

# Or convert
state_obj = State(**result)
value = state_obj.field  # Attribute access
```

### 2. No Output Validation

```python
class State(BaseModel):
    count: int = Field(ge=0)

def bad_node(state: State):
    # This will NOT raise an error!
    return {"count": -10}  # Violates constraint

# The invalid state persists in the graph
```

### 3. Pydantic Validation Errors Don't Show Node Names

```python
# Error message:
# ValidationError: 1 validation error for State
#   count
#     ensure this value is greater than or equal to 0

# Missing: "Error occurred in node 'my_node'"
```

**Workaround**: Add try/catch in nodes with custom logging.

### 4. Performance Impact

Pydantic validation on every node input adds overhead:
- Validation logic execution
- Type coercion
- Field validation
- Custom validators

**Benchmark** (approximate):
- TypedDict: 1x (baseline)
- Dataclass: 1.1x
- Pydantic: 2-5x (depending on complexity)

### 5. Nested Models Become Dicts in Output

```python
class Inner(BaseModel):
    value: str

class Outer(BaseModel):
    inner: Inner

result = graph.invoke({"inner": {"value": "test"}})
print(type(result["inner"]))  # dict, not Inner!
```

### 6. Type Coercion Can Be Surprising

```python
class State(BaseModel):
    count: int
    flag: bool

# Unexpected coercions
graph.invoke({"count": "123", "flag": "yes"})
# count = 123 (int)
# flag = True (bool) - "yes" is truthy!
```

---

## Migration Guide

### From TypedDict to Pydantic

**Before**:
```python
from typing_extensions import TypedDict

class State(TypedDict):
    text: str
    count: int
```

**After**:
```python
from pydantic import BaseModel

class State(BaseModel):
    text: str
    count: int
```

**Update node signatures** (no change needed):
```python
def my_node(state: State):
    return {"count": state.count + 1}
```

**Update graph invocation**:
```python
# Before: result was already a dict
result = graph.invoke({"text": "hello", "count": 0})

# After: result is STILL a dict, but you can convert
result = graph.invoke({"text": "hello", "count": 0})
validated = State(**result)  # Convert to Pydantic
```

### From Pydantic to TypedDict (Performance)

**Before**:
```python
from pydantic import BaseModel

class State(BaseModel):
    text: str
    count: int
```

**After**:
```python
from typing_extensions import TypedDict

class State(TypedDict):
    text: str
    count: int
```

**Update nodes** (change attribute access to dict access):
```python
# Before
def my_node(state: State):
    return {"count": state.count + 1}

# After
def my_node(state: State):
    return {"count": state["count"] + 1}
```

---

## Related Documentation

- [LangGraph Conditional Parallel Execution](./langgraph-conditional-parallel-execution.md) - State management with reducers
- [LangGraph Human-in-the-Loop Complete Guide](./langgraph-human-in-the-loop-complete-guide.md) - Checkpointing and state persistence
- [LangGraph Observability, Tracing, and MLflow Integration](./langgraph-observability-tracing-mlflow.md) - Tracking state changes

---

## Summary

| Aspect | Behavior |
|--------|----------|
| State schema | Can be TypedDict, dataclass, or Pydantic BaseModel |
| Node input type | Pydantic instance (if using BaseModel schema) |
| Node output type | Dictionary (always) |
| Graph output type | Dictionary (never Pydantic instance) |
| Input validation | ✅ Yes (on node inputs) |
| Output validation | ❌ No |
| Nested Pydantic models | Deserialized to dicts in output |
| Performance | TypedDict > dataclass > Pydantic |
| Checkpointer serialization | Via JsonPlusSerializer (supports LangChain types, datetimes, enums) |
| Custom objects | Requires pickle_fallback=True |
| Best for validation | Use Pydantic |
| Best for performance | Use TypedDict |
| Best for defaults | Use dataclass or Pydantic |

**Golden Rule**: When using Pydantic as state schema, always remember that `graph.invoke()` returns a **dictionary**, not a Pydantic instance. Convert explicitly when you need Pydantic features: `State(**result)`.
