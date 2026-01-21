# LangGraph Conditional Parallel Node Execution

**Source**: Official LangGraph Documentation (langchain-ai/langgraph)
**Date Retrieved**: 2025-12-10
**Version Coverage**: LangGraph 0.2.74+

## Summary

LangGraph supports conditional parallel node execution through two primary mechanisms: (1) returning a list of node names from a routing function, and (2) using the `Send` API for dynamic fan-out with custom state per parallel task. Parallel nodes execute concurrently in the same superstep, and their state updates are merged using reducer functions. This enables powerful fan-out/fan-in patterns for map-reduce workflows, conditional branching to multiple paths, and dynamic parallel task creation.

---

## Key Concepts

### 1. Parallel Execution via List Return

**Pattern**: A routing function returns `Sequence[str]` or `list[str]` containing multiple node names.

**Behavior**:
- All nodes in the returned list execute **in parallel** during the next superstep
- Each parallel node receives the **same state** as input
- State updates from all parallel nodes are **merged** using reducer functions
- All parallel nodes must complete before proceeding to subsequent nodes

**Basic Example**:

```python
from typing import Sequence
from langgraph.graph import StateGraph, START, END

def route_bc_or_cd(state: State) -> Sequence[str]:
    """Route to multiple nodes based on state condition."""
    if state["which"] == "cd":
        return ["c", "d"]  # Both c and d execute in parallel
    return ["b", "c"]       # Both b and c execute in parallel

builder = StateGraph(State)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("c", node_c)
builder.add_node("d", node_d)

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route_bc_or_cd)
# All parallel nodes route to same next node
builder.add_edge("b", END)
builder.add_edge("c", END)
builder.add_edge("d", END)

graph = builder.compile()
```

**TypeScript Equivalent**:

```typescript
const routeBcOrCd = (state: z.infer<typeof State>): string[] => {
  if (state.which === "cd") {
    return ["c", "d"];
  }
  return ["b", "c"];
};
```

### 2. Dynamic Parallel Execution with Send API

**Pattern**: Use `Send` objects to dynamically create parallel tasks with **custom state** for each task.

**Behavior**:
- The routing function returns `list[Send]` objects
- Each `Send` specifies: (1) target node name, (2) custom state for that node
- Enables **map-reduce** patterns where each parallel task gets different input
- Number of parallel tasks is determined at runtime based on state

**Send Object Structure**:

```python
from langgraph.types import Send

Send(node_name: str, state: dict)
```

**Map-Reduce Example**:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import Annotated
import operator

class OverallState(TypedDict):
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]  # Reducer for aggregation
    best_joke: str

def generate_topics(state: OverallState):
    """Phase 1: Generate list of topics."""
    return {"subjects": ["lions", "elephants", "penguins"]}

def continue_to_jokes(state: OverallState) -> list[Send]:
    """Fan-out: Create parallel tasks for each subject."""
    return [
        Send("generate_joke", {"subject": s})
        for s in state["subjects"]
    ]

def generate_joke(state: OverallState):
    """Worker node: Generate joke for one subject."""
    joke_map = {
        "lions": "Why don't lions like fast food? Because they can't catch it!",
        "elephants": "Why don't elephants use computers? They're afraid of the mouse!",
        "penguins": "Why don't penguins like talking to strangers? They find it hard to break the ice."
    }
    # Reducer automatically appends to 'jokes' list
    return {"jokes": [joke_map[state["subject"]]]}

def best_joke(state: OverallState):
    """Fan-in: Aggregate results from all workers."""
    return {"best_joke": state["jokes"][0]}

builder = StateGraph(OverallState)
builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)  # Executed N times in parallel
builder.add_node("best_joke", best_joke)

builder.add_edge(START, "generate_topics")
# Fan-out: conditional edge returns list[Send]
builder.add_conditional_edges(
    "generate_topics",
    continue_to_jokes,
    ["generate_joke"]  # Target node for all Send objects
)
# Fan-in: all workers route to aggregator
builder.add_edge("generate_joke", "best_joke")
builder.add_edge("best_joke", END)

graph = builder.compile()
```

**TypeScript Equivalent**:

```typescript
import { Send } from "@langchain/langgraph";

const continueToJokes = (state: z.infer<typeof OverallState>) => {
  return state.subjects.map((subject) =>
    new Send("generateJoke", { subject })
  );
};

graph.addConditionalEdges("generateTopics", continueToJokes);
```

### 3. State Management with Reducers

**Critical Requirement**: When nodes execute in parallel and update the same state field, you **must** define a reducer function to merge updates.

**Common Reducers**:

| Reducer | Behavior | Use Case |
|---------|----------|----------|
| `operator.add` | Concatenate lists/strings | Aggregating results from parallel workers |
| `operator.or_` | Logical OR for booleans | Any worker sets flag to True |
| Custom function | Custom merge logic | Complex aggregation scenarios |

**Python State Definition with Reducer**:

```python
import operator
from typing import Annotated
from typing_extensions import TypedDict

class State(TypedDict):
    # Reducer makes this append-only (list concatenation)
    aggregate: Annotated[list[str], operator.add]

    # Regular field (no reducer) - only one node should update
    single_value: str
```

**TypeScript State Definition with Reducer**:

```typescript
import "@langchain/langgraph/zod";
import { z } from "zod";

const State = z.object({
  // Reducer makes this append-only
  aggregate: z.array(z.string()).langgraph.reducer((x, y) => x.concat(y)),

  // Regular field (no reducer)
  singleValue: z.string(),
});
```

**How Reducers Work**:

```python
# Node A returns: {"aggregate": ["A"]}
# Node B returns: {"aggregate": ["B"]}
# Node C returns: {"aggregate": ["C"]}

# After parallel execution, reducer merges:
# final_state["aggregate"] = ["A", "B", "C"]
```

**Without Reducer - ERROR**:

```python
# ❌ WRONG - No reducer defined
class State(TypedDict):
    results: list[str]  # No reducer!

# When parallel nodes both update 'results', you get:
# INVALID_CONCURRENT_GRAPH_UPDATE error
```

**Fix with Reducer**:

```python
# ✅ CORRECT - Reducer defined
class State(TypedDict):
    results: Annotated[list[str], operator.add]  # Has reducer!

# Now parallel updates are safely merged
```

---

## Implementation Patterns

### Pattern 1: Conditional Fan-Out to Multiple Nodes (Same State)

**Use Case**: Based on flags in state, execute 1-3 nodes in parallel, all receiving the same state.

**Your Phase 3 Scenario**:
- Phase 2.5 outputs: `time_properties`, `non_time_properties`, `supports_vector_search`
- Phase 3: If `time_properties` is True, run `time_node`; if `non_time_properties` is True, run `non_time_node`; if `supports_vector_search` is True, run `vector_node`
- All selected nodes execute in parallel
- Phase 4: Merge results

**Implementation**:

```python
import operator
from typing import Sequence, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    # Phase 2.5 outputs
    time_properties: bool
    non_time_properties: bool
    supports_vector_search: bool

    # Phase 3 outputs (merged via reducer)
    time_results: Annotated[list[dict], operator.add]
    non_time_results: Annotated[list[dict], operator.add]
    vector_results: Annotated[list[dict], operator.add]

    # Phase 4 output
    merged_results: dict

def route_phase_3(state: State) -> Sequence[str]:
    """Route to 1-3 nodes based on flags."""
    nodes_to_execute = []

    if state["time_properties"]:
        nodes_to_execute.append("time_node")
    if state["non_time_properties"]:
        nodes_to_execute.append("non_time_node")
    if state["supports_vector_search"]:
        nodes_to_execute.append("vector_node")

    # If no nodes selected, route directly to merge
    if not nodes_to_execute:
        return ["merge_node"]

    return nodes_to_execute

def time_node(state: State):
    """Process time properties."""
    # Do time-related processing
    return {"time_results": [{"processed": "time_data"}]}

def non_time_node(state: State):
    """Process non-time properties."""
    # Do non-time-related processing
    return {"non_time_results": [{"processed": "non_time_data"}]}

def vector_node(state: State):
    """Process vector search."""
    # Do vector search processing
    return {"vector_results": [{"processed": "vector_data"}]}

def merge_node(state: State):
    """Phase 4: Merge all results."""
    merged = {
        "time": state.get("time_results", []),
        "non_time": state.get("non_time_results", []),
        "vector": state.get("vector_results", [])
    }
    return {"merged_results": merged}

# Build graph
builder = StateGraph(State)
builder.add_node("phase_2_5", phase_2_5_node)
builder.add_node("time_node", time_node)
builder.add_node("non_time_node", non_time_node)
builder.add_node("vector_node", vector_node)
builder.add_node("merge_node", merge_node)

builder.add_edge(START, "phase_2_5")

# Conditional parallel execution
builder.add_conditional_edges(
    "phase_2_5",
    route_phase_3,
    # All possible destinations
    ["time_node", "non_time_node", "vector_node", "merge_node"]
)

# All Phase 3 nodes converge to merge
builder.add_edge("time_node", "merge_node")
builder.add_edge("non_time_node", "merge_node")
builder.add_edge("vector_node", "merge_node")
builder.add_edge("merge_node", END)

graph = builder.compile()
```

**Key Points**:
- Routing function returns `Sequence[str]` with 0-3 node names
- All returned nodes execute in parallel with the **same state**
- Each node updates different state fields (no conflicts)
- Merge node waits for all parallel nodes to complete
- If no nodes selected, route directly to merge

### Pattern 2: Dynamic Map-Reduce with Send

**Use Case**: Number of parallel tasks unknown until runtime, each task needs different input.

**Example**: Process multiple documents in parallel.

```python
from langgraph.types import Send

class State(TypedDict):
    documents: list[str]
    processed_docs: Annotated[list[dict], operator.add]  # Aggregates results

def fan_out_documents(state: State) -> list[Send]:
    """Create one task per document."""
    return [
        Send("process_document", {"doc_id": i, "doc_text": doc})
        for i, doc in enumerate(state["documents"])
    ]

def process_document(state: State):
    """Worker: Process one document."""
    doc_id = state["doc_id"]
    doc_text = state["doc_text"]

    # Process document
    result = {"doc_id": doc_id, "summary": f"Summary of {doc_text[:50]}"}

    # Append to aggregated results
    return {"processed_docs": [result]}

def aggregate_results(state: State):
    """Collect all processed documents."""
    return {"final_output": state["processed_docs"]}

builder = StateGraph(State)
builder.add_node("fan_out", fan_out_documents)
builder.add_node("process_document", process_document)
builder.add_node("aggregate", aggregate_results)

builder.add_edge(START, "fan_out")
builder.add_conditional_edges(
    "fan_out",
    fan_out_documents,
    ["process_document"]
)
builder.add_edge("process_document", "aggregate")
builder.add_edge("aggregate", END)

graph = builder.compile()
```

### Pattern 3: Fan-Out/Fan-In with State Accumulation

**Use Case**: Parallel nodes accumulate results into shared state fields.

```python
import operator
from typing import Annotated

class State(TypedDict):
    # Shared accumulator field
    aggregate: Annotated[list[str], operator.add]
    input_data: str

def node_a(state: State):
    return {"aggregate": ["result_from_a"]}

def node_b(state: State):
    return {"aggregate": ["result_from_b"]}

def node_c(state: State):
    return {"aggregate": ["result_from_c"]}

def merge_node(state: State):
    """All parallel results are in state['aggregate']."""
    print(f"Merged results: {state['aggregate']}")
    return {}

builder = StateGraph(State)
builder.add_node("splitter", splitter_node)
builder.add_node("a", node_a)
builder.add_node("b", node_b)
builder.add_node("c", node_c)
builder.add_node("merge", merge_node)

# Splitter decides which nodes to execute
builder.add_conditional_edges(
    "splitter",
    lambda state: ["a", "b", "c"],  # All three in parallel
)

# All parallel nodes route to merge
builder.add_edge("a", "merge")
builder.add_edge("b", "merge")
builder.add_edge("c", "merge")

graph = builder.compile()

# Result: state["aggregate"] will be ["result_from_a", "result_from_b", "result_from_c"]
```

### Pattern 4: Conditional Routing with Mapping Dictionary

**Use Case**: Map routing function outputs to specific nodes.

```python
from typing import Literal

def route_by_type(state: State) -> Literal["process_text", "process_image", "process_both"]:
    """Route based on content type."""
    if state["has_text"] and state["has_image"]:
        return "process_both"
    elif state["has_text"]:
        return "process_text"
    else:
        return "process_image"

builder.add_conditional_edges(
    "classifier",
    route_by_type,
    {
        "process_text": "text_node",
        "process_image": "image_node",
        "process_both": "combined_node"  # Routes to one node, not parallel
    }
)
```

**For Parallel Execution**:

```python
def route_parallel(state: State) -> Sequence[str]:
    """Return list for parallel execution."""
    nodes = []
    if state["has_text"]:
        nodes.append("text_node")
    if state["has_image"]:
        nodes.append("image_node")
    return nodes

builder.add_conditional_edges(
    "classifier",
    route_parallel,
    ["text_node", "image_node"]  # Both can execute in parallel
)
```

---

## Graph Wiring for Parallel Execution

### Option 1: All Parallel Nodes to Same Next Node

```python
builder.add_conditional_edges(
    "phase_2",
    routing_function,
    ["node_a", "node_b", "node_c"]
)

# All route to same merge node
builder.add_edge("node_a", "merge")
builder.add_edge("node_b", "merge")
builder.add_edge("node_c", "merge")
```

### Option 2: Direct Edges After Conditional

```python
# Conditional edges handle fan-out
builder.add_conditional_edges("splitter", route_fn)

# Regular edges handle fan-in
builder.add_edge("worker_a", "aggregator")
builder.add_edge("worker_b", "aggregator")
builder.add_edge("worker_c", "aggregator")
```

### Option 3: Fan-Out with Send API

```python
builder.add_conditional_edges(
    "orchestrator",
    fan_out_function,  # Returns list[Send]
    ["worker"]  # All Send objects target this node
)

# Single edge from worker to aggregator
builder.add_edge("worker", "aggregator")
```

---

## State Handling Details

### What State Do Parallel Nodes Receive?

**Answer**: All parallel nodes receive the **same state snapshot** from the previous superstep.

```python
# After phase_2 completes:
state = {"flag_a": True, "flag_b": True, "data": "shared"}

# Both node_a and node_b receive identical state:
# node_a sees: {"flag_a": True, "flag_b": True, "data": "shared"}
# node_b sees: {"flag_a": True, "flag_b": True, "data": "shared"}
```

### How Are State Updates Merged?

**With Reducer** (Safe for Parallel Updates):

```python
class State(TypedDict):
    results: Annotated[list[str], operator.add]

# node_a returns: {"results": ["A"]}
# node_b returns: {"results": ["B"]}
# Merged state: {"results": ["A", "B"]}
```

**Without Reducer** (Conflict):

```python
class State(TypedDict):
    results: list[str]  # No reducer!

# node_a returns: {"results": ["A"]}
# node_b returns: {"results": ["B"]}
# ERROR: INVALID_CONCURRENT_GRAPH_UPDATE
```

### Best Practice: Separate Fields for Parallel Nodes

```python
class State(TypedDict):
    # Each node updates its own field (no reducer needed)
    node_a_result: dict
    node_b_result: dict
    node_c_result: dict

    # Or use single field with reducer
    all_results: Annotated[list[dict], operator.add]
```

---

## Common Gotchas and Pitfalls

### 1. Missing Reducer for Concurrent Updates

**Problem**: Parallel nodes update the same state field without a reducer.

```python
# ❌ WRONG
class State(TypedDict):
    results: list[str]  # No reducer!

def node_a(state):
    return {"results": ["A"]}

def node_b(state):
    return {"results": ["B"]}

# Error: INVALID_CONCURRENT_GRAPH_UPDATE
```

**Fix**: Add a reducer.

```python
# ✅ CORRECT
class State(TypedDict):
    results: Annotated[list[str], operator.add]  # Has reducer!
```

### 2. Forgetting to Include All Possible Destinations

**Problem**: Conditional edge can return node names not listed in destinations.

```python
# ❌ WRONG
builder.add_conditional_edges(
    "router",
    routing_function,  # Can return ["a", "b", "c"]
    ["a", "b"]  # Missing "c"!
)
```

**Fix**: Include all possible destinations.

```python
# ✅ CORRECT
builder.add_conditional_edges(
    "router",
    routing_function,
    ["a", "b", "c"]  # All possible destinations
)
```

### 3. Returning Empty List from Routing Function

**Problem**: What happens if routing function returns `[]`?

```python
def route(state: State) -> Sequence[str]:
    if not state["any_flags_set"]:
        return []  # No nodes to execute
    return ["node_a"]
```

**Solution**: Route to a default node or END.

```python
def route(state: State) -> Sequence[str]:
    nodes = []
    if state["flag_a"]:
        nodes.append("node_a")
    if state["flag_b"]:
        nodes.append("node_b")

    if not nodes:
        return ["skip_node"]  # Default path
    return nodes

builder.add_node("skip_node", lambda state: {})
builder.add_edge("skip_node", "merge_node")
```

### 4. Mixing Send and Regular Returns

**Problem**: You cannot mix `Send` objects with node name strings.

```python
# ❌ WRONG
def route(state: State):
    return [
        Send("worker", {"task": 1}),
        "regular_node"  # Can't mix!
    ]
```

**Fix**: Use all `Send` or all strings.

```python
# ✅ CORRECT (all Send)
def route(state: State):
    return [
        Send("worker", {"task": 1}),
        Send("worker", {"task": 2})
    ]

# ✅ CORRECT (all strings)
def route(state: State):
    return ["node_a", "node_b"]
```

### 5. Parallel Nodes Not Converging to Same Node

**Problem**: Parallel nodes have different next nodes, making merge difficult.

```python
# ❌ PROBLEMATIC
builder.add_edge("node_a", "downstream_1")
builder.add_edge("node_b", "downstream_2")
# How to merge results?
```

**Fix**: Route all parallel nodes to same merge node.

```python
# ✅ CORRECT
builder.add_edge("node_a", "merge_node")
builder.add_edge("node_b", "merge_node")
builder.add_edge("node_c", "merge_node")
```

### 6. Accessing Results from Optional Parallel Nodes

**Problem**: Merge node tries to access results from nodes that didn't execute.

```python
def merge_node(state: State):
    # If node_a didn't execute, this KeyError!
    result_a = state["node_a_result"]
```

**Fix**: Use `.get()` with defaults.

```python
def merge_node(state: State):
    result_a = state.get("node_a_result", {})
    result_b = state.get("node_b_result", {})
    result_c = state.get("node_c_result", {})

    return {"merged": {
        "a": result_a,
        "b": result_b,
        "c": result_c
    }}
```

---

## Best Practices

### 1. Use Type Hints for Clarity

```python
from typing import Sequence, Literal

def route_phase_3(state: State) -> Sequence[str]:
    """Clear return type for parallel execution."""
    nodes: list[str] = []
    if state["flag_a"]:
        nodes.append("node_a")
    return nodes
```

### 2. Document Parallel Execution in Routing Functions

```python
def route_workers(state: State) -> list[Send]:
    """
    Fan-out: Create parallel worker tasks for each item.

    Returns:
        list[Send]: One Send object per item, all execute in parallel.
    """
    return [Send("worker", {"item": item}) for item in state["items"]]
```

### 3. Use Reducers Consistently

```python
# If ANY parallel nodes update a field, use a reducer
class State(TypedDict):
    # Parallel nodes update this
    results: Annotated[list[dict], operator.add]

    # Only one node updates this (no reducer needed)
    final_summary: str
```

### 4. Handle Empty Parallel Node Lists

```python
def route(state: State) -> Sequence[str]:
    nodes = []
    if state["condition_a"]:
        nodes.append("node_a")
    if state["condition_b"]:
        nodes.append("node_b")

    # Always have a fallback
    if not nodes:
        nodes.append("default_node")

    return nodes
```

### 5. Use Descriptive Node Names

```python
# ❌ BAD
builder.add_node("n1", func1)
builder.add_node("n2", func2)

# ✅ GOOD
builder.add_node("process_time_properties", process_time_node)
builder.add_node("process_non_time_properties", process_non_time_node)
builder.add_node("process_vector_search", process_vector_node)
```

### 6. Test with Different Parallel Combinations

```python
# Test all combinations:
# 1. No nodes execute (all flags False)
# 2. One node executes
# 3. Two nodes execute
# 4. All nodes execute

test_cases = [
    {"time": False, "non_time": False, "vector": False},  # None
    {"time": True, "non_time": False, "vector": False},   # One
    {"time": True, "non_time": True, "vector": False},    # Two
    {"time": True, "non_time": True, "vector": True},     # All
]

for test_state in test_cases:
    result = graph.invoke(test_state, config)
    assert "merged_results" in result
```

---

## Complete Example: Phase 3 Level Builder

Here's a complete implementation for your Phase 3 scenario:

```python
import operator
from typing import Sequence, Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class LevelBuilderState(TypedDict):
    # Phase 2.5 outputs
    time_properties: bool
    non_time_properties: bool
    supports_vector_search: bool

    # Context from earlier phases
    entity_info: dict
    relationship_info: dict

    # Phase 3 outputs (each node updates its own field)
    time_constraints: dict | None
    non_time_constraints: dict | None
    vector_search_config: dict | None

    # Phase 4 output
    final_level_config: dict

def phase_2_5_analysis(state: LevelBuilderState):
    """Analyze what types of processing are needed."""
    # Determine what flags to set based on entity/relationship analysis
    return {
        "time_properties": True,  # Has temporal properties
        "non_time_properties": True,  # Has other properties
        "supports_vector_search": False  # No vector search needed
    }

def route_phase_3(state: LevelBuilderState) -> Sequence[str]:
    """
    Route to 1-3 Phase 3 nodes based on flags.
    All selected nodes execute in parallel.
    """
    nodes_to_execute = []

    if state["time_properties"]:
        nodes_to_execute.append("process_time_constraints")

    if state["non_time_properties"]:
        nodes_to_execute.append("process_non_time_constraints")

    if state["supports_vector_search"]:
        nodes_to_execute.append("process_vector_search")

    # If no processing needed, skip to merge
    if not nodes_to_execute:
        return ["merge_phase_3_results"]

    return nodes_to_execute

def process_time_constraints(state: LevelBuilderState):
    """Phase 3a: Process temporal properties."""
    # Build time-based constraints
    time_config = {
        "type": "time",
        "constraints": ["before_date", "after_date"],
        "entity": state["entity_info"]
    }
    return {"time_constraints": time_config}

def process_non_time_constraints(state: LevelBuilderState):
    """Phase 3b: Process non-temporal properties."""
    # Build property-based constraints
    non_time_config = {
        "type": "properties",
        "constraints": ["equals", "contains"],
        "entity": state["entity_info"]
    }
    return {"non_time_constraints": non_time_config}

def process_vector_search(state: LevelBuilderState):
    """Phase 3c: Configure vector search."""
    # Build vector search configuration
    vector_config = {
        "type": "vector",
        "similarity_threshold": 0.8,
        "top_k": 10
    }
    return {"vector_search_config": vector_config}

def merge_phase_3_results(state: LevelBuilderState):
    """Phase 4: Merge results from all executed Phase 3 nodes."""
    final_config = {}

    # Safely access results from optional parallel nodes
    if state.get("time_constraints"):
        final_config["time"] = state["time_constraints"]

    if state.get("non_time_constraints"):
        final_config["non_time"] = state["non_time_constraints"]

    if state.get("vector_search_config"):
        final_config["vector"] = state["vector_search_config"]

    return {"final_level_config": final_config}

# Build the graph
builder = StateGraph(LevelBuilderState)

# Add all nodes
builder.add_node("phase_2_5", phase_2_5_analysis)
builder.add_node("process_time_constraints", process_time_constraints)
builder.add_node("process_non_time_constraints", process_non_time_constraints)
builder.add_node("process_vector_search", process_vector_search)
builder.add_node("merge_phase_3_results", merge_phase_3_results)

# Wire the graph
builder.add_edge(START, "phase_2_5")

# Conditional parallel fan-out
builder.add_conditional_edges(
    "phase_2_5",
    route_phase_3,
    [
        "process_time_constraints",
        "process_non_time_constraints",
        "process_vector_search",
        "merge_phase_3_results"  # Direct path if no processing needed
    ]
)

# Fan-in: All Phase 3 nodes converge to merge
builder.add_edge("process_time_constraints", "merge_phase_3_results")
builder.add_edge("process_non_time_constraints", "merge_phase_3_results")
builder.add_edge("process_vector_search", "merge_phase_3_results")

builder.add_edge("merge_phase_3_results", END)

# Compile
graph = builder.compile()

# Usage
result = graph.invoke({
    "entity_info": {"type": "Person", "name": "John"},
    "relationship_info": {"type": "KNOWS"}
})

print(result["final_level_config"])
```

---

## Summary of API Options

| Mechanism | Return Type | Use Case | State Per Task |
|-----------|-------------|----------|----------------|
| List of strings | `Sequence[str]` or `list[str]` | Fixed nodes, same state | All tasks get same state |
| Send API | `list[Send]` | Dynamic tasks, custom state | Each task gets custom state |
| Single string | `str` or `Literal[...]` | No parallel execution | N/A |
| Mapping dict | Return value maps to node | Explicit routing, no parallel | N/A |

---

## Reference Links

- [Graph API Documentation](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/graph-api.md)
- [Low-Level Concepts](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/concepts/low_level.md)
- [Map-Reduce Tutorial](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/map-reduce.md)
- [INVALID_CONCURRENT_GRAPH_UPDATE Error](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/troubleshooting/errors/INVALID_CONCURRENT_GRAPH_UPDATE.md)
- [Durable Execution](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/concepts/durable_execution.md)
