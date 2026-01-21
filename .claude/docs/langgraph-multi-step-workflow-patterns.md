# LangGraph Multi-Step Workflow Design Patterns

**Source**: Official LangGraph Documentation (langchain-ai/langgraph)
**Date Retrieved**: 2025-12-18
**Version Coverage**: LangGraph 0.2.74+

## Summary

This guide covers best practices for designing multi-step sequential workflows with embedded parallel execution in LangGraph. It demonstrates state management, node chaining, conditional routing, and parallel fan-out/fan-in patterns specifically for complex multi-phase workflows like hypothesis generation, report building, and analysis pipelines.

---

## Core Workflow Architecture

### Multi-Step Sequential Pattern with Parallel Phases

**Structure**: A → B → (C1 || C2) → D

```
Step 1: Analyze Post
   ↓
Step 2: Diagnose Gaps
   ↓
   ├─→ Context Query 1 ─┐
   |                     ├─→ Step 4: Form Hypotheses
   └─→ Context Query 2 ─┘
```

This pattern combines:
- **Sequential execution** (Steps 1, 2, 4)
- **Parallel execution** (Context queries)
- **Fan-in convergence** (Merging parallel results)
- **Conditional routing** (Optional steps based on state)

---

## State Definition for Multi-Step Workflows

### Pattern 1: Phase-Based State (Recommended)

Organize state fields by workflow phase for clarity:

```python
import operator
from typing import Annotated, Optional
from typing_extensions import TypedDict

class HypothesisWorkflowState(TypedDict):
    # Input
    post_text: str
    user_context: dict

    # Phase A.1: Analyze Post
    extracted_signals: dict
    signal_confidence: float

    # Phase A.2: Diagnose Gaps
    identified_gaps: list[str]
    context_questions: list[str]  # Exactly 2 questions

    # Phase A.3: Context Search (parallel results)
    query_1_results: Optional[dict]
    query_2_results: Optional[dict]
    # Alternative: Use reducer for aggregated results
    all_context_results: Annotated[list[dict], operator.add]

    # Phase A.4: Form Hypotheses
    hypotheses: list[dict]  # 2 diverse hypotheses
    confidence_scores: list[float]
```

**Key Design Decisions**:
- **Separate fields for parallel nodes**: `query_1_results` vs `query_2_results` (no reducer needed)
- **OR use reducer**: `all_context_results` with `operator.add` for aggregation
- **Optional fields**: Use `Optional[dict]` for results from conditional nodes
- **Typed lists**: Specify expected structure in docstrings or use Pydantic

### Pattern 2: Pydantic State with Nested Models

For complex validation and nested structures:

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Signal(BaseModel):
    type: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)

class ContextQuestion(BaseModel):
    question: str
    search_type: str  # "time_based", "entity_based", "vector"
    priority: int = Field(ge=1, le=10)

class ContextResult(BaseModel):
    question_id: int
    results: list[dict]
    retrieval_confidence: float

class Hypothesis(BaseModel):
    id: int
    description: str
    supporting_evidence: list[str]
    confidence: float = Field(ge=0.0, le=1.0)

class HypothesisWorkflowState(BaseModel):
    # Input
    post_text: str

    # Phase A.1
    signals: List[Signal] = []

    # Phase A.2
    context_questions: List[ContextQuestion] = []

    # Phase A.3 (with reducer for parallel aggregation)
    context_results: Annotated[List[ContextResult], operator.add] = []

    # Phase A.4
    hypotheses: List[Hypothesis] = []
```

**Tradeoffs**:
- ✅ Runtime validation on node inputs
- ✅ Self-documenting structure
- ✅ IDE autocomplete support
- ❌ Performance overhead (2-5x slower than TypedDict)
- ⚠️ Graph output is still a dict, not Pydantic instance (requires manual conversion)

**Important**: When using Pydantic, convert graph output back to Pydantic:
```python
result = graph.invoke(input_state)
validated_result = HypothesisWorkflowState(**result)
```

---

## Node Implementation Patterns

### Sequential Node Pattern

Each node receives full state, returns partial update:

```python
def analyze_post(state: HypothesisWorkflowState) -> dict:
    """
    Phase A.1: Extract signals from post text.

    Returns:
        Partial state update with extracted signals.
    """
    post_text = state["post_text"]

    # Use LLM or DSPy module to extract signals
    signals = extract_signals_from_text(post_text)

    return {
        "extracted_signals": signals,
        "signal_confidence": calculate_confidence(signals)
    }

def diagnose_gaps(state: HypothesisWorkflowState) -> dict:
    """
    Phase A.2: Identify missing context, generate 2 questions.

    Returns:
        Partial state update with questions for context retrieval.
    """
    signals = state["extracted_signals"]
    user_context = state["user_context"]

    # Analyze what context is missing
    gaps = identify_missing_context(signals, user_context)

    # Generate exactly 2 context questions
    questions = generate_context_questions(gaps, limit=2)

    return {
        "identified_gaps": gaps,
        "context_questions": questions
    }
```

**Best Practices**:
1. **Single responsibility**: Each node handles one phase
2. **Explicit inputs**: Access only state fields needed for this phase
3. **Partial updates**: Return only fields that change
4. **Type hints**: Annotate state parameter and return type
5. **Docstrings**: Document phase number, purpose, and outputs

### Parallel Worker Node Pattern

Nodes that execute in parallel:

```python
def context_search_query_1(state: HypothesisWorkflowState) -> dict:
    """
    Phase A.3a: Execute first context search query.

    Runs in parallel with query 2.
    """
    question = state["context_questions"][0]

    # Execute graph search
    results = execute_graph_search(question["text"])

    return {
        "query_1_results": {
            "question": question,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    }

def context_search_query_2(state: HypothesisWorkflowState) -> dict:
    """
    Phase A.3b: Execute second context search query.

    Runs in parallel with query 1.
    """
    question = state["context_questions"][1]

    # Execute graph search
    results = execute_graph_search(question["text"])

    return {
        "query_2_results": {
            "question": question,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    }
```

**Alternative: Using Reducer for Aggregation**:

```python
def context_search_query_1(state: HypothesisWorkflowState) -> dict:
    """Parallel worker that appends to aggregated results."""
    question = state["context_questions"][0]
    results = execute_graph_search(question["text"])

    # Reducer automatically appends to list
    return {
        "all_context_results": [{
            "question_id": 0,
            "question": question,
            "results": results
        }]
    }

# State definition with reducer:
class HypothesisWorkflowState(TypedDict):
    context_questions: list[str]
    all_context_results: Annotated[list[dict], operator.add]  # Reducer!
```

### Aggregator/Merge Node Pattern

Node that waits for all parallel workers to complete:

```python
def form_hypotheses(state: HypothesisWorkflowState) -> dict:
    """
    Phase A.4: Generate 2 diverse hypotheses from context.

    Waits for both context queries to complete.
    """
    signals = state["extracted_signals"]
    gaps = state["identified_gaps"]

    # Access parallel results safely (they might not exist if queries failed)
    query_1 = state.get("query_1_results", {})
    query_2 = state.get("query_2_results", {})

    # Combine context from both queries
    combined_context = merge_context_results(query_1, query_2)

    # Generate 2 diverse hypotheses
    hypotheses = generate_diverse_hypotheses(
        signals=signals,
        gaps=gaps,
        context=combined_context,
        num_hypotheses=2
    )

    return {
        "hypotheses": hypotheses,
        "confidence_scores": [h["confidence"] for h in hypotheses]
    }
```

**Best Practices for Merge Nodes**:
1. **Use `.get()` for optional fields**: Handle cases where parallel nodes didn't execute
2. **Validate completeness**: Check if all expected parallel results are present
3. **Error handling**: Handle partial failures in parallel branches
4. **Combine intelligently**: Don't just concatenate—synthesize information

---

## Graph Construction Patterns

### Pattern 1: Fixed Parallel Execution

When you always execute a fixed number of parallel nodes:

```python
from langgraph.graph import StateGraph, START, END

def build_hypothesis_workflow():
    """Build hypothesis generation workflow with fixed parallel queries."""

    builder = StateGraph(HypothesisWorkflowState)

    # Add all nodes
    builder.add_node("analyze_post", analyze_post)
    builder.add_node("diagnose_gaps", diagnose_gaps)
    builder.add_node("context_query_1", context_search_query_1)
    builder.add_node("context_query_2", context_search_query_2)
    builder.add_node("form_hypotheses", form_hypotheses)

    # Sequential: START → A.1 → A.2
    builder.add_edge(START, "analyze_post")
    builder.add_edge("analyze_post", "diagnose_gaps")

    # Parallel: A.2 → (A.3a || A.3b)
    builder.add_edge("diagnose_gaps", "context_query_1")
    builder.add_edge("diagnose_gaps", "context_query_2")

    # Fan-in: (A.3a || A.3b) → A.4
    builder.add_edge("context_query_1", "form_hypotheses")
    builder.add_edge("context_query_2", "form_hypotheses")

    # Finish: A.4 → END
    builder.add_edge("form_hypotheses", END)

    return builder.compile()

# Visualize
graph = build_hypothesis_workflow()
display(Image(graph.get_graph().draw_mermaid_png()))

# Invoke
result = graph.invoke({
    "post_text": "I'm feeling nostalgic about my childhood summers.",
    "user_context": {"age": 30, "location": "California"}
})

print(f"Generated {len(result['hypotheses'])} hypotheses")
for i, hyp in enumerate(result['hypotheses'], 1):
    print(f"{i}. {hyp['description']} (confidence: {hyp['confidence']:.2f})")
```

**Flow Diagram**:
```
START → analyze_post → diagnose_gaps ─┬→ context_query_1 ─┐
                                      └→ context_query_2 ─┴→ form_hypotheses → END
```

### Pattern 2: Conditional Parallel Execution

When parallel nodes execute conditionally based on state flags:

```python
from typing import Sequence

def route_context_queries(state: HypothesisWorkflowState) -> Sequence[str]:
    """
    Conditionally route to 0-3 context query types.

    Returns:
        List of node names to execute in parallel.
    """
    questions = state["context_questions"]
    nodes_to_execute = []

    # Check which query types are needed
    has_time_query = any(q["search_type"] == "time_based" for q in questions)
    has_entity_query = any(q["search_type"] == "entity_based" for q in questions)
    has_vector_query = any(q["search_type"] == "vector" for q in questions)

    if has_time_query:
        nodes_to_execute.append("time_based_search")
    if has_entity_query:
        nodes_to_execute.append("entity_based_search")
    if has_vector_query:
        nodes_to_execute.append("vector_search")

    # If no queries needed, skip directly to hypothesis generation
    if not nodes_to_execute:
        return ["form_hypotheses"]

    return nodes_to_execute

def build_conditional_hypothesis_workflow():
    """Build workflow with conditional parallel query execution."""

    builder = StateGraph(HypothesisWorkflowState)

    # Add nodes
    builder.add_node("analyze_post", analyze_post)
    builder.add_node("diagnose_gaps", diagnose_gaps)
    builder.add_node("time_based_search", time_based_search_node)
    builder.add_node("entity_based_search", entity_based_search_node)
    builder.add_node("vector_search", vector_search_node)
    builder.add_node("form_hypotheses", form_hypotheses)

    # Sequential edges
    builder.add_edge(START, "analyze_post")
    builder.add_edge("analyze_post", "diagnose_gaps")

    # Conditional parallel fan-out
    builder.add_conditional_edges(
        "diagnose_gaps",
        route_context_queries,
        [
            "time_based_search",
            "entity_based_search",
            "vector_search",
            "form_hypotheses"  # Direct path if no queries needed
        ]
    )

    # Fan-in: All search nodes converge to hypothesis generation
    builder.add_edge("time_based_search", "form_hypotheses")
    builder.add_edge("entity_based_search", "form_hypotheses")
    builder.add_edge("vector_search", "form_hypotheses")

    # End
    builder.add_edge("form_hypotheses", END)

    return builder.compile()
```

**Key Points**:
- Routing function returns `Sequence[str]` for parallel execution
- All listed nodes in the return value execute in parallel
- Include direct path to merge node if no parallel execution needed
- Each parallel node should update different state fields OR use reducers

### Pattern 3: Dynamic Parallel Execution with Send API

When the number of parallel tasks is unknown until runtime:

```python
from langgraph.types import Send

class DynamicHypothesisState(TypedDict):
    post_text: str
    signals: list[dict]

    # Number of questions determined at runtime
    context_questions: list[str]

    # Aggregated results from N parallel queries
    all_context_results: Annotated[list[dict], operator.add]

    hypotheses: list[dict]

def fan_out_context_queries(state: DynamicHypothesisState) -> list[Send]:
    """
    Create one parallel task per context question.

    Returns:
        List of Send objects, one per question.
    """
    return [
        Send("execute_query", {
            "query_id": i,
            "question": question
        })
        for i, question in enumerate(state["context_questions"])
    ]

def execute_query(state: DynamicHypothesisState) -> dict:
    """
    Worker node: Execute one context query.

    Receives custom state with query_id and question.
    """
    query_id = state["query_id"]
    question = state["question"]

    # Execute search
    results = execute_graph_search(question)

    # Append to aggregated results (reducer merges automatically)
    return {
        "all_context_results": [{
            "query_id": query_id,
            "question": question,
            "results": results
        }]
    }

def build_dynamic_hypothesis_workflow():
    """Build workflow with dynamic parallel query execution."""

    builder = StateGraph(DynamicHypothesisState)

    builder.add_node("analyze_post", analyze_post)
    builder.add_node("diagnose_gaps", diagnose_gaps)
    builder.add_node("execute_query", execute_query)  # Executed N times
    builder.add_node("form_hypotheses", form_hypotheses)

    # Sequential
    builder.add_edge(START, "analyze_post")
    builder.add_edge("analyze_post", "diagnose_gaps")

    # Dynamic fan-out using Send API
    builder.add_conditional_edges(
        "diagnose_gaps",
        fan_out_context_queries,
        ["execute_query"]  # All Send objects target this node
    )

    # Fan-in
    builder.add_edge("execute_query", "form_hypotheses")
    builder.add_edge("form_hypotheses", END)

    return builder.compile()
```

**When to Use Send API**:
- ✅ Number of parallel tasks unknown until runtime
- ✅ Each parallel task needs different input
- ✅ Implementing map-reduce patterns
- ❌ Fixed number of parallel nodes (use Pattern 1 or 2)

---

## Conditional Routing Best Practices

### Routing Function Patterns

#### 1. Binary Routing (Continue or End)

```python
from typing import Literal

def should_continue(state: State) -> Literal["continue", "end"]:
    """Simple binary decision based on state."""
    if len(state["hypotheses"]) >= 2:
        return "end"
    return "continue"

# Wire with mapping
builder.add_conditional_edges(
    "form_hypotheses",
    should_continue,
    {
        "continue": "diagnose_gaps",  # Loop back for more context
        "end": END
    }
)
```

#### 2. Multi-Way Routing (Switch Case)

```python
from typing import Literal

def route_by_confidence(state: State) -> Literal["high", "medium", "low"]:
    """Route based on confidence score."""
    confidence = state["signal_confidence"]

    if confidence > 0.8:
        return "high"
    elif confidence > 0.5:
        return "medium"
    else:
        return "low"

builder.add_conditional_edges(
    "analyze_post",
    route_by_confidence,
    {
        "high": "form_hypotheses",  # Skip gap diagnosis
        "medium": "diagnose_gaps",   # Normal flow
        "low": "request_clarification"  # Need human input
    }
)
```

#### 3. Parallel Fan-Out Routing

```python
from typing import Sequence

def route_to_processors(state: State) -> Sequence[str]:
    """Route to 1-N processors based on flags."""
    processors = []

    if state.get("has_time_signals"):
        processors.append("time_processor")
    if state.get("has_location_signals"):
        processors.append("location_processor")
    if state.get("has_sentiment_signals"):
        processors.append("sentiment_processor")

    # Always have a fallback
    if not processors:
        processors.append("default_processor")

    return processors

# No mapping dict needed - returns list of node names
builder.add_conditional_edges(
    "analyze_post",
    route_to_processors,
    [
        "time_processor",
        "location_processor",
        "sentiment_processor",
        "default_processor"
    ]
)
```

### Routing Best Practices

1. **Type Annotations**: Use `Literal` or `Sequence[str]` for clarity
   ```python
   # ✅ Good - Clear return type
   def route(state: State) -> Literal["a", "b", "c"]:
       ...

   # ❌ Bad - Unclear return type
   def route(state: State):
       return "a"  # What are the other options?
   ```

2. **Always Include All Possible Destinations**:
   ```python
   # ✅ Good - All destinations listed
   builder.add_conditional_edges(
       "router",
       route_function,  # Can return "a", "b", or "c"
       ["a", "b", "c"]
   )

   # ❌ Bad - Missing destination "c"
   builder.add_conditional_edges(
       "router",
       route_function,
       ["a", "b"]  # Will error if route_function returns "c"
   )
   ```

3. **Handle Empty Route Lists**:
   ```python
   def route_parallel(state: State) -> Sequence[str]:
       nodes = []
       if state["condition_a"]:
           nodes.append("node_a")
       if state["condition_b"]:
           nodes.append("node_b")

       # Always provide fallback
       if not nodes:
           return ["skip_node"]  # Or route directly to next phase

       return nodes
   ```

4. **Document Routing Logic**:
   ```python
   def route_context_queries(state: State) -> Sequence[str]:
       """
       Route to specialized search nodes based on question types.

       Returns:
           - ["time_search"] if questions are time-based
           - ["entity_search"] if questions are entity-based
           - ["time_search", "entity_search"] if both types present
           - ["form_hypotheses"] if no queries needed (direct path)
       """
       ...
   ```

5. **Use Mapping Dicts for Clarity**:
   ```python
   # ✅ Good - Explicit mapping
   builder.add_conditional_edges(
       "analyzer",
       route_function,
       {
           "needs_context": "diagnose_gaps",
           "ready": "form_hypotheses",
           "unclear": "request_clarification"
       }
   )

   # ❌ Less clear - Implicit mapping
   builder.add_conditional_edges(
       "analyzer",
       route_function  # Returns node name directly
   )
   ```

---

## State Management Best Practices

### 1. Reducer Selection

**When to Use Reducers**:

```python
# ✅ Use reducer when parallel nodes update same field
class State(TypedDict):
    results: Annotated[list[dict], operator.add]  # Multiple nodes append

def parallel_node_1(state: State):
    return {"results": [{"source": "node1", "data": "..."}]}

def parallel_node_2(state: State):
    return {"results": [{"source": "node2", "data": "..."}]}

# After parallel execution:
# state["results"] = [{"source": "node1", ...}, {"source": "node2", ...}]
```

**When NOT to Use Reducers**:

```python
# ✅ Separate fields - no reducer needed
class State(TypedDict):
    node1_result: dict  # Only node1 updates this
    node2_result: dict  # Only node2 updates this

def parallel_node_1(state: State):
    return {"node1_result": {"data": "..."}}

def parallel_node_2(state: State):
    return {"node2_result": {"data": "..."}}
```

**Common Reducers**:

| Reducer | Behavior | Use Case |
|---------|----------|----------|
| `operator.add` | Concatenate lists/strings | Aggregating parallel results |
| `operator.or_` | Logical OR | Any worker sets flag to True |
| `lambda x, y: x \| y` | Merge dicts | Combine dictionaries |
| Custom function | Custom logic | Complex merging scenarios |

### 2. Optional vs Required Fields

```python
from typing import Optional

class State(TypedDict):
    # Required fields (always present)
    post_text: str
    signals: dict

    # Optional fields (may not be set)
    query_1_results: Optional[dict]  # Only if query 1 executes
    query_2_results: Optional[dict]  # Only if query 2 executes

def merge_node(state: State):
    # Safe access with .get()
    q1 = state.get("query_1_results")
    q2 = state.get("query_2_results")

    if q1 and q2:
        # Both queries completed
        return {"combined": merge_both(q1, q2)}
    elif q1 or q2:
        # Only one completed
        return {"combined": q1 or q2}
    else:
        # Neither completed
        return {"combined": {}}
```

### 3. State Field Naming Conventions

```python
# ✅ Good - Clear phase-based naming
class State(TypedDict):
    # Phase identifier in field name
    phase1_extracted_signals: dict
    phase2_context_questions: list[str]
    phase3_query_results: list[dict]
    phase4_hypotheses: list[dict]

# ✅ Good - Descriptive worker outputs
class State(TypedDict):
    time_based_results: dict
    entity_based_results: dict
    vector_search_results: dict

# ❌ Bad - Ambiguous naming
class State(TypedDict):
    data1: dict
    data2: dict
    output: dict
```

### 4. State Validation Strategies

**Option A: Pydantic State (Runtime Validation)**:
```python
from pydantic import BaseModel, Field, validator

class State(BaseModel):
    context_questions: list[str] = Field(min_items=2, max_items=2)
    hypotheses: list[dict] = Field(min_items=2, max_items=2)

    @validator('hypotheses')
    def validate_hypotheses(cls, v):
        for h in v:
            if 'confidence' not in h:
                raise ValueError("Each hypothesis must have confidence")
        return v
```

**Option B: Manual Validation in Nodes**:
```python
def diagnose_gaps(state: State) -> dict:
    """Generate exactly 2 context questions."""
    gaps = identify_missing_context(state["signals"])
    questions = generate_questions(gaps)

    # Validate before returning
    if len(questions) != 2:
        raise ValueError(f"Expected 2 questions, got {len(questions)}")

    return {"context_questions": questions}
```

**Option C: Validation Node**:
```python
def validate_hypotheses(state: State) -> dict:
    """Validation node between phases."""
    hypotheses = state["hypotheses"]

    if len(hypotheses) != 2:
        raise ValueError("Must have exactly 2 hypotheses")

    for h in hypotheses:
        if h["confidence"] < 0.5:
            raise ValueError(f"Hypothesis confidence too low: {h['confidence']}")

    return {}  # No state update, just validation

# Add validation node in graph
builder.add_edge("form_hypotheses", "validate_hypotheses")
builder.add_edge("validate_hypotheses", END)
```

---

## Complete Example: Hypothesis Building Workflow

Here's a production-ready implementation:

```python
import operator
from typing import Annotated, Optional, Sequence, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from datetime import datetime

# ============================================================================
# STATE DEFINITION
# ============================================================================

class HypothesisWorkflowState(TypedDict):
    """State for hypothesis building workflow with 4 sequential phases."""

    # Input
    post_text: str
    user_id: str
    user_context: dict

    # Phase A.1: Analyze Post
    signals: dict
    signal_types: list[str]
    confidence: float

    # Phase A.2: Diagnose Gaps
    identified_gaps: list[str]
    context_questions: list[dict]  # [{"text": str, "type": str, "priority": int}]

    # Phase A.3: Context Search (parallel)
    query_1_results: Optional[dict]
    query_2_results: Optional[dict]
    search_errors: Annotated[list[str], operator.add]  # Accumulate errors

    # Phase A.4: Form Hypotheses
    hypotheses: list[dict]  # [{"id": int, "description": str, "confidence": float}]

    # Metadata
    workflow_start_time: str
    workflow_end_time: Optional[str]

# ============================================================================
# NODE IMPLEMENTATIONS
# ============================================================================

def analyze_post(state: HypothesisWorkflowState) -> dict:
    """
    Phase A.1: Extract signals from post text.

    Analyzes post to identify:
    - Emotional signals (sentiment, mood)
    - Temporal signals (time references, recency)
    - Entity signals (people, places, objects)
    - Intent signals (desire, nostalgia, planning)
    """
    post_text = state["post_text"]

    # Use DSPy module or LLM for signal extraction
    signals = {
        "emotional": extract_emotional_signals(post_text),
        "temporal": extract_temporal_signals(post_text),
        "entities": extract_entity_signals(post_text),
        "intent": extract_intent_signals(post_text)
    }

    signal_types = [k for k, v in signals.items() if v]
    confidence = calculate_overall_confidence(signals)

    return {
        "signals": signals,
        "signal_types": signal_types,
        "confidence": confidence,
        "workflow_start_time": datetime.now().isoformat()
    }

def diagnose_gaps(state: HypothesisWorkflowState) -> dict:
    """
    Phase A.2: Identify missing context, generate 2 questions.

    Analyzes signals and user context to determine:
    - What information is missing
    - What questions would fill those gaps
    - Priority and type of each question

    Returns exactly 2 context questions.
    """
    signals = state["signals"]
    user_context = state["user_context"]

    # Identify what's missing
    gaps = []

    if signals["temporal"] and not user_context.get("timeline"):
        gaps.append("missing_temporal_context")

    if signals["entities"] and not user_context.get("relationships"):
        gaps.append("missing_entity_relationships")

    if signals["intent"] and not user_context.get("past_behaviors"):
        gaps.append("missing_behavioral_history")

    # Generate 2 highest priority questions
    questions = generate_context_questions(
        signals=signals,
        gaps=gaps,
        user_context=user_context,
        limit=2
    )

    # Validate
    if len(questions) != 2:
        raise ValueError(f"Expected 2 questions, generated {len(questions)}")

    return {
        "identified_gaps": gaps,
        "context_questions": questions
    }

def context_search_query_1(state: HypothesisWorkflowState) -> dict:
    """
    Phase A.3a: Execute first context search query.

    Runs in parallel with query 2.
    """
    try:
        question = state["context_questions"][0]

        # Execute knowledge graph search
        results = execute_graph_search(
            query=question["text"],
            search_type=question["type"],
            user_id=state["user_id"]
        )

        return {
            "query_1_results": {
                "question": question,
                "results": results,
                "result_count": len(results),
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        return {
            "search_errors": [f"Query 1 failed: {str(e)}"]
        }

def context_search_query_2(state: HypothesisWorkflowState) -> dict:
    """
    Phase A.3b: Execute second context search query.

    Runs in parallel with query 1.
    """
    try:
        question = state["context_questions"][1]

        # Execute knowledge graph search
        results = execute_graph_search(
            query=question["text"],
            search_type=question["type"],
            user_id=state["user_id"]
        )

        return {
            "query_2_results": {
                "question": question,
                "results": results,
                "result_count": len(results),
                "timestamp": datetime.now().isoformat()
            }
        }
    except Exception as e:
        return {
            "search_errors": [f"Query 2 failed: {str(e)}"]
        }

def form_hypotheses(state: HypothesisWorkflowState) -> dict:
    """
    Phase A.4: Generate 2 diverse hypotheses.

    Synthesizes:
    - Original post signals
    - Identified gaps
    - Context from parallel queries

    Generates 2 hypotheses that are maximally diverse in their
    interpretations while being grounded in the available evidence.
    """
    signals = state["signals"]
    gaps = state["identified_gaps"]

    # Safely access parallel query results
    query_1 = state.get("query_1_results", {})
    query_2 = state.get("query_2_results", {})
    errors = state.get("search_errors", [])

    # Check if we have sufficient context
    if not query_1 and not query_2:
        # Both queries failed - use only signals
        context = {"source": "signals_only", "data": signals}
    else:
        # Merge available context
        context = merge_context_results(query_1, query_2)

    # Generate 2 diverse hypotheses
    hypotheses = generate_diverse_hypotheses(
        signals=signals,
        gaps=gaps,
        context=context,
        num_hypotheses=2,
        diversity_threshold=0.7  # Ensure hypotheses are distinct
    )

    # Validate
    if len(hypotheses) != 2:
        raise ValueError(f"Expected 2 hypotheses, generated {len(hypotheses)}")

    return {
        "hypotheses": hypotheses,
        "workflow_end_time": datetime.now().isoformat()
    }

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_hypothesis_workflow():
    """
    Build the complete hypothesis generation workflow.

    Flow:
        START → analyze_post → diagnose_gaps
                                    ├─→ context_search_query_1 ─┐
                                    └─→ context_search_query_2 ─┴→ form_hypotheses → END

    Returns:
        Compiled LangGraph workflow.
    """
    builder = StateGraph(HypothesisWorkflowState)

    # Add all nodes
    builder.add_node("analyze_post", analyze_post)
    builder.add_node("diagnose_gaps", diagnose_gaps)
    builder.add_node("context_search_query_1", context_search_query_1)
    builder.add_node("context_search_query_2", context_search_query_2)
    builder.add_node("form_hypotheses", form_hypotheses)

    # Sequential: START → Phase A.1 → Phase A.2
    builder.add_edge(START, "analyze_post")
    builder.add_edge("analyze_post", "diagnose_gaps")

    # Parallel: Phase A.2 → (Phase A.3a || Phase A.3b)
    builder.add_edge("diagnose_gaps", "context_search_query_1")
    builder.add_edge("diagnose_gaps", "context_search_query_2")

    # Fan-in: (Phase A.3a || Phase A.3b) → Phase A.4
    builder.add_edge("context_search_query_1", "form_hypotheses")
    builder.add_edge("context_search_query_2", "form_hypotheses")

    # Finish: Phase A.4 → END
    builder.add_edge("form_hypotheses", END)

    return builder.compile()

# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    # Build workflow
    workflow = build_hypothesis_workflow()

    # Visualize
    from IPython.display import Image, display
    display(Image(workflow.get_graph().draw_mermaid_png()))

    # Execute
    result = workflow.invoke({
        "post_text": "I miss the summers at grandma's lake house.",
        "user_id": "user_123",
        "user_context": {
            "age": 30,
            "location": "California",
            "interests": ["family", "nostalgia", "travel"]
        }
    })

    # Display results
    print(f"\n{'='*60}")
    print("HYPOTHESIS GENERATION RESULTS")
    print(f"{'='*60}\n")

    print(f"Post: {result['post_text']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"\nSignal Types: {', '.join(result['signal_types'])}")
    print(f"Identified Gaps: {', '.join(result['identified_gaps'])}")

    if result.get('search_errors'):
        print(f"\n⚠️  Search Errors: {', '.join(result['search_errors'])}")

    print(f"\n{'='*60}")
    print("GENERATED HYPOTHESES")
    print(f"{'='*60}\n")

    for i, hyp in enumerate(result['hypotheses'], 1):
        print(f"{i}. {hyp['description']}")
        print(f"   Confidence: {hyp['confidence']:.2f}")
        if 'supporting_evidence' in hyp:
            print(f"   Evidence: {', '.join(hyp['supporting_evidence'][:3])}")
        print()

    # Timing
    start = datetime.fromisoformat(result['workflow_start_time'])
    end = datetime.fromisoformat(result['workflow_end_time'])
    duration = (end - start).total_seconds()
    print(f"Workflow Duration: {duration:.2f} seconds")
```

---

## Advanced Patterns

### Retry and Error Handling

```python
from typing import Literal

class StateWithRetry(TypedDict):
    signals: dict
    hypotheses: list[dict]
    retry_count: int
    max_retries: int
    errors: Annotated[list[str], operator.add]

def form_hypotheses_with_retry(state: StateWithRetry) -> dict:
    """Generate hypotheses with error tracking."""
    try:
        hypotheses = generate_hypotheses(state["signals"])
        return {"hypotheses": hypotheses}
    except Exception as e:
        return {
            "errors": [f"Attempt {state['retry_count']}: {str(e)}"],
            "retry_count": state["retry_count"] + 1
        }

def should_retry(state: StateWithRetry) -> Literal["retry", "fail", "success"]:
    """Determine if we should retry hypothesis generation."""
    if state.get("hypotheses"):
        return "success"
    elif state["retry_count"] >= state["max_retries"]:
        return "fail"
    else:
        return "retry"

# Wire with retry loop
builder.add_conditional_edges(
    "form_hypotheses",
    should_retry,
    {
        "success": END,
        "retry": "form_hypotheses",  # Loop back
        "fail": "handle_failure"
    }
)
```

### Human-in-the-Loop for Low Confidence

```python
from langgraph.checkpoint.postgres import PostgresSaver

def route_by_confidence(state: State) -> Literal["auto", "review"]:
    """Route to human review if confidence is low."""
    if state["confidence"] < 0.6:
        return "review"
    return "auto"

# Build with checkpointing for interruption
checkpointer = PostgresSaver.from_conn_string(SUPABASE_DB_URI)

builder.add_conditional_edges(
    "form_hypotheses",
    route_by_confidence,
    {
        "auto": END,
        "review": "human_review"  # Interrupt for human input
    }
)

graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["human_review"]  # Pause execution
)

# Execute with thread for resumption
config = {"configurable": {"thread_id": "hypothesis_workflow_123"}}
result = graph.invoke(input_state, config)

# Later, after human review
graph.invoke(None, config)  # Resume from checkpoint
```

### Progressive Refinement Loop

```python
class RefinementState(TypedDict):
    post_text: str
    hypotheses: list[dict]
    refinement_iterations: int
    max_iterations: int
    quality_score: float

def refine_hypotheses(state: RefinementState) -> dict:
    """Improve hypothesis quality iteratively."""
    current_hypotheses = state["hypotheses"]

    refined = improve_hypotheses(current_hypotheses)
    quality = evaluate_quality(refined)

    return {
        "hypotheses": refined,
        "refinement_iterations": state["refinement_iterations"] + 1,
        "quality_score": quality
    }

def should_refine(state: RefinementState) -> Literal["refine", "done"]:
    """Continue refining until quality threshold or max iterations."""
    if state["quality_score"] >= 0.9:
        return "done"
    elif state["refinement_iterations"] >= state["max_iterations"]:
        return "done"
    else:
        return "refine"

builder.add_conditional_edges(
    "form_hypotheses",
    should_refine,
    {
        "done": END,
        "refine": "refine_hypotheses"
    }
)
builder.add_edge("refine_hypotheses", "form_hypotheses")  # Loop back
```

---

## Testing Strategies

### Unit Test Individual Nodes

```python
def test_analyze_post():
    """Test signal extraction from post."""
    state = {
        "post_text": "I miss my childhood dog.",
        "user_context": {}
    }

    result = analyze_post(state)

    assert "signals" in result
    assert "emotional" in result["signals"]
    assert result["confidence"] > 0
    assert len(result["signal_types"]) > 0

def test_diagnose_gaps():
    """Test gap identification and question generation."""
    state = {
        "signals": {
            "emotional": ["nostalgia", "sadness"],
            "temporal": ["childhood"],
            "entities": ["dog"],
            "intent": ["remembering"]
        },
        "user_context": {}
    }

    result = diagnose_gaps(state)

    assert "context_questions" in result
    assert len(result["context_questions"]) == 2
    for q in result["context_questions"]:
        assert "text" in q
        assert "type" in q
```

### Integration Test Full Workflow

```python
def test_hypothesis_workflow_end_to_end():
    """Test complete workflow execution."""
    workflow = build_hypothesis_workflow()

    input_state = {
        "post_text": "I'm feeling nostalgic about summers at the lake.",
        "user_id": "test_user",
        "user_context": {"age": 30, "location": "California"}
    }

    result = workflow.invoke(input_state)

    # Validate state progression
    assert "signals" in result
    assert "context_questions" in result
    assert len(result["context_questions"]) == 2
    assert "hypotheses" in result
    assert len(result["hypotheses"]) == 2

    # Validate hypotheses
    for hyp in result["hypotheses"]:
        assert "description" in hyp
        assert "confidence" in hyp
        assert 0 <= hyp["confidence"] <= 1

    # Validate timing
    assert "workflow_start_time" in result
    assert "workflow_end_time" in result

def test_parallel_execution_timing():
    """Verify queries execute in parallel, not sequentially."""
    import time

    workflow = build_hypothesis_workflow()

    start = time.time()
    result = workflow.invoke(test_input)
    duration = time.time() - start

    # If queries are parallel, total time should be ~max(query1, query2)
    # not sum(query1 + query2)
    # Assuming each query takes ~1 second, parallel should be <2s
    # Sequential would be ~2s
    assert duration < 1.5  # Allow some overhead
```

### Test Conditional Routing

```python
def test_conditional_parallel_routing():
    """Test that routing function selects correct parallel nodes."""

    # Test case 1: Time-based query only
    state1 = {
        "context_questions": [
            {"text": "When did this happen?", "type": "time_based"}
        ]
    }
    result1 = route_context_queries(state1)
    assert result1 == ["time_based_search"]

    # Test case 2: Both query types
    state2 = {
        "context_questions": [
            {"text": "When?", "type": "time_based"},
            {"text": "Who?", "type": "entity_based"}
        ]
    }
    result2 = route_context_queries(state2)
    assert set(result2) == {"time_based_search", "entity_based_search"}

    # Test case 3: No queries (direct path)
    state3 = {"context_questions": []}
    result3 = route_context_queries(state3)
    assert result3 == ["form_hypotheses"]
```

---

## Performance Optimization

### 1. Minimize State Size

```python
# ❌ Bad - Storing large intermediate data
class State(TypedDict):
    raw_llm_outputs: list[str]  # Can be huge
    all_intermediate_results: list[dict]  # Memory intensive

# ✅ Good - Store only essential data
class State(TypedDict):
    extracted_signals: dict  # Compact summary
    hypothesis_ids: list[int]  # Reference, not full data
```

### 2. Parallel Execution for Independent Operations

```python
# ✅ Optimal - Parallel LLM calls
builder.add_edge("diagnose_gaps", "query_1")
builder.add_edge("diagnose_gaps", "query_2")
builder.add_edge("diagnose_gaps", "query_3")  # All execute concurrently

# ❌ Suboptimal - Sequential LLM calls
builder.add_edge("diagnose_gaps", "query_1")
builder.add_edge("query_1", "query_2")
builder.add_edge("query_2", "query_3")  # Slower by 3x
```

### 3. Use TypedDict for Performance

```python
# ✅ Fastest - TypedDict
class State(TypedDict):
    signals: dict
    hypotheses: list[dict]

# ❌ Slower - Pydantic (2-5x overhead)
class State(BaseModel):
    signals: dict
    hypotheses: list[dict]
```

### 4. Batch LLM Calls When Possible

```python
# ✅ Better - Batch processing
def extract_multiple_signals(state: State) -> dict:
    texts = [state["post_text"], state["user_bio"], state["recent_posts"]]

    # Single batched LLM call
    all_signals = llm.batch([
        {"text": t} for t in texts
    ])

    return {"all_signals": all_signals}

# ❌ Less efficient - Sequential calls
def extract_signals_sequentially(state: State) -> dict:
    signal1 = llm.invoke(state["post_text"])
    signal2 = llm.invoke(state["user_bio"])
    signal3 = llm.invoke(state["recent_posts"])
    ...
```

---

## Common Pitfalls and Solutions

### 1. Parallel Nodes Updating Same Field Without Reducer

**Problem**:
```python
# ❌ Error - No reducer defined
class State(TypedDict):
    results: list[dict]  # Multiple nodes try to update this

def query_1(state): return {"results": [{"data": "q1"}]}
def query_2(state): return {"results": [{"data": "q2"}]}

# Runtime error: INVALID_CONCURRENT_GRAPH_UPDATE
```

**Solution**:
```python
# ✅ Add reducer
class State(TypedDict):
    results: Annotated[list[dict], operator.add]  # Reducer merges updates
```

### 2. Accessing Nonexistent Parallel Results

**Problem**:
```python
# ❌ KeyError if query didn't execute
def merge(state):
    result = state["query_1_results"]  # Might not exist!
```

**Solution**:
```python
# ✅ Use .get() with default
def merge(state):
    result = state.get("query_1_results", {})
    if not result:
        # Handle missing result
        ...
```

### 3. Forgetting to List All Conditional Destinations

**Problem**:
```python
# ❌ Missing destination
builder.add_conditional_edges(
    "router",
    route_function,  # Can return "a", "b", or "c"
    ["a", "b"]  # Missing "c" - runtime error!
)
```

**Solution**:
```python
# ✅ List all possible destinations
builder.add_conditional_edges(
    "router",
    route_function,
    ["a", "b", "c"]  # All destinations
)
```

### 4. Not Handling Empty Parallel Route Lists

**Problem**:
```python
# ❌ Can return empty list
def route(state) -> Sequence[str]:
    nodes = []
    if state["flag"]:
        nodes.append("node_a")
    return nodes  # Empty if flag is False!
```

**Solution**:
```python
# ✅ Always provide fallback
def route(state) -> Sequence[str]:
    nodes = []
    if state["flag"]:
        nodes.append("node_a")

    if not nodes:
        return ["default_node"]  # Fallback
    return nodes
```

### 5. Graph Output is Dict, Not Pydantic

**Problem**:
```python
# ❌ Expecting Pydantic instance
result = graph.invoke(input)
result.hypotheses  # AttributeError! result is dict
```

**Solution**:
```python
# ✅ Convert back to Pydantic if needed
result = graph.invoke(input)
state_obj = State(**result)
state_obj.hypotheses  # Now works
```

---

## Related Documentation

- [LangGraph Conditional Parallel Execution](./langgraph-conditional-parallel-execution.md) - Detailed parallel execution patterns
- [LangGraph State Serialization with Pydantic](./langgraph-state-serialization-pydantic.md) - State schema options
- [LangGraph Human-in-the-Loop Complete Guide](./langgraph-human-in-the-loop-complete-guide.md) - Checkpointing and interruption
- [LangGraph Observability, Tracing, and MLflow Integration](./langgraph-observability-tracing-mlflow.md) - Monitoring workflows

---

## Summary

**Key Takeaways**:

1. **State Design**: Use phase-based field naming, TypedDict for performance, Pydantic for validation
2. **Sequential Nodes**: Each node returns partial state update, receives full state
3. **Parallel Execution**: Use reducers when nodes update same field, separate fields otherwise
4. **Conditional Routing**: Return `Sequence[str]` for parallel, `Literal` for single, `list[Send]` for dynamic
5. **Fan-In Pattern**: Merge node uses `.get()` to safely access optional parallel results
6. **Best Practices**: Type hints, docstrings, error handling, fallback paths
7. **Testing**: Unit test nodes, integration test full workflow, verify parallelism timing

**Quick Reference**:

| Pattern | Use Case | Return Type |
|---------|----------|-------------|
| Sequential | A → B → C | Regular edges |
| Fixed Parallel | A → (B \|\| C) → D | `add_edge` to both |
| Conditional Parallel | A → (B? \|\| C? \|\| D?) → E | `Sequence[str]` |
| Dynamic Parallel | A → (N workers) → B | `list[Send]` |
| Binary Route | if/else decision | `Literal["a", "b"]` |
| Multi-way Route | switch case | `Literal["a", "b", "c"]` |
