# MLflow LangGraph Tracing: Unified Graph Traces vs Separate Node Traces

**Source**: MLflow Official Documentation, GitHub Issues #16880, #18216
**Date Retrieved**: 2026-01-06
**MLflow Version Coverage**: 2.14.0 - 2.21.3+
**LangGraph Version**: Compatible with LangChain 0.3.12 - 1.1.3

## Summary

MLflow provides automatic tracing for LangGraph through `mlflow.langchain.autolog()`, leveraging LangChain's callback framework. However, there are **known bugs** that cause LangGraph workflows to create separate traces for each node instead of a unified hierarchical trace. This document explains the root cause, known issues, workarounds, and best practices for achieving proper trace hierarchy in LangGraph workflows.

---

## Key Concepts

### How MLflow Traces LangGraph

1. **Integration Method**: MLflow traces LangGraph as an extension of its LangChain integration
2. **Mechanism**: Uses LangChain's callback framework via `MlflowLangchainTracer`
3. **Activation**: Enabled by calling `mlflow.langchain.autolog()`
4. **Scope**: Only traces are logged for LangGraph (not models or other artifacts)
5. **Runnable Interface**: LangGraph implements LangChain's Runnable interface, making it compatible with autolog

### Trace Structure Components

**Spans**: Individual units of work within a trace
- **Root Span**: Top-level span representing the entire graph execution
- **Child Spans**: Nested spans for individual nodes, LLM calls, tool executions

**Traces**: Collection of related spans forming a complete execution path
- **Unified Trace**: Single trace with hierarchical spans (desired)
- **Separate Traces**: Multiple disconnected root-level traces (problematic)

---

## Known Issues and Bugs

### Issue #18216: Combining Manual and Automatic Tracing

**Problem**: When using `mlflow.langchain.autolog()` with manual `@mlflow.trace` decorators on LangGraph agents, traces appear as separate root-level traces instead of nested spans.

**Expected Behavior**:
```
Trace (Root)
  └─ Manual @mlflow.trace span (Parent)
      ├─ Graph execution span
      │   ├─ Node 1 span
      │   ├─ Node 2 span
      │   └─ Node 3 span
      └─ Other manual spans
```

**Actual Behavior**:
```
Trace 1 (Root) - Manual @mlflow.trace span
Trace 2 (Root) - Graph execution (disconnected)
  ├─ Node 1 span
  ├─ Node 2 span
  └─ Node 3 span
```

**Root Cause**: The parent span isn't accessible in the context that executes the node in the graph.

**Impact**:
- Breaks logical flow visualization
- Prevents collection of user feedback on traces
- Prevents updates to trace metadata
- Loss of end-to-end observability

**Status**: Open bug (as of October 2024)

**GitHub Issue**: [#18216](https://github.com/mlflow/mlflow/issues/18216)

---

### Issue #16880: Async Trace Hierarchy Flattening

**Problem**: When using `mlflow.langchain.autolog()` with manual `@mlflow.trace` decorators on **async functions**, the hierarchical trace structure is flattened into separate root-level traces.

**Specific Context**:
- Affects async methods: `ainvoke()`, `abatch()`, `astream()`
- Synchronous versions (`invoke()`, `batch()`, `stream()`) work correctly
- Problem relates to Python's `ContextVar` mechanism not propagating across async boundaries

**Root Cause**: Context variable propagation failure in `MlflowLangchainTracer._get_parent_span()` during async execution.

**Impact**:
- Manual traces and autolog traces appear disconnected
- Impossible to follow complete execution flow of a single request
- Loss of parent-child relationships in async workflows

**Status**: Open bug (as of July 2024)

**GitHub Issue**: [#16880](https://github.com/mlflow/mlflow/issues/16880)

---

## Critical Configuration: `run_tracer_inline`

### What It Does

The `run_tracer_inline` parameter controls whether the MLflow tracer callback runs in the main async task or is offloaded to a thread pool.

```python
mlflow.langchain.autolog(run_tracer_inline=True)
```

### When to Use

**REQUIRED for**:
- Async graph methods (`ainvoke()`, `astream()`, `abatch()`)
- Combining autolog with manual `@mlflow.trace` decorators
- Workflows using async nodes or tools
- Proper context propagation in async scenarios

**Benefits**:
- Ensures proper context propagation across async boundaries
- Maintains parent-child span relationships
- Prevents flattening of trace hierarchy

### Performance Trade-off

While `run_tracer_inline=True` ensures correct tracing:
- The logging operation is **not asynchronous** and may block the main thread
- The invocation function itself is still non-blocking and returns a coroutine
- Logging overhead may slow down execution slightly

---

## Working Approaches and Workarounds

### Approach 1: Autolog Only (Simplest)

Use only autolog without manual tracing decorators.

```python
import mlflow
from langgraph.graph import StateGraph

# Enable autolog
mlflow.langchain.autolog()

# For async workflows, use run_tracer_inline
# mlflow.langchain.autolog(run_tracer_inline=True)

# Build and compile graph
builder = StateGraph(State)
builder.add_node("node1", node1_func)
builder.add_node("node2", node2_func)
# ... add more nodes
graph = builder.compile()

# Invoke - automatically traced
result = graph.invoke(inputs, config={"configurable": {"thread_id": "123"}})
```

**Pros**:
- Simple setup
- Automatic trace creation
- No manual instrumentation needed

**Cons**:
- Limited granularity within nodes
- Can't add custom metadata to graph-level trace
- No control over trace naming

---

### Approach 2: Manual Spans Inside Nodes (Recommended)

Add manual child spans **inside** node functions for detailed insights.

```python
import mlflow
from mlflow.types.core import SpanType
from langgraph.graph import StateGraph

# Enable autolog with inline tracing for async
mlflow.langchain.autolog(run_tracer_inline=True)

def composition_analysis_node(state: State):
    """Analyze composition with detailed tracing."""

    # Manual child span for entity extraction
    with mlflow.start_span(name="extract_entities", span_type=SpanType.TOOL) as span:
        span.set_inputs({"text": state["composition_text"]})
        entities = extract_entities_logic(state["composition_text"])
        span.set_outputs({"entities": entities})
        span.set_attribute("entity_count", len(entities))

    # Manual child span for emotion analysis
    with mlflow.start_span(name="analyze_emotions", span_type=SpanType.TOOL) as span:
        span.set_inputs({"text": state["composition_text"]})
        emotions = analyze_emotions_logic(state["composition_text"])
        span.set_outputs({"emotions": emotions})

    return {"entities": entities, "emotions": emotions}

# Build graph
builder = StateGraph(State)
builder.add_node("analyze", composition_analysis_node)
graph = builder.compile()

# Invoke - autolog creates parent trace, manual spans are children
result = graph.invoke(inputs, config={"configurable": {"thread_id": "123"}})
```

**Trace Structure**:
```
Graph Execution (autolog root span)
  └─ analyze node
      ├─ extract_entities (manual span)
      └─ analyze_emotions (manual span)
```

**Pros**:
- Detailed granularity within nodes
- Works with current MLflow versions
- Proper parent-child relationships
- Custom metadata and attributes

**Cons**:
- Requires manual instrumentation in each node
- More verbose code

---

### Approach 3: Custom Stream Mode Integration

Use LangGraph's custom stream mode to emit MLflow events.

```python
import mlflow
import time
from langgraph.config import get_stream_writer

def tracked_node(state: State):
    """Node with custom MLflow event streaming."""
    writer = get_stream_writer()

    # Emit MLflow tracking events
    writer({
        "mlflow_event": "node_start",
        "node_name": "tracked_node",
        "timestamp": time.time()
    })

    # Do work
    result = perform_analysis(state)

    # Emit metrics
    writer({
        "mlflow_event": "log_metric",
        "metric_name": "items_processed",
        "metric_value": len(result)
    })

    writer({
        "mlflow_event": "node_end",
        "node_name": "tracked_node",
        "duration": time.time() - start_time
    })

    return result

# Consumer side
with mlflow.start_run(run_name="langgraph_execution"):
    for chunk in graph.stream(
        inputs,
        config,
        stream_mode=["updates", "custom"]
    ):
        mode, data = chunk

        if mode == "custom" and data.get("mlflow_event") == "log_metric":
            mlflow.log_metric(data["metric_name"], data["metric_value"])
```

**Pros**:
- Fine-grained control over what's logged
- Works around autolog limitations
- Can integrate with existing streaming patterns

**Cons**:
- Most complex approach
- Requires consumer to process events
- More boilerplate code

---

## Non-Working Approaches (Known Bugs)

### ❌ Wrapping graph.invoke() with @mlflow.trace

**Attempted Code**:
```python
import mlflow

mlflow.langchain.autolog()

@mlflow.trace(name="composition_analysis_graph")
def run_analysis(graph, inputs, config):
    """Wrapper to create parent span for graph execution."""
    return graph.invoke(inputs, config)

# This creates SEPARATE traces, not nested!
result = run_analysis(graph, inputs, config)
```

**Expected**: Single trace with wrapper as parent, graph execution as child
**Actual**: Two separate root-level traces

**Why It Fails**: See Issue #18216 - parent span context is not accessible when nodes execute

---

### ❌ Using @mlflow.trace on Async Graph Methods Without run_tracer_inline

**Attempted Code**:
```python
import mlflow

mlflow.langchain.autolog()  # Missing run_tracer_inline=True!

@mlflow.trace
async def run_async_analysis(graph, inputs, config):
    return await graph.ainvoke(inputs, config)

# This flattens the trace hierarchy!
result = await run_async_analysis(graph, inputs, config)
```

**Expected**: Hierarchical trace structure
**Actual**: Flattened separate traces

**Why It Fails**: See Issue #16880 - context variable propagation fails in async scenarios

**Fix**: Add `run_tracer_inline=True`
```python
mlflow.langchain.autolog(run_tracer_inline=True)
```

---

## Best Practices

### 1. Configuration

**For Sync Workflows**:
```python
mlflow.langchain.autolog()
```

**For Async Workflows** (REQUIRED):
```python
mlflow.langchain.autolog(run_tracer_inline=True)
```

**For Background Initialization** (as in the project):
```python
def enable_mlflow_autolog():
    """Enable autolog in background thread."""
    import mlflow
    mlflow.langchain.autolog(run_tracer_inline=True)  # Critical for async!

# In main.py lifespan
import threading
autolog_thread = threading.Thread(target=enable_mlflow_autolog)
autolog_thread.start()
```

### 2. Graph Invocation

**Use Descriptive Config**:
```python
config = {
    "run_name": "composition_analysis",
    "configurable": {
        "thread_id": f"user-{user_id}",
    },
    "tags": ["production", "composition", f"user-{user_id}"],
    "metadata": {
        "user_id": user_id,
        "composition_id": composition_id,
        "environment": "production",
        "version": "0.4.0"
    }
}

result = graph.invoke(inputs, config)
```

**Benefits**:
- Easier trace identification in MLflow UI
- Filterable by tags and metadata
- Contextual information for debugging

### 3. Node-Level Tracing

**Add Manual Spans for Complex Operations**:
```python
def complex_node(state: State):
    """Node with multiple distinct operations."""

    # Operation 1
    with mlflow.start_span(name="database_query", span_type=SpanType.TOOL) as span:
        span.set_inputs({"query": state["query"]})
        data = query_database(state["query"])
        span.set_outputs({"rows": len(data)})

    # Operation 2
    with mlflow.start_span(name="transform_data", span_type=SpanType.TOOL) as span:
        span.set_inputs({"data_size": len(data)})
        transformed = transform(data)
        span.set_outputs({"transformed_size": len(transformed)})

    return {"data": transformed}
```

### 4. Error Handling

**Spans Automatically Capture Exceptions**:
```python
def node_with_error_handling(state: State):
    with mlflow.start_span(name="risky_operation", span_type=SpanType.TOOL) as span:
        span.set_inputs(state)
        try:
            result = risky_function(state)
            span.set_outputs(result)
            return result
        except Exception as e:
            # MLflow automatically sets span status to ERROR
            # and records exception details
            span.set_attribute("error_type", type(e).__name__)
            span.set_attribute("error_message", str(e))
            raise
```

### 5. Avoid Common Pitfalls

**❌ Don't**: Nest `@mlflow.trace` decorators with autolog
**✅ Do**: Use manual spans inside nodes

**❌ Don't**: Use autolog for async without `run_tracer_inline=True`
**✅ Do**: Always enable inline tracing for async workflows

**❌ Don't**: Try to wrap entire graph execution with decorators
**✅ Do**: Rely on autolog for graph-level tracing

**❌ Don't**: Create manual root traces when autolog is enabled
**✅ Do**: Create child spans within autolog traces

---

## Comparison: LangSmith vs MLflow Tracing

### LangSmith (Native LangGraph Observability)

**Pros**:
- Native integration, no configuration needed
- Automatic hierarchical tracing (works correctly)
- Purpose-built for LangChain/LangGraph
- Visual graph execution debugging
- Thread and checkpoint history

**Cons**:
- Requires LangSmith account
- Data sent to LangSmith cloud (or self-hosted instance)
- Separate from MLflow experiment tracking

**Setup**:
```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-api-key"
export LANGCHAIN_PROJECT="project-name"
```

### MLflow Tracing

**Pros**:
- Integrated with existing MLflow experiment tracking
- Local storage option (no external service required)
- Unified platform for ML and GenAI observability
- Works with DSPy autolog integration

**Cons**:
- Known bugs with trace hierarchy (see issues above)
- Requires workarounds for proper nesting
- Less optimized for LangGraph specifics
- Async tracing limitations

**Setup**:
```python
import mlflow
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("langgraph-workflows")
mlflow.langchain.autolog(run_tracer_inline=True)
```

### Recommendation for This Project

**Dual Approach** (if possible):
1. **LangSmith** for development and debugging (superior LangGraph tracing)
2. **MLflow** for production metrics and unified observability

**MLflow-Only Approach** (current setup):
- Use `run_tracer_inline=True` for async workflows
- Add manual spans inside critical nodes
- Accept limitations with graph-level trace wrapping
- Monitor GitHub issues for bug fixes

---

## Complete Working Example

```python
import mlflow
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from mlflow.types.core import SpanType

# ============================================================================
# SETUP
# ============================================================================

# Set MLflow tracking
mlflow.set_tracking_uri("./mlruns")
mlflow.set_experiment("composition-analysis")

# Enable autolog with inline tracing for async support
mlflow.langchain.autolog(run_tracer_inline=True)

# ============================================================================
# STATE DEFINITION
# ============================================================================

class CompositionState(TypedDict):
    composition_text: str
    entities: list
    emotions: list
    summary: str

# ============================================================================
# NODE FUNCTIONS WITH MANUAL SPANS
# ============================================================================

def extract_entities_node(state: CompositionState):
    """Extract entities with detailed tracking."""

    with mlflow.start_span(name="entity_extraction", span_type=SpanType.LLM) as span:
        span.set_inputs({"text": state["composition_text"]})

        # Simulate entity extraction
        entities = ["person1", "location1", "event1"]

        span.set_outputs({"entities": entities})
        span.set_attribute("entity_count", len(entities))
        span.set_attribute("text_length", len(state["composition_text"]))

    return {"entities": entities}

def analyze_emotions_node(state: CompositionState):
    """Analyze emotions with detailed tracking."""

    with mlflow.start_span(name="emotion_analysis", span_type=SpanType.LLM) as span:
        span.set_inputs({
            "text": state["composition_text"],
            "entities": state["entities"]
        })

        # Simulate emotion analysis
        emotions = ["joy", "curiosity"]

        span.set_outputs({"emotions": emotions})
        span.set_attribute("emotion_count", len(emotions))

    return {"emotions": emotions}

def generate_summary_node(state: CompositionState):
    """Generate summary with detailed tracking."""

    with mlflow.start_span(name="summary_generation", span_type=SpanType.LLM) as span:
        span.set_inputs({
            "text": state["composition_text"],
            "entities": state["entities"],
            "emotions": state["emotions"]
        })

        # Simulate summary generation
        summary = f"Composition about {', '.join(state['entities'])} with {', '.join(state['emotions'])}"

        span.set_outputs({"summary": summary})
        span.set_attribute("summary_length", len(summary))

    return {"summary": summary}

# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_composition_graph():
    """Build the composition analysis graph."""
    builder = StateGraph(CompositionState)

    # Add nodes
    builder.add_node("extract_entities", extract_entities_node)
    builder.add_node("analyze_emotions", analyze_emotions_node)
    builder.add_node("generate_summary", generate_summary_node)

    # Add edges
    builder.add_edge(START, "extract_entities")
    builder.add_edge("extract_entities", "analyze_emotions")
    builder.add_edge("analyze_emotions", "generate_summary")
    builder.add_edge("generate_summary", END)

    # Compile with checkpointer
    return builder.compile(checkpointer=MemorySaver())

# ============================================================================
# EXECUTION
# ============================================================================

def run_analysis(user_id: str, composition_id: str, text: str):
    """Run composition analysis with proper tracing."""

    graph = build_composition_graph()

    # Configuration with metadata
    config = {
        "configurable": {
            "thread_id": f"user-{user_id}-comp-{composition_id}"
        },
        "tags": ["production", f"user-{user_id}"],
        "metadata": {
            "user_id": user_id,
            "composition_id": composition_id,
            "environment": "production",
        }
    }

    # Inputs
    inputs = {
        "composition_text": text
    }

    # Execute - autolog creates trace automatically
    result = graph.invoke(inputs, config)

    # Get trace for reference
    trace_id = mlflow.get_last_active_trace_id()
    print(f"Created trace: {trace_id}")

    return result

# ============================================================================
# USAGE
# ============================================================================

if __name__ == "__main__":
    result = run_analysis(
        user_id="user123",
        composition_id="comp456",
        text="A beautiful sunset over the mountains with friends."
    )

    print(f"Analysis complete: {result}")
```

**Expected Trace Structure**:
```
Graph Execution (autolog root)
  ├─ extract_entities node
  │   └─ entity_extraction (manual span)
  ├─ analyze_emotions node
  │   └─ emotion_analysis (manual span)
  └─ generate_summary node
      └─ summary_generation (manual span)
```

---

## Project-Specific Recommendations

Based on the current setup in `main.py`:

### Current Setup Analysis

```python
# In main.py lifespan()
def enable_mlflow_autolog():
    import mlflow
    mlflow.langchain.autolog()  # ⚠️ Missing run_tracer_inline=True!
    mlflow.dspy.autolog()

autolog_thread = threading.Thread(target=enable_mlflow_autolog)
autolog_thread.start()
```

### Recommended Changes

**1. Enable Inline Tracing**:
```python
def enable_mlflow_autolog():
    import mlflow
    mlflow.langchain.autolog(run_tracer_inline=True)  # ✅ Critical for async!
    mlflow.dspy.autolog()
```

**2. Add Manual Spans in Step A Agent Nodes**:

In `src/agent/step_a_agent/nodes/form_hypotheses.py`:
```python
import mlflow
from mlflow.types.core import SpanType

def form_hypotheses_node(state: StepAState):
    """Form hypotheses with detailed tracking."""

    with mlflow.start_span(name="hypothesis_generation", span_type=SpanType.LLM) as span:
        span.set_inputs({"composition": state["composition"]})

        # Existing logic
        hypotheses = generate_hypotheses(state)

        span.set_outputs({"hypotheses": hypotheses})
        span.set_attribute("hypothesis_count", len(hypotheses))

    return {"hypotheses": hypotheses}
```

In `src/agent/step_a_agent/nodes/search.py`:
```python
import mlflow
from mlflow.types.core import SpanType

def search_node(state: StepAState):
    """Search for information with detailed tracking."""

    with mlflow.start_span(name="graph_search", span_type=SpanType.TOOL) as span:
        span.set_inputs({
            "hypotheses": state["hypotheses"],
            "search_queries": state["search_queries"]
        })

        # Existing search logic
        search_results = execute_search(state)

        span.set_outputs({"results": search_results})
        span.set_attribute("result_count", len(search_results))

    return {"search_results": search_results}
```

**3. Use Descriptive Config When Invoking**:

In `src/routers/composition/composition_routers.py`:
```python
config = {
    "configurable": {
        "thread_id": f"user-{user_id}-comp-{composition_id}"
    },
    "tags": ["composition_analysis", f"user-{user_id}"],
    "metadata": {
        "user_id": user_id,
        "composition_id": composition_id,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "version": "0.4.0"
    }
}

result = step_a_graph.invoke(inputs, config)
```

---

## Troubleshooting

### Symptom: Multiple Separate Traces Instead of One Unified Trace

**Possible Causes**:
1. Using `@mlflow.trace` decorator with autolog (known bug)
2. Async workflow without `run_tracer_inline=True`
3. Manual trace creation conflicting with autolog

**Solutions**:
- Remove `@mlflow.trace` decorators from graph wrapper functions
- Add `run_tracer_inline=True` to autolog call
- Use manual spans inside nodes, not around graph invocation

### Symptom: "No active trace found" Error

**Possible Causes**:
1. Context not properly propagated to nodes
2. Background task execution breaking context

**Solutions**:
- Enable `run_tracer_inline=True`
- Avoid creating background tasks within nodes
- Use `asyncio.gather()` instead of background tasks

### Symptom: Flat Trace Hierarchy (All Spans at Root Level)

**Possible Causes**:
1. Async context propagation failure
2. Missing `run_tracer_inline=True` parameter

**Solutions**:
- Add `run_tracer_inline=True` to `mlflow.langchain.autolog()`
- Verify async/await patterns are correct
- Check that manual spans are created within node execution context

---

## Future Improvements to Monitor

**GitHub Issues to Watch**:
- [#18216](https://github.com/mlflow/mlflow/issues/18216) - Manual + autolog tracing combination
- [#16880](https://github.com/mlflow/mlflow/issues/16880) - Async trace hierarchy flattening

**Potential Future Features**:
- Native support for wrapping graph execution with parent span
- Improved async context propagation
- Better integration with LangGraph-specific features
- Automatic graph topology visualization

---

## Related Documentation

- [LangGraph Observability, Tracing, and MLflow Integration](./langgraph-observability-tracing-mlflow.md)
- [DSPy LM Tracking and Monitoring](./dspy-lm-tracking-monitoring.md)
- [LangGraph Multi-Step Workflow Patterns](./langgraph-multi-step-workflow-patterns.md)

---

## References and Sources

1. [MLflow LangGraph Tracing Documentation](https://mlflow.org/docs/latest/genai/tracing/integrations/listing/langgraph)
2. [MLflow LangChain Autologging](https://mlflow.org/docs/latest/genai/flavors/langchain/autologging/)
3. [GitHub Issue #18216: Combining Manual and Automatic Tracing](https://github.com/mlflow/mlflow/issues/18216)
4. [GitHub Issue #16880: Async Trace Hierarchy Flattening](https://github.com/mlflow/mlflow/issues/16880)
5. [MLflow Manual Tracing Guide](https://mlflow.org/docs/latest/genai/tracing/app-instrumentation/manual-tracing/)
6. [Tracing and Evaluating LangGraph AI Agents with MLflow](https://www.advancinganalytics.co.uk/blog/tracing-and-evaluating-langgraph-ai-agents-with-mlflow)

---

## Summary

**The Problem**: LangGraph workflows create separate traces for each node instead of a unified hierarchical trace when using MLflow autolog, especially with async workflows or when combining manual and automatic tracing.

**Root Cause**: Known bugs in MLflow's context propagation, particularly in async scenarios (Issues #16880, #18216).

**Solution**:
1. Use `mlflow.langchain.autolog(run_tracer_inline=True)` for async workflows
2. Add manual spans **inside** node functions, not around graph invocation
3. Avoid combining `@mlflow.trace` decorators with autolog at graph level
4. Use descriptive config with metadata for better trace organization

**Working Pattern**:
```python
# Setup
mlflow.langchain.autolog(run_tracer_inline=True)

# Node with manual spans
def my_node(state):
    with mlflow.start_span(name="operation", span_type=SpanType.TOOL) as span:
        # Work here
        pass
    return result

# Invoke - autolog handles graph-level tracing
result = graph.invoke(inputs, config)
```

**Non-Working Patterns**:
- ❌ Wrapping `graph.invoke()` with `@mlflow.trace`
- ❌ Using autolog without `run_tracer_inline=True` for async
- ❌ Creating manual root traces when autolog is enabled
