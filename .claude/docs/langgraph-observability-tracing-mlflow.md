# LangGraph Observability, Tracing, and MLflow Integration

**Source**: Official LangGraph Documentation (langchain-ai/langgraph)
**Date Retrieved**: 2025-12-09
**Version Coverage**: LangGraph 0.2.74 - 1.0.3

## Summary

LangGraph provides comprehensive observability primarily through **LangSmith** (its native tracing platform), with support for custom instrumentation via streaming modes, callbacks, and the LangChain Runnable interface. While there is no direct native MLflow integration, LangGraph can be instrumented for MLflow using custom streaming, callbacks, and manual span creation. This document covers what can be tracked, native observability features, streaming modes, and strategies for MLflow integration.

---

## 1. What Can Be Tracked in LangGraph

### 1.1 Graph-Level Tracking

LangGraph automatically tracks the following at the graph execution level:

- **Graph Invocations**: Complete execution traces from input to output
- **State Snapshots**: Full state at each super-step (checkpoint)
- **Node Executions**: Which nodes executed, in what order, and their outputs
- **Edge Transitions**: Flow between nodes and conditional routing decisions
- **Execution Metadata**: Run IDs, thread IDs, timestamps, configuration
- **Errors and Exceptions**: Error states, stack traces, and failure points
- **Checkpoints**: Persistent state snapshots for recovery and time-travel

### 1.2 Node-Level Tracking

Within individual nodes:

- **Node Input**: State received by each node
- **Node Output**: State updates returned by each node
- **Execution Time**: Duration of each node execution
- **Tool Calls**: Tools invoked within nodes, their arguments, and results
- **LLM Calls**: Model invocations, prompts, responses, and token usage
- **Custom Events**: User-defined events emitted via stream writers

### 1.3 LLM and Tool Tracking

Detailed tracking for language model and tool usage:

- **LLM Tokens**: Individual token streaming from any LLM call
- **Prompts**: System and user prompts sent to LLMs
- **Model Responses**: Full responses and partial chunks
- **Token Usage**: Input tokens, output tokens, cached tokens
- **Tool Arguments**: Parameters passed to tool functions
- **Tool Results**: Return values and execution status
- **Parallel Tool Calls**: Multiple simultaneous tool executions
- **Tool Errors**: Exceptions and error messages from tools

### 1.4 State and Memory Tracking

State management and persistence:

- **State Changes**: Incremental updates to graph state
- **State History**: Complete checkpoint history for a thread
- **Checkpoint Metadata**: Tags, user info, timestamps
- **Thread Context**: Conversation history and session state
- **State Schema**: Structure and types of state channels
- **Reducers**: How state updates are merged

### 1.5 Subgraph Tracking

For nested graph architectures:

- **Subgraph Executions**: Traces from subgraphs included in parent traces
- **Subgraph Namespaces**: Identification of which graph/subgraph produced output
- **State Transformations**: How state is mapped between parent and child graphs
- **Nested Interrupts**: Human-in-the-loop pauses in subgraphs

---

## 2. Native Observability: LangSmith Integration

### 2.1 LangSmith Overview

LangSmith is LangGraph's official observability platform, providing:

- Visual trace debugging and inspection
- Thread and checkpoint history viewing
- Performance monitoring and analytics
- Production deployment tracing
- Test and evaluation frameworks

### 2.2 Enabling LangSmith Tracing

**Environment Variables**:

```bash
export LANGSMITH_TRACING=true
export LANGSMITH_API_KEY="your-api-key-here"
export LANGCHAIN_PROJECT="your-project-name"  # Optional, defaults to "default"
```

**Python Example**:

```python
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "snipreel-backend"

# All graph executions are now automatically traced
result = graph.invoke(inputs, config)
```

**Self-Hosted LangSmith**:

```bash
export LANGSMITH_ENDPOINT="https://your-langsmith-instance.com"
export LANGSMITH_API_KEY="your-api-key"
```

### 2.3 Custom Run Metadata for LangSmith

LangGraph implements the LangChain `Runnable` interface, allowing rich metadata:

```python
import uuid
from typing import Any

# Custom run configuration
config = {
    "run_id": uuid.uuid4(),  # Must be UUID
    "run_name": "composition_analysis_run",
    "tags": ["production", "composition", "user-123"],
    "metadata": {
        "user_id": "user-123",
        "composition_id": "comp-456",
        "environment": "production",
        "version": "0.4.0"
    }
}

# Apply to any Runnable method
result = graph.invoke(inputs, config)
result = graph.stream(inputs, config, stream_mode="values")
await graph.ainvoke(inputs, config)
```

**Filtering and Searching in LangSmith**:

- Filter traces by `tags` (e.g., all production runs)
- Search by `metadata` fields (e.g., specific users)
- Custom `run_name` for easy identification
- Specific `run_id` for debugging exact executions

---

## 3. Streaming Modes: Real-Time Observability

LangGraph provides multiple streaming modes for different observability needs.

### 3.1 Available Stream Modes

| Mode | Purpose | Output Format | Use Case |
|------|---------|---------------|----------|
| `values` | Full state after each step | Complete state object | Monitoring complete state progression |
| `updates` | Only state changes | `{node_name: state_update}` | Tracking incremental changes |
| `messages` | LLM token streaming | `(message_chunk, metadata)` | Real-time LLM output display |
| `debug` | Detailed execution info | Full debug information | Troubleshooting and deep inspection |
| `custom` | User-defined data | Custom objects | Progress updates, metrics, custom events |
| `events` | All events | Comprehensive event stream | Complete execution monitoring |

### 3.2 Stream Mode: `values`

**Purpose**: Stream the **complete state** after each node execution.

```python
for chunk in graph.stream(inputs, stream_mode="values"):
    print(f"State after step: {chunk}")
    # chunk is the full state dictionary

# Output example:
# {'messages': [...], 'entities': [...], 'step': 1}
# {'messages': [...], 'entities': [...], 'step': 2}
```

**When to use**:
- Monitoring full state evolution
- Debugging state accumulation issues
- Displaying complete context at each step

### 3.3 Stream Mode: `updates`

**Purpose**: Stream **only the changes** to state from each node.

```python
for chunk in graph.stream(inputs, stream_mode="updates"):
    print(f"Node: {chunk}")
    # chunk = {node_name: state_update}

# Output example:
# {'analyze_entities': {'entities': [...]}}
# {'extract_emotions': {'emotions': [...]}}
```

**When to use**:
- Tracking what each node contributes
- Optimizing state update logic
- Reducing data transfer in streaming scenarios

### 3.4 Stream Mode: `messages` (LLM Token Streaming)

**Purpose**: Stream **individual LLM tokens** as they're generated.

```python
for message_chunk, metadata in graph.stream(
    inputs,
    stream_mode="messages"
):
    if message_chunk.content:
        print(message_chunk.content, end="", flush=True)

    # metadata contains:
    # - langgraph_node: which node made the LLM call
    # - tags: tags associated with the LLM
    # - other context info
```

**Filtering by Node**:

```python
for msg, metadata in graph.stream(inputs, stream_mode="messages"):
    # Only show tokens from specific node
    if metadata.get("langgraph_node") == "call_model":
        print(msg.content, end="|")
```

**Filtering by Tags**:

```python
# Tag LLMs to differentiate outputs
llm_joke = ChatOpenAI(model="gpt-4o-mini", tags=["joke"])
llm_poem = ChatOpenAI(model="gpt-4o-mini", tags=["poem"])

for msg, metadata in graph.stream(inputs, stream_mode="messages"):
    if "joke" in metadata.get("tags", []):
        print(f"Joke: {msg.content}")
```

**When to use**:
- Real-time UI updates for LLM responses
- Tracking token usage per node
- Debugging prompt/response issues

### 3.5 Stream Mode: `custom`

**Purpose**: Emit **custom user-defined data** from nodes or tools.

**In Nodes**:

```python
from langgraph.config import get_stream_writer

def my_node(state: State):
    writer = get_stream_writer()

    # Emit progress updates
    writer({"progress": 0, "status": "starting"})

    # Do work...
    process_data()
    writer({"progress": 50, "status": "processing"})

    # More work...
    finalize()
    writer({"progress": 100, "status": "complete"})

    return {"result": "done"}

# Stream custom data
for chunk in graph.stream(inputs, stream_mode="custom"):
    print(f"Progress: {chunk}")
    # Output: {'progress': 0, 'status': 'starting'}
    #         {'progress': 50, 'status': 'processing'}
    #         {'progress': 100, 'status': 'complete'}
```

**In Tools**:

```python
from langchain_core.tools import tool
from langgraph.config import get_stream_writer

@tool
def query_database(query: str) -> str:
    """Query the database with progress tracking."""
    writer = get_stream_writer()

    writer({"data": "Retrieved 0/100 records", "type": "progress"})
    # Perform query...
    writer({"data": "Retrieved 50/100 records", "type": "progress"})
    # More query work...
    writer({"data": "Retrieved 100/100 records", "type": "progress"})

    return "query results"
```

**Multiple Stream Modes**:

```python
# Combine custom with other modes
for chunk in graph.stream(
    inputs,
    stream_mode=["updates", "custom"]
):
    # chunk will be a tuple: (mode, data)
    mode, data = chunk
    if mode == "custom":
        print(f"Custom event: {data}")
    elif mode == "updates":
        print(f"Node update: {data}")
```

**When to use**:
- Progress bars and status indicators
- Custom metrics and telemetry
- Real-time notifications to external systems
- Integration points for monitoring tools (like MLflow)

### 3.6 Stream Mode: `debug`

**Purpose**: Get **detailed execution information** for troubleshooting.

```python
for event in graph.stream(inputs, stream_mode="debug"):
    print(f"Debug event: {event}")
    # Includes: node names, full state, execution metadata, errors
```

**When to use**:
- Deep debugging of graph execution
- Understanding complex conditional flows
- Investigating performance issues

### 3.7 Streaming from Subgraphs

Include outputs from nested subgraphs in the stream:

```python
# Enable subgraph streaming
for chunk in graph.stream(
    inputs,
    stream_mode="updates",
    subgraphs=True  # Include subgraph outputs
):
    # chunk includes namespace indicating which graph
    namespace = chunk.get("__namespace__", "parent")
    print(f"From {namespace}: {chunk}")
```

---

## 4. State and Checkpoint Tracking

### 4.1 Retrieving Current State

```python
config = {"configurable": {"thread_id": "conversation-123"}}

# Get current state
state = graph.get_state(config)

print(state.values)        # Current state dictionary
print(state.next)          # Next nodes to execute
print(state.config)        # Configuration with checkpoint_id
print(state.metadata)      # Checkpoint metadata
print(state.created_at)    # Timestamp
```

### 4.2 State History (Time-Travel)

Retrieve complete execution history:

```python
config = {"configurable": {"thread_id": "conversation-123"}}

# Get all checkpoints (newest first)
history = list(graph.get_state_history(config))

for i, checkpoint in enumerate(history):
    print(f"Step {i}:")
    print(f"  State: {checkpoint.values}")
    print(f"  Next: {checkpoint.next}")
    print(f"  Checkpoint ID: {checkpoint.config['configurable']['checkpoint_id']}")
    print(f"  Metadata: {checkpoint.metadata}")
    print()
```

### 4.3 Replaying from Checkpoint

Resume execution from any historical checkpoint:

```python
# Select a specific checkpoint
past_checkpoint_id = history[2].config["configurable"]["checkpoint_id"]

# Create config for that checkpoint
time_travel_config = {
    "configurable": {
        "thread_id": "conversation-123",
        "checkpoint_id": past_checkpoint_id
    }
}

# Continue from that point (will replay then continue)
result = graph.invoke(new_inputs, config=time_travel_config)
```

---

## 5. Tool and LLM Call Tracking

### 5.1 Tracking Tool Executions

LangGraph automatically tracks tool calls via `ToolMessage` objects:

```python
from langchain_core.messages import ToolMessage

# Tool execution is automatically tracked
# Each tool call creates a ToolMessage with:
# - tool_call_id: Links back to the AI's tool call
# - name: Tool name
# - content: Tool result
# - metadata: Additional context

# Access via state
for message in state["messages"]:
    if isinstance(message, ToolMessage):
        print(f"Tool: {message.name}")
        print(f"Result: {message.content}")
        print(f"Call ID: {message.tool_call_id}")
```

### 5.2 Tracking LLM Calls

LLM calls are tracked through:

1. **Message History** (in state):

```python
from langchain_core.messages import AIMessage, HumanMessage

for message in state["messages"]:
    if isinstance(message, AIMessage):
        print(f"AI Response: {message.content}")
        print(f"Tool Calls: {message.tool_calls}")
        print(f"Usage Metadata: {message.usage_metadata}")
```

2. **Token Streaming** (via `messages` mode):

```python
for token, metadata in graph.stream(inputs, stream_mode="messages"):
    # Track token-by-token
    print(f"Token from {metadata['langgraph_node']}: {token.content}")
```

3. **LangSmith Traces** (automatic when enabled):
   - Full prompt and response
   - Token counts
   - Model name and parameters
   - Latency and cost

---

## 6. MLflow Integration Strategies

### 6.1 Why No Native MLflow Integration?

LangGraph focuses on LangSmith as its native observability platform. However, integration with MLflow is possible through:

- **LangChain Callbacks**: LangGraph implements the Runnable interface
- **Custom Streaming**: Use `custom` stream mode to emit MLflow events
- **Manual Instrumentation**: Wrap graph executions with MLflow spans
- **DSPy Integration**: Track DSPy modules within LangGraph nodes (already done in this project)

### 6.2 Strategy 1: Custom Stream Mode for MLflow

Emit MLflow tracking data via custom streaming:

```python
import mlflow
from langgraph.config import get_stream_writer

def composition_analysis_node(state: State):
    writer = get_stream_writer()

    # Emit MLflow tracking events
    writer({
        "mlflow_event": "node_start",
        "node_name": "composition_analysis",
        "timestamp": time.time()
    })

    # Do the work
    result = analyze_composition(state["composition"])

    # Track metrics
    writer({
        "mlflow_event": "log_metric",
        "metric_name": "entities_extracted",
        "metric_value": len(result["entities"])
    })

    writer({
        "mlflow_event": "node_end",
        "node_name": "composition_analysis",
        "duration": time.time() - start_time
    })

    return result

# In your MLflow consumer
for chunk in graph.stream(inputs, stream_mode="custom"):
    if chunk.get("mlflow_event") == "log_metric":
        mlflow.log_metric(
            chunk["metric_name"],
            chunk["metric_value"]
        )
```

### 6.3 Strategy 2: Wrap Graph Execution with MLflow Spans

Manually create MLflow tracking around graph invocations:

```python
import mlflow
from mlflow import MlflowClient

def run_graph_with_mlflow(graph, inputs, config):
    """Execute graph with MLflow tracking."""

    with mlflow.start_run(run_name="langgraph_execution"):
        # Log input
        mlflow.log_param("thread_id", config["configurable"]["thread_id"])
        mlflow.log_param("graph_name", graph.__class__.__name__)
        mlflow.log_dict(inputs, "inputs.json")

        # Track execution
        start_time = time.time()

        try:
            # Execute graph and collect outputs
            outputs = []
            for chunk in graph.stream(inputs, config, stream_mode="updates"):
                outputs.append(chunk)

                # Log each node execution
                for node_name, node_output in chunk.items():
                    mlflow.log_dict(
                        node_output,
                        f"node_outputs/{node_name}.json"
                    )

            # Log final state
            final_state = graph.get_state(config)
            mlflow.log_dict(final_state.values, "final_state.json")

            # Log metrics
            duration = time.time() - start_time
            mlflow.log_metric("execution_duration_seconds", duration)
            mlflow.log_metric("num_steps", len(outputs))

            return final_state.values

        except Exception as e:
            mlflow.log_param("error", str(e))
            mlflow.log_param("status", "failed")
            raise
        else:
            mlflow.log_param("status", "success")
```

### 6.4 Strategy 3: LangChain Callback Handler for MLflow

Create a custom callback handler (LangGraph supports LangChain callbacks):

```python
from langchain.callbacks.base import BaseCallbackHandler
import mlflow

class MLflowCallbackHandler(BaseCallbackHandler):
    """Track LangGraph execution in MLflow."""

    def __init__(self):
        self.run_stack = []

    def on_chain_start(self, serialized, inputs, **kwargs):
        """Called when a chain (or graph) starts."""
        run_id = kwargs.get("run_id")
        parent_run_id = kwargs.get("parent_run_id")

        # Start MLflow span
        span_name = serialized.get("name", "unknown")
        self.run_stack.append({
            "span_name": span_name,
            "start_time": time.time(),
            "run_id": str(run_id)
        })

    def on_chain_end(self, outputs, **kwargs):
        """Called when a chain ends."""
        if self.run_stack:
            span_info = self.run_stack.pop()
            duration = time.time() - span_info["start_time"]

            # Log to MLflow
            mlflow.log_metric(
                f"{span_info['span_name']}_duration",
                duration
            )

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Track LLM calls."""
        mlflow.log_param("llm_prompts", len(prompts))

    def on_llm_end(self, response, **kwargs):
        """Track LLM responses."""
        # Extract token usage
        token_usage = response.llm_output.get("token_usage", {})
        mlflow.log_metric("llm_total_tokens", token_usage.get("total_tokens", 0))
        mlflow.log_metric("llm_prompt_tokens", token_usage.get("prompt_tokens", 0))
        mlflow.log_metric("llm_completion_tokens", token_usage.get("completion_tokens", 0))

    def on_tool_start(self, serialized, input_str, **kwargs):
        """Track tool usage."""
        tool_name = serialized.get("name", "unknown")
        mlflow.log_param(f"tool_{tool_name}", input_str)

    def on_tool_end(self, output, **kwargs):
        """Track tool results."""
        pass

# Use with graph
mlflow_handler = MLflowCallbackHandler()

config = {
    "configurable": {"thread_id": "123"},
    "callbacks": [mlflow_handler]
}

result = graph.invoke(inputs, config)
```

### 6.5 Strategy 4: Integrate with Existing DSPy MLflow Tracking

Since this project already tracks DSPy with MLflow, extend the pattern:

```python
from src.utils.dspy_tracking import (
    extract_lm_history,
    extract_token_usage,
    set_mlflow_span_tracking
)
import mlflow

def langgraph_node_with_dspy(state: State):
    """Node that uses DSPy and tracks to MLflow."""

    with mlflow.start_span(name="composition_analysis_node") as span:
        # Execute DSPy module
        prediction = dspy_module(**state["input"])

        # Extract tracking data
        history_data = extract_lm_history(dspy.settings.lm, logger)
        token_usage = extract_token_usage(prediction, logger)

        # Set MLflow span data
        set_mlflow_span_tracking(
            span,
            history_data,
            token_usage,
            prediction.toDict(),
            fallback_inputs=state["input"]
        )

        return {"output": prediction}
```

### 6.6 Best Practices for MLflow Integration

1. **Use Thread IDs as MLflow Run Names**:
   ```python
   run_name = f"langgraph_{config['configurable']['thread_id']}"
   with mlflow.start_run(run_name=run_name):
       result = graph.invoke(inputs, config)
   ```

2. **Log Checkpoint IDs as MLflow Tags**:
   ```python
   state = graph.get_state(config)
   checkpoint_id = state.config["configurable"]["checkpoint_id"]
   mlflow.set_tag("checkpoint_id", checkpoint_id)
   ```

3. **Track Node-Level Metrics**:
   ```python
   for node_name, node_output in chunk.items():
       with mlflow.start_span(name=node_name):
           # Log node-specific metrics
           mlflow.log_metric(f"{node_name}_output_size", len(str(node_output)))
   ```

4. **Log State Changes**:
   ```python
   for chunk in graph.stream(inputs, stream_mode="updates"):
       for node_name, update in chunk.items():
           mlflow.log_dict(update, f"updates/{node_name}_{step}.json")
   ```

---

## 7. Production Deployment Monitoring

### 7.1 LangGraph Platform Metrics

When deployed to LangGraph Platform (Cloud or Self-Hosted), automatic monitoring includes:

- **Infrastructure Metrics**:
  - CPU and memory usage
  - Container restarts
  - Replica count (autoscaling)
  - Postgres resource usage

- **Application Metrics**:
  - Run success/error counts
  - API latency
  - Queue depth (pending/active runs)
  - Throughput (runs per second)

### 7.2 Autoscaling

Production deployments automatically scale based on:

- CPU utilization (target: 75%)
- Memory utilization (target: 75%)
- Pending run count (target: 10 runs per container)
- Maximum: 10 containers
- Scale-down cooldown: 30 minutes

### 7.3 Custom Monitoring Integration

Use environment variables for third-party monitoring:

**Datadog**:
```bash
export DD_API_KEY="your-datadog-key"
export DD_SITE="datadoghq.com"
export DD_ENV="production"
export DD_SERVICE="langgraph-agent"
export DD_TRACE_ENABLED="true"
```

---

## 8. Best Practices

### 8.1 What to Track at Graph Level vs Node Level

**Graph Level**:
- Overall execution time
- Success/failure rates
- Input/output schemas
- Thread/conversation context
- High-level business metrics

**Node Level**:
- Individual node duration
- State transformations per node
- Tool calls and their results
- LLM token usage per node
- Node-specific errors

### 8.2 Tracking Nested Graphs/Subgraphs

```python
# Enable subgraph streaming
for chunk in graph.stream(
    inputs,
    stream_mode=["updates", "debug"],
    subgraphs=True
):
    # Check namespace to identify graph hierarchy
    namespace = chunk.get("__namespace__")
    if namespace:
        print(f"From subgraph {namespace}: {chunk}")
```

**Best practices**:
- Use distinct `run_name` for parent and subgraphs
- Tag subgraph executions with parent context
- Log state transformations between graphs
- Track handoff points between agents

### 8.3 Performance Considerations

**Minimize Overhead**:

1. **Use appropriate stream modes**:
   - `updates` is lighter than `values`
   - `messages` only for user-facing streaming
   - `debug` only in development

2. **Batch logging in custom streams**:
   ```python
   # Bad: Log every token
   for token in tokens:
       writer({"token": token})

   # Good: Batch tokens
   writer({"tokens": tokens, "count": len(tokens)})
   ```

3. **Use checkpointer durability modes**:
   ```python
   # Best performance (checkpoints only on exit)
   from langgraph.checkpoint.postgres import PostgresSaver

   checkpointer = PostgresSaver.from_conn_string(
       DB_URI,
       checkpoint_mode="exit"  # Only save on graph completion
   )
   ```

4. **Filter streaming data early**:
   ```python
   for msg, metadata in graph.stream(inputs, stream_mode="messages"):
       # Filter immediately
       if metadata.get("langgraph_node") != "target_node":
           continue
       # Only process relevant data
       process(msg)
   ```

### 8.4 Structuring Traces for LangGraph Workflows

**Hierarchical Structure**:

```
Run: langgraph_execution
  ├─ Span: graph_invoke
  │   ├─ Span: node_1
  │   │   ├─ Span: llm_call
  │   │   └─ Span: tool_call
  │   ├─ Span: node_2
  │   │   └─ Span: llm_call
  │   └─ Span: node_3
```

**Implementation**:

```python
with mlflow.start_run(run_name="langgraph_execution"):
    with mlflow.start_span(name="graph_invoke"):
        for chunk in graph.stream(inputs, stream_mode="updates"):
            for node_name, output in chunk.items():
                with mlflow.start_span(name=node_name):
                    # Log node execution
                    mlflow.log_dict(output, f"{node_name}_output.json")
```

### 8.5 Recommended Metadata

**Always include**:
- `thread_id`: Conversation context
- `run_id`: Unique execution identifier
- `user_id`: User performing the action
- `environment`: dev/staging/production
- `version`: Application version

**Helpful for debugging**:
- `checkpoint_id`: For time-travel debugging
- `error_type`: Classification of failures
- `retry_count`: For retry logic tracking
- `upstream_service`: If part of larger system

---

## 9. Common Patterns and Gotchas

### 9.1 Tracking Multi-Agent Handoffs

```python
# Track which agent is active
def track_handoff(state: State):
    writer = get_stream_writer()

    # Log handoff event
    writer({
        "event": "agent_handoff",
        "from_agent": state["current_agent"],
        "to_agent": state["next_agent"],
        "reason": state["handoff_reason"]
    })

    return state
```

### 9.2 Tracking Errors and Retries

```python
def resilient_node(state: State, config: RunnableConfig):
    max_retries = 3

    for attempt in range(max_retries):
        try:
            writer = get_stream_writer()
            writer({
                "event": "node_attempt",
                "attempt": attempt + 1,
                "max_retries": max_retries
            })

            result = perform_operation(state)

            writer({
                "event": "node_success",
                "attempt": attempt + 1
            })

            return result

        except Exception as e:
            writer({
                "event": "node_error",
                "attempt": attempt + 1,
                "error": str(e)
            })

            if attempt == max_retries - 1:
                raise

            time.sleep(2 ** attempt)  # Exponential backoff
```

### 9.3 Gotcha: Stream Mode vs Invoke

**Important**: Some stream modes don't work with `.invoke()`:

```python
# This works
result = graph.invoke(inputs, config)

# This also works - returns final state
for chunk in graph.stream(inputs, config, stream_mode="values"):
    pass

# But you can't get "messages" or "custom" from invoke
# You MUST use stream() for those modes
```

### 9.4 Gotcha: Subgraph Streaming Configuration

Runtime `interrupt_before`/`interrupt_after` don't work for subgraphs - must be compile-time:

```python
# ❌ Won't work
subgraph.invoke(inputs, interrupt_before=["node_a"])

# ✅ Configure at compile time
subgraph = sub_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["node_a"]
)
```

---

## 10. Complete Example: MLflow-Instrumented LangGraph

```python
import mlflow
import time
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer, RunnableConfig
from langchain_openai import ChatOpenAI

# State definition
class CompositionState(TypedDict):
    composition_text: str
    entities: list
    emotions: list
    summary: str

# Node with MLflow tracking
def extract_entities(state: CompositionState, config: RunnableConfig):
    """Extract entities with MLflow tracking."""
    writer = get_stream_writer()

    # Log start
    writer({
        "mlflow_event": "node_start",
        "node": "extract_entities",
        "timestamp": time.time()
    })

    # Simulate entity extraction
    llm = ChatOpenAI(model="gpt-4o-mini")
    response = llm.invoke([
        {"role": "system", "content": "Extract entities from text."},
        {"role": "user", "content": state["composition_text"]}
    ])

    entities = ["entity1", "entity2"]  # Parsed from response

    # Log metrics
    writer({
        "mlflow_event": "log_metric",
        "metric_name": "entities_extracted",
        "metric_value": len(entities)
    })

    writer({
        "mlflow_event": "node_end",
        "node": "extract_entities"
    })

    return {"entities": entities}

# Build graph
def build_composition_graph():
    graph = StateGraph(CompositionState)
    graph.add_node("extract_entities", extract_entities)
    # Add more nodes...
    graph.add_edge(START, "extract_entities")
    graph.add_edge("extract_entities", END)
    return graph.compile()

# Execute with MLflow
def run_with_mlflow(graph, inputs):
    with mlflow.start_run(run_name="composition_analysis"):
        mlflow.log_param("input_length", len(inputs["composition_text"]))

        start_time = time.time()
        config = {"configurable": {"thread_id": "user-123"}}

        # Stream with both updates and custom
        for chunk in graph.stream(
            inputs,
            config,
            stream_mode=["updates", "custom"]
        ):
            mode, data = chunk

            if mode == "custom":
                # Process MLflow events
                if data.get("mlflow_event") == "log_metric":
                    mlflow.log_metric(
                        data["metric_name"],
                        data["metric_value"]
                    )

            elif mode == "updates":
                # Log node outputs
                for node_name, output in data.items():
                    mlflow.log_dict(output, f"{node_name}_output.json")

        # Log execution time
        duration = time.time() - start_time
        mlflow.log_metric("total_duration_seconds", duration)

        # Get final state
        final_state = graph.get_state(config)
        mlflow.log_dict(final_state.values, "final_state.json")

        return final_state.values

# Usage
graph = build_composition_graph()
result = run_with_mlflow(graph, {
    "composition_text": "A beautiful sunset over the mountains..."
})
```

---

## 11. Related Documentation

- [LangGraph Human-in-the-Loop Complete Guide](./langgraph-human-in-the-loop-complete-guide.md)
- [DSPy LM Tracking and Monitoring](./dspy-lm-tracking-monitoring.md)
- [DSPy Training with Existing Data](./dspy-training-existing-data-complete-guide.md)

## 12. Summary Table

| Feature | Native Support | MLflow Integration Strategy |
|---------|---------------|----------------------------|
| Graph execution traces | ✅ LangSmith | Custom callbacks + spans |
| Node-level tracking | ✅ Stream modes | Custom stream mode |
| LLM token streaming | ✅ `messages` mode | Callback handlers |
| Tool call tracking | ✅ Automatic | Parse from state messages |
| State snapshots | ✅ Checkpoints | Log to MLflow artifacts |
| Custom metrics | ✅ `custom` mode | Direct MLflow logging |
| Subgraph tracking | ✅ With `subgraphs=True` | Hierarchical spans |
| Error tracking | ✅ LangSmith | Exception logging + tags |
| Performance metrics | ✅ Platform metrics | Custom span instrumentation |

---

## Conclusion

LangGraph provides comprehensive observability primarily through **LangSmith**, its native tracing platform. For MLflow integration, use a combination of:

1. **Custom streaming** (`custom` mode) to emit MLflow events
2. **LangChain callbacks** to hook into graph execution lifecycle
3. **Manual span wrapping** for graph invocations
4. **Existing DSPy integration patterns** for nodes using DSPy

The key is leveraging LangGraph's flexible streaming modes and callback system to bridge to MLflow's tracking APIs, creating a unified observability solution across LangGraph and DSPy components.
