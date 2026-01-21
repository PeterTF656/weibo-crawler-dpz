# LangGraph Human-in-the-Loop Complete Guide

**Source**: Official LangGraph Documentation (langchain-ai/langgraph)
**Date Retrieved**: 2025-11-30
**Version Coverage**: LangGraph 0.2.74 - 1.0.3

## Summary

Human-in-the-loop (HITL) in LangGraph enables pausing graph execution at strategic points to gather human input, approvals, or decisions before proceeding. This is achieved through two primary mechanisms: dynamic interrupts (using the `interrupt()` function) and static breakpoints (configured at compile-time or runtime). All HITL functionality requires a checkpointer to persist graph state during pauses, enabling resumption from the exact point of interruption.

---

## Key Concepts

### 1. The `interrupt()` Function

The `interrupt()` function is the core mechanism for dynamic human-in-the-loop workflows:

- **Purpose**: Pauses graph execution at a specific point within a node
- **Behavior**: Works like Python's `input()` - halts execution and waits for external input
- **Payload**: Accepts any JSON-serializable value to surface to the human reviewer
- **Return Value**: When resumed, returns the human-provided input for use in the graph
- **Requirement**: Must have a checkpointer configured to persist state during the pause

**Basic Pattern**:
```python
from langgraph.types import interrupt

def human_approval(state: State):
    # Pause and surface data for review
    decision = interrupt({
        "question": "Do you approve this action?",
        "data_to_review": state["llm_output"]
    })

    # decision contains the human's response when resumed
    if decision == "approve":
        return {"status": "approved"}
    else:
        return {"status": "rejected"}
```

### 2. Static Breakpoints

Static breakpoints pause execution before or after specific nodes, configured at compile-time or runtime:

**Compile-time Configuration**:
```python
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["node_a"],  # Pause BEFORE node_a executes
    interrupt_after=["node_b", "node_c"]  # Pause AFTER these nodes execute
)
```

**Runtime Configuration** (per-invocation):
```python
graph.invoke(
    inputs,
    config={"configurable": {"thread_id": "thread-1"}},
    interrupt_before=["node_a"],
    interrupt_after=["node_b"]
)
```

### 3. Checkpointers - The Foundation

Checkpointers are **mandatory** for all HITL functionality. They save a snapshot of the graph state at each super-step.

**Key Checkpoint Properties**:
- `config`: Configuration associated with the checkpoint (thread_id, checkpoint_id)
- `metadata`: Additional metadata about the checkpoint
- `values`: Current state values at this point in time
- `next`: Tuple of node names to execute next
- `tasks`: Information about pending tasks, including interrupt data and error information

**Available Checkpointers**:

| Checkpointer | Use Case | Import |
|--------------|----------|--------|
| `InMemorySaver` | Development/testing only | `langgraph.checkpoint.memory` |
| `SqliteSaver` | Development/single-user apps | `langgraph.checkpoint.sqlite` |
| `PostgresSaver` | Production multi-user apps | `langgraph.checkpoint.postgres` |

**Production Setup (PostgreSQL)**:
```python
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://user:pass@localhost:5432/dbname?sslmode=disable"
checkpointer = PostgresSaver.from_conn_string(DB_URI)
checkpointer.setup()  # Call once to create tables

graph = builder.compile(checkpointer=checkpointer)
```

### 4. Threads and Thread IDs

Every graph invocation with a checkpointer **must** specify a `thread_id`:

```python
config = {"configurable": {"thread_id": "conversation-123"}}
graph.invoke(inputs, config=config)
```

- **Thread**: Represents an isolated execution context (e.g., a user conversation)
- **Thread ID**: Unique identifier that separates different graph execution contexts
- **Persistence**: All checkpoints for a thread are stored together and can be retrieved later
- **Isolation**: Different thread IDs maintain completely separate state

---

## Implementation Patterns

### Pattern 1: Dynamic Interrupt for Tool Approval

**Use Case**: Pause before executing sensitive tools to get human approval.

```python
from langgraph.types import interrupt, Command
from langchain_core.tools import tool

@tool
def delete_records(record_ids: list[str]) -> str:
    """Delete records from database. Requires confirmation."""
    confirmation = interrupt({
        "action": "delete_records",
        "record_ids": record_ids,
        "count": len(record_ids),
        "warning": "This action cannot be undone!"
    })

    if confirmation.get("confirmed"):
        # Execute the deletion
        db.delete_many(record_ids)
        return f"Deleted {len(record_ids)} records"
    else:
        return "Deletion cancelled"

# Use in agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent

memory = InMemorySaver()
agent = create_react_agent(
    model=model,
    tools=[delete_records],
    checkpointer=memory  # REQUIRED for interrupts
)

# Run until interrupt
config = {"configurable": {"thread_id": "session-1"}}
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Delete records 1, 2, 3"}]},
    config=config
)

# Check if interrupted
state = agent.get_state(config)
if state.next == ("__interrupt__",):
    print(f"Approval needed: {state.values.get('__interrupt__')}")

    # Resume with approval
    from langgraph.types import Command
    agent.invoke(
        Command(resume={"confirmed": True}),
        config=config
    )
```

### Pattern 2: Approval/Rejection Workflow with Conditional Routing

**Use Case**: Route to different nodes based on human approval.

```python
from typing import Literal
from langgraph.types import interrupt, Command
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    llm_output: str
    decision: str

def human_approval(state: State) -> Command[Literal["approved_path", "rejected_path"]]:
    decision = interrupt({
        "question": "Do you approve the following output?",
        "llm_output": state["llm_output"]
    })

    if decision == "approve":
        return Command(goto="approved_path", update={"decision": "approved"})
    else:
        return Command(goto="rejected_path", update={"decision": "rejected"})

def approved_node(state: State) -> State:
    print("✅ Approved path taken.")
    return state

def rejected_node(state: State) -> State:
    print("❌ Rejected path taken.")
    return state

# Build graph
builder = StateGraph(State)
builder.add_node("generate_llm_output", generate_llm_output)
builder.add_node("human_approval", human_approval)
builder.add_node("approved_path", approved_node)
builder.add_node("rejected_path", rejected_node)

builder.add_edge(START, "generate_llm_output")
builder.add_edge("generate_llm_output", "human_approval")
builder.add_edge("approved_path", END)
builder.add_edge("rejected_path", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# Run and resume
config = {"configurable": {"thread_id": "approval-flow-1"}}

# First run - hits interrupt
result = graph.invoke({}, config=config)
print(result["__interrupt__"])  # Shows interrupt payload

# Resume with approval
final_result = graph.invoke(Command(resume="approve"), config=config)
```

### Pattern 3: Static Breakpoints for Node Inspection

**Use Case**: Pause execution before/after specific nodes to inspect state.

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

checkpointer = InMemorySaver()
graph = builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["step_3"]  # Pause before step_3
)

config = {"configurable": {"thread_id": "1"}}

# Run until breakpoint
for event in graph.stream({"input": "hello"}, config, stream_mode="values"):
    print(event)

# Inspect state at breakpoint
state = graph.get_state(config)
print(f"State at breakpoint: {state.values}")
print(f"Next node: {state.next}")  # Should be ('step_3',)

# Resume execution (pass None as input)
for event in graph.stream(None, config, stream_mode="values"):
    print(event)
```

### Pattern 4: Input Validation with Interrupts

**Use Case**: Keep asking for input until valid data is provided.

```python
from langgraph.types import interrupt

def human_node(state):
    """Human node with validation loop."""
    question = "What is your age?"

    while True:
        answer = interrupt(question)

        # Validate answer
        if not isinstance(answer, int) or answer < 0:
            question = f"'{answer}' is not a valid age. What is your age?"
            continue
        else:
            break

    return {"age": answer}
```

### Pattern 5: Multiple Parallel Interrupts

**Use Case**: Handle multiple simultaneous human inputs when nodes run in parallel.

```python
from langgraph.types import Command

# When parallel nodes both have interrupts, you get multiple interrupt IDs
# Resume all at once with a mapping:

resume_values = {
    "interrupt_id_1": {"approved": True},
    "interrupt_id_2": {"confirmed": True}
}

graph.invoke(
    Command(resume=resume_values),
    config=config
)
```

---

## Advanced Features

### 1. Accessing Interrupt Data

When a graph hits an interrupt, the return value contains interrupt information:

**Python (v0.4.0+)**:
```python
result = graph.invoke(inputs, config)
if "__interrupt__" in result:
    interrupt_data = result["__interrupt__"]
    print(interrupt_data)
```

**Alternative (all versions)**:
```python
# Use get_state() to retrieve interrupt info
state = graph.get_state(config)
if state.next == ("__interrupt__",):
    # Interrupts are in state.tasks
    for task in state.tasks:
        if hasattr(task, 'interrupts'):
            print(task.interrupts)
```

**TypeScript**:
```typescript
const result = await graph.invoke(inputs, config);
if (result.__interrupt__) {
    console.log(result.__interrupt__);
}
```

### 2. Conditional/Sensitive Tool Interrupts

Only interrupt for sensitive operations, let safe operations proceed:

```python
# Categorize tools
SAFE_TOOLS = ["search", "get_weather", "lookup_info"]
SENSITIVE_TOOLS = ["delete_record", "update_flight", "book_hotel"]

def should_interrupt(tool_name: str) -> bool:
    return tool_name in SENSITIVE_TOOLS

# In your node logic
if should_interrupt(tool_call.name):
    approval = interrupt({
        "tool": tool_call.name,
        "args": tool_call.args
    })
    # ... handle approval
else:
    # Execute directly without interruption
    result = tool_call.execute()
```

### 3. Time Travel and Checkpoint History

Retrieve full execution history and resume from any checkpoint:

```python
# Get all checkpoints for a thread
config = {"configurable": {"thread_id": "conversation-1"}}
history = list(graph.get_state_history(config))

# history[0] is most recent, history[-1] is oldest
for i, checkpoint in enumerate(history):
    print(f"Step {i}: {checkpoint.values}")
    print(f"Checkpoint ID: {checkpoint.config['configurable']['checkpoint_id']}")

# Resume from a specific past checkpoint
past_checkpoint_id = history[2].config["configurable"]["checkpoint_id"]
time_travel_config = {
    "configurable": {
        "thread_id": "conversation-1",
        "checkpoint_id": past_checkpoint_id
    }
}

# Continue from that point in history
result = graph.invoke({"messages": [...]}, config=time_travel_config)
```

### 4. Encrypted Checkpoints

For sensitive data, use encrypted serialization:

```python
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# Reads AES key from LANGGRAPH_AES_KEY environment variable
serde = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db"), serde=serde)

graph = builder.compile(checkpointer=checkpointer)
```

---

## Common Gotchas and Pitfalls

### 1. **Missing Checkpointer - Most Common Error**

**Problem**: Interrupts silently fail or raise errors when no checkpointer is configured.

```python
# ❌ WRONG - No checkpointer
graph = builder.compile()
# interrupt() calls will fail!

# ✅ CORRECT
from langgraph.checkpoint.memory import InMemorySaver
graph = builder.compile(checkpointer=InMemorySaver())
```

### 2. **Missing or Inconsistent thread_id**

**Problem**: Each invocation needs the same thread_id to maintain state.

```python
# ❌ WRONG - Different thread IDs
graph.invoke(inputs, config={"configurable": {"thread_id": "1"}})
graph.invoke(Command(resume=data), config={"configurable": {"thread_id": "2"}})
# State is lost! Different threads are isolated.

# ✅ CORRECT - Same thread ID
config = {"configurable": {"thread_id": "conversation-1"}}
graph.invoke(inputs, config=config)
graph.invoke(Command(resume=data), config=config)  # Same config
```

### 3. **Using InMemorySaver in Production**

**Problem**: InMemorySaver loses all state when the process restarts.

```python
# ❌ WRONG for production
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()  # Lost on restart!

# ✅ CORRECT for production
from langgraph.checkpoint.postgres import PostgresSaver
DB_URI = "postgresql://user:pass@localhost:5432/dbname"
checkpointer = PostgresSaver.from_conn_string(DB_URI)
checkpointer.setup()  # Create tables (run once)
```

### 4. **Incorrect Postgres Connection Setup**

**Problem**: Manual psycopg connections need specific parameters.

```python
# ❌ WRONG - Missing required parameters
import psycopg
with psycopg.connect(DB_URI) as conn:
    checkpointer = PostgresSaver(conn)
    # Will fail with: TypeError: tuple indices must be integers or slices, not str

# ✅ CORRECT - Use from_conn_string
checkpointer = PostgresSaver.from_conn_string(DB_URI)

# ✅ ALSO CORRECT - Manual connection with required params
with psycopg.connect(DB_URI, autocommit=True, row_factory=dict_row) as conn:
    checkpointer = PostgresSaver(conn)
```

### 5. **Forgetting to Resume with Command**

**Problem**: Trying to resume with regular input instead of Command object.

```python
# ❌ WRONG - Regular invoke after interrupt
result = graph.invoke(inputs, config)  # Hits interrupt
result = graph.invoke(inputs, config)  # Starts over, doesn't resume!

# ✅ CORRECT - Use Command to resume
from langgraph.types import Command
result = graph.invoke(inputs, config)  # Hits interrupt
result = graph.invoke(Command(resume=user_input), config)  # Resumes
```

### 6. **Not Passing None to Resume Static Breakpoints**

**Problem**: Static breakpoints require `None` as input to resume.

```python
# ❌ WRONG - Passing new input
graph.invoke(inputs, config)  # Hits breakpoint
graph.invoke(inputs, config)  # Starts execution over!

# ✅ CORRECT - Pass None to resume
graph.invoke(inputs, config)  # Hits breakpoint
graph.invoke(None, config)    # Resumes from breakpoint
```

### 7. **Double Texting / Concurrent Invocations**

**Problem**: New user input arrives while graph is still executing or paused.

**Considerations**:
- Interrupting concurrent execution saves work done so far, but may create edge cases
- Example: Tool call made but result not yet received creates a "dangling" tool call
- Need to handle cleanup in graph logic
- See the [double texting guide](https://docs.langchain.ai/docs/concepts/double_texting) for strategies

### 8. **Recursion Limits and Infinite Loops**

**Problem**: Graphs with loops can run indefinitely without proper termination.

```python
# Set recursion limit to prevent runaway execution
graph.invoke(
    inputs,
    config={
        "configurable": {"thread_id": "1"},
        "recursion_limit": 50  # Max 50 super-steps
    }
)
```

**Best Practice**: Use conditional edges to route to END node when done:
```python
def should_continue(state):
    if state["iterations"] > 10:
        return "end"
    return "continue"

builder.add_conditional_edges(
    "process_node",
    should_continue,
    {"continue": "process_node", "end": END}
)
```

### 9. **Interrupt Return Value Not Available (Old Versions)**

**Problem**: In versions before 0.4.0, `invoke()` doesn't return `__interrupt__`.

```python
# If on version < 0.4.0
# ❌ This won't work:
result = graph.invoke(inputs, config)
print(result["__interrupt__"])  # KeyError!

# ✅ Use stream instead:
for chunk in graph.stream(inputs, config):
    if "__interrupt__" in chunk:
        print(chunk["__interrupt__"])

# ✅ Or use get_state:
graph.invoke(inputs, config)
state = graph.get_state(config)
# Check state.tasks for interrupt info
```

### 10. **Interrupt in Subgraphs**

**Limitation**: Runtime interrupt configuration (`interrupt_before`/`interrupt_after` in invoke) is not supported for subgraphs.

```python
# ❌ WRONG - Runtime interrupts don't work for subgraphs
subgraph.invoke(inputs, interrupt_before=["node_a"])  # Won't work!

# ✅ CORRECT - Configure at compile time
subgraph = sub_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["node_a"]
)
```

---

## Best Practices

### 1. **Always Use Durable Checkpointers in Production**

```python
# Development
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()  # OK for dev/testing

# Production
from langgraph.checkpoint.postgres import PostgresSaver
checkpointer = PostgresSaver.from_conn_string(DB_URI)
checkpointer.setup()  # Run once
```

### 2. **Use Descriptive Thread IDs**

```python
# ❌ Generic
thread_id = "1"

# ✅ Descriptive
thread_id = f"user-{user_id}-conversation-{conversation_id}"
config = {"configurable": {"thread_id": thread_id}}
```

### 3. **Structure Interrupt Payloads Clearly**

```python
# ✅ Well-structured interrupt payload
interrupt({
    "type": "approval_required",
    "action": "delete_user",
    "user_id": user_id,
    "user_email": user_email,
    "warning": "This action cannot be undone",
    "timestamp": datetime.now().isoformat()
})
```

### 4. **Validate Human Input**

```python
def human_approval(state):
    while True:
        response = interrupt({"question": "Approve? (yes/no)"})

        if response.lower() in ["yes", "no"]:
            return {"approved": response.lower() == "yes"}
        else:
            # Ask again with error message
            continue
```

### 5. **Add Metadata to Checkpoints**

```python
# When manually creating checkpoints, include useful metadata
checkpointer.put(
    config,
    checkpoint_data,
    metadata={
        "user": "alice@example.com",
        "session": "debug-session-1",
        "environment": "production"
    },
    new_versions={}
)
```

### 6. **Use Type Hints for Clarity**

```python
from typing import Literal
from langgraph.types import Command

def approval_node(state: State) -> Command[Literal["approved", "rejected"]]:
    decision = interrupt({"question": "Approve?"})

    if decision == "approve":
        return Command(goto="approved")
    else:
        return Command(goto="rejected")
```

---

## TypeScript Examples

### Basic Interrupt Pattern

```typescript
import { interrupt, Command } from "@langchain/langgraph";
import { MemorySaver } from "@langchain/langgraph";

const humanApproval = (state: State): Command => {
  const decision = interrupt({
    question: "Do you approve the following output?",
    llmOutput: state.llmOutput
  });

  if (decision === "approve") {
    return new Command({
      goto: "approvedPath",
      update: { decision: "approved" }
    });
  } else {
    return new Command({
      goto: "rejectedPath",
      update: { decision: "rejected" }
    });
  }
};

const checkpointer = new MemorySaver();
const graph = builder.compile({ checkpointer });

// Run and resume
const config = { configurable: { thread_id: "1" } };
const result = await graph.invoke({}, config);
console.log(result.__interrupt__);

// Resume
await graph.invoke(new Command({ resume: "approve" }), config);
```

---

## Summary of Key Requirements

| Feature | Requirement | Notes |
|---------|-------------|-------|
| Dynamic Interrupts (`interrupt()`) | Checkpointer + thread_id | Required for persistence |
| Static Breakpoints | Checkpointer + thread_id | Same as interrupts |
| Resume Dynamic Interrupt | `Command(resume=value)` | Value is returned by interrupt() |
| Resume Static Breakpoint | `invoke(None, config)` | Pass None as input |
| Production Use | PostgresSaver or SqliteSaver | Never InMemorySaver |
| Concurrent Threads | Unique thread_id per thread | Ensures isolation |
| Time Travel | checkpoint_id in config | Optional - defaults to latest |

---

## Reference Links

- [Official Human-in-the-Loop Guide](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/how-tos/human_in_the_loop/add-human-in-the-loop.md)
- [Persistence Concepts](https://github.com/langchain-ai/langgraph/blob/main/docs/docs/concepts/persistence.md)
- [Checkpointer Libraries](https://github.com/langchain-ai/langgraph/tree/main/libs/checkpoint)
- [PostgresSaver README](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-postgres/README.md)
- [SqliteSaver README](https://github.com/langchain-ai/langgraph/blob/main/libs/checkpoint-sqlite/README.md)
