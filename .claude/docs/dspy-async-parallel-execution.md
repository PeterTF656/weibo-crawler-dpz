# DSPy Async & Parallel Execution Research

**Date**: 2025-12-12
**Context**: Investigation into why LangGraph parallel nodes are executing LLM calls sequentially
**Codebase**: Level Agent Phase 3 parallel property extraction

---

## Executive Summary

**Finding**: The 3 parallel extraction nodes in Phase 3 are executing **sequentially** instead of in parallel due to **synchronous DSPy module calls** that block thread execution.

**Root Cause**:
- LangGraph correctly runs nodes in parallel using `ThreadPoolExecutor`
- But DSPy modules use synchronous `litellm.completion()` calls that block threads
- Each thread waits for the previous HTTP request to complete before starting the next

**Solution**: Convert nodes to async and use DSPy's async API (`acall()` + `aforward()`)

**Performance Impact**:
- Current: **~4.2s** (sequential LLM calls)
- Expected: **~2.3s** (parallel LLM calls)
- **Savings: ~1.9s per query** (~45% improvement)

---

## Investigation Timeline

### Test 1: LangGraph Parallel Execution Model

**Objective**: Determine if LangGraph supports parallel node execution

**Test Code**:
```python
# 3 sync nodes with 0.5s sleep each
def sync_node_a(state): time.sleep(0.5); return {"a": "done"}
def sync_node_b(state): time.sleep(0.5); return {"b": "done"}
def sync_node_c(state): time.sleep(0.5); return {"c": "done"}

graph.add_conditional_edges(START, route_parallel, ["node_a", "node_b", "node_c"])
# All nodes fan-in to merge node
```

**Result**: ✅ **0.51s** (expected ~0.5s if parallel, ~1.5s if sequential)

**Conclusion**: LangGraph DOES run sync nodes in parallel using `ThreadPoolExecutor`

---

### Test 2: DSPy Async Support

**Objective**: Check if DSPy supports async execution

**Findings**:

| Component | Sync Method | Async Method | Is Async? |
|-----------|-------------|--------------|-----------|
| `dspy.Module` | `__call__()` | `acall()` | ✅ Yes |
| `dspy.Module` | `forward()` | `aforward()` | ✅ Yes |
| `dspy.LM` | `__call__()` | `acall()` | ✅ Yes |
| `dspy.LM` | `forward()` | `aforward()` | ✅ Yes |
| LiteLLM | `completion()` | `acompletion()` | ✅ Yes |

**Internal Execution Path**:

```
SYNC PATH (Current - Sequential):
extractor()
  → dspy.Module.__call__()
    → self.forward()
      → self.extractor() (another dspy.Module)
        → dspy.LM.forward()
          → litellm_completion()  ← BLOCKS THREAD HERE
            → httpx.Client.post()  (synchronous HTTP)

ASYNC PATH (Proposed - Parallel):
await extractor.acall()
  → dspy.Module.acall()
    → await self.aforward()
      → await self.extractor.acall()
        → await dspy.LM.aforward()
          → await alitellm_completion()  ← NON-BLOCKING
            → await httpx.AsyncClient.post()  (async HTTP)
```

---

### Test 3: DSPy Async Parallel Execution

**Test Code**:
```python
class TestModule(dspy.Module):
    async def aforward(self, text: str) -> str:
        await asyncio.sleep(0.5)
        return f"async: {text}"

# Run 3 in parallel
results = await asyncio.gather(
    module.acall(text="task1"),
    module.acall(text="task2"),
    module.acall(text="task3")
)
```

**Result**: ✅ **0.50s** (expected ~0.5s)

**Conclusion**: DSPy `acall()` + `aforward()` supports true parallel execution

---

## Root Cause Analysis

### Why Are LLM Calls Sequential?

1. **LangGraph spawns 3 parallel threads** ✅
   ```
   12:03:15.066 → Thread 1: time_property_extraction_node starts
   12:03:15.078 → Thread 2: nontime_property_extraction_node starts (+12ms)
   12:03:15.093 → Thread 3: vector_search_extraction_node starts (+27ms)
   ```

2. **Each thread calls synchronous DSPy module** ❌
   ```python
   # phase3_property_extraction.py:145
   extractor = TimePropertyExtractor(lm=get_qwen3_235b_a22b_instruct())
   time_props = extractor(...)  # ← BLOCKS thread until HTTP completes
   ```

3. **LiteLLM makes synchronous HTTP calls** ❌
   ```python
   # dspy.LM.forward() internally calls:
   response = litellm.completion(  # ← Blocking I/O
       model="openrouter/openai/gpt-oss-120b",
       messages=[...],
   )
   ```

4. **HTTP requests execute sequentially** ❌
   ```
   12:03:16.321 → HTTP Request 1 (Time)     [Thread 1 blocked]
   12:03:17.047 → HTTP Request 2 (NonTime)  [Thread 2 blocked] (+726ms)
   12:03:18.172 → HTTP Request 3 (Vector)   [Thread 3 blocked] (+1125ms)
   ```

### Why Does This Happen?

**Synchronous blocking I/O in threaded environments**:
- Each thread makes a blocking HTTP call
- Python's GIL (Global Interpreter Lock) doesn't help here - the blocking is in I/O, not CPU
- `httpx.Client` (sync) serializes requests through connection pooling
- Even with multiple threads, sync HTTP clients often share connection pools

**The Fix**: Use async/await which allows the event loop to interleave I/O operations:
- Thread 1: Start HTTP request, yield control
- Thread 2: Start HTTP request, yield control
- Thread 3: Start HTTP request, yield control
- All 3 requests are now in-flight simultaneously
- Event loop wakes each coroutine when its response arrives

---

## Solution Architecture

### Current Implementation (Sequential)

```python
# Sync node (blocks thread)
def time_property_extraction_node(state: LevelState, config: RunnableConfig) -> LevelState:
    extractor = TimePropertyExtractor(lm=get_qwen3_235b_a22b_instruct())
    time_props = extractor(...)  # ← Blocks until HTTP completes
    return {"time_extraction_result": time_props}
```

### Proposed Implementation (Parallel)

```python
# Async node (non-blocking)
async def time_property_extraction_node(state: LevelState, config: RunnableConfig) -> LevelState:
    extractor = TimePropertyExtractor(lm=get_qwen3_235b_a22b_instruct())
    time_props = await extractor.acall(...)  # ← Non-blocking, yields control
    return {"time_extraction_result": time_props}
```

**Required Changes**:

1. **Node functions**: Convert from `def` to `async def`
2. **Module invocation**: Change `extractor()` to `await extractor.acall()`
3. **Module methods**: Add `async def aforward()` to all 3 extractors
4. **Graph invocation**: Use `await graph.ainvoke()` instead of `graph.invoke()`

---

## Implementation Checklist

### Phase 3 Nodes (phase3_property_extraction.py)

- [ ] Convert `time_property_extraction_node` to `async def`
- [ ] Convert `nontime_property_extraction_node` to `async def`
- [ ] Convert `vector_search_extraction_node` to `async def`
- [ ] Replace `extractor()` calls with `await extractor.acall()`

### Extractor Modules

**time_property_extractor.py**:
- [ ] Add `async def aforward()` method
- [ ] Use `await self.lm.aforward()` or DSPy async patterns

**nontime_property_extractor.py**:
- [ ] Add `async def aforward()` method
- [ ] Use `await self.lm.aforward()` or DSPy async patterns

**vector_search_extractor.py**:
- [ ] Add `async def aforward()` method
- [ ] Use `await self.lm.aforward()` or DSPy async patterns

### Graph Invocation (agent_simplified.py)

- [ ] Convert `invoke_level_agent()` to async
- [ ] Use `await graph.ainvoke()` instead of `graph.invoke()`
- [ ] Update all callers to use `await invoke_level_agent()`

---

## Expected Performance Improvement

### Current Timeline (Sequential)
```
Phase 3 Start:        12:03:15.066
Time LLM:             12:03:16.321 (+1.255s)
NonTime LLM:          12:03:17.047 (+0.726s) ← 726ms wasted
Vector LLM:           12:03:18.172 (+1.125s) ← 1125ms wasted
Phase 3 End:          12:03:19.325 (4.259s total)
```

### Expected Timeline (Parallel)
```
Phase 3 Start:        12:03:15.066
All 3 LLMs fire:      12:03:15.070 (nearly simultaneous)
Longest completes:    12:03:17.370 (~2.3s - longest single request)
Phase 3 End:          12:03:17.400 (2.334s total)
```

**Improvement**: 4.259s → 2.334s = **1.925s saved per query** (~45% faster)

---

## Technical Details

### DSPy Async API

**Module Methods**:
```python
# Synchronous (current)
result = module(arg1, arg2)
result = module.forward(arg1, arg2)

# Asynchronous (proposed)
result = await module.acall(arg1, arg2)
result = await module.aforward(arg1, arg2)
```

**LM Methods**:
```python
# Synchronous
with dspy.context(lm=my_lm):
    result = module(...)  # Uses lm.forward() → litellm.completion()

# Asynchronous
with dspy.context(lm=my_lm):
    result = await module.acall(...)  # Uses lm.aforward() → litellm.acompletion()
```

### LangGraph Async API

**Node Definition**:
```python
# Synchronous
def my_node(state: State, config: RunnableConfig) -> State:
    return {"key": "value"}

# Asynchronous
async def my_node(state: State, config: RunnableConfig) -> State:
    return {"key": "value"}
```

**Graph Invocation**:
```python
# Synchronous (current)
result = graph.invoke(initial_state)

# Asynchronous (proposed)
result = await graph.ainvoke(initial_state)
```

**Important**: If nodes are async, you MUST use `ainvoke()`, not `invoke()`
- Error: `TypeError: No synchronous function provided to "node_b"`

---

## LiteLLM Internals

### Completion Functions

**Synchronous (current)**:
```python
# dspy/clients/lm.py
def litellm_completion(request, num_retries, cache):
    return litellm.completion(**request)  # Blocking HTTP call
```

**Asynchronous (proposed)**:
```python
# dspy/clients/lm.py
async def alitellm_completion(request, num_retries, cache):
    return await litellm.acompletion(**request)  # Non-blocking HTTP call
```

### HTTP Client

**Synchronous path**:
```python
httpx.Client().post("https://openrouter.ai/api/v1/chat/completions")
# Blocks thread until response received
```

**Asynchronous path**:
```python
await httpx.AsyncClient().post("https://openrouter.ai/api/v1/chat/completions")
# Yields control, allows other tasks to run
```

---

## Verification Tests

### Test 1: Confirm Async Conversion Works
```python
import asyncio
from src.tools.graphdb.search.nl_to_cypher.sequential.agent.level.phases.modules.time_property_extractor import TimePropertyExtractor

async def test():
    extractor = TimePropertyExtractor(lm=get_qwen3_235b_a22b_instruct())
    result = await extractor.acall(
        branch_intent="test",
        selected_label="Idea",
        time_properties=["created_at"],
        current_datetime="2025-12-12T00:00:00Z"
    )
    print(f"Result: {result}")

asyncio.run(test())
```

### Test 2: Verify Parallel Execution
```python
import asyncio
import time

async def test_parallel():
    extractor1 = TimePropertyExtractor(lm=get_qwen3_235b_a22b_instruct())
    extractor2 = NonTimePropertyExtractor(lm=get_gpt_oss_120b())
    extractor3 = VectorSearchExtractor(lm=get_gpt_oss_120b())

    start = time.perf_counter()
    results = await asyncio.gather(
        extractor1.acall(...),
        extractor2.acall(...),
        extractor3.acall(...)
    )
    duration = time.perf_counter() - start

    print(f"Duration: {duration:.2f}s (should be ~2.3s, not ~4.2s)")

asyncio.run(test_parallel())
```

---

## References

- DSPy Documentation: https://dspy-docs.vercel.app/
- DSPy GitHub: https://github.com/stanfordnlp/dspy
- LangGraph Async: https://langchain-ai.github.io/langgraph/
- LiteLLM Async: https://docs.litellm.ai/docs/completion/async

---

## Version Info

- **DSPy**: 3.0.4
- **LangGraph**: (check via `pip show langgraph`)
- **LiteLLM**: (check via `pip show litellm`)
- **Python**: 3.13
