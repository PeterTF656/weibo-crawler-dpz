# Async Migration Plan - Level Agent Phase 3

**Objective**: Convert Phase 3 parallel extraction nodes from sync to async to achieve true parallel LLM execution

**Expected Performance**: 4.3s → 2.3s (~45% improvement)

**Risk Level**: HIGH - Core execution path changes

---

## Migration Strategy: Additive First, Then Replace

**Principle**: Add async methods WITHOUT removing sync methods initially. This allows:
- Testing async versions in isolation
- Easy rollback if issues arise
- Gradual migration with backward compatibility
- Verification at each step

---

## Stage 1: Add Async to TimePropertyExtractor ✓

**File**: `src/tools/graphdb/search/nl_to_cypher/sequential/agent/level/phases/modules/time_property_extractor.py`

**Changes**:
1. Add `async def aforward()` method (keeps existing `forward()`)
2. Use DSPy async patterns: `await self.lm.aforward()` OR rely on DSPy's internal async handling
3. Test in isolation

**Rollback**: Simply delete the new `aforward()` method

**Verification**:
```python
# Test async version
extractor = TimePropertyExtractor(lm=get_qwen3_235b_a22b_instruct())
result = await extractor.acall(...)
assert result is not None
```

---

## Stage 2: Add Async to NonTimePropertyExtractor ✓

**File**: `src/tools/graphdb/search/nl_to_cypher/sequential/agent/level/phases/modules/nontime_property_extractor.py`

**Changes**:
1. Add `async def aforward()` method
2. Test in isolation

**Rollback**: Delete `aforward()` method

---

## Stage 3: Add Async to VectorSearchExtractor ✓

**File**: `src/tools/graphdb/search/nl_to_cypher/sequential/agent/level/phases/modules/vector_search_extractor.py`

**Changes**:
1. Add `async def aforward()` method
2. Test in isolation

**Rollback**: Delete `aforward()` method

---

## Stage 4: Convert time_property_extraction_node to Async ⚠️

**File**: `src/tools/graphdb/search/nl_to_cypher/sequential/agent/level/phases/phase3_property_extraction.py`

**Changes**:
```python
# BEFORE
def time_property_extraction_node(state: LevelState, config: RunnableConfig) -> LevelState:
    extractor = TimePropertyExtractor(lm=get_qwen3_235b_a22b_instruct())
    time_props = extractor(...)  # Sync call
    return {"time_extraction_result": time_props}

# AFTER
async def time_property_extraction_node(state: LevelState, config: RunnableConfig) -> LevelState:
    extractor = TimePropertyExtractor(lm=get_qwen3_235b_a22b_instruct())
    time_props = await extractor.acall(...)  # Async call
    return {"time_extraction_result": time_props}
```

**CRITICAL**: This breaks the graph unless ALL parallel nodes are async

**Rollback**: Revert function signature to `def` and call to `extractor()`

---

## Stage 5: Convert nontime_property_extraction_node to Async ⚠️

**File**: Same as Stage 4

**Changes**: Convert to `async def`, use `await extractor.acall()`

**Rollback**: Revert to sync

---

## Stage 6: Convert vector_search_extraction_node to Async ⚠️

**File**: Same as Stage 4

**Changes**: Convert to `async def`, use `await extractor.acall()`

**Rollback**: Revert to sync

**IMPORTANT**: Stages 4-6 should be done together in a single commit for atomicity

---

## Stage 7: Update Graph Invocation to Async ⚠️⚠️

**File**: `src/tools/graphdb/search/nl_to_cypher/sequential/agent/level/agent_simplified.py`

**Changes**:
```python
# BEFORE
def invoke_level_agent(state: LevelState, config: RunnableConfig):
    graph = build_level_graph()
    result = graph.invoke(state, config)
    return result

# AFTER
async def invoke_level_agent(state: LevelState, config: RunnableConfig):
    graph = build_level_graph()
    result = await graph.ainvoke(state, config)
    return result
```

**CRITICAL**: All callers must be updated to use `await invoke_level_agent()`

**Rollback**: Revert to sync `invoke()`

---

## Stage 8: Update All Callers ⚠️⚠️⚠️

**Files to Update**: Any file that calls `invoke_level_agent()`

**Search Pattern**: `rg "invoke_level_agent" -t py`

**Changes**: Convert all calling functions to async and use `await`

**Risk**: HIGHEST - May affect multiple modules

---

## Rollback Plan

### Quick Rollback (Any Stage)
```bash
git checkout src/tools/graphdb/search/nl_to_cypher/sequential/agent/level/
```

### Selective Rollback
- Stage 1-3: Delete `aforward()` methods from extractors
- Stage 4-6: Revert node functions to `def` and sync calls
- Stage 7-8: Revert graph invocation to sync

---

## Testing Strategy

### After Each Stage

**Unit Test**:
```python
# Test async method works
async def test_extractor():
    extractor = TimePropertyExtractor(lm=get_qwen3_235b_a22b_instruct())
    result = await extractor.acall(
        branch_intent="test",
        selected_label="Idea",
        time_properties=["created_at"],
        current_datetime="2025-12-12T00:00:00Z"
    )
    assert result is not None
    print(f"✅ Stage test passed: {result}")
```

### After Stages 4-6 (All Nodes Async)

**Integration Test**:
```python
async def test_parallel_nodes():
    import asyncio
    import time

    # Build graph with async nodes
    graph = build_level_graph()

    state = {
        "current_label": "Composition",
        "target_label": "Idea",
        "branch_intent": "Find ideas from user compositions",
        "time_properties": ["created_at"],
        "non_time_properties": ["mongodb_id"],
        "supports_vector_search": True,
        "vector_search_types": ["Title", "Logic"]
    }

    start = time.perf_counter()
    result = await graph.ainvoke(state, config)
    duration = time.perf_counter() - start

    print(f"Duration: {duration:.2f}s")
    print(f"Expected: ~2.3s (parallel), Current baseline: ~4.3s")

    if duration < 3.0:
        print("✅ PARALLEL EXECUTION ACHIEVED!")
    else:
        print("❌ Still sequential - investigate")
```

### After Stage 8 (End-to-End)

**Full System Test**:
- Run actual API endpoint
- Monitor logs for parallel timing
- Verify results match previous implementation

---

## Success Criteria

✅ **Stage 1-3**: Each extractor has working `aforward()` method
✅ **Stage 4-6**: All 3 nodes are async, graph compiles without errors
✅ **Stage 7**: Graph invocation uses `await ainvoke()`
✅ **Stage 8**: All HTTP requests start within 50ms of each other
✅ **Final**: Phase 3 duration < 3.0s (down from 4.3s)

---

## Risk Mitigation

1. **Backup First**: Create branch before starting
2. **One Stage at a Time**: Get user approval before each stage
3. **Test After Each Stage**: Don't proceed until tests pass
4. **Monitor Logs**: Watch for timing changes
5. **Keep Sync Methods**: Don't delete until fully verified

---

## Current Status

- [✅] Stage 1: TimePropertyExtractor async
- [✅] Stage 2: NonTimePropertyExtractor async
- [✅] Stage 3: VectorSearchExtractor async
- [✅] Stage 4-6: All nodes async (atomic commit)
- [✅] Stage 7: Graph invocation async
- [✅] Stage 8: Update all callers
- [ ] Verification: Parallel execution confirmed (NEXT STEP)

---

## Next Step

**Start with Stage 1**: Add `aforward()` to TimePropertyExtractor

**User Approval Required**: Proceed to Stage 1?
