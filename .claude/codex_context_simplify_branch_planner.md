# Context: Simplify branch_planner.py (Preserve All Functionality)

## Task Overview

Implement all simplifications recommended by code-simplifier while **PRESERVING ALL FUNCTIONALITY**.

**Critical Requirements**:
- ✅ All existing functionality must work exactly as before
- ✅ No behavioral changes
- ✅ DSPy integration must remain intact
- ✅ MLflow tracking must work correctly
- ✅ Tool tracking/observability must be preserved
- ✅ Both sync and async methods must work identically
- ✅ Validate with syntax check after changes

---

## Simplification Priority List

### 1. CRITICAL: Extract Duplicated Logic (Priority 1)

**Location**: `forward()` and `aforward()` methods (lines 397-731)

**Problem**: ~165 lines of nearly identical code duplicated between sync and async versions.

**Only Difference**:
```python
# forward() - Line 455
prediction = react_module(
    user_query=user_query,
    branch_intent=branch_intent,
    topology_context=topology_context,
    known_data=known_data,
)

# aforward() - Line 624
prediction = await react_module.acall(
    user_query=user_query,
    branch_intent=branch_intent,
    topology_context=topology_context,
    known_data=known_data,
)
```

**Everything Else is Duplicated**:
1. Setup (lines 429-450 / 598-619):
   - Get topology context
   - Log branch_intent
   - Create tool_tracker
   - Determine effective_lm
   - Create react_module
2. Tracking extraction (lines 463-467 / 631-635)
3. Tool usage logging (lines 469-482 / 637-650)
4. Token usage logging (lines 484-491 / 652-660)
5. Tracking data compilation (lines 493-503 / 662-672)
6. MLflow span tracking (lines 505-537 / 674-706)
7. Debug logging (lines 539-549 / 708-718)
8. Validation (line 552 / 721)
9. Final logging (lines 554-557 / 723-726)
10. Return statement (lines 559-562 / 728-731)

**Solution**: Extract into private helper methods:

```python
def _prepare_execution(
    self,
    branch_intent: str,
    mlflow_span: Any | None,
    lm: dspy.LM | None,
) -> tuple[dspy.ReAct, Any, dspy.LM, Dict[str, Any]]:
    """Prepare ReAct module, tool tracker, effective LM, and timing dict.

    Returns:
        (react_module, tool_tracker, effective_lm, timing)
    """
    timing: Dict[str, Any] = {}

    # Get topology context (cached after first call)
    topology_context = self._get_topology_context()

    logger.info(f"Planning path for: {branch_intent[:50]}...")

    # Create fresh tool tracker for this call
    tool_tracker = self._tracker_class(mlflow_parent_span=mlflow_span)
    tracked_tools = tool_tracker.wrap_tools(self._base_tools)

    # LM priority: call-level > module-level > global settings
    effective_lm = lm or self._lm or dspy.settings.lm

    # Create ReAct module with tracked tools
    react_module = dspy.ReAct(
        self._signature,
        tools=tracked_tools,
        max_iters=self._max_iters,
    )

    return react_module, tool_tracker, effective_lm, timing, topology_context


def _process_results(
    self,
    prediction: dspy.Prediction,
    effective_lm: dspy.LM,
    tool_tracker: Any,
    timing: Dict[str, Any],
    mlflow_span: Any | None,
    user_query: str,
    branch_intent: str,
    known_data: Dict[str, Any],
    is_async: bool = False,
) -> dspy.Prediction:
    """Process prediction results: tracking, validation, and return.

    Args:
        prediction: Raw prediction from ReAct module
        effective_lm: The LM used for this call
        tool_tracker: Tool call tracker
        timing: Timing information dict
        mlflow_span: Optional MLflow span
        user_query: Original query
        branch_intent: Branch intent string
        known_data: Context data
        is_async: Whether this is async call (for logging)

    Returns:
        Final dspy.Prediction with validated planned_levels and tracking
    """
    # Extract tracking data
    lm_history = extract_lm_history(effective_lm, logger)
    token_usage = extract_token_usage(prediction, logger)
    tool_usage = tool_tracker.get_summary()

    # Log tool usage summary
    if tool_usage["total_calls"] > 0:
        logger.info(
            f"[Tool Usage] Total calls: {tool_usage['total_calls']}, "
            f"by tool: {tool_usage['call_counts']}, "
            f"total duration: {tool_usage['total_duration_ms']}ms"
        )
        # Log individual calls
        for call in tool_usage["calls"]:
            logger.info(
                f"  - {call['tool']} #{call['call_index'] + 1}: "
                f"input={call['input']}, duration={call['duration_ms']}ms"
            )
            logger.debug(f"    output: {call['output'][:200]}...")
    else:
        logger.info("[Tool Usage] No tools were called")

    # Log token usage
    if token_usage:
        logger.info(
            f"[Token Usage] input={token_usage.get('input_tokens', 0)}, "
            f"output={token_usage.get('output_tokens', 0)}, "
            f"total={token_usage.get('total_tokens', 0)}"
        )

    # Compile tracking data
    tracking_data: Dict[str, Any] = {
        "timing": timing,
        "token_usage": token_usage,
        "tool_usage": tool_usage,
        "lm_history": {
            "system_prompt": lm_history.get("system_prompt"),
            "user_prompt": lm_history.get("user_prompt"),
            "assistant_response": lm_history.get("assistant_response"),
            "cost": lm_history.get("cost"),
        },
    }

    # Set MLflow span tracking if provided
    if mlflow_span:
        set_mlflow_span_tracking(
            span=mlflow_span,
            history_data=lm_history,
            token_usage=token_usage,
            output_response={"planned_levels": prediction.planned_levels},
            fallback_inputs={
                "user_query": user_query,
                "branch_intent": branch_intent,
                "known_data": known_data,
            },
            timing_breakdown=timing,
            custom_logger=logger,
        )
        # Add tool usage to span attributes
        if tool_usage["total_calls"] > 0:
            mlflow_span.set_attributes(
                {
                    "tool_total_calls": tool_usage["total_calls"],
                    "tool_call_counts": str(tool_usage["call_counts"]),
                    "tool_total_duration_ms": tool_usage["total_duration_ms"],
                }
            )
            # Add individual tool call details
            for i, call in enumerate(tool_usage["calls"]):
                mlflow_span.set_attributes(
                    {
                        f"tool_call_{i}_name": call["tool"],
                        f"tool_call_{i}_input": str(call["input"]),
                        f"tool_call_{i}_duration_ms": call["duration_ms"],
                    }
                )

    # Debug: Log raw prediction output before validation
    raw_planned_levels = prediction.planned_levels
    logger.debug(f"Raw prediction.planned_levels type: {type(raw_planned_levels)}")
    logger.debug(f"Raw prediction.planned_levels value: {raw_planned_levels}")
    if isinstance(raw_planned_levels, list) and len(raw_planned_levels) > 0:
        logger.debug(
            f"First item type: {type(raw_planned_levels[0])}, "
            f"value: {raw_planned_levels[0]}"
        )

    # Validate minimal schema
    planned_levels = self._validate_output_schema(prediction.planned_levels)

    async_suffix = " (async)" if is_async else ""
    logger.info(
        f"Planned {len(planned_levels)} levels{async_suffix}: "
        f"{[level['label'] for level in planned_levels]}"
    )

    return dspy.Prediction(
        planned_levels=planned_levels,
        tracking=tracking_data,
    )
```

**New forward() Implementation**:
```python
def forward(
    self,
    branch_intent: str,
    user_query: str = "",
    known_data: Dict[str, Any] | None = None,
    lm: dspy.LM | None = None,
    mlflow_span: Any | None = None,
) -> dspy.Prediction:
    """Execute path planning with tool-augmented reasoning.

    [Keep existing docstring]
    """
    known_data = known_data or {}

    # Prepare execution
    react_module, tool_tracker, effective_lm, timing, topology_context = (
        self._prepare_execution(branch_intent, mlflow_span, lm)
    )

    # Execute with timing (SYNC version)
    with track_duration("llm_call", timing, logger):
        with dspy.context(lm=effective_lm):
            prediction = react_module(
                user_query=user_query,
                branch_intent=branch_intent,
                topology_context=topology_context,
                known_data=known_data,
            )

    # Process results and return
    return self._process_results(
        prediction=prediction,
        effective_lm=effective_lm,
        tool_tracker=tool_tracker,
        timing=timing,
        mlflow_span=mlflow_span,
        user_query=user_query,
        branch_intent=branch_intent,
        known_data=known_data,
        is_async=False,
    )
```

**New aforward() Implementation**:
```python
async def aforward(
    self,
    branch_intent: str,
    user_query: str = "",
    known_data: Dict[str, Any] | None = None,
    lm: dspy.LM | None = None,
    mlflow_span: Any | None = None,
) -> dspy.Prediction:
    """Execute path planning with tool-augmented reasoning (ASYNC version).

    [Keep existing docstring]
    """
    known_data = known_data or {}

    # Prepare execution (same as sync)
    react_module, tool_tracker, effective_lm, timing, topology_context = (
        self._prepare_execution(branch_intent, mlflow_span, lm)
    )

    # Execute with timing (ASYNC version)
    with track_duration("llm_call", timing, logger):
        with dspy.context(lm=effective_lm):
            prediction = await react_module.acall(
                user_query=user_query,
                branch_intent=branch_intent,
                topology_context=topology_context,
                known_data=known_data,
            )

    # Process results and return (same as sync)
    return self._process_results(
        prediction=prediction,
        effective_lm=effective_lm,
        tool_tracker=tool_tracker,
        timing=timing,
        mlflow_span=mlflow_span,
        user_query=user_query,
        branch_intent=branch_intent,
        known_data=known_data,
        is_async=True,
    )
```

**Critical**: topology_context needs to be returned from _prepare_execution and passed to react_module call!

---

### 2. Remove Unused Import (Priority 2)

**Location**: Lines 29-31

**Current**:
```python
from src.tools.graphdb.search.nl_to_cypher.sequential.schema_generator import (
    get_properties_for_label,  # noqa: F401 - Used as LLM tool
)
```

**Why It's Unused**: The actual tool is imported at lines 276-279:
```python
from src.tools.graphdb.search.nl_to_cypher.sequential.branch.dspy.modules.tools import (
    ToolCallTracker,
    properties_tool,
)
```

**Fix**: Delete lines 29-31

---

### 3. Fix Debug Logging Level (Priority 3)

**Location**: Multiple places (lines 539-549, 708-718 in original)

**Current**:
```python
logger.info(f"[DEBUG] Raw prediction.planned_levels type: {type(raw_planned_levels)}")
logger.info(f"[DEBUG] Raw prediction.planned_levels value: {raw_planned_levels}")
logger.info(f"[DEBUG] First item type: {type(raw_planned_levels[0])}, value: {raw_planned_levels[0]}")
```

**Fix**: Change to `logger.debug()` and remove "[DEBUG]" prefix:
```python
logger.debug(f"Raw prediction.planned_levels type: {type(raw_planned_levels)}")
logger.debug(f"Raw prediction.planned_levels value: {raw_planned_levels}")
logger.debug(f"First item type: {type(raw_planned_levels[0])}, value: {raw_planned_levels[0]}")
```

**Already done in _process_results helper above**

---

### 4. Remove F-String Concatenation (Priority 4)

**Location**: Multiple logging statements

**Pattern to Fix**:
```python
# BEFORE (with + operator)
logger.info(
    f"[Tool Usage] Total calls: {tool_usage['total_calls']}, "
    + f"by tool: {tool_usage['call_counts']}, "
    + f"total duration: {tool_usage['total_duration_ms']}ms"
)

# AFTER (no + operator)
logger.info(
    f"[Tool Usage] Total calls: {tool_usage['total_calls']}, "
    f"by tool: {tool_usage['call_counts']}, "
    f"total duration: {tool_usage['total_duration_ms']}ms"
)
```

**Locations to fix** (in original code):
- Lines 469-472, 477-478
- Lines 487-489
- Lines 555-556
- Lines 638-641, 645-646
- Lines 655-658
- Lines 723-725

**Already done in _process_results helper above**

---

### 5. Update Factory Function Docstring (Priority 5)

**Location**: Lines 734-762 (after refactoring will be different)

**Current Docstring Issue**:
```python
"""Create PathPlanner with tools for path planning.

Features:
- Minimal output schema: {label, hop_index, reasoning} per level
```

**Fix**: Update "Minimal output schema" to "6-field output schema":
```python
"""Create PathPlanner with tools for path planning.

Features:
- 6-field output schema: {label, hop_index, reasoning, temporal_filter, ownership_filter, vector_search} per level
```

---

## Testing Requirements

After implementing all changes:

1. **Syntax Validation**:
   ```bash
   python3 -m py_compile src/tools/graphdb/search/nl_to_cypher/sequential/branch/dspy/modules/branch_planner.py
   ```

2. **Import Validation**:
   ```bash
   python3 -c "from src.tools.graphdb.search.nl_to_cypher.sequential.branch.dspy.modules.branch_planner import PathPlanner; print('✓ Import successful')"
   ```

3. **Verify No Behavioral Changes**:
   - Both forward() and aforward() should produce identical outputs (when comparing sync vs async execution)
   - All tracking data should be preserved
   - MLflow integration should work
   - Tool tracking should work

---

## Critical Reminders

1. ✅ **Preserve exact behavior** - no functional changes
2. ✅ **Keep all docstrings** in forward() and aforward() methods
3. ✅ **topology_context must be returned** from _prepare_execution and used in react_module call
4. ✅ **Logger calls** - use logger.debug() for debug logs
5. ✅ **F-string concatenation** - remove + operators
6. ✅ **Test after changes** - syntax and import validation
7. ✅ **Line numbers will change** - the specific line numbers in this doc are from original code

---

## Expected Outcome

**Before**: 756 lines
**After**: ~620 lines (estimate)

**Reduction**: ~136 lines eliminated through extraction of duplicated logic

**Benefits**:
- Single source of truth for setup/tracking logic
- Easier maintenance (fix bugs once, not twice)
- Clearer code structure
- Same functionality, better organization

---

## Reference Files

**Must Read**:
1. `src/tools/graphdb/search/nl_to_cypher/sequential/branch/dspy/modules/branch_planner.py` - File to refactor

**For Context** (if needed):
2. `src/tools/graphdb/search/nl_to_cypher/sequential/branch/docs/phase1.md` - Understanding the module
3. `src/tools/graphdb/search/nl_to_cypher/sequential/branch/model.py` - PlanningLevel schema

---

## Codex Execution Instructions

**Model**: gpt-5.2-codex (coding task - refactoring)
**Reasoning**: xhigh (complex refactoring requiring extreme care)
**Sandbox**: danger-full-access (always required)

**Task**:
1. Read branch_planner.py carefully
2. Implement all 5 simplifications in order (Priority 1 → Priority 5)
3. Preserve all functionality exactly as-is
4. Validate syntax and imports after changes
5. Report what was changed and confirm all tests pass
