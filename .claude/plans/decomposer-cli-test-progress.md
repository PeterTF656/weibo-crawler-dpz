# Progress Log: Decomposer CLI Test

## Session: 2026-01-14

### Actions Taken

1. **[DONE] Read existing codebase**
   - Explored decomposer module structure
   - Read unit_test_pipeline.py (existing test)
   - Read cli_test_step_a.py (reference pattern)
   - Read lm.py (LLM configurations)
   - Read mlflow_setup.py (tracing setup)

2. **[DONE] Created planning files**
   - task_plan.md
   - findings.md
   - progress.md

3. **[DONE] Design CLI test**
   - CLI arguments: --query, --lm, --json, --verbose, --list, --all, --test-case, --async, --no-mlflow
   - MLflow integration via mlflow.start_span() for async mode
   - Support both sync and async modes

4. **[DONE] Implement cli_test_decomposer.py**
   - Created file at tests/nl_to_cypher/decomposer/cli_test_decomposer.py
   - Features:
     - 15 LLM model options (OpenRouter + Tuzi)
     - 8 sample test cases
     - Pretty and JSON output formats
     - Batch testing with validation
     - MLflow tracing integration

5. **[DONE] Fixed MLflow tracing issue**
   - **Root Cause:** `mlflow.start_span()` requires an active trace context
   - **Fix:** Added `@mlflow.trace()` decorator to create trace context first
   - Created `_run_decomposer_with_trace()` helper function with decorator
   - Child spans now properly recorded within the trace

### Next Steps
- [ ] Test with sample queries
- [ ] Verify MLflow traces appear in UI

---

## Files Modified
| File | Action | Status |
|------|--------|--------|
| tests/nl_to_cypher/decomposer/cli_test_decomposer.py | Create | Done |
| .claude/plans/decomposer-cli-test-*.md | Create | Done |

---

## Test Commands (To Run Later)
```bash
# Basic usage
python tests/nl_to_cypher/decomposer/cli_test_decomposer.py \
    --query "Find my compositions about school"

# With specific LM
python tests/nl_to_cypher/decomposer/cli_test_decomposer.py \
    --query "Find compositions about health" \
    --lm gpt_4o_mini

# Run all sample tests
python tests/nl_to_cypher/decomposer/cli_test_decomposer.py --all

# JSON output
python tests/nl_to_cypher/decomposer/cli_test_decomposer.py \
    --query "Find compositions" --json
```
