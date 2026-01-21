# Task Plan: Decomposer CLI Test Tool

## Goal
Create a CLI test file for the decomposer module that supports:
- Custom input queries
- LM configurations (multiple model options)
- MLflow tracing integration
- Pretty and JSON output formats

**Target Output:** `tests/nl_to_cypher/decomposer/cli_test_decomposer.py`

---

## Phases

### Phase 1: Understand Existing Code [complete]
- [x] Read `unit_test_pipeline.py` - existing decomposer test structure
- [x] Read `cli_test_step_a.py` - reference CLI pattern
- [x] Read `decomposer_module.py` - DSPy module with MLflow support
- [x] Read `lm.py` - available LLM configurations
- [x] Read `mlflow_setup.py` - MLflow setup utilities

### Phase 2: Design CLI Test [complete]
- [x] Define CLI arguments (--query, --lm, --json, --verbose, --async, --no-mlflow, etc.)
- [x] Support both sync and async modes
- [x] Integrate MLflow tracing
- [x] Support sample test cases

### Phase 3: Implementation [complete]
- [x] Create `cli_test_decomposer.py`
- [x] Implement argument parsing
- [x] Implement MLflow setup
- [x] Implement decomposer execution
- [x] Implement output formatting

### Phase 4: Verification [complete]
- [x] Syntax check passed
- [x] --help output verified
- [x] --list output verified

---

## Key Design Decisions

### 1. LM Options (from lm.py)
```python
LM_OPTIONS = [
    # OpenRouter models
    "gemini_flash",
    "gemini_2p5_pro",
    "gpt_4o_mini",
    "gpt_4o",
    "gpt_5",
    "gpt_oss_120b",
    "gpt_oss_20b",
    "anthropic_haiku",
    "deepseek_v3",
    "qwen3_235b",
    "kimi_k2",
    "minimax",
    "glm",
    # Tuzi models
    "tuzi_gemini_3_pro",
    "tuzi_gpt_5",
]
```

### 2. MLflow Setup (from cli_test_step_a.py pattern)
```python
from src.utils.mlflow_setup import setup_mlflow_tracking
if setup_mlflow_tracking(experiment_name="decomposer_cli_tests"):
    # Enable DSPy autolog or use manual tracing
    ...
```

### 3. Output Format
- Pretty format: Human-readable staged output
- JSON format: Machine-parseable full result

---

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | - | - |

---

## Files to Create/Modify
1. `tests/nl_to_cypher/decomposer/cli_test_decomposer.py` - Main CLI test file
