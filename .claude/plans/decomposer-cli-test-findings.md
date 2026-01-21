# Findings: Decomposer CLI Test

## Codebase Discoveries

### 1. Decomposer Module Structure
**Location:** `src/tools/graphdb/search/nl_to_cypher/sequential/agent/decomposer/agent/`

Key files:
- `model.py` - DecomposedQuery model with anchor_label + branch intents
- `dspy/decomposer_module.py` - QueryDecomposerModule (sync/async with MLflow)
- `unit_test_pipeline.py` - Existing test pipeline (lacks MLflow integration)

### 2. Output Model (DecomposedQuery)
```python
class DecomposedQuery(BaseModel):
    anchor_label: str  # Composition | Matching | Creator
    intersection_branches: List[str]  # AND semantics
    union_branches: List[str]  # OR semantics
```

### 3. QueryDecomposerModule API
```python
# Sync usage
module = QueryDecomposerModule(lm=lm)
result = module.forward(user_query="...")
decomposed = result.decomposed_query

# Async usage with MLflow span
result = await module.aforward(user_query="...", mlflow_span=span)
```

### 4. Available LLMs (from lm.py)
| Model | Provider | Function |
|-------|----------|----------|
| gemini_flash | OpenRouter | get_gemini_flash() |
| gpt_4o_mini | OpenRouter | get_gpt_4o_mini() |
| kimi_k2 | OpenRouter | get_kimi_k2() |
| deepseek_v3 | OpenRouter | get_deepseek_v3() |
| anthropic_haiku | OpenRouter | get_anthropic_haiku() |
| tuzi_gemini_3_pro | Tuzi | get_tuzi_gemini_3_pro() |

### 5. MLflow Integration Pattern
From `cli_test_step_a.py`:
```python
import mlflow
from src.utils.mlflow_setup import setup_mlflow_tracking

if setup_mlflow_tracking(experiment_name="decomposer_cli_tests"):
    print("[MLflow] Tracing enabled")
```

For DSPy, use manual tracing:
```python
@mlflow.trace(name="decomposer", span_type="LLM")
async def run_decomposer(...):
    result = await module.aforward(user_query=query)
    return result
```

### 6. Existing unit_test_pipeline.py Patterns
- Uses argparse for CLI
- Supports --query, --mock, --all, --model, --json, --quiet, --list, --test-case
- Has SAMPLE_TEST_CASES for batch testing
- Pretty and JSON output formatters
- ValidationResult for test assertions

**Missing from unit_test_pipeline.py:**
- MLflow tracing integration
- .env file loading
- Async support for MLflow spans

### 7. CLI Test Pattern (from cli_test_step_a.py)
Key elements:
1. Load .env file BEFORE imports
2. Add project root to sys.path
3. Parse args with argparse
4. Setup MLflow tracking
5. Initialize LLM
6. Run workflow (sync or async)
7. Print formatted results

---

## Implementation Notes

### Required Imports
```python
import argparse
import asyncio
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
import mlflow
```

### Env Loading Pattern
```python
def load_env_file(env_file: str | None = None) -> str:
    if env_file:
        env_path = Path(env_file)
    else:
        env_path = Path(project_root) / ".env"
    load_dotenv(env_path, override=True)
    return str(env_path)
```
