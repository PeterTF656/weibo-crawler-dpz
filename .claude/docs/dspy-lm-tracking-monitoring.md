# DSPy Language Model Tracking and Monitoring - Official Documentation

> Researched from DSPy Official Documentation (dspy.ai)

## Overview

DSPy provides comprehensive built-in capabilities for tracking and monitoring language model usage, including token counts, costs, call history, and integration with external observability platforms like MLflow. This document covers all official methods for monitoring LM usage in DSPy applications.

---

## Table of Contents

1. [Enabling Usage Tracking](#enabling-usage-tracking)
2. [Accessing Usage Statistics](#accessing-usage-statistics)
3. [LM History and Metadata](#lm-history-and-metadata)
4. [Cost Tracking](#cost-tracking)
5. [MLflow Integration](#mlflow-integration)
6. [Custom Callbacks for Observability](#custom-callbacks-for-observability)
7. [Configuration Options](#configuration-options)
8. [Best Practices](#best-practices)
9. [Important Notes](#important-notes)

---

## Enabling Usage Tracking

### Basic Usage Tracking Configuration

DSPy provides a built-in `track_usage` parameter in `dspy.configure()` that enables automatic token usage tracking for all LM calls.

```python
import dspy

# Enable usage tracking globally
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=False),
    track_usage=True
)

result = dspy.ChainOfThought(BasicQA)(question="What is 2+2?")
print(f"Token usage: {result.get_lm_usage()}")
```

**Source:** https://dspy.ai/cheatsheet_h=basicqa

### Usage Tracking with Multiple Modules

When using programs with multiple prediction modules, usage tracking aggregates token counts across all LM calls within that execution.

```python
import dspy

dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=False),
    track_usage=True
)

class MyProgram(dspy.Module):
    def __init__(self):
        self.predict1 = dspy.ChainOfThought("question -> answer")
        self.predict2 = dspy.ChainOfThought("question, answer -> score")

    def __call__(self, question: str) -> str:
        answer = self.predict1(question=question)
        score = self.predict2(question=question, answer=answer)
        return score

program = MyProgram()
output = program(question="What is the capital of France?")
print(output.get_lm_usage())  # Shows aggregated token usage for both calls
```

**Source:** https://dspy.ai/learn/programming/modules

### Usage Tracking with Caching

**Important:** When caching is enabled, cached responses will show empty usage statistics since no actual LM call is made.

```python
# Enable caching
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=True),
    track_usage=True
)

program = MyProgram()

# First call - will show usage statistics
output = program(question="What is the capital of Zambia?")
print(output.get_lm_usage())  # Shows token usage: {'model': {'prompt_tokens': X, 'completion_tokens': Y}}

# Second call - same question, will use cache
output = program(question="What is the capital of Zambia?")
print(output.get_lm_usage())  # Shows empty dict: {}
```

**Source:** https://dspy.ai/learn/programming/modules

---

## Accessing Usage Statistics

### Retrieving Usage from Prediction Objects

Every DSPy `Prediction` object has built-in methods for accessing LM usage information.

```python
# Get LM usage from prediction
def get_lm_usage(self):
    return self._lm_usage

# Set LM usage manually (for custom tracking)
def set_lm_usage(self, value):
    self._lm_usage = value
```

**Source:** https://dspy.ai/api/evaluation/EvaluationResult

### Internal Usage Tracking Implementation

DSPy modules automatically track usage when `settings.track_usage` is enabled:

```python
@with_callbacks
def __call__(self, *args, **kwargs) -> Prediction:
    from dspy.dsp.utils.settings import thread_local_overrides

    caller_modules = settings.caller_modules or []
    caller_modules = list(caller_modules)
    caller_modules.append(self)

    with settings.context(caller_modules=caller_modules):
        if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
            with track_usage() as usage_tracker:
                output = self.forward(*args, **kwargs)
            tokens = usage_tracker.get_total_tokens()
            self._set_lm_usage(tokens, output)

            return output

        return self.forward(*args, **kwargs)
```

**Source:** https://dspy.ai/api/evaluation/SemanticF1

### Async Usage Tracking

Usage tracking also works with async DSPy modules:

```python
@with_callbacks
async def acall(self, *args, **kwargs) -> Prediction:
    from dspy.dsp.utils.settings import thread_local_overrides

    caller_modules = settings.caller_modules or []
    caller_modules = list(caller_modules)
    caller_modules.append(self)

    with settings.context(caller_modules=caller_modules):
        if settings.track_usage and thread_local_overrides.get().get("usage_tracker") is None:
            with track_usage() as usage_tracker:
                output = await self.aforward(*args, **kwargs)
                tokens = usage_tracker.get_total_tokens()
                self._set_lm_usage(tokens, output)

                return output

        return await self.aforward(*args, **kwargs)
```

**Source:** https://dspy.ai/api/evaluation/SemanticF1

---

## LM History and Metadata

### Accessing LM History

Every `dspy.LM` object maintains a history of all interactions, which can be accessed to retrieve detailed metadata about each call.

```python
import dspy

# Assuming 'lm' is a configured dspy.LM object
# Make some calls to populate the history...

# Get the number of calls made to the LM
num_calls = len(lm.history)
print(f"Number of LM calls: {num_calls}")

# Access metadata of the last call
if lm.history:
    last_call_metadata = lm.history[-1]
    print("Keys in last call metadata:", last_call_metadata.keys())
    # Example: print prompt and response
    # print("Prompt:", last_call_metadata['prompt'])
    # print("Response:", last_call_metadata['response'])
```

**Source:** https://dspy.ai/learn/programming/language_models

### Inspecting Recent History

DSPy provides a built-in `inspect_history()` method for displaying formatted LM history:

```python
# Inspect the last call
dspy.inspect_history(n=1)

# Inspect the last 5 calls
dspy.inspect_history(n=5)
```

**Source:** https://dspy.ai/tutorials/observability

### History Management

DSPy manages history size automatically with configurable limits:

```python
def update_history(self, entry):
    if settings.disable_history:
        return

    # Global LM history
    if len(GLOBAL_HISTORY) >= MAX_HISTORY_SIZE:
        GLOBAL_HISTORY.pop(0)

    GLOBAL_HISTORY.append(entry)

    if settings.max_history_size == 0:
        return

    # dspy.LM.history
    if len(self.history) >= settings.max_history_size:
        self.history.pop(0)

    self.history.append(entry)

    # Per-module history
    caller_modules = settings.caller_modules or []
    for module in caller_modules:
        if len(module.history) >= settings.max_history_size:
            module.history.pop(0)
        module.history.append(entry)
```

**Source:** https://dspy.ai/api/models/LM

---

## Cost Tracking

### Calculating Total LM Costs

DSPy integrates with LiteLLM to automatically calculate costs for supported providers. Costs are stored in the LM history and can be summed:

```python
# Calculate total cost in USD for all LM calls
cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])
print(f"Total cost: ${cost:.4f} USD")
```

**Source:** https://dspy.ai/tutorials/entity_extraction, https://dspy.ai/tutorials/rag

### Cost Tracking During Optimization

When running DSPy optimizers, costs can be significant. Here are typical costs:

- **Simple optimization run**: ~$2 USD, ~20 minutes
- **BootstrapFewShotWithRandomSearch** (7 candidates, 10 threads):
  - Time: ~6 minutes
  - API calls: ~3200
  - Input tokens: ~2.7M
  - Output tokens: ~156K
  - Cost: ~$3 USD (at GPT-3.5-turbo-1106 pricing)

**Source:** https://dspy.ai/faqs, https://dspy.ai/index

### Cost Optimization Strategies

From the official documentation:

> "To improve the quality of your DSPy programs, you can explore different system architectures, such as asking the LM to generate search queries for a retriever. You can also experiment with different prompt optimizers or weight optimizers, scale inference time compute using DSPy Optimizers (e.g., via ensembling), and reduce costs by distilling to a smaller LM."

**Source:** https://dspy.ai/tutorials/rag

---

## MLflow Integration

DSPy provides native integration with MLflow for comprehensive observability and experiment tracking.

### Installing MLflow

```bash
# For basic tracing (minimum version 2.18.0)
pip install -U mlflow>=2.18.0

# For optimizer tracking (minimum version 2.21.1)
pip install mlflow>=2.21.1

# For full LLMOps integration (version 3.0.0+)
pip install mlflow>=3.0.0
```

**Source:** https://dspy.ai/tutorials/observability, https://dspy.ai/tutorials/optimizer_tracking, https://dspy.ai/tutorials/custom_module

### Starting MLflow Server

```bash
# Start MLflow server with SQLite backend (recommended for production)
mlflow server --backend-store-uri sqlite:///mydb.sqlite

# Server runs on http://127.0.0.1:5000 by default
```

**Source:** https://dspy.ai/tutorials/observability

### Basic MLflow Configuration

```python
import dspy
import mlflow
import os

os.environ["OPENAI_API_KEY"] = "{your_openai_api_key}"

# Configure MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Set experiment name
mlflow.set_experiment("DSPy")

# Configure DSPy
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)
```

**Source:** https://dspy.ai/tutorials/observability

### MLflow Autologging for Optimization

MLflow provides comprehensive autologging for DSPy optimization processes:

```python
import mlflow
import dspy

# Enable autologging with all features
mlflow.dspy.autolog(
    log_compiles=True,              # Track optimization process
    log_evals=True,                 # Track evaluation results
    log_traces_from_compile=True    # Track program traces during optimization
)

# Configure MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DSPy-Optimization")
```

**Important Note:** For large datasets, disable trace logging to prevent memory issues:

```python
log_traces_from_compile=False  # Optimize memory usage for large datasets
```

**Source:** https://dspy.ai/tutorials/optimizer_tracking

### Logging Evaluation Results to MLflow

```python
import mlflow
import dspy

with mlflow.start_run(run_name="agent_evaluation"):
    evaluate = dspy.Evaluate(
        devset=devset,
        metric=top5_recall,
        num_threads=16,
        display_progress=True,
    )

    # Evaluate the program
    result = evaluate(cot)

    # Log the aggregated score
    mlflow.log_metric("top5_recall", result.score)

    # Log detailed evaluation results as a table
    mlflow.log_table(
        {
            "Claim": [example.claim for example in eval_set],
            "Expected Titles": [example.titles for example in eval_set],
            "Predicted Titles": [output[1] for output in result.results],
            "Top 5 Recall": [output[2] for output in result.results],
        },
        artifact_file="eval_results.json",
    )
```

**Source:** https://dspy.ai/tutorials/agents, https://dspy.ai/tutorials/multihop_search

### MLflow Tracing Capabilities

When MLflow tracing is enabled, it automatically captures:

- All LLM calls with prompts and responses
- Retriever invocations
- Tool executions
- Component relationships
- Parameters and configurations
- Latency metrics
- Token usage and costs

**Source:** https://dspy.ai/tutorials/observability

---

## Custom Callbacks for Observability

DSPy supports custom callbacks for implementing custom logging and monitoring logic.

### Creating a Custom Callback

```python
import dspy
from dspy.utils.callback import BaseCallback

# 1. Define a custom callback class that extends BaseCallback
class AgentLoggingCallback(BaseCallback):

    # 2. Implement on_module_end handler to run custom logging code
    def on_module_end(self, call_id, outputs, exception):
        step = "Reasoning" if self._is_reasoning_output(outputs) else "Acting"
        print(f"== {step} Step ==")
        for k, v in outputs.items():
            print(f"  {k}: {v}")
        print("\n")

    def _is_reasoning_output(self, outputs):
        return any(k.startswith("Thought") for k in outputs.keys())

# 3. Set the callback in DSPy settings
dspy.configure(callbacks=[AgentLoggingCallback()])
```

**Source:** https://dspy.ai/tutorials/observability

### Callback Integration Points

Callbacks are automatically invoked during module execution through the `@with_callbacks` decorator that wraps the `__call__` and `acall` methods.

**Source:** https://dspy.ai/api/modules/CodeAct

---

## Configuration Options

### DSPy Configure Parameters

```python
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=False),
    track_usage=True,           # Enable usage tracking
    callbacks=[...],            # Add custom callbacks
)
```

### Cache Configuration

```python
# Disable all caching
dspy.configure_cache(
    enable_disk_cache=False,
    enable_memory_cache=False,
)

# Configure cache limits
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_size_limit_bytes=YOUR_DESIRED_VALUE,
    memory_max_entries=YOUR_DESIRED_VALUE,
)
```

**Source:** https://dspy.ai/tutorials/cache

### LM-Specific Configuration

```python
# Disable caching for specific LM instance
lm = dspy.LM('openai/gpt-4o-mini', cache=False)

# Configure generation parameters
gpt_4o_mini = dspy.LM(
    'openai/gpt-4o-mini',
    temperature=0.9,
    max_tokens=3000,
    cache=False
)
```

**Source:** https://dspy.ai/faqs, https://dspy.ai/learn/programming/language_models

### Environment Variables

```python
import os

# Set cache directory for notebooks
os.environ["DSP_NOTEBOOK_CACHEDIR"] = os.path.join(os.getcwd(), 'cache')

# Set fine-tuning directory
os.environ["DSPY_FINETUNEDIR"] = "/path/to/dir"

# Control GPU usage
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

**Source:** https://dspy.ai/faqs, https://dspy.ai/tutorials/classification_finetuning

### History Configuration

DSPy provides settings to control history tracking:

- `settings.disable_history`: Disable all history tracking
- `settings.max_history_size`: Maximum number of entries to keep in history (per LM, per module)

**Source:** https://dspy.ai/api/models/LM

### Logging Configuration

```python
import logging

# Set DSPy logging level to WARNING (reduce verbosity)
logging.getLogger("dspy").setLevel(logging.WARNING)

# Disable DSPy logging completely
from dspy.utils import disable_logging
disable_logging()

# Re-enable DSPy logging
from dspy.utils import enable_logging
enable_logging()
```

**Source:** https://dspy.ai/faqs, https://dspy.ai/api/utils/disable_logging

---

## Best Practices

### 1. Always Enable Usage Tracking for Cost Monitoring

```python
# RECOMMENDED: Enable usage tracking by default
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=False),
    track_usage=True
)
```

### 2. Use MLflow for Production Monitoring

For production systems, MLflow provides the most comprehensive observability:

```python
import mlflow
import dspy

# Configure MLflow
mlflow.set_tracking_uri("http://your-mlflow-server:5000")
mlflow.set_experiment("Production-DSPy-App")

# Enable autologging
mlflow.dspy.autolog(
    log_compiles=True,
    log_evals=True,
    log_traces_from_compile=False  # Disable for large-scale production
)
```

### 3. Monitor Costs Regularly

```python
# Check costs after execution
total_cost = sum([x['cost'] for x in lm.history if x['cost'] is not None])
if total_cost > BUDGET_THRESHOLD:
    logger.warning(f"Cost exceeded threshold: ${total_cost:.4f}")
```

### 4. Disable Caching for Accurate Usage Tracking

When you need precise usage statistics, disable caching:

```python
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=False),  # No cache for accurate tracking
    track_usage=True
)
```

### 5. Use Custom Callbacks for Integration with Existing Systems

Implement custom callbacks to integrate DSPy monitoring with your existing observability stack:

```python
class CustomMonitoringCallback(BaseCallback):
    def on_module_end(self, call_id, outputs, exception):
        # Send metrics to your monitoring system
        send_to_datadog(outputs)
        send_to_prometheus(outputs)
```

### 6. Optimize for Large Datasets

When working with large datasets during optimization:

```python
# Disable trace logging to prevent memory issues
mlflow.dspy.autolog(
    log_compiles=True,
    log_evals=True,
    log_traces_from_compile=False  # IMPORTANT for large datasets
)
```

**Source:** https://dspy.ai/tutorials/optimizer_tracking

### 7. Choose the Right Optimizer

From the official documentation:

- **10 examples**: Use `BootstrapFewShot`
- **50+ examples**: Try `BootstrapFewShotWithRandomSearch`
- **200+ examples, 40+ trials**: Use `MIPROv2`
- **Large LM (7B+ parameters)**: Consider `BootstrapFinetune` for efficiency

**Source:** https://dspy.ai/learn/optimization/optimizers

---

## Important Notes

### Cache Impact on Usage Tracking

**Critical:** When caching is enabled, repeated calls with identical inputs will return cached results with empty usage statistics:

```python
# Cached call returns: {}
# Non-cached call returns: {'model': {'prompt_tokens': X, 'completion_tokens': Y}}
```

### LiteLLM Integration

DSPy uses LiteLLM under the hood for cost calculation. Costs are automatically calculated for supported providers (OpenAI, Anthropic, etc.). The `cost` field in history entries may be `None` for providers not supported by LiteLLM.

### History Metadata Structure

The LM history contains rich metadata for each call. Common keys include:

- `prompt`: The input prompt
- `response`: The model response
- `cost`: Cost in USD (if available)
- `usage`: Token usage information
- `model`: Model identifier
- `cache_hit`: Whether the response came from cache

### Global vs. Local History

DSPy maintains three levels of history:

1. **Global History** (`GLOBAL_HISTORY`): All LM calls across all instances
2. **LM Instance History** (`lm.history`): Calls for a specific LM instance
3. **Module History** (`module.history`): Calls for a specific module

All three are managed with size limits to prevent memory issues.

**Source:** https://dspy.ai/api/models/LM

### Forward Method Integration

Usage tracking is integrated at the forward method level:

```python
async def aforward(self, prompt: str | None = None, messages: list[dict[str, Any]] | None = None, **kwargs):
    # ... request handling ...

    results = await completion(...)

    # Automatic usage tracking
    if not getattr(results, "cache_hit", False) and dspy.settings.usage_tracker and hasattr(results, "usage"):
        settings.usage_tracker.add_usage(self.model, dict(results.usage))

    return results
```

**Source:** https://dspy.ai/api/models/LM

---

## Summary

DSPy provides comprehensive LM tracking and monitoring through:

1. **Built-in Usage Tracking**: Enable with `track_usage=True` in `dspy.configure()`
2. **Prediction-Level Metrics**: Access via `prediction.get_lm_usage()`
3. **LM History**: Full metadata for all calls via `lm.history`
4. **Cost Tracking**: Automatic cost calculation via LiteLLM integration
5. **MLflow Integration**: Production-grade observability with experiment tracking
6. **Custom Callbacks**: Flexible integration with existing monitoring systems
7. **Configuration Options**: Fine-grained control over caching, history, and logging

For production DSPy applications with MLflow integration, the recommended approach is:

```python
import dspy
import mlflow

# Configure MLflow
mlflow.set_tracking_uri("http://mlflow-server:5000")
mlflow.set_experiment("Your-Experiment")
mlflow.dspy.autolog(log_compiles=True, log_evals=True)

# Configure DSPy with usage tracking
dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=True),
    track_usage=True
)

# Your DSPy program execution
# All metrics, traces, and costs are automatically logged to MLflow
```

This setup provides complete observability across LM calls, token usage, costs, and execution traces.
