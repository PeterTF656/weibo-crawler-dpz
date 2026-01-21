# DSPy Tool Calling vs Structured Output Patterns - Best Practices

> Researched from DSPy Official Documentation

## Overview

This document addresses architectural decisions for DSPy implementations when choosing between:
1. **Native function calling** (LLM calls tools directly)
2. **Structured output pattern** (LLM generates structured data → post-processing → tool invocation)

Based on your context: experiencing "Tool choice is none, but model called a tool" errors with some models, and needing reliable time filter extraction from natural language.

---

## 1. DSPy Recommendations for Reliable Tool Calling

### Model Compatibility Issues

**Critical Finding**: Not all models support native function calling reliably.

From DSPy documentation:

> **DSPy automatically detects whether a language model supports native function calling** by using `litellm.supports_function_calling()`. This ensures compatibility and provides a fallback mechanism. **If a model does not inherently support native function calling, DSPy will gracefully revert to manual text-based parsing** for tool execution, even if the `use_native_function_calling=True` parameter has been explicitly set.

**Key Points**:
- ✅ Anthropic models (Claude) support native function calling properly
- ⚠️ Some models do NOT support native function calling
- ❌ Error "Tool choice is none, but model called a tool" indicates model incompatibility
- 🔄 DSPy attempts graceful fallback but may still fail

### Adapter Defaults and Configuration

Different adapters have different defaults:

```python
import dspy

# ChatAdapter: use_native_function_calling=False by default
chat_adapter = dspy.ChatAdapter(use_native_function_calling=False)

# JSONAdapter: use_native_function_calling=True by default
json_adapter = dspy.JSONAdapter(use_native_function_calling=True)

# Configure DSPy to use the adapter
dspy.configure(lm=dspy.LM(model="openai/gpt-4o"), adapter=chat_adapter)
```

**Documentation Quote**:
> The `ChatAdapter` uses `use_native_function_calling=False` by default, meaning it relies on text parsing for tool execution. In contrast, the `JSONAdapter` defaults to `use_native_function_calling=True`, utilizing native function calling. Users can override these default behaviors by explicitly setting the `use_native_function_calling` parameter.

### When to Use Tool Calling vs Manual Handling

From DSPy documentation:

**Use `dspy.ReAct` (automatic tool calling) when:**
- You want automatic reasoning and tool selection
- The task requires multiple tool calls
- You need built-in error recovery
- You want to focus on tool implementation rather than orchestration

**Use manual tool handling when:**
- ✅ You need precise control over tool execution
- ✅ You want custom error handling logic
- ✅ You want to minimize the latency
- ✅ Your tool returns nothing (void function)

**Recommendation for Your Use Case**: Given the model compatibility errors, manual handling is preferred.

---

## 2. LLM Output → Post-Processing → Tool Pattern

DSPy supports multiple patterns for separating LLM output generation from tool execution.

### Pattern 1: ChainOfThought with Manual Tool Execution

This is the **recommended production pattern** from existing documentation.

```python
import dspy

class TimeExtractionSignature(dspy.Signature):
    """Extract time filters from natural language query."""

    # Inputs
    query: str = dspy.InputField(desc="Natural language query")
    base_datetime: str = dspy.InputField(desc="Current datetime as ISO string")

    # Outputs (NO tool_calls field!)
    reasoning: str = dspy.OutputField(desc="Reasoning about time expressions")
    time_expression: str | None = dspy.OutputField(
        desc="Extracted time expression like 'last 30 minutes', '2 hours ago', or None"
    )
    offset_minutes: int | None = dspy.OutputField(
        desc="Calculated offset in minutes (negative for past), or None"
    )

class TimeFilterModule(dspy.Module):
    """Extracts time filters using LLM → post-processing pattern."""

    def __init__(self, lm: Any | None = None):
        super().__init__()
        self._lm_override = lm
        self.extractor = dspy.ChainOfThought(TimeExtractionSignature)

    def _convert_to_iso_datetime(
        self,
        base_datetime: datetime,
        offset_minutes: int
    ) -> str:
        """Deterministic tool invocation in Python."""
        target_time = base_datetime + timedelta(minutes=offset_minutes)
        return target_time.isoformat()

    def _parse_with_fallback(
        self,
        time_expression: str | None,
        query: str
    ) -> int | None:
        """Rule-based fallback parser."""
        if not time_expression:
            # Regex-based extraction
            patterns = [
                (r'last\s+(\d+)\s+minutes?', lambda m: -int(m.group(1))),
                (r'(\d+)\s+hours?\s+ago', lambda m: -int(m.group(1)) * 60),
                # ... more patterns
            ]
            for pattern, converter in patterns:
                match = re.search(pattern, query.lower())
                if match:
                    return converter(match)
        return None

    def forward(self, query: str, base_datetime: datetime) -> dspy.Prediction:
        """Extract time filter using separated pattern."""

        # Step 1: LLM generates structured output (no tool calling)
        lm = self._lm_override or dspy.settings.lm

        # Use ChatAdapter WITHOUT native function calling
        with dspy.context(
            lm=lm,
            adapter=dspy.ChatAdapter(use_native_function_calling=False),
        ):
            result = self.extractor(
                query=query,
                base_datetime=base_datetime.isoformat()
            )

        # Step 2: Post-processing and validation
        offset = result.offset_minutes
        time_expr = result.time_expression

        # Step 3: Fallback to rule-based parsing if LLM failed
        if offset is None:
            offset = self._parse_with_fallback(time_expr, query)

        # Step 4: Deterministic tool invocation
        if offset is not None:
            iso_datetime = self._convert_to_iso_datetime(
                base_datetime,
                offset
            )
            return dspy.Prediction(
                reasoning=result.reasoning,
                time_expression=time_expr,
                offset_minutes=offset,
                target_datetime=iso_datetime,
                success=True
            )
        else:
            return dspy.Prediction(
                reasoning=result.reasoning,
                time_expression=None,
                offset_minutes=None,
                target_datetime=None,
                success=False
            )
```

**Key Advantages**:
- ✅ No model compatibility issues (no native function calling)
- ✅ Deterministic tool execution
- ✅ Easy to debug and test
- ✅ Fallback mechanisms built-in
- ✅ Works with any model

### Pattern 2: TwoStepAdapter for Post-Processing

DSPy provides `TwoStepAdapter` for using a secondary model to extract structured data:

```python
from dspy.adapters import TwoStepAdapter
import dspy

# Use a smaller, faster model for extraction
extraction_lm = dspy.LM("openai/gpt-4o-mini")

# TwoStepAdapter: reasoning model → extraction model
adapter = TwoStepAdapter(extraction_model=extraction_lm)

class TimeExtraction(dspy.Signature):
    """Extract time information from text."""
    query: str = dspy.InputField()
    time_offset: int | None = dspy.OutputField()

module = dspy.Predict(TimeExtraction)

with dspy.context(adapter=adapter):
    result = module(query="events in the last 30 minutes")
```

**How TwoStepAdapter Works**:
1. Main LM generates raw text completion
2. Extraction LM parses text into structured output
3. Both models work together for reliability

From documentation:
> The `acall` method asynchronously invokes a language model, processes its output, and **uses a secondary model to extract structured data**. It handles raw text, log probabilities, and tool calls, returning a list of parsed output dictionaries.

---

## 3. Tool Calling Failure and Retry Handling

### Error Handling in ChatAdapter

DSPy adapters have built-in fallback mechanisms:

```python
def __call__(
    self,
    lm: LM,
    lm_kwargs: dict[str, Any],
    signature: type[Signature],
    demos: list[dict[str, Any]],
    inputs: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        return super().__call__(lm, lm_kwargs, signature, demos, inputs)
    except Exception as e:
        # fallback to JSONAdapter
        from dspy.adapters.json_adapter import JSONAdapter

        if (
            isinstance(e, ContextWindowExceededError)
            or isinstance(self, JSONAdapter)
            or not self.use_json_adapter_fallback
        ):
            raise e
        return JSONAdapter()(lm, lm_kwargs, signature, demos, inputs)
```

**Key Points**:
- ChatAdapter automatically falls back to JSONAdapter on errors
- Can disable fallback with `use_json_adapter_fallback=False`
- JSONAdapter uses structured output format with graceful degradation

### Retry Pattern: BestOfN

Use `dspy.BestOfN` to retry predictions with reward-based selection:

```python
import dspy

def validate_time_filter(args, pred: dspy.Prediction) -> float:
    """Reward function: returns 1.0 if time filter is valid, 0.0 otherwise."""
    if pred.offset_minutes is None:
        return 0.0

    # Validate offset is reasonable (e.g., within last 24 hours)
    if -1440 <= pred.offset_minutes <= 0:
        return 1.0
    return 0.0

time_extractor = dspy.ChainOfThought(TimeExtractionSignature)

# Wrap with BestOfN for automatic retry
best_of_3 = dspy.BestOfN(
    module=time_extractor,
    N=3,  # Try up to 3 times
    reward_fn=validate_time_filter,
    threshold=1.0,  # Stop when reward >= 1.0
    fail_count=1  # Raise error after 1 failure
)

result = best_of_3(
    query="events in the last 30 minutes",
    base_datetime=datetime.now().isoformat()
)
```

**How BestOfN Works**:
1. Runs module up to N times with different rollout IDs
2. Uses `temperature=1.0` for diversity
3. Returns first prediction that meets threshold
4. Otherwise returns best prediction by reward score

From documentation:
> Runs a module up to `N` times with different rollout IDs at `temperature=1.0` and returns the best prediction out of `N` attempts or the first prediction that passes the `threshold`.

### Iterative Refinement: Refine

Use `dspy.Refine` for automatic feedback loops:

```python
import dspy

refine_extractor = dspy.Refine(
    module=time_extractor,
    N=3,
    reward_fn=validate_time_filter,
    threshold=1.0,
    fail_count=1
)

result = refine_extractor(
    query="events in the last 30 minutes",
    base_datetime=datetime.now().isoformat()
)
```

**How Refine Differs from BestOfN**:
- Generates automatic feedback after each unsuccessful attempt
- Uses feedback as hints for subsequent runs
- More intelligent than simple retry
- Better for complex tasks requiring reasoning

From documentation:
> After each unsuccessful attempt, it generates detailed feedback about module performance and uses it as hints for subsequent runs, improving output quality iteratively.

---

## 4. Structured Output Without Tool Calling

### Using Pydantic Models with DSPy Signatures

DSPy natively supports Pydantic models for structured output:

```python
import dspy
from pydantic import BaseModel
from typing import List

class TimeFilter(BaseModel):
    """Structured time filter output."""
    expression: str | None
    offset_minutes: int | None
    confidence: float

class TimeExtractionSignature(dspy.Signature):
    """Extract time filters with Pydantic model output."""

    query: str = dspy.InputField()
    base_datetime: str = dspy.InputField()

    # Pydantic model as output type!
    time_filter: TimeFilter = dspy.OutputField(
        desc="Extracted time filter information"
    )

# Use with ChainOfThought
extractor = dspy.ChainOfThought(TimeExtractionSignature)

# Use ChatAdapter (no function calling needed for structured output)
with dspy.context(adapter=dspy.ChatAdapter()):
    result = extractor(
        query="show me posts from last hour",
        base_datetime=datetime.now().isoformat()
    )

# Access structured output
print(result.time_filter.offset_minutes)  # -60
print(result.time_filter.confidence)  # 0.95
```

**Key Benefits**:
- Type-safe output with Pydantic validation
- No tool calling required
- Works with all models
- ChatAdapter handles JSON schema automatically

### JSONAdapter for Guaranteed JSON Output

Use `JSONAdapter` when you need guaranteed JSON output:

```python
import dspy

# JSONAdapter ensures JSON output with fallback
json_adapter = dspy.JSONAdapter(use_native_function_calling=False)

class TimeData(BaseModel):
    offset_minutes: int | None
    expression: str | None

class TimeSignature(dspy.Signature):
    query: str = dspy.InputField()
    data: TimeData = dspy.OutputField()

extractor = dspy.Predict(TimeSignature)

with dspy.context(adapter=json_adapter):
    result = extractor(query="events from 2 hours ago")
```

**JSONAdapter Features**:
- Tries structured output format first
- Falls back to `{"type": "json_object"}` mode on error
- More robust than ChatAdapter for pure JSON needs

From documentation:
```python
try:
    structured_output_model = _get_structured_outputs_response_format(
        signature, self.use_native_function_calling
    )
    lm_kwargs["response_format"] = structured_output_model
    return super().__call__(lm, lm_kwargs, signature, demos, inputs)
except Exception:
    logger.warning("Failed to use structured output format, falling back to JSON mode.")
    lm_kwargs["response_format"] = {"type": "json_object"}
    return super().__call__(lm, lm_kwargs, signature, demos, inputs)
```

---

## 5. Validation and Fallback Patterns

### Pattern 1: Post-Processing with Validation

```python
import dspy
from pydantic import BaseModel, validator

class ValidatedTimeFilter(BaseModel):
    """Time filter with built-in validation."""

    offset_minutes: int | None
    expression: str | None

    @validator('offset_minutes')
    def validate_offset(cls, v):
        if v is not None:
            # Must be in the past and within reasonable range
            if not (-10080 <= v <= 0):  # Within last week
                raise ValueError(f"Offset {v} outside valid range")
        return v

class TimeExtractionSignature(dspy.Signature):
    query: str = dspy.InputField()
    base_datetime: str = dspy.InputField()
    time_data: ValidatedTimeFilter = dspy.OutputField()

extractor = dspy.ChainOfThought(TimeExtractionSignature)

def extract_with_validation(query: str, base_datetime: datetime):
    """Extract time filter with validation and fallback."""
    try:
        result = extractor(
            query=query,
            base_datetime=base_datetime.isoformat()
        )
        # Pydantic validation happens automatically
        return result.time_data
    except Exception as e:
        logger.warning(f"LLM extraction failed: {e}")
        # Fallback to rule-based extraction
        return rule_based_parser(query)
```

### Pattern 2: Reward Function for Validation

```python
def validation_reward(args, pred: dspy.Prediction) -> float:
    """Comprehensive validation reward function."""

    # Check 1: Required fields present
    if pred.time_filter is None:
        return 0.0

    # Check 2: Offset within valid range
    offset = pred.time_filter.offset_minutes
    if offset is None or not (-10080 <= offset <= 0):
        return 0.2  # Partial credit for trying

    # Check 3: Expression matches offset
    expr = pred.time_filter.expression
    if expr and "hour" in expr.lower():
        expected_range = (-120, -30)  # Hours typically mean 1-2 hours
        if expected_range[0] <= offset <= expected_range[1]:
            return 1.0

    return 0.5  # Partial credit

# Use with BestOfN or Refine
validated_extractor = dspy.BestOfN(
    module=extractor,
    N=3,
    reward_fn=validation_reward,
    threshold=1.0
)
```

### Pattern 3: Parsing with Error Recovery

```python
class TimeFilterModule(dspy.Module):
    """Time filter extraction with comprehensive error recovery."""

    def __init__(self):
        super().__init__()
        self.extractor = dspy.ChainOfThought(TimeExtractionSignature)

    def _parse_llm_output(self, result) -> dict:
        """Parse LLM output with error handling."""
        try:
            return {
                'offset': result.time_data.offset_minutes,
                'expression': result.time_data.expression,
                'source': 'llm'
            }
        except (AttributeError, ValidationError) as e:
            logger.warning(f"Failed to parse LLM output: {e}")
            return None

    def _rule_based_parse(self, query: str) -> dict:
        """Fallback to rule-based parsing."""
        patterns = {
            r'last\s+(\d+)\s+min': lambda m: -int(m.group(1)),
            r'(\d+)\s+hour.*ago': lambda m: -int(m.group(1)) * 60,
            r'past\s+(\d+)\s+hour': lambda m: -int(m.group(1)) * 60,
        }

        for pattern, converter in patterns.items():
            match = re.search(pattern, query.lower())
            if match:
                return {
                    'offset': converter(match),
                    'expression': match.group(0),
                    'source': 'regex'
                }
        return None

    def forward(self, query: str, base_datetime: datetime):
        """Extract with multi-level fallback."""

        # Level 1: Try LLM extraction
        try:
            result = self.extractor(
                query=query,
                base_datetime=base_datetime.isoformat()
            )
            parsed = self._parse_llm_output(result)
            if parsed:
                return dspy.Prediction(**parsed, success=True)
        except Exception as e:
            logger.warning(f"LLM extraction failed: {e}")

        # Level 2: Try rule-based parsing
        parsed = self._rule_based_parse(query)
        if parsed:
            return dspy.Prediction(**parsed, success=True)

        # Level 3: Return failure with reason
        return dspy.Prediction(
            offset=None,
            expression=None,
            source='none',
            success=False
        )
```

---

## Summary: Recommended Pattern for Your Use Case

Based on your requirements (reliable time filter extraction, model compatibility issues), here's the recommended approach:

### Architecture: Separated Pattern (LLM → Post-Processing → Tool)

```python
import dspy
from pydantic import BaseModel
from datetime import datetime, timedelta
import re

class TimeFilterData(BaseModel):
    """Validated time filter data."""
    time_expression: str | None
    offset_minutes: int | None

class TimeExtractionSignature(dspy.Signature):
    """Extract time information from query."""
    query: str = dspy.InputField()
    base_datetime: str = dspy.InputField()
    reasoning: str = dspy.OutputField()
    filter_data: TimeFilterData = dspy.OutputField()

class TimeFilterExtractor(dspy.Module):
    """Production-ready time filter extractor."""

    def __init__(self, lm=None):
        super().__init__()
        self._lm = lm

        # Use ChainOfThought with structured output (NO tool calling)
        base_extractor = dspy.ChainOfThought(TimeExtractionSignature)

        # Wrap with BestOfN for retry logic
        self.extractor = dspy.BestOfN(
            module=base_extractor,
            N=3,
            reward_fn=self._validation_reward,
            threshold=1.0,
            fail_count=1
        )

    def _validation_reward(self, args, pred) -> float:
        """Validate extracted time filter."""
        try:
            offset = pred.filter_data.offset_minutes
            if offset is None:
                return 0.0
            # Valid range: last 7 days
            if -10080 <= offset <= 0:
                return 1.0
            return 0.0
        except:
            return 0.0

    def _rule_based_fallback(self, query: str) -> int | None:
        """Regex-based fallback parser."""
        patterns = [
            (r'last\s+(\d+)\s+min', lambda m: -int(m.group(1))),
            (r'(\d+)\s+hour.*ago', lambda m: -int(m.group(1)) * 60),
        ]
        for pattern, converter in patterns:
            match = re.search(pattern, query.lower())
            if match:
                return converter(match)
        return None

    def _convert_to_iso(self, base: datetime, offset: int) -> str:
        """Deterministic tool invocation."""
        return (base + timedelta(minutes=offset)).isoformat()

    def forward(self, query: str, base_datetime: datetime):
        """Extract time filter with multi-level fallback."""

        lm = self._lm or dspy.settings.lm

        # Use ChatAdapter WITHOUT native function calling
        with dspy.context(
            lm=lm,
            adapter=dspy.ChatAdapter(use_native_function_calling=False)
        ):
            try:
                result = self.extractor(
                    query=query,
                    base_datetime=base_datetime.isoformat()
                )
                offset = result.filter_data.offset_minutes
            except Exception as e:
                logger.warning(f"LLM extraction failed: {e}")
                offset = None

        # Fallback to rule-based
        if offset is None:
            offset = self._rule_based_fallback(query)

        # Execute tool deterministically
        if offset is not None:
            target_iso = self._convert_to_iso(base_datetime, offset)
            return dspy.Prediction(
                offset_minutes=offset,
                target_datetime=target_iso,
                success=True
            )

        return dspy.Prediction(
            offset_minutes=None,
            target_datetime=None,
            success=False
        )
```

### Why This Pattern is Recommended

1. **No Model Compatibility Issues**: Uses structured output, not native function calling
2. **Reliable**: Multi-level fallback (LLM → BestOfN retry → regex)
3. **Deterministic Tools**: Tool invocation happens in Python, not LLM
4. **Easy to Debug**: Clear separation of concerns
5. **Works with Any Model**: No dependency on function calling support
6. **Production-Ready**: Error handling, validation, and logging built-in

### Key Differences from Tool Calling Pattern

| Aspect | Tool Calling Pattern | Recommended Pattern |
|--------|---------------------|-------------------|
| **LLM Role** | Calls tools directly | Generates structured data |
| **Tool Execution** | During LLM inference | Post-processing in Python |
| **Model Requirements** | Must support function calling | Any model with JSON support |
| **Error Handling** | LLM-dependent | Explicit Python code |
| **Fallback** | Limited | Multi-level (retry + regex) |
| **Debugging** | Difficult (LLM black box) | Easy (step through code) |
| **Reliability** | Model-dependent | Deterministic |

---

## Additional Resources

- **Existing Documentation**: `.claude/docs/dspy-tools-complete-reference.md` covers production tool patterns
- **DSPy Tools**: https://dspy.ai/learn/programming/tools
- **DSPy Adapters**: https://dspy.ai/learn/programming/adapters
- **BestOfN/Refine**: https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine
