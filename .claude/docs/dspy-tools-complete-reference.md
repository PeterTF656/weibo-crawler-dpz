# DSPy Tools - Complete Reference

> **Updated with production patterns from SnipReel codebase**

## Overview

DSPy tools enable Language Models to call external functions and integrate with various APIs. This guide covers both **production patterns** (ChainOfThought with manual execution) and **prototyping patterns** (ReAct with automatic execution).

**Key Insight:** There are two fundamentally different approaches to tool calling in DSPy:
1. **Manual Execution (Production)**: ChainOfThought plans tool calls, you execute them
2. **Automatic Execution (Prototyping)**: ReAct executes tools automatically in a loop

## Quick Comparison: ReAct vs ChainOfThought

| Feature | ChainOfThought (Production) | ReAct (Prototyping) |
|---------|----------------------------|---------------------|
| **Tool execution** | Manual (you implement) | Automatic (internal loop) |
| **Iterations** | Single pass | Multiple (max_iters) |
| **Tools passed** | Input during forward() | Constructor parameter |
| **Control** | High (explicit) | Low (black box) |
| **ChatAdapter required** | ✅ Yes (for native function calling) | ❌ No |
| **Error handling** | You implement | Internal |
| **Use case** | Production systems | Quick prototypes |
| **Debugging** | Easy (step through) | Hard (black box) |

## Core Concepts

### What are DSPy Tools?

DSPy tools are functions that LLMs can call to:
- Retrieve external information (APIs, databases, search)
- Perform computations (math, data processing)
- Execute actions (booking, cancellation, file operations)
- Access internal systems (memory, preferences, user data)

### Tool vs Module vs Signature

- **Tool**: A callable function wrapped in `dspy.Tool` (e.g., `dspy.Tool(func=get_weather, name="get_weather", desc="...")`)
- **Module**: A DSPy component that orchestrates LM calls (e.g., `dspy.ChainOfThought`, `dspy.ReAct`)
- **Signature**: Defines the input/output schema for a DSPy module with fields for tools and tool calls

---

## Part 1: Production Pattern (Recommended)

### Pattern Overview: ChainOfThought + Manual Execution

**This is the recommended pattern for production systems.** It gives you full control over tool execution, error handling, and result processing.

**How it works:**
1. LLM receives tools as **input** and plans which tools to call
2. LLM outputs `tool_calls` (structured plan) or `direct_result` (direct answer)
3. **You manually execute** the planned tool calls
4. You process and use the results

### Step 1: Define and Wrap Tools

**Always wrap tools explicitly with `dspy.Tool`:**

```python
import dspy

# Define the function
def get_property_schema_for_labels(labels: list[str]) -> str:
    """Get valid property keys for Neo4j node labels.

    Args:
        labels: List of label names (e.g., ['Composition', 'Status'])

    Returns:
        Formatted string with properties for each label
    """
    result = []
    for label in labels:
        props = schema_registry.get(label, [])
        result.append(f"Label: {label}\nProperties: {', '.join(props)}")
    return "\n\n".join(result)

# Wrap in dspy.Tool with explicit metadata
property_schema_tool = dspy.Tool(
    func=get_property_schema_for_labels,
    name="get_property_schema_for_labels",
    desc=(
        "Get valid property keys for Neo4j node labels. "
        "Use this to discover which properties can be used for each label. "
        "Pass full label names like 'Composition', 'Status', 'Emotion', etc."
    ),
)
```

**Why explicit wrapping is required:**
- ✅ Clear, explicit tool names and descriptions
- ✅ Control over how tools appear to the LLM
- ✅ Type-safe and debuggable
- ✅ Works reliably across all scopes

### Step 2: Define Complete Signature

**The signature must have specific input and output fields:**

```python
class ToolExecutorSignature(dspy.Signature):
    """Execute a task using tools or direct reasoning.

    The LLM can choose to either:
    1. Call tools (populate tool_calls, leave direct_result empty)
    2. Answer directly (populate direct_result, leave tool_calls empty)
    """

    # === INPUTS ===
    task: str = dspy.InputField(
        desc="The task to execute"
    )
    context: str = dspy.InputField(
        desc="Available context and background information"
    )
    tools: list[dspy.Tool] = dspy.InputField(
        desc=(
            "Available tools. Use get_property_schema_for_labels(labels) "
            "to get valid properties for node labels."
        )
    )

    # === OUTPUTS (ALL THREE REQUIRED) ===
    reasoning: str = dspy.OutputField(
        desc=(
            "Your thought process: Do you need to call tools, or can you "
            "answer directly with available information?"
        )
    )
    tool_calls: dspy.ToolCalls = dspy.OutputField(
        desc=(
            "Tool calls to make. Leave empty if answering directly. "
            "Call tools to get information you don't have."
        )
    )
    direct_result: str | None = dspy.OutputField(
        desc=(
            "Direct answer if no tool needed. Set to None if making tool_calls. "
            "Use this when you already have enough information."
        )
    )
```

**Critical requirements:**
- ✅ `tools` as **InputField** (not passed to constructor)
- ✅ `reasoning` output - LLM's thought process
- ✅ `tool_calls` output - Planned tool calls (can be empty)
- ✅ `direct_result` output - Direct answer option (can be None)

**This dual-path pattern** allows the LLM to choose between calling tools or answering directly.

### Step 3: Implement Module with Manual Execution

**Complete production implementation:**

```python
import dspy
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

class ToolExecutorModule(dspy.Module):
    """Production module with manual tool execution.

    Pattern: ChainOfThought + ChatAdapter + Manual Execution
    """

    def __init__(self, lm: Any | None = None):
        """Initialize the module.

        Args:
            lm: Optional language model override
        """
        super().__init__()
        self._lm_override = lm

        # Define tools - must be dspy.Tool wrapped objects
        self._tools: list[dspy.Tool] = [
            property_schema_tool,
            available_labels_tool,
        ]

        # Use ChainOfThought (NOT ReAct!)
        self.executor = dspy.ChainOfThought(ToolExecutorSignature)

    def _build_tool_functions_map(self) -> dict[str, Callable[..., Any]]:
        """Build explicit mapping of tool names to callable functions.

        This is REQUIRED for manual tool execution. Do not rely on
        automatic discovery in production code.
        """
        functions_map: dict[str, Callable[..., Any]] = {}
        for tool in self._tools:
            if tool.name:
                functions_map[tool.name] = tool.func
        return functions_map

    def _execute_tool_calls(self, tool_calls: dspy.ToolCalls) -> list[str]:
        """Execute tool calls synchronously with proper error handling.

        Args:
            tool_calls: The tool calls planned by the LLM

        Returns:
            List of tool result strings
        """
        results: list[str] = []
        functions_map = self._build_tool_functions_map()

        for call in tool_calls.tool_calls:
            tool_name = call.name
            tool_args = call.args

            # Validate tool exists
            if tool_name not in functions_map:
                error_msg = (
                    f"Tool '{tool_name}' not found in available tools: "
                    f"{list(functions_map.keys())}"
                )
                logger.warning(error_msg)
                results.append(f"Error: {error_msg}")
                continue

            # Execute with error handling
            try:
                func = functions_map[tool_name]
                logger.info(f"Executing tool '{tool_name}' with args: {tool_args}")

                result = func(**tool_args)
                result_str = str(result)
                results.append(result_str)

                # Log preview
                log_preview = (
                    result_str[:200] + "..."
                    if len(result_str) > 200
                    else result_str
                )
                logger.info(f"Tool '{tool_name}' returned: {log_preview}")

            except Exception as e:
                error_msg = f"Error executing tool '{tool_name}': {str(e)}"
                logger.exception(error_msg)
                results.append(f"Error: {error_msg}")

        return results

    def forward(self, task: str, context: str) -> dspy.Prediction:
        """Execute task using tools or direct reasoning.

        Args:
            task: The task to execute
            context: Available context

        Returns:
            Prediction with:
            - reasoning: LLM's thought process
            - tool_calls: Tool calls made (if any)
            - tool_results: Results from tool execution (if tools called)
            - result: Final result (direct_result or combined tool results)
        """
        lm = self._lm_override or dspy.settings.lm

        # CRITICAL: Use ChatAdapter for native function calling
        with dspy.context(
            lm=lm,
            adapter=dspy.ChatAdapter(use_native_function_calling=True),
            tool_choice="auto",  # "auto", "required", or "none"
            track_usage=True,
        ):
            res = self.executor(
                task=task,
                context=context,
                tools=self._tools,  # Tools as INPUT, not constructor param!
            )

        # Extract usage tracking
        original_usage = None
        if hasattr(res, "get_lm_usage"):
            try:
                original_usage = res.get_lm_usage()
            except Exception as e:
                logger.warning(f"Failed to get_lm_usage(): {e}")

        # Check if tool calls were made
        tool_calls = getattr(res, "tool_calls", None)
        direct_result = getattr(res, "direct_result", None)

        # Path 1: Tool calls made - execute them manually
        if tool_calls and hasattr(tool_calls, "tool_calls") and tool_calls.tool_calls:
            logger.info(f"Executing {len(tool_calls.tool_calls)} tool calls...")
            tool_results = self._execute_tool_calls(tool_calls)

            # Combine results
            final_result = "\n---\n".join(tool_results)

            # Create new prediction with results
            new_prediction = dspy.Prediction(
                reasoning=res.reasoning,
                tool_calls=tool_calls,
                tool_results=tool_results,
                result=final_result,
            )

            # Preserve usage tracking
            if original_usage and hasattr(new_prediction, "set_lm_usage"):
                try:
                    new_prediction.set_lm_usage(original_usage)
                except Exception as e:
                    logger.warning(f"Failed to set_lm_usage(): {e}")

            return new_prediction

        # Path 2: Direct result - no tool calls needed
        else:
            logger.info("No tool calls made, using direct result")
            new_prediction = dspy.Prediction(
                reasoning=res.reasoning,
                tool_calls=None,
                tool_results=[],
                result=direct_result or "No result generated.",
            )

            # Preserve usage tracking
            if original_usage and hasattr(new_prediction, "set_lm_usage"):
                try:
                    new_prediction.set_lm_usage(original_usage)
                except Exception as e:
                    logger.warning(f"Failed to set_lm_usage(): {e}")

            return new_prediction
```

### Step 4: Use the Module

```python
# Initialize
module = ToolExecutorModule(lm=my_lm)

# Execute
result = module(
    task="Get valid properties for Composition and Status labels",
    context="Working with Neo4j graph database"
)

# Access results
print("Reasoning:", result.reasoning)
if result.tool_calls:
    print("Tool calls made:", len(result.tool_calls.tool_calls))
    print("Tool results:", result.tool_results)
print("Final result:", result.result)
```

### Critical Requirements Checklist

✅ **Tools wrapped explicitly** with `dspy.Tool(func, name, desc)`
✅ **Signature has 3 outputs**: `reasoning`, `tool_calls`, `direct_result`
✅ **Tools passed as INPUT** during `forward()`, not to constructor
✅ **ChatAdapter used** with `use_native_function_calling=True`
✅ **Manual execution implemented** via `_execute_tool_calls()`
✅ **Explicit function mapping** via `_build_tool_functions_map()`
✅ **Error handling** for missing tools and execution failures
✅ **Both paths handled**: tool calling and direct answer

---

## Part 2: Prototyping Pattern (ReAct)

### When to Use ReAct

Use `dspy.ReAct` for:
- ✅ Quick prototypes and demos
- ✅ Simple use cases with trusted tools
- ✅ Exploratory development

**Do NOT use ReAct for:**
- ❌ Production systems requiring control
- ❌ Complex error handling requirements
- ❌ Custom tool execution logic
- ❌ Debugging and monitoring needs

### ReAct Pattern

`dspy.ReAct` automatically handles tool execution in an internal loop:

```python
import dspy

# Define tools
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"The weather in {city} is sunny and 75°F"

def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Search results for '{query}': [relevant information...]"

# Create ReAct agent
react_agent = dspy.ReAct(
    signature="question -> answer",
    tools=[get_weather, search_web],  # Tools to constructor
    max_iters=5
)

# Use the agent
result = react_agent(question="What's the weather like in Tokyo?")
print(result.answer)
print("Trajectory:", result.trajectory)
```

**How ReAct Works:**
1. Receives a question
2. Reasons about the current situation (`next_thought`)
3. Selects a tool to call (`next_tool_name`)
4. Provides arguments for the tool (`next_tool_args`)
5. **Automatically executes** the tool and observes the result
6. Repeats until calling the `finish` tool
7. Extracts the final answer from the trajectory

**Limitations:**
- ❌ No control over execution
- ❌ Hard to debug failures
- ❌ Can't customize error handling
- ❌ Black box behavior
- ❌ May use many tokens (iterative loop)

---

## Defining DSPy Tools

### Basic Function Definition

The simplest way to create a DSPy tool:

```python
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    # In a real implementation, this would call a weather API
    return f"The weather in {city} is sunny and 75°F"

def search_web(query: str) -> str:
    """Search the web for information."""
    # In a real implementation, this would call a search API
    return f"Search results for '{query}': [relevant information...]"
```

**Key Requirements:**
- **Type hints**: Required for all parameters and return type
- **Docstring**: Used as the tool description for the LLM
- **Simple types**: Use `str`, `int`, `float`, `bool`, `list`, `dict`, or Pydantic models

### Explicit dspy.Tool Wrapper (Recommended)

**Always wrap functions explicitly in production code:**

```python
import dspy

def my_function(param1: str, param2: int = 5) -> str:
    """A sample function with parameters."""
    return f"Processed {param1} with value {param2}"

# Create a tool with explicit metadata
tool = dspy.Tool(
    func=my_function,
    name="my_function",  # Explicit name
    desc="A sample function with parameters.",  # Explicit description
)

# Access tool properties
print(tool.name)  # "my_function"
print(tool.desc)  # "A sample function with parameters."
print(tool.args)  # {'param1': {'type': 'string'}, 'param2': {'type': 'integer', 'default': 5}}
```

### Custom Tool Configuration

Override automatic inference:

```python
tool = dspy.Tool(
    func=my_function,
    name="custom_processor",              # Override function name
    desc="Custom description for LLM",    # Override docstring
    args={'input': {'type': 'string'}},   # Override arg schema
    arg_types={'input': str},             # Override arg types
    arg_desc={'input': 'The input text'}  # Override arg descriptions
)
```

## Best Practices

### 1. Clear Docstrings

```python
def good_tool(city: str, units: str = "celsius") -> str:
    """Get weather information for a specific city.

    Args:
        city: The name of the city to get weather for
        units: Temperature units, either 'celsius' or 'fahrenheit'

    Returns:
        A string describing the current weather conditions
    """
    if not city.strip():
        return "Error: City name cannot be empty"

    # Weather logic here...
    return f"Weather in {city}: 25°{units[0].upper()}, sunny"
```

### 2. Type Hints

```python
# Good - Clear type hints
def calculate_percentage(value: float, total: float) -> float:
    """Calculate percentage of value relative to total."""
    return (value / total) * 100 if total != 0 else 0.0

# Good - Using Pydantic models for complex types
from pydantic import BaseModel

class UserProfile(BaseModel):
    user_id: str
    name: str
    email: str

def get_user_info(name: str) -> UserProfile:
    """Fetch the user profile from database with given name."""
    return user_database.get(name)
```

### 3. Error Handling

```python
def weather(city: str) -> str:
    """Get weather information for a city."""
    try:
        if not city.strip():
            return "Error: City name cannot be empty"

        result = call_weather_api(city)
        return f"The weather in {city} is {result}"
    except Exception as e:
        return f"Error fetching weather: {str(e)}"
```

### 4. Default Parameters

```python
def search_database(query: str, table: str = "users") -> list[dict]:
    """Search database table with query."""
    return [{"id": 1, "name": "Example", "query": query, "table": table}]

# The default value will be included in the tool schema
tool = dspy.Tool(search_database)
print(tool.args)
# {'query': {'type': 'string'},
#  'table': {'type': 'string', 'default': 'users'}}
```

---

## Common Mistakes and Solutions

### ❌ Mistake 1: Using ReAct Instead of ChainOfThought

**Symptom:** You want full control over tool execution in production.

**Wrong:**
```python
agent = dspy.ReAct(
    signature="question -> answer",
    tools=[get_weather, search_web]  # Black box, automatic execution
)
```

**Right:**
```python
# Define signature with tools input and tool_calls output
class MySignature(dspy.Signature):
    question: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField()
    reasoning: str = dspy.OutputField()
    tool_calls: dspy.ToolCalls = dspy.OutputField()
    direct_result: str | None = dspy.OutputField()

# Use ChainOfThought with manual execution
predictor = dspy.ChainOfThought(MySignature)

with dspy.context(adapter=dspy.ChatAdapter(use_native_function_calling=True)):
    result = predictor(question="...", tools=self._tools)

# Manually execute tools
if result.tool_calls and result.tool_calls.tool_calls:
    tool_results = self._execute_tool_calls(result.tool_calls)
```

### ❌ Mistake 2: Missing ChatAdapter

**Symptom:** Tool calls not being generated, or malformed tool_calls output.

**Wrong:**
```python
# No ChatAdapter - tool calls may not work!
result = predictor(tools=[...], question="...")
```

**Right:**
```python
# ChatAdapter is REQUIRED for ChainOfThought tool calling
with dspy.context(
    adapter=dspy.ChatAdapter(use_native_function_calling=True),
    tool_choice="auto",
):
    result = predictor(tools=[...], question="...")
```

**Why ChatAdapter is required:**
- Enables native LLM function calling format
- Ensures `tool_calls` are properly generated
- Required for GPT-4, Claude, and most modern LLMs

### ❌ Mistake 3: Passing Raw Functions Instead of dspy.Tool

**Symptom:** Tools not discovered or names/descriptions incorrect.

**Wrong:**
```python
def get_weather(city: str) -> str:
    """Get weather."""
    return f"Weather in {city}"

# Passing raw function
self._tools = [get_weather]  # May not work reliably
```

**Right:**
```python
def get_weather(city: str) -> str:
    """Get weather."""
    return f"Weather in {city}"

# Wrap explicitly
weather_tool = dspy.Tool(
    func=get_weather,
    name="get_weather",
    desc="Get the current weather for a specified city"
)

self._tools = [weather_tool]  # Explicit and reliable
```

### ❌ Mistake 4: Tools Passed to Constructor

**Symptom:** Following ReAct examples for ChainOfThought.

**Wrong:**
```python
# Wrong for ChainOfThought
predictor = dspy.ChainOfThought(MySignature, tools=[...])
```

**Right:**
```python
# Right - tools as INPUT during forward()
predictor = dspy.ChainOfThought(MySignature)

result = predictor(
    question="...",
    tools=self._tools  # Tools as input field!
)
```

### ❌ Mistake 5: Incomplete Signature

**Symptom:** Only one output field, missing dual-path support.

**Wrong:**
```python
class ToolSignature(dspy.Signature):
    question: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField()
    outputs: dspy.ToolCalls = dspy.OutputField()  # Only 1 output!
```

**Right:**
```python
class ToolSignature(dspy.Signature):
    question: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField()

    # All 3 outputs required for dual-path pattern
    reasoning: str = dspy.OutputField()
    tool_calls: dspy.ToolCalls = dspy.OutputField()
    direct_result: str | None = dspy.OutputField()
```

### ❌ Mistake 6: Not Implementing Manual Execution

**Symptom:** Expected automatic execution with ChainOfThought.

**Wrong:**
```python
result = predictor(tools=[...], question="...")
# Expecting tools to run automatically - they won't!
print(result.answer)  # No answer!
```

**Right:**
```python
result = predictor(tools=[...], question="...")

# Must manually execute tool calls
if result.tool_calls and result.tool_calls.tool_calls:
    tool_results = self._execute_tool_calls(result.tool_calls)
    # Now process tool_results
```

### ❌ Mistake 7: Relying on Automatic Tool Discovery

**Symptom:** Tools not found or wrong tools called.

**Wrong:**
```python
# Relying on magic discovery
for call in tool_calls.tool_calls:
    result = call.execute()  # May fail in production!
```

**Right:**
```python
# Explicit function mapping
def _build_tool_functions_map(self) -> dict[str, Callable]:
    return {tool.name: tool.func for tool in self._tools}

def _execute_tool_calls(self, tool_calls):
    results = []
    functions_map = self._build_tool_functions_map()

    for call in tool_calls.tool_calls:
        if call.name not in functions_map:
            results.append(f"Error: Tool '{call.name}' not found")
            continue

        func = functions_map[call.name]
        result = func(**call.args)
        results.append(result)

    return results
```

---

## Advanced Patterns

### Using Pydantic Models

```python
from pydantic import BaseModel

class Date(BaseModel):
    year: int
    month: int
    day: int
    hour: int

class Flight(BaseModel):
    flight_id: str
    date_time: Date
    origin: str
    destination: str
    duration: float
    price: float

def fetch_flight_info(date: Date, origin: str, destination: str) -> list[Flight]:
    """Fetch flight information from origin to destination on the given date"""
    flights = []
    for flight_id, flight in flight_database.items():
        if (flight.date_time.year == date.year and
            flight.date_time.month == date.month and
            flight.date_time.day == date.day and
            flight.origin == origin and
            flight.destination == destination):
            flights.append(flight)
    return flights
```

### Asynchronous Tools

```python
import asyncio
import dspy

async def async_weather(city: str) -> str:
    """Get weather information asynchronously."""
    await asyncio.sleep(0.1)  # Simulate async API call
    return f"The weather in {city} is sunny"

# Create tool
tool = dspy.Tool(async_weather)

# Use acall for async tools
result = await tool.acall(city="New York")
print(result)
```

### Integration with MCP (Model Context Protocol)

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import asyncio

async def run():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # List available tools
            tools = await session.list_tools()

            # Convert MCP tools to DSPy tools
            dspy_tools = []
            for tool in tools.tools:
                dspy_tools.append(dspy.Tool.from_mcp_tool(session, tool))

            print(f"Loaded {len(dspy_tools)} tools")

asyncio.run(run())
```

### Integration with LangChain Tools

```python
import dspy
from langchain.tools import tool as lc_tool

@lc_tool
def add(x: int, y: int):
    """Add two numbers together."""
    return x + y

# Convert LangChain tool to DSPy tool
dspy_tool = dspy.Tool.from_langchain(add)

# Use asynchronously
async def run_tool():
    return await dspy_tool.acall(x=1, y=2)

print(asyncio.run(run_tool()))  # Output: 3
```

---

## Summary

### Production Checklist

**Tool Definition:**
- ✅ Type hints for all parameters and return type
- ✅ Clear, descriptive docstring
- ✅ Explicit wrapping with `dspy.Tool(func, name, desc)`
- ✅ Error handling for edge cases
- ✅ Default values where appropriate

**Module Implementation:**
- ✅ Use `dspy.ChainOfThought` (not ReAct)
- ✅ Signature with 3 outputs: `reasoning`, `tool_calls`, `direct_result`
- ✅ Tools as `InputField` in signature
- ✅ ChatAdapter with `use_native_function_calling=True`
- ✅ Tools passed as input during `forward()`
- ✅ Manual execution via `_execute_tool_calls()`
- ✅ Explicit function mapping via `_build_tool_functions_map()`
- ✅ Error handling for missing tools and execution failures
- ✅ Handle both tool calling and direct answer paths

### Key Takeaways

1. **For Production**: Use ChainOfThought + manual execution pattern
2. **For Prototyping**: Use ReAct for quick demos
3. **Always wrap tools** explicitly with `dspy.Tool`
4. **ChatAdapter is required** for ChainOfThought tool calling
5. **Tools are inputs**, not constructor parameters
6. **Implement manual execution** - don't expect automatic execution
7. **Use explicit function mapping** - don't rely on discovery
8. **Handle errors gracefully** - both missing tools and execution failures
9. **Support dual paths** - tool calling and direct answers
10. **Test thoroughly** - verify tool calls are actually executed

### Pattern Reference

```python
# Complete production pattern
class MyModule(dspy.Module):
    def __init__(self):
        self._tools = [dspy.Tool(func, name, desc), ...]
        self.executor = dspy.ChainOfThought(MySignature)

    def _build_tool_functions_map(self):
        return {tool.name: tool.func for tool in self._tools}

    def _execute_tool_calls(self, tool_calls):
        # Manual execution with error handling
        ...

    def forward(self, task, context):
        with dspy.context(
            adapter=dspy.ChatAdapter(use_native_function_calling=True),
            tool_choice="auto",
        ):
            res = self.executor(task=task, context=context, tools=self._tools)

        if res.tool_calls and res.tool_calls.tool_calls:
            tool_results = self._execute_tool_calls(res.tool_calls)
            # Process results
        else:
            # Use direct_result

        return prediction
```
