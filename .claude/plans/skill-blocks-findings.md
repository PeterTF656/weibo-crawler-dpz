# Findings: Skill Blocks Design

## Overview
This file captures research findings for the markdown-based skill blocks design.

---

## Source 1: Graph Topology (model.py)

### GRAPH_TOPOLOGY
All allowed edges between node types:
```python
GRAPH_TOPOLOGY = {
    ("Creator", "Composition"): [CREATE],
    ("Creator", "Status"): [HAVE],
    ("Creator", "Matching"): [CREATE],
    ("Status", "Emotion"): [SHOW],
    ("Status", "Motive"): [REFLECT],
    ("Status", "SelfPresenting"): [REFLECT],
    ("Composition", "Status"): [REFLECT],
    ("Composition", "Composition"): [HAPPEN_AFTER],
    ("Composition", "Narrative"): [REFLECT],
    ("Composition", "Idea"): [REFLECT],
    ("Composition", "Structure"): [REFLECT],
    ("Composition", "Matching"): [BELONG_TO],
}
```
**Total**: 12 valid edge transitions

### Valid Node Labels
```python
ValidNodeLabel = Literal[
    "Creator", "Composition", "Emotion", "Idea", "Matching",
    "Motive", "Narrative", "Structure", "Status", "SelfPresenting"
]
```

### Vector-Searchable Labels
```python
VALID_VECTOR_SEARCH_LABELS = [
    "Emotion", "Idea", "Motive", "Narrative", "Structure", "SelfPresenting"
]
```
**Vector Search Types**: `Description`, `Logic`, `Title`

### Helper Functions
| Function | Purpose |
|----------|---------|
| `get_valid_next_labels_from_topology(label)` | Get reachable labels (bidirectional) |
| `get_valid_edges_for_transition(from, to)` | Get valid edges for a transition |
| `get_property_keys_for_label(label)` | Get valid property keys |

### Graph Structure
- **Hub nodes**: Composition (6 connections), Creator (3), Status (3)
- **Leaf nodes**: Emotion, Idea, Motive, Narrative, Structure, SelfPresenting, Matching

---

## Source 2: Property Schema (property_schema.py + schema_types.py)

### Per-Label Properties
| Label | Properties |
|-------|------------|
| Creator | `user_id` |
| Composition | `mongodb_id`, `created_at`, `creator_id` |
| Status | `mongodb_id`, `created_at` |
| Matching | `mongodb_id`, `created_at`, `task_id` |
| Emotion, Idea, Motive, Narrative, SelfPresenting | `created_at` |
| Structure | `mongodb_id`, `created_at` |

### Property Operators
```python
Literal["eq", "ne", "gt", "gte", "lt", "lte"]  # default: "eq"
```

### Property Type Validation
```python
PROPERTY_TYPE_SCHEMA = {
    "created_at": "iso_datetime",
    "updated_at": "iso_datetime",
    "user_id": "uuid_or_objectid",
    "creator_id": "uuid_or_objectid",
    "mongodb_id": "uuid_or_objectid",
}
```

### Usage Contexts
1. `node.include` - Nodes MUST match (AND logic)
2. `nodeExclusion` - Nodes must NOT match (OR logic)
3. `nodeConditions` - Additional WHERE clauses

---

## Source 3: DSPy Integration Patterns

### Signature Pattern
```python
class MySignature(dspy.Signature):
    """Docstring with LLM instructions"""

    input_field: Type = dspy.InputField(desc="...")
    tools: List[dspy.Tool] = dspy.InputField(desc="...")

    reasoning: str = dspy.OutputField(desc="...")
    result: Type = dspy.OutputField(desc="...")
```

### Module Pattern
```python
class MyModule(dspy.Module):
    def __init__(self, lm=None):
        super().__init__()
        self._tools = [tool1, tool2]
        self.executor = dspy.ChainOfThought(MySignature)

    def forward(self, **inputs):
        res = self.executor(tools=self._tools, **inputs)
        tool_results = self._execute_tool_calls(res.tool_calls)
        return dspy.Prediction(result=parsed_results)
```

### Tool Definition Pattern
```python
def my_tool_func(arg1: str, arg2: int) -> Dict:
    """USE THIS TOOL WHEN: [guidance]

    This tool:
    - [capability]

    Args:
        arg1: [description]

    Returns:
        [description]
    """
    # implementation

tool = dspy.Tool(func=my_tool_func, name="my_tool")
```

### LangGraph Node Pattern
```python
@mlflow.trace(name="node_name")
async def my_node(state: AgentState, config: RunnableConfig):
    # Get tools from state
    tools = state.get("tools", {})

    # Call DSPy module
    module = MyModule()
    result = await module.aforward(**inputs)

    # Return state updates
    return {**state, "output": result}
```

---

## Source 4: ComplexQueryInput Runtime Model

### Structure
```
ComplexQueryInput
├── branches: List[Branch]
│   └── Branch.levels: List[Level]
│       ├── node: NodeFilter (labels, include)
│       ├── nodeExclusion: List[NodeProperty]
│       ├── nodeConditions: List[NodeProperty]
│       ├── vectorSearches: List[VectorSearch]  # 1-5 max
│       ├── path: Path  # to reach NEXT level
│       │   ├── edgeTypes: List[RelationshipType]
│       │   ├── minHops, maxHops  # maxHops-minHops < 2!
│       │   └── direction: IN|OUT|BIDIRECTIONAL
│       └── weight: float (0-1)
├── projection: List[str]
└── limit: int
```

### Key Constraints
- **Topology validation**: Branch sequences must follow GRAPH_TOPOLOGY
- **Hop constraint**: `maxHops - minHops < 2` (memory explosion prevention)
- **Vector search limit**: 1-5 per level
- **Min k**: Vector search k >= 10

---

## Key Insights

1. **Single Source of Truth**: `schema_types.py` Literal types -> `property_schema.py` runtime dicts
2. **Existing tools are ready**: Helper functions exist, just need `dspy.Tool` wrappers
3. **Markdown skills should embed into DSPy signatures**: Use docstrings or load from files
4. **Validation is structural**: Happens at query construction, provides precise error locations
5. **LEGO philosophy fits perfectly**: Atomic blocks -> Segments -> Branches -> Query

---

## Open Questions Resolved

1. **Should segment skills be loaded on-demand or pre-loaded?**
   - **Answer**: Load on-demand. Skills are documentation templates, actual validation uses existing helpers.

2. **How to handle the 5 vector search per level limit?**
   - **Answer**: Document in skill template constraints section. Agent must split across levels if needed.

3. **Should skills include example Cypher output?**
   - **Answer**: Yes, for educational purposes. Examples help LLMs understand intended output.

4. **How to version/update generated skills when topology changes?**
   - **Answer**: Regenerate from `GRAPH_TOPOLOGY`. Skills are derived, not hand-written.
