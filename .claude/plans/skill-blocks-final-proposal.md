# Final Proposal: Markdown-Based Skill Blocks for nl_to_cypher

**Date**: 2026-01-13
**Status**: Ready for Review

---

## Executive Summary

This proposal defines a markdown-based skill block system for AI agents to convert natural language queries into topology-valid Cypher. The system uses **programmatically generated** skill templates derived from `GRAPH_TOPOLOGY` and `LABEL_PROPERTIES`, ensuring skills stay synchronized with the schema.

### Key Design Decisions

1. **Markdown over YAML**: AI agents process markdown naturally; supports tables, code blocks, and nested structures
2. **Programmatic generation**: All primitive blocks derived from existing schema definitions
3. **Skills as templates, not code**: Skills document how to use tools; validation happens in existing helpers
4. **LEGO composition**: Atomic blocks -> Segments -> Branches -> Query

---

## Part 1: Markdown Format Specifications

### 1.1 Primitive Block Types

#### NodeType Block

```markdown
# NodeType: Composition

## Overview
| Attribute | Value |
|-----------|-------|
| **Type** | `NodeType` |
| **Label** | `Composition` |
| **Vector Search** | No |

## Properties
| Key | Type | Description |
|-----|------|-------------|
| `mongodb_id` | `uuid_or_objectid` | MongoDB document ID |
| `created_at` | `iso_datetime` | Creation timestamp |
| `creator_id` | `uuid_or_objectid` | Owner's user ID |

## Connectivity
### Outgoing Edges (FROM Composition)
| Target | Edge | Description |
|--------|------|-------------|
| `Status` | `REFLECT` | Composition reflects a status |
| `Composition` | `HAPPEN_AFTER` | Temporal sequence |
| `Narrative` | `REFLECT` | Narrative aspect extraction |
| `Idea` | `REFLECT` | Ideational aspect extraction |
| `Structure` | `REFLECT` | Structural aspect extraction |
| `Matching` | `BELONG_TO` | Assigned to matching task |

### Incoming Edges (TO Composition)
| Source | Edge | Description |
|--------|------|-------------|
| `Creator` | `CREATE` | Creator authored this |
| `Composition` | `HAPPEN_AFTER` | Follows another composition |

## Usage Notes
- Hub node with highest connectivity (6 outgoing paths)
- Common anchor for user-facing queries
- Does NOT support vector search directly (use connected Idea/Narrative nodes)
```

#### EdgeType Block

```markdown
# EdgeType: REFLECT

## Overview
| Attribute | Value |
|-----------|-------|
| **Type** | `EdgeType` |
| **Relationship** | `REFLECT` |

## Valid Transitions
| Source | Target | Semantic Meaning |
|--------|--------|------------------|
| `Status` | `Motive` | Status reflects motivational aspects |
| `Status` | `SelfPresenting` | Status reflects self-presentation |
| `Composition` | `Status` | Composition reflects a status update |
| `Composition` | `Narrative` | Composition contains narrative elements |
| `Composition` | `Idea` | Composition contains ideas |
| `Composition` | `Structure` | Composition has structural elements |

## Path Configuration
```python
Path(
    edgeTypes=[RelationshipType.REFLECT],
    minHops=1,
    maxHops=1,
    direction="OUT"  # typically outgoing
)
```
```

#### PropertyConstraint Block

```markdown
# PropertyConstraint

## Structure
| Field | Type | Required | Default |
|-------|------|----------|---------|
| `key` | `AnyPropertyKey` | Yes | - |
| `value` | `Any` | Yes | - |
| `operator` | `Literal["eq","ne","gt","gte","lt","lte"]` | No | `"eq"` |

## Operators
| Operator | Cypher | Example |
|----------|--------|---------|
| `eq` | `=` | `creator_id = "user123"` |
| `ne` | `<>` | `status != "draft"` |
| `gt` | `>` | `created_at > "2025-01-01"` |
| `gte` | `>=` | `score >= 0.5` |
| `lt` | `<` | `age < 30` |
| `lte` | `<=` | `priority <= 3` |

## Validation Rules
1. `key` must exist in `LABEL_TO_PROPERTY_KEYS[label]`
2. `value` type must match `PROPERTY_TYPE_SCHEMA[key]`:
   - `iso_datetime`: ISO 8601 format string
   - `uuid_or_objectid`: UUID v4 or 24-char hex ObjectId
   - `any`: No validation

## Example
```python
NodeProperty(
    key="created_at",
    operator="gte",
    value="2025-01-01T00:00:00Z"
)
```
```

#### VectorSearchSpec Block

```markdown
# VectorSearchSpec

## Overview
Semantic similarity search on vector-indexed node labels.

## Structure
| Field | Type | Required | Default |
|-------|------|----------|---------|
| `queryText` | `str` | Yes | - |
| `vectorSearchType` | `Literal["Description","Logic","Title"]` | Yes | - |
| `k` | `int` (>= 10) | Yes | - |
| `vectorIndex` | `str` | No | auto-derived |
| `queryEmbedding` | `List[float]` | No | auto-computed |

## Valid Labels
Only these labels support vector search:
- `Emotion`
- `Idea`
- `Motive`
- `Narrative`
- `Structure`
- `SelfPresenting`

## Vector Search Types
| Type | Embedding Field | Use When |
|------|-----------------|----------|
| `Description` | `description_embedding` | General semantic meaning |
| `Logic` | `logic_embedding` | Reasoning/logic patterns |
| `Title` | `title_embedding` | Brief topic matching |

## Example
```python
VectorSearch(
    queryText="feelings of happiness and joy",
    vectorSearchType="Description",
    k=20
)
```

## Constraints
- Maximum 5 vector searches per Level
- k must be >= 10
- Label must be in VALID_VECTOR_SEARCH_LABELS
```

---

### 1.2 EdgeSegment Block (LEGO Connector)

```markdown
# EdgeSegment: Composition -> REFLECT -> Idea

## Overview
| Attribute | Value |
|-----------|-------|
| **Type** | `EdgeSegment` |
| **Source** | `Composition` |
| **Edge** | `REFLECT` |
| **Target** | `Idea` |
| **Hops** | `1` (fixed) |

## Validation
- **Topology Valid**: `GRAPH_TOPOLOGY[("Composition", "Idea")] = [REFLECT]`
- **Bidirectional**: Can traverse IN or OUT direction

## As Level Path
```python
# Level N (Composition) -> Level N+1 (Idea)
Level(
    node=NodeFilter(labels=["Composition"]),
    path=Path(
        edgeTypes=[RelationshipType.REFLECT],
        minHops=1,
        maxHops=1,
        direction="OUT"
    )
)
```

## Cypher Pattern
```cypher
MATCH (comp:Composition)-[:REFLECT]->(idea:Idea)
```

## Chaining Rules
- **Can chain FROM**: `Creator` (via CREATE), `Composition` (via HAPPEN_AFTER)
- **Can chain TO**: Nothing (Idea is a leaf node)
```

---

### 1.3 Composition SKILLs (Templates for Agents)

#### SKILL: compose_branch

```markdown
# SKILL: compose_branch

## Purpose
Assemble a sequence of EdgeSegments into a Branch that terminates at an anchor.

## When to Use
- Building a constraint path from entry point to anchor
- Connecting semantic searches to result type
- Traversing multi-hop relationships

## Input
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `entry` | `NodeBlock` | Yes | Where the branch starts |
| `segments` | `List[EdgeSegment]` | Yes | Ordered traversal steps |
| `anchor` | `NodeBlock` | Yes | Where branch ends (usually result type) |

## Output
| Field | Type | Description |
|-------|------|-------------|
| `branch` | `Branch` | ComplexQueryInput.Branch ready for query |

## Validation Rules
1. **Segment Chaining**: `segments[i].target == segments[i+1].source`
2. **Entry Match**: `entry.label == segments[0].source`
3. **Anchor Match**: `segments[-1].target == anchor.label`
4. **Topology Valid**: Each segment must exist in GRAPH_TOPOLOGY

## Tools Available
```python
get_valid_next_labels_from_topology(current_label) -> List[str]
get_valid_edges_for_transition(from_label, to_label) -> List[RelationshipType]
get_property_keys_for_label(label) -> List[str]
```

## Workflow
1. Start at `entry` NodeBlock
2. For each segment, validate topology
3. Convert segments to Level objects with paths
4. Add decorations (properties, vector search) to appropriate levels
5. Return assembled Branch

## Example
**Query**: "Find compositions about school topics"

```python
# Entry: Idea node with vector search
entry = NodeBlock(
    label="Idea",
    vector_search=VectorSearchSpec(
        queryText="school education learning",
        vectorSearchType="Description",
        k=20
    )
)

# Single segment: Idea <- REFLECT <- Composition
segments = [
    EdgeSegment(source="Idea", edge="REFLECT", target="Composition", direction="IN")
]

# Anchor: Composition (result type)
anchor = NodeBlock(label="Composition")

# Result: Branch with 2 levels
branch = compose_branch(entry, segments, anchor)
```

## Common Patterns
| Pattern | Entry | Path | Anchor |
|---------|-------|------|--------|
| Topic search | Idea (vector) | Idea <- Composition | Composition |
| Emotion filter | Emotion (vector) | Emotion <- Status <- Creator -> Composition | Composition |
| My compositions | Creator (props) | Creator -> Composition | Composition |
| Time filter | Composition (props) | Direct | Composition |
```

#### SKILL: compose_query

```markdown
# SKILL: compose_query

## Purpose
Combine multiple branches into a complete query with boolean semantics.

## When to Use
- Combining multiple independent constraints
- Building AND/OR logic across branches
- Creating the final ComplexQueryInput

## Input
| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `anchor` | `ValidNodeLabel` | Yes | Result type (e.g., "Composition") |
| `branches` | `List[Branch]` | Yes | All must terminate at anchor |
| `operator` | `Literal["AND", "OR"]` | Yes | How to combine branch results |
| `projection` | `List[str]` | No | Properties to return |
| `limit` | `int` | No | Max results (default 100) |

## Output
| Field | Type | Description |
|-------|------|-------------|
| `query` | `ComplexQueryInput` | Ready for Cypher generation |

## Semantics
| Operator | Meaning | Cypher Logic |
|----------|---------|--------------|
| `AND` | All branches must match | Intersection of results |
| `OR` | Any branch may match | Union of results |

## Validation Rules
1. **Anchor Consistency**: All branches must end at `anchor`
2. **Branch Validity**: Each branch must pass topology validation
3. **Reasonable Size**: Total levels across branches should be manageable

## Example
**Query**: "My compositions about school that are connected to happy emotions"

```python
# Branch A: Topic constraint (school)
topic_branch = compose_branch(
    entry=NodeBlock(label="Idea", vector_search=...),
    segments=[EdgeSegment("Idea", "REFLECT", "Composition", "IN")],
    anchor=NodeBlock(label="Composition")
)

# Branch B: Ownership constraint (my)
owner_branch = compose_branch(
    entry=NodeBlock(label="Creator", property_constraints=[
        PropertyConstraint(key="user_id", op="eq", value=current_user_id)
    ]),
    segments=[EdgeSegment("Creator", "CREATE", "Composition", "OUT")],
    anchor=NodeBlock(label="Composition")
)

# Branch C: Emotion constraint (happy) - requires multi-hop
emotion_branch = compose_branch(
    entry=NodeBlock(label="Emotion", vector_search=...),
    segments=[
        EdgeSegment("Emotion", "SHOW", "Status", "IN"),
        EdgeSegment("Status", "REFLECT", "Composition", "IN")
    ],
    anchor=NodeBlock(label="Composition")
)

# Combine with AND
query = compose_query(
    anchor="Composition",
    branches=[topic_branch, owner_branch, emotion_branch],
    operator="AND",
    limit=50
)
```

## Output Format
```python
ComplexQueryInput(
    branches=[...],
    projection=["mongodb_id", "created_at", "creator_id"],
    limit=50,
    debug=False
)
```
```

---

## Part 2: Programmatic Generation Approach

### 2.1 Generator Architecture

```
GRAPH_TOPOLOGY + LABEL_PROPERTIES + VALID_VECTOR_SEARCH_LABELS
                              |
                              v
                    ┌─────────────────┐
                    │   Generator     │
                    │   Functions     │
                    └─────────────────┘
                              |
              ┌───────────────┼───────────────┐
              v               v               v
        NodeType.md     EdgeSegment.md   Composition
        (10 files)      (12 files)       Skills.md
                                         (static)
```

### 2.2 Generator Functions

```python
# src/tools/graphdb/search/nl_to_cypher/skill_generator.py

from typing import List
from src.tools.graphdb.search.nl_to_cypher.model import (
    GRAPH_TOPOLOGY,
    LABEL_TO_PROPERTY_KEYS,
    VALID_VECTOR_SEARCH_LABELS,
    get_valid_next_labels_from_topology,
)
from src.tools.graphdb.search.nl_to_cypher.property_schema import (
    get_expected_type_for_property,
)


def generate_node_type_skill(label: str) -> str:
    """Generate markdown skill for a NodeType."""
    properties = LABEL_TO_PROPERTY_KEYS.get(label, [])
    is_vector_searchable = label in VALID_VECTOR_SEARCH_LABELS
    next_labels = get_valid_next_labels_from_topology(label)

    # Get outgoing edges
    outgoing = []
    for (src, tgt), edges in GRAPH_TOPOLOGY.items():
        if src == label:
            for edge in edges:
                outgoing.append((tgt, edge.value))

    # Get incoming edges
    incoming = []
    for (src, tgt), edges in GRAPH_TOPOLOGY.items():
        if tgt == label:
            for edge in edges:
                incoming.append((src, edge.value))

    md = f"""# NodeType: {label}

## Overview
| Attribute | Value |
|-----------|-------|
| **Type** | `NodeType` |
| **Label** | `{label}` |
| **Vector Search** | {"Yes" if is_vector_searchable else "No"} |

## Properties
| Key | Type | Description |
|-----|------|-------------|
"""
    for prop in properties:
        prop_type = get_expected_type_for_property(prop)
        md += f"| `{prop}` | `{prop_type}` | - |\n"

    if not properties:
        md += "| *(none)* | - | - |\n"

    md += f"""
## Connectivity
### Outgoing Edges (FROM {label})
| Target | Edge | Description |
|--------|------|-------------|
"""
    for tgt, edge in outgoing:
        md += f"| `{tgt}` | `{edge}` | - |\n"

    if not outgoing:
        md += "| *(leaf node - no outgoing edges)* | - | - |\n"

    md += f"""
### Incoming Edges (TO {label})
| Source | Edge | Description |
|--------|------|-------------|
"""
    for src, edge in incoming:
        md += f"| `{src}` | `{edge}` | - |\n"

    if not incoming:
        md += "| *(root node - no incoming edges)* | - | - |\n"

    if is_vector_searchable:
        md += """
## Vector Search
This label supports semantic search with types:
- `Description` - General semantic meaning
- `Logic` - Reasoning patterns
- `Title` - Brief topic matching
"""

    return md


def generate_edge_segment_skill(source: str, edge: str, target: str) -> str:
    """Generate markdown skill for an EdgeSegment."""
    return f"""# EdgeSegment: {source} -> {edge} -> {target}

## Overview
| Attribute | Value |
|-----------|-------|
| **Type** | `EdgeSegment` |
| **Source** | `{source}` |
| **Edge** | `{edge}` |
| **Target** | `{target}` |
| **Hops** | `1` (fixed) |

## Validation
- **Topology Valid**: `GRAPH_TOPOLOGY[("{source}", "{target}")] = [{edge}]`
- **Bidirectional**: Can traverse IN or OUT direction

## As Level Path
```python
Level(
    node=NodeFilter(labels=["{source}"]),
    path=Path(
        edgeTypes=[RelationshipType.{edge}],
        minHops=1,
        maxHops=1,
        direction="OUT"
    )
)
```

## Cypher Pattern
```cypher
MATCH (s:{source})-[:{edge}]->(t:{target})
```
"""


def generate_all_skills() -> dict[str, str]:
    """Generate all skill markdown files."""
    skills = {}

    # Generate NodeType skills
    all_labels = set()
    for (src, tgt), _ in GRAPH_TOPOLOGY.items():
        all_labels.add(src)
        all_labels.add(tgt)

    for label in sorted(all_labels):
        skills[f"nodes/{label}.md"] = generate_node_type_skill(label)

    # Generate EdgeSegment skills
    for (src, tgt), edges in GRAPH_TOPOLOGY.items():
        for edge in edges:
            filename = f"segments/{src}.{edge.value}.{tgt}.md"
            skills[filename] = generate_edge_segment_skill(src, edge.value, tgt)

    return skills
```

### 2.3 Usage: Generate Skills on Demand

```python
# In DSPy module or agent node
from src.tools.graphdb.search.nl_to_cypher.skill_generator import (
    generate_node_type_skill,
    generate_edge_segment_skill,
)

# Get skill for a specific node type
composition_skill = generate_node_type_skill("Composition")

# Get skill for a specific segment
segment_skill = generate_edge_segment_skill("Creator", "CREATE", "Composition")

# Embed in DSPy signature docstring
class QueryComposerSignature(dspy.Signature):
    f"""Convert natural language to graph query.

    ## Available Skills

    {generate_node_type_skill("Composition")}

    {generate_edge_segment_skill("Creator", "CREATE", "Composition")}
    """

    query: str = dspy.InputField()
    result: ComplexQueryInput = dspy.OutputField()
```

---

## Part 3: DSPy Integration

### 3.1 Skill-Aware Signature

```python
# src/tools/graphdb/search/nl_to_cypher/dspy/skill_signatures.py

import dspy
from src.tools.graphdb.search.nl_to_cypher.skill_generator import (
    generate_all_skills,
    generate_node_type_skill,
)
from src.tools.graphdb.search.nl_to_cypher.utils import build_topology_context_for_llm


def build_skill_context(labels: list[str]) -> str:
    """Build skill context for specific labels."""
    context = "## Relevant Node Skills\n\n"
    for label in labels:
        context += generate_node_type_skill(label) + "\n---\n"
    return context


class SkillAwareQuerySignature(dspy.Signature):
    """Convert natural language to topology-valid graph query using skills.

    You are a graph query composer. Use the provided skills to build
    valid queries. Each skill documents:
    - What nodes/edges are valid
    - What properties can be filtered
    - What vector searches are available

    RULES:
    1. Only use edges that appear in skill connectivity tables
    2. Only use properties listed in skill property tables
    3. Only use vector search on labels marked "Vector Search: Yes"
    4. Chain segments where target of one = source of next
    """

    natural_language: str = dspy.InputField(
        desc="User's search intent in natural language"
    )
    topology_context: str = dspy.InputField(
        desc="Graph topology summary from build_topology_context_for_llm()"
    )
    skill_context: str = dspy.InputField(
        desc="Relevant skill markdown loaded dynamically"
    )
    tools: list[dspy.Tool] = dspy.InputField(
        desc="Validation tools"
    )

    reasoning: str = dspy.OutputField(
        desc="Step-by-step reasoning using skills"
    )
    anchor: str = dspy.OutputField(
        desc="Result type label (usually Composition)"
    )
    branches_json: str = dspy.OutputField(
        desc="JSON array of branch specifications"
    )
```

### 3.2 Skill-Aware Module

```python
# src/tools/graphdb/search/nl_to_cypher/dspy/skill_modules.py

import dspy
import json
from typing import List

from src.tools.graphdb.search.nl_to_cypher.model import (
    ComplexQueryInput,
    get_valid_edges_for_transition,
    get_valid_next_labels_from_topology,
    get_property_keys_for_label,
)
from src.tools.graphdb.search.nl_to_cypher.skill_generator import (
    generate_node_type_skill,
    generate_edge_segment_skill,
)
from src.tools.graphdb.search.nl_to_cypher.utils import build_topology_context_for_llm


# Wrap existing helpers as dspy.Tool
def validate_segment(source: str, edge: str, target: str) -> str:
    """Validate if a segment is topology-valid.

    USE THIS TOOL WHEN: You need to verify an edge connection before using it.

    Args:
        source: Source node label
        edge: Relationship type name
        target: Target node label

    Returns:
        "VALID" if segment exists in topology, else error message
    """
    from src.models.model import RelationshipType

    try:
        edge_enum = RelationshipType(edge)
    except ValueError:
        return f"INVALID: '{edge}' is not a valid RelationshipType"

    valid_edges = get_valid_edges_for_transition(source, target)
    if edge_enum in valid_edges:
        return "VALID"
    else:
        valid_str = [e.value for e in valid_edges] if valid_edges else "none"
        return f"INVALID: {source}->{target} allows edges: {valid_str}, not {edge}"


def get_node_skill(label: str) -> str:
    """Get the skill documentation for a node type.

    USE THIS TOOL WHEN: You need to understand a node's properties and connections.

    Args:
        label: Node label (e.g., "Composition", "Idea")

    Returns:
        Markdown skill documentation
    """
    return generate_node_type_skill(label)


def get_segment_skill(source: str, edge: str, target: str) -> str:
    """Get the skill documentation for an edge segment.

    USE THIS TOOL WHEN: You need to understand how to traverse between two node types.

    Args:
        source: Source node label
        edge: Relationship type
        target: Target node label

    Returns:
        Markdown skill documentation
    """
    return generate_edge_segment_skill(source, edge, target)


# Create tool instances
validate_segment_tool = dspy.Tool(
    func=validate_segment,
    name="validate_segment"
)

get_node_skill_tool = dspy.Tool(
    func=get_node_skill,
    name="get_node_skill"
)

get_segment_skill_tool = dspy.Tool(
    func=get_segment_skill,
    name="get_segment_skill"
)


class SkillBasedQueryComposer(dspy.Module):
    """Compose graph queries using skill-based guidance."""

    def __init__(self, lm=None):
        super().__init__()
        self._lm_override = lm
        self._tools = [
            validate_segment_tool,
            get_node_skill_tool,
            get_segment_skill_tool,
        ]
        self.executor = dspy.ChainOfThought(SkillAwareQuerySignature)

    def forward(
        self,
        natural_language: str,
        relevant_labels: List[str] = None,
    ) -> dspy.Prediction:
        # Build context
        topology_context = build_topology_context_for_llm()

        # Build skill context for relevant labels
        if relevant_labels:
            skill_context = "\n---\n".join(
                generate_node_type_skill(label) for label in relevant_labels
            )
        else:
            # Default: include hub nodes
            skill_context = "\n---\n".join(
                generate_node_type_skill(label)
                for label in ["Composition", "Creator", "Status"]
            )

        # Execute with skills
        with dspy.context(lm=self._lm_override) if self._lm_override else nullcontext():
            result = self.executor(
                natural_language=natural_language,
                topology_context=topology_context,
                skill_context=skill_context,
                tools=self._tools,
            )

        # Parse and validate result
        try:
            branches_data = json.loads(result.branches_json)
            # Convert to ComplexQueryInput (with validation)
            query = self._build_query(result.anchor, branches_data)
            return dspy.Prediction(
                reasoning=result.reasoning,
                query=query,
                valid=True,
            )
        except Exception as e:
            return dspy.Prediction(
                reasoning=result.reasoning,
                error=str(e),
                valid=False,
            )

    def _build_query(self, anchor: str, branches_data: list) -> ComplexQueryInput:
        # Implementation converts branch specs to ComplexQueryInput
        # Uses existing Pydantic validation
        pass
```

---

## Part 4: File Organization

### 4.1 Directory Structure

```
src/tools/graphdb/search/nl_to_cypher/
├── skill-blocks/
│   ├── README.md                    # Design philosophy (existing)
│   ├── generated/                   # Auto-generated skills (gitignored)
│   │   ├── nodes/
│   │   │   ├── Composition.md
│   │   │   ├── Creator.md
│   │   │   └── ...                  # 10 files
│   │   └── segments/
│   │       ├── Creator.CREATE.Composition.md
│   │       ├── Composition.REFLECT.Idea.md
│   │       └── ...                  # 12 files
│   └── templates/                   # Static skill templates
│       ├── compose_branch.md
│       ├── compose_query.md
│       ├── decorate_node_with_properties.md
│       └── decorate_node_with_vector_search.md
├── skill_generator.py               # Generator functions
└── dspy/
    ├── skill_signatures.py          # DSPy signatures with skills
    └── skill_modules.py             # DSPy modules using skills
```

### 4.2 Generation Script

```python
# scripts/generate_skills.py

from pathlib import Path
from src.tools.graphdb.search.nl_to_cypher.skill_generator import generate_all_skills

OUTPUT_DIR = Path("src/tools/graphdb/search/nl_to_cypher/skill-blocks/generated")


def main():
    """Generate all skill markdown files."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    skills = generate_all_skills()

    for filepath, content in skills.items():
        output_path = OUTPUT_DIR / filepath
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)
        print(f"Generated: {output_path}")

    print(f"\nTotal: {len(skills)} skill files generated")


if __name__ == "__main__":
    main()
```

---

## Part 5: Evaluation Criteria

### 5.1 Validation Checklist

| Check | Description | Implementation |
|-------|-------------|----------------|
| Segment validity | Every segment in GRAPH_TOPOLOGY | `validate_segment()` tool |
| Branch chaining | `segment[i].target == segment[i+1].source` | Pydantic validator |
| Anchor consistency | All branches end at same anchor | Query builder check |
| Property validity | Keys in LABEL_TO_PROPERTY_KEYS | Existing validation |
| Vector search validity | Labels in VALID_VECTOR_SEARCH_LABELS | Existing validation |
| Hop constraint | `maxHops - minHops < 2` | Pydantic validator |

### 5.2 Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Skill coverage | 100% | All topology entries have skills |
| Topology validation | 0 errors | No invalid segments in output |
| Query parse rate | > 95% | Successfully parse to ComplexQueryInput |
| Fallback rate | < 10% | Queries requiring fallback to general AI |

---

## Part 6: Implementation Roadmap

### Phase 1: Generator Implementation (Week 1)
- [ ] Implement `skill_generator.py` with all generator functions
- [ ] Create generation script
- [ ] Generate initial skill files
- [ ] Review and refine markdown format

### Phase 2: DSPy Integration (Week 2)
- [ ] Create skill-aware signatures
- [ ] Wrap existing helpers as dspy.Tool
- [ ] Implement SkillBasedQueryComposer module
- [ ] Add MLflow tracing

### Phase 3: Testing & Evaluation (Week 3)
- [ ] Create test suite for generated skills
- [ ] Test skill-based query composition
- [ ] Measure validation rates
- [ ] Iterate on skill format based on results

### Phase 4: Production Integration (Week 4)
- [ ] Integrate with existing nl_to_cypher pipeline
- [ ] Add skill caching/loading optimization
- [ ] Documentation and developer guide
- [ ] Deployment

---

## Appendix: Example Generated Skills

### Example: Creator.md

```markdown
# NodeType: Creator

## Overview
| Attribute | Value |
|-----------|-------|
| **Type** | `NodeType` |
| **Label** | `Creator` |
| **Vector Search** | No |

## Properties
| Key | Type | Description |
|-----|------|-------------|
| `user_id` | `uuid_or_objectid` | - |

## Connectivity
### Outgoing Edges (FROM Creator)
| Target | Edge | Description |
|--------|------|-------------|
| `Composition` | `CREATE` | - |
| `Status` | `HAVE` | - |
| `Matching` | `CREATE` | - |

### Incoming Edges (TO Creator)
| Source | Edge | Description |
|--------|------|-------------|
| *(root node - no incoming edges)* | - | - |
```

### Example: Creator.CREATE.Composition.md

```markdown
# EdgeSegment: Creator -> CREATE -> Composition

## Overview
| Attribute | Value |
|-----------|-------|
| **Type** | `EdgeSegment` |
| **Source** | `Creator` |
| **Edge** | `CREATE` |
| **Target** | `Composition` |
| **Hops** | `1` (fixed) |

## Validation
- **Topology Valid**: `GRAPH_TOPOLOGY[("Creator", "Composition")] = [CREATE]`
- **Bidirectional**: Can traverse IN or OUT direction

## As Level Path
```python
Level(
    node=NodeFilter(labels=["Creator"]),
    path=Path(
        edgeTypes=[RelationshipType.CREATE],
        minHops=1,
        maxHops=1,
        direction="OUT"
    )
)
```

## Cypher Pattern
```cypher
MATCH (s:Creator)-[:CREATE]->(t:Composition)
```
```
