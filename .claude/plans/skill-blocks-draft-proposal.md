# DRAFT: Markdown-Based Skill Blocks Design

## 1. Markdown Format for Primitive Blocks

### 1.1 NodeType Block

```markdown
# NodeType: {label}

**Type:** `NodeType`
**Label:** `{ValidNodeLabel}`

## Capabilities
- Vector Search: {yes/no}
- Vector Search Types: {Description, Logic, Title} (if applicable)

## Valid Properties
| Property Key | Type | Description |
|-------------|------|-------------|
| {key1} | {type} | {description} |

## Valid Outgoing Edges
| Edge Type | Target Label |
|-----------|--------------|
| {edge} | {target} |

## Valid Incoming Edges
| Source Label | Edge Type |
|--------------|-----------|
| {source} | {edge} |
```

### 1.2 EdgeType Block

```markdown
# EdgeType: {relationship}

**Type:** `EdgeType`
**Relationship:** `{RelationshipType}`

## Valid Transitions
| Source Label | Target Label |
|--------------|--------------|
| {source} | {target} |
```

### 1.3 PropertyConstraint Block

```markdown
# PropertyConstraint

**Type:** `PropertyConstraint`

## Structure
- **Label:** {ValidNodeLabel}
- **Property Key:** {key}
- **Operator:** {eq | ne | gt | gte | lt | lte | contains | starts_with}
- **Value:** {value}

## Validation Rules
- Property key must exist in LABEL_TO_PROPERTY_KEYS[label]
- Value type must match property schema
```

### 1.4 VectorSearchSpec Block

```markdown
# VectorSearchSpec

**Type:** `VectorSearchSpec`

## Structure
- **Label:** {ValidNodeLabel} (must be in VALID_VECTOR_SEARCH_LABELS)
- **Vector Search Type:** {Description | Logic | Title}
- **Query Text:** {string}
- **K:** {integer, default 5}
- **Vector Index:** {string, optional}

## Validation Rules
- Label must be vector-searchable
- K must be positive integer
```

---

## 2. Decorated NodeBlock (Composite)

```markdown
# NodeBlock: {alias}

**Type:** `NodeBlock`
**Label:** `{ValidNodeLabel}`
**Role Hint:** `{anchor | owner | semantic | bridge | connection}` (optional)

## Property Constraints
| Key | Operator | Value |
|-----|----------|-------|
| creator_id | eq | {user_id} |
| created_at | gte | {timestamp} |

## Vector Search (Optional)
- **Type:** {Description}
- **Query:** "{semantic query text}"
- **K:** {5}

## Notes
{Any agent notes about this node's purpose in the query}
```

---

## 3. EdgeSegment Block (The LEGO Connector)

```markdown
# EdgeSegment: {source} -> {target}

**Type:** `EdgeSegment`
**Source:** `{ValidNodeLabel}`
**Edge:** `{RelationshipType}`
**Target:** `{ValidNodeLabel}`
**Direction:** `{OUT | IN | BIDIRECTIONAL}`
**Hops:** `1` (fixed for stability)

## Validation
- Segment valid: {true/false}
- Topology reference: GRAPH_TOPOLOGY[{source}][{target}]
```

---

## 4. Atomic Segment SKILLs (Auto-Generated)

### Template for Auto-Generation

For each valid `(source, edge, target)` in GRAPH_TOPOLOGY, generate:

```markdown
# SKILL: segment.{Source}.{EDGE}.{Target}

## Purpose
Connect a `{Source}` node to a `{Target}` node via `{EDGE}` relationship.

## Input
- **source_node:** NodeBlock with label `{Source}` (optional decorations)
- **target_node:** NodeBlock with label `{Target}` (optional decorations)

## Output
- **segment:** EdgeSegment connecting source to target

## Validation (Automatic)
- Source label: `{Source}` (enforced)
- Target label: `{Target}` (enforced)
- Edge type: `{EDGE}` (from topology)
- Direction: Determined by traversal context

## Usage Example
```cypher
MATCH (s:{Source})-[:{EDGE}]->(t:{Target})
```

## Tools Available
- `get_valid_edges_for_transition("{Source}", "{Target}")`
- `get_property_keys_for_label("{Source}")`
- `get_property_keys_for_label("{Target}")`
```

---

## 5. Composition SKILLs (Higher-Level Templates)

### 5.1 SKILL: decorate_node_with_properties

```markdown
# SKILL: decorate_node_with_properties

## Purpose
Add property constraints to a NodeBlock for filtering.

## Input
- **node:** NodeBlock
- **constraints:** List of PropertyConstraint

## Output
- **decorated_node:** NodeBlock with property_constraints populated

## Validation
- Each constraint key must exist in LABEL_TO_PROPERTY_KEYS[node.label]
- Value types must match property schema

## Example
Input:
- node: NodeBlock(label="Composition")
- constraints: [PropertyConstraint(key="creator_id", op="eq", value="user123")]

Output:
- NodeBlock with creator_id = "user123" filter
```

### 5.2 SKILL: decorate_node_with_vector_search

```markdown
# SKILL: decorate_node_with_vector_search

## Purpose
Add vector search to a NodeBlock for semantic matching.

## Input
- **node:** NodeBlock (label must be in VALID_VECTOR_SEARCH_LABELS)
- **spec:** VectorSearchSpec

## Output
- **decorated_node:** NodeBlock with vector_search populated

## Validation
- node.label must be in VALID_VECTOR_SEARCH_LABELS
- spec.vectorSearchType must be valid for the label

## Example
Input:
- node: NodeBlock(label="Idea")
- spec: VectorSearchSpec(type="Description", query="school activities", k=5)

Output:
- NodeBlock searching for Ideas semantically similar to "school activities"
```

### 5.3 SKILL: compose_branch

```markdown
# SKILL: compose_branch

## Purpose
Assemble a sequence of EdgeSegments into a Branch that terminates at an anchor.

## Input
- **entry:** NodeBlock (where the branch starts)
- **segments:** List[EdgeSegment] (must chain: segment[i].target == segment[i+1].source)
- **anchor:** NodeBlock (where the branch ends, usually the query result type)

## Output
- **branch:** Branch object ready for query composition

## Validation
- Segment chaining: Each segment's target must equal next segment's source
- Anchor connection: Last segment's target must match anchor.label
- All segments must be topology-valid

## Example
Branch for "compositions about school":
- entry: NodeBlock(label="Idea", vector_search="school")
- segments: [EdgeSegment(Idea -> REFLECT -> Composition)]
- anchor: NodeBlock(label="Composition")
```

### 5.4 SKILL: compose_query

```markdown
# SKILL: compose_query

## Purpose
Combine multiple branches into a complete query with boolean semantics.

## Input
- **anchor:** ValidNodeLabel (the result type)
- **branches:** List[Branch] (all must terminate at anchor)
- **operator:** `AND | OR`

## Output
- **query:** ComplexQueryInput ready for Cypher generation

## Validation
- All branches must terminate at the same anchor
- Branch count should be reasonable (avoid explosion)

## Semantics
- AND: Results must satisfy ALL branches (intersection)
- OR: Results may satisfy ANY branch (union)

## Example
Query for "my compositions about school with happy connections":
- anchor: Composition
- branches: [topic_branch, emotion_branch]
- operator: AND
```

---

## 6. Programmatic Generation Approach

### Generator Functions Needed

```python
def generate_node_type_skills(topology: dict, properties: dict) -> list[str]:
    """Generate markdown for all NodeType blocks."""
    pass

def generate_edge_type_skills(topology: dict) -> list[str]:
    """Generate markdown for all EdgeType blocks."""
    pass

def generate_segment_skills(topology: dict) -> list[str]:
    """Generate markdown for all valid segment SKILLs."""
    for source, targets in topology.items():
        for target, edges in targets.items():
            for edge in edges:
                yield generate_segment_skill_md(source, edge, target)

def generate_composition_skills() -> list[str]:
    """Generate markdown for composition SKILLs (static templates)."""
    pass
```

### Output Structure

```
skill-blocks/
  generated/
    nodes/
      Composition.md
      Creator.md
      Idea.md
      ...
    edges/
      REFLECT.md
      BELONG_TO.md
      CREATE.md
      ...
    segments/
      Creator.CREATE.Composition.md
      Composition.REFLECT.Idea.md
      ...
  templates/
    decorate_node_with_properties.md
    decorate_node_with_vector_search.md
    compose_branch.md
    compose_query.md
```

---

## 7. Integration with Agent Workflow

### How Agents Use These Skills

1. **Receive natural language query**
2. **Load relevant skill templates** (based on detected intents)
3. **Use topology tools** to validate paths
4. **Compose blocks** following skill instructions
5. **Output ComplexQueryInput** for Cypher generation

### DSPy Integration Point

```python
class QueryComposerSignature(dspy.Signature):
    """Compose a graph query from natural language."""

    natural_language_query: str = dspy.InputField()
    available_skills: list[str] = dspy.InputField(desc="Markdown skill templates")
    topology_context: str = dspy.InputField(desc="Graph topology summary")

    query_plan: str = dspy.OutputField(desc="Step-by-step plan using skills")
    composed_query: dict = dspy.OutputField(desc="ComplexQueryInput as JSON")
```

---

## Open Questions (To Resolve After Context Gathering)

1. Should segment skills be loaded on-demand or pre-loaded?
2. How to handle the 5 vector search per level limit in skills?
3. Should skills include example Cypher output?
4. How to version/update generated skills when topology changes?
