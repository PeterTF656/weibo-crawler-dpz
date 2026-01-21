# Chain-of-Thought Prompt Template for Query Decomposition

## Recommended Prompt Structure

### System Instructions (DSPy Signature Instructions)

```
You are a graph query decomposition specialist. Your job is to parse natural language
search queries into structured constraints for a Neo4j knowledge graph.

Follow this 7-phase reasoning process step-by-step. Show your work for each phase.

# Phase 1: Identify Parties & Intent

Count ownership contexts mentioned in the query:
- "my", "I", "current user" → current_user
- "other", "their", "matched user" → other_users
- No ownership specified → any

Count distinct ownership contexts:
- 1 context → Single-party query
- 2 contexts (both current_user AND other_users) → Two-party query

Identify user intent:
- What type of entity does the user want to find?
  • Posts/content → looking for Composition
  • User profiles → looking for Creator
  • Connections/sessions → looking for Matching

Format:
```
**Phase 1 Output:**
- Ownership contexts: [list]
- Party count: [1 or 2]
- User intent: [find compositions/users/matchings]
```

# Phase 2: Derive Result Label

Apply these rules in order:

Rule 1 (Two-Party Hub Rule):
IF party_count == 2:
    result_label = "Matching"
    REASON: Matching is the ONLY hub connecting current_user and other_users

Rule 2 (Single-Party Intent Mapping):
ELSE IF user_intent == "find compositions":
    result_label = "Composition"
ELSE IF user_intent == "find users":
    result_label = "Creator"
ELSE IF user_intent == "find matchings":
    result_label = "Matching"

Format:
```
**Phase 2 Output:**
- result_label: [Composition/Matching/Creator]
- Reasoning: [explain which rule was applied]
```

# Phase 3: Extract Constraints with Context

For EACH constraint phrase in the query, create a table entry:

Constraint Classification:
1. **Target Node**: Which node type? (Composition, Idea, Emotion, Status, etc.)
2. **Position**: Relative to result_label:
   - "anchor" = constraint ON the result_label itself
   - "intermediate" = constraint on a node IN THE PATH between leaf and result_label
   - "leaf" = constraint on the SEARCH TARGET node (where semantic search happens)
3. **Type**:
   - "property" = direct attribute (creator_id, created_at, etc.)
   - "path" = requires traversal + semantic search
4. **Ownership**: current_user / other_users / any
5. **Details**: What filter to apply

Format as table:
```
**Phase 3 Output:**

| # | Phrase | Target Node | Position | Type | Ownership | Details |
|---|--------|-------------|----------|------|-----------|---------|
| 1 | "my" | Composition | anchor | property | current_user | creator_id = $user_id |
| 2 | "about politics" | Idea | leaf | path | any | semantic: "politics" |
| 3 | "felt sad" | Emotion | leaf | path | other_users | semantic: "sad" |
| 4 | "last week" | Status | intermediate | property | other_users | created_at >= $last_week |
```

**Critical**: Distinguish between anchor/intermediate/leaf positions carefully!

Examples:
- "my compositions from last week" → "last week" is on Composition (ANCHOR)
- "where creator felt sad last week" → "last week" is on Status (INTERMEDIATE)

# Phase 4: Validate Reachability

For each constraint target node, verify it can reach result_label using the topology summary.

Topology reference:
- From Composition: Can reach Idea, Status, Narrative, Structure, Matching, Creator
  - Status → Emotion, Motive, SelfPresenting
- From Matching: Can reach Composition, Creator
- From Creator: Can reach Composition, Status, Matching

Format:
```
**Phase 4 Output:**

Validation checks:
- [Node1] → [result_label]: ✅ Reachable via [path]
- [Node2] → [result_label]: ✅ Reachable via [path]
- [Node3] → [result_label]: ❌ NOT reachable (ERROR)

Overall: [PASS/FAIL]
```

If FAIL: Stop and suggest corrections.

# Phase 5: Classify AND vs OR

Examine logical connectors in the query:

Rules:
- Explicit "OR" → union_filters
- Explicit "AND" → intersection_filters
- Implicit conjunction (commas, "also", "and") → intersection_filters
- Default → intersection_filters

Format:
```
**Phase 5 Output:**

Constraints for intersection_filters (AND):
- [list constraint #s]

Constraints for union_filters (OR):
- [list constraint #s]
```

# Phase 6: Assign Priorities

Order constraints by importance:

Rules:
- Core intent, must-have → priority = 0
- Secondary, nice-to-have → priority = 1, 2, 3...

Format:
```
**Phase 6 Output:**

| Constraint # | Priority | Reasoning |
|--------------|----------|-----------|
| 2 | 0 | Core intent: "about politics" |
| 3 | 1 | Secondary condition: "felt sad" |
| 4 | 1 | Refines #3, same priority |
```

# Phase 7: Structure Output with Context

Build the final ParsedQuery:

1. **property_filters**: Constraints with position="anchor" and type="property"

2. **intersection_filters**: Path constraints with AND semantics
   - For each: create FilterSpec with label, semantic_query, priority, ownership
   - **context field**: Write natural language description of intermediate constraints

   Context field examples:
   - Simple path: context = null or omit
   - With ownership: context = "via current user's composition"
   - With intermediate time: context = "via Creator's Status from last week"
   - Complex: context = "via other users' Composition in Matching from past week"

3. **union_filters**: Path constraints with OR semantics (same structure)

Format:
```
**Phase 7 Output:**

ParsedQuery(
    result_label="[Composition/Matching/Creator]",
    property_filters=[
        PropertyFilter(key="creator_id", op="eq", value="$user_id"),
        PropertyFilter(key="created_at", op="gte", value="$last_week"),
    ],
    intersection_filters=[
        FilterSpec(
            label="Idea",
            semantic_query="politics",
            priority=0,
            ownership="current_user",
            context="via current user's composition"
        ),
        FilterSpec(
            label="Emotion",
            semantic_query="sad",
            priority=1,
            ownership="other_users",
            context="via other users' Status from last week"
        ),
    ],
    union_filters=[],
)
```

# Summary

Always show your reasoning for EACH phase before proceeding to the next.
This helps catch errors early and makes your logic transparent.
```

### Example: Full Reasoning Trace

**Query**: "Find matchings where my composition is about politics and the other user felt sad last week"

**Phase 1 Output:**
- Ownership contexts: [current_user, other_users]
- Party count: 2
- User intent: find matchings

**Phase 2 Output:**
- result_label: Matching
- Reasoning: Party count = 2, so Two-Party Hub Rule applies. Matching is the only hub connecting both parties.

**Phase 3 Output:**

| # | Phrase | Target Node | Position | Type | Ownership | Details |
|---|--------|-------------|----------|------|-----------|---------|
| 1 | "my composition" | Composition | intermediate | property | current_user | creator_id = $user_id |
| 2 | "about politics" | Idea | leaf | path | current_user | semantic: "politics" |
| 3 | "other user felt sad" | Emotion | leaf | path | other_users | semantic: "sad" |
| 4 | "last week" | Status | intermediate | property | other_users | created_at >= $last_week |

**Phase 4 Output:**

Validation checks:
- Idea → Matching: ✅ Reachable via [Idea → Composition → Matching]
- Emotion → Matching: ✅ Reachable via [Emotion → Status → Creator → Matching]

Overall: PASS

**Phase 5 Output:**

Constraints for intersection_filters (AND):
- #2 (about politics)
- #3 (felt sad) + #4 (last week)

Constraints for union_filters (OR):
- None

**Phase 6 Output:**

| Constraint # | Priority | Reasoning |
|--------------|----------|-----------|
| 2 | 0 | Core intent for current_user side |
| 3 | 1 | Core intent for other_users side |

**Phase 7 Output:**

```python
ParsedQuery(
    result_label="Matching",
    property_filters=[],  # No anchor-level properties
    intersection_filters=[
        FilterSpec(
            label="Idea",
            semantic_query="politics",
            priority=0,
            ownership="current_user",
            context="via current user's composition"
        ),
        FilterSpec(
            label="Emotion",
            semantic_query="sad",
            priority=1,
            ownership="other_users",
            context="via other users' Status from last week"
        ),
    ],
    union_filters=[],
)
```

## Implementation Tips

### For DSPy Integration

1. **Use TypedPredictor or TypedChainOfThought**:
   ```python
   from dspy.functional import TypedChainOfThought

   predictor = TypedChainOfThought(QueryDecomposerSignature)
   ```

2. **Enforce structured reasoning**:
   - Include phase headers in the instructions
   - Use clear formatting (tables, numbered lists)
   - Request explicit "Phase N Output:" sections

3. **Add few-shot examples**:
   - Include 2-3 complete reasoning traces in the signature
   - Cover edge cases: two-party, intermediate constraints, OR semantics

### For Prompt Optimization

1. **Test with diverse queries**:
   - Simple: "my compositions about health"
   - Complex: "matchings where my composition about politics and their emotion was sad last week"
   - Edge: "users who felt happy OR excited in recent matchings"

2. **Monitor failure modes**:
   - Track where the LLM makes mistakes (likely Phase 3 position classification)
   - Add clarifying examples to that phase's instructions

3. **Iterative refinement**:
   - Start with the 7-phase structure
   - If LLM struggles with a phase, break it into sub-steps
   - If a phase is trivial, consider merging with adjacent phase

## Advanced: Intermediate Node Constraint Detection

This is the hardest part. Here are specific strategies:

### Strategy 1: Temporal Clue Analysis

Teach the LLM to ask: "WHEN did this event happen?"

```
Query: "compositions where creator felt sad last week"

Question: When did the creator feel sad?
Answer: Last week

Question: What node represents "when creator felt"?
Answer: Status (Status has created_at timestamp)

Conclusion: "last week" constraint is on Status (intermediate), NOT Composition
```

### Strategy 2: Ownership + Time Pattern

```
Pattern: [ownership] [action] [time]

Examples:
- "my compositions from last week" → time on Composition (anchor)
- "creator felt sad last week" → time on Status (when they felt)
- "via recent matchings" → time on Matching (intermediate)

Rule: The time constraint applies to the node that "owns" the timestamp of the action.
```

### Strategy 3: Path Visualization

Ask LLM to visualize the path:

```
Query: "compositions where creator felt sad last week"

Path: Composition ← Creator ← Status ← Emotion
        ↑                       ↑          ↑
      anchor            intermediate     leaf
      (return here)   (time constraint) (search here)

"last week" modifies "felt" → felt is recorded in Status → constraint on Status
```

## Recommended Next Steps

1. **Update your DSPy signature** with the new 7-phase instructions
2. **Add position classification** to Phase 3 constraint table
3. **Create few-shot examples** covering:
   - Two-party queries
   - Intermediate time constraints
   - OR semantics
4. **Test with your query dataset** and measure accuracy
5. **Iterate on failure cases** by enhancing specific phase instructions
