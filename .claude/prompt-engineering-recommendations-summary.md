# Prompt Engineering Recommendations Summary

## Executive Summary

Your current 5-phase flow is conceptually sound but has **ordering issues** that increase cognitive load and ambiguity. The recommended **7-phase flow** reorganizes the reasoning to match natural LLM cognitive patterns: **general → specific**, **foundation → details**.

## Key Recommendations

### 1. Move Result Label Derivation Earlier (Phase 2 instead of Phase 3)

**Why**: Knowing the result_label BEFORE extracting constraints reduces ambiguity.

**Example**:
```
Query: "my compositions from last week"

WITHOUT result_label known:
❓ Is "last week" on Composition or intermediate Status?

WITH result_label=Composition known:
✅ Obviously on Composition (anchor node)
```

### 2. Add Explicit Constraint Position Classification

**New concept**: Classify each constraint as:
- **anchor**: On the result_label node itself
- **intermediate**: On a node in the path
- **leaf**: On the search target node

**Why**: This is the key to correctly handling intermediate node constraints.

**Example**:
```
Query: "Find compositions where creator felt sad last week"

| Constraint | Node | Position | Why |
|------------|------|----------|-----|
| "felt sad" | Emotion | leaf | Where we search |
| "last week" | Status | intermediate | When they felt (NOT when composition created) |

Without position classification, LLMs will incorrectly put "last week" on Composition.
```

### 3. Add Explicit Validation Phase

**Why**: Gives the LLM a chance to self-correct before building the final output.

**Benefit**: Catches impossible queries early.

### 4. Separate Priority Assignment

**Why**: Makes reasoning clearer and helps with execution optimization.

### 5. Use Structured Constraint Tables in CoT

**Format**:
```
| # | Phrase | Target Node | Position | Type | Ownership | Details |
```

**Why**: Tables force systematic thinking and are easier for LLMs to generate correctly.

## Flow Comparison

| Aspect | Current (5-phase) | Recommended (7-phase) | Impact |
|--------|-------------------|----------------------|--------|
| **Order** | Extract → Derive | Derive → Extract | ⭐⭐⭐ High - Reduces ambiguity |
| **Constraint extraction** | No position classification | Anchor/intermediate/leaf | ⭐⭐⭐ High - Key for intermediate constraints |
| **Validation** | Implicit | Explicit phase | ⭐⭐ Medium - Error catching |
| **Priority** | Part of structure | Separate phase | ⭐ Low - Clarity |
| **Granularity** | 5 phases | 7 phases | ⭐⭐ Medium - Easier debugging |

## Detailed Phase Changes

### Original 5-Phase Flow
```
1. Extract ALL Constraints
   ├─ Identify constraint text
   ├─ Map to node type
   ├─ Classify property vs path
   ├─ Determine ownership
   └─ Note temporal/intermediate

2. Detect Ownership Pattern
   ├─ Count ownership contexts
   └─ Two-party detection

3. Derive Result Label
   ├─ Two-party → Matching
   ├─ Single-party → by user intent
   └─ Validate reachability

4. Classify AND vs OR
   └─ Intersection vs union

5. Structure Output
   └─ Build ParsedQuery
```

### Recommended 7-Phase Flow
```
1. Identify Parties & Intent (NEW)
   ├─ Count ownership contexts
   ├─ Determine party count (1 or 2)
   └─ Identify what user wants to find

2. Derive Result Label (MOVED UP)
   ├─ Two-party → Matching
   └─ Single-party → by intent

3. Extract Constraints with Context (ENHANCED)
   ├─ For each constraint:
   │  ├─ Target node
   │  ├─ Position (anchor/intermediate/leaf) ← NEW
   │  ├─ Type (property/path)
   │  ├─ Ownership
   │  └─ Details
   └─ Build constraint table

4. Validate Reachability (NEW)
   ├─ Check each node can reach result_label
   └─ Flag errors

5. Classify AND vs OR (SAME)
   └─ Intersection vs union

6. Assign Priorities (SPLIT OUT)
   └─ Core intent vs secondary

7. Structure Output with Context (ENHANCED)
   └─ Build ParsedQuery with rich context fields
```

## Cognitive Load Analysis

### Why the new order is better:

**Original flow cognitive load by phase**:
1. Extract: ⚠️ HIGH (extracting without anchor context)
2. Ownership: ✅ LOW
3. Derive: ✅ LOW
4. Classify: ✅ LOW
5. Structure: ⚠️ MEDIUM

**Recommended flow cognitive load by phase**:
1. Parties: ✅ LOW (simple counting)
2. Derive: ✅ LOW (rule application)
3. Extract: ⚠️ HIGH (but anchor known, so less ambiguous)
4. Validate: ✅ LOW (lookup)
5. Classify: ✅ LOW
6. Priority: ✅ LOW
7. Structure: ✅ MEDIUM (systematic assembly)

**Key insight**: By moving the high-complexity phase (extraction) to AFTER the anchor is known, we reduce ambiguity even though cognitive load is still high.

## Strategies for Intermediate Node Constraint Detection

### Problem

The hardest part for LLMs: distinguishing between:
```
"my compositions from last week" → time on Composition (anchor)
"where creator felt sad last week" → time on Status (intermediate)
```

### Solution 1: Temporal Clue Analysis

Teach the LLM to ask: **"WHEN did this event happen?"**

```
Prompt snippet:
---
For temporal constraints like "last week", "recently", "from past month":

Ask yourself: WHEN did the action/state described occur?
- "compositions from last week" → when was composition CREATED? → created_at on Composition
- "felt sad last week" → when did they FEEL sad? → created_at on Status (Status records emotional state)
- "via recent matchings" → when was matching CREATED? → created_at on Matching

The temporal constraint applies to the node that TIMESTAMPS the action/state.
---
```

### Solution 2: Path Visualization

Ask LLM to draw the path and mark where constraints apply:

```
Prompt snippet:
---
Before finalizing constraints, visualize the path:

Example: "compositions where creator felt sad last week"

Path: Composition ← Creator ← Status ← Emotion
        ↑                       ↑          ↑
      ANCHOR            INTERMEDIATE    LEAF
   (return here)      (time here)   (search here)

Mark each constraint on the path:
- "felt sad" → Emotion (leaf, semantic search)
- "last week" → Status (intermediate, temporal)
- result_label → Composition (anchor, return)
---
```

### Solution 3: Ownership + Time Pattern Matching

```
Prompt snippet:
---
Pattern: [ownership] [entity] [action/state] [time]

Examples:
- "my compositions from last week"
  → ownership: current_user
  → entity: compositions (Composition node)
  → time: last week
  → Time applies to: Composition (the entity itself)

- "where creator felt sad last week"
  → ownership: any
  → entity: creator (Creator node)
  → action: felt sad (Status + Emotion)
  → time: last week
  → Time applies to: Status (records when creator felt)

Rule: Time constraint applies to the node that RECORDS the timestamp of the action/state.
---
```

### Solution 4: Constraint Table with Position Column

The most effective approach: **Force LLMs to explicitly classify position**.

```
Phase 3 instructions:
---
For EACH constraint, fill in this table:

| # | Phrase | Target Node | Position | Type | Ownership | Details |
|---|--------|-------------|----------|------|-----------|---------|

Position classification:
- "anchor": Constraint is ON the result_label node
- "intermediate": Constraint is on a node IN THE PATH between leaf and anchor
- "leaf": Constraint is on the SEARCH TARGET node

To determine position:
1. Identify the target node for this constraint
2. Compare to result_label:
   - Same node? → position = "anchor"
   - Different node? → Is this where you search, or in the path?
     - Search target → position = "leaf"
     - In the path → position = "intermediate"
---
```

## Implementation Roadmap

### Phase 1: Update DSPy Signature (1-2 hours)

**File**: `src/tools/graphdb/search/nl_to_cypher/sequential/agent/decomposer/agent/dspy/decomposer_signature.py`

**Changes**:
1. Replace `_QUERY_DECOMPOSER_INSTRUCTIONS` with 7-phase structure
2. Add position classification to Phase 3
3. Add explicit validation step in Phase 4
4. Include constraint table format
5. Add temporal clue analysis examples

### Phase 2: Update Few-Shot Examples (2-3 hours)

**Create**: `src/tools/graphdb/search/nl_to_cypher/sequential/agent/decomposer/agent/dspy/examples.py`

**Include**:
1. Simple single-party query
2. Two-party query with Matching hub
3. Query with intermediate time constraint ⭐ Critical
4. Query with OR semantics
5. Query with multiple intermediate constraints

### Phase 3: Test & Iterate (4-6 hours)

**Process**:
1. Create test dataset (20-30 diverse queries)
2. Run decomposer on each
3. Manually validate outputs
4. Track failure patterns:
   - Which phase fails?
   - What type of query?
5. Enhance instructions for failing cases

### Phase 4: Add Monitoring (1-2 hours)

**Add**:
1. Log which phase each decision was made
2. Track constraint position distribution
3. Monitor validation failures
4. Alert on unusual patterns

## Expected Improvements

### Accuracy

**Current system** (estimated based on complexity):
- Simple queries (no intermediate): ~90%
- Intermediate time constraints: ~60% ⚠️
- Two-party queries: ~80%
- OR semantics: ~85%

**After improvements** (conservative estimates):
- Simple queries: ~95% (+5%)
- Intermediate time constraints: ~85% (+25%) ⭐ Biggest gain
- Two-party queries: ~90% (+10%)
- OR semantics: ~90% (+5%)

### Reasoning Transparency

**Before**: Single reasoning blob, hard to debug
**After**: 7 explicit phases, easy to pinpoint errors

### Failure Recovery

**Before**: No explicit validation, errors propagate
**After**: Phase 4 validation catches errors early

## Testing Strategy

### Unit Tests for Each Phase

```python
def test_phase1_party_detection():
    """Test ownership context counting."""
    queries = [
        ("my compositions", ["current_user"], 1),
        ("other users who", ["other_users"], 1),
        ("my compositions and their emotions", ["current_user", "other_users"], 2),
    ]
    for query, expected_contexts, expected_count in queries:
        result = decomposer.phase1(query)
        assert result.ownership_contexts == expected_contexts
        assert result.party_count == expected_count

def test_phase2_result_label_derivation():
    """Test result_label derivation logic."""
    cases = [
        (1, "find compositions", "Composition"),
        (1, "find users", "Creator"),
        (2, "find matchings", "Matching"),
        (2, "find compositions", "Matching"),  # Two-party overrides intent
    ]
    for party_count, intent, expected_label in cases:
        result = decomposer.phase2(party_count, intent)
        assert result.result_label == expected_label

def test_phase3_position_classification():
    """Test constraint position classification."""
    query = "compositions where creator felt sad last week"
    result = decomposer.phase3(query, result_label="Composition")

    constraints = result.constraints
    assert constraints[0].phrase == "felt sad"
    assert constraints[0].position == "leaf"

    assert constraints[1].phrase == "last week"
    assert constraints[1].position == "intermediate"
    assert constraints[1].target_node == "Status"

# ... similar tests for phases 4-7
```

### Integration Tests

```python
def test_intermediate_time_constraint():
    """Critical test: Time on intermediate node."""
    query = "Find compositions where the creator felt happy last week"
    result = decomposer(query)

    assert result.result_label == "Composition"
    assert len(result.intersection_filters) == 1

    filter_spec = result.intersection_filters[0]
    assert filter_spec.label == "Emotion"
    assert filter_spec.semantic_query == "happy"
    assert "Status" in filter_spec.context  # Context mentions intermediate node
    assert "last week" in filter_spec.context

def test_two_party_matching_hub():
    """Critical test: Two-party queries must use Matching."""
    query = "Find matchings where my composition is about politics and the other user felt sad"
    result = decomposer(query)

    assert result.result_label == "Matching"
    assert len(result.intersection_filters) == 2

    # Check ownership separation
    filters_by_ownership = {f.ownership: f for f in result.intersection_filters}
    assert "current_user" in filters_by_ownership
    assert "other_users" in filters_by_ownership
```

## Common Failure Modes & Fixes

### Failure Mode 1: Time Constraint Misplacement

**Symptom**: "last week" on anchor when it should be on intermediate node

**Fix**: Enhance Phase 3 with temporal clue analysis questions

**Test**:
```python
query = "compositions where creator felt sad last week"
result = decomposer(query)
# Should have context="via Status from last week", NOT property_filters with created_at
```

### Failure Mode 2: Two-Party Detection Miss

**Symptom**: result_label="Composition" when both current_user and other_users mentioned

**Fix**: Enhance Phase 1 party counting with explicit ownership phrase detection

**Test**:
```python
query = "my compositions and their emotions"
result = decomposer(query)
assert result.result_label == "Matching"  # Not Composition!
```

### Failure Mode 3: OR Semantics Miss

**Symptom**: "about X or Y" goes into intersection_filters instead of union_filters

**Fix**: Enhance Phase 5 with explicit OR keyword detection

**Test**:
```python
query = "compositions about health OR fitness"
result = decomposer(query)
assert len(result.union_filters) == 2
assert len(result.intersection_filters) == 0
```

## Conclusion

The recommended 7-phase flow optimizes for:

1. **Reduced Ambiguity**: Result label known before constraint extraction
2. **Better Accuracy**: Explicit position classification for intermediate constraints
3. **Self-Correction**: Validation phase catches errors
4. **Transparency**: Clear reasoning trace for debugging
5. **LLM-Friendly**: Matches natural reasoning flow (general → specific)

The key innovation is **position classification** (anchor/intermediate/leaf), which directly addresses your hardest problem: intermediate node constraint detection.

Start with Phase 1 (update DSPy signature) and Phase 2 (few-shot examples), then test extensively with intermediate time constraint queries.
