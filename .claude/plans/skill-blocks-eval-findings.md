# Findings: Skill-Blocks Pattern Evaluation

## Overview
Evaluating skill-blocks design for nl_to_cypher system.

---

## Phase 1 Findings: Design Principles

### Core Principles from README.md ✅
1. **Topology is the contract**: Only traversals in `GRAPH_TOPOLOGY` are allowed; if `(source,target)` pair isn't in topology, generation must refuse/fallback.
2. **Validation at smallest composable unit**:
   - Traversal unit = `EdgeSegment` (the "LEGO connector")
   - Node decorations validated independently (property constraints, vector search specs)
   - Composition rules enforce connectability
3. **Agents assemble; deterministic tools validate**: Agent plans paths/branches; helper functions enforce constraints.

### Mapping to Runtime Model ✅
- `NodeBlock` → `ComplexQueryInput.Branch.Level`
- `EdgeSegment` → `Level.path`
- Property constraints → `Level.nodeConditions`
- Vector search → `Level.vectorSearches`

### Helper Functions Status ✅ (ALL EXIST)
| Function | Location | Status |
|----------|----------|--------|
| `get_valid_next_labels_from_topology` | `model.py:127` | ✅ Found |
| `get_valid_edges_for_transition` | `model.py:148` | ✅ Found |
| `get_property_keys_for_label` | `model.py:170` | ✅ Found |
| `validate_property_value_types` | `utils.py:182` | ✅ Found |
| `build_topology_context_for_llm` | `utils.py:348` | ✅ Found |

---

## Phase 2 Findings: Graph Topology

### Node Labels (ValidNodeLabel)
`Creator`, `Composition`, `Emotion`, `Idea`, `Matching`, `Motive`, `Narrative`, `Structure`, `Status`, `SelfPresenting`

### Edge Types (RelationshipType)
`CREATE`, `REFLECT`, `SHOW`, `HAVE`, `BELONG_TO`, `HAPPEN_AFTER` (+ others not in topology)

### Valid Transitions (GRAPH_TOPOLOGY) - Bidirectional
| Source | Edge | Target |
|--------|------|--------|
| Creator | CREATE | Composition |
| Creator | HAVE | Status |
| Creator | CREATE | Matching |
| Status | SHOW | Emotion |
| Status | REFLECT | Motive |
| Status | REFLECT | SelfPresenting |
| Composition | REFLECT | Status |
| Composition | HAPPEN_AFTER | Composition |
| Composition | REFLECT | Narrative |
| Composition | REFLECT | Idea |
| Composition | REFLECT | Structure |
| Composition | BELONG_TO | Matching |

### Vector Search Labels
`Emotion`, `Idea`, `Motive`, `Narrative`, `Structure`, `SelfPresenting`

### Property Keys by Label
| Label | Property Keys |
|-------|---------------|
| Creator | `user_id` |
| Composition | `mongodb_id`, `created_at`, `creator_id` |
| Status | `mongodb_id`, `created_at` |
| Matching | `mongodb_id`, `created_at`, `task_id` |
| Emotion | `created_at` |
| Idea | `created_at` |
| Motive | `created_at` |
| Narrative | `created_at` |
| Structure | `mongodb_id`, `created_at` |
| SelfPresenting | `created_at` |

---

## Phase 3 Findings: Pattern Validation

### Valid Patterns ✅
All patterns in `patterns.md` are topologically valid:
- Single-node filters with properties
- `Creator - Matching`, `Creator - Status`
- `Idea|Narrative|Structure - Composition`
- `Emotion|Motive|SelfPresenting - Status - Composition`
- Multi-party patterns involving `Matching`

### Invalid Patterns ❌
**None** - All listed path segments follow allowed transitions.

### Missing Patterns 🔴
| Missing Pattern | Chinese Description | Topology Basis |
|-----------------|---------------------|----------------|
| `Creator - Composition` | 用户到帖子 (直接) | Creator <-> Composition (CREATE) |
| `Composition - Status` | 内容到状态 | Composition <-> Status (REFLECT) |
| `Composition - Status - (Emotion\|Motive\|SelfPresenting)` | 内容到特质 | Via Status |
| `Creator - Composition - (Idea\|Narrative\|Structure)` | 用户到议题 | Via Composition |
| `Creator - Status - (Emotion\|Motive\|SelfPresenting)` | 用户到形象/情绪/动机 | Via Status |
| `Creator - Composition - Matching` | 用户到匹配 (via posts) | Via Composition |
| `Composition - Composition` | 时序链 (HAPPEN_AFTER) | HAPPEN_AFTER edge |

---

## Phase 4 Findings: Recommendations

### Summary Assessment

| Aspect | Status | Notes |
|--------|--------|-------|
| Design Principles | ✅ Solid | Well-documented, topology-first approach |
| Helper Functions | ✅ All Exist | 5/5 functions implemented as documented |
| Runtime Model Mapping | ✅ Accurate | NodeBlock→Level, EdgeSegment→Path |
| Patterns Validity | ✅ All Valid | No topological errors in patterns.md |
| Pattern Completeness | ⚠️ Incomplete | 7 missing patterns identified |
| Edge Case Coverage | ⚠️ Sparse | Needs more documentation |

---

### 1. Missing Patterns to Add

**Priority 1 - Core Single-Party Patterns:**
```yaml
# 用户到帖子 (直接)
Pattern: Creator - Composition
Use case: "Show all posts by user X"
Segments: [Creator] --CREATE--> [Composition]

# 内容到状态
Pattern: Composition - Status
Use case: "Get the status associated with this post"
Segments: [Composition] --REFLECT--> [Status]
```

**Priority 2 - Extended Single-Party Patterns:**
```yaml
# 用户到议题 (via 帖子)
Pattern: Creator - Composition - (Idea|Narrative|Structure)
Use case: "What topics has user X posted about?"
Segments: [Creator] --CREATE--> [Composition] --REFLECT--> [Idea|Narrative|Structure]

# 用户到形象/情绪/动机 (via 状态)
Pattern: Creator - Status - (Emotion|Motive|SelfPresenting)
Use case: "What emotions has user X expressed?"
Segments: [Creator] --HAVE--> [Status] --SHOW/REFLECT--> [Emotion|Motive|SelfPresenting]

# 内容到特质 (via 状态)
Pattern: Composition - Status - (Emotion|Motive|SelfPresenting)
Use case: "What emotion does this post reflect?"
Segments: [Composition] --REFLECT--> [Status] --SHOW/REFLECT--> [Emotion|Motive|SelfPresenting]
```

**Priority 3 - Temporal Patterns:**
```yaml
# 时序链 (HAPPEN_AFTER)
Pattern: Composition - Composition
Use case: "Posts that happened after post X" / "Post sequence"
Segments: [Composition] --HAPPEN_AFTER--> [Composition]
```

---

### 2. Structural Improvements to patterns.md

**Current Organization Issues:**
1. Mixed Chinese/English labels inconsistent
2. Path filter vs Property filter distinction unclear
3. Multi-party section lacks systematic coverage

**Recommended Structure:**
```markdown
# Patterns Catalog

## 1. Single-Node Patterns (Property Filters Only)
   - Composition by creator_id
   - Composition by created_at
   - Matching by task_id

## 2. Single-Party Patterns (1 hop)
   - Creator → Composition (CREATE)
   - Creator → Status (HAVE)
   - Creator → Matching (CREATE)
   - Composition → Status (REFLECT)
   - Composition → (Idea|Narrative|Structure) (REFLECT)
   - Status → (Emotion|Motive|SelfPresenting) (SHOW/REFLECT)

## 3. Single-Party Patterns (2+ hops)
   - Creator → Composition → (Idea|Narrative|Structure)
   - Creator → Status → (Emotion|Motive|SelfPresenting)
   - Composition → Status → (Emotion|Motive|SelfPresenting)

## 4. Multi-Party Patterns (via Matching)
   - Existence: Composition ↔ Matching
   - Other-party Composition: ... → Matching → Composition (creator_id ≠ X)
   - Other-party Creator: ... → Matching → Creator (user_id ≠ X)
   - Chained Matching: ... → Creator → Matching (other matchings)

## 5. Temporal Patterns
   - Composition → Composition (HAPPEN_AFTER)
```

---

### 3. Edge Cases to Document

| Edge Case | Risk | Mitigation |
|-----------|------|------------|
| **Anchor ambiguity** | User intent unclear (Creator vs Composition vs Matching as result) | Default to Composition; add clarification prompt |
| **Direction confusion** | Traversal direction matters for some edges | Always store direction on segment; helpers check both |
| **AND/OR nesting** | "(A and B) or C" needs explicit grouping | Current model uses flat branches; document limitation |
| **Hop limit** | `maxHops - minHops < 2` enforced | Keep segments fixed-hop (1); document variable-length limitations |
| **Vector search limit** | Max 5 per level | Split across branches/levels if exceeded |
| **Self-referential Composition** | HAPPEN_AFTER creates cycles | Limit depth; add cycle detection |
| **Empty results** | Valid query, no matches | Return empty gracefully; don't treat as error |

---

### 4. Evaluation Metrics to Add

The README mentions evaluation checklist but patterns.md should include test queries:

```markdown
## Test Queries (Regression Suite)

| Query | Expected Pattern | Valid? |
|-------|------------------|--------|
| "我的帖子" | Composition (creator_id = X) | ✅ |
| "我关于学校的帖子" | Idea(vector:"学校") → Composition (creator_id = X) | ✅ |
| "我有匹配的帖子" | Composition → Matching | ✅ |
| "匹配对象的帖子" | Composition → Matching → Composition (creator_id ≠ X) | ✅ |
| "匹配对象最近情绪" | ... → Creator → Status → Emotion | ✅ |
```

---

### 5. Actionable Next Steps

1. **Immediate**: Update `patterns.md` with 7 missing patterns
2. **Short-term**: Restructure patterns.md with recommended organization
3. **Medium-term**: Add edge case documentation section
4. **Ongoing**: Build regression test suite for pattern validation

---

---

## Phase 5 Findings: Multi-Party Deep Analysis

### The Core Problem: Logical vs Technical Anchor Mismatch

**Current Branch Intersection Mechanism** (`nodes_multiple_branch_cypher.py:90`):
- Branches are intersected by **terminal node ID intersection** (AND semantics)
- `target_node_name = node_names[-1]` per branch
- IDs collected via `elementId(item.<target_node>)`
- Final result = `[id IN branch_0_ids WHERE id IN branch_1_ids AND ...]`

**Critical Limitation**:
- **No explicit `return_anchor` field** in `ComplexQueryInput`
- The join point IS the return point (hard-wired to last level)
- **Cannot "return X but join on Y"**
- Different terminal labels across branches = **empty results (silent failure)**

---

### Case-by-Case Multi-Party Analysis (patterns.md:29-43)

#### Scenario 1: 某用户的发帖，且该帖子有匹配 (lines 31-33)

```
Branch 1: Composition (creator_id: x) → Matching
Branch 2: Composition (creator_id: ≠x) → Matching
```

| Aspect | Value |
|--------|-------|
| **User Intent** | Find user X's posts that have a match |
| **Logical Anchor** | `Composition` (user X's post) |
| **Technical Anchor** | `Matching` (join point for both parties) |
| **Mismatch?** | ✅ YES - User wants posts, system joins on Matching |
| **Edge Case** | Result returns `Matching` IDs, not `Composition` IDs |

**Problem**: To return `Composition`, need post-processing to extract from branch 1.

---

#### Scenario 2: 发帖+内容相关+匹配对象用户有负面情绪帖子 (lines 34-36)

```
Branch 1: Idea(vector) → Composition (creator_id: x) → Matching
Branch 2: Emotion(vector) → Status → Composition (creator_id: ≠x) → Matching
```

| Aspect | Value |
|--------|-------|
| **User Intent** | X's posts (topic-related) where matched USER has negative-emotion posts |
| **Logical Anchor** | `Composition` (user X's post) |
| **Technical Anchor** | `Matching` (as written) OR `Creator` (if intent is "user has ANY negative post") |
| **Mismatch?** | ✅ YES - Plus semantic ambiguity |
| **Edge Case** | Pattern joins on Matching, meaning the negative-emotion post IS the matched post, not just "matched user has some negative post somewhere" |

**Problem**: The pattern as written is MORE restrictive than the user intent. User says "对象用户有负面情绪帖子" (matched user HAS negative posts), but pattern requires the specific matched post to show negative emotion.

**Correct Pattern for User Intent**:
```
Branch 1: Idea(vector) → Composition (creator_id: x) → Matching → Creator (user_id: ≠x)
Branch 2: Emotion(vector) → Status → Composition → Creator (user_id: ≠x)
Join on: Creator
```

But this STILL can't express "same creator" across branches (no variable binding).

---

#### Scenario 3: 发帖+内容相关+匹配的对象帖子展现负面情绪 (lines 37-39)

```
Branch 1: Idea(vector) → Composition (creator_id: x) → Matching → Composition (creator_id: ≠x)
Branch 2: Emotion(vector) → Status → Composition (creator_id: ≠x)
```

| Aspect | Value |
|--------|-------|
| **User Intent** | X's posts where the MATCHED POST shows negative emotion |
| **Logical Anchor** | `Composition` (user X's post) |
| **Technical Anchor** | `Composition` (the OTHER party's post) |
| **Mismatch?** | ✅ YES - Same label, different role |
| **Edge Case** | Join happens on counterpart's Composition, not X's Composition |

**Problem**: Result set is keyed by OTHER party's `Composition` IDs. To get X's posts, need to traverse back from the Matching.

---

#### Scenario 4: 发帖+内容相关+匹配对象用户有其他匹配 (lines 40-42)

```
Branch 1: Idea(vector) → Composition (creator_id: x) → Matching → Creator (user_id: ≠x)
Branch 2: Matching → Creator (user_id: ≠x)
```

| Aspect | Value |
|--------|-------|
| **User Intent** | X's posts where matched user has OTHER matchings |
| **Logical Anchor** | `Composition` (user X's post) |
| **Technical Anchor** | `Creator` (matched user) |
| **Mismatch?** | ✅ YES |
| **Edge Case** | Branch 2 only proves "user has A matching", doesn't enforce "DIFFERENT matching" |

**Problem**: Cannot express "a different Matching instance than branch 1" - no cross-branch distinctness constraint. Branch 2 will always be satisfied if branch 1 is satisfied (same Matching counts).

---

### Summary: Anchor Placement Problem

| Scenario | Logical Anchor | Technical Anchor | Can Express? | Returns What User Wants? |
|----------|---------------|------------------|--------------|--------------------------|
| 1 | Composition (mine) | Matching | ✅ | ❌ Returns Matching |
| 2 | Composition (mine) | Matching or Creator | ⚠️ Semantic gap | ❌ |
| 3 | Composition (mine) | Composition (theirs) | ✅ | ❌ Returns their Composition |
| 4 | Composition (mine) | Creator | ⚠️ No distinctness | ❌ |

---

### Fundamental Limitations in ComplexQueryInput

1. **No `return_anchor` field**: Join point = Return point (hard-coded)
2. **No cross-branch variable binding**: Cannot say "same Creator as reached via Matching"
3. **No cross-branch distinctness**: Cannot say "different Matching instance"
4. **Silent failure on label mismatch**: Different terminal labels → empty result, no error

---

### Recommendations for Multi-Party Patterns

**Short-term (Documentation)**:
- Document that multi-party queries return the TECHNICAL anchor, not the logical anchor
- Add post-processing guidance for extracting the logical anchor from results

**Medium-term (Model Enhancement)**:
- Add `return_anchor: Optional[ValidNodeLabel]` to `ComplexQueryInput`
- Add `return_level_index: Optional[int]` per branch to specify which level to return

**Long-term (Fundamental)**:
- Add cross-branch variable binding capability
- Add distinctness constraints for "other" semantics

---

### 6. Overall Verdict

**The skill-blocks design is sound for single-party queries.** The core principles are well-thought-out, all helper functions exist in code, and the patterns listed are topologically valid.

**For multi-party queries, there are fundamental limitations**:
1. The logical anchor (what user wants) often differs from the technical anchor (where branches must join)
2. The current `ComplexQueryInput` model cannot express this distinction
3. Patterns in `patterns.md` are topologically valid but may not return what users expect

The product manager and Neo4j specialist have created a solid foundation. However, multi-party scenarios require either:
- Model enhancements to support separate return anchors
- Clear documentation that results need post-processing
- User-facing clarification about what will be returned
