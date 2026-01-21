# Graph Pattern Analysis & NL-to-Cypher Optimization Plan

## Context

The current NL-to-Cypher system converts natural language queries into Neo4j Cypher queries for social network data traversal. The project leader has identified that the current approach is:
- **Flexible** but **LLM-dependent** and **slow**

### User's Initial Graph Patterns
1. A specific user's specific state's specific node (单用户单状态单节点)
2. Other (non-specific) user's node connected through connection node M to A (他用户通过连接节点到A)

---

# ANALYSIS RESULTS

## 1. Graph Schema & Topology

### Valid Node Labels (10 total)
| Label | Type | Vector Search | Description |
|-------|------|---------------|-------------|
| Creator | Hub | No | User/author node |
| Composition | Hub | No | Post/content node |
| Matching | Hub | No | Connection between users |
| Status | Intermediate | No | User state at a point in time |
| Emotion | Leaf | Yes | Emotional state (vector searchable) |
| Idea | Leaf | Yes | Topic/concept (vector searchable) |
| Motive | Leaf | Yes | Motivation (vector searchable) |
| Narrative | Leaf | Yes | Story structure (vector searchable) |
| Structure | Leaf | Yes | Content structure (vector searchable) |
| SelfPresenting | Leaf | Yes | Self-presentation (vector searchable) |

### Graph Topology (Allowed Edges)
```
Creator -[CREATE]-> Composition
Creator -[HAVE]-> Status
Creator -[CREATE]-> Matching
Status -[SHOW]-> Emotion
Status -[REFLECT]-> Motive
Status -[REFLECT]-> SelfPresenting
Composition -[REFLECT]-> Status
Composition -[HAPPEN_AFTER]-> Composition
Composition -[REFLECT]-> Narrative
Composition -[REFLECT]-> Idea
Composition -[REFLECT]-> Structure
Composition -[BELONG_TO]-> Matching
```

### Anchor Types (Return Targets)
- **Composition**: User posts/content
- **Matching**: Connections between users
- **Creator**: User profiles

### Hub/Connection Nodes
- **Matching**: Primary connection node linking two users' compositions

---

## 2. Common Graph Query Patterns

### Pattern A: Single-User Content Query (单用户单状态单节点)
**Example**: "Find my posts about health"
```
Creator(user_id=X) -[CREATE]-> Composition -[REFLECT]-> Idea(semantic: "health")
```
**Complexity**: Simple
**LLM Dependency**: Semantic search term extraction only

### Pattern B: Single-User Emotion Query
**Example**: "Find my posts where I felt happy"
```
Creator(user_id=X) -[HAVE]-> Status -[SHOW]-> Emotion(semantic: "happy")
                   -[CREATE]-> Composition -[REFLECT]-> Status
```
**Complexity**: Medium (2-hop)
**LLM Dependency**: Emotion interpretation

### Pattern C: Two-Party Connection Query (他用户通过连接节点到A)
**Example**: "Find my posts matched with others who felt happy"
```
Emotion(semantic: "happy") <-[SHOW]- Status(other) <-[HAVE]- Creator(NOT user_id=X)
                                                   -[CREATE]-> Composition(other)
                                                              -[BELONG_TO]-> Matching
                                                                            <-[BELONG_TO]- Composition(mine)
```
**Complexity**: Complex (multi-hop, multi-party)
**LLM Dependency**: Full decomposition required

### Pattern D: Connection Existence Query
**Example**: "Find my posts that have matching records"
```
Matching -[BELONG_TO]-> Composition(creator_id=X)
```
**Complexity**: Simple
**LLM Dependency**: Minimal (just anchor identification)

### Pattern E: Temporal Filter Query
**Example**: "Find my posts from last week"
```
Creator(user_id=X) -[CREATE]-> Composition(created_at >= 7_days_ago)
```
**Complexity**: Simple
**LLM Dependency**: Time parsing only

---

## 3. Performance Bottleneck Analysis

### Current Pipeline Timing
```
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage                    │ Type          │ Time      │ LLM Calls       │
├─────────────────────────────────────────────────────────────────────────┤
│ 1. Decomposition         │ LLM (DSPy)    │ ~400ms    │ 1               │
│ 2. Branch Planning       │ LLM (DSPy)    │ ~400ms    │ 1 per branch    │
│ 3. Path Validation       │ Deterministic │ ~1ms      │ 0               │
│ 4. Level Building        │ Mixed         │ ~800ms    │ 1-4 (parallel)  │
│    - Phase 1: Schema Gen │ Deterministic │ ~1ms      │ 0               │
│    - Phase 2: Selection  │ LLM/Determ    │ 0-400ms   │ 0-1             │
│    - Phase 2.5: Narrow   │ Deterministic │ ~1ms      │ 0               │
│    - Phase 3: Extraction │ LLM (parallel)│ ~400ms    │ 0-3 (parallel)  │
│    - Phase 4: Assembly   │ Deterministic │ ~1ms      │ 0               │
│ 5. Cypher Generation     │ Deterministic │ ~10ms     │ 0               │
└─────────────────────────────────────────────────────────────────────────┘

TOTAL: 1.5-2.5 seconds (4-7 LLM calls per query)
```

### Bottleneck Ranking
1. **Decomposition LLM** (~400ms) - Always required
2. **Branch Planning LLM** (~400ms per branch) - Always required
3. **Phase 3 Extraction** (~400ms) - Parallelized but still slow
4. **Phase 2 Selection** (~400ms) - Sometimes deterministic

### Key Insight: Deterministic Shortcuts Already Exist
The system already has deterministic paths:
- Phase 1: Always deterministic (topology lookup)
- Phase 2: Deterministic when `level_label` is known AND single edge option
- Phase 2.5: Always deterministic (schema narrowing)
- Phase 4: Always deterministic (assembly)

---

## 4. Optimization Strategies

### Strategy 1: Pattern-Based Query Classification (HIGH IMPACT)
**Description**: Classify queries into known patterns before LLM decomposition

**Implementation**:
```python
PATTERN_TEMPLATES = {
    "single_user_content": {
        "regex": r"(my|我的).*(posts?|compositions?|帖子)",
        "anchor": "Composition",
        "branches": ["Creator(user_id=X) -> Composition"]
    },
    "single_user_emotion": {
        "regex": r"(felt|感到|情绪).*(happy|sad|angry|开心|难过)",
        "anchor": "Composition",
        "branches": ["Emotion -> Status -> Composition"]
    },
    # ... more patterns
}
```

**Latency Reduction**: 400-800ms (skip decomposition for ~60% of queries)
**Complexity**: Medium
**Trade-off**: May miss edge cases; needs fallback to LLM

### Strategy 2: Template-Based Cypher Generation (HIGH IMPACT)
**Description**: Pre-defined Cypher templates for common patterns

**Implementation**:
```python
CYPHER_TEMPLATES = {
    "user_content_with_idea": """
        MATCH (c:Creator {user_id: $user_id})-[:CREATE]->(comp:Composition)
        MATCH (comp)-[:REFLECT]->(i:Idea)
        WHERE i.description_embedding <-> $query_embedding < $threshold
        RETURN comp
        LIMIT $limit
    """,
    # ... more templates
}
```

**Latency Reduction**: 800-1500ms (skip entire level building)
**Complexity**: Low
**Trade-off**: Less flexible; requires template maintenance

### Strategy 3: Hybrid Fast/Slow Path (RECOMMENDED)
**Description**: Route simple queries to fast deterministic path, complex to LLM

```
Query → Classifier → ┬─ Simple Pattern → Template Cypher (~50ms)
                     └─ Complex Query  → Full LLM Pipeline (~2000ms)
```

**Implementation**:
1. Rule-based classifier (regex + keyword matching)
2. Confidence threshold for routing
3. Fallback to LLM when uncertain

**Latency Reduction**:
- Simple queries: 1500-2000ms saved
- Complex queries: No change
- Average: ~1000ms saved (assuming 60% simple)

**Complexity**: Medium
**Trade-off**: Best balance of speed and flexibility

### Strategy 4: Caching Layer (MEDIUM IMPACT)
**Description**: Cache query results and intermediate computations

**Levels**:
1. **Query-level cache**: Same query → same Cypher (Redis, TTL: 5min)
2. **Embedding cache**: Same text → same embedding (persistent)
3. **Schema cache**: Already implemented via `@lru_cache`

**Latency Reduction**: 100% for cache hits
**Complexity**: Low
**Trade-off**: Cache invalidation complexity

### Strategy 5: Smaller Models for Simple Tasks (LOW IMPACT)
**Description**: Use faster/cheaper models for simpler extraction tasks

**Current**: All DSPy modules use same model
**Proposed**:
- Decomposition: Full model (complex reasoning)
- Time extraction: Small model (simple parsing)
- Property extraction: Small model (lookup-like)

**Latency Reduction**: ~100-200ms per small model call
**Complexity**: Low
**Trade-off**: Potential accuracy loss

---

## 5. Recommendations

### Priority 1: Implement Hybrid Fast/Slow Path
**Effort**: 2-3 days
**Impact**: ~1000ms average latency reduction

Steps:
1. Define 5-10 common query patterns with regex/keyword rules
2. Create Cypher templates for each pattern
3. Build classifier with confidence scoring
4. Add fallback to full LLM pipeline

### Priority 2: Add Query-Level Caching
**Effort**: 1 day
**Impact**: 100% latency reduction for repeated queries

Steps:
1. Hash query + user_id as cache key
2. Store generated Cypher in Redis
3. Set appropriate TTL (5-15 minutes)

### Priority 3: Optimize Phase 2 Selection
**Effort**: 0.5 days
**Impact**: ~400ms for single-option cases

The code already has deterministic shortcuts, but they may not be fully utilized. Audit and ensure:
- `level_label` is always passed when known
- Single-edge cases skip LLM

### Not Recommended (Yet)
- **Smaller models**: Marginal gain, accuracy risk
- **Full template system**: Too rigid for current use cases

---

## 6. Summary for Project Leader

### Current State
- **Latency**: 1.5-2.5 seconds per query
- **LLM Calls**: 4-7 per query
- **Flexibility**: High (handles complex multi-hop queries)

### Proposed Improvements
| Strategy | Latency Saved | Effort | Recommended |
|----------|---------------|--------|-------------|
| Hybrid Fast/Slow | ~1000ms avg | Medium | **Yes** |
| Query Caching | 100% (hits) | Low | **Yes** |
| Phase 2 Audit | ~400ms | Low | **Yes** |
| Smaller Models | ~200ms | Low | Later |
| Full Templates | ~1500ms | High | No |

### Expected Outcome
- **Simple queries**: ~50-200ms (from ~2000ms)
- **Complex queries**: ~1500-2000ms (unchanged)
- **Average**: ~500-800ms (from ~2000ms)

### Key Insight
The system is already well-architected with deterministic shortcuts. The main opportunity is **routing simple queries around the LLM pipeline entirely**, not optimizing the LLM calls themselves.
