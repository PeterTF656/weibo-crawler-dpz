# Recommended 7-Phase LLM Reasoning Flow for Graph Query Decomposition

## Overview

This flow is optimized for LLM cognitive patterns, moving from high-level intent to detailed constraint extraction.

## The 7-Phase Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Identify Parties & Intent (Foundation Setting)        │
│                                                                 │
│ Questions:                                                      │
│ - How many ownership contexts? (1 vs 2 parties)                │
│ - What does user want to find? (posts/users/connections)       │
│                                                                 │
│ Output: Party count, high-level intent                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 2: Derive Result Label (Anchor Selection)                │
│                                                                 │
│ Logic:                                                          │
│ - Two parties → result_label = "Matching" (hub rule)           │
│ - Single party → derive from intent:                           │
│   • "find compositions" → "Composition"                        │
│   • "find users" → "Creator"                                   │
│   • "find matchings" → "Matching"                              │
│                                                                 │
│ Output: result_label (what to RETURN)                          │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 3: Extract Constraints with Context                      │
│                                                                 │
│ For EACH constraint phrase:                                    │
│ 1. Identify target node type (Idea/Emotion/Status/etc.)       │
│ 2. Classify position relative to result_label:                │
│    - anchor: constraint ON the result_label                   │
│    - intermediate: on a node IN THE PATH                       │
│    - leaf: on the SEARCH TARGET node                           │
│ 3. Determine ownership (current_user/other_users/any)         │
│ 4. Classify type (property vs path)                            │
│                                                                 │
│ Use constraint table format:                                   │
│ | Phrase | Node | Position | Type | Ownership | Details |     │
│                                                                 │
│ Output: Structured constraint table                            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 4: Validate Reachability                                 │
│                                                                 │
│ For each constraint node:                                      │
│ - Can it reach result_label?                                   │
│ - Check against topology summary                               │
│                                                                 │
│ If any constraint unreachable:                                 │
│ - Flag as invalid                                              │
│ - Suggest correction                                           │
│                                                                 │
│ Output: Validation result (pass/fail + corrections)            │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 5: Classify AND vs OR                                    │
│                                                                 │
│ Rules:                                                          │
│ - Explicit "or" → union_filters                               │
│ - "and" or implicit conjunction → intersection_filters        │
│ - Default: intersection_filters                                │
│                                                                 │
│ Output: Constraints sorted into intersection/union groups      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 6: Assign Priorities                                     │
│                                                                 │
│ Rules:                                                          │
│ - Core intent, must-have → priority = 0                       │
│ - Secondary conditions → priority = 1, 2, 3...                │
│                                                                 │
│ Helps execution engine with:                                   │
│ - Early stopping (empty high-priority result)                  │
│ - Query optimization                                            │
│                                                                 │
│ Output: Priority assignments                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ Phase 7: Structure Output with Context                         │
│                                                                 │
│ Build ParsedQuery:                                             │
│ - result_label                                                  │
│ - property_filters (anchor node attributes)                    │
│ - intersection_filters (AND semantics)                         │
│ - union_filters (OR semantics)                                 │
│                                                                 │
│ For each FilterSpec:                                           │
│ - label: WHERE to apply filter                                 │
│ - semantic_query: WHAT to search for                           │
│ - priority: execution order                                    │
│ - ownership: path guidance                                     │
│ - context: natural language description of intermediate        │
│           constraints and path logic                            │
│                                                                 │
│ Output: Complete ParsedQuery                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Why This Order?

### 1. Parties & Intent First
**Cognitive Load**: Low - just counting ownership mentions
**Benefit**: Sets mental context for everything else

### 2. Result Label Second
**Cognitive Load**: Low - simple rule application
**Benefit**: Disambiguates constraint extraction

Example:
```
Query: "compositions from last week"

Without result_label: Is "last week" on Composition or intermediate Status?
With result_label=Composition: Obviously on Composition (anchor)
```

### 3. Constraint Extraction Third
**Cognitive Load**: High - detailed parsing
**Benefit**: Result label already known helps position classification

Knowing result_label="Composition" helps LLM understand:
- "my" → property filter on Composition (anchor)
- "about politics" → path filter via Idea (leaf)
- "creator felt sad last week" → "last week" is on Status (intermediate), not Composition

### 4. Validation Fourth
**Cognitive Load**: Low - topology lookup
**Benefit**: Catch errors before building output

### 5-7. Classify, Prioritize, Structure
**Cognitive Load**: Low-Medium - rule application
**Benefit**: Systematic construction of output

## Key Differences from Original 5-Phase Flow

| Aspect | Original | Recommended | Why Changed |
|--------|----------|-------------|-------------|
| **Phase order** | Extract → Ownership → Derive | Parties → Derive → Extract | Knowing result_label helps extraction |
| **Constraint extraction** | All at once | After result_label | Less ambiguity |
| **Validation** | Implicit | Explicit phase | Self-correction opportunity |
| **Priority assignment** | Part of structure | Separate phase | Clearer reasoning |
| **Phases** | 5 | 7 | More granular, easier to debug |

## Constraint Position Classification (New)

This is the key enhancement in Phase 3:

```
Position types:
- anchor: Constraint ON the result_label node
- intermediate: Constraint on a node IN THE PATH (Status, Composition in path)
- leaf: Constraint on the SEARCH TARGET node (Idea, Emotion, etc.)

Example:
Query: "Find compositions where creator felt sad last week"
result_label: Composition

| Phrase | Node | Position | Why |
|--------|------|----------|-----|
| "creator felt sad" | Emotion | leaf | Search target |
| "last week" | Status | intermediate | Time is on Status, not Composition |

Path: Composition ← Creator ← Status ← Emotion
                     ↑                    ↑
                   anchor            leaf (filter here)
                              ↑
                         intermediate (time constraint here)
```

## Prompt Structure Recommendation

See the Chain-of-Thought prompt template in the implementation section below.

## Benefits

1. **Reduced Ambiguity**: Result label known before constraint extraction
2. **Self-Correction**: Explicit validation phase catches errors
3. **Better Context**: Position classification helps write context field
4. **Easier Debugging**: More granular phases = clearer reasoning trace
5. **Optimized for LLM**: Matches natural reasoning flow (general → specific)
