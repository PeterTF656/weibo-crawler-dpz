# Findings: Sequential Agent Architecture

## Overview

The nl_to_cypher/sequential module implements a **3-tier hierarchical agent system** for converting natural language queries to Neo4j Cypher queries.

---

## Architecture Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MAIN AGENT (Orchestrator)                         │
│  Decomposes query → Fans out to parallel branches → Aggregates → Searches   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Send API (×N branches)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        BRANCH AGENT (Per-Branch)                             │
│  Plans path → Validates topology → Replans if invalid → Builds levels       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ asyncio.gather (×M levels)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LEVEL AGENT (Per-Level)                              │
│  Schema → Selection → Narrowing → Parallel Extraction → Merge → Assemble    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Layer 1: Main Agent

**Location:** `agent/main_agent/`

**Graph Flow:**
```
START → decompose → fan_out_branches → [branch_worker ×N] → aggregate → search → data_processing → END
```

**Key Components:**
| Node | Purpose | Type |
|------|---------|------|
| decompose | DSPy QueryDecomposerModule → DecomposedQuery | LLM |
| fan_out_branches | Creates Send objects for parallel execution | Router |
| branch_worker | Calls arun_branch_planning() per branch | Worker |
| aggregate | Separates intersection vs union results | Reducer |
| search | Executes Neo4j queries via read_data_multi_branch_parallel | Executor |
| data_processing | Builds CleanSearchResult | Formatter |

**Key Patterns:**
- **Send API**: Dynamic parallelization (N branches at runtime)
- **operator.add reducer**: Accumulates branch_results from parallel workers
- **InMemorySaver**: Fast checkpointing without Postgres overhead

---

## Layer 2: Branch Agent

**Location:** `branch/`

**Graph Flow:**
```
START → plan_path → validate_path → [conditional]
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         │                               │                               │
    path_valid=True              path_valid=False &              path_valid=False &
         │                       replan_count < max               replan_count >= max
         ▼                               │                               │
   build_levels ←────────────────────────┘                               ▼
         │                                                          END (failure)
         ▼
   link_levels
         │
         ▼
   END (success)
```

**Key Components:**
| Node | Purpose | Type |
|------|---------|------|
| plan_path | DSPy PathPlanner → List[PlanningLevel] | LLM |
| validate_path | Checks GRAPH_TOPOLOGY validity | Validator |
| build_levels | asyncio.gather() calls to Level Agent | Parallel Executor |
| link_levels | Populates path fields between levels | Linker |

**Key Patterns:**
- **Replan Loop**: Up to 3 attempts with validation feedback
- **State-driven replanning**: replan_guidance passed to LLM on retry

---

## Layer 3: Level Agent

**Location:** `agent/level/`

**Graph Flow (8-node pipeline):**
```
START → schema_generation → structural_selection → schema_narrowing
                                                         │
                              ┌──────────────────────────┼──────────────────────────┐
                              │                          │                          │
                    time_property_extraction   nontime_property_extraction   vector_search_extraction
                              │                          │                          │
                              └──────────────────────────┼──────────────────────────┘
                                                         │
                                                         ▼
                                               merge_property_results
                                                         │
                                                         ▼
                                                   assemble_level
                                                         │
                                                         ▼
                                                        END
```

**Key Components:**
| Phase | Node | Purpose | Type | Time |
|-------|------|---------|------|------|
| 1 | schema_generation | Topology lookup (get_valid_next_hops) | Deterministic | ~1ms |
| 2 | structural_selection | LLM selects best hop | LLM | ~400ms |
| 2.5 | schema_narrowing | Categorize properties | Deterministic | ~1ms |
| 3 | time_property_extraction | Extract time filters | LLM (conditional) | ~400ms |
| 3 | nontime_property_extraction | Extract other filters | LLM (conditional) | ~400ms |
| 3 | vector_search_extraction | Extract semantic search | LLM (conditional) | ~400ms |
| 3.5 | merge_property_results | Combine extraction outputs | Deterministic | ~1ms |
| 4 | assemble_level | Build final Level object | Deterministic | ~1ms |

**Key Patterns:**
- **Conditional Parallel Routing**: route_property_extraction returns Sequence[str] of nodes
- **Isolated State Updates**: Each extraction node updates unique field (no conflicts)
- **Dynamic DSPy Signatures**: Literal-constrained types at runtime

---

## Data Flow

```
User Query
    │
    ▼
DecomposedQuery = {
  anchor_label: "Composition",
  intersection_branches: ["Find by emotion...", "Filter by time..."],
  union_branches: ["Or match by topic..."]
}
    │
    ▼ (per branch_intent)
BranchGraphState = {
  planned_path: [PlanningLevel, ...],
  built_levels: [Level, ...],
  path_valid: bool
}
    │
    ▼ (per level in path)
LevelState = {
  selected_hop_option: NextHopOption,
  assembled_level: Level
}
    │
    ▼ (aggregated)
ComplexQueryInput = {
  branches: [Branch(levels=[Level, ...]), ...],
  projection: [...],
  limit: int
}
    │
    ▼
SearchResult → CleanSearchResult → API Response
```

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| 3-tier hierarchy | Separation of concerns: decomposition, planning, building |
| Send API for branches | Dynamic N branches determined at runtime |
| asyncio.gather for levels | Fixed M levels per branch, simpler coordination |
| Conditional parallel routing | Only run extraction nodes that apply |
| InMemorySaver default | Eliminates 5-7s Postgres connection overhead |
| Replan loop with feedback | Self-correction with LLM-readable error messages |

---

## Performance Characteristics

| Stage | Typical Time | LLM Calls |
|-------|--------------|-----------|
| Decomposition | ~500ms | 1 |
| Branch Planning (per branch) | ~800ms | 1-4 (with replans) |
| Level Building (per level) | ~800ms | 2-4 (parallel) |
| Search Execution | ~200-500ms | 0 |
| Data Processing | ~10ms | 0 |

**Total (1 branch, 2 levels):** ~2-3 seconds

---

## Files Reference

| Component | Key Files |
|-----------|-----------|
| Main Agent | graph.py, nodes.py, state.py, models.py |
| Branch Agent | agent.py, state.py, model.py, dspy/modules/branch_planner.py |
| Level Agent | level_agent.py, state.py, phases/*.py |
| Decomposer | agent/decomposer/agent/dspy/decomposer_module.py |
