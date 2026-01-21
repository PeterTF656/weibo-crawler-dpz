# Task Plan: Markdown-Based Skill Blocks for nl_to_cypher

## Goal
Design and implement markdown-based skill blocks that can be programmatically generated from graph topology and used by AI agents to convert natural language queries to Cypher.

## Context
The project leader has defined a "skill blocks" approach in `src/tools/graphdb/search/nl_to_cypher/skill-blocks/README.md`. The current design uses YAML format, but markdown is preferred for AI agent consumption.

## Success Criteria
- [ ] All primitive blocks can be generated programmatically from existing topology/property definitions
- [ ] Skills are defined in markdown format suitable for AI agent templates
- [ ] Clear composition grammar for chaining blocks together
- [ ] Integration path with existing DSPy/LangGraph infrastructure

---

## Phase 1: Context Gathering
**Status:** `complete`
**Started:** 2026-01-13
**Completed:** 2026-01-13

### Tasks
- [x] Read skill-blocks README.md (initial context)
- [x] Analyze graph topology definition (`model.py`)
- [x] Analyze property schema (`property_schema.py`)
- [x] Review existing helper functions
- [x] Understand ComplexQueryInput runtime model

### Dependencies
- None (initial phase)

---

## Phase 2: Block Type Analysis
**Status:** `complete`

### Tasks
- [x] Map NodeType blocks to markdown representation
- [x] Map EdgeType blocks to markdown representation
- [x] Map PropertyConstraint blocks to markdown format
- [x] Map VectorSearchSpec blocks to markdown format
- [x] Define NodeBlock decorated structure in markdown

---

## Phase 3: Skill Template Design
**Status:** `complete`

### Tasks
- [x] Design atomic segment SKILL markdown template
- [x] Design composition SKILL markdown template
- [x] Define tool integration points for DSPy/LangGraph
- [x] Create programmatic generation approach

---

## Phase 4: Implementation Proposal
**Status:** `complete`

### Tasks
- [x] Write Python generator for atomic segment skills
- [x] Write Python generator for composition skills
- [x] Define agent workflow integration
- [x] Create evaluation framework

---

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| (none yet) | | |

---

## Decisions Log
| Decision | Rationale | Date |
|----------|-----------|------|
| Use markdown over YAML | AI agents process markdown better; human-readable | 2026-01-13 |
| Planning files in .claude/plans/ | Keep organized with other plans | 2026-01-13 |
