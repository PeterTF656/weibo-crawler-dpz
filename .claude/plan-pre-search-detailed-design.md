# Task Plan: Pre-Search Sub-Step Detailed Design

## Goal
Design detailed Pydantic models and formatting strategy for the pre-search optimization in Step A.3, enabling direct retrieval of user compositions when data volume is below threshold.

## Phases
- [x] Phase 1: Analyze current models and data flow
- [x] Phase 2: Specialist team consultation on detailed design
- [x] Phase 3: Finalize model specifications
- [x] Phase 4: Implementation complete

## Status
**COMPLETED** - All implementation tasks finished

---

# Specialist Team Consultation

## Executive Summary

- **One-liner**: Design clean, minimal model extensions that support both retrieval methods while maintaining backward compatibility
- **Key Decision**: Extend `ContextResult` with retrieval metadata, add `UserContextStats` as a lightweight stats model
- **Complexity**: Low-Medium (mostly additive changes, no breaking modifications)

---

## Team Discussion

### Agent Context & Memory Engineer

**Question 1: UserContextStats Model Design**

The `UserContextStats` should be minimal and focused on pre-search decision making:

```python
class UserContextStats(BaseModel):
    """Lightweight user statistics for pre-search routing decisions.

    Cached at user profile level, updated on composition/conversation writes.
    NOT fetched from DB on every request - should be passed in or cached.
    """
    composition_count: int = Field(
        ge=0,
        description="Total number of user's compositions/moments"
    )
    conversation_count: int = Field(
        ge=0,
        description="Total number of conversations user participated in"
    )
    total_message_count: int = Field(
        ge=0,
        description="Total messages across all conversations"
    )
    last_stats_update: str | None = Field(
        default=None,
        description="ISO timestamp of last stats update (for cache invalidation)"
    )
```

**Key design decision**: This model should be OPTIONAL in the state. If not provided, default to semantic search (safe fallback).

```python
# In StepAState
user_context_stats: NotRequired[UserContextStats]
"""User's content statistics for pre-search routing. Optional - if missing, defaults to semantic search."""
```

**Why Optional?**
1. Backwards compatible - existing callers don't break
2. Graceful degradation - missing stats = use semantic search
3. Separation of concerns - caller provides stats, agent doesn't fetch them

---

### LangGraph & LangChain Developer

**Question 2: ContextResult Updates for Direct Retrieval**

The current `ContextResult` model needs to distinguish between retrieval methods:

```python
# Current model
class ContextResult(BaseModel):
    question: str
    results: list[dict[str, Any]]
    source: Literal["graph", "web"]  # ← Too limited
    confidence: float
```

**Proposed extension:**

```python
class ContextResult(BaseModel):
    """Output from A.3: Context Search.

    Unified result format for all retrieval methods:
    - Semantic search (graph or web)
    - Direct retrieval (full data dump)
    """
    question: str = Field(
        description="The original question being answered"
    )
    results: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Retrieved context items"
    )
    source: Literal["graph", "web"] = Field(
        description="Data source: 'graph' for user data, 'web' for external knowledge"
    )
    retrieval_method: Literal["semantic", "direct"] = Field(
        default="semantic",
        description="How data was retrieved: 'semantic' for search, 'direct' for full retrieval"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Result quality score (0.0-1.0). Direct retrieval = 1.0 (complete data)"
    )
    coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Data coverage ratio. 1.0 = retrieved all available data. Only meaningful for direct retrieval."
    )
    total_available: int = Field(
        default=0,
        ge=0,
        description="Total items available (for direct retrieval tracking)"
    )
    formatted_context: str | None = Field(
        default=None,
        description="Pre-formatted context string for LLM consumption. Optional optimization."
    )
```

**Key additions:**
- `retrieval_method`: Distinguishes HOW we got the data
- `coverage`: Important for A.4 to know if we have complete picture
- `total_available`: Tells downstream how much data exists
- `formatted_context`: Optional pre-formatting to avoid re-processing

**Backward compatibility**: All new fields have defaults, so existing code works unchanged.

---

### Prompt Engineer

**Question 3: Formatting Strategy for Direct Retrieval**

When we retrieve ALL compositions (e.g., user has 12 posts), we need structured formatting for the LLM:

**BAD Approach (token-wasteful, hard to parse):**
```
Here are all 12 compositions from this user:
1. {"mongodb_id": "abc123", "content": "刚提了新车！人生第一辆宝马！", "created_at": "2024-01-15T10:30:00Z", "type": "text", ...}
2. {"mongodb_id": "def456", ...}
...
```

**GOOD Approach (structured, scannable, actionable):**

```markdown
## User Context: Complete Composition History

**Total Posts**: 12 | **Date Range**: 2024-01-01 to 2024-01-15 | **Coverage**: 100%

### Recent Posts (Chronological)

| # | Date | Type | Summary | Key Themes |
|---|------|------|---------|------------|
| 1 | Jan 15 | showing_off | 刚提了新车！人生第一辆宝马！ | car, achievement |
| 2 | Jan 14 | venting | 今天又加班到12点... | work, stress |
| 3 | Jan 12 | emo | 有时候觉得一个人也挺好的 | loneliness |
...

### Detected Patterns

**Emotional Trajectory**: Mixed (positive achievement posts interspersed with work stress)

**Recurring Topics**:
- Work/career (5 posts) - stress, overtime, achievement
- Relationships (3 posts) - loneliness, reflection
- Material/lifestyle (4 posts) - car, food, activities

**Notable Context**:
- User recently purchased a car (showing_off pattern)
- Work stress appears to be ongoing theme
- Some posts suggest relationship contemplation
```

**Formatting Function Design:**

```python
class CompositionFormatted(BaseModel):
    """Single composition formatted for LLM context."""
    index: int
    date_short: str  # "Jan 15"
    post_type: str | None  # From A.1 classification if available
    content_preview: str  # First 100 chars
    key_themes: list[str]  # Extracted keywords
    mongodb_id: str  # For reference


class FormattedCompositionContext(BaseModel):
    """Complete formatted context for direct retrieval results."""
    total_count: int
    date_range: str  # "2024-01-01 to 2024-01-15"
    coverage: float  # 1.0 for complete
    compositions: list[CompositionFormatted]
    detected_patterns: dict[str, Any]  # Optional pattern extraction
    formatted_string: str  # Final markdown string for LLM
```

**Formatting Rules:**
1. **Recency bias**: Most recent posts first
2. **Content preview**: Max 100 chars, preserve Chinese characters
3. **Pattern detection**: Simple keyword/theme extraction (no LLM call)
4. **Date normalization**: "Jan 15" not "2024-01-15T10:30:00Z"
5. **Table format**: Scannable at a glance

---

### DSPy Developer

**Important Note**: The formatting should NOT use an LLM call. The whole point of direct retrieval is to SKIP the semantic search overhead. Adding an LLM summarization defeats the purpose.

**Lightweight formatting alternatives:**
1. **Keyword extraction**: Use simple NLP (jieba for Chinese) to extract keywords
2. **Date grouping**: Group by week/month for temporal patterns
3. **Theme clustering**: Simple frequency-based theme detection

```python
def extract_simple_themes(content: str) -> list[str]:
    """Extract themes without LLM using keyword frequency."""
    import jieba

    # Chinese word segmentation
    words = jieba.cut(content)

    # Filter by predefined theme keywords
    theme_keywords = {
        "car": ["车", "宝马", "开车", "驾驶"],
        "work": ["加班", "工作", "公司", "上班"],
        "relationship": ["恋爱", "暗恋", "分手", "喜欢"],
        # ... more themes
    }

    themes = []
    for theme, keywords in theme_keywords.items():
        if any(kw in content for kw in keywords):
            themes.append(theme)

    return themes
```

---

### Senior Python/FastAPI Backend Engineer

**Implementation Considerations:**

1. **Where to format?** In `search.py` at retrieval time, NOT in A.4 node
2. **Caching formatted results**: Store `formatted_context` in ContextResult to avoid re-formatting
3. **Error handling**: If formatting fails, fall back to raw JSON dump

```python
async def direct_retrieve_compositions(
    user_id: str,
    question: ContextQuestion,
    stats: UserContextStats,
) -> ContextResult:
    """Direct retrieval path for users with few compositions."""

    # 1. Fetch all compositions from Supabase
    compositions = await fetch_all_user_compositions(user_id)

    # 2. Format for LLM consumption (no LLM call)
    formatted = format_compositions_for_context(
        compositions=compositions,
        question=question,
    )

    # 3. Return unified ContextResult
    return ContextResult(
        question=question.question,
        results=[c.model_dump() for c in compositions],  # Raw data preserved
        source="graph",
        retrieval_method="direct",
        confidence=1.0,  # We have everything
        coverage=1.0,  # Complete data
        total_available=stats.composition_count,
        formatted_context=formatted.formatted_string,  # Pre-formatted for A.4
    )
```

---

### Startup Mentor

**Reality check on complexity:**

The team is adding a lot of structure. Let me simplify:

**MVP Approach (What to build first):**

1. `UserContextStats` - YES, minimal version
2. `ContextResult` extensions - YES, but only `retrieval_method` and `formatted_context`
3. `FormattedCompositionContext` - SKIP for now, just use a formatting function that returns string
4. Theme extraction with jieba - MAYBE LATER, start with simple approach

**Simpler formatting first:**

```python
def format_compositions_simple(compositions: list[dict], question: str) -> str:
    """Simple formatting without NLP overhead."""
    lines = [
        f"## User Composition History ({len(compositions)} posts)",
        f"**Question**: {question}",
        "",
        "### Posts (Recent First)",
        "",
    ]

    for i, c in enumerate(compositions[:15], 1):  # Max 15
        date = c.get("created_at", "")[:10]  # Just the date part
        content = (c.get("content") or c.get("final_content") or "")[:100]
        lines.append(f"{i}. [{date}] {content}...")

    return "\n".join(lines)
```

This is ~20 lines of code, no dependencies, gets the job done. Add sophistication later when you have data on what A.4 actually needs.

---

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| `UserContextStats` is Optional in state | Backwards compatibility, graceful degradation |
| Add `retrieval_method` to ContextResult | Clear signal to A.4 about data completeness |
| Add `formatted_context` to ContextResult | Avoid re-processing in A.4 |
| Start with simple string formatting | Ship fast, measure, iterate |
| No LLM call in formatting | Defeats purpose of skipping semantic search |
| Threshold = 15 compositions | Balance between context size and search value |

---

## Final Model Specifications

### 1. UserContextStats (NEW)

```python
class UserContextStats(BaseModel):
    """Lightweight user statistics for pre-search routing decisions."""

    composition_count: int = Field(
        ge=0,
        description="Total number of user's compositions/moments"
    )
    conversation_count: int = Field(
        default=0,
        ge=0,
        description="Total number of conversations (Phase 2 feature)"
    )
    total_message_count: int = Field(
        default=0,
        ge=0,
        description="Total messages across all conversations (Phase 2 feature)"
    )
```

### 2. ContextResult (EXTENDED)

```python
class ContextResult(BaseModel):
    """Output from A.3: Context Search."""

    # Existing fields (unchanged)
    question: str = Field(description="The original question")
    results: list[dict[str, Any]] = Field(default_factory=list, description="Retrieved context")
    source: Literal["graph", "web"] = Field(description="Data source")
    confidence: float = Field(ge=0.0, le=1.0, description="Result quality score")

    # NEW fields
    retrieval_method: Literal["semantic", "direct"] = Field(
        default="semantic",
        description="How data was retrieved"
    )
    formatted_context: str | None = Field(
        default=None,
        description="Pre-formatted context string for LLM consumption"
    )
    total_available: int = Field(
        default=0,
        ge=0,
        description="Total items available in source"
    )
```

### 3. StepAState (EXTENDED)

```python
class StepAState(TypedDict):
    # ... existing fields ...

    # NEW field
    user_context_stats: NotRequired[UserContextStats]
    """User's content statistics for pre-search routing. If missing, defaults to semantic search."""
```

---

## Formatting Strategy

### Function Signature

```python
def format_compositions_for_context(
    compositions: list[dict[str, Any]],
    question: str,
    max_items: int = 15,
    max_content_length: int = 150,
) -> str:
    """Format compositions for LLM context window.

    Args:
        compositions: Raw composition dicts from DB
        question: The gap question being answered
        max_items: Maximum compositions to include (default 15)
        max_content_length: Max chars per content preview

    Returns:
        Formatted markdown string for LLM consumption
    """
```

### Output Format

```markdown
## User Composition History

**Posts Retrieved**: 12 | **Method**: Direct (complete history)
**Gap Question**: 用户过去发过关于什么话题的内容？

### Recent Posts

| # | Date | Content Preview |
|---|------|-----------------|
| 1 | 2024-01-15 | 刚提了新车！人生第一辆宝马！终于实现了一个小目标... |
| 2 | 2024-01-14 | 今天又加班到12点，真的累了，不知道这样的日子什么时候是个头... |
| 3 | 2024-01-12 | 有时候觉得一个人也挺好的，不用迁就别人，想干嘛就干嘛... |
...

### Context Notes
- This is the user's complete composition history
- Use this context to understand user's patterns, interests, and emotional state
```

### Key Formatting Rules

1. **Recency first**: Sort by `created_at` DESC
2. **Preview length**: 150 chars max, preserve word boundaries
3. **Date format**: YYYY-MM-DD for clarity
4. **Include metadata header**: Shows retrieval method and completeness
5. **Include the gap question**: Reminds LLM what to look for

---

## Implementation Order

1. **state.py**: Add `UserContextStats` model and state field
2. **state.py**: Extend `ContextResult` with new fields
3. **search.py**: Add `format_compositions_for_context()` function
4. **search.py**: Add `should_use_direct_retrieval()` routing function
5. **search.py**: Add `direct_retrieve_compositions()` function
6. **search.py**: Update `_execute_search()` to check routing
7. **Tests**: Add unit tests for formatting and routing logic

---

## Errors Encountered
- None yet

## Open Questions
1. Should `formatted_context` be computed lazily in A.4 instead of eagerly in A.3?
   - **Decision**: Eager in A.3 - avoids re-computation and keeps A.4 simple
2. Should we track conversation stats in Phase 1?
   - **Decision**: Include fields but don't use them yet (Phase 2 feature)
