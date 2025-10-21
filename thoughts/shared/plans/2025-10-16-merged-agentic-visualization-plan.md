# Merged Plan: Agentic Five Lenses with Rich Visualization

**Date**: 2025-10-16
**Status**: Architecture Design
**Strategy**: Parallel development using git worktrees

---

## Executive Summary

This plan merges two complementary initiatives:
1. **Agentic Orchestration** (track-a): Natural language interface with LangGraph coordination
2. **Rich Visualization** (main): Plotly charts, markdown tables, executive summaries

**Key Insight**: The orchestration layer provides *intelligence* (what to analyze), while visualization provides *clarity* (how to present it). Together they create a powerful business analytics interface.

**Timeline**: 2-3 weeks with parallel development
**Development Strategy**: Use git worktrees to develop formatters and orchestration simultaneously

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Claude Desktop User                        │
└────────────────────────┬────────────────────────────────────┘
                         │ Natural Language Query
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              UNIFIED MCP SERVER (track-a)                    │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   Query Interpreter (LLM)                              │ │
│  │   "What's our customer health?" → Intent + Parameters  │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   ↓                                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   LangGraph Coordinator (EXISTING - track-a)           │ │
│  │   - Parallel lens execution                            │ │
│  │   - Dependency management (Lens 2 needs Lens 1)        │ │
│  │   - Error handling & partial results                   │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   ↓                                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   Result Formatter (NEW - from main)                   │ │
│  │   - Markdown tables (lens outputs)                     │ │
│  │   - Plotly charts (trends, distributions, gauges)      │ │
│  │   - Executive summaries (actionable insights)          │ │
│  └────────────────┬───────────────────────────────────────┘ │
│                   ↓                                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │   Lens Services (EXISTING - track-a)                   │ │
│  │   lens1_tool, lens2_tool, ..., lens5_tool             │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │ Direct calls
                      ↓
┌─────────────────────────────────────────────────────────────┐
│   Five Lenses Analytical Core (customer_base_audit/)        │
│   - Immutable dataclasses                                   │
│   - 486 passing tests                                       │
└─────────────────────────────────────────────────────────────┘
```

**Key Principle**: One MCP server (track-a) with two modes:
1. **Orchestrated mode**: Natural language → multi-lens analysis → rich visualization
2. **Direct mode**: Individual lens tools for power users

---

## Current State Analysis

### Track A (Orchestration) - `.worktrees/track-a/`

**Completed** ✅:
- MCP server infrastructure (`main.py`, `observability.py`)
- All 5 lens services (`tools/lens1.py` through `lens5.py`)
- Foundation services (`tools/data_mart.py`, `tools/rfm.py`, `tools/cohorts.py`)
- LangGraph coordinator (`orchestration/coordinator.py`)
- Orchestrated analysis tool (`tools/orchestrated_analysis.py`)
- State management (`state.py`)

**Gaps**:
- ❌ No visualization/formatting
- ❌ Plain JSON outputs only
- ❌ No executive summaries

### Main Branch (Visualization) - Current Working Directory

**Completed** ✅:
- JSON serialization utilities (`customer_base_audit/mcp/serializers.py`)
- Pydantic schemas (`customer_base_audit/mcp/schemas.py`)
- MCP server scaffold (`customer_base_audit/mcp/server.py`)
- Tool wrappers (`customer_base_audit/mcp/tools.py`)
- Comprehensive tests (14 passing)

**Planned** (Phase 2 from original plan):
- ❌ Markdown table formatters
- ❌ Plotly chart generators
- ❌ Executive summary templates

---

## Merged Implementation Strategy

### Phase 1: Formatter Development (Main Branch) - WEEK 1

**Goal**: Build visualization components independently in main branch

**Why main branch?**: Formatters are self-contained and don't need orchestration infrastructure

**Deliverables**:
1. `customer_base_audit/mcp/formatters/markdown_tables.py`
   - `format_lens1_table()` - Snapshot metrics table
   - `format_lens2_table()` - Migration matrix + rates
   - `format_lens4_decomposition_table()` - Cohort comparison
   - `format_lens5_health_summary_table()` - Health scorecard

2. `customer_base_audit/mcp/formatters/plotly_charts.py`
   - `create_retention_trend_chart()` - Lens 3 cohort retention curves
   - `create_revenue_concentration_pie()` - Lens 1 top customer visualization
   - `create_health_score_gauge()` - Lens 5 health meter
   - `create_executive_dashboard()` - Multi-lens combined view

3. `customer_base_audit/mcp/formatters/executive_summaries.py`
   - `generate_health_summary()` - Lens 5 narrative
   - `generate_retention_insights()` - Lens 2 + 3 synthesis
   - `generate_cohort_comparison()` - Lens 4 analysis

4. `customer_base_audit/mcp/formatters/__init__.py`
   - Public API exports

**Tests**: `tests/test_mcp_formatters.py`
- Test each formatter with sample lens outputs
- Verify markdown validity
- Verify Plotly JSON schema
- Test with edge cases (empty data, single customer, etc.)

**Success Criteria**:
- [x] All formatters tested independently
- [x] Markdown renders correctly in Claude Desktop
- [x] Plotly charts display interactively
- [x] Executive summaries are actionable

---

### Phase 2: Integration into Track A - WEEK 2

**Goal**: Merge formatters into track-a's orchestration layer

**Work in**: `.worktrees/track-a/`

**Steps**:

#### 2.1. Copy Formatters to Track A
```bash
# From main branch
# Note: Path verified to exist in track-a worktree (2025-10-20)
cp -r customer_base_audit/mcp/formatters/ \
  .worktrees/track-a/analytics/services/mcp_server/formatters/

# Update imports for track-a structure
# customer_base_audit.analyses → analytics.services.mcp_server imports
```

#### 2.2. Enhance Orchestration Coordinator

**File**: `.worktrees/track-a/analytics/services/mcp_server/orchestration/coordinator.py`

Add new node to LangGraph:

```python
def format_results(state: AnalysisState) -> AnalysisState:
    """Format lens results with visualizations and narratives."""
    from analytics.services.mcp_server.formatters import (
        format_lens1_table,
        create_health_score_gauge,
        generate_health_summary,
    )

    formatted_outputs = {}

    # Format each lens that executed successfully
    if state["lens1_result"]:
        formatted_outputs["lens1_table"] = format_lens1_table(state["lens1_result"])
        formatted_outputs["lens1_chart"] = create_revenue_concentration_pie(state["lens1_result"])

    if state["lens5_result"]:
        formatted_outputs["health_summary"] = generate_health_summary(state["lens5_result"])
        formatted_outputs["health_gauge"] = create_health_score_gauge(state["lens5_result"])

    state["formatted_outputs"] = formatted_outputs
    return state

# Add to workflow
workflow.add_node("format_results", format_results)
workflow.add_edge("synthesize_results", "format_results")
```

#### 2.3. Update Response Models

**File**: `.worktrees/track-a/analytics/services/mcp_server/tools/orchestrated_analysis.py`

```python
class OrchestrationResponse(BaseModel):
    """Enhanced response with formatted outputs."""
    raw_results: dict  # Original lens outputs
    formatted_outputs: dict  # Markdown tables, charts, summaries
    insights: list[str]  # LLM-generated insights
    execution_time_seconds: float
    lenses_executed: list[str]
```

#### 2.4. Add Direct Formatting Tools (Optional)

For power users who want individual lens + formatting:

```python
@mcp.tool()
async def lens1_snapshot_formatted(
    ctx: Context,
    rfm_data: list[dict],
) -> dict:
    """Lens 1 with formatted output."""
    # Execute lens1
    result = await lens1_snapshot(ctx, rfm_data)

    # Add formatting
    from formatters import format_lens1_table, create_revenue_concentration_pie

    return {
        "raw": result,
        "table": format_lens1_table(result),
        "chart": create_revenue_concentration_pie(result),
    }
```

**Success Criteria**:
- [ ] Orchestrated analysis returns formatted outputs
- [ ] Charts embed correctly in Claude Desktop responses
- [ ] Markdown tables render properly
- [ ] Executive summaries provide actionable insights

---

### Phase 3: Enhanced Natural Language Interface - WEEK 3

**Goal**: Improve query interpretation and result synthesis

**Work in**: `.worktrees/track-a/`

#### 3.1. Query Interpreter Enhancement

**File**: `.worktrees/track-a/analytics/services/mcp_server/orchestration/query_interpreter.py` (new)

```python
from anthropic import Anthropic

class QueryInterpreter:
    """LLM-powered query interpretation."""

    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)

    async def interpret(self, query: str) -> dict:
        """Convert natural language to lens execution plan."""
        prompt = f"""Analyze this customer analytics query and determine which lenses to execute:

Query: {query}

Available lenses:
- Lens 1: Single-period snapshot (revenue concentration, one-time buyers)
- Lens 2: Period comparison (retention, churn, growth)
- Lens 3: Cohort evolution (retention curves over time)
- Lens 4: Multi-cohort comparison (cohort quality trends)
- Lens 5: Overall health assessment (composite health score)

Return JSON with:
{{
  "lenses": ["lens1", "lens2", ...],
  "time_period": "last_quarter" | "last_month" | "custom",
  "focus_areas": ["retention", "revenue", "health"],
  "comparison_needed": true/false
}}"""

        response = await self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )

        return json.loads(response.content[0].text)
```

#### 3.2. Result Synthesizer

**File**: `.worktrees/track-a/analytics/services/mcp_server/orchestration/result_synthesizer.py` (new)

```python
class ResultSynthesizer:
    """Generate insights from multi-lens analysis."""

    async def synthesize(self, lens_results: dict) -> dict:
        """Create executive summary and recommendations."""

        # Combine all lens insights
        context = self._prepare_context(lens_results)

        prompt = f"""You are a customer analytics expert. Analyze these results and provide:

1. Executive Summary (2-3 sentences)
2. Top 3 Insights (ordered by business impact)
3. Recommended Actions (prioritized by ROI)

Data:
{json.dumps(context, indent=2)}

Focus on actionable insights that drive revenue and retention."""

        response = await self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}]
        )

        return self._parse_synthesis(response.content[0].text)
```

**Success Criteria**:
- [ ] Natural language queries route to correct lenses
- [ ] Multi-lens results synthesize into coherent narratives
- [ ] Recommendations are specific and actionable
- [ ] Cost per query < $0.50

---

## Parallel Development Strategy Using Worktrees

### Setup: Create Development Worktrees

```bash
# If you want a dedicated worktree for formatters
git worktree add .worktrees/formatters-dev feature/formatters-development

# Work on formatters in isolation
cd .worktrees/formatters-dev
# Implement Phase 1 formatters here

# Meanwhile, continue orchestration work in track-a
cd .worktrees/track-a
# Enhance coordinator, add health checks, etc.
```

### Workflow: Parallel Streams

**Stream 1: Formatters** (Developer A or Solo: Monday-Wednesday)
- Branch: `feature/formatters-development` in main or formatters-dev worktree
- Work: Implement all formatters, charts, summaries
- Tests: Isolated formatter tests
- No dependencies on track-a

**Stream 2: Orchestration Enhancement** (Developer B or Solo: Monday-Wednesday)
- Branch: `feature/orchestration-enhancement` in track-a worktree
- Work: Add health checks, improve coordinator, optimize performance
- Tests: Orchestration integration tests
- No dependencies on formatters

**Stream 3: Integration** (Thursday-Friday)
- Merge formatters into track-a
- Update coordinator to use formatters
- Integration testing
- End-to-end testing with Claude Desktop

### Benefits of This Approach

1. **Isolation**: Formatters can be tested without orchestration complexity
2. **Parallelism**: Two developers (or one developer context-switching) can work simultaneously
3. **Safety**: Track-a remains stable while formatters are developed
4. **Testability**: Each component has independent test suite
5. **Flexibility**: Can merge formatters when ready, not before

---

## Migration Path for Existing Work

### What to Do with Main Branch MCP Server

**Option 1: Archive It** (Recommended)
```bash
git checkout main
git mv customer_base_audit/mcp customer_base_audit/mcp_archive
git commit -m "Archive initial MCP server POC, migrating to track-a"
```

Keep the serializers and schemas as reference, but use track-a as primary server.

**Option 2: Keep as Standalone Tool**
- Maintain both servers for different use cases
- Main: Direct lens access for analysts
- Track-a: Orchestrated analysis for executives
- Configure different ports in Claude Desktop

**Recommendation**: Archive main's MCP server, migrate formatters to track-a.

---

## File Structure (Post-Merge)

```
AutoCLV/
├── customer_base_audit/           # Core analytics (unchanged)
│   ├── analyses/
│   │   ├── lens1.py
│   │   ├── lens2.py
│   │   ├── lens3.py
│   │   ├── lens4.py
│   │   └── lens5.py
│   └── mcp_archive/               # Archived POC
│
└── .worktrees/
    └── track-a/
        └── analytics/
            └── services/
                └── mcp_server/
                    ├── main.py                    # MCP server entry
                    ├── observability.py
                    ├── state.py
                    │
                    ├── orchestration/
                    │   ├── coordinator.py         # LangGraph workflow
                    │   ├── query_interpreter.py   # NEW: NL → intent
                    │   └── result_synthesizer.py  # NEW: Results → insights
                    │
                    ├── formatters/                # NEW: From main branch
                    │   ├── __init__.py
                    │   ├── markdown_tables.py
                    │   ├── plotly_charts.py
                    │   └── executive_summaries.py
                    │
                    └── tools/
                        ├── lens1.py               # Existing
                        ├── lens2.py
                        ├── lens3.py
                        ├── lens4.py
                        ├── lens5.py
                        ├── orchestrated_analysis.py  # Enhanced with formatters
                        ├── data_mart.py
                        ├── rfm.py
                        └── cohorts.py
```

---

## Testing Strategy

### Unit Tests (Per Worktree)

**Formatters** (main or formatters-dev):
```bash
pytest tests/test_mcp_formatters.py -v
# Test markdown rendering
# Test Plotly JSON schemas
# Test edge cases (empty data, single customer)
```

**Orchestration** (track-a):
```bash
cd .worktrees/track-a
pytest tests/services/mcp_server/test_orchestration.py -v
# Test LangGraph workflows
# Test parallel execution
# Test error handling
```

### Integration Tests (Track-A After Merge)

```bash
cd .worktrees/track-a
pytest tests/services/mcp_server/test_integration_formatted.py -v
# Test orchestrated_analysis with formatters
# Verify charts + tables + summaries in response
# Test end-to-end with Claude Desktop
```

---

## Success Metrics

### Technical
- [ ] All 486 core tests still passing
- [ ] Formatter tests: 20+ tests covering all output types
- [ ] Integration tests: 10+ tests for formatted orchestration
- [ ] No regression in track-a orchestration performance

### User Experience
- [ ] Natural language query → formatted results in < 10 seconds
- [ ] Charts render interactively in Claude Desktop
- [ ] Executive summaries are < 200 words, actionable
- [ ] Health score gauge updates in real-time

### Business
- [ ] Cost per analysis < $0.50 (Claude API usage)
- [ ] 90% of common queries handled without clarification
- [ ] Executives can make decisions from summaries alone (no raw data needed)

---

## Timeline & Milestones

### Week 1: Parallel Development
- **Mon-Wed**: Formatters (main branch) + Orchestration polish (track-a)
- **Thu**: Integration prep, merge formatters to track-a
- **Fri**: Integration testing, bug fixes

### Week 2: Enhanced Orchestration
- **Mon-Tue**: Query interpreter implementation
- **Wed-Thu**: Result synthesizer implementation
- **Fri**: End-to-end testing with Claude Desktop

### Week 3: Polish & Deploy
- **Mon-Tue**: Performance optimization, caching
- **Wed**: Documentation (user guide, API docs)
- **Thu**: Staging deployment, user acceptance testing
- **Fri**: Production deployment, monitoring setup

---

## Risks & Mitigations

### Risk 1: Formatters Don't Integrate Cleanly
**Mitigation**: Define clear interfaces early (input: lens dataclass, output: dict with markdown/chart)

### Risk 2: LLM Costs Exceed Budget
**Mitigation**: Cache common queries, use cheaper models for synthesis, implement request throttling

### Risk 3: Track-A and Main Diverge Too Much
**Mitigation**: Daily syncs of `customer_base_audit/` core from main → track-a

### Risk 4: Plotly Charts Don't Render in Claude Desktop
**Mitigation**: Test early with MCP Inspector, fallback to PNG export if needed

---

## Next Steps

1. **Immediate** (Today):
   - Create `feature/formatters-development` branch in main
   - Set up formatter module structure
   - Write first formatter test

2. **Week 1 Day 1**:
   - Implement markdown table formatters
   - Create Plotly chart generators
   - Test in isolation

3. **Week 1 Day 4**:
   - Merge formatters to track-a
   - Update orchestration coordinator
   - Integration testing

4. **Week 2**:
   - Build query interpreter
   - Implement result synthesizer
   - Polish user experience

---

## Decision Points

**Choose One**:

**A. Conservative Path** (Recommended for solo developer)
- Week 1: Formatters only (in main branch)
- Week 2: Merge to track-a + integration
- Week 3: Natural language enhancement

**B. Aggressive Path** (For team or experienced developer)
- Parallel streams from Day 1
- Formatters + orchestration simultaneously
- Higher risk, faster delivery

**C. Minimal Path** (Skip NL for now)
- Just add formatters to track-a
- Skip query interpreter and synthesizer
- Users still use direct tool calls, but get rich output

---

## Recommendation

**Start with Conservative Path**:
1. Complete formatter development in main (Week 1)
2. Test formatters thoroughly in isolation
3. Merge to track-a when stable (Week 1 end)
4. Enhance orchestration with formatters (Week 2)
5. Add NL interface if time permits (Week 3)

This gives you working, tested components at each stage and allows for early user feedback on visualizations before investing in NL processing.
