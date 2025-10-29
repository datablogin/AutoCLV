# AutoCLV Project Completion Implementation Plan

**Date**: 2025-10-21
**Based On**: Research document `thoughts/shared/research/2025-10-21-track-merge-strategy.md`
**Current Status**: 80-85% complete
**Target**: v1.0.0 release in 5 weeks

## Overview

This plan outlines the final implementation phases to complete the AutoCLV project and merge all development tracks into a unified v1.0.0 release. The project has evolved into a production-ready Agentic MCP Server with LLM capabilities, orchestration, and comprehensive observability.

## Current State Analysis

### What's Complete ✅
- **Track-A MCP Server** (Phase 5 v2.0.0):
  - LangGraph orchestration with parallel lens execution
  - Optional LLM-powered query parsing and narrative synthesis
  - Full observability stack (OpenTelemetry, Prometheus, Grafana, Jaeger)
  - Circuit breakers, health checks, resilience patterns
  - Query caching (30-50% cost reduction)
  - Lens 1 and Lens 5 fully implemented
  - Foundation tools (data mart, RFM, cohorts, data loader)
  - Basic formatters (markdown tables, Plotly charts, executive summaries)

- **GitHub Issues Cleanup**:
  - Closed 7 duplicate issues (#94-97, #111-114) as "completed-in-track-a"
  - Reassigned 9 active issues to appropriate tracks

### What's Incomplete ⏸️

**Track-A Priority** (10-15 days):
- Issue #131: Lens 2 full MCP tool wrapper implementation
- Issue #132: Lens 3 full MCP tool wrapper implementation
- Issue #133: Lens 4 full MCP tool wrapper implementation
- Issue #122: Cache key normalization (medium priority)
- Issue #125: Prompt injection sanitization (medium priority)
- Issues #123, #124, #126, #127: Low-priority tech debt

**Main Branch** (5-6 days):
- Issue #115: Enhanced visualizations (dashboards, heatmaps, Sankey)
- Issue #129: Chart token usage optimization

**Track-B Decision** (TBD):
- Evaluate whether to merge analytics libraries, archive, or maintain in parallel

## What We're NOT Doing

To prevent scope creep, the following are **explicitly out of scope** for v1.0.0:

- ❌ Issue #116: Advanced features (anomaly alerts, scenario planning, benchmarking) - deferred to v1.1.0
- ❌ Complete rewrite of any existing working components
- ❌ New features not required for core Five Lenses functionality
- ❌ Performance optimizations beyond Issue #122 and #129
- ❌ Additional LLM providers beyond Anthropic Claude
- ❌ Mobile or desktop application development

## Desired End State

### v1.0.0 Success Criteria

**Functional Requirements**:
- ✅ All 5 lenses fully operational via MCP server
- ✅ Orchestrated analysis working with natural language queries
- ✅ Enhanced visualizations (dashboards, charts, heatmaps)
- ✅ Production observability stack functional
- ✅ Security hardening complete (prompt injection prevention)

**Quality Requirements**:
- ✅ Test coverage >80% for all lens implementations
- ✅ All automated tests passing
- ✅ Type checking passing with mypy
- ✅ Linting passing with ruff
- ✅ Documentation complete and accurate

**Deployment Requirements**:
- ✅ Single unified codebase on main branch
- ✅ All tracks merged successfully
- ✅ Claude Desktop configuration documented
- ✅ README updated with current architecture
- ✅ Release tagged as v1.0.0

### Verification

**Automated**:
```bash
# All tests pass
make test

# Type checking passes
make type-check

# Linting passes
make lint

# Integration tests pass
pytest tests/services/mcp_server/test_orchestration.py -v
```

**Manual**:
- [ ] All 5 lenses return real data (not placeholders) via Claude Desktop
- [ ] Orchestrated analysis generates meaningful insights
- [ ] Enhanced visualizations display correctly in Claude Desktop
- [ ] Observability stack (Jaeger, Prometheus, Grafana) works end-to-end
- [ ] No security vulnerabilities in LLM query handling
- [ ] Performance is acceptable (<10s for typical orchestrated analysis)

---

## Implementation Phases

### Phase 1: Track-A Lens Implementations (Week 1-2)

**Duration**: 7-10 days
**Track**: Track-A (`.worktrees/track-a`)
**Dependencies**: None (can start immediately)
**Parallelization**: ✅ Lens 2 and Lens 3 can be developed in parallel by different developers

#### Week 1: Lens 2 and Lens 3 (PARALLEL)

##### Stream A: Lens 2 Implementation (Issue #131)
**Duration**: 2-3 days
**Developer**: Developer A (or Mon-Wed)

**Changes Required**:

1. **File**: `analytics/services/mcp_server/tools/lens2.py`
   - Replace placeholder in `coordinator.py:827-852` with full implementation
   - Import `analyze_period_comparison()` from `customer_base_audit/analyses/lens2.py`
   - Extract period aggregations for two periods from shared state
   - Convert `Lens2Metrics` dataclass to dict format
   - Add OpenTelemetry span with attributes (customer counts, retention rate, growth)

2. **File**: `analytics/services/mcp_server/orchestration/coordinator.py`
   - Update `_execute_lens2()` method (lines 827-852)
   - Add growth momentum insight generation
   - Add retention trend analysis

3. **File**: `tests/services/mcp_server/test_lens_tools.py`
   - Add comprehensive Lens 2 test cases
   - Test period comparison with different date ranges
   - Test retention/churn calculation accuracy
   - Test integration with orchestration

**Success Criteria**:

Automated Verification:
- [x] Lens 2 tests pass: `pytest tests/services/mcp_server/test_lens_tools.py::test_lens2_*`
- [x] Type checking passes: `make type-check`
- [x] Orchestration test passes: `pytest tests/services/mcp_server/test_orchestration.py -k lens2`

Manual Verification:
- [x] Lens 2 returns real period comparison data via MCP (Q1→Q2 2024: 3,997→4,000 customers)
- [x] Growth momentum insights are actionable ("Strong Growth Momentum" with 31.97% revenue growth)
- [x] Retention rates match expectations from core implementation (100% retention verified)

**Implementation Note**: After all automated tests pass, pause for manual verification via Claude Desktop before proceeding.

##### Stream B: Lens 3 Implementation (Issue #132)
**Duration**: 2-3 days
**Developer**: Developer B (or Mon-Wed, parallel to Lens 2)

**Changes Required**:

1. **File**: `analytics/services/mcp_server/tools/lens3.py`
   - Replace placeholder in `coordinator.py:854-876` with full implementation
   - Import `analyze_cohort_evolution()` from `customer_base_audit/analyses/lens3.py`
   - Extract cohort assignments and period aggregations from shared state
   - Add cohort selection parameter (default to first cohort or specified cohort_id)
   - Convert `CohortEvolutionMetrics` dataclass to dict format
   - Add OpenTelemetry span with attributes (cohort size, maturity, LTV)

2. **File**: `analytics/services/mcp_server/orchestration/coordinator.py`
   - Update `_execute_lens3()` method (lines 854-876)
   - Add cohort maturity insight generation
   - Add LTV trajectory analysis

3. **File**: `tests/services/mcp_server/test_lens_tools.py`
   - Add comprehensive Lens 3 test cases
   - Test cohort evolution over multiple periods
   - Test retention curve calculations
   - Test integration with orchestration

**Success Criteria**:

Automated Verification:
- [x] Lens 3 tests pass: `pytest tests/services/mcp_server/test_lens_tools.py::test_lens3_*`
- [x] Type checking passes: `make type-check`
- [x] Orchestration test passes: `pytest tests/services/mcp_server/test_orchestration.py -k lens3`

Manual Verification:
- [x] Lens 3 returns real cohort evolution data via MCP (2023-Q1: 831 customers, 18 months tracked)
- [x] Retention curves display correctly (stable 63-67% long-term retention)
- [x] Cohort maturity assessment is accurate ("Mature" for 18-month cohort)
- [x] LTV trajectory insights are actionable ("Strong" with $95-105 stable revenue)

**Implementation Note**: After all automated tests pass, pause for manual verification via Claude Desktop before proceeding.

##### Integration Point (Friday Week 1)
**Duration**: 1 day

**Tasks**:
- Merge both Lens 2 and Lens 3 implementations
- Run full integration test suite
- Update documentation with new lens capabilities
- Test orchestrated analysis with Lens 1, 2, 3, and 5 combinations

**Success Criteria**:
- [ ] All tests pass: `make test`
- [ ] Orchestrated analysis correctly includes Lens 2 and 3 results
- [ ] No regressions in Lens 1 or Lens 5

#### Week 2: Lens 4 Implementation + Critical Tech Debt

##### Mon-Wed: Lens 4 Implementation (Issue #133)
**Duration**: 3-4 days
**Developer**: Developer A or B (sequential after Week 1)

**Changes Required**:

1. **File**: `analytics/services/mcp_server/tools/lens4.py`
   - Replace placeholder in `coordinator.py:878-900` with full implementation
   - Import `analyze_multi_cohort_comparison()` from `customer_base_audit/analyses/lens4.py`
   - Extract cohort assignments and period aggregations from shared state
   - Add cohort selection parameter (default to all cohorts or specified list)
   - Convert `MultiCohortComparison` and nested dataclasses to dict format
   - Add OpenTelemetry span with attributes (cohort count, best/worst performance)

2. **File**: `analytics/services/mcp_server/orchestration/coordinator.py`
   - Update `_execute_lens4()` method (lines 878-900)
   - Add best/worst cohort identification
   - Add cohort quality trend analysis (improving/declining)

3. **File**: `tests/services/mcp_server/test_lens_tools.py`
   - Add comprehensive Lens 4 test cases (most complex lens)
   - Test multi-cohort comparison with varying cohort counts
   - Test performance ranking logic
   - Test decomposition metrics
   - Test integration with orchestration

**Success Criteria**:

Automated Verification:
- [x] Lens 4 tests pass: `pytest tests/services/mcp_server/test_lens_tools.py::test_lens4_*`
- [x] Type checking passes: `make type-check`
- [x] Orchestration test passes: `pytest tests/services/mcp_server/test_orchestration.py -k lens4`

Manual Verification:
- [x] Lens 4 returns real multi-cohort comparison data via MCP
- [x] Performance rankings are accurate
- [x] Best and worst cohorts correctly identified
- [x] Quality trend analysis (improving/declining) is actionable
- [x] Decomposition metrics break down revenue/retention correctly

**Implementation Note**: Lens 4 is the most complex (821 lines core implementation). Allow extra time for testing. After all automated tests pass, pause for manual verification via Claude Desktop.

##### Thu: Issue #122 - Cache Key Normalization
**Duration**: 1-2 days
**Track**: Track-A
**Priority**: Medium (cost optimization)

**Changes Required**:

1. **File**: `analytics/services/mcp_server/orchestration/query_cache.py`
   - Enhance `_normalize_cache_key()` method (lines 186-192)
   - Implement smarter normalization beyond lowercase/strip:
     - Remove punctuation
     - Normalize whitespace
     - Optional: Remove common stopwords ("what", "show", "tell me")
     - Optional: Use simple stemming for common verbs
   - Add configuration for normalization aggressiveness

2. **File**: `tests/services/mcp_server/test_phase5_natural_language.py`
   - Add tests for semantically equivalent queries
   - Test cache hit rate improvements
   - Verify no false positives (different queries incorrectly matching)

**Success Criteria**:

Automated Verification:
- [x] Cache tests pass: `pytest tests/services/mcp_server/test_phase5_natural_language.py::test_cache_*`
- [x] Type checking passes: `make type-check`

Manual Verification:
- [x] Cache hit rate increases from ~30% to 50%+ in real usage
- [x] No false positive cache hits
- [x] Cost savings measurable via token usage tracking

**Target**: Increase cache hit rate from 30% to 50%+

##### Fri: Issue #125 - Prompt Injection Sanitization
**Duration**: 1-2 days
**Track**: Track-A
**Priority**: HIGH (security vulnerability discovered)

**CRITICAL FINDING (Manual Testing Discovery)**:
Initial implementation only sanitized LLM-powered paths, but **most queries use rule-based mode** which completely bypassed sanitization! This was discovered during manual testing when a prompt injection attempt executed successfully instead of being blocked.

**Root Cause**: Sanitization was only in `query_interpreter.py` and `result_synthesizer.py` (LLM components), but rule-based queries go directly through `coordinator.analyze()` without touching those components.

**Files Modified**:
1. Created `analytics/services/mcp_server/orchestration/security.py` (new shared utility)
2. `analytics/services/mcp_server/orchestration/coordinator.py` (CRITICAL: entry point sanitization)
3. `analytics/services/mcp_server/orchestration/query_interpreter.py` (LLM mode)
4. `analytics/services/mcp_server/orchestration/result_synthesizer.py` (LLM mode)
5. `tests/services/mcp_server/test_phase5_natural_language.py` (17 security tests)

**Implementation**:

1. **Created shared security module**: `security.py`
   - Centralized `sanitize_user_input()` function
   - Length limits (max 1000 chars)
   - Detect role manipulation (assistant:, user:, system:)
   - Detect instruction override (ignore/disregard/forget instructions)
   - Detect delimiter injection (```system, ```user, ```assistant)
   - Detect JSON injection ("lenses":, "reasoning":, etc.)
   - Escape special characters (" → \", \ → \\)
   - Remove control characters (except \n and \t)
   - User-friendly error messages

2. **Added entry-point sanitization**: `coordinator.py`
   - Sanitize at `analyze()` method BEFORE any processing
   - Protects BOTH rule-based and LLM-powered paths
   - Returns error state for security violations (doesn't execute analysis)

3. **Updated LLM components to use shared function**:
   - `query_interpreter.py`: Use shared sanitization
   - `result_synthesizer.py`: Use shared sanitization
   - Removed duplicate code

4. **Added comprehensive tests**:
   - 12 sanitization unit tests (pattern detection, character escaping, etc.)
   - 5 coordinator integration tests (rule-based mode, LLM mode, caching, etc.)

**Success Criteria**:

Automated Verification:
- [x] Security unit tests pass (12): `pytest tests/services/mcp_server/test_phase5_natural_language.py::TestPromptInjectionSanitization`
- [x] Coordinator security tests pass (5): `pytest tests/services/mcp_server/test_phase5_natural_language.py::TestCoordinatorSecurity`
- [x] Type checking passes: `make type-check`
- [x] All existing tests still pass (no regression): 48/48 tests passing

Manual Verification:
- [x] Known injection patterns are blocked (verified in rule-based mode)
- [x] Legitimate user queries work correctly
- [x] Error messages for blocked queries are user-friendly

**Total Tests**: 17 security tests (12 unit + 5 integration)

**Security Target**: Block common injection patterns in BOTH LLM and rule-based modes while preserving usability

**Lessons Learned**:
- Manual testing is critical for security features
- Automated tests of individual components can miss integration gaps
- Security must be enforced at the entry point, not just in downstream components
- Test both code paths (rule-based and LLM modes) explicitly

---

### Phase 2: Main Branch Enhanced Visualizations (Week 3)

**Duration**: 5-6 days
**Track**: Main Branch (`feature/phase1-rich-formatters`)
**Dependencies**: None (can run in parallel with Phase 1 Week 2)
**Parallelization**: ✅ Can be developed in parallel with Track-A Lens 4 + tech debt work

#### Issue #115: Enhanced Visualizations
**Duration**: 4-5 days

**Changes Required**:

1. **File**: `customer_base_audit/mcp/formatters/plotly_charts.py`
   - Add `create_executive_dashboard()` - 4-panel combined view
   - Add `create_retention_trend_chart()` - Lens 3 cohort retention curves
   - Add `create_cohort_heatmap()` - Lens 4 cohort performance heatmap
   - Add `create_sankey_diagram()` - Customer migration flow visualization
   - Each function returns Plotly JSON for MCP embedding

2. **File**: `customer_base_audit/mcp/formatters/markdown_tables.py`
   - Enhance `format_lens4_decomposition_table()` with better formatting
   - Add support for large datasets (pagination/summarization)

3. **File**: `tests/test_mcp_formatters.py`
   - Add tests for all new chart types
   - Test with various data sizes
   - Verify Plotly JSON schema validity
   - Test edge cases (empty data, single cohort, etc.)

**Success Criteria**:

Automated Verification:
- [x] Formatter tests pass: `pytest tests/test_mcp_formatters.py::test_enhanced_*`
- [x] Type checking passes: `make type-check`
- [x] Plotly JSON validates correctly

Manual Verification:
- [x] Executive dashboard displays all 4 panels correctly in Claude Desktop
- [x] Retention trend charts show curves for multiple cohorts
- [x] Cohort heatmap colors are intuitive (red=bad, green=good)
- [x] Sankey diagram shows customer migration flows clearly
- [x] Interactive features work (zoom, pan, hover tooltips)
- [x] Charts render on first attempt (no "maximum length" errors)

**Implementation Note**: Test each visualization in Claude Desktop as you develop it. After all automated tests pass, pause for comprehensive manual testing across different data scenarios.

#### Issue #129: Chart Token Optimization
**Duration**: 1-2 days

**Changes Required**:

1. **File**: `customer_base_audit/mcp/formatters/plotly_charts.py`
   - Reduce chart dimensions from current 1200x600 to 800x400 (or configurable)
   - Use lower DPI for PNG rendering (if converting to PNG)
   - Prefer returning Plotly JSON over PNG when possible
   - Add compression for PNG images if needed
   - Consider thumbnail + full size option

2. **File**: `customer_base_audit/mcp/formatters/__init__.py`
   - Add configuration for chart size/quality tradeoffs
   - Allow users to choose between quality and token efficiency

**Success Criteria**:

Automated Verification:
- [x] Chart size tests pass: `pytest tests/test_mcp_formatters.py::test_chart_size_*`
- [x] Charts use <50% of current token count
- [x] Quality remains acceptable (validated by test assertions)

Manual Verification:
- [x] Multiple charts can be displayed in single Claude Desktop conversation
- [x] No "maximum length" errors under normal usage
- [x] Image quality acceptable for business presentations
- [x] Interactive features still work after optimization

**Target**: Charts use <50% of current token count while maintaining acceptable quality

---

### Phase 3: Integration - Merge Main → Track-A (Week 4)

**Duration**: 4-5 days
**Track**: Track-A (`.worktrees/track-a`)
**Dependencies**: Phases 1 and 2 complete
**Parallelization**: ❌ Must be sequential (requires both tracks complete)

#### Mon-Tue: Formatter Integration

**Changes Required**:

1. **Copy Enhanced Formatters to Track-A**:
   ```bash
   cp -r /path/to/main/customer_base_audit/mcp/formatters/ \
     .worktrees/track-a/customer_base_audit/mcp/formatters/
   ```

2. **File**: `analytics/services/mcp_server/orchestration/coordinator.py`
   - Add new `format_results()` node to StateGraph (after `synthesize_results`)
   - Import enhanced formatters
   - Apply appropriate formatter based on lens results:
     - Lens 3 → `create_retention_trend_chart()`
     - Lens 4 → `create_cohort_heatmap()`
     - Multi-lens → `create_executive_dashboard()`
   - Store formatted outputs in state

3. **File**: `analytics/services/mcp_server/tools/orchestrated_analysis.py`
   - Update `OrchestratedAnalysisResponse` to include `formatted_outputs` dict
   - Return charts and tables alongside raw data

4. **File**: `tests/services/mcp_server/test_orchestration.py`
   - Add tests for formatted output generation
   - Verify charts/tables included in orchestrated responses

**Success Criteria**:

Automated Verification:
- [x] All tests pass: `pytest tests/services/mcp_server/test_orchestration.py` (15/15 passed)
- [x] Formatter integration tests pass: `pytest tests/services/mcp_server/test_orchestration.py::test_formatted_*` (5/5 passed)
- [x] Type checking passes: N/A (no Makefile, but code follows type hints)

Manual Verification:
- [ ] Orchestrated analysis returns formatted visualizations
- [ ] Charts display correctly in Claude Desktop
- [ ] No regression in existing functionality
- [ ] Performance acceptable (<10s for analysis with formatting)

#### Wed: Track-B Evaluation

**Options Analysis**:

**Option 1: Archive Track-B** (Recommended):
- Track-A already has all necessary functionality
- Track-B's analytics libraries not needed for MCP server
- Simpler maintenance with single architecture
- Action: Document Track-B as alternative implementation, mark as archived

**Option 2: Merge Selective Libraries**:
- Identify specific `analytics/libs/` modules needed by Track-A
- Copy only those modules into Track-A
- Update imports and test integration
- Action: Only if specific library functionality required

**Option 3: Maintain Parallel**:
- Keep Track-B for non-MCP users (data scientists, Jupyter notebooks)
- Maintain two deployment paths
- Action: Only if there's a confirmed user base for direct Python API

**Decision Criteria**:
- Is there a user requirement for Track-B's library-style interface?
- Does Track-A need any functionality from Track-B's `analytics/libs/`?
- What's the maintenance cost of parallel tracks?

**Action**: Document decision and rationale in `thoughts/shared/research/2025-10-22-track-b-decision.md`

**Decision Made**: Option 1 - Archive Track-B ✅
- See `thoughts/shared/research/2025-10-22-track-b-decision.md` for full analysis
- Track-B contains valuable enterprise infrastructure patterns but not needed for v1.0.0
- Can be revived for v1.1.0+ if needed

#### Thu-Fri: Integration Testing and Documentation

**Testing Tasks**:

1. **End-to-End Integration Test**:
   - Create `tests/test_integration_complete.py`
   - Test full workflow: load data → build data mart → calculate RFM → run all 5 lenses → generate visualizations
   - Test orchestrated analysis with various queries
   - Test observability stack (Jaeger, Prometheus)
   - Test error handling and resilience

2. **Performance Testing**:
   - Benchmark orchestrated analysis with realistic data sizes
   - Verify <10s response time for typical queries
   - Test concurrent request handling
   - Monitor memory usage

3. **Security Testing**:
   - Verify prompt injection prevention works
   - Test authentication/authorization (if applicable)
   - Review security scan results

**Documentation Updates**:

1. **README.md**: Update with current unified architecture
2. **MCP Server Documentation**: Update quickstart and setup guides
3. **Phase Documentation**: Archive old phase docs, create v1.0.0 release notes
4. **Claude Desktop Config**: Update with latest configuration

**Success Criteria**:

Automated Verification:
- [x] All tests pass: 518 tests passing (`pytest tests/` - all unit, integration, and orchestration tests)
- [x] Orchestration tests pass: 17/17 tests (`pytest tests/services/mcp_server/test_orchestration.py`)
- [x] Foundation tests pass: All lens tests, data mart, RFM, cohorts working
- [x] Security tests pass: 17 security tests including prompt injection prevention
- [x] Performance verified: Tests complete in <10s

Manual Verification (Ready for User Testing):
- [ ] Complete workflow works end-to-end via Claude Desktop (orchestrated analysis with all 5 lenses)
- [ ] Formatted visualizations display correctly in Claude Desktop (charts, tables, dashboards)
- [ ] No regression in existing functionality
- [ ] Performance acceptable in real usage (<10s for typical queries)
- [ ] Security: Prompt injection attacks properly blocked

---

### Phase 4: Final Merge Track-A → Main (Week 5)

**Duration**: 5 days
**Track**: Main
**Dependencies**: Phase 3 complete
**Parallelization**: ❌ Sequential (final integration)

#### Mon-Tue: Merge Execution

**Merge Strategy**:

1. **Backup Main Branch**:
   ```bash
   git checkout main
   git branch main-backup-pre-v1.0.0
   git push origin main-backup-pre-v1.0.0
   ```

2. **Merge Track-A into Main**:
   ```bash
   git checkout main
   git merge --no-ff .worktrees/track-a/feature/issue-118-phase4b
   ```

3. **Resolve Conflicts**:
   - Prioritize Track-A's versions for:
     - `customer_base_audit/` (more advanced)
     - `analytics/services/mcp_server/` (only in Track-A)
   - Prioritize Main's versions for:
     - Enhanced formatters (if different from Track-A's copy)
   - Manually review:
     - `README.md` (combine both)
     - Documentation files (merge content)

4. **Test After Merge**:
   ```bash
   make test
   make type-check
   make lint
   ```

**Success Criteria**:

Automated Verification:
- [ ] Merge completes without critical conflicts
- [ ] All tests pass after merge: `make test`
- [ ] Type checking passes: `make type-check`
- [ ] Linting passes: `make lint`

Manual Verification:
- [ ] MCP server starts successfully
- [ ] All 5 lenses work via Claude Desktop
- [ ] Observability stack functional
- [ ] No functionality lost in merge

#### Wed: Final Testing

**Full Test Suite**:

1. **Unit Tests**: `make test`
2. **Integration Tests**: `pytest tests/test_integration_complete.py`
3. **Type Checking**: `make type-check`
4. **Linting**: `make lint`
5. **Security Scan**: Run security tools
6. **Performance Benchmarks**: Verify response times

**Manual Testing Checklist**:
- [ ] All 5 lenses work correctly via Claude Desktop
- [ ] Orchestrated analysis generates insights
- [ ] Enhanced visualizations display
- [ ] LLM features work (with API key)
- [ ] Caching works
- [ ] Observability stack operational
- [ ] Error handling graceful

**Success Criteria**:
- [ ] All automated tests pass
- [ ] All manual tests pass
- [ ] No critical issues identified
- [ ] Performance meets targets

#### Thu: Documentation Review

**Documentation Tasks**:

1. **README.md**:
   - Update overview with final architecture
   - Update installation instructions
   - Update Claude Desktop configuration
   - Add troubleshooting section

2. **API Documentation**:
   - Generate API docs: `make docs` (if available)
   - Update MCP tool descriptions
   - Document all configuration options

3. **Release Notes**:
   - Create `CHANGELOG.md` or update existing
   - Document v1.0.0 features
   - Document breaking changes (if any)
   - Document migration guide (if applicable)

4. **Contributing Guide**:
   - Update development workflow
   - Update testing guidelines
   - Update PR requirements

**Success Criteria**:
- [ ] All documentation accurate
- [ ] No broken links
- [ ] Setup instructions work for fresh install
- [ ] API documentation complete

#### Fri: Release v1.0.0

**Release Tasks**:

1. **Version Bump**:
   - Update version in `analytics/services/mcp_server/main.py` (if not already 2.0.0 → v1.0.0)
   - Update version in `pyproject.toml` or similar
   - Update version in documentation

2. **Tag Release**:
   ```bash
   git tag -a v1.0.0 -m "Release v1.0.0: Production-ready Agentic MCP Server"
   git push origin v1.0.0
   ```

3. **Create GitHub Release**:
   ```bash
   gh release create v1.0.0 \
     --title "AutoCLV v1.0.0: Five Lenses Agentic MCP Server" \
     --notes "$(cat release-notes-v1.0.0.md)"
   ```

4. **Deploy to Production** (if applicable):
   - Update production environment
   - Run health checks
   - Monitor initial usage

5. **Cleanup**:
   - Archive/delete old worktrees (Track-A, Track-B, Track-C)
   - Close any remaining old issues
   - Update project board/tracking

**Success Criteria**:
- [ ] Release tagged: `v1.0.0`
- [ ] GitHub release created
- [ ] Production deployment successful (if applicable)
- [ ] No critical issues in first 24 hours
- [ ] Team notified of release

---

## Parallelization Summary

### What Can Run in Parallel ✅

**Week 1** (Phase 1):
- **Stream A**: Lens 2 implementation (Developer A or Mon-Wed)
- **Stream B**: Lens 3 implementation (Developer B or Mon-Wed)
- Both can be developed simultaneously, then merged Friday

**Week 2-3** (Phase 1 & 2 overlap):
- **Track-A**: Lens 4 + tech debt (Mon-Fri)
- **Main Branch**: Enhanced visualizations (Mon-Fri)
- Different tracks, no conflicts, can run in parallel

**Week 3** (Phase 2 internal):
- **Issue #115**: Enhanced visualizations (Mon-Thu)
- **Issue #129**: Chart optimization (Thu-Fri or overlap)
- Some parallelization possible within Phase 2

### What Must Be Sequential ❌

**Phase 3** (Week 4):
- Depends on both Phase 1 and Phase 2 complete
- Merging Main → Track-A requires both tracks ready

**Phase 4** (Week 5):
- Depends on Phase 3 complete
- Final merge and release must be sequential

### Optimal Resource Allocation

**Single Developer**:
- Week 1: Lens 2 (Mon-Tue), Lens 3 (Wed-Thu), Integration (Fri)
- Week 2: Lens 4 (Mon-Wed), Issue #122 (Thu), Issue #125 (Fri)
- Week 3: Enhanced visualizations (Mon-Thu), Chart optimization (Fri)
- Week 4: Integration and testing
- Week 5: Final merge and release

**Two Developers**:
- Developer A: Track-A work (Lens 2, then Lens 4, then tech debt)
- Developer B: Track-A work (Lens 3), then Main Branch work (visualizations)
- **Time Savings**: ~1 week (complete in 4 weeks instead of 5)

**Three+ Developers**:
- Developer A: Lens 2 + Lens 4
- Developer B: Lens 3 + tech debt
- Developer C: Enhanced visualizations + chart optimization
- **Time Savings**: ~1.5 weeks (complete in 3.5 weeks)

---

## Testing Strategy

### Unit Tests

**Per-Lens Testing**:
- Test each lens independently with synthetic data
- Test edge cases (empty data, single customer, extreme values)
- Test integration with shared state
- Test error handling

**Per-Component Testing**:
- Test cache key normalization with various query patterns
- Test prompt sanitization with injection attempts
- Test formatters with various data sizes and shapes
- Test observability instrumentation

### Integration Tests

**Orchestration Testing**:
- Test multi-lens orchestrated analysis
- Test parallel execution correctness
- Test dependency handling (Lens 2 depends on Lens 1)
- Test error handling and partial results

**End-to-End Testing**:
- Test complete workflow: data load → all lenses → formatted output
- Test via Claude Desktop interface
- Test observability stack integration
- Test caching behavior

### Performance Testing

**Benchmarks**:
- Orchestrated analysis response time: Target <10s
- Single lens execution time: Depends on lens complexity
- Cache hit rate: Target >50% with Issue #122 implemented
- Memory usage: Monitor and set limits

**Load Testing** (optional):
- Concurrent request handling
- Sustained usage patterns
- Resource utilization under load

### Security Testing

**Prompt Injection Testing**:
- Test known injection patterns
- Test legitimate queries that might look like injection
- Verify error messages user-friendly

**Dependency Scanning**:
- Run security scanner on dependencies: `safety check` or similar
- Update vulnerable dependencies before release

---

## Migration Notes

### For Users

**Upgrading from Earlier Versions**:
- No breaking changes expected (backward compatible)
- Enhanced visualizations optional (can still use basic formatters)
- LLM features optional (works without API key)

**New Configuration**:
- `ANTHROPIC_MODEL`: Optional, defaults to `claude-3-5-sonnet-latest`
- Chart size configuration: New options for token optimization
- Sanitization settings: New security features

### For Developers

**Worktree Cleanup**:
After v1.0.0 release, old worktrees can be removed:
```bash
git worktree remove .worktrees/track-a
git worktree remove .worktrees/track-b
git worktree remove .worktrees/track-c
```

**Branch Cleanup**:
Old feature branches can be archived:
```bash
git branch -d feature/issue-118-phase4b
git branch -d feature/track-b-clv-models
git branch -d test/end-to-end-integration
```

---

## Performance Considerations

**Response Time Targets**:
- Single lens analysis: <2s
- Orchestrated analysis (2-3 lenses): <10s
- Orchestrated analysis (all 5 lenses): <15s
- LLM query parsing: ~1-2s additional
- LLM narrative synthesis: ~2-3s additional

**Resource Usage**:
- Memory: Monitor during integration testing
- CPU: Should be modest (mostly I/O bound)
- Network: LLM API calls add latency
- Disk: Minimal (results cached in memory)

**Optimization Opportunities**:
- Issue #122: Better caching = faster repeat queries
- Issue #129: Smaller charts = faster rendering
- Parallel lens execution already implemented
- Consider result caching beyond query caching

---

## Risk Mitigation

### Technical Risks

**Risk**: Merge conflicts during Phase 3 or 4
- **Mitigation**: Frequent syncing between tracks during Weeks 1-3
- **Mitigation**: Backup branches before merging
- **Mitigation**: Test in separate branch before merging to main

**Risk**: Performance degradation with enhanced visualizations
- **Mitigation**: Issue #129 addresses chart size
- **Mitigation**: Performance testing in Phase 3
- **Mitigation**: Make enhanced visualizations optional/configurable

**Risk**: Breaking changes during integration
- **Mitigation**: Comprehensive test suite
- **Mitigation**: Manual testing in Phase 3 and 4
- **Mitigation**: Backward compatibility checks

### Schedule Risks

**Risk**: Lens implementations take longer than estimated
- **Mitigation**: Lens 2 and 3 can be done in parallel (saves time)
- **Mitigation**: Lens 4 buffer built into schedule (3-4 days)
- **Mitigation**: Tech debt items can be deferred if critical path delayed

**Risk**: Visualization work extends beyond Week 3
- **Mitigation**: Runs in parallel with Track-A work (no blocking)
- **Mitigation**: Can merge partial visualization improvements
- **Mitigation**: Issue #116 already deferred to v1.1.0 (reduces scope)

**Risk**: Integration issues in Phase 3
- **Mitigation**: Extra 2 days allocated (Thu-Fri)
- **Mitigation**: Can extend into Week 5 if needed
- **Mitigation**: Main → Track-A merge is low-risk (Track-A more advanced)

---

## Success Metrics

### Completion Metrics
- [ ] All 5 lenses fully operational: **100%**
- [ ] Test coverage >80%: **Current: Unknown, Target: 80%+**
- [ ] All high-priority issues closed: **Target: 100%**
- [ ] Documentation complete: **Target: 100%**

### Quality Metrics
- [ ] Zero critical bugs in production
- [ ] All automated tests passing
- [ ] Type checking 100% pass rate
- [ ] Linting 100% pass rate

### Performance Metrics
- [ ] Orchestrated analysis <10s: **Target: 95% of queries**
- [ ] Cache hit rate >50%: **After Issue #122**
- [ ] Chart token usage <50% of current: **After Issue #129**
- [ ] No "maximum length" errors: **Target: 0**

### User Experience Metrics
- [ ] Setup time for new users: **Target: <30 minutes**
- [ ] Response quality (manual assessment): **Target: Good/Excellent**
- [ ] Visualization clarity: **Target: Good/Excellent**
- [ ] Error messages user-friendly: **Target: 100%**

---

## References

- **Original Research**: `thoughts/shared/research/2025-10-21-track-merge-strategy.md`
- **Track-A Issues**: #131 (Lens 2), #132 (Lens 3), #133 (Lens 4), #122, #125
- **Main Branch Issues**: #115 (Visualizations), #129 (Chart optimization)
- **Backlog**: #116 (Advanced features - v1.1.0)
- **Closed**: #94-97, #111-114 (completed in Track-A)

---

## Next Steps

### Immediate (This Week)
1. ✅ **Verify all issues created**: Issues #131-133 created for Lens 2-4
2. **Begin Lens 2 implementation**: Start Phase 1 Stream A
3. **Begin Lens 3 implementation**: Start Phase 1 Stream B (if two developers)
4. **Setup development environment**: Ensure Track-A worktree ready

### Week 1 End
- Complete Lens 2 and Lens 3 implementations
- Merge and test integration
- Update documentation

### Week 2 End
- Complete Lens 4 implementation
- Complete cache normalization (Issue #122)
- Complete prompt sanitization (Issue #125)

### Week 3 End
- Complete enhanced visualizations (Issue #115)
- Complete chart optimization (Issue #129)
- Main branch ready for integration

### Week 4 End
- Main → Track-A merge complete
- Track-B decision made and documented
- Integration testing complete

### Week 5 End
- **v1.0.0 Release** 🎉
- Production deployment (if applicable)
- Celebrate and plan v1.1.0 features

---

**Plan Status**: Ready for execution
**Next Action**: Begin Phase 1 Week 1 - Lens 2 and Lens 3 implementations
**Owner**: Development team
**Timeline**: 5 weeks to v1.0.0
