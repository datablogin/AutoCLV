# Phase 3 Manual Testing Prompts for Claude Desktop

**Date**: 2025-10-22
**Purpose**: Verify Phase 3 formatter integration via Claude Desktop
**Prerequisites**: MCP server running, data loaded (data mart, RFM, cohorts)

---

## Setup Verification

### 1. Check MCP Server Status
**Prompt**:
```
Can you check the health of the AutoCLV MCP server?
```

**Expected Response**:
- Server status: healthy
- Phase 5 with 16 tools registered
- Foundation tools available

---

## Test 1: Foundation Data Setup

Before testing formatters, ensure data is loaded.

### 1a. Load Sample Data (if needed)
**Prompt**:
```
I need to set up the AutoCLV analysis. Can you:
1. Build the customer data mart from synthetic_transactions.json
2. Calculate RFM metrics for two periods (Q1 2024 and Q2 2024)
3. Create customer cohorts

Use the synthetic transaction data that should be available.
```

**Expected Actions**:
- Calls `build_customer_data_mart` tool
- Calls `calculate_rfm_metrics` for period 1
- Calls `calculate_rfm_metrics` for period 2 (with different date range)
- Calls `create_customer_cohorts` tool

**Success Criteria**:
- All tools execute successfully
- Data mart shows thousands of transactions
- RFM metrics calculated for ~4,000 customers per period
- Multiple cohorts created (2023-Q1, 2023-Q2, etc.)

### 1b. Verify Foundation Data
**Prompt**:
```
Can you verify that the foundation data is ready? Check:
- Is the data mart built?
- Are RFM metrics calculated?
- Are customer cohorts created?
```

**Expected Response**:
- Data mart: ✅ Ready
- RFM metrics: ✅ Ready (period 1 and period 2)
- Cohorts: ✅ Ready

---

## Test 2: Lens 2 Sankey Diagram (Customer Migration)

### 2a. Run Lens 2 Analysis
**Prompt**:
```
Run a Lens 2 period comparison analysis. I want to see customer migration patterns between the two periods.
```

**Expected Response Structure**:
```json
{
  "query": "lens 2...",
  "lenses_executed": ["lens2"],
  "formatted_outputs": {
    "lens2_sankey": {
      "data": [...],  // Plotly Sankey diagram data
      "layout": {...}  // Chart layout
    }
  },
  "lens2_result": {
    "period1_customers": 3997,
    "period2_customers": 4000,
    "retained_customers": 3997,
    "churned_customers": 0,
    "new_customers": 3,
    "retention_rate": 100.0,
    "growth_momentum": "Strong Growth Momentum"
  }
}
```

**What to Verify**:
- ✅ `formatted_outputs` field exists
- ✅ `lens2_sankey` key present in formatted_outputs
- ✅ Sankey diagram has `data` and `layout` fields
- ✅ Diagram shows customer flow: Period 1 → Retained/Churned, New → Period 2
- ✅ Numbers match lens2_result metrics

**Visual Check** (if Claude Desktop renders it):
- Sankey shows 4 nodes: Period 1, Period 2, Retained, Churned, New
- Flow colors are distinct and readable
- Hover tooltips show customer counts

### 2b. Ask About Sankey Visualization
**Prompt**:
```
Can you explain what the Sankey diagram shows? What are the key customer migration insights?
```

**Expected Response**:
- Explanation of customer flows
- Retention/churn patterns
- Growth drivers (new vs retained customers)

---

## Test 3: Lens 3 Retention Trend Chart (Cohort Evolution)

### 3a. Run Lens 3 Analysis
**Prompt**:
```
Analyze the cohort evolution for the first cohort. I want to see retention trends over time.
```

**Expected Response Structure**:
```json
{
  "query": "lens 3...",
  "lenses_executed": ["lens3"],
  "formatted_outputs": {
    "lens3_retention_chart": {
      "data": [...],  // Plotly line chart data
      "layout": {...}  // Chart layout with retention % on y-axis, periods on x-axis
    }
  },
  "lens3_result": {
    "cohort_id": "2023-Q1",
    "cohort_size": 831,
    "periods_analyzed": 18,
    "cohort_maturity": "Mature",
    "ltv_trajectory": "Strong",
    "retention_curve": {
      "0": 100.0,
      "1": 67.3,
      "2": 65.8,
      ...
    }
  }
}
```

**What to Verify**:
- ✅ `formatted_outputs.lens3_retention_chart` exists
- ✅ Chart has retention data for multiple periods
- ✅ Retention curve shows typical decay pattern (high → stabilization)
- ✅ Chart includes activation and revenue curves if available

**Visual Check**:
- Line chart with period number (0-18) on x-axis
- Retention % (0-100%) on y-axis
- Retention curve starts at 100% and decays
- Smooth curve indicating stable cohort behavior

### 3b. Deep Dive on Cohort Health
**Prompt**:
```
What does the retention trend chart tell us about this cohort's health? Is it performing well?
```

**Expected Response**:
- Analysis of retention stability
- LTV trajectory insights ("Strong" = stable revenue)
- Cohort maturity assessment
- Actionable recommendations

---

## Test 4: Multi-Lens Executive Dashboard

### 4a. Run Multi-Lens Analysis
**Prompt**:
```
Give me a comprehensive customer base health analysis. Run Lens 1 and Lens 5 together and show me an executive dashboard.
```

**Expected Response Structure**:
```json
{
  "query": "...",
  "lenses_executed": ["lens1", "lens5"],
  "formatted_outputs": {
    "executive_dashboard": {
      "data": [...],  // 4-panel dashboard with multiple charts
      "layout": {
        "grid": {...},  // Subplot layout
        "title": "Executive Dashboard - Customer Base Health"
      }
    }
  }
}
```

**What to Verify**:
- ✅ Dashboard created when 2+ lenses executed
- ✅ Dashboard includes data from both Lens 1 and Lens 5
- ✅ Multiple panels/subplots visible
- ✅ Each panel has appropriate chart type (bar, pie, gauge, etc.)

**Visual Check**:
- 4 distinct panels in grid layout
- Panel 1: Customer health metrics (Lens 1)
- Panel 2: Health score gauge (Lens 5)
- Panel 3: RFM distribution or concentration risk
- Panel 4: Health grade breakdown

### 4b. Three-Lens Dashboard
**Prompt**:
```
Run a comprehensive analysis with Lens 1, Lens 2, and Lens 5. Show me the executive dashboard with all these perspectives.
```

**Expected Response**:
- Executive dashboard with data from all 3 lenses
- Richer insights combining snapshot, period comparison, and overall health

---

## Test 5: Lens 4 Placeholder Handling

### 5a. Run Lens 4 (Placeholder)
**Prompt**:
```
Run a Lens 4 multi-cohort comparison analysis.
```

**Expected Response**:
- Lens 4 executes (placeholder implementation)
- No formatted outputs for Lens 4 (graceful handling)
- No errors or crashes
- Placeholder data returned with recommendations

**What to Verify**:
- ✅ Analysis completes without errors
- ✅ Lens 4 marked as executed
- ✅ Placeholder message in recommendations
- ✅ No `lens4_heatmap` or `lens4_table` in formatted_outputs (because it's a placeholder)

---

## Test 6: Performance and Error Handling

### 6a. Performance Test
**Prompt**:
```
Run an orchestrated analysis with "lens1 and lens2 and lens3 and lens5". Measure the execution time.
```

**Expected Response**:
- All 4 lenses execute successfully
- `execution_time_ms` < 10,000 (10 seconds)
- Multiple formatted outputs generated
- No timeouts or performance issues

**What to Verify**:
- ✅ Execution time acceptable
- ✅ All formatted outputs included
- ✅ No memory issues
- ✅ Executive dashboard combines all lens data

### 6b. Error Handling (No Data)
**Prompt**:
```
What happens if I try to run lens2 without having two periods of RFM data?
```

**Expected Response**:
- Graceful error message
- Recommendations on how to fix (calculate period 2 RFM)
- No crashes or unhandled exceptions
- Empty formatted_outputs or omitted

---

## Test 7: Natural Language Orchestration

### 7a. Customer Health Question
**Prompt**:
```
How healthy is my customer base? Show me retention trends and growth momentum.
```

**Expected Behavior**:
- Automatically selects appropriate lenses (likely Lens 2, 3, 5)
- Generates formatted visualizations
- Synthesizes insights across lenses
- Returns executive dashboard if multiple lenses run

### 7b. Period Comparison Question
**Prompt**:
```
Compare my customer base between Q1 2024 and Q2 2024. Did we grow or shrink?
```

**Expected Behavior**:
- Triggers Lens 2 (period comparison)
- Returns Sankey diagram showing migration
- Provides growth momentum assessment
- Shows retention/churn metrics

### 7c. Cohort Performance Question
**Prompt**:
```
Show me how my oldest cohort has performed over time.
```

**Expected Behavior**:
- Triggers Lens 3 (cohort evolution)
- Returns retention trend chart
- Analyzes cohort maturity and LTV trajectory
- Provides actionable recommendations

---

## Test 8: Chart Quality and Token Efficiency

### 8a. Chart Size Verification
**Prompt**:
```
Run lens2 and check the formatted output. What are the chart dimensions?
```

**Expected Response**:
- Charts default to 800x400 pixels (medium quality)
- Token-efficient sizing (56% reduction from original 1200x600)
- Charts still readable and professional

**What to Verify**:
- ✅ Chart width: 800px (or configured size)
- ✅ Chart height: 400px (or configured size)
- ✅ Quality setting: "medium" (default)
- ✅ Multiple charts fit in conversation without "maximum length" errors

---

## Success Criteria Summary

Phase 3 manual testing passes if:

- [ ] **All formatted outputs present**: Sankey (Lens 2), retention chart (Lens 3), dashboard (multi-lens)
- [ ] **Charts render correctly**: Valid Plotly JSON, appropriate chart types, readable labels
- [ ] **Data accuracy**: Chart data matches lens result metrics
- [ ] **Performance acceptable**: <10s for typical orchestrated analysis with formatting
- [ ] **Error handling**: Graceful failures, helpful error messages
- [ ] **No regressions**: Existing Lens 1, 2, 3, 5 functionality still works
- [ ] **Token efficiency**: Multiple charts displayable without length errors

---

## Troubleshooting

### Issue: No formatted_outputs in response
**Solution**: Check that Phase 3 integration is complete and format_results node is in graph

### Issue: Charts don't render in Claude Desktop
**Solution**: Verify Plotly JSON structure is valid; check that data/layout fields exist

### Issue: "AttributeError: 'dict' object has no attribute 'migration'"
**Solution**: Ensure original metrics objects (lens2_metrics, lens3_metrics) are stored in state

### Issue: Performance slower than expected
**Solution**: Check data size, reduce chart quality, verify parallel execution working

---

## Quick Test Suite (5 minutes)

For rapid verification:

1. **Foundation check**: `"Are RFM metrics and cohorts ready?"`
2. **Lens 2 Sankey**: `"Run lens2"`
3. **Lens 3 retention**: `"Run lens3"`
4. **Multi-lens dashboard**: `"Run lens1 and lens5"`
5. **Performance**: Check execution times in responses

If all 5 pass, Phase 3 is working correctly.

---

**End of Phase 3 Manual Test Prompts**
