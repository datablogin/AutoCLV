# Manual Testing Instructions for Image Serialization Fix

## Quick Start

The image serialization bug has been **fixed** in the coordinator code. All PNG charts now properly convert from Plotly JSON to FastMCP Image objects.

## How to Test in Claude Desktop

### Prerequisites
Foundation data must be loaded (you mentioned this is already done based on your earlier successful analysis).

### Test Queries

#### Test 1: Comprehensive Health Analysis
```
Run comprehensive customer base health analysis with visualizations
```

**Expected Results:**
- ✅ `lens5_health_gauge`: PNG image of health score gauge
- ✅ `lens5_table`: Markdown table with health metrics
- ✅ `health_summary`: Executive summary with insights

#### Test 2: Revenue Snapshot (if foundation data supports Lens 1)
```
Show me revenue snapshot with visualizations
```

**Expected Results:**
- ✅ `lens1_revenue_pie`: PNG pie chart of revenue concentration
- ✅ `lens1_table`: Markdown table with revenue metrics

#### Test 3: Multi-Lens Analysis (if supported)
```
Analyze revenue and customer health with visualizations
```

**Expected Results:**
- ✅ `lens1_revenue_pie`: Revenue pie chart (PNG)
- ✅ `lens1_table`: Revenue table (markdown)
- ✅ `lens5_health_gauge`: Health gauge (PNG)
- ✅ `lens5_table`: Health table (markdown)
- ✅ `health_summary`: Health summary (markdown)
- ✅ `executive_dashboard`: Combined dashboard (PNG)

## What Fixed Images Look Like

### Before Fix ❌
```python
formatted_outputs = {
    "lens2_sankey": {  # Dict - cannot serialize!
        "data": [...],
        "layout": {...}
    }
}
```

**Result:** Serialization error, no images in Claude Desktop

### After Fix ✅
```python
formatted_outputs = {
    "lens2_sankey": Image(  # FastMCP Image object
        data=b'\x89PNG\r\n\x1a\n...',  # PNG bytes
        format="png"
    )
}
```

**Result:** Images display inline in Claude Desktop

## Verification Checklist

When testing in Claude Desktop, verify:

- [ ] **PNG Images Display Inline**
  - Images should appear as actual images, not base64 strings
  - Should be viewable directly in the chat

- [ ] **Markdown Tables Render**
  - Tables should have proper column alignment
  - Headers should be bold

- [ ] **Executive Summaries Readable**
  - Should contain insights and recommendations
  - Formatted as markdown text

- [ ] **No Serialization Errors**
  - Check MCP server logs (stderr)
  - Should see: `formatted_outputs_complete` with non-zero count

- [ ] **Performance Acceptable**
  - Single lens: < 10 seconds
  - Multi-lens: < 30 seconds

## Checking MCP Server Logs

If images don't appear, check the MCP server logs:

```bash
# Logs are in stderr, Claude Desktop shows them in developer console
# Look for:
```

**Success Indicators:**
```
[info] generating_formatted_outputs lenses_executed=['lens5']
[debug] lens5_health_gauge_generated size_bytes=45123
[debug] lens5_table_generated length=1234
[debug] health_summary_generated length=2345
[info] formatted_outputs_complete output_count=3 output_keys=['lens5_health_gauge', 'lens5_table', 'health_summary']
```

**Failure Indicators:**
```
[warning] lens5_health_gauge_generation_failed error='...'
[info] formatted_outputs_complete output_count=0 output_keys=[]
```

## Troubleshooting

### Issue: No images appear
**Check:**
1. Is `include_visualizations=True` in your query?
2. Did lenses execute successfully? (check `lenses_executed` vs `lenses_failed`)
3. Check MCP server logs for PNG generation errors

### Issue: Images are broken/corrupt
**Check:**
1. PNG header validation in logs (should show `size_bytes` for each image)
2. kaleido is installed: `python -c "import kaleido"`

### Issue: Some images work, others don't
**Check:**
1. Which lenses executed successfully?
2. Check logs for specific formatter errors
3. Verify metrics data is complete for that lens

## Code Validation

Automated tests confirm the fix is correct:

```bash
# Run pattern verification
python test_format_results_pattern.py

# Expected output:
# ✓ SUCCESS! All PNG charts use correct Image conversion pattern
#   - No direct dict assignments found
#   - All charts use go.Figure -> to_image -> Image() pattern
```

## What Was Fixed

**Files Modified:**
- `analytics/services/mcp_server/orchestration/coordinator.py`

**Locations Fixed:**
1. Line 1570: `lens2_sankey` - Sankey diagram
2. Line 1598: `lens3_retention_trend` - Retention trend chart
3. Line 1625: `lens4_heatmap` - Cohort heatmap
4. Line 1710: `executive_dashboard` - Multi-lens dashboard

**Pattern Applied:**
```python
chart_json = create_chart_formatter(metrics)
fig = go.Figure(data=chart_json["data"], layout=chart_json["layout"])
img_bytes = fig.to_image(format="png", width=W, height=H)
formatted_outputs["chart_name"] = Image(data=img_bytes, format="png")
```

## Success Criteria

For Phase 4 to be complete:
- [x] Automated verification passed (pattern test)
- [ ] Manual test 1: Health analysis with visualizations works
- [ ] Manual test 2: Images display inline in Claude Desktop
- [ ] Manual test 3: No serialization errors
- [ ] Performance acceptable (< 30s for multi-lens)

## Next Steps After Successful Testing

Once manual testing confirms images work:
1. Mark Phase 4 as complete in the plan
2. Proceed to Phase 5: Merge track-a back to main branch
3. Clean up test files (optional)

---

**Created:** 2025-11-01
**Fix Status:** ✅ Applied and verified at code level
**Testing Status:** ⏳ Awaiting manual verification in Claude Desktop
