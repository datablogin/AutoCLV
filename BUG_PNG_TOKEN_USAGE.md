# BUG: PNG Images Causing Excessive Token Usage

## Issue Summary
Phase 3 PNG conversion implementation works correctly (images display in Claude Desktop), but causes conversation length limits to be hit after only 3-5 commands, even on Claude Max plan.

## Severity
**CRITICAL** - Makes the feature unusable in practice

## Reproduction
1. Run `lens2 analysis`
2. Run `lens3 analysis`
3. Run `lens4 analysis`
4. **Result**: "Claude has hit the maximum length for this conversation"

## Root Cause Analysis

### Token Usage Breakdown
- Single PNG chart: ~15KB-50KB raw
- Base64 encoding: Adds ~33% overhead → ~20KB-67KB base64
- Multi-chart responses (executive dashboard): Multiple images in single response
- **Cumulative effect**: Each analysis adds 20KB-200KB+ to conversation context

### Why This Happens
1. **Images returned in tool response**: PNG images embedded in JSON response are part of conversation history
2. **Context accumulation**: Claude Desktop keeps full conversation history including all images
3. **Multiple charts per analysis**: Some lenses generate multiple visualizations
4. **No compression**: PNGs at 800x400px default size

## Impact
- ✅ Technical implementation works (images display correctly)
- ❌ Unusable for real workflows (conversation dies after 3-5 commands)
- ❌ Defeats purpose of formatted outputs (can't do multi-step analysis)

## Measured Data
- Lens 3 retention chart: ~20,736 chars base64 (~15KB PNG)
- Sankey diagram: ~55,160 chars base64 (~40KB PNG)
- Executive dashboard: Multiple images → 100KB+ total
- **3 analyses with charts**: ~150KB+ of image data in context

## Potential Solutions

### Option 1: Reduce Image Size (Quick Fix)
**Pros**: Immediate reduction in token usage
**Cons**: Lower quality, may still hit limits with many commands
```python
# Change default from 800x400 to 400x200
ChartConfig(width=400, height=200)  # ~5KB instead of ~15KB
```

### Option 2: Compress PNGs Aggressively
**Pros**: Maintains resolution, reduces file size
**Cons**: May degrade visual quality, still significant overhead
```python
# Use PIL optimization
image.save(buffer, format="PNG", optimize=True, compress_level=9)
```

### Option 3: Make Images Optional (Recommended Short-term)
**Pros**: Users can opt-in when needed, default to lean responses
**Cons**: Requires API change
```python
@mcp.tool()
async def run_orchestrated_analysis(
    query: str,
    include_visualizations: bool = False  # Default to False
)
```

### Option 4: Use MCP Resources Instead of Tool Returns (Best Long-term)
**Pros**: Images not in conversation context, only fetched when needed
**Cons**: Significant refactoring, may not work with current MCP clients
```python
# Store images as MCP resources
@mcp.resource("image://lens2_sankey")
def get_sankey_image() -> bytes:
    return generate_sankey_png()
```

### Option 5: Return Image References, Not Image Data
**Pros**: Minimal token usage, fast responses
**Cons**: Requires external storage, more complex architecture
```python
# Store in temp directory, return file path
formatted_outputs["lens2_sankey"] = {
    "image_url": "file:///tmp/lens2_sankey_abc123.png",
    "plotly_json": {...}
}
```

## Recommended Immediate Action

**Phase 1 (Immediate)**:
1. Make PNG generation **opt-in** via parameter (default: False)
2. Document token usage implications in tool docstring
3. Add size information to response so users know what they're getting

**Phase 2 (Short-term)**:
1. Reduce default image size to 600x300 (60% reduction)
2. Add aggressive PNG compression
3. Only generate images for the "most important" chart per lens

**Phase 3 (Long-term)**:
1. Investigate MCP Resources API for image delivery
2. Consider switching to image references with temp file storage
3. Add configuration to let users control image size/quality tradeoff

## Code Impact
- Files to modify:
  - `analytics/services/mcp_server/tools/orchestrated_analysis.py` (add include_visualizations param)
  - `customer_base_audit/mcp/formatters/plotly_charts.py` (add compression, reduce default size)
  - `analytics/services/mcp_server/orchestration/coordinator.py` (conditional formatting)

## Testing Requirements
1. Verify token usage with different image sizes
2. Test 10+ sequential analyses without hitting limits
3. Measure quality degradation with compression
4. User acceptance testing for image quality

## Priority
**P0** - Blocks production use of Phase 3 formatted outputs feature

## Related Issues
- Phase 3 PNG conversion implementation (completed)
- Issue #137 - PNG visualizations token exhaustion (THIS ISSUE - fixed by PR #138)
- Future: MCP Resources API investigation for image delivery
- Future: Chart size configuration enhancements

## Reported By
User on Claude Max plan - hitting limits after 3-5 commands

## Date Discovered
2025-10-27
