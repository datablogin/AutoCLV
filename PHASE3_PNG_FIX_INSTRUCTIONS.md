# Phase 3 PNG Conversion Fix - Compact Instructions

**Issue**: Claude Desktop cannot render Plotly JSON. Customers see code instead of charts.
**Solution**: Convert Plotly JSON → PNG images using kaleido (already installed)
**Timeline**: 2-3 hours
**Branch**: feature/lens2-lens3-implementation (track-a worktree)

---

## Current State ✅

**What's Working**:
- Formatter integration complete in `coordinator.py:1328-1443`
- `_format_results()` node generates Plotly JSON
- Original metrics stored: `lens2_metrics`, `lens3_metrics`
- All 15 orchestration tests passing
- Response includes `formatted_outputs` field

**What's NOT Working**:
- Plotly JSON doesn't render in Claude Desktop
- Customers see iframes/code, not charts ❌

---

## Implementation Plan

### 1. Add PNG Conversion Helper (15 min)

**File**: `customer_base_audit/mcp/formatters/plotly_charts.py`

Add after line 60:
```python
import plotly.graph_objects as go
import base64
from io import BytesIO

def _convert_plotly_to_base64_png(fig_dict: dict[str, Any]) -> str:
    """Convert Plotly JSON to base64-encoded PNG.

    Uses kaleido to render static image for Claude Desktop display.
    """
    fig = go.Figure(fig_dict)
    img_bytes = fig.to_image(format="png", engine="kaleido")
    return base64.b64encode(img_bytes).decode('utf-8')
```

### 2. Update Chart Functions (30 min)

Modify these functions to return both JSON and PNG:

**Functions to update**:
- `create_retention_trend_chart()` (line 62)
- `create_sankey_diagram()` (line 620)
- `create_cohort_heatmap()` (line 488)
- `create_executive_dashboard()` (line 286)

**Pattern**:
```python
def create_retention_trend_chart(metrics: Lens3Metrics) -> dict[str, Any]:
    # ... existing code to build fig_dict ...

    result = {
        "plotly_json": fig_dict,  # Keep JSON for programmatic access
        "image_base64": _convert_plotly_to_base64_png(fig_dict),
        "format": "png",
        "width": width,
        "height": height
    }
    return result
```

### 3. Update Coordinator Format Node (15 min)

**File**: `analytics/services/mcp_server/orchestration/coordinator.py`

No changes needed! Already calls formatters correctly.
The formatter return value changes will automatically propagate.

### 4. Update Response Model (10 min)

**File**: `analytics/services/mcp_server/tools/orchestrated_analysis.py`

Update docstring for `formatted_outputs` field (line 80):
```python
# Formatted outputs (Phase 3) - PNG images + Plotly JSON
formatted_outputs: dict[str, Any] | None = None  # Charts as base64 PNG + JSON
```

### 5. Add Tests (45 min)

**File**: `tests/services/mcp_server/test_orchestration.py`

Update existing tests (lines 278-528) to check for both formats:

```python
# Check formatted_outputs
formatted_outputs = result.get("formatted_outputs", {})

if "lens2" in result.get("lenses_executed", []):
    assert "lens2_sankey" in formatted_outputs
    sankey = formatted_outputs["lens2_sankey"]

    # Verify dual format
    assert "plotly_json" in sankey
    assert "image_base64" in sankey
    assert sankey["format"] == "png"
    assert isinstance(sankey["image_base64"], str)
    assert len(sankey["image_base64"]) > 1000  # Non-trivial PNG
```

### 6. Test Image Display (30 min)

Create test file to verify PNG generation:

**File**: `test_png_conversion.py` (root)
```python
#!/usr/bin/env python3
"""Quick test to verify PNG conversion works."""
import base64
from customer_base_audit.mcp.formatters.plotly_charts import (
    _convert_plotly_to_base64_png
)

# Simple test figure
test_fig = {
    "data": [{"type": "bar", "x": [1, 2, 3], "y": [4, 5, 6]}],
    "layout": {"title": "Test", "width": 800, "height": 400}
}

try:
    png_b64 = _convert_plotly_to_base64_png(test_fig)
    print(f"✅ PNG generated: {len(png_b64)} chars")

    # Write to file for visual inspection
    png_bytes = base64.b64decode(png_b64)
    with open("test_chart.png", "wb") as f:
        f.write(png_bytes)
    print("✅ Saved to test_chart.png - check visually")
except Exception as e:
    print(f"❌ Error: {e}")
```

Run: `python test_png_conversion.py`

---

## Testing Checklist

After implementation:

```bash
# 1. Unit tests pass
pytest tests/services/mcp_server/test_orchestration.py -v

# 2. PNG conversion works
python test_png_conversion.py
# Opens test_chart.png - verify it's a valid chart image

# 3. Full orchestration test
pytest tests/services/mcp_server/test_orchestration.py::test_formatted_outputs_lens2 -v
```

**Manual Test** (Claude Desktop):
```
Run lens2 and show me the formatted_outputs field.
```

Expected response should include:
```json
{
  "formatted_outputs": {
    "lens2_sankey": {
      "image_base64": "iVBORw0KGgoAAAANSU...(long base64 string)",
      "plotly_json": {...},
      "format": "png",
      "width": 800,
      "height": 400
    }
  }
}
```

Claude should display the PNG image inline, not code.

---

## Edge Cases

1. **Kaleido not installed**: Already in pyproject.toml:26 ✅
2. **Import error**: Add try/except around kaleido import, fall back to JSON only
3. **Large images**: 800x400px default = ~50-100KB PNG (acceptable)
4. **Token limit**: Base64 adds ~33% overhead, but within budget

---

## Rollback Plan

If PNG conversion causes issues:

1. Remove `_convert_plotly_to_base64_png()` calls
2. Return only `plotly_json` (original behavior)
3. Document as known limitation

---

## Files Modified Summary

1. `customer_base_audit/mcp/formatters/plotly_charts.py` - Add conversion helper + update 4 chart functions
2. `analytics/services/mcp_server/tools/orchestrated_analysis.py` - Update docstring
3. `tests/services/mcp_server/test_orchestration.py` - Update test assertions
4. `test_png_conversion.py` - New test file (delete after verification)

---

## Success Criteria

✅ Tests pass (15/15)
✅ PNG images generated and valid
✅ Claude Desktop displays images inline (not code)
✅ Performance <10s for multi-lens analysis
✅ Image quality acceptable for customers

**Estimated Time**: 2-3 hours total

---

## Troubleshooting

**Error: "kaleido not found"**
```bash
pip install kaleido==0.2.1
```

**Error: "No module named 'plotly.graph_objects'"**
Already installed, check import path

**Images too large (>200KB)**
Reduce dimensions in ChartConfig or use lower DPI

**Images too small/blurry**
Increase dimensions (may impact tokens)

---

END OF INSTRUCTIONS
