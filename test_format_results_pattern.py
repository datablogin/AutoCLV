#!/usr/bin/env python
"""Test that _format_results() uses correct PNG conversion pattern.

This test verifies that the coordinator code correctly converts Plotly JSON
to PNG Images for all chart types.
"""

import re
import sys


def analyze_format_results_method():
    """Analyze the _format_results() method to verify PNG conversion pattern."""
    print("=" * 80)
    print("Analyzing _format_results() PNG Conversion Pattern")
    print("=" * 80)

    # Read the coordinator file
    with open("analytics/services/mcp_server/orchestration/coordinator.py") as f:
        content = f.read()

    # Find all places where formatted_outputs is assigned
    assignments = re.findall(
        r'formatted_outputs\["([^"]+)"\]\s*=\s*(.+)', content, re.MULTILINE
    )

    print(f"\nFound {len(assignments)} formatted_outputs assignments:\n")

    # Track which ones use the correct pattern
    correct_png_pattern = 0
    incorrect_dict_pattern = 0
    string_assignments = 0
    errors = []

    for key, value in assignments:
        # Check if it's an Image object (correct for PNGs)
        if "Image(data=" in value and 'format="png"' in value:
            correct_png_pattern += 1
            print(f"  ✓ {key}: Correct PNG Image pattern")
        # Check if it's a raw dict assignment (incorrect!)
        elif any(var in value for var in ["_result", "_json"]):
            # If it assigns a variable ending in _result or _json directly without Image()
            if "Image(" not in value and "go.Figure" not in value:
                incorrect_dict_pattern += 1
                errors.append(f"{key}: Assigns dict directly (should be Image object)")
                print(f"  ✗ {key}: ERROR - Assigns dict directly!")
                print(f"      Pattern: {value[:80]}")
            else:
                correct_png_pattern += 1
                print(f"  ✓ {key}: Correct PNG Image pattern")
        # String assignments (markdown tables, summaries)
        elif any(
            var in value
            for var in ["table_md", "summary_md", "insights_md", "comparison_md"]
        ):
            string_assignments += 1
            print(f"  ✓ {key}: String (table/summary)")
        else:
            # Unknown pattern
            if "Image(" in value:
                correct_png_pattern += 1
                print(f"  ✓ {key}: Image object")
            else:
                print(f"  ? {key}: Unknown pattern - {value[:50]}")

    # Check for specific problematic patterns from the bug
    problematic_patterns = [
        (
            r'formatted_outputs\["lens2_sankey"\]\s*=\s*sankey_result(?!\s*#|\n)',
            "lens2_sankey: Direct dict assignment",
        ),
        (
            r'formatted_outputs\["lens3_retention_trend"\]\s*=\s*retention_chart_result(?!\s*#|\n)',
            "lens3_retention_trend: Direct dict assignment",
        ),
        (
            r'formatted_outputs\["lens4_heatmap"\]\s*=\s*heatmap_result(?!\s*#|\n)',
            "lens4_heatmap: Direct dict assignment",
        ),
        (
            r'formatted_outputs\["executive_dashboard"\]\s*=\s*dashboard_result(?!\s*#|\n)',
            "executive_dashboard: Direct dict assignment",
        ),
    ]

    print("\n" + "=" * 80)
    print("Checking for Known Problematic Patterns")
    print("=" * 80)

    found_problems = []
    for pattern, description in problematic_patterns:
        if re.search(pattern, content):
            found_problems.append(description)
            print(f"  ✗ FOUND: {description}")

    if not found_problems:
        print("  ✓ No problematic patterns found!")

    # Verify correct pattern exists for all PNG charts
    required_png_charts = [
        "lens1_revenue_pie",
        "lens2_sankey",
        "lens3_retention_trend",
        "lens4_heatmap",
        "lens5_health_gauge",
        "executive_dashboard",
    ]

    print("\n" + "=" * 80)
    print("Verifying Correct Pattern for Required PNG Charts")
    print("=" * 80)

    missing_correct_pattern = []
    for chart in required_png_charts:
        # Check if the correct pattern exists: go.Figure(...).to_image() -> Image()
        pattern = rf"{chart}.*?go\.Figure.*?to_image.*?Image\("
        if re.search(pattern, content, re.DOTALL):
            print(f"  ✓ {chart}: Uses correct go.Figure -> to_image -> Image pattern")
        else:
            missing_correct_pattern.append(chart)
            print(f"  ✗ {chart}: Missing correct pattern!")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total assignments: {len(assignments)}")
    print(f"  PNG Images (correct): {correct_png_pattern}")
    print(f"  Strings (tables/summaries): {string_assignments}")
    print(f"  Direct dict assignments (incorrect): {incorrect_dict_pattern}")

    success = (
        len(found_problems) == 0
        and len(missing_correct_pattern) == 0
        and incorrect_dict_pattern == 0
    )

    if success:
        print("\n✓ SUCCESS! All PNG charts use correct Image conversion pattern")
        print("  - No direct dict assignments found")
        print("  - All charts use go.Figure -> to_image -> Image() pattern")
        return 0
    else:
        print("\n✗ FAILURE!")
        if found_problems:
            print(f"  - Found {len(found_problems)} problematic pattern(s)")
        if missing_correct_pattern:
            print(
                f"  - {len(missing_correct_pattern)} chart(s) missing correct pattern"
            )
        if incorrect_dict_pattern > 0:
            print(f"  - {incorrect_dict_pattern} direct dict assignment(s)")
        return 1


if __name__ == "__main__":
    sys.exit(analyze_format_results_method())
