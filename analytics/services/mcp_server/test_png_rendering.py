"""Manual test script to verify PNG rendering for all chart types.

Run this script to verify that all Plotly charts can be rendered to PNG using kaleido.

Usage:
    cd /Users/robertwelborn/PycharmProjects/AutoCLV/.worktrees/track-a
    python analytics/services/mcp_server/test_png_rendering.py
"""

from datetime import datetime, timezone
from decimal import Decimal
import plotly.graph_objects as go

# Import formatters
from analytics.services.mcp_server.formatters import (
    create_retention_trend_chart,
    create_revenue_concentration_pie,
    create_health_score_gauge,
    create_executive_dashboard,
)

# Import metrics classes from main branch (track-a has them)
from customer_base_audit.analyses.lens1 import Lens1Metrics
from customer_base_audit.analyses.lens3 import Lens3Metrics, CohortPeriodMetrics
from customer_base_audit.analyses.lens5 import (
    Lens5Metrics,
    CustomerBaseHealthScore,
    CohortRevenuePeriod,
    CohortRepeatBehavior,
)


def create_sample_lens1() -> Lens1Metrics:
    """Create sample Lens 1 data for testing."""
    return Lens1Metrics(
        total_customers=1250,
        one_time_buyers=450,
        one_time_buyer_pct=Decimal("36.00"),
        total_revenue=Decimal("487500.00"),
        top_10pct_revenue_contribution=Decimal("48.50"),
        top_20pct_revenue_contribution=Decimal("67.20"),
        avg_orders_per_customer=Decimal("3.20"),
        median_customer_value=Decimal("325.00"),
        rfm_distribution={
            "555": 125,
            "554": 150,
            "544": 175,
            "444": 200,
            "333": 250,
            "222": 200,
            "111": 150,
        },
    )


def create_sample_lens3() -> Lens3Metrics:
    """Create sample Lens 3 data for testing."""
    return Lens3Metrics(
        cohort_name="2023-Q4",
        acquisition_date=datetime(2023, 10, 1, tzinfo=timezone.utc),
        cohort_size=500,
        periods=[
            CohortPeriodMetrics(0, 500, 1.0, 1.5, 50.0, 1.5, 50.0, 25000.0),
            CohortPeriodMetrics(1, 400, 0.95, 1.3, 45.0, 1.04, 36.0, 18000.0),
            CohortPeriodMetrics(2, 320, 0.90, 1.2, 42.0, 0.768, 26.88, 13440.0),
            CohortPeriodMetrics(3, 275, 0.88, 1.15, 40.0, 0.6325, 22.0, 11000.0),
            CohortPeriodMetrics(4, 250, 0.86, 1.1, 38.0, 0.55, 19.0, 9500.0),
        ],
    )


def create_sample_lens5() -> Lens5Metrics:
    """Create sample Lens 5 data for testing."""
    health = CustomerBaseHealthScore(
        total_customers=2500,
        total_active_customers=2000,
        overall_retention_rate=Decimal("80.00"),
        cohort_quality_trend="improving",
        revenue_predictability_pct=Decimal("88.00"),
        acquisition_dependence_pct=Decimal("12.00"),
        health_score=Decimal("84.50"),
        health_grade="B",
    )

    revenue_contribs = [
        CohortRevenuePeriod(
            cohort_id="2023-Q1",
            period_start=datetime(2023, 1, 1, tzinfo=timezone.utc),
            total_revenue=Decimal("25000.00"),
            pct_of_period_revenue=Decimal("40.00"),
            active_customers=400,
            avg_revenue_per_customer=Decimal("62.50"),
        ),
        CohortRevenuePeriod(
            cohort_id="2023-Q1",
            period_start=datetime(2023, 2, 1, tzinfo=timezone.utc),
            total_revenue=Decimal("22000.00"),
            pct_of_period_revenue=Decimal("38.00"),
            active_customers=380,
            avg_revenue_per_customer=Decimal("57.89"),
        ),
    ]

    repeat_behavior = [
        CohortRepeatBehavior(
            cohort_id="2023-Q1",
            cohort_size=400,
            one_time_buyers=160,
            repeat_buyers=240,
            repeat_rate=Decimal("60.00"),
            avg_orders_per_repeat_buyer=Decimal("3.50"),
        ),
        CohortRepeatBehavior(
            cohort_id="2023-Q2",
            cohort_size=450,
            one_time_buyers=153,
            repeat_buyers=297,
            repeat_rate=Decimal("66.00"),
            avg_orders_per_repeat_buyer=Decimal("3.80"),
        ),
    ]

    return Lens5Metrics(
        cohort_revenue_contributions=revenue_contribs,
        cohort_repeat_behavior=repeat_behavior,
        health_score=health,
        analysis_start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        analysis_end_date=datetime(2023, 12, 31, tzinfo=timezone.utc),
    )


def test_revenue_pie_rendering():
    """Test Lens 1 revenue concentration pie chart PNG rendering."""
    print("Testing revenue pie chart...")

    lens1_metrics = create_sample_lens1()
    chart_json = create_revenue_concentration_pie(lens1_metrics)
    fig = go.Figure(data=chart_json["data"], layout=chart_json["layout"])
    img_bytes = fig.to_image(format="png", width=800, height=600)

    assert len(img_bytes) > 0, "PNG bytes should not be empty"
    assert img_bytes[:8] == b"\x89PNG\r\n\x1a\n", "Should be valid PNG header"

    print(f"✓ Revenue pie chart rendered successfully ({len(img_bytes)} bytes)")


def test_retention_trend_rendering():
    """Test Lens 3 retention trend chart PNG rendering."""
    print("Testing retention trend chart...")

    lens3_metrics = create_sample_lens3()
    chart_json = create_retention_trend_chart(lens3_metrics)
    fig = go.Figure(data=chart_json["data"], layout=chart_json["layout"])
    img_bytes = fig.to_image(format="png", width=1200, height=600)

    assert len(img_bytes) > 0, "PNG bytes should not be empty"
    assert img_bytes[:8] == b"\x89PNG\r\n\x1a\n", "Should be valid PNG header"

    print(f"✓ Retention trend chart rendered successfully ({len(img_bytes)} bytes)")


def test_health_gauge_rendering():
    """Test Lens 5 health score gauge PNG rendering."""
    print("Testing health score gauge...")

    lens5_metrics = create_sample_lens5()
    chart_json = create_health_score_gauge(lens5_metrics)
    fig = go.Figure(data=chart_json["data"], layout=chart_json["layout"])
    img_bytes = fig.to_image(format="png", width=600, height=400)

    assert len(img_bytes) > 0, "PNG bytes should not be empty"
    assert img_bytes[:8] == b"\x89PNG\r\n\x1a\n", "Should be valid PNG header"

    print(f"✓ Health gauge rendered successfully ({len(img_bytes)} bytes)")


def test_executive_dashboard_rendering():
    """Test multi-lens executive dashboard PNG rendering."""
    print("Testing executive dashboard...")

    lens1_metrics = create_sample_lens1()
    lens5_metrics = create_sample_lens5()

    chart_json = create_executive_dashboard(lens1_metrics, lens5_metrics)
    fig = go.Figure(data=chart_json["data"], layout=chart_json["layout"])
    img_bytes = fig.to_image(format="png", width=1400, height=1000)

    assert len(img_bytes) > 0, "PNG bytes should not be empty"
    assert img_bytes[:8] == b"\x89PNG\r\n\x1a\n", "Should be valid PNG header"

    print(f"✓ Executive dashboard rendered successfully ({len(img_bytes)} bytes)")


if __name__ == "__main__":
    print("Starting PNG rendering tests...\n")

    try:
        test_revenue_pie_rendering()
        test_retention_trend_rendering()
        test_health_gauge_rendering()
        test_executive_dashboard_rendering()
        print("\n✓ All PNG rendering tests passed!")
    except Exception as e:
        print(f"\n✗ PNG rendering test failed: {e}")
        import traceback

        traceback.print_exc()
        raise
