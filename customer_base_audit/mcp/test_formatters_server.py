"""Temporary MCP server for testing formatters with Claude Desktop.

This server provides sample data and demonstrates all formatter capabilities.
Use this to verify that formatters work correctly in Claude Desktop before
integrating into the main orchestration layer.
"""

from datetime import datetime, timezone
from decimal import Decimal

from fastmcp import FastMCP
from fastmcp.utilities.types import Image

from customer_base_audit.analyses.lens1 import Lens1Metrics
from customer_base_audit.analyses.lens2 import Lens2Metrics, CustomerMigration
from customer_base_audit.analyses.lens3 import Lens3Metrics, CohortPeriodMetrics
from customer_base_audit.analyses.lens4 import (
    Lens4Metrics,
    CohortDecomposition,
    TimeToSecondPurchase,
)
from customer_base_audit.analyses.lens5 import (
    Lens5Metrics,
    CustomerBaseHealthScore,
    CohortRepeatBehavior,
)
from customer_base_audit.mcp.formatters import (
    format_lens1_table,
    format_lens2_table,
    format_lens4_decomposition_table,
    format_lens5_health_summary_table,
    create_retention_trend_chart,
    create_revenue_concentration_pie,
    create_health_score_gauge,
    create_executive_dashboard,
    generate_health_summary,
    generate_retention_insights,
    generate_cohort_comparison,
)

# Initialize FastMCP server
mcp = FastMCP("Formatter Test Server", version="0.1.0")


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


def create_sample_lens2() -> Lens2Metrics:
    """Create sample Lens 2 data for testing."""
    p1 = Lens1Metrics(
        1000, 400, Decimal("40.00"), Decimal("400000.00"),
        Decimal("45.00"), Decimal("65.00"), Decimal("3.00"),
        Decimal("400.00"), {}
    )
    p2 = Lens1Metrics(
        1100, 420, Decimal("38.18"), Decimal("462000.00"),
        Decimal("47.50"), Decimal("66.50"), Decimal("3.15"),
        Decimal("420.00"), {}
    )
    migration = CustomerMigration(
        retained=frozenset([f"C{i}" for i in range(750)]),
        churned=frozenset([f"C{i}" for i in range(750, 1000)]),
        new=frozenset([f"C{i}" for i in range(1000, 1350)]),
        reactivated=frozenset([f"C{i}" for i in range(1000, 1050)])
    )
    return Lens2Metrics(
        period1_metrics=p1,
        period2_metrics=p2,
        migration=migration,
        retention_rate=Decimal("75.00"),
        churn_rate=Decimal("25.00"),
        reactivation_rate=Decimal("14.29"),
        customer_count_change=100,
        revenue_change_pct=Decimal("15.50"),
        avg_order_value_change_pct=Decimal("5.00")
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
        ]
    )


def create_sample_lens4() -> Lens4Metrics:
    """Create sample Lens 4 data for testing."""
    decomps = [
        CohortDecomposition(
            cohort_id="2023-Q1",
            period_number=0,
            cohort_size=400,
            active_customers=400,
            pct_active=Decimal("100.00"),
            total_orders=600,
            aof=Decimal("1.50"),
            total_revenue=Decimal("60000.00"),
            aov=Decimal("100.00"),
            margin=Decimal("100.00"),
            revenue=Decimal("60000.00")
        ),
        CohortDecomposition(
            cohort_id="2023-Q2",
            period_number=0,
            cohort_size=450,
            active_customers=450,
            pct_active=Decimal("100.00"),
            total_orders=720,
            aof=Decimal("1.60"),
            total_revenue=Decimal("75600.00"),
            aov=Decimal("105.00"),
            margin=Decimal("100.00"),
            revenue=Decimal("75600.00")
        ),
        CohortDecomposition(
            cohort_id="2023-Q3",
            period_number=0,
            cohort_size=500,
            active_customers=500,
            pct_active=Decimal("100.00"),
            total_orders=850,
            aof=Decimal("1.70"),
            total_revenue=Decimal("93500.00"),
            aov=Decimal("110.00"),
            margin=Decimal("100.00"),
            revenue=Decimal("93500.00")
        ),
    ]

    ttsp = [
        TimeToSecondPurchase(
            cohort_id="2023-Q1",
            customers_with_repeat=240,
            repeat_rate=Decimal("60.00"),
            median_days=Decimal("35.00"),
            mean_days=Decimal("42.00"),
            cumulative_repeat_by_period={}
        ),
        TimeToSecondPurchase(
            cohort_id="2023-Q2",
            customers_with_repeat=297,
            repeat_rate=Decimal("66.00"),
            median_days=Decimal("30.00"),
            mean_days=Decimal("38.00"),
            cumulative_repeat_by_period={}
        ),
        TimeToSecondPurchase(
            cohort_id="2023-Q3",
            customers_with_repeat=350,
            repeat_rate=Decimal("70.00"),
            median_days=Decimal("28.00"),
            mean_days=Decimal("35.00"),
            cumulative_repeat_by_period={}
        ),
    ]

    return Lens4Metrics(
        cohort_decompositions=decomps,
        time_to_second_purchase=ttsp,
        cohort_comparisons=[],
        alignment_type="left-aligned"
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
        health_grade="B"
    )

    cohort_behavior = [
        CohortRepeatBehavior(
            cohort_id="2023-Q1",
            cohort_size=400,
            one_time_buyers=160,
            repeat_buyers=240,
            repeat_rate=Decimal("60.00"),
            avg_orders_per_repeat_buyer=Decimal("4.20")
        ),
        CohortRepeatBehavior(
            cohort_id="2023-Q2",
            cohort_size=450,
            one_time_buyers=153,
            repeat_buyers=297,
            repeat_rate=Decimal("66.00"),
            avg_orders_per_repeat_buyer=Decimal("4.50")
        ),
        CohortRepeatBehavior(
            cohort_id="2023-Q3",
            cohort_size=500,
            one_time_buyers=150,
            repeat_buyers=350,
            repeat_rate=Decimal("70.00"),
            avg_orders_per_repeat_buyer=Decimal("4.80")
        ),
    ]

    return Lens5Metrics(
        cohort_revenue_contributions=[],
        cohort_repeat_behavior=cohort_behavior,
        health_score=health,
        analysis_start_date=datetime(2023, 1, 1, tzinfo=timezone.utc),
        analysis_end_date=datetime(2023, 12, 31, tzinfo=timezone.utc)
    )


@mcp.tool()
def test_markdown_tables() -> str:
    """Test all markdown table formatters with sample data.

    Returns formatted tables for Lens 1, 2, 4, and 5.
    """
    lens1 = create_sample_lens1()
    lens2 = create_sample_lens2()
    lens4 = create_sample_lens4()
    lens5 = create_sample_lens5()

    output = "# Markdown Table Formatters Test\n\n"
    output += format_lens1_table(lens1) + "\n\n---\n\n"
    output += format_lens2_table(lens2) + "\n\n---\n\n"
    output += format_lens4_decomposition_table(lens4) + "\n\n---\n\n"
    output += format_lens5_health_summary_table(lens5)

    return output


@mcp.tool()
def test_plotly_charts() -> str:
    """Test Plotly chart generators with sample data.

    Returns JSON specifications for retention trends, revenue concentration pie,
    health gauge, and executive dashboard.
    """
    import json

    lens1 = create_sample_lens1()
    lens3 = create_sample_lens3()
    lens5 = create_sample_lens5()

    output = "# Plotly Chart Generators Test\n\n"
    output += "## Retention Trend Chart (Lens 3)\n\n"
    output += "```json\n" + json.dumps(create_retention_trend_chart(lens3), indent=2) + "\n```\n\n"
    output += "## Revenue Concentration Pie (Lens 1)\n\n"
    output += "```json\n" + json.dumps(create_revenue_concentration_pie(lens1), indent=2) + "\n```\n\n"
    output += "## Health Score Gauge (Lens 5)\n\n"
    output += "```json\n" + json.dumps(create_health_score_gauge(lens5), indent=2) + "\n```\n\n"
    output += "## Executive Dashboard (Lens 1 + 5)\n\n"
    output += "```json\n" + json.dumps(create_executive_dashboard(lens1, lens5), indent=2) + "\n```\n"

    return output


@mcp.tool()
def test_executive_summaries() -> str:
    """Test executive summary generators with sample data.

    Returns narrative summaries with actionable insights for health,
    retention, and cohort comparison.
    """
    lens2 = create_sample_lens2()
    lens3 = create_sample_lens3()
    lens4 = create_sample_lens4()
    lens5 = create_sample_lens5()

    output = "# Executive Summary Generators Test\n\n"
    output += generate_health_summary(lens5) + "\n\n---\n\n"
    output += generate_retention_insights(lens2, lens3) + "\n\n---\n\n"
    output += generate_cohort_comparison(lens4)

    return output


@mcp.tool()
def test_all_formatters() -> str:
    """Test ALL formatters at once.

    Comprehensive demonstration of all formatting capabilities including
    markdown tables, Plotly charts, and executive summaries.
    """
    import json

    lens1 = create_sample_lens1()
    lens2 = create_sample_lens2()
    lens3 = create_sample_lens3()
    lens4 = create_sample_lens4()
    lens5 = create_sample_lens5()

    output = "# Complete Formatter Test Suite\n\n"
    output += "This demonstrates all formatter capabilities with sample customer data.\n\n"
    output += "---\n\n"

    # Markdown tables
    output += "# Part 1: Markdown Tables\n\n"
    output += format_lens1_table(lens1) + "\n\n"
    output += format_lens2_table(lens2) + "\n\n"
    output += format_lens4_decomposition_table(lens4) + "\n\n"
    output += format_lens5_health_summary_table(lens5) + "\n\n"
    output += "---\n\n"

    # Executive summaries
    output += "# Part 2: Executive Summaries\n\n"
    output += generate_health_summary(lens5) + "\n\n"
    output += generate_retention_insights(lens2, lens3) + "\n\n"
    output += generate_cohort_comparison(lens4) + "\n\n"
    output += "---\n\n"

    # Charts (JSON format)
    output += "# Part 3: Interactive Plotly Charts\n\n"
    output += "## Retention Trend Chart\n\n"
    output += "```json\n" + json.dumps(create_retention_trend_chart(lens3), indent=2) + "\n```\n\n"
    output += "## Revenue Concentration Pie\n\n"
    output += "```json\n" + json.dumps(create_revenue_concentration_pie(lens1), indent=2) + "\n```\n\n"
    output += "## Health Score Gauge\n\n"
    output += "```json\n" + json.dumps(create_health_score_gauge(lens5), indent=2) + "\n```\n"

    return output


@mcp.tool()
def show_retention_chart() -> Image:
    """Show the retention trend chart as a visual image.

    Displays the cohort retention curve with active customer counts.
    """
    import plotly.graph_objects as go

    lens3 = create_sample_lens3()
    chart_json = create_retention_trend_chart(lens3)

    fig = go.Figure(chart_json)

    # Render as PNG image
    img_bytes = fig.to_image(format="png", width=1200, height=600)

    return Image(data=img_bytes, format="png")


@mcp.tool()
def show_revenue_pie_chart() -> Image:
    """Show the revenue concentration pie chart as a visual image.

    Displays Pareto analysis of revenue distribution across customer segments.
    """
    import plotly.graph_objects as go

    lens1 = create_sample_lens1()
    chart_json = create_revenue_concentration_pie(lens1)

    fig = go.Figure(chart_json)

    # Render as PNG image
    img_bytes = fig.to_image(format="png", width=800, height=600)

    return Image(data=img_bytes, format="png")


@mcp.tool()
def show_health_gauge() -> Image:
    """Show the health score gauge as a visual image.

    Displays overall customer base health score with color-coded ranges.
    """
    import plotly.graph_objects as go

    lens5 = create_sample_lens5()
    chart_json = create_health_score_gauge(lens5)

    fig = go.Figure(chart_json)

    # Render as PNG image
    img_bytes = fig.to_image(format="png", width=600, height=400)

    return Image(data=img_bytes, format="png")


@mcp.tool()
def show_all_charts() -> str:
    """Show information about all available charts.

    Note: Due to size limits, view charts individually using:
    - show_retention_chart
    - show_revenue_pie_chart
    - show_health_gauge
    """
    output = "# Available Visualizations\n\n"
    output += "Due to response size limits in Claude Desktop, please view charts individually:\n\n"
    output += "## 1. Retention Trend Chart\n"
    output += "Use `show_retention_chart` to see cohort retention over time.\n\n"
    output += "## 2. Revenue Concentration Pie Chart\n"
    output += "Use `show_revenue_pie_chart` to see Pareto revenue distribution.\n\n"
    output += "## 3. Health Score Gauge\n"
    output += "Use `show_health_gauge` to see the overall health score (84.5/100, Grade B).\n\n"
    output += "---\n\n"
    output += "**Try**: \"Show the retention chart\" or \"Show the health gauge\"\n"

    return output


if __name__ == "__main__":
    mcp.run()
