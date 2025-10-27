"""Plotly chart generators for Five Lenses analysis results.

Creates interactive Plotly charts in JSON format for display in
Claude Desktop and other visualization tools.

Charts are optimized for token efficiency by default (800x400px).
Use ChartConfig to adjust size/quality tradeoffs.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from customer_base_audit.analyses.lens1 import Lens1Metrics
    from customer_base_audit.analyses.lens2 import Lens2Metrics
    from customer_base_audit.analyses.lens3 import Lens3Metrics
    from customer_base_audit.analyses.lens4 import Lens4Metrics
    from customer_base_audit.analyses.lens5 import Lens5Metrics


def _get_default_height(base_height: int | None = None) -> int:
    """Get default height from chart config.

    Parameters
    ----------
    base_height:
        Optional override height

    Returns
    -------
    int:
        Height in pixels
    """
    if base_height is not None:
        return base_height
    from customer_base_audit.mcp.formatters import get_chart_config

    return get_chart_config().height


def _get_default_width(base_width: int | None = None) -> int:
    """Get default width from chart config.

    Parameters
    ----------
    base_width:
        Optional override width

    Returns
    -------
    int:
        Width in pixels
    """
    if base_width is not None:
        return base_width
    from customer_base_audit.mcp.formatters import get_chart_config

    return get_chart_config().width


def _convert_plotly_to_base64_png(fig_dict: dict[str, Any]) -> str:
    """Convert Plotly JSON to base64-encoded PNG.

    Uses kaleido to render static image for Claude Desktop display.

    Parameters
    ----------
    fig_dict:
        Plotly figure specification as JSON-serializable dict

    Returns
    -------
    str:
        Base64-encoded PNG image data

    Raises
    ------
    ImportError:
        If plotly.graph_objects is not available
    Exception:
        If PNG conversion fails
    """
    import plotly.graph_objects as go

    fig = go.Figure(fig_dict)
    img_bytes = fig.to_image(format="png", engine="kaleido")
    return base64.b64encode(img_bytes).decode("utf-8")


def create_retention_trend_chart(metrics: Lens3Metrics) -> dict[str, Any]:
    """Create retention trend chart from Lens 3 cohort evolution data.

    Visualizes how a cohort's retention evolves over time using
    cumulative activation rates.

    Parameters
    ----------
    metrics:
        Lens 3 cohort evolution results

    Returns
    -------
    dict:
        Plotly figure specification as JSON-serializable dict

    Examples
    --------
    >>> from datetime import datetime
    >>> from customer_base_audit.analyses.lens3 import (
    ...     Lens3Metrics, CohortPeriodMetrics
    ... )
    >>> metrics = Lens3Metrics(
    ...     cohort_name="2023-Q1",
    ...     acquisition_date=datetime(2023, 1, 1),
    ...     cohort_size=100,
    ...     periods=[
    ...         CohortPeriodMetrics(0, 100, 1.0, 1.5, 50.0, 1.5, 50.0, 5000.0),
    ...         CohortPeriodMetrics(1, 80, 0.9, 1.2, 40.0, 0.96, 32.0, 3200.0),
    ...     ]
    ... )
    >>> chart = create_retention_trend_chart(metrics)
    >>> chart['data'][0]['type']
    'scatter'
    """
    periods = [p.period_number for p in metrics.periods]
    activation_rates = [p.cumulative_activation_rate * 100 for p in metrics.periods]
    active_customers = [p.active_customers for p in metrics.periods]

    # Create retention curve
    retention_trace = {
        "type": "scatter",
        "mode": "lines+markers",
        "name": "Cumulative Activation %",
        "x": periods,
        "y": activation_rates,
        "line": {"color": "rgb(55, 128, 191)", "width": 2},
        "marker": {"size": 8},
        "hovertemplate": "Period %{x}<br>Activation: %{y:.1f}%<extra></extra>",
    }

    # Create active customers bar chart
    active_trace = {
        "type": "bar",
        "name": "Active Customers",
        "x": periods,
        "y": active_customers,
        "yaxis": "y2",
        "marker": {"color": "rgba(255, 165, 0, 0.6)"},
        "hovertemplate": "Period %{x}<br>Active: %{y}<extra></extra>",
    }

    width = _get_default_width()
    height = _get_default_height()

    layout = {
        "title": {
            "text": f"Cohort Retention Trend: {metrics.cohort_name}",
            "x": 0.5,
            "xanchor": "center",
        },
        "xaxis": {
            "title": "Periods Since Acquisition",
            "tickmode": "linear",
            "dtick": 1,
        },
        "yaxis": {
            "title": "Cumulative Activation Rate (%)",
            "range": [0, 105],
            "side": "left",
        },
        "yaxis2": {
            "title": "Active Customers",
            "overlaying": "y",
            "side": "right",
        },
        "hovermode": "x unified",
        "showlegend": True,
        "legend": {"x": 0.01, "y": 0.99, "xanchor": "left", "yanchor": "top"},
        "width": width,
        "height": height,
    }

    fig_dict = {"data": [retention_trace, active_trace], "layout": layout}

    # Return dual format: PNG for display + JSON for programmatic access
    return {
        "plotly_json": fig_dict,
        "image_base64": _convert_plotly_to_base64_png(fig_dict),
        "format": "png",
        "width": width,
        "height": height,
    }


def create_revenue_concentration_pie(metrics: Lens1Metrics) -> dict[str, Any]:
    """Create revenue concentration pie chart from Lens 1 data.

    Visualizes Pareto distribution of revenue across customer segments.

    Parameters
    ----------
    metrics:
        Lens 1 snapshot results

    Returns
    -------
    dict:
        Plotly figure specification as JSON-serializable dict

    Examples
    --------
    >>> from decimal import Decimal
    >>> from customer_base_audit.analyses.lens1 import Lens1Metrics
    >>> metrics = Lens1Metrics(
    ...     total_customers=100,
    ...     one_time_buyers=40,
    ...     one_time_buyer_pct=Decimal("40.00"),
    ...     total_revenue=Decimal("10000.00"),
    ...     top_10pct_revenue_contribution=Decimal("45.00"),
    ...     top_20pct_revenue_contribution=Decimal("62.00"),
    ...     avg_orders_per_customer=Decimal("2.50"),
    ...     median_customer_value=Decimal("100.00"),
    ...     rfm_distribution={}
    ... )
    >>> chart = create_revenue_concentration_pie(metrics)
    >>> chart['data'][0]['type']
    'pie'
    """
    # Calculate segment contributions
    top_10_pct = float(metrics.top_10pct_revenue_contribution)
    top_20_pct = float(metrics.top_20pct_revenue_contribution)
    remaining_80_pct = 100 - top_20_pct

    # Calculate middle 10% (11-20%)
    middle_10_pct = top_20_pct - top_10_pct

    labels = ["Top 10% Customers", "Next 10% (11-20%)", "Remaining 80% Customers"]
    values = [top_10_pct, middle_10_pct, remaining_80_pct]
    colors = ["rgb(55, 128, 191)", "rgb(255, 165, 0)", "rgb(220, 220, 220)"]

    pie_trace = {
        "type": "pie",
        "labels": labels,
        "values": values,
        "marker": {"colors": colors},
        "textinfo": "label+percent",
        "textposition": "auto",
        "hovertemplate": "%{label}<br>Revenue Share: %{value:.1f}%<extra></extra>",
    }

    width = _get_default_width()
    height = _get_default_height()

    layout = {
        "title": {
            "text": "Revenue Concentration (Pareto Analysis)",
            "x": 0.5,
            "xanchor": "center",
        },
        "showlegend": True,
        "legend": {"x": 0.85, "y": 0.5},
        "width": width,
        "height": height,
    }

    fig_dict = {"data": [pie_trace], "layout": layout}

    # Return dual format: PNG for display + JSON for programmatic access
    return {
        "plotly_json": fig_dict,
        "image_base64": _convert_plotly_to_base64_png(fig_dict),
        "format": "png",
        "width": width,
        "height": height,
    }


def create_health_score_gauge(metrics: Lens5Metrics) -> dict[str, Any]:
    """Create health score gauge chart from Lens 5 data.

    Displays overall customer base health score as an indicator gauge.

    Parameters
    ----------
    metrics:
        Lens 5 health assessment results

    Returns
    -------
    dict:
        Plotly figure specification as JSON-serializable dict

    Examples
    --------
    >>> from datetime import datetime
    >>> from decimal import Decimal
    >>> from customer_base_audit.analyses.lens5 import (
    ...     Lens5Metrics, CustomerBaseHealthScore
    ... )
    >>> health = CustomerBaseHealthScore(
    ...     total_customers=1000,
    ...     total_active_customers=800,
    ...     overall_retention_rate=Decimal("80.00"),
    ...     cohort_quality_trend="improving",
    ...     revenue_predictability_pct=Decimal("85.00"),
    ...     acquisition_dependence_pct=Decimal("15.00"),
    ...     health_score=Decimal("82.50"),
    ...     health_grade="B"
    ... )
    >>> metrics = Lens5Metrics(
    ...     cohort_revenue_contributions=[],
    ...     cohort_repeat_behavior=[],
    ...     health_score=health,
    ...     analysis_start_date=datetime(2023, 1, 1),
    ...     analysis_end_date=datetime(2023, 12, 31)
    ... )
    >>> chart = create_health_score_gauge(metrics)
    >>> chart['data'][0]['type']
    'indicator'
    """
    health = metrics.health_score
    score = float(health.health_score)

    # Determine color based on grade
    grade_colors = {
        "A": "rgb(34, 139, 34)",  # Green
        "B": "rgb(0, 128, 255)",  # Blue
        "C": "rgb(255, 165, 0)",  # Orange
        "D": "rgb(255, 140, 0)",  # Dark Orange
        "F": "rgb(220, 20, 60)",  # Red
    }
    color = grade_colors.get(health.health_grade, "rgb(128, 128, 128)")

    gauge_trace = {
        "type": "indicator",
        "mode": "gauge+number+delta",
        "value": score,
        "title": {"text": f"Health Score (Grade {health.health_grade})"},
        "delta": {"reference": 70, "increasing": {"color": "green"}},
        "gauge": {
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 60], "color": "rgba(220, 20, 60, 0.2)"},  # F range
                {"range": [60, 70], "color": "rgba(255, 140, 0, 0.2)"},  # D range
                {"range": [70, 80], "color": "rgba(255, 165, 0, 0.2)"},  # C range
                {"range": [80, 90], "color": "rgba(0, 128, 255, 0.2)"},  # B range
                {"range": [90, 100], "color": "rgba(34, 139, 34, 0.2)"},  # A range
            ],
            "threshold": {
                "line": {"color": "black", "width": 2},
                "thickness": 0.75,
                "value": score,
            },
        },
    }

    width = _get_default_width()
    height = _get_default_height()

    layout = {
        "title": {
            "text": "Customer Base Health Assessment",
            "x": 0.5,
            "xanchor": "center",
        },
        "width": width,
        "height": height,
    }

    fig_dict = {"data": [gauge_trace], "layout": layout}

    # Return dual format: PNG for display + JSON for programmatic access
    return {
        "plotly_json": fig_dict,
        "image_base64": _convert_plotly_to_base64_png(fig_dict),
        "format": "png",
        "width": width,
        "height": height,
    }


def create_executive_dashboard(
    lens1: Lens1Metrics,
    lens5: Lens5Metrics,
) -> dict[str, Any]:
    """Create multi-lens executive dashboard.

    Combines key metrics from multiple lenses into a single dashboard view.

    Parameters
    ----------
    lens1:
        Lens 1 snapshot metrics
    lens5:
        Lens 5 health assessment metrics

    Returns
    -------
    dict:
        Plotly figure specification with subplots

    Examples
    --------
    >>> from datetime import datetime
    >>> from decimal import Decimal
    >>> from customer_base_audit.analyses.lens1 import Lens1Metrics
    >>> from customer_base_audit.analyses.lens5 import (
    ...     Lens5Metrics, CustomerBaseHealthScore
    ... )
    >>> lens1 = Lens1Metrics(
    ...     total_customers=100,
    ...     one_time_buyers=40,
    ...     one_time_buyer_pct=Decimal("40.00"),
    ...     total_revenue=Decimal("10000.00"),
    ...     top_10pct_revenue_contribution=Decimal("45.00"),
    ...     top_20pct_revenue_contribution=Decimal("62.00"),
    ...     avg_orders_per_customer=Decimal("2.50"),
    ...     median_customer_value=Decimal("100.00"),
    ...     rfm_distribution={}
    ... )
    >>> health = CustomerBaseHealthScore(
    ...     total_customers=1000,
    ...     total_active_customers=800,
    ...     overall_retention_rate=Decimal("80.00"),
    ...     cohort_quality_trend="improving",
    ...     revenue_predictability_pct=Decimal("85.00"),
    ...     acquisition_dependence_pct=Decimal("15.00"),
    ...     health_score=Decimal("82.50"),
    ...     health_grade="B"
    ... )
    >>> lens5 = Lens5Metrics(
    ...     cohort_revenue_contributions=[],
    ...     cohort_repeat_behavior=[],
    ...     health_score=health,
    ...     analysis_start_date=datetime(2023, 1, 1),
    ...     analysis_end_date=datetime(2023, 12, 31)
    ... )
    >>> dashboard = create_executive_dashboard(lens1, lens5)
    >>> len(dashboard['data']) > 0
    True
    """
    # Key metrics summary
    health = lens5.health_score

    # Create KPI indicators
    kpi_trace = {
        "type": "indicator",
        "mode": "number",
        "value": lens1.total_customers,
        "title": {"text": "Total Customers"},
        "domain": {"x": [0, 0.25], "y": [0.7, 1]},
    }

    revenue_trace = {
        "type": "indicator",
        "mode": "number",
        "value": float(lens1.total_revenue),
        "number": {"prefix": "$", "valueformat": ",.0f"},
        "title": {"text": "Total Revenue"},
        "domain": {"x": [0.25, 0.5], "y": [0.7, 1]},
    }

    retention_trace = {
        "type": "indicator",
        "mode": "number",
        "value": float(health.overall_retention_rate),
        "number": {"suffix": "%"},
        "title": {"text": "Retention Rate"},
        "domain": {"x": [0.5, 0.75], "y": [0.7, 1]},
    }

    health_trace = {
        "type": "indicator",
        "mode": "number+gauge",
        "value": float(health.health_score),
        "title": {"text": f"Health Score ({health.health_grade})"},
        "gauge": {"axis": {"range": [0, 100]}},
        "domain": {"x": [0.75, 1], "y": [0.7, 1]},
    }

    # Revenue concentration pie (bottom half)
    top_10_pct = float(lens1.top_10pct_revenue_contribution)
    top_20_pct = float(lens1.top_20pct_revenue_contribution)
    middle_10_pct = top_20_pct - top_10_pct
    remaining_80_pct = 100 - top_20_pct

    pie_trace = {
        "type": "pie",
        "labels": ["Top 10%", "Next 10%", "Bottom 80%"],
        "values": [top_10_pct, middle_10_pct, remaining_80_pct],
        "marker": {
            "colors": ["rgb(55, 128, 191)", "rgb(255, 165, 0)", "rgb(220, 220, 220)"]
        },
        "domain": {"x": [0, 0.5], "y": [0, 0.6]},
        "name": "Revenue Distribution",
    }

    # Cohort quality breakdown (bottom right)
    components = [
        "Retention",
        "Cohort Quality",
        "Predictability",
        "Independence",
    ]
    scores = [
        float(health.overall_retention_rate),
        80
        if health.cohort_quality_trend == "improving"
        else (50 if health.cohort_quality_trend == "stable" else 20),
        float(health.revenue_predictability_pct),
        100 - float(health.acquisition_dependence_pct),
    ]

    bar_trace = {
        "type": "bar",
        "x": components,
        "y": scores,
        "marker": {"color": "rgb(55, 128, 191)"},
        "xaxis": "x2",
        "yaxis": "y2",
        "name": "Health Components",
    }

    width = _get_default_width()
    height = _get_default_height(800)  # Larger dashboard, but configurable

    layout = {
        "title": {
            "text": "Customer Base Executive Dashboard",
            "x": 0.5,
            "xanchor": "center",
        },
        "xaxis2": {
            "domain": [0.55, 1],
            "anchor": "y2",
        },
        "yaxis2": {
            "domain": [0, 0.6],
            "anchor": "x2",
            "range": [0, 100],
        },
        "width": width,
        "height": height,
        "showlegend": False,
    }

    fig_dict = {
        "data": [
            kpi_trace,
            revenue_trace,
            retention_trace,
            health_trace,
            pie_trace,
            bar_trace,
        ],
        "layout": layout,
    }

    # Return dual format: PNG for display + JSON for programmatic access
    return {
        "plotly_json": fig_dict,
        "image_base64": _convert_plotly_to_base64_png(fig_dict),
        "format": "png",
        "width": width,
        "height": height,
    }


def create_cohort_heatmap(metrics: Lens4Metrics) -> dict[str, Any]:
    """Create cohort performance heatmap from Lens 4 data.

    Visualizes cohort performance over periods using a color-coded heatmap.
    Shows retention rates or revenue per customer across cohorts and periods.

    Parameters
    ----------
    metrics:
        Lens 4 multi-cohort comparison results

    Returns
    -------
    dict:
        Plotly figure specification as JSON-serializable dict

    Examples
    --------
    >>> from decimal import Decimal
    >>> from customer_base_audit.analyses.lens4 import (
    ...     Lens4Metrics, CohortDecomposition
    ... )
    >>> decomp1 = CohortDecomposition(
    ...     cohort_id="2023-Q1",
    ...     period_number=0,
    ...     cohort_size=100,
    ...     active_customers=100,
    ...     pct_active=Decimal("100.00"),
    ...     total_orders=150,
    ...     aof=Decimal("1.50"),
    ...     total_revenue=Decimal("15000.00"),
    ...     aov=Decimal("100.00"),
    ...     margin=Decimal("100.00"),
    ...     revenue=Decimal("15000.00")
    ... )
    >>> decomp2 = CohortDecomposition(
    ...     cohort_id="2023-Q1",
    ...     period_number=1,
    ...     cohort_size=100,
    ...     active_customers=80,
    ...     pct_active=Decimal("80.00"),
    ...     total_orders=120,
    ...     aof=Decimal("1.50"),
    ...     total_revenue=Decimal("12000.00"),
    ...     aov=Decimal("100.00"),
    ...     margin=Decimal("100.00"),
    ...     revenue=Decimal("12000.00")
    ... )
    >>> metrics = Lens4Metrics(
    ...     cohort_decompositions=[decomp1, decomp2],
    ...     time_to_second_purchase=[],
    ...     cohort_comparisons=[],
    ...     alignment_type="left-aligned"
    ... )
    >>> chart = create_cohort_heatmap(metrics)
    >>> chart['data'][0]['type']
    'heatmap'
    """
    # Group decompositions by cohort
    cohorts_data: dict[str, list[tuple[int, float]]] = {}
    for decomp in metrics.cohort_decompositions:
        if decomp.cohort_id not in cohorts_data:
            cohorts_data[decomp.cohort_id] = []
        # Use retention rate (pct_active) as the metric
        retention_pct = float(decomp.pct_active)
        cohorts_data[decomp.cohort_id].append((decomp.period_number, retention_pct))

    # Sort cohorts by name (chronological)
    cohort_ids = sorted(cohorts_data.keys())

    # Find max period across all cohorts
    max_period = max(
        period for periods in cohorts_data.values() for period, _ in periods
    )

    # Build heatmap matrix
    z_values: list[list[float | None]] = []
    for cohort_id in cohort_ids:
        row: list[float | None] = [None] * (max_period + 1)
        for period, retention in cohorts_data[cohort_id]:
            row[period] = retention
        z_values.append(row)

    # Create heatmap
    heatmap_trace = {
        "type": "heatmap",
        "z": z_values,
        "x": list(range(max_period + 1)),
        "y": cohort_ids,
        "colorscale": [
            [0.0, "rgb(220, 20, 60)"],  # Red for low retention
            [0.5, "rgb(255, 165, 0)"],  # Orange for medium
            [1.0, "rgb(34, 139, 34)"],  # Green for high retention
        ],
        "colorbar": {
            "title": "Retention %",
            "ticksuffix": "%",
        },
        "hovertemplate": (
            "Cohort: %{y}<br>Period: %{x}<br>Retention: %{z:.1f}%<extra></extra>"
        ),
        "zmin": 0,
        "zmax": 100,
    }

    # Calculate dynamic height based on cohort count, but respect config min/max
    base_height = _get_default_height()
    dynamic_height = max(base_height, min(len(cohort_ids) * 30, base_height * 2))
    width = _get_default_width()

    layout = {
        "title": {
            "text": f"Cohort Retention Heatmap ({metrics.alignment_type.title()})",
            "x": 0.5,
            "xanchor": "center",
        },
        "xaxis": {
            "title": "Period Number",
            "tickmode": "linear",
            "dtick": 1,
        },
        "yaxis": {
            "title": "Cohort",
            "autorange": "reversed",  # Newest cohorts at top
        },
        "width": width,
        "height": dynamic_height,
    }

    fig_dict = {"data": [heatmap_trace], "layout": layout}

    # Return dual format: PNG for display + JSON for programmatic access
    return {
        "plotly_json": fig_dict,
        "image_base64": _convert_plotly_to_base64_png(fig_dict),
        "format": "png",
        "width": width,
        "height": dynamic_height,
    }


def create_sankey_diagram(metrics: Lens2Metrics) -> dict[str, Any]:
    """Create customer migration Sankey diagram from Lens 2 data.

    Visualizes customer flow between periods, showing retention, churn,
    new acquisitions, and reactivations.

    Parameters
    ----------
    metrics:
        Lens 2 period comparison results

    Returns
    -------
    dict:
        Plotly figure specification as JSON-serializable dict

    Examples
    --------
    >>> from decimal import Decimal
    >>> from customer_base_audit.analyses.lens1 import Lens1Metrics
    >>> from customer_base_audit.analyses.lens2 import (
    ...     Lens2Metrics, CustomerMigration
    ... )
    >>> p1 = Lens1Metrics(
    ...     total_customers=100,
    ...     one_time_buyers=40,
    ...     one_time_buyer_pct=Decimal("40.00"),
    ...     total_revenue=Decimal("10000.00"),
    ...     top_10pct_revenue_contribution=Decimal("45.00"),
    ...     top_20pct_revenue_contribution=Decimal("62.00"),
    ...     avg_orders_per_customer=Decimal("2.50"),
    ...     median_customer_value=Decimal("100.00"),
    ...     rfm_distribution={}
    ... )
    >>> p2 = Lens1Metrics(
    ...     total_customers=120,
    ...     one_time_buyers=50,
    ...     one_time_buyer_pct=Decimal("41.67"),
    ...     total_revenue=Decimal("12000.00"),
    ...     top_10pct_revenue_contribution=Decimal("47.00"),
    ...     top_20pct_revenue_contribution=Decimal("64.00"),
    ...     avg_orders_per_customer=Decimal("2.80"),
    ...     median_customer_value=Decimal("100.00"),
    ...     rfm_distribution={}
    ... )
    >>> migration = CustomerMigration(
    ...     retained=frozenset(["C1", "C2"]),
    ...     churned=frozenset(["C3"]),
    ...     new=frozenset(["C4", "C5"]),
    ...     reactivated=frozenset(["C6"])
    ... )
    >>> metrics = Lens2Metrics(
    ...     period1_metrics=p1,
    ...     period2_metrics=p2,
    ...     migration=migration,
    ...     retention_rate=Decimal("66.67"),
    ...     churn_rate=Decimal("33.33"),
    ...     reactivation_rate=Decimal("16.67"),
    ...     customer_count_change=20,
    ...     revenue_change_pct=Decimal("20.00"),
    ...     avg_order_value_change_pct=Decimal("5.00")
    ... )
    >>> chart = create_sankey_diagram(metrics)
    >>> chart['data'][0]['type']
    'sankey'
    """
    # Calculate customer counts
    retained = len(metrics.migration.retained)
    churned = len(metrics.migration.churned)
    new = len(metrics.migration.new)
    reactivated = len(metrics.migration.reactivated)

    # Define nodes
    # Node indices: 0=Period1, 1=Retained, 2=Churned, 3=New, 4=Reactivated, 5=Period2
    nodes = {
        "label": [
            "Period 1 Customers",
            "Retained",
            "Churned",
            "New Customers",
            "Reactivated",
            "Period 2 Customers",
        ],
        "color": [
            "rgb(128, 128, 128)",  # Period 1 (gray)
            "rgb(34, 139, 34)",  # Retained (green)
            "rgb(220, 20, 60)",  # Churned (red)
            "rgb(0, 128, 255)",  # New (blue)
            "rgb(255, 165, 0)",  # Reactivated (orange)
            "rgb(128, 128, 128)",  # Period 2 (gray)
        ],
    }

    # Define flows (links)
    links: dict[str, list[int | str]] = {
        "source": [],
        "target": [],
        "value": [],
        "color": [],
    }

    # Flow: Period 1 → Retained → Period 2
    if retained > 0:
        links["source"].extend([0, 1])
        links["target"].extend([1, 5])
        links["value"].extend([retained, retained])
        links["color"].extend(
            ["rgba(34, 139, 34, 0.4)", "rgba(34, 139, 34, 0.4)"]  # Green
        )

    # Flow: Period 1 → Churned
    if churned > 0:
        links["source"].append(0)
        links["target"].append(2)
        links["value"].append(churned)
        links["color"].append("rgba(220, 20, 60, 0.4)")  # Red

    # Flow: New → Period 2
    if new > 0:
        links["source"].append(3)
        links["target"].append(5)
        links["value"].append(new)
        links["color"].append("rgba(0, 128, 255, 0.4)")  # Blue

    # Flow: Reactivated → Period 2
    if reactivated > 0:
        links["source"].append(4)
        links["target"].append(5)
        links["value"].append(reactivated)
        links["color"].append("rgba(255, 165, 0, 0.4)")  # Orange

    sankey_trace = {
        "type": "sankey",
        "node": nodes,
        "link": links,
        "textfont": {"size": 12},
    }

    width = _get_default_width()
    height = _get_default_height(500)

    layout = {
        "title": {
            "text": "Customer Migration Flow",
            "x": 0.5,
            "xanchor": "center",
        },
        "font": {"size": 10},
        "width": width,
        "height": height,
    }

    fig_dict = {"data": [sankey_trace], "layout": layout}

    # Return dual format: PNG for display + JSON for programmatic access
    return {
        "plotly_json": fig_dict,
        "image_base64": _convert_plotly_to_base64_png(fig_dict),
        "format": "png",
        "width": width,
        "height": height,
    }
