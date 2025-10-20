"""Plotly chart generators for Five Lenses analysis results.

Creates interactive Plotly charts in JSON format for display in
Claude Desktop and other visualization tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from customer_base_audit.analyses.lens1 import Lens1Metrics
    from customer_base_audit.analyses.lens3 import Lens3Metrics
    from customer_base_audit.analyses.lens5 import Lens5Metrics


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
    }

    return {"data": [retention_trace, active_trace], "layout": layout}


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

    layout = {
        "title": {
            "text": "Revenue Concentration (Pareto Analysis)",
            "x": 0.5,
            "xanchor": "center",
        },
        "showlegend": True,
        "legend": {"x": 0.85, "y": 0.5},
    }

    return {"data": [pie_trace], "layout": layout}


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

    layout = {
        "title": {
            "text": "Customer Base Health Assessment",
            "x": 0.5,
            "xanchor": "center",
        },
        "height": 400,
    }

    return {"data": [gauge_trace], "layout": layout}


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
        80 if health.cohort_quality_trend == "improving" else (50 if health.cohort_quality_trend == "stable" else 20),
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
        "height": 800,
        "showlegend": False,
    }

    return {
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
