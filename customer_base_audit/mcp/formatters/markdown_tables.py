"""Markdown table formatters for Five Lenses analysis results.

Formats lens analysis results as clean markdown tables suitable for
display in Claude Desktop and other markdown renderers.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from customer_base_audit.analyses.lens1 import Lens1Metrics
    from customer_base_audit.analyses.lens2 import Lens2Metrics
    from customer_base_audit.analyses.lens4 import Lens4Metrics
    from customer_base_audit.analyses.lens5 import Lens5Metrics


def format_lens1_table(metrics: Lens1Metrics) -> str:
    """Format Lens 1 snapshot metrics as a markdown table.

    Parameters
    ----------
    metrics:
        Lens 1 analysis results

    Returns
    -------
    str:
        Markdown-formatted table with key metrics

    Examples
    --------
    >>> from decimal import Decimal
    >>> from customer_base_audit.analyses.lens1 import Lens1Metrics
    >>> metrics = Lens1Metrics(
    ...     total_customers=1000,
    ...     one_time_buyers=400,
    ...     one_time_buyer_pct=Decimal("40.00"),
    ...     total_revenue=Decimal("500000.00"),
    ...     top_10pct_revenue_contribution=Decimal("45.20"),
    ...     top_20pct_revenue_contribution=Decimal("62.80"),
    ...     avg_orders_per_customer=Decimal("3.50"),
    ...     median_customer_value=Decimal("350.00"),
    ...     rfm_distribution={}
    ... )
    >>> print(format_lens1_table(metrics))
    """
    # Format revenue with commas
    revenue_formatted = f"${metrics.total_revenue:,.2f}"
    median_value_formatted = f"${metrics.median_customer_value:,.2f}"

    table = f"""## Lens 1: Single Period Snapshot

| Metric | Value |
|--------|-------|
| Total Customers | {metrics.total_customers:,} |
| One-Time Buyers | {metrics.one_time_buyers:,} ({metrics.one_time_buyer_pct}%) |
| Total Revenue | {revenue_formatted} |
| Top 10% Revenue Share | {metrics.top_10pct_revenue_contribution}% |
| Top 20% Revenue Share | {metrics.top_20pct_revenue_contribution}% |
| Avg Orders/Customer | {metrics.avg_orders_per_customer} |
| Median Customer Value | {median_value_formatted} |
"""

    # Add RFM distribution if available
    if metrics.rfm_distribution:
        table += "\n### RFM Segment Distribution\n\n"
        table += "| RFM Score | Customer Count |\n"
        table += "|-----------|---------------|\n"
        for score in sorted(metrics.rfm_distribution.keys(), reverse=True):
            count = metrics.rfm_distribution[score]
            table += f"| {score} | {count:,} |\n"

    return table


def format_lens2_table(metrics: Lens2Metrics) -> str:
    """Format Lens 2 period comparison metrics as markdown tables.

    Parameters
    ----------
    metrics:
        Lens 2 analysis results

    Returns
    -------
    str:
        Markdown-formatted tables showing migration and changes

    Examples
    --------
    >>> from decimal import Decimal
    >>> from customer_base_audit.analyses.lens1 import Lens1Metrics
    >>> from customer_base_audit.analyses.lens2 import Lens2Metrics, CustomerMigration
    >>> p1 = Lens1Metrics(100, 40, Decimal("40.00"), Decimal("10000.00"),
    ...                   Decimal("45.00"), Decimal("62.00"), Decimal("2.50"),
    ...                   Decimal("100.00"), {})
    >>> p2 = Lens1Metrics(120, 50, Decimal("41.67"), Decimal("12000.00"),
    ...                   Decimal("47.00"), Decimal("64.00"), Decimal("2.80"),
    ...                   Decimal("100.00"), {})
    >>> migration = CustomerMigration(
    ...     retained=frozenset(["C1", "C2"]),
    ...     churned=frozenset(["C3"]),
    ...     new=frozenset(["C4", "C5"]),
    ...     reactivated=frozenset(["C5"])
    ... )
    >>> metrics = Lens2Metrics(
    ...     period1_metrics=p1,
    ...     period2_metrics=p2,
    ...     migration=migration,
    ...     retention_rate=Decimal("66.67"),
    ...     churn_rate=Decimal("33.33"),
    ...     reactivation_rate=Decimal("50.00"),
    ...     customer_count_change=20,
    ...     revenue_change_pct=Decimal("20.00"),
    ...     avg_order_value_change_pct=Decimal("5.00")
    ... )
    >>> print(format_lens2_table(metrics))
    """
    # Migration summary
    table = f"""## Lens 2: Period-to-Period Comparison

### Customer Migration

| Category | Count |
|----------|-------|
| Retained Customers | {len(metrics.migration.retained):,} |
| Churned Customers | {len(metrics.migration.churned):,} |
| New Customers | {len(metrics.migration.new):,} |
| Reactivated Customers | {len(metrics.migration.reactivated):,} |

### Key Rates

| Metric | Value |
|--------|-------|
| Retention Rate | {metrics.retention_rate}% |
| Churn Rate | {metrics.churn_rate}% |
| Reactivation Rate | {metrics.reactivation_rate}% |

### Period Comparison

| Metric | Period 1 | Period 2 | Change |
|--------|----------|----------|--------|
| Total Customers | {metrics.period1_metrics.total_customers:,} | {metrics.period2_metrics.total_customers:,} | {metrics.customer_count_change:+,} |
| Total Revenue | ${metrics.period1_metrics.total_revenue:,.2f} | ${metrics.period2_metrics.total_revenue:,.2f} | {_format_change(metrics.revenue_change_pct)} |
| Avg Orders/Customer | {metrics.period1_metrics.avg_orders_per_customer} | {metrics.period2_metrics.avg_orders_per_customer} | {_format_change(metrics.avg_order_value_change_pct)} |
"""
    return table


def format_lens4_decomposition_table(metrics: Lens4Metrics, max_cohorts: int = 10) -> str:
    """Format Lens 4 cohort decomposition as markdown tables.

    Parameters
    ----------
    metrics:
        Lens 4 analysis results
    max_cohorts:
        Maximum number of cohorts to display (default: 10)

    Returns
    -------
    str:
        Markdown-formatted tables showing cohort comparisons

    Examples
    --------
    >>> from decimal import Decimal
    >>> from customer_base_audit.analyses.lens4 import (
    ...     Lens4Metrics, CohortDecomposition, TimeToSecondPurchase
    ... )
    >>> decomp = CohortDecomposition(
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
    >>> ttsp = TimeToSecondPurchase(
    ...     cohort_id="2023-Q1",
    ...     customers_with_repeat=60,
    ...     repeat_rate=Decimal("60.00"),
    ...     median_days=Decimal("30.00"),
    ...     mean_days=Decimal("35.00"),
    ...     cumulative_repeat_by_period={}
    ... )
    >>> metrics = Lens4Metrics(
    ...     cohort_decompositions=[decomp],
    ...     time_to_second_purchase=[ttsp],
    ...     cohort_comparisons=[],
    ...     alignment_type="left-aligned"
    ... )
    >>> print(format_lens4_decomposition_table(metrics))
    """
    # Group decompositions by cohort
    cohorts = {}
    for decomp in metrics.cohort_decompositions:
        if decomp.cohort_id not in cohorts:
            cohorts[decomp.cohort_id] = []
        cohorts[decomp.cohort_id].append(decomp)

    # Limit to max_cohorts
    cohort_ids = sorted(cohorts.keys())[:max_cohorts]

    table = f"""## Lens 4: Multi-Cohort Comparison

**Alignment Type**: {metrics.alignment_type.title()}

### Cohort Performance Overview

"""

    # Create summary table for period 0
    table += "| Cohort | Size | Active % | AOF | AOV | Revenue |\n"
    table += "|--------|------|----------|-----|-----|----------|\n"

    for cohort_id in cohort_ids:
        period_0_decomps = [d for d in cohorts[cohort_id] if d.period_number == 0]
        if period_0_decomps:
            decomp = period_0_decomps[0]
            table += (
                f"| {cohort_id} | {decomp.cohort_size:,} | "
                f"{decomp.pct_active}% | {decomp.aof} | "
                f"${decomp.aov:,.2f} | ${decomp.total_revenue:,.2f} |\n"
            )

    # Add time to second purchase summary
    if metrics.time_to_second_purchase:
        table += "\n### Time to Second Purchase\n\n"
        table += "| Cohort | Repeat Rate | Median Days | Mean Days |\n"
        table += "|--------|-------------|-------------|----------|\n"

        for ttsp in metrics.time_to_second_purchase[:max_cohorts]:
            table += (
                f"| {ttsp.cohort_id} | {ttsp.repeat_rate}% | "
                f"{ttsp.median_days} | {ttsp.mean_days} |\n"
            )

    return table


def format_lens5_health_summary_table(metrics: Lens5Metrics) -> str:
    """Format Lens 5 customer base health metrics as markdown table.

    Parameters
    ----------
    metrics:
        Lens 5 analysis results

    Returns
    -------
    str:
        Markdown-formatted table with health scorecard

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
    >>> print(format_lens5_health_summary_table(metrics))
    """
    health = metrics.health_score

    # Determine trend emoji
    trend_indicators = {
        "improving": "📈",
        "stable": "➡️",
        "declining": "📉"
    }
    trend_emoji = trend_indicators.get(health.cohort_quality_trend, "")

    table = f"""## Lens 5: Customer Base Health Score

### Overall Health: Grade {health.health_grade} ({health.health_score}/100)

| Component | Value | Weight |
|-----------|-------|--------|
| Overall Retention Rate | {health.overall_retention_rate}% | 30% |
| Cohort Quality Trend | {health.cohort_quality_trend.title()} {trend_emoji} | 30% |
| Revenue Predictability | {health.revenue_predictability_pct}% | 20% |
| Acquisition Independence | {100 - health.acquisition_dependence_pct}% | 20% |

### Customer Base Metrics

| Metric | Value |
|--------|-------|
| Total Customers | {health.total_customers:,} |
| Active Customers | {health.total_active_customers:,} |
| Retention Rate | {health.overall_retention_rate}% |
| Revenue from Established Cohorts | {health.revenue_predictability_pct}% |
| Revenue from New Acquisitions | {health.acquisition_dependence_pct}% |
"""

    # Add cohort repeat behavior summary
    if metrics.cohort_repeat_behavior:
        table += "\n### Cohort Repeat Behavior\n\n"
        table += "| Cohort | Size | Repeat Rate | Avg Orders (Repeat Buyers) |\n"
        table += "|--------|------|-------------|---------------------------|\n"

        for crb in metrics.cohort_repeat_behavior[:5]:  # Show top 5 cohorts
            table += (
                f"| {crb.cohort_id} | {crb.cohort_size:,} | {crb.repeat_rate}% | "
                f"{crb.avg_orders_per_repeat_buyer} |\n"
            )

    return table


def _format_change(pct_change: Decimal) -> str:
    """Format percentage change with sign and color indicators.

    Parameters
    ----------
    pct_change:
        Percentage change value

    Returns
    -------
    str:
        Formatted change string with sign
    """
    if pct_change > 0:
        return f"+{pct_change}%"
    elif pct_change < 0:
        return f"{pct_change}%"
    else:
        return "0%"
