"""Executive summary generators for Five Lenses analysis results.

Generates narrative summaries and actionable insights from
lens analysis results for business stakeholders.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from customer_base_audit.analyses.lens2 import Lens2Metrics
    from customer_base_audit.analyses.lens3 import Lens3Metrics
    from customer_base_audit.analyses.lens4 import Lens4Metrics
    from customer_base_audit.analyses.lens5 import Lens5Metrics


def generate_health_summary(metrics: Lens5Metrics) -> str:
    """Generate executive summary from Lens 5 health assessment.

    Parameters
    ----------
    metrics:
        Lens 5 health assessment results

    Returns
    -------
    str:
        Narrative summary with key insights and recommendations

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
    >>> summary = generate_health_summary(metrics)
    >>> "Grade B" in summary
    True
    """
    health = metrics.health_score

    # Determine grade context
    grade_context = {
        "A": ("excellent", "maintain this strong performance"),
        "B": ("good", "focus on incremental improvements"),
        "C": ("moderate", "address areas of concern proactively"),
        "D": ("concerning", "take immediate action to improve retention"),
        "F": ("critical", "urgent intervention required"),
    }
    grade_desc, grade_advice = grade_context.get(
        health.health_grade, ("unknown", "review metrics carefully")
    )

    # Determine trend narrative
    trend_narrative = {
        "improving": "showing positive momentum with newer cohorts outperforming older ones",
        "stable": "maintaining consistent quality across cohorts",
        "declining": "showing concerning degradation in newer cohort performance",
    }
    trend_text = trend_narrative.get(
        health.cohort_quality_trend, "showing unclear trend patterns"
    )

    # Assess revenue predictability
    if health.revenue_predictability_pct >= 80:
        predictability_text = "strong revenue base from established customers"
    elif health.revenue_predictability_pct >= 60:
        predictability_text = "moderate revenue base with room to improve retention"
    else:
        predictability_text = "heavy reliance on new customer acquisition for revenue"

    # Assess acquisition dependence
    if health.acquisition_dependence_pct >= 30:
        dependence_warning = (
            f"\n\n**⚠️ Warning**: {health.acquisition_dependence_pct}% of revenue comes from "
            "newest cohort. This high acquisition dependence indicates vulnerability to "
            "acquisition channel disruptions."
        )
    else:
        dependence_warning = ""

    summary = f"""# Customer Base Health Assessment

## Overall Grade: {health.health_grade} ({health.health_score}/100)

Your customer base health is **{grade_desc}**. {grade_advice.capitalize()}.

### Key Findings

**Retention**: {
        health.overall_retention_rate
    }% of your historical customer base remains active.
This represents {health.total_active_customers:,} active customers out of {
        health.total_customers:,} total.

**Cohort Quality**: Your cohorts are {trend_text}. This trend accounts for 30% of your
overall health score and directly impacts long-term revenue sustainability.

**Revenue Stability**: {
        health.revenue_predictability_pct
    }% of revenue is predictable from
established cohorts, indicating a {predictability_text}. Only {
        health.acquisition_dependence_pct
    }%
depends on the newest cohort.{dependence_warning}

### Recommended Actions

1. **Retention Focus**: With {
        health.overall_retention_rate
    }% retention, prioritize strategies
   to re-engage inactive customers and prevent churn in active cohorts.

2. **Cohort Quality**: {
        "Continue investing in initiatives that improve new customer quality"
        if health.cohort_quality_trend == "improving"
        else (
            "Monitor cohort performance closely for early warning signs"
            if health.cohort_quality_trend == "stable"
            else "Investigate root causes of declining cohort quality immediately"
        )
    }.

3. **Revenue Diversification**: {
        "Maintain current balance between retention and acquisition"
        if health.acquisition_dependence_pct < 20
        else "Reduce acquisition dependence by improving retention and expansion revenue"
    }.
"""

    # Add cohort-specific insights if available
    if metrics.cohort_repeat_behavior:
        repeat_rates = [crb.repeat_rate for crb in metrics.cohort_repeat_behavior]
        avg_repeat_rate = sum(repeat_rates) / len(repeat_rates)

        summary += "\n### Cohort Behavior Insights\n\n"
        summary += (
            f"Average repeat purchase rate across cohorts: {avg_repeat_rate:.1f}%\n\n"
        )

        # Identify best and worst cohorts
        sorted_cohorts = sorted(
            metrics.cohort_repeat_behavior,
            key=lambda x: x.repeat_rate,
            reverse=True,
        )

        if len(sorted_cohorts) >= 2:
            best = sorted_cohorts[0]
            worst = sorted_cohorts[-1]
            summary += f"**Best Performing**: {best.cohort_id} ({best.repeat_rate}% repeat rate)\n"
            summary += f"**Needs Attention**: {worst.cohort_id} ({worst.repeat_rate}% repeat rate)\n"

    return summary


def generate_retention_insights(
    lens2: Lens2Metrics,
    lens3: Lens3Metrics | None = None,
) -> str:
    """Generate retention-focused insights from Lens 2 and optionally Lens 3.

    Parameters
    ----------
    lens2:
        Lens 2 period comparison results
    lens3:
        Optional Lens 3 cohort evolution results

    Returns
    -------
    str:
        Narrative summary focused on retention and churn patterns

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
    >>> lens2 = Lens2Metrics(
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
    >>> insights = generate_retention_insights(lens2)
    >>> "66.67%" in insights
    True
    """
    # Assess retention performance
    if lens2.retention_rate >= 80:
        retention_assessment = "excellent retention"
    elif lens2.retention_rate >= 60:
        retention_assessment = "moderate retention"
    elif lens2.retention_rate >= 40:
        retention_assessment = "concerning retention"
    else:
        retention_assessment = "critical churn levels"

    # Assess growth trajectory
    if lens2.customer_count_change > 0:
        growth_text = f"grew by {lens2.customer_count_change:,} customers ({abs(lens2.customer_count_change / lens2.period1_metrics.total_customers * 100):.1f}%)"
    elif lens2.customer_count_change < 0:
        growth_text = f"declined by {abs(lens2.customer_count_change):,} customers ({abs(lens2.customer_count_change / lens2.period1_metrics.total_customers * 100):.1f}%)"
    else:
        growth_text = "remained flat"

    # Assess reactivation
    if lens2.reactivation_rate > 10:
        reactivation_note = f"\n\n**Positive Signal**: {lens2.reactivation_rate}% of new customers are reactivations, showing successful win-back efforts."
    elif len(lens2.migration.reactivated) > 0:
        reactivation_note = f"\n\n**Note**: {len(lens2.migration.reactivated):,} customers reactivated ({lens2.reactivation_rate}%)."
    else:
        reactivation_note = ""

    insights = f"""# Retention & Churn Analysis

## Period-to-Period Performance

Your customer base {growth_text}, with {retention_assessment}.

### Migration Breakdown

- **Retained**: {len(lens2.migration.retained):,} customers ({lens2.retention_rate}%)
- **Churned**: {len(lens2.migration.churned):,} customers ({lens2.churn_rate}%)
- **New**: {len(lens2.migration.new):,} customers
- **Reactivated**: {len(lens2.migration.reactivated):,} customers ({
        lens2.reactivation_rate
    }% of new){reactivation_note}

### Revenue Impact

Revenue {"increased" if lens2.revenue_change_pct > 0 else "decreased"} by {
        abs(lens2.revenue_change_pct)
    }%
(${lens2.period1_metrics.total_revenue:,.2f} → ${
        lens2.period2_metrics.total_revenue:,.2f}).

Average order value {
        "increased" if lens2.avg_order_value_change_pct > 0 else "decreased"
    } by
{abs(lens2.avg_order_value_change_pct)}%, indicating {
        "improved customer spend patterns"
        if lens2.avg_order_value_change_pct > 0
        else "pricing pressure or customer downgrading"
    }.

### Strategic Recommendations

"""

    # Add targeted recommendations
    if lens2.retention_rate < 60:
        insights += (
            "1. **Urgent**: Implement churn prevention program for at-risk customers\n"
        )
        insights += "2. Conduct exit surveys to understand churn drivers\n"
        insights += "3. Create win-back campaigns for recently churned customers\n"
    elif lens2.retention_rate < 80:
        insights += "1. Analyze characteristics of churned customers to identify at-risk patterns\n"
        insights += (
            "2. Implement proactive engagement for customers showing churn signals\n"
        )
        insights += "3. Strengthen onboarding and early-lifecycle experience\n"
    else:
        insights += "1. Maintain current retention programs that are driving strong performance\n"
        insights += "2. Document and scale best practices across customer segments\n"
        insights += "3. Focus on expanding revenue from retained customer base\n"

    # Add Lens 3 insights if provided
    if lens3 is not None:
        insights += "\n### Cohort Evolution Patterns\n\n"

        if len(lens3.periods) >= 2:
            period_0 = lens3.periods[0]
            latest = lens3.periods[-1]

            retention_decay = (
                period_0.cumulative_activation_rate - latest.cumulative_activation_rate
            ) * 100

            insights += f"**{lens3.cohort_name}** cohort retention:\n"
            insights += (
                f"- Started with {period_0.active_customers:,} active customers\n"
            )
            insights += f"- Now has {latest.active_customers:,} active customers (Period {latest.period_number})\n"
            insights += f"- Cumulative activation rate: {latest.cumulative_activation_rate * 100:.1f}%\n"

            if retention_decay > 20:
                insights += f"\n⚠️ Significant retention decay of {retention_decay:.1f} percentage points suggests strong churn in early lifecycle.\n"

    return insights


def generate_cohort_comparison(metrics: Lens4Metrics, max_cohorts: int = 5) -> str:
    """Generate cohort comparison summary from Lens 4 analysis.

    Parameters
    ----------
    metrics:
        Lens 4 multi-cohort comparison results
    max_cohorts:
        Maximum number of cohorts to analyze in detail (default: 5)

    Returns
    -------
    str:
        Narrative summary comparing cohort performance

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
    >>> summary = generate_cohort_comparison(metrics)
    >>> "2023-Q1" in summary
    True
    """
    if not metrics.cohort_decompositions:
        return "# Cohort Comparison\n\nNo cohort data available for analysis."

    # Group by cohort
    cohorts = {}
    for decomp in metrics.cohort_decompositions:
        if decomp.cohort_id not in cohorts:
            cohorts[decomp.cohort_id] = []
        cohorts[decomp.cohort_id].append(decomp)

    cohort_ids = sorted(cohorts.keys())[:max_cohorts]

    summary = f"""# Multi-Cohort Performance Comparison

## Analysis Overview

Comparing **{len(cohort_ids)}** cohorts using **{metrics.alignment_type}** analysis.
"""

    if metrics.alignment_type == "left-aligned":
        summary += "\nCohorts are compared at equivalent lifecycle stages (Period 0 = acquisition period).\n"
    else:
        summary += "\nCohorts are compared within same calendar periods.\n"

    # Analyze period 0 performance
    summary += "\n### Acquisition Period Performance\n\n"

    period_0_decomps = []
    for cohort_id in cohort_ids:
        period_0 = [d for d in cohorts[cohort_id] if d.period_number == 0]
        if period_0:
            period_0_decomps.append(period_0[0])

    if period_0_decomps:
        # Sort by revenue per customer
        sorted_by_revenue = sorted(
            period_0_decomps,
            key=lambda d: d.total_revenue / Decimal(str(d.cohort_size)),
            reverse=True,
        )

        summary += "| Cohort | Size | Revenue/Customer | AOV | AOF |\n"
        summary += "|--------|------|------------------|-----|-----|\n"

        for decomp in sorted_by_revenue:
            rev_per_customer = decomp.total_revenue / Decimal(str(decomp.cohort_size))
            summary += (
                f"| {decomp.cohort_id} | {decomp.cohort_size:,} | "
                f"${rev_per_customer:,.2f} | ${decomp.aov:,.2f} | {decomp.aof} |\n"
            )

    # Time to second purchase analysis
    if metrics.time_to_second_purchase:
        summary += "\n### Time to Second Purchase\n\n"

        ttsp_data = [
            t for t in metrics.time_to_second_purchase if t.cohort_id in cohort_ids
        ]

        if ttsp_data:
            # Sort by repeat rate
            sorted_by_repeat = sorted(
                ttsp_data, key=lambda t: t.repeat_rate, reverse=True
            )

            summary += "| Cohort | Repeat Rate | Median Days | Mean Days |\n"
            summary += "|--------|-------------|-------------|----------|\n"

            for ttsp in sorted_by_repeat:
                summary += (
                    f"| {ttsp.cohort_id} | {ttsp.repeat_rate}% | "
                    f"{ttsp.median_days} | {ttsp.mean_days} |\n"
                )

            # Identify best performer
            best = sorted_by_repeat[0]
            summary += f"\n**Best Performer**: {best.cohort_id} with {best.repeat_rate}% repeat rate "
            summary += (
                f"and median time to second purchase of {best.median_days} days.\n"
            )

    # Key insights from comparisons
    if metrics.cohort_comparisons:
        summary += "\n### Cohort Quality Trends\n\n"

        # Analyze if newer cohorts are better or worse
        pct_active_changes = [
            c.pct_active_change_pct for c in metrics.cohort_comparisons
        ]
        aov_changes = [c.aov_change_pct for c in metrics.cohort_comparisons]

        avg_pct_active_change = sum(pct_active_changes) / len(pct_active_changes)
        avg_aov_change = sum(aov_changes) / len(aov_changes)

        if avg_pct_active_change > 5:
            summary += "✅ Newer cohorts show improving retention rates (average +"
            summary += f"{avg_pct_active_change:.1f}%)\n"
        elif avg_pct_active_change < -5:
            summary += "⚠️ Newer cohorts show declining retention rates (average "
            summary += f"{avg_pct_active_change:.1f}%)\n"
        else:
            summary += "➡️ Cohort retention rates are relatively stable across cohorts\n"

        if avg_aov_change > 5:
            summary += "✅ Newer cohorts have higher average order values (average +"
            summary += f"{avg_aov_change:.1f}%)\n"
        elif avg_aov_change < -5:
            summary += "⚠️ Newer cohorts have lower average order values (average "
            summary += f"{avg_aov_change:.1f}%)\n"
        else:
            summary += "➡️ Average order values are stable across cohorts\n"

    summary += "\n### Strategic Implications\n\n"

    if metrics.cohort_comparisons and avg_pct_active_change < -5:
        summary += "1. **Urgent**: Investigate declining cohort quality - this threatens long-term growth\n"
        summary += "2. Compare acquisition channels, onboarding, and early experience across cohorts\n"
        summary += (
            "3. Consider pausing underperforming channels until quality improves\n"
        )
    else:
        summary += "1. Continue monitoring cohort performance to detect early quality degradation\n"
        summary += (
            "2. Document and replicate success patterns from best-performing cohorts\n"
        )
        summary += (
            "3. Optimize for long-term customer value, not just acquisition volume\n"
        )

    return summary
