"""Lens 5 MCP Tool - Phase 2 Lens Service

Wraps Lens 5 (Overall Customer Base Health) as an MCP tool for agentic orchestration.
"""

from decimal import Decimal
from datetime import datetime

import structlog
from customer_base_audit.analyses.lens5 import (
    Lens5Metrics,
    assess_customer_base_health,
)
from fastmcp import Context
from pydantic import BaseModel, Field

from analytics.services.mcp_server.main import mcp
from analytics.services.mcp_server.state import get_shared_state

logger = structlog.get_logger(__name__)


class Lens5Request(BaseModel):
    """Request for Lens 5 analysis."""

    analysis_name: str = Field(
        default="Customer Base Health Assessment",
        description="Name for this health assessment",
    )
    analysis_start_date: datetime | None = Field(
        default=None,
        description="Start of analysis window (defaults to earliest period)",
    )
    analysis_end_date: datetime | None = Field(
        default=None,
        description="End of analysis window (defaults to latest period)",
    )


class CohortRevenueSummary(BaseModel):
    """Summary of cohort revenue contribution."""

    cohort_id: str
    total_revenue: float
    pct_of_total_revenue: float
    active_customers: int


class CohortBehaviorSummary(BaseModel):
    """Summary of cohort repeat behavior."""

    cohort_id: str
    cohort_size: int
    repeat_rate: float
    avg_orders_per_repeat_buyer: float


class Lens5Response(BaseModel):
    """Lens 5 analysis response."""

    analysis_name: str
    date_range: tuple[str, str]

    # Overall health
    health_score: float  # 0-100
    health_grade: str  # A-F
    total_customers: int
    total_active_customers: int

    # Key metrics
    overall_retention_rate: float
    cohort_quality_trend: str  # "improving", "stable", "declining"
    revenue_predictability_pct: float
    acquisition_dependence_pct: float

    # Summaries
    cohort_count: int
    top_revenue_cohorts: list[CohortRevenueSummary]
    top_repeat_cohorts: list[CohortBehaviorSummary]

    # Insights
    health_assessment: str  # Narrative health assessment
    key_strengths: list[str]
    key_risks: list[str]
    recommendations: list[str]


def _generate_health_assessment(metrics: Lens5Metrics) -> str:
    """Generate narrative health assessment."""
    hs = metrics.health_score
    grade = hs.health_grade
    score = float(hs.health_score)

    if grade == "A":
        return (
            f"Exceptional customer base health (score: {score:.1f}/100). "
            f"Strong retention ({hs.overall_retention_rate}%), "
            f"{hs.cohort_quality_trend} cohort quality, and "
            f"highly predictable revenue ({hs.revenue_predictability_pct}%)."
        )
    elif grade == "B":
        return (
            f"Good customer base health (score: {score:.1f}/100). "
            f"Solid retention ({hs.overall_retention_rate}%) and "
            f"{hs.cohort_quality_trend} cohort quality with "
            f"room for improvement in acquisition independence."
        )
    elif grade == "C":
        return (
            f"Moderate customer base health (score: {score:.1f}/100). "
            f"Retention at {hs.overall_retention_rate}% with "
            f"{hs.cohort_quality_trend} cohort quality. "
            f"Focus needed on retention and revenue predictability."
        )
    elif grade == "D":
        return (
            f"Below-average customer base health (score: {score:.1f}/100). "
            f"Retention concerns ({hs.overall_retention_rate}%) and "
            f"{hs.cohort_quality_trend} cohort quality. "
            f"Immediate attention required."
        )
    else:  # F
        return (
            f"Critical customer base health issues (score: {score:.1f}/100). "
            f"Low retention ({hs.overall_retention_rate}%), "
            f"{hs.cohort_quality_trend} cohort quality, and "
            f"high acquisition dependence. Urgent intervention needed."
        )


def _identify_key_strengths(metrics: Lens5Metrics) -> list[str]:
    """Identify key strengths in customer base."""
    hs = metrics.health_score
    strengths = []

    if hs.overall_retention_rate >= Decimal("80"):
        strengths.append(
            f"Strong customer retention at {hs.overall_retention_rate}%"
        )

    if hs.cohort_quality_trend == "improving":
        strengths.append("Cohort quality is improving over time")

    if hs.revenue_predictability_pct >= Decimal("70"):
        strengths.append(
            f"High revenue predictability ({hs.revenue_predictability_pct}%) "
            f"from existing cohorts"
        )

    if hs.acquisition_dependence_pct <= Decimal("20"):
        strengths.append(
            f"Low acquisition dependence ({hs.acquisition_dependence_pct}%) "
            f"indicates sustainable growth"
        )

    # Check cohort repeat rates
    high_repeat_cohorts = [
        c for c in metrics.cohort_repeat_behavior if c.repeat_rate >= Decimal("60")
    ]
    if len(high_repeat_cohorts) >= len(metrics.cohort_repeat_behavior) * 0.5:
        strengths.append(
            f"{len(high_repeat_cohorts)} cohorts show strong repeat purchase behavior (>60%)"
        )

    if not strengths:
        strengths.append("Baseline customer base established")

    return strengths


def _identify_key_risks(metrics: Lens5Metrics) -> list[str]:
    """Identify key risks in customer base."""
    hs = metrics.health_score
    risks = []

    if hs.overall_retention_rate < Decimal("60"):
        risks.append(
            f"CRITICAL: Low retention rate ({hs.overall_retention_rate}%) "
            f"indicates high customer churn"
        )
    elif hs.overall_retention_rate < Decimal("75"):
        risks.append(
            f"MEDIUM: Retention rate ({hs.overall_retention_rate}%) below optimal"
        )

    if hs.cohort_quality_trend == "declining":
        risks.append(
            "HIGH: Cohort quality is declining - newer cohorts performing worse"
        )

    if hs.revenue_predictability_pct < Decimal("50"):
        risks.append(
            f"HIGH: Low revenue predictability ({hs.revenue_predictability_pct}%) "
            f"creates forecasting challenges"
        )

    if hs.acquisition_dependence_pct > Decimal("40"):
        risks.append(
            f"HIGH: High acquisition dependence ({hs.acquisition_dependence_pct}%) "
            f"indicates growth driven by new customers rather than retention"
        )

    # Check for weak cohorts
    weak_cohorts = [
        c for c in metrics.cohort_repeat_behavior if c.repeat_rate < Decimal("30")
    ]
    if weak_cohorts:
        risks.append(
            f"MEDIUM: {len(weak_cohorts)} cohorts show poor repeat rates (<30%)"
        )

    if not risks:
        risks.append("No significant risks identified at this time")

    return risks


def _generate_recommendations(metrics: Lens5Metrics) -> list[str]:
    """Generate actionable recommendations."""
    hs = metrics.health_score
    recs = []

    # Retention recommendations
    if hs.overall_retention_rate < Decimal("70"):
        recs.append(
            "PRIORITY 1: Implement comprehensive retention program "
            "targeting at-risk customers with personalized engagement"
        )

    # Cohort quality recommendations
    if hs.cohort_quality_trend == "declining":
        recs.append(
            "PRIORITY 1: Analyze recent acquisition channels and "
            "customer onboarding to improve cohort quality"
        )
    elif hs.cohort_quality_trend == "stable":
        recs.append(
            "Experiment with acquisition improvements to drive "
            "cohort quality from stable to improving"
        )

    # Revenue predictability recommendations
    if hs.revenue_predictability_pct < Decimal("60"):
        recs.append(
            "PRIORITY 2: Focus on customer lifetime value optimization "
            "to increase revenue predictability"
        )

    # Acquisition dependence recommendations
    if hs.acquisition_dependence_pct > Decimal("35"):
        recs.append(
            "PRIORITY 2: Reduce acquisition dependence by investing in "
            "customer success, upsell, and cross-sell programs"
        )

    # Weak cohort recommendations
    weak_cohorts = [
        c for c in metrics.cohort_repeat_behavior if c.repeat_rate < Decimal("30")
    ]
    if weak_cohorts:
        cohort_ids = [c.cohort_id for c in weak_cohorts[:3]]
        recs.append(
            f"Investigate why cohorts {', '.join(cohort_ids)} have low repeat rates "
            f"and implement targeted retention campaigns"
        )

    # Positive recommendations
    if hs.health_grade in ("A", "B"):
        recs.append(
            "Maintain current strategies while testing incremental improvements "
            "to retention and cohort quality"
        )

    if not recs:
        recs.append(
            "Continue monitoring customer base health metrics "
            "and adjust strategies as needed"
        )

    return recs


async def _assess_customer_base_health_impl(
    request: Lens5Request, ctx: Context
) -> Lens5Response:
    """Implementation of Lens 5 analysis logic."""
    shared_state = get_shared_state()
    await ctx.info("Starting Lens 5: Overall Customer Base Health Analysis")

    # Get period aggregations and cohort assignments from context
    period_aggregations = shared_state.get("period_aggregations")
    cohort_assignments = shared_state.get("cohort_assignments")

    if period_aggregations is None:
        raise ValueError(
            "Period aggregations not found. Run build_customer_data_mart first."
        )

    if cohort_assignments is None:
        raise ValueError(
            "Cohort assignments not found. Run create_customer_cohorts first."
        )

    # Determine analysis window
    all_dates = [p.period_start for p in period_aggregations]
    min_date = min(all_dates)
    max_date = max(all_dates)

    analysis_start = request.analysis_start_date or min_date
    analysis_end = request.analysis_end_date or max_date

    await ctx.report_progress(
        0.2, f"Analyzing from {analysis_start.date()} to {analysis_end.date()}..."
    )

    # Run Lens 5 analysis
    lens5_result = assess_customer_base_health(
        period_aggregations=period_aggregations,
        cohort_assignments=cohort_assignments,
        analysis_start_date=analysis_start,
        analysis_end_date=analysis_end,
    )

    await ctx.report_progress(0.6, "Calculating cohort summaries...")

    # Generate summaries
    # Top revenue cohorts
    top_revenue = sorted(
        lens5_result.cohort_revenue_contributions,
        key=lambda x: x.total_revenue,
        reverse=True,
    )[:5]

    top_revenue_summaries = []
    total_revenue = sum(c.total_revenue for c in lens5_result.cohort_revenue_contributions)

    for contrib in top_revenue:
        if total_revenue > 0:
            pct = (float(contrib.total_revenue) / float(total_revenue)) * 100
        else:
            pct = 0.0

        top_revenue_summaries.append(
            CohortRevenueSummary(
                cohort_id=contrib.cohort_id,
                total_revenue=float(contrib.total_revenue),
                pct_of_total_revenue=pct,
                active_customers=contrib.active_customers,
            )
        )

    # Top repeat behavior cohorts
    top_repeat = sorted(
        lens5_result.cohort_repeat_behavior,
        key=lambda x: x.repeat_rate,
        reverse=True,
    )[:5]

    top_repeat_summaries = [
        CohortBehaviorSummary(
            cohort_id=c.cohort_id,
            cohort_size=c.cohort_size,
            repeat_rate=float(c.repeat_rate),
            avg_orders_per_repeat_buyer=float(c.avg_orders_per_repeat_buyer),
        )
        for c in top_repeat
    ]

    await ctx.report_progress(0.8, "Generating insights and recommendations...")

    # Generate insights
    health_assessment = _generate_health_assessment(lens5_result)
    key_strengths = _identify_key_strengths(lens5_result)
    key_risks = _identify_key_risks(lens5_result)
    recommendations = _generate_recommendations(lens5_result)

    # Store in context
    shared_state.set("lens5_result", lens5_result)

    await ctx.info(
        f"Lens 5 analysis complete - "
        f"Grade: {lens5_result.health_score.health_grade}, "
        f"Score: {float(lens5_result.health_score.health_score):.1f}/100"
    )

    return Lens5Response(
        analysis_name=request.analysis_name,
        date_range=(
            analysis_start.isoformat(),
            analysis_end.isoformat(),
        ),
        health_score=float(lens5_result.health_score.health_score),
        health_grade=lens5_result.health_score.health_grade,
        total_customers=lens5_result.health_score.total_customers,
        total_active_customers=lens5_result.health_score.total_active_customers,
        overall_retention_rate=float(lens5_result.health_score.overall_retention_rate),
        cohort_quality_trend=lens5_result.health_score.cohort_quality_trend,
        revenue_predictability_pct=float(
            lens5_result.health_score.revenue_predictability_pct
        ),
        acquisition_dependence_pct=float(
            lens5_result.health_score.acquisition_dependence_pct
        ),
        cohort_count=len(lens5_result.cohort_repeat_behavior),
        top_revenue_cohorts=top_revenue_summaries,
        top_repeat_cohorts=top_repeat_summaries,
        health_assessment=health_assessment,
        key_strengths=key_strengths,
        key_risks=key_risks,
        recommendations=recommendations,
    )


@mcp.tool()
async def assess_overall_customer_base_health(
    request: Lens5Request, ctx: Context
) -> Lens5Response:
    """
    Lens 5: Overall customer base health assessment.

    Provides an integrative view of customer base health by analyzing:
    - Revenue contributions across cohorts (C3 data)
    - Repeat purchase behavior by cohort
    - Overall health score and grade (0-100, A-F)
    - Cohort quality trends and revenue predictability

    This is the highest-level analysis that combines insights from
    all other lenses to provide a complete health assessment.

    Args:
        request: Configuration for Lens 5 analysis

    Returns:
        Comprehensive customer base health assessment with actionable insights
    """
    return await _assess_customer_base_health_impl(request, ctx)
