"""Lens 3 MCP Tool - Phase 2 Lens Service

Wraps Lens 3 (Single Cohort Evolution) as an MCP tool for agentic orchestration.

Phase 4B: Added distributed tracing for Lens 3 execution
"""

import structlog
from customer_base_audit.analyses.lens3 import Lens3Metrics, analyze_cohort_evolution
from fastmcp import Context
from opentelemetry import trace
from pydantic import BaseModel, Field

from analytics.services.mcp_server.instance import mcp
from analytics.services.mcp_server.state import get_shared_state

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)


class Lens3Request(BaseModel):
    """Request for Lens 3 analysis."""

    cohort_id: str = Field(description="Cohort identifier to analyze")


class Lens3Response(BaseModel):
    """Lens 3 analysis response."""

    cohort_id: str
    cohort_size: int
    acquisition_date: str
    periods_analyzed: int

    # Key metrics by period
    activation_curve: dict[int, float]  # period -> cumulative activation rate
    revenue_curve: dict[int, float]  # period -> avg revenue per cohort member
    retention_curve: dict[int, float]  # period -> active customer rate

    # Insights
    cohort_maturity: str  # "early", "growth", "mature", "declining"
    ltv_trajectory: str  # "strong", "moderate", "weak"
    recommendations: list[str]


def _assess_cohort_maturity(metrics: Lens3Metrics) -> str:
    """Assess cohort maturity based on lifecycle stage."""
    num_periods = len(metrics.periods)

    if num_periods <= 2:
        return "early"
    elif num_periods <= 6:
        # Check if growth is still happening
        if num_periods >= 3:
            recent_retention_pct = (
                float(metrics.periods[-1].active_customers) / metrics.cohort_size * 100
            )
            if recent_retention_pct > 30:
                return "growth"
        return "growth"
    elif num_periods <= 12:
        # Check retention stability
        if num_periods >= 3:
            recent_retention_pct = (
                float(metrics.periods[-1].active_customers) / metrics.cohort_size * 100
            )
            if recent_retention_pct > 20:
                return "mature"
        return "mature"
    else:
        # Long-lived cohort - check if declining
        if num_periods >= 3:
            recent_retention_pct = (
                float(metrics.periods[-1].active_customers) / metrics.cohort_size * 100
            )
            if recent_retention_pct < 10:
                return "declining"
        return "mature"


def _assess_ltv_trajectory(metrics: Lens3Metrics) -> str:
    """Assess lifetime value trajectory."""
    if len(metrics.periods) < 2:
        return "moderate"

    # Calculate total revenue per cohort member
    total_revenue_per_member = sum(
        float(p.avg_revenue_per_cohort_member) for p in metrics.periods
    )

    # Strong LTV: > $500 total per member
    if total_revenue_per_member > 500:
        return "strong"

    # Check revenue growth trend
    if len(metrics.periods) >= 3:
        recent_revenue = float(metrics.periods[-1].avg_revenue_per_cohort_member)
        early_revenue = float(metrics.periods[0].avg_revenue_per_cohort_member)

        # If recent period has higher revenue than acquisition period
        if recent_revenue > early_revenue * 0.5:
            return "moderate"

    # Weak trajectory if revenue is low or declining quickly
    if total_revenue_per_member < 100:
        return "weak"

    return "moderate"


def _generate_lens3_recommendations(metrics: Lens3Metrics) -> list[str]:
    """Generate actionable recommendations based on cohort evolution."""
    recs = []

    # Retention recommendations
    if len(metrics.periods) >= 2:
        period1_retention = (
            float(metrics.periods[1].active_customers) / metrics.cohort_size * 100
        )

        if period1_retention < 30:
            recs.append(
                f"HIGH PRIORITY: First-period retention is {period1_retention:.1f}%. "
                f"Improve onboarding and early-stage engagement."
            )

    # Revenue per member recommendations
    if len(metrics.periods) >= 3:
        avg_revenue = sum(
            float(p.avg_revenue_per_cohort_member) for p in metrics.periods
        ) / len(metrics.periods)

        if avg_revenue < 50:
            recs.append(
                f"MEDIUM: Average revenue per member is low (${avg_revenue:.2f}). "
                f"Focus on increasing purchase frequency and order value."
            )

    # Activation rate recommendations
    if len(metrics.periods) >= 1:
        cumulative_activation = float(metrics.periods[-1].cumulative_activation_rate)

        if cumulative_activation < 50:
            recs.append(
                f"MEDIUM: Only {cumulative_activation:.1f}% of cohort has activated. "
                f"Implement activation campaigns for dormant customers."
            )

    # Cohort-specific insights
    if len(metrics.periods) >= 3:
        # Get last 3 periods safely
        recent_3_retention = [
            (float(p.active_customers) / metrics.cohort_size * 100)
            for p in metrics.periods[-3:]
        ]
        if recent_3_retention and all(r < 10 for r in recent_3_retention):
            recs.append(
                "Cohort showing signs of exhaustion. Consider targeted win-back campaigns."
            )

    if not recs:
        recs.append("Cohort performance is healthy. Continue monitoring.")

    return recs


async def _analyze_cohort_lifecycle_impl(
    request: Lens3Request, ctx: Context
) -> Lens3Response:
    """Implementation of Lens 3 analysis logic."""
    with tracer.start_as_current_span("lens3_execution") as span:
        shared_state = get_shared_state()
        await ctx.info(f"Starting Lens 3 analysis for cohort {request.cohort_id}")

        span.set_attribute("cohort_id", request.cohort_id)

        # Get cohort data from context
        cohort_definitions = shared_state.get("cohort_definitions")
        cohort_assignments = shared_state.get("cohort_assignments")

        if cohort_definitions is None or cohort_assignments is None:
            raise ValueError(
                "Cohort data not found. Run create_customer_cohorts first."
            )

        # Find the requested cohort
        target_cohort = None
        for cohort_def in cohort_definitions:
            if cohort_def.cohort_id == request.cohort_id:
                target_cohort = cohort_def
                break

        if target_cohort is None:
            available_cohorts = [c.cohort_id for c in cohort_definitions]
            raise ValueError(
                f"Cohort '{request.cohort_id}' not found. "
                f"Available cohorts: {available_cohorts}"
            )

        # Get customer IDs for this cohort
        cohort_customer_ids = [
            cust_id
            for cust_id, assigned_cohort in cohort_assignments.items()
            if assigned_cohort == request.cohort_id
        ]

        # Validate cohort has customers assigned
        if not cohort_customer_ids:
            raise ValueError(
                f"No customers assigned to cohort '{request.cohort_id}'. "
                f"Cohort may be empty or cohort assignments may be incorrect."
            )

        span.set_attribute("cohort_size", len(cohort_customer_ids))
        span.set_attribute("acquisition_date", target_cohort.start_date.isoformat())

        # Get data mart from context
        mart = shared_state.get("data_mart")
        if mart is None:
            raise ValueError("Data mart not found. Run build_customer_data_mart first.")

        # Get period aggregations (use first granularity)
        if not mart.periods:
            raise ValueError(
                "Data mart has no period aggregations. Ensure data mart was built correctly."
            )
        first_granularity = list(mart.periods.keys())[0]
        period_aggregations = mart.periods[first_granularity]

        await ctx.report_progress(0.3, "Analyzing cohort evolution...")

        # Run Lens 3 analysis
        with tracer.start_as_current_span("lens3_calculate"):
            lens3_result = analyze_cohort_evolution(
                cohort_name=request.cohort_id,
                acquisition_date=target_cohort.start_date,
                period_aggregations=period_aggregations,
                cohort_customer_ids=cohort_customer_ids,
            )

        await ctx.report_progress(0.7, "Generating insights...")

        # Calculate insights
        with tracer.start_as_current_span("lens3_generate_insights"):
            cohort_maturity = _assess_cohort_maturity(lens3_result)
            ltv_trajectory = _assess_ltv_trajectory(lens3_result)
            recommendations = _generate_lens3_recommendations(lens3_result)

        # Build metric curves
        activation_curve = {
            p.period_number: float(p.cumulative_activation_rate)
            for p in lens3_result.periods
        }
        revenue_curve = {
            p.period_number: float(p.avg_revenue_per_cohort_member)
            for p in lens3_result.periods
        }
        retention_curve = {
            p.period_number: (
                float(p.active_customers) / lens3_result.cohort_size * 100
            )
            for p in lens3_result.periods
        }

        # Store in context
        shared_state.set("lens3_result", lens3_result)

        # Add result metrics to span
        span.set_attribute("periods_analyzed", len(lens3_result.periods))
        span.set_attribute("cohort_maturity", cohort_maturity)
        span.set_attribute("ltv_trajectory", ltv_trajectory)
        if lens3_result.periods:
            span.set_attribute(
                "final_activation_rate",
                float(lens3_result.periods[-1].cumulative_activation_rate),
            )

        await ctx.info(f"Lens 3 analysis complete for cohort {request.cohort_id}")

        return Lens3Response(
            cohort_id=request.cohort_id,
            cohort_size=lens3_result.cohort_size,
            acquisition_date=lens3_result.acquisition_date.isoformat(),
            periods_analyzed=len(lens3_result.periods),
            activation_curve=activation_curve,
            revenue_curve=revenue_curve,
            retention_curve=retention_curve,
            cohort_maturity=cohort_maturity,
            ltv_trajectory=ltv_trajectory,
            recommendations=recommendations,
        )


@mcp.tool()
async def analyze_cohort_lifecycle(
    request: Lens3Request, ctx: Context
) -> Lens3Response:
    """
    Lens 3: Single cohort evolution analysis.

    Tracks a single acquisition cohort through their lifecycle:
    - Cumulative activation rates over time
    - Revenue per cohort member by period
    - Retention patterns and churn dynamics

    This lens reveals cohort maturity and lifetime value patterns.

    Args:
        request: Configuration for Lens 3 analysis

    Returns:
        Cohort lifecycle metrics with maturity assessment
    """
    return await _analyze_cohort_lifecycle_impl(request, ctx)
