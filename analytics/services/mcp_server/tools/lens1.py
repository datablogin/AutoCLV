"""Lens 1 MCP Tool - Phase 2 Lens Service

Wraps Lens 1 (Single Period Analysis) as an MCP tool for agentic orchestration.
"""

from decimal import Decimal

import structlog
from customer_base_audit.analyses.lens1 import analyze_single_period, Lens1Metrics
from fastmcp import Context
from pydantic import BaseModel, Field

from analytics.services.mcp_server.main import mcp
from analytics.services.mcp_server.state import get_shared_state

logger = structlog.get_logger(__name__)


class Lens1Request(BaseModel):
    """Request for Lens 1 analysis."""

    period_name: str = Field(
        default="Current Period", description="Name for this analysis period"
    )


class Lens1Response(BaseModel):
    """Lens 1 analysis response."""

    period_name: str
    total_customers: int
    one_time_buyers: int
    one_time_buyer_pct: float
    total_revenue: float
    top_10pct_revenue_contribution: float
    top_20pct_revenue_contribution: float
    avg_orders_per_customer: float
    median_customer_value: float
    rfm_distribution: dict[str, int]

    # Insights
    customer_health_score: float  # 0-100
    concentration_risk: str  # "low", "medium", "high"
    recommendations: list[str]


def _calculate_customer_health_score(metrics: Lens1Metrics) -> float:
    """Calculate 0-100 health score based on key indicators."""
    score = 100.0

    # Penalize high one-time buyer percentage
    if metrics.one_time_buyer_pct > Decimal("70"):
        score -= 30
    elif metrics.one_time_buyer_pct > Decimal("50"):
        score -= 15

    # Penalize extreme revenue concentration
    if metrics.top_10pct_revenue_contribution > Decimal("80"):
        score -= 20
    elif metrics.top_10pct_revenue_contribution > Decimal("60"):
        score -= 10

    # Reward healthy repeat purchase behavior
    if metrics.avg_orders_per_customer > Decimal("3"):
        score += 10

    return max(0.0, min(100.0, score))


def _assess_concentration_risk(metrics: Lens1Metrics) -> str:
    """Assess revenue concentration risk."""
    top10 = float(metrics.top_10pct_revenue_contribution)

    if top10 > 70:
        return "high"
    elif top10 > 50:
        return "medium"
    else:
        return "low"


def _generate_lens1_recommendations(metrics: Lens1Metrics) -> list[str]:
    """Generate actionable recommendations."""
    recs = []

    if metrics.one_time_buyer_pct > Decimal("60"):
        recs.append(
            f"HIGH: {metrics.one_time_buyer_pct}% one-time buyers. "
            f"Implement retention campaigns targeting first-time purchasers."
        )

    if metrics.top_10pct_revenue_contribution > Decimal("70"):
        recs.append(
            f"MEDIUM: Top 10% contribute {metrics.top_10pct_revenue_contribution}% of revenue. "
            f"Diversify customer base to reduce concentration risk."
        )

    if metrics.avg_orders_per_customer < Decimal("2"):
        recs.append(
            "MEDIUM: Low average orders per customer. "
            "Focus on repeat purchase incentives and loyalty programs."
        )

    if not recs:
        recs.append("Customer base health is strong. Maintain current strategies.")

    return recs


async def _analyze_single_period_snapshot_impl(
    request: Lens1Request, ctx: Context
) -> Lens1Response:
    """Implementation of Lens 1 analysis logic."""
    shared_state = get_shared_state()
    await ctx.info("Starting Lens 1 analysis")

    # Get RFM metrics from context
    rfm_metrics = shared_state.get("rfm_metrics")
    rfm_scores = shared_state.get("rfm_scores")

    if rfm_metrics is None:
        raise ValueError("RFM metrics not found. Run calculate_rfm_metrics first.")

    await ctx.report_progress(0.3, "Analyzing customer distribution...")

    # Run Lens 1 analysis
    lens1_result = analyze_single_period(
        rfm_metrics=rfm_metrics, rfm_scores=rfm_scores if rfm_scores else None
    )

    await ctx.report_progress(0.7, "Generating insights...")

    # Calculate insights
    health_score = _calculate_customer_health_score(lens1_result)
    concentration_risk = _assess_concentration_risk(lens1_result)
    recommendations = _generate_lens1_recommendations(lens1_result)

    # Store in context
    shared_state.set("lens1_result", lens1_result)

    await ctx.info("Lens 1 analysis complete")

    return Lens1Response(
        period_name=request.period_name,
        total_customers=lens1_result.total_customers,
        one_time_buyers=lens1_result.one_time_buyers,
        one_time_buyer_pct=float(lens1_result.one_time_buyer_pct),
        total_revenue=float(lens1_result.total_revenue),
        top_10pct_revenue_contribution=float(
            lens1_result.top_10pct_revenue_contribution
        ),
        top_20pct_revenue_contribution=float(
            lens1_result.top_20pct_revenue_contribution
        ),
        avg_orders_per_customer=float(lens1_result.avg_orders_per_customer),
        median_customer_value=float(lens1_result.median_customer_value),
        rfm_distribution=lens1_result.rfm_distribution,
        customer_health_score=health_score,
        concentration_risk=concentration_risk,
        recommendations=recommendations,
    )


@mcp.tool()
async def analyze_single_period_snapshot(
    request: Lens1Request, ctx: Context
) -> Lens1Response:
    """
    Lens 1: Single-period snapshot analysis.

    Analyzes customer base health for a single observation period, including:
    - Customer counts and one-time buyer percentage
    - Revenue distribution and concentration (Pareto analysis)
    - Average order frequency and median customer value
    - RFM segment distribution

    This is the foundation lens for understanding current customer base state.

    Args:
        request: Configuration for Lens 1 analysis

    Returns:
        Comprehensive single-period metrics with actionable insights
    """
    return await _analyze_single_period_snapshot_impl(request, ctx)
