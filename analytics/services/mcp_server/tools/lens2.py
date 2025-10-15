"""Lens 2 MCP Tool - Phase 2 Lens Service

Wraps Lens 2 (Period-to-Period Comparison) as an MCP tool for agentic orchestration.
"""

from decimal import Decimal

import structlog
from customer_base_audit.analyses.lens2 import Lens2Metrics, analyze_period_comparison
from fastmcp import Context
from pydantic import BaseModel, Field

from analytics.services.mcp_server.main import mcp
from analytics.services.mcp_server.state import get_shared_state

logger = structlog.get_logger(__name__)


class Lens2Request(BaseModel):
    """Request for Lens 2 analysis."""

    period1_name: str = Field(default="Period 1", description="Name for first period")
    period2_name: str = Field(default="Period 2", description="Name for second period")
    period1_rfm_key: str = Field(
        default="rfm_metrics",
        description="Context key for period 1 RFM metrics (default: 'rfm_metrics')",
    )
    period2_rfm_key: str = Field(
        default="rfm_metrics_period2",
        description="Context key for period 2 RFM metrics",
    )
    include_reactivation_analysis: bool = Field(
        default=False,
        description="Enable reactivation analysis (requires customer history in context)",
    )


class Lens2Response(BaseModel):
    """Lens 2 analysis response."""

    # Period summaries
    period1_name: str
    period2_name: str
    period1_customers: int
    period2_customers: int

    # Migration metrics
    retained_customers: int
    churned_customers: int
    new_customers: int
    reactivated_customers: int

    # Rates
    retention_rate: float
    churn_rate: float
    reactivation_rate: float

    # Growth metrics
    customer_count_change: int
    revenue_change_pct: float
    avg_order_value_change_pct: float | None

    # Insights
    growth_momentum: str  # "strong", "moderate", "declining", "negative"
    key_drivers: list[str]
    recommendations: list[str]


def _assess_growth_momentum(metrics: Lens2Metrics) -> str:
    """Assess growth momentum based on key metrics."""
    # Strong growth: positive customer growth and revenue growth > 10%
    if metrics.customer_count_change > 0 and metrics.revenue_change_pct > Decimal("10"):
        return "strong"

    # Moderate growth: positive customer growth or revenue growth 0-10%
    if metrics.customer_count_change > 0 or metrics.revenue_change_pct > Decimal("0"):
        return "moderate"

    # Declining: negative customer growth but positive revenue
    if metrics.customer_count_change < 0 and metrics.revenue_change_pct > Decimal("0"):
        return "declining"

    # Negative: negative customer and revenue growth
    return "negative"


def _identify_key_drivers(metrics: Lens2Metrics) -> list[str]:
    """Identify key drivers of period-to-period changes."""
    drivers = []

    # Retention impact
    if metrics.retention_rate > Decimal("80"):
        drivers.append(
            f"Strong retention: {metrics.retention_rate}% of customers remained active"
        )
    elif metrics.retention_rate < Decimal("50"):
        drivers.append(
            f"Weak retention: Only {metrics.retention_rate}% of customers remained active"
        )

    # Churn impact
    if metrics.churn_rate > Decimal("30"):
        drivers.append(
            f"High churn: {metrics.churn_rate}% of customers churned from period 1"
        )

    # Acquisition impact
    new_customer_pct = (
        len(metrics.migration.new)
        / float(metrics.period2_metrics.total_customers)
        * 100
        if metrics.period2_metrics.total_customers > 0
        else 0
    )
    if new_customer_pct > 30:
        drivers.append(
            f"Strong acquisition: {new_customer_pct:.1f}% of period 2 customers are new"
        )

    # Revenue dynamics
    if abs(metrics.revenue_change_pct) > Decimal("20"):
        direction = "increased" if metrics.revenue_change_pct > 0 else "decreased"
        drivers.append(
            f"Revenue {direction} significantly: {abs(metrics.revenue_change_pct)}%"
        )

    return drivers


def _generate_lens2_recommendations(metrics: Lens2Metrics) -> list[str]:
    """Generate actionable recommendations based on period comparison."""
    recs = []

    # Retention recommendations
    if metrics.retention_rate < Decimal("60"):
        recs.append(
            f"HIGH PRIORITY: Retention rate is {metrics.retention_rate}%. "
            f"Implement win-back campaigns and loyalty programs."
        )

    # Churn recommendations
    if metrics.churn_rate > Decimal("40"):
        recs.append(
            f"HIGH PRIORITY: Churn rate is {metrics.churn_rate}%. "
            f"Analyze churned customer segments to identify at-risk patterns."
        )

    # Growth recommendations
    if metrics.customer_count_change < 0:
        recs.append(
            "MEDIUM: Customer base shrinking. Increase acquisition efforts "
            "and improve retention programs."
        )

    # Revenue recommendations
    if metrics.revenue_change_pct < Decimal("-10"):
        recs.append(
            f"HIGH PRIORITY: Revenue declined {abs(metrics.revenue_change_pct)}%. "
            f"Focus on increasing average order value and purchase frequency."
        )

    # Positive momentum
    if (
        metrics.retention_rate > Decimal("70")
        and metrics.customer_count_change > 0
        and metrics.revenue_change_pct > Decimal("5")
    ):
        recs.append(
            "Business momentum is strong. Continue current strategies and "
            "monitor for sustainability."
        )

    if not recs:
        recs.append("Period-to-period dynamics are stable. Monitor key metrics.")

    return recs


async def _analyze_period_comparison_impl(
    request: Lens2Request, ctx: Context
) -> Lens2Response:
    """Implementation of Lens 2 analysis logic."""
    shared_state = get_shared_state()
    await ctx.info("Starting Lens 2 analysis")

    # Get RFM metrics for both periods from context
    period1_rfm = shared_state.get(request.period1_rfm_key)
    period2_rfm = shared_state.get(request.period2_rfm_key)

    if period1_rfm is None:
        raise ValueError(
            f"Period 1 RFM metrics not found in context (key: '{request.period1_rfm_key}'). "
            f"Calculate RFM for period 1 first."
        )

    if period2_rfm is None:
        raise ValueError(
            f"Period 2 RFM metrics not found in context (key: '{request.period2_rfm_key}'). "
            f"Calculate RFM for period 2 first."
        )

    await ctx.report_progress(0.3, "Analyzing customer migration...")

    # Get customer history if reactivation analysis enabled
    all_customer_history = None
    if request.include_reactivation_analysis:
        all_customer_history = shared_state.get("customer_history")
        if all_customer_history is None:
            logger.warning(
                "Reactivation analysis enabled but customer_history not found in context"
            )

    # Run Lens 2 analysis
    lens2_result = analyze_period_comparison(
        period1_rfm=period1_rfm,
        period2_rfm=period2_rfm,
        all_customer_history=all_customer_history,
    )

    await ctx.report_progress(0.7, "Generating insights...")

    # Calculate insights
    growth_momentum = _assess_growth_momentum(lens2_result)
    key_drivers = _identify_key_drivers(lens2_result)
    recommendations = _generate_lens2_recommendations(lens2_result)

    # Store in context
    shared_state.set("lens2_result", lens2_result)

    await ctx.info("Lens 2 analysis complete")

    # Calculate AOV change if available
    aov_change_pct = None
    if lens2_result.avg_order_value_change_pct is not None:
        aov_change_pct = float(lens2_result.avg_order_value_change_pct)

    return Lens2Response(
        period1_name=request.period1_name,
        period2_name=request.period2_name,
        period1_customers=lens2_result.period1_metrics.total_customers,
        period2_customers=lens2_result.period2_metrics.total_customers,
        retained_customers=len(lens2_result.migration.retained),
        churned_customers=len(lens2_result.migration.churned),
        new_customers=len(lens2_result.migration.new),
        reactivated_customers=len(lens2_result.migration.reactivated),
        retention_rate=float(lens2_result.retention_rate),
        churn_rate=float(lens2_result.churn_rate),
        reactivation_rate=float(lens2_result.reactivation_rate),
        customer_count_change=lens2_result.customer_count_change,
        revenue_change_pct=float(lens2_result.revenue_change_pct),
        avg_order_value_change_pct=aov_change_pct,
        growth_momentum=growth_momentum,
        key_drivers=key_drivers,
        recommendations=recommendations,
    )


@mcp.tool()
async def analyze_period_to_period_comparison(
    request: Lens2Request, ctx: Context
) -> Lens2Response:
    """
    Lens 2: Period-to-period comparison analysis.

    Compares two time periods to track customer migration patterns:
    - Retention, churn, and reactivation rates
    - New customer acquisition
    - Revenue and AOV trends

    This lens reveals customer lifecycle dynamics and business momentum.

    Note: This tool requires two separate RFM calculations to be stored in context.
    By default, it looks for 'rfm_metrics' (period 1) and 'rfm_metrics_period2' (period 2).

    Args:
        request: Configuration for Lens 2 analysis

    Returns:
        Period comparison metrics with growth insights
    """
    return await _analyze_period_comparison_impl(request, ctx)
