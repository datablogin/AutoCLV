"""Lens 4 MCP Tool - Phase 2 Lens Service

Wraps Lens 4 (Multi-Cohort Comparison) as an MCP tool for agentic orchestration.
"""

import structlog
from customer_base_audit.analyses.lens4 import Lens4Metrics, compare_cohorts
from fastmcp import Context
from pydantic import BaseModel, Field

from analytics.services.mcp_server.instance import mcp
from analytics.services.mcp_server.state import get_shared_state

logger = structlog.get_logger(__name__)


class Lens4Thresholds:
    """Detection thresholds for cohort comparison analysis.

    These thresholds are calibrated to surface actionable insights while avoiding
    noise from minor statistical variations.
    """

    # Alert when best cohort generates >50% more revenue than worst
    REVENUE_VARIANCE_PCT = 50

    # Alert when average order value differs by >30%
    AOV_VARIANCE_PCT = 30

    # Alert when purchase frequency differs by >50% (1.5x multiplier)
    FREQUENCY_MULTIPLIER = 1.5

    # Alert when time-to-second-purchase differs by >50% (1.5x multiplier)
    TTSP_MULTIPLIER = 1.5

    # HIGH priority recommendation when revenue gap exceeds 2x
    HIGH_PRIORITY_REVENUE_GAP = 2.0

    # MEDIUM priority thresholds for frequency and AOV
    FREQUENCY_PRIORITY_MULTIPLIER = 1.5
    AOV_PRIORITY_MULTIPLIER = 1.3

    # Post-purchase engagement threshold (days)
    LONG_TTSP_DAYS = 60


class Lens4Request(BaseModel):
    """Request for Lens 4 analysis."""

    alignment_type: str = Field(
        default="left-aligned",
        description="Alignment mode: 'left-aligned' (by cohort age) or 'time-aligned' (by calendar period)",
    )
    include_margin: bool = Field(
        default=False, description="Include margin analysis (requires margin data)"
    )


class CohortSummary(BaseModel):
    """Summary metrics for a single cohort."""

    cohort_id: str
    cohort_size: int
    total_revenue: float
    avg_revenue_per_member: float
    avg_frequency: float
    avg_order_value: float


class Lens4Response(BaseModel):
    """Lens 4 analysis response."""

    cohort_count: int
    alignment_type: str

    # Decomposition summary
    cohort_summaries: list[CohortSummary]

    # Comparative insights
    best_performing_cohort: str | None
    worst_performing_cohort: str | None
    key_differences: list[str]
    recommendations: list[str]


def _generate_cohort_summaries(metrics: Lens4Metrics) -> list[CohortSummary]:
    """Generate summary statistics for each cohort."""
    # Group decompositions by cohort
    # Optimized: initialize on first encounter to avoid defaultdict lambda overhead
    cohort_data: dict[str, dict] = {}

    for decomp in metrics.cohort_decompositions:
        if decomp.cohort_id not in cohort_data:
            cohort_data[decomp.cohort_id] = {
                "cohort_size": decomp.cohort_size,
                "total_revenue": 0.0,
                "total_orders": 0,
                "periods": 0,
            }

        cohort_data[decomp.cohort_id]["total_revenue"] += float(decomp.revenue)
        cohort_data[decomp.cohort_id]["total_orders"] += decomp.total_orders
        cohort_data[decomp.cohort_id]["periods"] += 1

    # Calculate summary metrics
    summaries = []
    for cohort_id, data in cohort_data.items():
        cohort_size = data["cohort_size"]
        total_revenue = data["total_revenue"]
        total_orders = data["total_orders"]

        avg_revenue_per_member = total_revenue / cohort_size if cohort_size > 0 else 0.0
        avg_frequency = total_orders / cohort_size if cohort_size > 0 else 0.0
        avg_order_value = total_revenue / total_orders if total_orders > 0 else 0.0

        summaries.append(
            CohortSummary(
                cohort_id=cohort_id,
                cohort_size=cohort_size,
                total_revenue=total_revenue,
                avg_revenue_per_member=avg_revenue_per_member,
                avg_frequency=avg_frequency,
                avg_order_value=avg_order_value,
            )
        )

    # Sort by total revenue descending, then by cohort_id for deterministic ordering
    summaries.sort(key=lambda s: (-s.total_revenue, s.cohort_id))

    return summaries


def _identify_best_worst_cohorts(
    summaries: list[CohortSummary],
) -> tuple[str | None, str | None]:
    """Identify best and worst performing cohorts by revenue per member."""
    if not summaries:
        return None, None

    # Sort by avg revenue per member
    sorted_by_revenue = sorted(
        summaries, key=lambda s: s.avg_revenue_per_member, reverse=True
    )

    best = sorted_by_revenue[0].cohort_id
    worst = sorted_by_revenue[-1].cohort_id if len(sorted_by_revenue) > 1 else None

    return best, worst


def _identify_key_differences(
    metrics: Lens4Metrics, summaries: list[CohortSummary]
) -> list[str]:
    """Identify key differences between cohorts.

    Detection thresholds are defined in Lens4Thresholds class:
    - Revenue variance: Revenue variance percentage threshold
    - Frequency gap: Frequency multiplier for significant differences
    - AOV variance: AOV variance percentage threshold
    - TTSP variance: Time-to-second-purchase multiplier

    These thresholds are calibrated to surface actionable insights while avoiding
    noise from minor variations.
    """
    differences: list[str] = []

    if len(summaries) < 2:
        return differences

    # Revenue per member variance
    # Calculate percentage difference: (max - min) / max * 100
    # This shows how much more revenue the best cohort generates vs. the worst
    revenues = [s.avg_revenue_per_member for s in summaries]
    max_revenue = max(revenues)
    min_revenue = min(revenues)
    revenue_variance_pct = (
        (max_revenue - min_revenue) / max_revenue * 100 if max_revenue > 0 else 0
    )

    # Alert on significant revenue variance (indicates cohort quality difference)
    if revenue_variance_pct > Lens4Thresholds.REVENUE_VARIANCE_PCT:
        differences.append(
            f"High revenue variance: Best cohort generates {revenue_variance_pct:.0f}% "
            f"more revenue per member than worst cohort"
        )

    # Frequency differences
    # Alert when max frequency is >threshold (significant difference in purchase behavior)
    frequencies = [s.avg_frequency for s in summaries]
    max_freq = max(frequencies)
    min_freq = min(frequencies)
    if max_freq > min_freq * Lens4Thresholds.FREQUENCY_MULTIPLIER:
        differences.append(
            f"Purchase frequency varies significantly: {max_freq:.1f} vs {min_freq:.1f} orders per customer"
        )

    # AOV differences
    # Alert on significant AOV variance (indicates different customer value segments)
    aovs = [s.avg_order_value for s in summaries]
    max_aov = max(aovs)
    min_aov = min(aovs)
    aov_variance_pct = (max_aov - min_aov) / max_aov * 100 if max_aov > 0 else 0

    if aov_variance_pct > Lens4Thresholds.AOV_VARIANCE_PCT:
        differences.append(
            f"Average order value varies: ${max_aov:.2f} vs ${min_aov:.2f}"
        )

    # Time to second purchase insights
    # Alert when TTSP varies by >1.5x (indicates different engagement patterns)
    if metrics.time_to_second_purchase:
        ttsp_data = [
            float(t.mean_days)
            for t in metrics.time_to_second_purchase
            if t.mean_days is not None
        ]
        if len(ttsp_data) >= 2:
            max_ttsp = max(ttsp_data)
            min_ttsp = min(ttsp_data)
            # Avoid division by zero and false positives when min_ttsp is 0
            if max_ttsp > min_ttsp * Lens4Thresholds.TTSP_MULTIPLIER and min_ttsp > 0:
                differences.append(
                    f"Time to second purchase varies: {min_ttsp:.0f} to {max_ttsp:.0f} days"
                )

    return differences


def _generate_lens4_recommendations(
    metrics: Lens4Metrics, summaries: list[CohortSummary]
) -> list[str]:
    """Generate actionable recommendations based on cohort comparison."""
    recs = []

    if len(summaries) < 2:
        recs.append("Not enough cohorts for comparison. Add more cohorts to analyze.")
        return recs

    best, worst = _identify_best_worst_cohorts(summaries)

    if best and worst:
        best_cohort = next((s for s in summaries if s.cohort_id == best), None)
        worst_cohort = next((s for s in summaries if s.cohort_id == worst), None)

        if not best_cohort or not worst_cohort:
            recs.append("Insufficient cohort data for detailed recommendations.")
            return recs

        revenue_gap = (
            best_cohort.avg_revenue_per_member / worst_cohort.avg_revenue_per_member
            if worst_cohort.avg_revenue_per_member > 0
            else 0
        )

        if revenue_gap > Lens4Thresholds.HIGH_PRIORITY_REVENUE_GAP:
            recs.append(
                f"HIGH PRIORITY: Best cohort ({best}) generates {revenue_gap:.1f}x more revenue "
                f"than worst cohort ({worst}). Analyze {best} acquisition channels and replicate."
            )

        # Frequency insights
        if (
            best_cohort.avg_frequency
            > worst_cohort.avg_frequency * Lens4Thresholds.FREQUENCY_PRIORITY_MULTIPLIER
        ):
            recs.append(
                f"MEDIUM: Best cohort has {best_cohort.avg_frequency:.1f} orders per customer "
                f"vs {worst_cohort.avg_frequency:.1f}. Focus on increasing purchase frequency "
                f"for underperforming cohorts."
            )

        # AOV insights
        if (
            best_cohort.avg_order_value
            > worst_cohort.avg_order_value * Lens4Thresholds.AOV_PRIORITY_MULTIPLIER
        ):
            recs.append(
                f"MEDIUM: Best cohort has ${best_cohort.avg_order_value:.2f} AOV "
                f"vs ${worst_cohort.avg_order_value:.2f}. Implement upselling strategies "
                f"for lower-AOV cohorts."
            )

    # Time to second purchase insights
    if metrics.time_to_second_purchase:
        ttsp_with_data = [
            float(t.mean_days)
            for t in metrics.time_to_second_purchase
            if t.mean_days is not None
        ]
        avg_ttsp = sum(ttsp_with_data) / len(ttsp_with_data) if ttsp_with_data else 0

        if avg_ttsp > Lens4Thresholds.LONG_TTSP_DAYS:
            recs.append(
                f"MEDIUM: Average time to second purchase is {avg_ttsp:.0f} days. "
                f"Implement post-purchase engagement campaigns to accelerate repeat purchases."
            )

    if not recs:
        recs.append(
            "Cohort performance is relatively consistent. Continue current acquisition strategies."
        )

    return recs


async def _compare_multiple_cohorts_impl(
    request: Lens4Request, ctx: Context
) -> Lens4Response:
    """Implementation of Lens 4 analysis logic."""
    shared_state = get_shared_state()
    await ctx.info("Starting Lens 4 analysis")

    # Get cohort assignments from context
    cohort_assignments = shared_state.get("cohort_assignments")
    if not cohort_assignments:
        raise ValueError(
            "Cohort assignments not found or empty. "
            "Run create_customer_cohorts first and ensure customers are assigned."
        )

    # Get data mart from context
    mart = shared_state.get("data_mart")
    if mart is None:
        raise ValueError("Data mart not found. Run build_customer_data_mart first.")

    # Get period aggregations (use first granularity)
    first_granularity = list(mart.periods.keys())[0]
    period_aggregations = mart.periods[first_granularity]

    await ctx.report_progress(0.3, None)

    # Run Lens 4 analysis
    lens4_result = compare_cohorts(
        period_aggregations=period_aggregations,
        cohort_assignments=cohort_assignments,
        alignment_type=request.alignment_type,
        include_margin=request.include_margin,
    )

    await ctx.report_progress(0.7, None)

    # Generate summary metrics
    cohort_summaries = _generate_cohort_summaries(lens4_result)
    best_cohort, worst_cohort = _identify_best_worst_cohorts(cohort_summaries)
    key_differences = _identify_key_differences(lens4_result, cohort_summaries)
    recommendations = _generate_lens4_recommendations(lens4_result, cohort_summaries)

    # Store in context
    shared_state.set("lens4_result", lens4_result)

    await ctx.info("Lens 4 analysis complete")

    return Lens4Response(
        cohort_count=len(cohort_summaries),
        alignment_type=request.alignment_type,
        cohort_summaries=cohort_summaries,
        best_performing_cohort=best_cohort,
        worst_performing_cohort=worst_cohort,
        key_differences=key_differences,
        recommendations=recommendations,
    )


@mcp.tool()
async def compare_multiple_cohorts(
    request: Lens4Request, ctx: Context
) -> Lens4Response:
    """
    Lens 4: Multi-cohort comparison analysis.

    Compares multiple acquisition cohorts to identify:
    - Cohort-level performance differences
    - AOF, AOV, and margin decomposition
    - Time-to-second-purchase patterns

    This lens reveals which acquisition periods yield the best customers.

    Args:
        request: Configuration for Lens 4 analysis

    Returns:
        Multi-cohort comparison with performance rankings
    """
    return await _compare_multiple_cohorts_impl(request, ctx)
