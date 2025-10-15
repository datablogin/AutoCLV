"""RFM MCP Tool - Phase 1 Foundation Service"""

from fastmcp import Context
from pydantic import BaseModel, Field
from datetime import datetime
from customer_base_audit.foundation.rfm import calculate_rfm, calculate_rfm_scores
from analytics.services.mcp_server.main import mcp
import structlog

logger = structlog.get_logger(__name__)


class CalculateRFMRequest(BaseModel):
    """Request to calculate RFM metrics."""
    observation_end: datetime = Field(
        description="End date for RFM observation period"
    )
    enable_parallel: bool = Field(
        default=True,
        description="Enable parallel processing for large datasets"
    )
    calculate_scores: bool = Field(
        default=True,
        description="Also calculate RFM scores (1-5 binning)"
    )


class RFMResponse(BaseModel):
    """RFM calculation response."""
    metrics_count: int
    score_count: int
    date_range: tuple[str, str]
    parallel_enabled: bool


async def _calculate_rfm_metrics_impl(
    request: CalculateRFMRequest,
    ctx: Context
) -> RFMResponse:
    """Implementation of RFM calculation logic."""
    await ctx.info("Starting RFM calculation")

    # Get data mart from context
    mart = ctx.get_state("data_mart")
    if mart is None:
        raise ValueError(
            "Data mart not found. Run build_customer_data_mart first."
        )

    # Get period aggregations (use first granularity)
    first_granularity = list(mart.periods.keys())[0]
    period_aggregations = mart.periods[first_granularity]

    await ctx.report_progress(0.2, "Calculating RFM metrics...")

    # Calculate RFM
    rfm_metrics = calculate_rfm(
        period_aggregations=period_aggregations,
        observation_end=request.observation_end,
        parallel=request.enable_parallel
    )

    await ctx.report_progress(0.7, "Calculating RFM scores...")

    # Calculate scores if requested
    rfm_scores = []
    if request.calculate_scores:
        rfm_scores = calculate_rfm_scores(rfm_metrics)

    # Store in context
    ctx.set_state("rfm_metrics", rfm_metrics)
    ctx.set_state("rfm_scores", rfm_scores)

    # Extract date range
    dates = [m.observation_start for m in rfm_metrics]
    date_range = (
        min(dates).isoformat() if dates else "",
        max(dates).isoformat() if dates else ""
    )

    await ctx.info(f"RFM calculation complete: {len(rfm_metrics)} customers")

    return RFMResponse(
        metrics_count=len(rfm_metrics),
        score_count=len(rfm_scores),
        date_range=date_range,
        parallel_enabled=request.enable_parallel
    )


@mcp.tool()
async def calculate_rfm_metrics(
    request: CalculateRFMRequest,
    ctx: Context
) -> RFMResponse:
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics.

    This tool transforms period aggregations into RFM metrics for each
    customer, which are the foundation for Lens 1 and Lens 2 analyses.

    Args:
        request: Configuration for RFM calculation

    Returns:
        Summary of calculated RFM metrics
    """
    return await _calculate_rfm_metrics_impl(request, ctx)
