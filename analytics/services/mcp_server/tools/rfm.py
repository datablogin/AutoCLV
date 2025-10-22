"""RFM MCP Tool - Phase 1 Foundation Service"""

from datetime import datetime

import structlog
from customer_base_audit.foundation.rfm import calculate_rfm, calculate_rfm_scores
from fastmcp import Context
from pydantic import BaseModel, Field

from analytics.services.mcp_server.main import mcp
from analytics.services.mcp_server.state import get_shared_state

logger = structlog.get_logger(__name__)


class CalculateRFMRequest(BaseModel):
    """Request to calculate RFM metrics."""

    observation_end: datetime = Field(description="End date for RFM observation period")
    enable_parallel: bool = Field(
        default=True, description="Enable parallel processing for large datasets"
    )
    calculate_scores: bool = Field(
        default=True, description="Also calculate RFM scores (1-5 binning)"
    )
    storage_key: str = Field(
        default="rfm_metrics",
        description="Key to store RFM metrics in shared state (default: 'rfm_metrics'). "
        "Use 'rfm_metrics_period2' for second period in Lens 2 analysis.",
    )


class RFMResponse(BaseModel):
    """RFM calculation response."""

    metrics_count: int
    score_count: int
    date_range: tuple[str, str]
    parallel_enabled: bool
    storage_key: str


async def _calculate_rfm_metrics_impl(
    request: CalculateRFMRequest, ctx: Context
) -> RFMResponse:
    """Implementation of RFM calculation logic."""
    await ctx.info("Starting RFM calculation")

    # Get data mart from shared state
    shared_state = get_shared_state()
    mart = shared_state.get("data_mart")
    if mart is None:
        raise ValueError("Data mart not found. Run build_customer_data_mart first.")

    # Get period aggregations (use first granularity)
    first_granularity = list(mart.periods.keys())[0]
    period_aggregations = mart.periods[first_granularity]

    await ctx.report_progress(0.2, "Calculating RFM metrics...")

    # Calculate RFM
    rfm_metrics = calculate_rfm(
        period_aggregations=period_aggregations,
        observation_end=request.observation_end,
        parallel=request.enable_parallel,
    )

    await ctx.report_progress(0.7, "Calculating RFM scores...")

    # Calculate scores if requested
    rfm_scores = []
    if request.calculate_scores:
        rfm_scores = calculate_rfm_scores(rfm_metrics)

    # Validate RFM metrics were calculated
    if not rfm_metrics:
        raise ValueError(
            "No RFM metrics calculated. "
            "Check that period aggregations contain customer data."
        )

    # Store in shared state for reuse across tool calls
    # Use custom storage key if provided (enables Lens 2 with two periods)
    shared_state.set(request.storage_key, rfm_metrics)

    # Store scores with matching suffix if custom key used
    if request.storage_key != "rfm_metrics":
        scores_key = request.storage_key.replace("_metrics", "_scores")
        shared_state.set(scores_key, rfm_scores)
    else:
        shared_state.set("rfm_scores", rfm_scores)

    # Extract date range
    dates = [m.observation_start for m in rfm_metrics]
    date_range = (min(dates).isoformat(), max(dates).isoformat())

    await ctx.info(
        f"RFM calculation complete: {len(rfm_metrics)} customers stored as '{request.storage_key}'"
    )

    return RFMResponse(
        metrics_count=len(rfm_metrics),
        score_count=len(rfm_scores),
        date_range=date_range,
        parallel_enabled=request.enable_parallel,
        storage_key=request.storage_key,
    )


@mcp.tool()
async def calculate_rfm_metrics(
    request: CalculateRFMRequest, ctx: Context
) -> RFMResponse:
    """
    Calculate RFM (Recency, Frequency, Monetary) metrics.

    For Lens 2 period-to-period analysis, calculate RFM twice with different
    observation_end dates and storage_key values:
    - Period 1: storage_key="rfm_metrics", observation_end=<end of period 1>
    - Period 2: storage_key="rfm_metrics_period2", observation_end=<end of period 2>

    This tool transforms period aggregations into RFM metrics for each
    customer, which are the foundation for Lens 1 and Lens 2 analyses.

    Args:
        request: Configuration for RFM calculation

    Returns:
        Summary of calculated RFM metrics
    """
    return await _calculate_rfm_metrics_impl(request, ctx)
