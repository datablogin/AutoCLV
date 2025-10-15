"""Data Mart MCP Tool - Phase 1 Foundation Service"""

from fastmcp import Context
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal
from pathlib import Path
import json
from customer_base_audit.foundation.data_mart import (
    CustomerDataMartBuilder,
    PeriodGranularity,
)
from analytics.services.mcp_server.main import mcp
import structlog

logger = structlog.get_logger(__name__)

MAX_INPUT_BYTES = 25 * 1024 * 1024  # 25 MiB cap to avoid accidental OOM


def _load_transactions(path: Path) -> list[dict]:
    """Load transactions from JSON file (adapted from cli.py)."""
    resolved = path.resolve()
    size = resolved.stat().st_size
    if size > MAX_INPUT_BYTES:
        raise ValueError(
            f"Input file {resolved} is {size} bytes; exceeds limit of {MAX_INPUT_BYTES} bytes"
        )
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, list):
        raise ValueError("Expected a list of transactions in the input file")
    return [dict(item) for item in payload]


class BuildDataMartRequest(BaseModel):
    """Request to build customer data mart."""
    transaction_data_path: str = Field(
        description="Path to transaction data (JSON)"
    )
    period_granularities: list[Literal["month", "quarter", "year"]] = Field(
        default=["quarter", "year"],
        description="Period granularities to compute"
    )


class DataMartResponse(BaseModel):
    """Data mart build response."""
    order_count: int
    period_count: int
    customer_count: int
    granularities: list[str]
    date_range: tuple[str, str]


async def _build_customer_data_mart_impl(
    request: BuildDataMartRequest,
    ctx: Context,
    transactions: list[dict] | None = None
) -> DataMartResponse:
    """Implementation of data mart building logic.

    Args:
        request: Configuration for data mart build
        ctx: MCP context
        transactions: Optional pre-loaded transactions (for testing)
    """
    await ctx.info(f"Building data mart from {request.transaction_data_path}")

    # Parse granularities
    granularities = tuple(
        PeriodGranularity(g)
        for g in request.period_granularities
    )

    # Build data mart
    builder = CustomerDataMartBuilder(period_granularities=granularities)

    # Load transactions (or use provided ones for testing)
    if transactions is None:
        transactions = _load_transactions(Path(request.transaction_data_path))

    await ctx.report_progress(0.3, "Aggregating orders...")
    mart = builder.build(transactions)

    await ctx.report_progress(0.9, "Finalizing...")

    # Extract summary
    all_periods = []
    for granularity, periods in mart.periods.items():
        all_periods.extend(periods)

    dates = [p.period_start for p in all_periods]
    date_range = (
        min(dates).isoformat() if dates else "",
        max(dates).isoformat() if dates else ""
    )

    # Store in context for reuse
    ctx.set_state("data_mart", mart)

    await ctx.info("Data mart built successfully")

    return DataMartResponse(
        order_count=len(mart.orders),
        period_count=len(all_periods),
        customer_count=len(set(p.customer_id for p in all_periods)),
        granularities=[g.value for g in granularities],
        date_range=date_range
    )


@mcp.tool()
async def build_customer_data_mart(
    request: BuildDataMartRequest,
    ctx: Context
) -> DataMartResponse:
    """
    Build customer data mart from raw transaction data.

    This tool aggregates raw transactions into order-level and period-level
    summaries, which are the foundation for all Four Lenses analyses.

    Args:
        request: Configuration for data mart build

    Returns:
        Summary statistics about the built data mart
    """
    return await _build_customer_data_mart_impl(request, ctx)
