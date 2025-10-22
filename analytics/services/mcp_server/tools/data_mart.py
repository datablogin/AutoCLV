"""Data Mart MCP Tool - Phase 1 Foundation Service"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import structlog
from customer_base_audit.foundation.data_mart import (
    CustomerDataMartBuilder,
    PeriodGranularity,
)
from fastmcp import Context
from pydantic import BaseModel, Field

from analytics.services.mcp_server.main import mcp
from analytics.services.mcp_server.state import get_shared_state

logger = structlog.get_logger(__name__)

MAX_INPUT_BYTES = 25 * 1024 * 1024  # 25 MiB cap to avoid accidental OOM

# Allowed base directory for transaction data files
# Default to current working directory, can be overridden via environment variable
ALLOWED_BASE_DIR = Path(os.environ.get("MCP_DATA_DIR", os.getcwd())).resolve()


def _load_transactions(path: Path) -> list[dict]:
    """Load transactions from JSON file and parse datetime fields.

    Validates that the resolved path is within the allowed base directory
    to prevent path traversal attacks.

    Raises:
        ValueError: If path is outside allowed directory or file is too large
    """
    resolved = path.resolve()

    # Validate path is within allowed directory
    try:
        resolved.relative_to(ALLOWED_BASE_DIR)
    except ValueError as e:
        raise ValueError(
            f"Path {resolved} is outside allowed directory {ALLOWED_BASE_DIR}. "
            f"Only files within the allowed directory can be loaded."
        ) from e

    # Validate file size
    size = resolved.stat().st_size
    if size > MAX_INPUT_BYTES:
        raise ValueError(
            f"Input file {resolved} is {size} bytes; exceeds limit of {MAX_INPUT_BYTES} bytes"
        )
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, list):
        raise ValueError("Expected a list of transactions in the input file")

    # Parse and validate datetime fields
    transactions = []
    for idx, item in enumerate(payload):
        txn = dict(item)

        # Parse or validate order_ts or event_ts if present
        for ts_field in ["order_ts", "event_ts"]:
            if ts_field in txn:
                value = txn[ts_field]

                # Parse string to datetime
                if isinstance(value, str):
                    try:
                        txn[ts_field] = datetime.fromisoformat(value)
                    except (ValueError, TypeError) as e:
                        raise ValueError(
                            f"Transaction {idx}: Failed to parse {ts_field} as datetime: {value}"
                        ) from e

                # Validate non-string values are datetime objects
                elif not isinstance(value, datetime):
                    raise TypeError(
                        f"Transaction {idx}: {ts_field} must be a datetime or ISO format string, "
                        f"got {type(value).__name__}"
                    )

        transactions.append(txn)

    return transactions


class BuildDataMartRequest(BaseModel):
    """Request to build customer data mart."""

    transaction_data_path: str = Field(description="Path to transaction data (JSON)")
    period_granularities: list[Literal["month", "quarter", "year"]] = Field(
        default=["quarter", "year"], description="Period granularities to compute"
    )
    max_transaction_date: datetime | None = Field(
        default=None,
        description="Optional: Filter transactions to only include those on or before this date. "
        "Required for period-to-period comparison (Lens 2) to create period-specific data marts.",
    )


class DataMartResponse(BaseModel):
    """Data mart build response."""

    order_count: int
    period_count: int
    customer_count: int
    granularities: list[str]
    date_range: tuple[str, str]


async def _build_customer_data_mart_impl(
    request: BuildDataMartRequest, ctx: Context, transactions: list[dict] | None = None
) -> DataMartResponse:
    """Implementation of data mart building logic.

    Args:
        request: Configuration for data mart build
        ctx: MCP context
        transactions: Optional pre-loaded transactions (for testing)
    """
    await ctx.info(f"Building data mart from {request.transaction_data_path}")

    # Parse granularities
    granularities = tuple(PeriodGranularity(g) for g in request.period_granularities)

    # Build data mart
    builder = CustomerDataMartBuilder(period_granularities=granularities)

    # Load transactions (or use provided ones for testing)
    if transactions is None:
        transactions = _load_transactions(Path(request.transaction_data_path))

    # Filter transactions by max_transaction_date if specified
    # Prefer order_ts if available, otherwise use event_ts for consistency
    if request.max_transaction_date is not None:
        original_count = len(transactions)
        transactions = [
            txn
            for txn in transactions
            if (
                # Prefer order_ts if available
                ("order_ts" in txn and txn["order_ts"] <= request.max_transaction_date)
                or (
                    # Use event_ts only if order_ts not present
                    "order_ts" not in txn
                    and "event_ts" in txn
                    and txn["event_ts"] <= request.max_transaction_date
                )
            )
        ]
        filtered_count = len(transactions)
        await ctx.info(
            f"Filtered transactions by max_transaction_date: {filtered_count} of {original_count} transactions retained "
            f"(preferring order_ts over event_ts for consistency)"
        )

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
        max(dates).isoformat() if dates else "",
    )

    # Store in shared state for reuse across tool calls
    shared_state = get_shared_state()
    shared_state.set("data_mart", mart)

    # Store period aggregations separately for Lens 5
    # Use first granularity (typically quarter or month)
    first_granularity = list(mart.periods.keys())[0]
    period_aggregations = mart.periods[first_granularity]
    shared_state.set("period_aggregations", period_aggregations)

    await ctx.info(
        "Data mart built successfully",
        extra=f"Stored {len(period_aggregations)} period aggregations for Lens 5",
    )

    return DataMartResponse(
        order_count=len(mart.orders),
        period_count=len(all_periods),
        customer_count=len(set(p.customer_id for p in all_periods)),
        granularities=[g.value for g in granularities],
        date_range=date_range,
    )


@mcp.tool()
async def build_customer_data_mart(
    request: BuildDataMartRequest, ctx: Context
) -> DataMartResponse:
    """
    Build customer data mart from raw transaction data.

    This tool aggregates raw transactions into order-level and period-level
    summaries, which are the foundation for all Four Lenses analyses.

    For Lens 2 period-to-period comparison, build the data mart twice with different
    max_transaction_date values:
    - Period 1: max_transaction_date=<end of period 1> (e.g., 2024-03-31)
    - Period 2: max_transaction_date=<end of period 2> (e.g., 2024-06-30)

    Then calculate RFM for each period and run Lens 2 analysis.

    Args:
        request: Configuration for data mart build

    Returns:
        Summary statistics about the built data mart
    """
    return await _build_customer_data_mart_impl(request, ctx)
