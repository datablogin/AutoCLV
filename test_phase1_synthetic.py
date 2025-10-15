#!/usr/bin/env python
"""Test Phase 1 Foundation Services with synthetic data.

This script demonstrates how to use the Phase 1 MCP tools with your own data.
"""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from analytics.services.mcp_server.tools.data_mart import (
    _build_customer_data_mart_impl as build_customer_data_mart,
    BuildDataMartRequest,
)
from analytics.services.mcp_server.tools.rfm import (
    _calculate_rfm_metrics_impl as calculate_rfm_metrics,
    CalculateRFMRequest,
)
from analytics.services.mcp_server.tools.cohorts import (
    _create_customer_cohorts_impl as create_customer_cohorts,
    CreateCohortsRequest,
)


def create_mock_context():
    """Create a mock MCP context for standalone testing."""
    ctx = AsyncMock()
    ctx.state = {}

    def get_state(key):
        return ctx.state.get(key)

    def set_state(key, value):
        ctx.state[key] = value

    ctx.get_state = MagicMock(side_effect=get_state)
    ctx.set_state = MagicMock(side_effect=set_state)
    ctx.info = AsyncMock()
    ctx.report_progress = AsyncMock()

    return ctx


async def test_with_json_file(json_path: str):
    """Test Phase 1 with a JSON file containing transactions.

    Args:
        json_path: Path to JSON file with transactions

    Expected JSON format:
    [
        {
            "order_id": "O1",
            "customer_id": "C1",
            "event_ts": "2023-01-15T10:00:00+00:00",  # ISO format or datetime object
            "unit_price": 100.0,
            "quantity": 2
        },
        ...
    ]
    """
    print(f"\n{'=' * 60}")
    print(f"Testing Phase 1 with: {json_path}")
    print(f"{'=' * 60}\n")

    # Load transactions from JSON
    with open(json_path, "r") as f:
        transactions = json.load(f)

    # Convert ISO strings to datetime objects if needed
    for txn in transactions:
        if isinstance(txn.get("event_ts"), str):
            txn["event_ts"] = datetime.fromisoformat(txn["event_ts"])
        elif isinstance(txn.get("order_ts"), str):
            txn["order_ts"] = datetime.fromisoformat(txn["order_ts"])

    print(f"Loaded {len(transactions)} transactions")

    # Create mock context
    ctx = create_mock_context()

    # Step 1: Build Data Mart
    print("\n--- Step 1: Building Data Mart ---")
    data_mart_request = BuildDataMartRequest(
        transaction_data_path=json_path, period_granularities=["quarter", "year"]
    )

    data_mart_response = await build_customer_data_mart(
        data_mart_request, ctx, transactions=transactions
    )

    print(f"✓ Data Mart Built:")
    print(f"  - Orders: {data_mart_response.order_count}")
    print(f"  - Customers: {data_mart_response.customer_count}")
    print(f"  - Periods: {data_mart_response.period_count}")
    print(f"  - Granularities: {', '.join(data_mart_response.granularities)}")
    print(
        f"  - Date Range: {data_mart_response.date_range[0]} to {data_mart_response.date_range[1]}"
    )

    # Step 2: Calculate RFM Metrics
    print("\n--- Step 2: Calculating RFM Metrics ---")

    # Determine observation end date (use latest transaction + 1 day)
    mart = ctx.state["data_mart"]
    latest_date = max(order.order_ts for order in mart.orders)
    observation_end = latest_date.replace(hour=23, minute=59, second=59)

    rfm_request = CalculateRFMRequest(
        observation_end=observation_end,
        enable_parallel=False,  # Set to True for large datasets
        calculate_scores=True,
    )

    rfm_response = await calculate_rfm_metrics(rfm_request, ctx)

    print(f"✓ RFM Metrics Calculated:")
    print(f"  - Customers: {rfm_response.metrics_count}")
    print(f"  - Scores: {rfm_response.score_count}")
    print(
        f"  - Date Range: {rfm_response.date_range[0]} to {rfm_response.date_range[1]}"
    )

    # Show sample RFM metrics
    rfm_metrics = ctx.state["rfm_metrics"]
    if rfm_metrics:
        print(f"\n  Sample RFM Metrics (first 5):")
        for i, metric in enumerate(rfm_metrics[:5]):
            print(
                f"    {metric.customer_id}: R={metric.recency_days}d, F={metric.frequency}, M=${metric.monetary}"
            )

    # Step 3: Create Cohorts
    print("\n--- Step 3: Creating Customer Cohorts ---")

    cohort_request = CreateCohortsRequest(
        cohort_type="quarterly"  # Options: "monthly", "quarterly", "yearly"
    )

    cohort_response = await create_customer_cohorts(cohort_request, ctx)

    print(f"✓ Cohorts Created:")
    print(f"  - Cohort Count: {cohort_response.cohort_count}")
    print(f"  - Customers: {cohort_response.customer_count}")
    print(f"  - Type: {cohort_response.cohort_type}")
    print(
        f"  - Date Range: {cohort_response.date_range[0]} to {cohort_response.date_range[1]}"
    )

    print(f"\n  Cohort Distribution:")
    for cohort_id, count in sorted(cohort_response.assignment_summary.items()):
        print(f"    {cohort_id}: {count} customers")

    print(f"\n{'=' * 60}")
    print("✓ Phase 1 Pipeline Complete!")
    print(f"{'=' * 60}\n")

    return ctx


async def test_with_python_data(transactions: list[dict]):
    """Test Phase 1 with in-memory transaction data.

    Args:
        transactions: List of transaction dictionaries with datetime objects
    """
    print(f"\n{'=' * 60}")
    print(f"Testing Phase 1 with in-memory data")
    print(f"{'=' * 60}\n")

    print(f"Using {len(transactions)} transactions")

    # Create mock context
    ctx = create_mock_context()

    # Build Data Mart
    print("\n--- Building Data Mart ---")
    data_mart_request = BuildDataMartRequest(
        transaction_data_path="memory", period_granularities=["quarter"]
    )

    data_mart_response = await build_customer_data_mart(
        data_mart_request, ctx, transactions=transactions
    )

    print(
        f"✓ Data Mart: {data_mart_response.customer_count} customers, "
        f"{data_mart_response.order_count} orders"
    )

    # Calculate RFM
    print("\n--- Calculating RFM ---")
    mart = ctx.state["data_mart"]
    latest_date = max(order.order_ts for order in mart.orders)

    rfm_request = CalculateRFMRequest(
        observation_end=latest_date, enable_parallel=False, calculate_scores=True
    )

    rfm_response = await calculate_rfm_metrics(rfm_request, ctx)
    print(f"✓ RFM: {rfm_response.metrics_count} customers analyzed")

    # Create Cohorts
    print("\n--- Creating Cohorts ---")
    cohort_request = CreateCohortsRequest(cohort_type="monthly")
    cohort_response = await create_customer_cohorts(cohort_request, ctx)
    print(f"✓ Cohorts: {cohort_response.cohort_count} cohorts created")

    print(f"\n{'=' * 60}")
    print("✓ Complete!")
    print(f"{'=' * 60}\n")

    return ctx


async def main():
    """Main test function."""
    import sys

    if len(sys.argv) > 1:
        # Test with provided JSON file
        json_path = sys.argv[1]
        if not Path(json_path).exists():
            print(f"Error: File not found: {json_path}")
            print("\nUsage:")
            print(f"  python {sys.argv[0]} <path-to-transactions.json>")
            print(f"  python {sys.argv[0]}  # Uses example data")
            return

        await test_with_json_file(json_path)

    else:
        # Test with example data
        print("No JSON file provided. Using example transactions...")

        example_transactions = [
            {
                "order_id": "O1",
                "customer_id": "C1",
                "event_ts": datetime(2023, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
                "unit_price": 100.0,
                "quantity": 2,
            },
            {
                "order_id": "O2",
                "customer_id": "C1",
                "event_ts": datetime(2023, 4, 10, 9, 15, 0, tzinfo=timezone.utc),
                "unit_price": 120.0,
                "quantity": 3,
            },
            {
                "order_id": "O3",
                "customer_id": "C2",
                "event_ts": datetime(2023, 1, 25, 16, 45, 0, tzinfo=timezone.utc),
                "unit_price": 50.0,
                "quantity": 1,
            },
            {
                "order_id": "O4",
                "customer_id": "C3",
                "event_ts": datetime(2023, 2, 5, 11, 20, 0, tzinfo=timezone.utc),
                "unit_price": 80.0,
                "quantity": 2,
            },
            {
                "order_id": "O5",
                "customer_id": "C3",
                "event_ts": datetime(2023, 5, 12, 13, 0, 0, tzinfo=timezone.utc),
                "unit_price": 90.0,
                "quantity": 1,
            },
        ]

        await test_with_python_data(example_transactions)


if __name__ == "__main__":
    asyncio.run(main())
