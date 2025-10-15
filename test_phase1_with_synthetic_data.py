#!/usr/bin/env python
"""Test Phase 1 Foundation Services with synthetic data generator.

This script uses the built-in synthetic data generator to test Phase 1 tools
with realistic customer scenarios.
"""

import asyncio
from datetime import datetime, timezone, date
from unittest.mock import AsyncMock, MagicMock

from customer_base_audit.synthetic import (
    generate_customers,
    generate_transactions,
    BASELINE_SCENARIO,
    HIGH_CHURN_SCENARIO,
    STABLE_BUSINESS_SCENARIO,
)

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


async def test_scenario(scenario_name: str, scenario_config: dict, num_customers: int = 100):
    """Test Phase 1 with a specific synthetic data scenario.

    Args:
        scenario_name: Name of the scenario (for display)
        scenario_config: Scenario configuration dict
        num_customers: Number of customers to generate
    """
    print(f"\n{'='*80}")
    print(f"TESTING: {scenario_name}")
    print(f"{'='*80}\n")

    # Generate synthetic customers and transactions
    print(f"Generating synthetic data...")
    print(f"  - Customers: {num_customers}")
    print(f"  - Scenario: {scenario_name}")

    start_date = date(2023, 1, 1)
    end_date = date(2023, 12, 31)

    customers = generate_customers(
        n=num_customers,
        start=start_date,
        end=end_date,
    )

    transaction_objects = generate_transactions(
        customers,
        start=start_date,
        end=end_date,
        scenario=scenario_config
    )

    # Convert Transaction objects to dicts for data mart builder
    transactions = [
        {
            "order_id": txn.order_id,
            "customer_id": txn.customer_id,
            "event_ts": txn.event_ts,
            "unit_price": txn.unit_price,
            "quantity": txn.quantity,
        }
        for txn in transaction_objects
    ]
    print(f"  - Generated {len(transactions)} transactions")

    # Create mock context
    ctx = create_mock_context()

    # Step 1: Build Data Mart
    print(f"\n--- Step 1: Building Data Mart ---")
    data_mart_request = BuildDataMartRequest(
        transaction_data_path="synthetic",
        period_granularities=["quarter", "year"]
    )

    data_mart_response = await build_customer_data_mart(
        data_mart_request,
        ctx,
        transactions=transactions
    )

    print(f"✓ Data Mart Built:")
    print(f"  - Orders: {data_mart_response.order_count}")
    print(f"  - Customers: {data_mart_response.customer_count}")
    print(f"  - Periods: {data_mart_response.period_count}")
    print(f"  - Date Range: {data_mart_response.date_range[0][:10]} to {data_mart_response.date_range[1][:10]}")

    # Step 2: Calculate RFM Metrics
    print(f"\n--- Step 2: Calculating RFM Metrics ---")

    rfm_request = CalculateRFMRequest(
        observation_end=datetime(2023, 12, 31, 23, 59, 59),  # Naive datetime to match synthetic data
        enable_parallel=True,  # Enable for realistic dataset sizes
        calculate_scores=True
    )

    rfm_response = await calculate_rfm_metrics(rfm_request, ctx)

    print(f"✓ RFM Metrics Calculated:")
    print(f"  - Customers Analyzed: {rfm_response.metrics_count}")
    print(f"  - RFM Scores Generated: {rfm_response.score_count}")
    print(f"  - Parallel Processing: {rfm_response.parallel_enabled}")

    # Analyze RFM distribution
    rfm_metrics = ctx.state["rfm_metrics"]
    rfm_scores = ctx.state["rfm_scores"]

    if rfm_metrics:
        avg_recency = sum(m.recency_days for m in rfm_metrics) / len(rfm_metrics)
        avg_frequency = sum(m.frequency for m in rfm_metrics) / len(rfm_metrics)
        avg_monetary = sum(float(m.monetary) for m in rfm_metrics) / len(rfm_metrics)

        print(f"\n  RFM Averages:")
        print(f"    - Recency: {avg_recency:.1f} days")
        print(f"    - Frequency: {avg_frequency:.2f} orders")
        print(f"    - Monetary: ${avg_monetary:.2f} per order")

    if rfm_scores:
        # Count RFM score distribution
        score_dist = {}
        for score in rfm_scores:
            key = score.rfm_score
            score_dist[key] = score_dist.get(key, 0) + 1

        print(f"\n  Top 5 RFM Segments:")
        for score, count in sorted(score_dist.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    {score}: {count} customers ({count/len(rfm_scores)*100:.1f}%)")

    # Step 3: Create Cohorts
    print(f"\n--- Step 3: Creating Customer Cohorts ---")

    cohort_request = CreateCohortsRequest(
        cohort_type="quarterly"
    )

    cohort_response = await create_customer_cohorts(cohort_request, ctx)

    print(f"✓ Cohorts Created:")
    print(f"  - Cohort Count: {cohort_response.cohort_count}")
    print(f"  - Customers Assigned: {cohort_response.customer_count}")

    print(f"\n  Cohort Distribution:")
    for cohort_id, count in sorted(cohort_response.assignment_summary.items()):
        pct = (count / cohort_response.customer_count) * 100
        print(f"    {cohort_id}: {count} customers ({pct:.1f}%)")

    print(f"\n{'='*80}")
    print(f"✓ {scenario_name} - Phase 1 Pipeline Complete!")
    print(f"{'='*80}\n")

    return ctx


async def main():
    """Run Phase 1 tests with different synthetic scenarios."""
    print("\n" + "="*80)
    print("Phase 1 Foundation Services - Synthetic Data Testing")
    print("="*80)

    # Test 1: Baseline Scenario
    await test_scenario(
        "Baseline Business Scenario",
        BASELINE_SCENARIO,
        num_customers=200
    )

    # Test 2: High Churn Scenario
    await test_scenario(
        "High Churn Scenario",
        HIGH_CHURN_SCENARIO,
        num_customers=200
    )

    # Test 3: Stable Business Scenario
    await test_scenario(
        "Stable Business Scenario",
        STABLE_BUSINESS_SCENARIO,
        num_customers=200
    )

    print("\n" + "="*80)
    print("✓ ALL SCENARIOS TESTED SUCCESSFULLY")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
