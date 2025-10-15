"""Integration tests for Phase 1 Foundation MCP Tools

Tests the three foundation tools:
1. build_customer_data_mart
2. calculate_rfm_metrics
3. create_customer_cohorts
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
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


def create_sample_transactions():
    """Create sample transaction data for testing."""
    return [
        # Customer C1: High-value repeat buyer
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
            "event_ts": datetime(2023, 2, 20, 14, 30, 0, tzinfo=timezone.utc),
            "unit_price": 150.0,
            "quantity": 1,
        },
        {
            "order_id": "O3",
            "customer_id": "C1",
            "event_ts": datetime(2023, 4, 10, 9, 15, 0, tzinfo=timezone.utc),
            "unit_price": 120.0,
            "quantity": 3,
        },
        # Customer C2: One-time buyer
        {
            "order_id": "O4",
            "customer_id": "C2",
            "event_ts": datetime(2023, 1, 25, 16, 45, 0, tzinfo=timezone.utc),
            "unit_price": 50.0,
            "quantity": 1,
        },
        # Customer C3: Moderate repeat buyer
        {
            "order_id": "O5",
            "customer_id": "C3",
            "event_ts": datetime(2023, 2, 5, 11, 20, 0, tzinfo=timezone.utc),
            "unit_price": 80.0,
            "quantity": 2,
        },
        {
            "order_id": "O6",
            "customer_id": "C3",
            "event_ts": datetime(2023, 5, 12, 13, 0, 0, tzinfo=timezone.utc),
            "unit_price": 90.0,
            "quantity": 1,
        },
        # Customer C4: New customer
        {
            "order_id": "O7",
            "customer_id": "C4",
            "event_ts": datetime(2023, 4, 20, 15, 30, 0, tzinfo=timezone.utc),
            "unit_price": 200.0,
            "quantity": 1,
        },
        # Customer C5: Another new customer
        {
            "order_id": "O8",
            "customer_id": "C5",
            "event_ts": datetime(2023, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
            "unit_price": 75.0,
            "quantity": 2,
        },
    ]


def create_mock_context():
    """Create a mock FastMCP Context for testing."""
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


@pytest.mark.asyncio
async def test_data_mart_build_workflow():
    """Test complete data mart build workflow."""
    transactions = create_sample_transactions()

    # Create request
    request = BuildDataMartRequest(
        transaction_data_path="test.json",  # Dummy path for testing
        period_granularities=["quarter", "year"]
    )

    # Create mock context
    ctx = create_mock_context()

    # Execute tool with pre-loaded transactions
    response = await build_customer_data_mart(request, ctx, transactions=transactions)

    # Verify response
    assert response.order_count == 8  # 8 orders
    assert response.customer_count == 5  # 5 customers
    assert response.period_count > 0  # Should have periods
    assert "quarter" in response.granularities
    assert "year" in response.granularities

    # Verify data mart was stored in context
    assert ctx.state["data_mart"] is not None
    mart = ctx.state["data_mart"]
    assert len(mart.orders) == 8
    assert len(mart.periods) == 2  # QUARTER and YEAR


@pytest.mark.asyncio
async def test_rfm_calculation_workflow():
    """Test RFM calculation with data mart context."""
    transactions = create_sample_transactions()
    ctx = create_mock_context()

    # Build data mart
    data_mart_request = BuildDataMartRequest(
        transaction_data_path="test.json",
        period_granularities=["quarter"]
    )
    await build_customer_data_mart(data_mart_request, ctx, transactions=transactions)

    # Calculate RFM
    rfm_request = CalculateRFMRequest(
        observation_end=datetime(2023, 6, 30, tzinfo=timezone.utc),
        enable_parallel=False,  # Disable parallel for deterministic testing
        calculate_scores=True
    )

    response = await calculate_rfm_metrics(rfm_request, ctx)

    # Verify response
    assert response.metrics_count == 5  # 5 customers
    assert response.score_count == 5  # 5 scores
    assert response.parallel_enabled is False

    # Verify RFM metrics were stored in context
    assert ctx.state["rfm_metrics"] is not None
    assert ctx.state["rfm_scores"] is not None
    assert len(ctx.state["rfm_metrics"]) == 5
    assert len(ctx.state["rfm_scores"]) == 5


@pytest.mark.asyncio
async def test_cohort_creation_workflow():
    """Test cohort creation and assignment."""
    transactions = create_sample_transactions()
    ctx = create_mock_context()

    # Build data mart
    data_mart_request = BuildDataMartRequest(
        transaction_data_path="test.json",
        period_granularities=["quarter"]
    )
    await build_customer_data_mart(data_mart_request, ctx, transactions=transactions)

    # Create cohorts
    cohort_request = CreateCohortsRequest(
        cohort_type="quarterly"
    )

    response = await create_customer_cohorts(cohort_request, ctx)

    # Verify response
    assert response.cohort_count > 0  # Should have cohorts
    assert response.customer_count == 5  # 5 customers
    assert response.cohort_type == "quarterly"
    assert len(response.assignment_summary) > 0

    # Verify cohorts were stored in context
    assert ctx.state["cohort_definitions"] is not None
    assert ctx.state["cohort_assignments"] is not None

    # Verify all customers are assigned
    assignments = ctx.state["cohort_assignments"]
    assert len(assignments) == 5


@pytest.mark.asyncio
async def test_full_foundation_pipeline():
    """Test complete foundation workflow: data mart → RFM → cohorts."""
    transactions = create_sample_transactions()
    ctx = create_mock_context()

    # Step 1: Build data mart
    data_mart_request = BuildDataMartRequest(
        transaction_data_path="test.json",
        period_granularities=["quarter"]
    )
    data_mart_response = await build_customer_data_mart(data_mart_request, ctx, transactions=transactions)
    assert data_mart_response.customer_count == 5

    # Step 2: Calculate RFM
    rfm_request = CalculateRFMRequest(
        observation_end=datetime(2023, 6, 30, tzinfo=timezone.utc),
        enable_parallel=False,
        calculate_scores=True
    )
    rfm_response = await calculate_rfm_metrics(rfm_request, ctx)
    assert rfm_response.metrics_count == 5

    # Step 3: Create cohorts
    cohort_request = CreateCohortsRequest(
        cohort_type="monthly"
    )
    cohort_response = await create_customer_cohorts(cohort_request, ctx)
    assert cohort_response.customer_count == 5

    # Verify all data is in context
    assert ctx.state["data_mart"] is not None
    assert ctx.state["rfm_metrics"] is not None
    assert ctx.state["rfm_scores"] is not None
    assert ctx.state["cohort_definitions"] is not None
    assert ctx.state["cohort_assignments"] is not None


@pytest.mark.asyncio
async def test_rfm_without_data_mart_raises_error():
    """Test that RFM calculation fails without data mart."""
    ctx = create_mock_context()

    rfm_request = CalculateRFMRequest(
        observation_end=datetime(2023, 6, 30, tzinfo=timezone.utc)
    )

    with pytest.raises(ValueError, match="Data mart not found"):
        await calculate_rfm_metrics(rfm_request, ctx)


@pytest.mark.asyncio
async def test_cohorts_without_data_mart_raises_error():
    """Test that cohort creation fails without data mart."""
    ctx = create_mock_context()

    cohort_request = CreateCohortsRequest(
        cohort_type="quarterly"
    )

    with pytest.raises(ValueError, match="Data mart not found"):
        await create_customer_cohorts(cohort_request, ctx)


@pytest.mark.asyncio
async def test_empty_transactions_raises_error():
    """Test that empty transaction list raises appropriate error."""
    ctx = create_mock_context()

    data_mart_request = BuildDataMartRequest(
        transaction_data_path="test.json",
        period_granularities=["quarter"]
    )

    # Should succeed building empty data mart
    response = await build_customer_data_mart(data_mart_request, ctx, transactions=[])

    # But should fail when trying to calculate RFM on empty data
    rfm_request = CalculateRFMRequest(
        observation_end=datetime(2023, 6, 30, tzinfo=timezone.utc),
        enable_parallel=False,
        calculate_scores=True
    )

    with pytest.raises(ValueError, match="No RFM metrics calculated"):
        await calculate_rfm_metrics(rfm_request, ctx)


@pytest.mark.asyncio
async def test_single_customer_workflow():
    """Test that single customer works through full pipeline."""
    transactions = [
        {
            "order_id": "O1",
            "customer_id": "C1",
            "event_ts": datetime(2023, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
            "unit_price": 100.0,
            "quantity": 1,
        },
        {
            "order_id": "O2",
            "customer_id": "C1",
            "event_ts": datetime(2023, 2, 20, 14, 30, 0, tzinfo=timezone.utc),
            "unit_price": 150.0,
            "quantity": 1,
        },
    ]

    ctx = create_mock_context()

    # Build data mart
    data_mart_request = BuildDataMartRequest(
        transaction_data_path="test.json",
        period_granularities=["quarter"]
    )
    data_mart_response = await build_customer_data_mart(data_mart_request, ctx, transactions=transactions)
    assert data_mart_response.customer_count == 1
    assert data_mart_response.order_count == 2

    # Calculate RFM
    rfm_request = CalculateRFMRequest(
        observation_end=datetime(2023, 6, 30, tzinfo=timezone.utc),
        enable_parallel=False,
        calculate_scores=True
    )
    rfm_response = await calculate_rfm_metrics(rfm_request, ctx)
    assert rfm_response.metrics_count == 1
    assert rfm_response.score_count == 1

    # Create cohorts
    cohort_request = CreateCohortsRequest(cohort_type="quarterly")
    cohort_response = await create_customer_cohorts(cohort_request, ctx)
    assert cohort_response.customer_count == 1
    assert cohort_response.cohort_count == 1
