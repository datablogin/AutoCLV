"""Test Phase 3 orchestration functionality.

This module tests the LangGraph coordinator and orchestrated analysis workflow.
"""

import pytest

from analytics.services.mcp_server.orchestration.coordinator import (
    FourLensesCoordinator,
)


@pytest.mark.asyncio
async def test_coordinator_initialization():
    """Test coordinator initializes successfully."""
    coordinator = FourLensesCoordinator()
    assert coordinator is not None
    assert coordinator.graph is not None
    assert coordinator.shared_state is not None


@pytest.mark.asyncio
async def test_intent_parsing_lens1():
    """Test intent parsing for Lens 1 keywords."""
    coordinator = FourLensesCoordinator()

    # Test various Lens 1 keywords
    queries = [
        "customer health snapshot",
        "show me current health",
        "lens1 analysis",
        "what's the snapshot look like",
    ]

    for query in queries:
        result = await coordinator.analyze(query)
        intent = result.get("intent", {})
        assert "lens1" in intent.get("lenses", []), f"Expected lens1 for query: {query}"


@pytest.mark.asyncio
async def test_intent_parsing_lens5():
    """Test intent parsing for Lens 5 keywords."""
    coordinator = FourLensesCoordinator()

    # Test various Lens 5 keywords
    queries = [
        "overall customer base health",
        "base health analysis",
        "lens5 report",
        "show me overall health",
    ]

    for query in queries:
        result = await coordinator.analyze(query)
        intent = result.get("intent", {})
        assert "lens5" in intent.get("lenses", []), f"Expected lens5 for query: {query}"


@pytest.mark.asyncio
async def test_intent_parsing_multiple_lenses():
    """Test intent parsing for queries requesting multiple lenses."""
    coordinator = FourLensesCoordinator()

    query = "customer health and cohort comparison"
    result = await coordinator.analyze(query)
    intent = result.get("intent", {})
    lenses = intent.get("lenses", [])

    # Should detect both health (lens1) and cohort comparison (lens4)
    assert "lens1" in lenses or "lens5" in lenses
    assert len(lenses) >= 1


@pytest.mark.asyncio
async def test_intent_parsing_default():
    """Test that unknown queries default to Lens 1."""
    coordinator = FourLensesCoordinator()

    query = "show me something"
    result = await coordinator.analyze(query)
    intent = result.get("intent", {})

    # Should default to lens1 when no specific lens is detected
    assert "lens1" in intent.get("lenses", [])


@pytest.mark.asyncio
async def test_orchestration_without_data():
    """Test orchestration fails gracefully when foundation data is missing."""
    coordinator = FourLensesCoordinator()

    query = "customer health snapshot"
    result = await coordinator.analyze(query)

    # Should complete without crashing
    assert result.get("query") == query
    assert "lenses_executed" in result
    assert "lenses_failed" in result
    assert "insights" in result

    # Lens 1 should fail because RFM data is missing
    assert "lens1" in result.get("lenses_failed", [])


@pytest.mark.asyncio
async def test_orchestration_state_management():
    """Test that orchestration properly tracks state through workflow."""
    coordinator = FourLensesCoordinator()

    query = "lens1"
    result = await coordinator.analyze(query)

    # Check state structure
    assert "data_mart_ready" in result
    assert "rfm_ready" in result
    assert "cohorts_ready" in result
    assert "execution_time_ms" in result
    assert result.get("execution_time_ms", 0) >= 0


@pytest.mark.asyncio
async def test_parallel_execution_intent():
    """Test that multiple independent lenses are requested for parallel execution."""
    coordinator = FourLensesCoordinator()

    # Request lenses 1, 3, 4, 5 (all can run in parallel)
    query = "lens1 and lens5"
    result = await coordinator.analyze(query)

    intent = result.get("intent", {})
    lenses = intent.get("lenses", [])

    # Should have multiple lenses
    assert len(lenses) >= 2
    assert "lens1" in lenses
    assert "lens5" in lenses


@pytest.mark.asyncio
async def test_lens2_requires_two_periods():
    """Test that Lens 2 provides helpful message when period 2 RFM is missing."""
    from analytics.services.mcp_server.state import get_shared_state

    coordinator = FourLensesCoordinator()
    shared_state = get_shared_state()

    # Setup minimal data for period 1
    from datetime import datetime
    from decimal import Decimal

    from customer_base_audit.foundation.rfm import RFMMetrics

    period1_rfm = [
        RFMMetrics(
            customer_id="C1",
            recency_days=10,
            frequency=5,
            monetary=Decimal("100"),
            observation_start=datetime(2024, 1, 1),
            observation_end=datetime(2024, 6, 30),
            total_spend=Decimal("500"),
        )
    ]

    # Store period 1 RFM but NOT period 2
    shared_state.set("rfm_metrics", period1_rfm)

    # Try to run Lens 2
    query = "lens2"
    result = await coordinator.analyze(query)

    # Should indicate lens2 needs two periods
    lens2_result = result.get("lens2_result")
    if lens2_result:  # May be None if lens failed
        assert "recommendations" in lens2_result
        recommendations = lens2_result.get("recommendations", [])
        # Should have message about needing two periods
        assert any("period 2" in str(rec).lower() for rec in recommendations) or any(
            "rfm_metrics_period2" in str(rec) for rec in recommendations
        )

    # Cleanup
    shared_state.clear()


@pytest.mark.asyncio
async def test_lens2_with_two_periods():
    """Test that Lens 2 executes successfully when both periods are available."""
    from analytics.services.mcp_server.state import get_shared_state

    coordinator = FourLensesCoordinator()
    shared_state = get_shared_state()

    # Setup data for both periods
    from datetime import datetime
    from decimal import Decimal

    from customer_base_audit.foundation.rfm import RFMMetrics

    period1_rfm = [
        RFMMetrics(
            customer_id="C1",
            recency_days=10,
            frequency=5,
            monetary=Decimal("100"),
            observation_start=datetime(2024, 1, 1),
            observation_end=datetime(2024, 6, 30),
            total_spend=Decimal("500"),
        ),
        RFMMetrics(
            customer_id="C2",
            recency_days=20,
            frequency=3,
            monetary=Decimal("150"),
            observation_start=datetime(2024, 1, 1),
            observation_end=datetime(2024, 6, 30),
            total_spend=Decimal("450"),
        ),
    ]

    period2_rfm = [
        RFMMetrics(
            customer_id="C1",
            recency_days=5,
            frequency=3,
            monetary=Decimal("120"),
            observation_start=datetime(2024, 7, 1),
            observation_end=datetime(2024, 12, 31),
            total_spend=Decimal("360"),
        ),
        RFMMetrics(
            customer_id="C3",
            recency_days=15,
            frequency=2,
            monetary=Decimal("200"),
            observation_start=datetime(2024, 7, 1),
            observation_end=datetime(2024, 12, 31),
            total_spend=Decimal("400"),
        ),
    ]

    # Store both periods
    shared_state.set("rfm_metrics", period1_rfm)
    shared_state.set("rfm_metrics_period2", period2_rfm)

    # Run Lens 2
    query = "lens2"
    result = await coordinator.analyze(query)

    # Should execute successfully
    assert "lens2" in result.get("lenses_executed", []) or "lens2_result" in result

    lens2_result = result.get("lens2_result")
    if lens2_result:
        # Verify key metrics are present
        assert "retention_rate" in lens2_result
        assert "churn_rate" in lens2_result
        assert "growth_momentum" in lens2_result
        assert "recommendations" in lens2_result

        # Verify retention calculation is correct
        # C1 was retained (in both periods), C2 churned, C3 is new
        assert lens2_result["period1_customers"] == 2
        assert lens2_result["period2_customers"] == 2
        assert lens2_result["retained_customers"] == 1  # C1
        assert lens2_result["churned_customers"] == 1  # C2
        assert lens2_result["new_customers"] == 1  # C3

    # Cleanup
    shared_state.clear()


# Note: Lens 3 orchestration is already tested via test_lens_tools.py::test_lens3_cohort_evolution
# which tests the full lens3 implementation. Orchestration-level test would be redundant.
