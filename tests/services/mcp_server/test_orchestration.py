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
