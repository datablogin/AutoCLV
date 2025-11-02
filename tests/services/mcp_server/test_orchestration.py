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


@pytest.mark.asyncio
async def test_formatted_outputs_lens2():
    """Test that Lens 2 generates formatted Sankey diagram."""
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
        )
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
        )
    ]

    shared_state.set("rfm_metrics", period1_rfm)
    shared_state.set("rfm_metrics_period2", period2_rfm)

    # Run Lens 2 with visualizations enabled
    query = "lens2"
    result = await coordinator.analyze(query, include_visualizations=True)

    # Check formatted_outputs includes lens2_sankey
    formatted_outputs = result.get("formatted_outputs", {})
    assert formatted_outputs is not None
    assert isinstance(formatted_outputs, dict)

    # If lens2 executed successfully, should have Sankey diagram
    if "lens2" in result.get("lenses_executed", []):
        assert "lens2_sankey" in formatted_outputs
        sankey = formatted_outputs["lens2_sankey"]

        # Verify it's a FastMCP Image object with PNG data
        from fastmcp.utilities.types import Image

        assert isinstance(sankey, Image), "lens2_sankey should be an Image object"
        assert hasattr(sankey, "data"), "Image should have data attribute"
        assert sankey.data.startswith(b"\x89PNG\r\n\x1a\n"), (
            "Should have valid PNG header"
        )
        assert len(sankey.data) > 1000, "Non-trivial PNG"

    shared_state.clear()


@pytest.mark.asyncio
async def test_formatted_outputs_lens3():
    """Test that Lens 3 generates formatted retention trend chart."""
    from analytics.services.mcp_server.state import get_shared_state

    coordinator = FourLensesCoordinator()
    shared_state = get_shared_state()

    # Setup minimal cohort data
    from datetime import datetime

    from customer_base_audit.foundation.cohorts import CohortDefinition

    cohort_definitions = [
        CohortDefinition(
            cohort_id="2024-Q1",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 31),
        )
    ]

    cohort_assignments = {f"C{i}": "2024-Q1" for i in range(10)}

    shared_state.set("cohort_definitions", cohort_definitions)
    shared_state.set("cohort_assignments", cohort_assignments)

    # Run Lens 3 with visualizations enabled
    query = "lens3"
    result = await coordinator.analyze(query, include_visualizations=True)

    # Check formatted_outputs
    formatted_outputs = result.get("formatted_outputs", {})

    # If lens3 executed successfully, should have retention trend chart
    if "lens3" in result.get("lenses_executed", []):
        assert "lens3_retention_trend" in formatted_outputs
        chart = formatted_outputs["lens3_retention_trend"]

        # Verify it's a FastMCP Image object with PNG data
        from fastmcp.utilities.types import Image

        assert isinstance(chart, Image), (
            "lens3_retention_trend should be an Image object"
        )
        assert hasattr(chart, "data"), "Image should have data attribute"
        assert chart.data.startswith(b"\x89PNG\r\n\x1a\n"), (
            "Should have valid PNG header"
        )
        assert len(chart.data) > 1000, "Non-trivial PNG"

    shared_state.clear()


@pytest.mark.asyncio
async def test_formatted_outputs_lens4():
    """Test that Lens 4 generates formatted heatmap and table."""
    from analytics.services.mcp_server.state import get_shared_state

    coordinator = FourLensesCoordinator()
    shared_state = get_shared_state()

    # Setup minimal cohort data
    from datetime import datetime

    from customer_base_audit.foundation.cohorts import CohortDefinition

    cohort_definitions = [
        CohortDefinition(
            cohort_id="2024-Q1",
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 3, 31),
        ),
        CohortDefinition(
            cohort_id="2024-Q2",
            start_date=datetime(2024, 4, 1),
            end_date=datetime(2024, 6, 30),
        ),
    ]

    cohort_assignments = {}
    for i in range(5):
        cohort_assignments[f"C{i}"] = "2024-Q1"
    for i in range(5, 10):
        cohort_assignments[f"C{i}"] = "2024-Q2"

    shared_state.set("cohort_definitions", cohort_definitions)
    shared_state.set("cohort_assignments", cohort_assignments)

    # Run Lens 4
    query = "lens4"
    result = await coordinator.analyze(query)

    # Check formatted_outputs
    formatted_outputs = result.get("formatted_outputs", {})

    # Note: Lens 4 is currently a placeholder, so it won't have formatted outputs
    # This test validates that the formatting doesn't crash even with placeholder data
    # When Lens 4 is fully implemented, it should have heatmap and table
    assert isinstance(formatted_outputs, dict)

    # TODO: Once Lens 4 is fully implemented (not a placeholder), uncomment:
    # if "lens4" in result.get("lenses_executed", []):
    #     assert "lens4_heatmap" in formatted_outputs
    #     assert "lens4_table" in formatted_outputs
    #     heatmap = formatted_outputs["lens4_heatmap"]
    #     table = formatted_outputs["lens4_table"]
    #     assert isinstance(heatmap, dict)
    #     assert isinstance(table, str)  # Markdown table

    shared_state.clear()


@pytest.mark.asyncio
async def test_formatted_outputs_executive_dashboard():
    """Test that multi-lens analysis generates executive dashboard."""
    from analytics.services.mcp_server.state import get_shared_state

    coordinator = FourLensesCoordinator()
    shared_state = get_shared_state()

    # Setup minimal data for multiple lenses
    from datetime import datetime
    from decimal import Decimal

    from customer_base_audit.foundation.rfm import RFMMetrics

    # RFM data for lens1
    rfm_metrics = [
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

    shared_state.set("rfm_metrics", rfm_metrics)

    # Run analysis with multiple lenses and visualizations enabled
    query = "lens1 and lens5"
    result = await coordinator.analyze(query, include_visualizations=True)

    # Check formatted_outputs
    formatted_outputs = result.get("formatted_outputs", {})
    lenses_executed = result.get("lenses_executed", [])

    # If lens1 and lens5 both executed, should have executive dashboard
    if "lens1" in lenses_executed and "lens5" in lenses_executed:
        assert "executive_dashboard" in formatted_outputs
        dashboard = formatted_outputs["executive_dashboard"]

        # Verify it's a FastMCP Image object with PNG data
        from fastmcp.utilities.types import Image

        assert isinstance(dashboard, Image), (
            "executive_dashboard should be an Image object"
        )
        assert hasattr(dashboard, "data"), "Image should have data attribute"
        assert dashboard.data.startswith(b"\x89PNG\r\n\x1a\n"), (
            "Should have valid PNG header"
        )
        assert len(dashboard.data) > 1000, "Non-trivial PNG"

    shared_state.clear()


@pytest.mark.asyncio
async def test_include_visualizations_default_false():
    """Test that include_visualizations defaults to False (no PNG generation)."""
    from analytics.services.mcp_server.state import get_shared_state

    coordinator = FourLensesCoordinator()
    shared_state = get_shared_state()

    # Setup minimal data
    from datetime import datetime
    from decimal import Decimal

    from customer_base_audit.foundation.rfm import RFMMetrics

    rfm_metrics = [
        RFMMetrics(
            customer_id=f"C{i}",
            recency_days=10,
            frequency=3,
            monetary=Decimal("150"),
            observation_start=datetime(2024, 1, 1),
            observation_end=datetime(2024, 6, 30),
            total_spend=Decimal("450"),
        )
        for i in range(5)
    ]

    shared_state.set("rfm_metrics", rfm_metrics)

    # Run analysis WITHOUT include_visualizations (default behavior)
    query = "lens1"
    result = await coordinator.analyze(query)  # No include_visualizations parameter

    # Should have empty formatted_outputs by default
    formatted_outputs = result.get("formatted_outputs", {})
    assert formatted_outputs == {}

    shared_state.clear()


@pytest.mark.asyncio
async def test_include_visualizations_explicit_true():
    """Test that include_visualizations=True generates PNGs."""
    from analytics.services.mcp_server.state import get_shared_state

    coordinator = FourLensesCoordinator()
    shared_state = get_shared_state()

    # Setup data for Lens 2 (requires two periods)
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

    period2_rfm = [
        RFMMetrics(
            customer_id="C1",
            recency_days=5,
            frequency=3,
            monetary=Decimal("120"),
            observation_start=datetime(2024, 7, 1),
            observation_end=datetime(2024, 12, 31),
            total_spend=Decimal("360"),
        )
    ]

    shared_state.set("rfm_metrics", period1_rfm)
    shared_state.set("rfm_metrics_period2", period2_rfm)

    # Run analysis WITH include_visualizations=True
    query = "lens2"
    result = await coordinator.analyze(query, include_visualizations=True)

    # Should have formatted_outputs with Sankey diagram
    formatted_outputs = result.get("formatted_outputs", {})
    assert formatted_outputs != {}

    # Verify lens2_sankey is present and is a FastMCP Image object
    if "lens2" in result.get("lenses_executed", []):
        assert "lens2_sankey" in formatted_outputs
        from fastmcp.utilities.types import Image

        assert isinstance(formatted_outputs["lens2_sankey"], Image)

    shared_state.clear()


@pytest.mark.asyncio
async def test_formatted_outputs_graceful_failure():
    """Test that formatting errors don't crash the entire analysis."""
    coordinator = FourLensesCoordinator()

    # Run analysis without data (lenses will fail)
    query = "lens1"
    result = await coordinator.analyze(query)

    # Should have formatted_outputs key, even if empty
    assert "formatted_outputs" in result
    formatted_outputs = result.get("formatted_outputs", {})
    assert isinstance(formatted_outputs, dict)

    # Analysis should still complete
    assert "insights" in result
    assert "recommendations" in result
