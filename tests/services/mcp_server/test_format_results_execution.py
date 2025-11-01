"""Execution test for _format_results() PNG generation.

Tests that PNG images are actually generated correctly with valid headers.
"""
from datetime import datetime
from decimal import Decimal

import pytest
from fastmcp.utilities.types import Image

from analytics.services.mcp_server.orchestration.coordinator import (
    FourLensesCoordinator,
)
from customer_base_audit.analyses.lens1 import Lens1Metrics
from customer_base_audit.analyses.lens5 import (
    CohortRepeatBehavior,
    CohortRevenuePeriod,
    CustomerBaseHealthScore,
    Lens5Metrics,
)


def create_mock_lens1_metrics() -> Lens1Metrics:
    """Create mock Lens1Metrics for testing."""
    return Lens1Metrics(
        total_customers=1000,
        one_time_buyers=50,
        one_time_buyer_pct=5.0,
        total_revenue=100000.0,
        top_10pct_revenue_contribution=30.0,
        top_20pct_revenue_contribution=50.0,
        avg_orders_per_customer=3.5,
        median_customer_value=85.0,
        rfm_distribution={
            "Champions": 200,
            "Loyal": 300,
            "Potential": 250,
            "At Risk": 150,
            "Lost": 100,
        },
    )


def create_mock_lens5_metrics() -> Lens5Metrics:
    """Create mock Lens5Metrics for testing."""
    return Lens5Metrics(
        cohort_revenue_contributions=[
            CohortRevenuePeriod(
                cohort_id="2024-01",
                period_start=datetime(2024, 1, 1),
                total_revenue=Decimal("5000.00"),
                pct_of_period_revenue=Decimal("50.0"),
                active_customers=100,
                avg_revenue_per_customer=Decimal("50.00"),
            )
        ],
        cohort_repeat_behavior=[
            CohortRepeatBehavior(
                cohort_id="2024-01",
                cohort_size=100,
                one_time_buyers=20,
                repeat_buyers=80,
                repeat_rate=Decimal("80.0"),
                avg_orders_per_repeat_buyer=Decimal("2.5"),
            )
        ],
        health_score=CustomerBaseHealthScore(
            health_score=Decimal("85.0"),
            health_grade="B",
            total_customers=1000,
            total_active_customers=900,
            overall_retention_rate=Decimal("90.0"),
            cohort_quality_trend="improving",
            revenue_predictability_pct=Decimal("95.0"),
            acquisition_dependence_pct=Decimal("5.0"),
        ),
        analysis_start_date=datetime(2024, 1, 1),
        analysis_end_date=datetime(2024, 12, 31),
    )


@pytest.mark.asyncio
async def test_format_results_generates_valid_pngs():
    """Test that _format_results() generates valid PNG images."""
    coordinator = FourLensesCoordinator()

    # Create mock state with lens results
    state = {
        "query": "test query",
        "intent": {"lenses": ["lens1", "lens5"]},
        "include_visualizations": True,
        "lenses_executed": ["lens1", "lens5"],
        "lenses_failed": [],
        # Lens 1 data
        "lens1_result": {
            "total_customers": 1000,
            "one_time_buyers": 50,
            "one_time_buyer_pct": 5.0,
            "total_revenue": 100000.0,
            "top_10pct_revenue_contribution": 30.0,
            "top_20pct_revenue_contribution": 50.0,
            "avg_orders_per_customer": 3.5,
            "median_customer_value": 85.0,
            "rfm_distribution": {
                "Champions": 200,
                "Loyal": 300,
                "Potential": 250,
                "At Risk": 150,
                "Lost": 100,
            },
        },
        # Lens 5 data (store full metrics object)
        "lens5_metrics": create_mock_lens5_metrics(),
        "lens5_result": {
            "health_score": 85.0,
            "health_grade": "B",
            "total_customers": 1000,
        },
    }

    # Execute _format_results()
    result_state = await coordinator._format_results(state)

    # Verify formatted_outputs exists
    assert "formatted_outputs" in result_state
    formatted_outputs = result_state["formatted_outputs"]

    # Verify PNG images are generated
    expected_images = ["lens1_revenue_pie", "lens5_health_gauge"]
    for image_key in expected_images:
        assert image_key in formatted_outputs, f"Missing {image_key}"
        img = formatted_outputs[image_key]

        # Verify it's an Image object
        assert isinstance(img, Image), f"{image_key} is not an Image object"

        # Verify PNG header (Image.data should contain PNG bytes)
        assert hasattr(img, "data"), f"{image_key} missing data attribute"
        assert img.data.startswith(
            b"\x89PNG\r\n\x1a\n"
        ), f"{image_key} does not have valid PNG header"

        # Verify non-empty
        assert len(img.data) > 0, f"{image_key} has empty data"

    # Verify tables are generated
    expected_tables = ["lens1_table", "lens5_table"]
    for table_key in expected_tables:
        assert table_key in formatted_outputs, f"Missing {table_key}"
        assert isinstance(
            formatted_outputs[table_key], str
        ), f"{table_key} is not a string"
        assert len(formatted_outputs[table_key]) > 0, f"{table_key} is empty"

    # Verify summaries are generated
    assert "health_summary" in formatted_outputs
    assert isinstance(formatted_outputs["health_summary"], str)
    assert len(formatted_outputs["health_summary"]) > 0


@pytest.mark.asyncio
async def test_format_results_skips_when_disabled():
    """Test that _format_results() skips generation when include_visualizations=False."""
    coordinator = FourLensesCoordinator()

    state = {
        "query": "test query",
        "include_visualizations": False,  # Disabled
        "lenses_executed": ["lens1", "lens5"],
        "lens1_result": {"total_customers": 1000},
        "lens5_metrics": create_mock_lens5_metrics(),
    }

    result_state = await coordinator._format_results(state)

    # Should return empty formatted_outputs
    assert result_state["formatted_outputs"] == {}


@pytest.mark.asyncio
async def test_format_results_handles_missing_kaleido():
    """Test that _format_results() handles missing kaleido gracefully."""
    # This test would require mocking the kaleido import, which is complex
    # For now, we verify that kaleido is installed
    try:
        import kaleido  # noqa: F401

        pytest.skip("kaleido is installed, cannot test missing dependency")
    except ImportError:
        # If kaleido is not installed, the function should return empty outputs
        coordinator = FourLensesCoordinator()

        state = {
            "query": "test query",
            "include_visualizations": True,
            "lenses_executed": ["lens1"],
            "lens1_result": {"total_customers": 1000},
        }

        result_state = await coordinator._format_results(state)
        assert result_state["formatted_outputs"] == {}
