"""Test Phase 4 observability and resilience features.

This module tests:
1. Metrics collector functionality
2. Retry logic with tenacity
3. OpenTelemetry tracing integration
4. Coordinator metrics integration

Note: Health check and execution metrics MCP tools are tested via end-to-end tests.
This file focuses on the underlying components.
"""

import pytest

from analytics.services.mcp_server.orchestration.coordinator import (
    FourLensesCoordinator,
    get_metrics_collector_instance,
)
from analytics.services.mcp_server.tools.execution_metrics import MetricsCollector


@pytest.mark.asyncio
async def test_metrics_collector_initial_state():
    """Test metrics collector starts with zero counts."""
    collector = MetricsCollector()

    metrics = collector.get_metrics()

    assert metrics["total_analyses"] == 0
    assert metrics["overall_avg_duration_ms"] == 0.0
    assert len(metrics["lens_metrics"]) == 0


@pytest.mark.asyncio
async def test_metrics_collector_record_analysis():
    """Test recording analysis metrics."""
    collector = MetricsCollector()

    collector.record_analysis_start()
    collector.record_analysis_duration(250.5)

    metrics = collector.get_metrics()

    assert metrics["total_analyses"] == 1
    assert metrics["overall_avg_duration_ms"] == 250.5


@pytest.mark.asyncio
async def test_metrics_collector_record_lens_success():
    """Test recording successful lens execution."""
    collector = MetricsCollector()

    collector.record_lens_execution("lens1", success=True, duration_ms=150.0)

    metrics = collector.get_metrics()

    assert "lens1" in metrics["lens_metrics"]
    lens1_metrics = metrics["lens_metrics"]["lens1"]

    assert lens1_metrics.total_executions == 1
    assert lens1_metrics.successful_executions == 1
    assert lens1_metrics.failed_executions == 0
    assert lens1_metrics.avg_duration_ms == 150.0
    assert lens1_metrics.success_rate_pct == 100.0


@pytest.mark.asyncio
async def test_metrics_collector_record_lens_failure():
    """Test recording failed lens execution."""
    collector = MetricsCollector()

    collector.record_lens_execution(
        "lens1", success=False, duration_ms=50.0, error_type="ValueError"
    )

    metrics = collector.get_metrics()

    lens1_metrics = metrics["lens_metrics"]["lens1"]

    assert lens1_metrics.total_executions == 1
    assert lens1_metrics.successful_executions == 0
    assert lens1_metrics.failed_executions == 1
    assert lens1_metrics.success_rate_pct == 0.0
    assert "ValueError" in lens1_metrics.error_types
    assert lens1_metrics.error_types["ValueError"] == 1


@pytest.mark.asyncio
async def test_metrics_collector_mixed_results():
    """Test recording mix of successes and failures."""
    collector = MetricsCollector()

    # Record 3 successes and 1 failure
    collector.record_lens_execution("lens1", success=True, duration_ms=100.0)
    collector.record_lens_execution("lens1", success=True, duration_ms=200.0)
    collector.record_lens_execution("lens1", success=True, duration_ms=300.0)
    collector.record_lens_execution(
        "lens1", success=False, duration_ms=50.0, error_type="ValueError"
    )

    metrics = collector.get_metrics()
    lens1_metrics = metrics["lens_metrics"]["lens1"]

    assert lens1_metrics.total_executions == 4
    assert lens1_metrics.successful_executions == 3
    assert lens1_metrics.failed_executions == 1
    assert lens1_metrics.success_rate_pct == 75.0  # 3/4 = 75%
    assert lens1_metrics.avg_duration_ms == 162.5  # (100+200+300+50)/4


@pytest.mark.asyncio
async def test_metrics_collector_duration_statistics():
    """Test min/max/avg duration calculations."""
    collector = MetricsCollector()

    durations = [100.0, 200.0, 300.0, 400.0, 500.0]
    for duration in durations:
        collector.record_lens_execution("lens1", success=True, duration_ms=duration)

    metrics = collector.get_metrics()
    lens1_metrics = metrics["lens_metrics"]["lens1"]

    assert lens1_metrics.min_duration_ms == 100.0
    assert lens1_metrics.max_duration_ms == 500.0
    assert lens1_metrics.avg_duration_ms == 300.0


@pytest.mark.asyncio
async def test_metrics_collector_multiple_lenses():
    """Test metrics for multiple different lenses."""
    collector = MetricsCollector()

    collector.record_lens_execution("lens1", success=True, duration_ms=100.0)
    collector.record_lens_execution("lens3", success=True, duration_ms=150.0)
    collector.record_lens_execution(
        "lens4", success=False, duration_ms=50.0, error_type="ValueError"
    )
    collector.record_lens_execution("lens5", success=True, duration_ms=200.0)

    metrics = collector.get_metrics()

    assert len(metrics["lens_metrics"]) == 4
    assert "lens1" in metrics["lens_metrics"]
    assert "lens3" in metrics["lens_metrics"]
    assert "lens4" in metrics["lens_metrics"]
    assert "lens5" in metrics["lens_metrics"]

    # Check lens4 failure
    lens4_metrics = metrics["lens_metrics"]["lens4"]
    assert lens4_metrics.success_rate_pct == 0.0
    assert "ValueError" in lens4_metrics.error_types


@pytest.mark.asyncio
async def test_metrics_collector_reset():
    """Test metrics collector reset functionality."""
    collector = MetricsCollector()

    # Add some data
    collector.record_analysis_start()
    collector.record_lens_execution("lens1", success=True, duration_ms=100.0)

    # Verify data exists
    metrics_before = collector.get_metrics()
    assert metrics_before["total_analyses"] == 1

    # Reset
    collector.reset()

    # Verify data is cleared
    metrics_after = collector.get_metrics()
    assert metrics_after["total_analyses"] == 0
    assert len(metrics_after["lens_metrics"]) == 0


@pytest.mark.asyncio
async def test_coordinator_retry_logic_exists():
    """Test that retry logic is properly configured on coordinator."""
    coordinator = FourLensesCoordinator()

    # Check that retry wrapper method exists
    assert hasattr(coordinator, "_execute_lens_with_retry")

    # Verify it's a retry-decorated function (has tenacity attributes)
    retry_wrapper = coordinator._execute_lens_with_retry
    assert hasattr(retry_wrapper, "retry")


@pytest.mark.asyncio
async def test_opentelemetry_tracer_configured():
    """Test that OpenTelemetry tracer is properly initialized."""
    from analytics.services.mcp_server.orchestration.coordinator import tracer

    # Tracer should be initialized
    assert tracer is not None

    # Should be able to create spans
    with tracer.start_as_current_span("test_span") as span:
        assert span is not None
        span.set_attribute("test_key", "test_value")


@pytest.mark.asyncio
async def test_coordinator_metrics_integration():
    """Test that coordinator creates metrics collector instance."""
    # This will trigger the lazy initialization
    collector = get_metrics_collector_instance()

    assert collector is not None
    assert isinstance(collector, MetricsCollector)


@pytest.mark.asyncio
async def test_multiple_error_types():
    """Test tracking multiple different error types."""
    collector = MetricsCollector()

    collector.record_lens_execution(
        "lens1", success=False, duration_ms=50.0, error_type="ValueError"
    )
    collector.record_lens_execution(
        "lens1", success=False, duration_ms=60.0, error_type="TimeoutError"
    )
    collector.record_lens_execution(
        "lens1", success=False, duration_ms=70.0, error_type="ValueError"
    )

    metrics = collector.get_metrics()
    lens1_metrics = metrics["lens_metrics"]["lens1"]

    assert len(lens1_metrics.error_types) == 2
    assert lens1_metrics.error_types["ValueError"] == 2
    assert lens1_metrics.error_types["TimeoutError"] == 1


@pytest.mark.asyncio
async def test_overall_success_rate():
    """Test overall success rate calculation across lenses."""
    collector = MetricsCollector()

    # Lens 1: 2 successes
    collector.record_lens_execution("lens1", success=True, duration_ms=100.0)
    collector.record_lens_execution("lens1", success=True, duration_ms=150.0)

    # Lens 3: 1 success, 1 failure
    collector.record_lens_execution("lens3", success=True, duration_ms=120.0)
    collector.record_lens_execution(
        "lens3", success=False, duration_ms=50.0, error_type="ValueError"
    )

    metrics = collector.get_metrics()

    # Total: 3 successes out of 4 executions = 75%
    assert metrics["overall_success_rate_pct"] == 75.0


@pytest.mark.asyncio
async def test_metrics_collector_uptime():
    """Test that uptime is tracked."""
    collector = MetricsCollector()

    metrics = collector.get_metrics()

    assert "uptime_seconds" in metrics
    assert metrics["uptime_seconds"] >= 0
    assert isinstance(metrics["uptime_seconds"], float)
