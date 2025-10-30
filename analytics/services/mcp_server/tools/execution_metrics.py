"""Execution Metrics MCP Tool - Phase 4

This tool provides performance metrics and execution statistics for the MCP server.
It tracks:
1. Lens execution counts and success rates
2. Average execution times per lens
3. Error rates and failure types
4. Overall system performance metrics

Usage:
    Call get_execution_metrics() to retrieve current performance statistics
    Call reset_execution_metrics() to clear metrics (useful for testing)
"""

import time
from collections import defaultdict, deque
from datetime import datetime
from threading import Lock
from typing import Any

import structlog
from fastmcp import Context
from pydantic import BaseModel, Field

from analytics.services.mcp_server.instance import mcp

logger = structlog.get_logger(__name__)


class LensMetrics(BaseModel):
    """Metrics for a single lens."""

    total_executions: int = Field(description="Total number of executions")
    successful_executions: int = Field(description="Number of successful executions")
    failed_executions: int = Field(description="Number of failed executions")
    avg_duration_ms: float = Field(description="Average execution duration in ms")
    min_duration_ms: float | None = Field(
        default=None, description="Minimum execution duration in ms"
    )
    max_duration_ms: float | None = Field(
        default=None, description="Maximum execution duration in ms"
    )
    success_rate_pct: float = Field(description="Success rate percentage (0-100)")
    error_types: dict[str, int] = Field(
        default_factory=dict, description="Count of errors by type"
    )


class ExecutionMetricsResponse(BaseModel):
    """Execution metrics response with performance statistics."""

    timestamp: str = Field(description="ISO timestamp of metrics snapshot")
    total_analyses: int = Field(description="Total orchestrated analyses run")
    lens_metrics: dict[str, LensMetrics] = Field(
        description="Per-lens execution metrics"
    )
    overall_avg_duration_ms: float = Field(
        description="Average duration across all analyses"
    )
    overall_success_rate_pct: float = Field(
        description="Overall success rate percentage"
    )
    uptime_seconds: float = Field(description="Metrics collection uptime in seconds")


# In-memory metrics storage
class MetricsCollector:
    """Thread-safe in-memory metrics collector."""

    def __init__(self):
        self._lock = Lock()
        self._start_time = time.time()
        self._total_analyses = 0
        # Bounded collection: keep last 10,000 analysis durations
        self._analysis_durations: deque = deque(maxlen=10000)

        # Per-lens metrics: {lens_name: {executions, successes, failures, durations, errors}}
        # Note: durations are bounded deques (1000 per lens) created on first use
        self._lens_data: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "executions": 0,
                "successes": 0,
                "failures": 0,
                "durations": deque(maxlen=1000),  # Keep last 1000 durations per lens
                "error_types": defaultdict(int),
            }
        )

    def record_analysis_start(self):
        """Record the start of an orchestrated analysis."""
        with self._lock:
            self._total_analyses += 1

    def record_analysis_duration(self, duration_ms: float):
        """Record an analysis execution duration."""
        with self._lock:
            self._analysis_durations.append(duration_ms)

    def record_lens_execution(
        self,
        lens_name: str,
        success: bool,
        duration_ms: float,
        error_type: str | None = None,
    ):
        """Record a lens execution result.

        Args:
            lens_name: Name of the lens (e.g., 'lens1', 'lens5')
            success: Whether execution succeeded
            duration_ms: Execution duration in milliseconds
            error_type: Type of error if failed (e.g., 'ValueError', 'TimeoutError')
        """
        with self._lock:
            data = self._lens_data[lens_name]
            data["executions"] += 1

            if success:
                data["successes"] += 1
            else:
                data["failures"] += 1
                if error_type:
                    data["error_types"][error_type] += 1

            data["durations"].append(duration_ms)

    def get_metrics(self) -> dict[str, Any]:
        """Get current metrics snapshot."""
        with self._lock:
            lens_metrics = {}

            for lens_name, data in self._lens_data.items():
                durations = data["durations"]
                total_execs = data["executions"]
                successes = data["successes"]

                avg_duration = sum(durations) / len(durations) if durations else 0.0
                min_duration = min(durations) if durations else None
                max_duration = max(durations) if durations else None
                success_rate = (
                    (successes / total_execs * 100) if total_execs > 0 else 0.0
                )

                lens_metrics[lens_name] = LensMetrics(
                    total_executions=total_execs,
                    successful_executions=successes,
                    failed_executions=data["failures"],
                    avg_duration_ms=avg_duration,
                    min_duration_ms=min_duration,
                    max_duration_ms=max_duration,
                    success_rate_pct=success_rate,
                    error_types=dict(data["error_types"]),
                )

            # Calculate overall metrics
            all_durations = self._analysis_durations
            overall_avg_duration = (
                sum(all_durations) / len(all_durations) if all_durations else 0.0
            )

            # Overall success rate: ratio of successful lens executions to total
            total_lens_execs = sum(d["executions"] for d in self._lens_data.values())
            total_lens_successes = sum(d["successes"] for d in self._lens_data.values())
            overall_success_rate = (
                (total_lens_successes / total_lens_execs * 100)
                if total_lens_execs > 0
                else 0.0  # No data means 0% success rate (not 100%)
            )

            uptime = time.time() - self._start_time

            return {
                "timestamp": datetime.now().isoformat(),
                "total_analyses": self._total_analyses,
                "lens_metrics": lens_metrics,
                "overall_avg_duration_ms": overall_avg_duration,
                "overall_success_rate_pct": overall_success_rate,
                "uptime_seconds": uptime,
            }

    def reset(self):
        """Reset all metrics (useful for testing)."""
        with self._lock:
            self._start_time = time.time()
            self._total_analyses = 0
            self._analysis_durations.clear()
            self._lens_data.clear()


# Global metrics collector instance
_metrics_collector = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Get the global metrics collector instance."""
    return _metrics_collector


@mcp.tool()
async def get_execution_metrics(ctx: Context) -> ExecutionMetricsResponse:
    """
    Get execution metrics for all lenses and analyses.

    This tool provides comprehensive performance statistics including:
    - Total number of orchestrated analyses run
    - Per-lens execution counts, success rates, and timings
    - Overall system performance metrics
    - Error type distribution for failed executions

    Returns:
        ExecutionMetricsResponse with detailed performance statistics

    Example:
        >>> result = await get_execution_metrics()
        >>> print(result.total_analyses)  # 42
        >>> print(result.lens_metrics['lens1'].avg_duration_ms)  # 150.5
        >>> print(result.overall_success_rate_pct)  # 98.5
    """
    logger.info("retrieving_execution_metrics")

    metrics_data = _metrics_collector.get_metrics()

    logger.info(
        "execution_metrics_retrieved",
        total_analyses=metrics_data["total_analyses"],
        lens_count=len(metrics_data["lens_metrics"]),
        overall_success_rate=metrics_data["overall_success_rate_pct"],
    )

    return ExecutionMetricsResponse(**metrics_data)


@mcp.tool()
async def reset_execution_metrics(ctx: Context) -> dict[str, str]:
    """
    Reset all execution metrics to zero.

    This tool clears all collected metrics and restarts tracking from zero.
    Useful for testing or starting fresh metrics collection after maintenance.

    Returns:
        Confirmation message with reset timestamp

    Example:
        >>> result = await reset_execution_metrics()
        >>> print(result['status'])  # 'metrics reset successfully'
    """
    logger.info("resetting_execution_metrics")

    _metrics_collector.reset()
    timestamp = datetime.now().isoformat()

    logger.info("execution_metrics_reset", timestamp=timestamp)

    return {
        "status": "metrics reset successfully",
        "timestamp": timestamp,
        "message": "All execution metrics have been cleared and restarted from zero",
    }
